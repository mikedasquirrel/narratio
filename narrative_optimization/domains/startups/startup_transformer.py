"""
Startup-Specific Narrative Transformer

Extracts narrative features specific to startup descriptions:
- Innovation language (required for startups)
- Team ensemble effects (founding team matters)
- Market positioning clarity
- Technical legitimacy signals
- Vision/ambition markers
- Execution credibility
"""

import numpy as np
import re
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.base_transformer import TextNarrativeTransformer


class StartupNarrativeTransformer(TextNarrativeTransformer):
    """
    Startup-specific narrative analyzer.
    
    Focuses on features that matter for startups:
    1. Innovation narrative (is this actually new?)
    2. Team narrative (founding team composition and chemistry)
    3. Market positioning (clear target, competitive advantage)
    4. Execution credibility (specific not vague)
    5. Vision/scale (thinking big vs small)
    """
    
    def __init__(self):
        super().__init__(
            narrative_id="startup_narrative",
            description="Startup success predicted by innovation clarity, team coherence, and execution credibility in founding narrative"
        )
        
        # Innovation markers
        self.innovation_words = {
            'new', 'first', 'revolutionary', 'innovative', 'novel', 'breakthrough',
            'reimagine', 'reinvent', 'disrupt', 'transform', 'change', 'future'
        }
        
        # Team strength markers
        self.team_strength_words = {
            'complementary', 'partnership', 'together', 'combined', 'expertise',
            'experience', 'background', 'team', 'co-founder', 'founding'
        }
        
        # Execution markers
        self.execution_words = {
            'build', 'built', 'create', 'develop', 'launch', 'ship', 'deliver',
            'execute', 'implement', 'scale', 'grow'
        }
        
        # Vague/weak markers (negative signals)
        self.vague_words = {
            'sort of', 'kind of', 'maybe', 'might', 'could', 'possibly',
            'perhaps', 'potentially', 'basically', 'simply'
        }
        
        # Scale/ambition markers
        self.ambition_words = {
            'everyone', 'world', 'global', 'billion', 'millions', 'all',
            'revolutionize', 'transform', 'change the world'
        }
        
        # Market clarity markers
        self.market_words = {
            'for', 'helps', 'enables', 'connects', 'marketplace', 'platform',
            'customers', 'users', 'market', 'industry'
        }
    
    def fit(self, X, y=None):
        """Learn startup narrative patterns from corpus."""
        # Calculate baseline ratios
        all_text = ' '.join(str(x).lower() for x in X)
        total_words = len(all_text.split())
        
        self.baseline_innovation_ratio = sum(all_text.count(w) for w in self.innovation_words) / total_words
        self.baseline_execution_ratio = sum(all_text.count(w) for w in self.execution_words) / total_words
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract startup narrative features."""
        features = []
        
        for text in X:
            text_str = str(text).lower()
            words = text_str.split()
            n_words = len(words) + 1
            
            # 1. Innovation Features (10 features)
            innovation_count = sum(1 for w in self.innovation_words if w in text_str)
            innovation_density = innovation_count / n_words
            has_innovation_narrative = 1.0 if innovation_count >= 2 else 0.0
            innovation_diversity = len([w for w in self.innovation_words if w in text_str]) / len(self.innovation_words)
            
            # Innovation specificity (specific > vague)
            specific_innovation = innovation_count > 0 and text_str.count('new') > 0
            
            # 2. Team Features (8 features)
            team_mention_count = sum(1 for w in self.team_strength_words if w in text_str)
            team_narrative_strength = team_mention_count / n_words
            mentions_complementarity = 1.0 if 'complementary' in text_str or 'complement' in text_str else 0.0
            mentions_partnership = 1.0 if any(w in text_str for w in ['partner', 'duo', 'together']) else 0.0
            
            # Founder count heuristic (count mentions of 'founder', 'co-founder')
            founder_mentions = text_str.count('founder') + text_str.count('co-founder')
            
            # 3. Market Positioning Features (8 features)
            market_clarity_count = sum(1 for w in self.market_words if w in text_str)
            has_clear_market = 1.0 if market_clarity_count >= 2 else 0.0
            target_specificity = 1.0 if 'for' in text_str else 0.0
            
            # Problem-solution clarity
            mentions_problem = 1.0 if any(w in text_str for w in ['problem', 'pain', 'challenge', 'difficult']) else 0.0
            mentions_solution = 1.0 if any(w in text_str for w in ['solution', 'solves', 'addresses', 'fixes']) else 0.0
            problem_solution_clarity = mentions_problem * mentions_solution
            
            # 4. Execution Credibility Features (8 features)
            execution_count = sum(1 for w in self.execution_words if w in text_str)
            execution_strength = execution_count / n_words
            has_execution_narrative = 1.0 if execution_count >= 2 else 0.0
            
            # Specificity vs vagueness
            vague_count = sum(1 for w in self.vague_words if w in text_str)
            vagueness_penalty = vague_count / n_words
            credibility_score = execution_strength - vagueness_penalty
            
            # Tense analysis (building vs built)
            building_tense = text_str.count('build') + text_str.count('building') + text_str.count('create')
            built_tense = text_str.count('built') + text_str.count('created') + text_str.count('launched')
            has_shipped = 1.0 if built_tense > building_tense else 0.0
            
            # 5. Vision/Ambition Features (6 features)
            ambition_count = sum(1 for w in self.ambition_words if w in text_str)
            ambition_score = ambition_count / n_words
            thinks_big = 1.0 if ambition_count >= 1 else 0.0
            
            # Specific scale markers
            mentions_billions = 1.0 if 'billion' in text_str else 0.0
            mentions_everyone = 1.0 if 'everyone' in text_str or 'anyone' in text_str else 0.0
            
            # 6. Nominative Features (5 features)
            # Company name analysis (from full text, not just description)
            company_names = re.findall(r'\b[A-Z][a-z]+\b', str(text))
            
            if company_names:
                name_length = np.mean([len(n) for n in company_names])
                name_memorability = len(company_names[0]) if company_names else 0
                name_has_tech_suffix = 1.0 if company_names and company_names[0].lower().endswith(('ly', 'ify', 'io', 'ai', 'tech')) else 0.0
            else:
                name_length = 0
                name_memorability = 0
                name_has_tech_suffix = 0
            
            # Combine all features
            feature_vector = [
                # Innovation (10)
                innovation_count,
                innovation_density,
                has_innovation_narrative,
                innovation_diversity,
                specific_innovation,
                text_str.count('first'),
                text_str.count('new'),
                text_str.count('revolutionary'),
                text_str.count('disrupt'),
                innovation_density - self.baseline_innovation_ratio,  # Relative innovation
                
                # Team (8)
                team_mention_count,
                team_narrative_strength,
                mentions_complementarity,
                mentions_partnership,
                founder_mentions,
                1.0 if founder_mentions >= 2 else 0.0,  # Multiple founders
                1.0 if team_mention_count == 0 else 0.0,  # Solo founder
                team_mention_count / (founder_mentions + 1),  # Team narrative per founder
                
                # Market Positioning (8)
                market_clarity_count,
                has_clear_market,
                target_specificity,
                mentions_problem,
                mentions_solution,
                problem_solution_clarity,
                market_clarity_count / n_words,
                1.0 if market_clarity_count >= 3 else 0.0,
                
                # Execution (8)
                execution_count,
                execution_strength,
                has_execution_narrative,
                vague_count,
                vagueness_penalty,
                credibility_score,
                has_shipped,
                execution_strength - self.baseline_execution_ratio,  # Relative execution
                
                # Ambition (6)
                ambition_count,
                ambition_score,
                thinks_big,
                mentions_billions,
                mentions_everyone,
                ambition_count / n_words,
                
                # Nominative (5)
                name_length,
                name_memorability,
                name_has_tech_suffix,
                1.0 if name_length <= 8 else 0.0,  # Short memorable names
                1.0 if name_length >= 12 else 0.0,  # Long descriptive names
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretation."""
        return [
            # Innovation
            'innovation_count', 'innovation_density', 'has_innovation_narrative',
            'innovation_diversity', 'specific_innovation', 'first_count', 'new_count',
            'revolutionary_count', 'disrupt_count', 'innovation_relative',
            
            # Team
            'team_mentions', 'team_density', 'mentions_complementarity',
            'mentions_partnership', 'founder_mentions', 'multiple_founders',
            'solo_founder', 'team_narrative_per_founder',
            
            # Market
            'market_clarity', 'has_clear_market', 'target_specificity',
            'mentions_problem', 'mentions_solution', 'problem_solution_clarity',
            'market_density', 'very_clear_market',
            
            # Execution
            'execution_count', 'execution_strength', 'has_execution_narrative',
            'vague_count', 'vagueness_penalty', 'credibility_score',
            'has_shipped', 'execution_relative',
            
            # Ambition
            'ambition_count', 'ambition_score', 'thinks_big',
            'mentions_billions', 'mentions_everyone', 'ambition_density',
            
            # Nominative
            'name_length', 'name_memorability', 'name_tech_suffix',
            'short_memorable_name', 'long_descriptive_name'
        ]
    
    def _generate_interpretation(self) -> str:
        return (
            f"Startup Narrative Analysis:\n"
            f"- Innovation baseline: {self.baseline_innovation_ratio:.3f}\n"
            f"- Execution baseline: {self.baseline_execution_ratio:.3f}\n"
            f"- Features: 45 dimensions (innovation, team, market, execution, ambition, nominative)\n"
            f"- Hypothesis: Success predicted by innovation clarity + team coherence + execution credibility"
        )


if __name__ == "__main__":
    print("Startup Narrative Transformer - Ready")
    print("Use with StartupDataCollector to analyze YC companies")

