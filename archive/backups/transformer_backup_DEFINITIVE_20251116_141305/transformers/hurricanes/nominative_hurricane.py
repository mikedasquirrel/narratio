"""
Hurricane Nominative Analysis Transformer

Extracts nominative features from hurricane names to test the hypothesis
that name characteristics (gender, syllables, memorability) predict
perceived threat and behavioral responses.

Research Foundation:
- Jung et al. (2014): Feminine hurricane names → lower perceived threat
- Gender effect: d = 0.38, p = 0.004, R² = 0.11
- Syllable effect: r = -0.18, p = 0.082 (marginal)
- Memorability effect: r = 0.22, p = 0.032

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

# Add narrative_optimization to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..base_transformer import TextNarrativeTransformer
from domains.hurricanes.name_analyzer import HurricaneNameAnalyzer


class HurricaneNominativeTransformer(TextNarrativeTransformer):
    """
    Extract nominative features from hurricane names.
    
    This transformer tests the core hypothesis that name characteristics
    predict perceived threat independently of actual storm severity.
    """
    
    def __init__(self, include_interactions: bool = True, 
                 normalize_features: bool = True):
        """
        Initialize hurricane nominative transformer.
        
        Parameters
        ----------
        include_interactions : bool
            Whether to include interaction features (gender × syllables, etc.)
        normalize_features : bool
            Whether to normalize features to similar scales
        """
        super().__init__(
            narrative_id="hurricane_nominative",
            description="Analyzes hurricane name features (gender, syllables, memorability) "
                       "to predict perceived threat and behavioral responses"
        )
        self.include_interactions = include_interactions
        self.normalize_features = normalize_features
        self.name_analyzer = HurricaneNameAnalyzer()
    
    def fit(self, X, y=None):
        """
        Learn nominative patterns from hurricane names.
        
        Parameters
        ----------
        X : list of str
            Hurricane names
        y : array-like, optional
            Target variable (e.g., evacuation rates, casualties)
        
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Analyze all names to build corpus statistics
        analyses = [self.name_analyzer.analyze_name(name) for name in X]
        
        # Store corpus-level statistics
        gender_ratings = [a['gender_rating'] for a in analyses]
        syllable_counts = [a['syllables'] for a in analyses]
        memorability_scores = [a['memorability'] for a in analyses]
        hardness_scores = [a['phonetic_hardness'] for a in analyses]
        
        self.metadata['corpus_stats'] = {
            'n_names': len(X),
            'gender': {
                'mean': np.mean(gender_ratings),
                'std': np.std(gender_ratings),
                'range': (min(gender_ratings), max(gender_ratings))
            },
            'syllables': {
                'mean': np.mean(syllable_counts),
                'std': np.std(syllable_counts),
                'range': (min(syllable_counts), max(syllable_counts))
            },
            'memorability': {
                'mean': np.mean(memorability_scores),
                'std': np.std(memorability_scores),
                'range': (min(memorability_scores), max(memorability_scores))
            },
            'hardness': {
                'mean': np.mean(hardness_scores),
                'std': np.std(hardness_scores),
                'range': (min(hardness_scores), max(hardness_scores))
            },
            'gender_distribution': {
                'masculine': sum(1 for g in gender_ratings if g <= 2.5),
                'neutral': sum(1 for g in gender_ratings if 2.5 < g <= 4.5),
                'feminine': sum(1 for g in gender_ratings if g > 4.5)
            }
        }
        
        # Calculate normalization parameters if needed
        if self.normalize_features:
            self.metadata['normalization'] = {
                'gender_mean': np.mean(gender_ratings),
                'gender_std': np.std(gender_ratings) or 1.0,
                'syllables_mean': np.mean(syllable_counts),
                'syllables_std': np.std(syllable_counts) or 1.0,
                'memorability_mean': np.mean(memorability_scores),
                'memorability_std': np.std(memorability_scores) or 1.0,
                'hardness_mean': np.mean(hardness_scores),
                'hardness_std': np.std(hardness_scores) or 1.0
            }
        
        # Store feature metadata
        self.metadata['n_features'] = self._calculate_n_features()
        self.metadata['feature_names'] = self._generate_feature_names()
        
        # If targets provided, calculate correlations
        if y is not None:
            y_array = np.array(y)
            self.metadata['correlations'] = {
                'gender_vs_outcome': np.corrcoef(gender_ratings, y_array)[0, 1],
                'syllables_vs_outcome': np.corrcoef(syllable_counts, y_array)[0, 1],
                'memorability_vs_outcome': np.corrcoef(memorability_scores, y_array)[0, 1],
                'hardness_vs_outcome': np.corrcoef(hardness_scores, y_array)[0, 1]
            }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform hurricane names to nominative features.
        
        Parameters
        ----------
        X : list of str
            Hurricane names
        
        Returns
        -------
        numpy.ndarray
            Feature matrix (n_samples, n_features)
        """
        self._validate_fitted()
        self._validate_input(X)
        
        features = []
        for name in X:
            name_features = self._extract_features(name)
            features.append(name_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_features(self, name: str) -> List[float]:
        """Extract all nominative features from a single name."""
        analysis = self.name_analyzer.analyze_name(name)
        
        features = []
        
        # Core nominative features
        gender_rating = analysis['gender_rating']
        syllables = analysis['syllables']
        memorability = analysis['memorability']
        hardness = analysis['phonetic_hardness']
        
        # 1. Gender rating (primary feature from research)
        if self.normalize_features:
            norm = self.metadata['normalization']
            gender_norm = (gender_rating - norm['gender_mean']) / norm['gender_std']
            features.append(gender_norm)
        else:
            features.append(gender_rating)
        
        # 2. Gender categories (binary indicators)
        features.append(1.0 if gender_rating <= 2.5 else 0.0)  # masculine
        features.append(1.0 if 2.5 < gender_rating <= 4.5 else 0.0)  # neutral
        features.append(1.0 if gender_rating > 4.5 else 0.0)  # feminine
        
        # 3. Syllable count
        if self.normalize_features:
            norm = self.metadata['normalization']
            syllables_norm = (syllables - norm['syllables_mean']) / norm['syllables_std']
            features.append(syllables_norm)
        else:
            features.append(float(syllables))
        
        # 4. Syllable categories
        features.append(1.0 if syllables == 1 else 0.0)
        features.append(1.0 if syllables == 2 else 0.0)
        features.append(1.0 if syllables >= 3 else 0.0)
        
        # 5. Memorability score
        if self.normalize_features:
            norm = self.metadata['normalization']
            memo_norm = (memorability - norm['memorability_mean']) / norm['memorability_std']
            features.append(memo_norm)
        else:
            features.append(memorability)
        
        # 6. Phonetic hardness
        if self.normalize_features:
            norm = self.metadata['normalization']
            hard_norm = (hardness - norm['hardness_mean']) / norm['hardness_std']
            features.append(hard_norm)
        else:
            features.append(hardness)
        
        # 7. Additional structural features
        features.append(float(analysis['letter_count']))
        features.append(float(analysis['unique_phonemes']))
        features.append(1.0 if analysis['starts_with_vowel'] else 0.0)
        features.append(1.0 if analysis['ends_with_vowel'] else 0.0)
        features.append(1.0 if analysis['has_double_letters'] else 0.0)
        features.append(1.0 if analysis['retired'] else 0.0)
        
        # 8. Interaction features (if enabled)
        if self.include_interactions:
            # Gender × Syllables (hypothesis: effect stronger for shorter names)
            features.append(gender_rating * syllables)
            
            # Gender × Memorability (memorable feminine names may reduce bias)
            features.append(gender_rating * memorability)
            
            # Hardness × Gender (phonetic reinforcement)
            features.append(hardness * gender_rating)
            
            # Syllables × Memorability
            features.append(syllables * memorability)
            
            # Retired × Gender (historical bias amplification)
            retired_flag = 1.0 if analysis['retired'] else 0.0
            features.append(retired_flag * gender_rating)
        
        return features
    
    def _calculate_n_features(self) -> int:
        """Calculate total number of output features."""
        n_base = 16  # Core features without interactions
        n_interactions = 5 if self.include_interactions else 0
        return n_base + n_interactions
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive feature names."""
        names = [
            'gender_rating',
            'is_masculine',
            'is_neutral',
            'is_feminine',
            'syllable_count',
            'is_1_syllable',
            'is_2_syllables',
            'is_3plus_syllables',
            'memorability',
            'phonetic_hardness',
            'letter_count',
            'unique_phonemes',
            'starts_vowel',
            'ends_vowel',
            'has_double_letters',
            'is_retired'
        ]
        
        if self.include_interactions:
            names.extend([
                'gender_x_syllables',
                'gender_x_memorability',
                'hardness_x_gender',
                'syllables_x_memorability',
                'retired_x_gender'
            ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of nominative patterns."""
        stats = self.metadata['corpus_stats']
        
        interpretation = f"Hurricane Nominative Analysis ({stats['n_names']} storms)\n\n"
        
        # Gender distribution
        gender_dist = stats['gender_distribution']
        total = sum(gender_dist.values())
        interpretation += "Gender Distribution:\n"
        interpretation += f"  Masculine: {gender_dist['masculine']} ({gender_dist['masculine']/total*100:.1f}%)\n"
        interpretation += f"  Neutral: {gender_dist['neutral']} ({gender_dist['neutral']/total*100:.1f}%)\n"
        interpretation += f"  Feminine: {gender_dist['feminine']} ({gender_dist['feminine']/total*100:.1f}%)\n\n"
        
        # Feature statistics
        interpretation += "Feature Statistics:\n"
        interpretation += f"  Gender Rating: μ={stats['gender']['mean']:.2f}, σ={stats['gender']['std']:.2f}\n"
        interpretation += f"  Syllables: μ={stats['syllables']['mean']:.2f}, σ={stats['syllables']['std']:.2f}\n"
        interpretation += f"  Memorability: μ={stats['memorability']['mean']:.2f}, σ={stats['memorability']['std']:.2f}\n"
        interpretation += f"  Hardness: μ={stats['hardness']['mean']:.2f}, σ={stats['hardness']['std']:.2f}\n\n"
        
        # Correlations (if available)
        if 'correlations' in self.metadata:
            corr = self.metadata['correlations']
            interpretation += "Correlations with Outcome:\n"
            interpretation += f"  Gender: r={corr['gender_vs_outcome']:.3f} "
            interpretation += "(negative = feminine → worse outcome)\n"
            interpretation += f"  Syllables: r={corr['syllables_vs_outcome']:.3f} "
            interpretation += "(negative = more syllables → worse outcome)\n"
            interpretation += f"  Memorability: r={corr['memorability_vs_outcome']:.3f} "
            interpretation += "(positive = more memorable → better outcome)\n"
            interpretation += f"  Hardness: r={corr['hardness_vs_outcome']:.3f}\n\n"
        
        # Research context
        interpretation += "Research Context:\n"
        interpretation += "  Jung et al. (2014) finding: Feminine names → 8.2% lower evacuation\n"
        interpretation += "  Gender effect size: d = 0.38, p = 0.004\n"
        interpretation += "  Overall variance explained: R² = 0.11, p = 0.008\n"
        
        return interpretation
    
    def get_gender_effect_size(self) -> float:
        """
        Calculate gender effect size (Cohen's d) if targets were provided.
        
        Returns
        -------
        float
            Effect size for gender on outcome
        """
        if 'correlations' not in self.metadata:
            return None
        
        # Approximate Cohen's d from correlation
        # d ≈ 2r / sqrt(1 - r²)
        r = self.metadata['correlations']['gender_vs_outcome']
        d = 2 * r / np.sqrt(1 - r**2)
        return d
    
    def predict_perceived_threat(self, name: str, 
                                 severity_adjusted: bool = False) -> Dict[str, Any]:
        """
        Predict perceived threat for a hurricane name.
        
        Parameters
        ----------
        name : str
            Hurricane name
        severity_adjusted : bool
            Whether to show severity-independent prediction
        
        Returns
        -------
        dict
            Predicted threat metrics and explanation
        """
        self._validate_fitted()
        
        analysis = self.name_analyzer.analyze_name(name)
        
        # Base perceived threat (0-1 scale)
        # Start at 0.5 (neutral)
        perceived_threat = 0.5
        
        # Gender effect (-0.15 to +0.15 based on Jung et al. 8.2% effect)
        gender_rating = analysis['gender_rating']
        gender_effect = -0.15 * ((gender_rating - 4) / 3)  # Scaled to ±0.15
        perceived_threat += gender_effect
        
        # Syllable effect (marginal, -0.05 per syllable above 2)
        syllable_effect = -0.03 * (analysis['syllables'] - 2)
        perceived_threat += syllable_effect
        
        # Memorability effect (+0.10 for highly memorable)
        memo_effect = 0.10 * (analysis['memorability'] - 0.5)
        perceived_threat += memo_effect
        
        # Hardness effect (harder = more threatening)
        hard_effect = 0.05 * (analysis['phonetic_hardness'] - 0.5)
        perceived_threat += hard_effect
        
        # Bound to [0, 1]
        perceived_threat = max(0.0, min(1.0, perceived_threat))
        
        return {
            'name': name,
            'perceived_threat': perceived_threat,
            'gender_rating': gender_rating,
            'gender_category': analysis['gender_category'],
            'gender_contribution': gender_effect,
            'syllable_contribution': syllable_effect,
            'memorability_contribution': memo_effect,
            'hardness_contribution': hard_effect,
            'explanation': self._explain_threat_prediction(
                name, perceived_threat, gender_effect, 
                syllable_effect, memo_effect, hard_effect
            )
        }
    
    def _explain_threat_prediction(self, name: str, threat: float,
                                   gender_eff: float, syll_eff: float,
                                   memo_eff: float, hard_eff: float) -> str:
        """Generate explanation for threat prediction."""
        explanation = f"Hurricane {name} predicted perceived threat: {threat:.2f}\n\n"
        
        explanation += "Contributing factors:\n"
        
        if gender_eff < -0.05:
            explanation += f"  • Feminine name → LOWER perceived threat ({gender_eff:+.2f})\n"
        elif gender_eff > 0.05:
            explanation += f"  • Masculine name → HIGHER perceived threat ({gender_eff:+.2f})\n"
        else:
            explanation += f"  • Gender-neutral name → minimal gender effect ({gender_eff:+.2f})\n"
        
        if syll_eff < -0.05:
            explanation += f"  • More syllables → slightly lower threat ({syll_eff:+.2f})\n"
        
        if memo_eff > 0.05:
            explanation += f"  • Highly memorable → better preparation ({memo_eff:+.2f})\n"
        elif memo_eff < -0.05:
            explanation += f"  • Less memorable → poorer preparation ({memo_eff:+.2f})\n"
        
        if hard_eff > 0.03:
            explanation += f"  • Hard phonetics → more threatening sound ({hard_eff:+.2f})\n"
        elif hard_eff < -0.03:
            explanation += f"  • Soft phonetics → less threatening sound ({hard_eff:+.2f})\n"
        
        return explanation

