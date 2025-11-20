"""
Contextual Narrative Analyzer

Intelligently identifies missing context variables and generates follow-up questions
to get a complete narrative picture. Works across all domains.
"""

from typing import Dict, List, Any, Optional
import re
from collections import Counter


class ContextualNarrativeAnalyzer:
    """
    Analyzes narrative text to identify missing contextual variables
    and generate intelligent follow-up questions.
    
    This analyzer understands what information is typically relevant
    for different domains and can identify gaps in the narrative.
    """
    
    def __init__(self):
        # Domain patterns
        self.domain_patterns = {
            'sports': {
                'keywords': ['team', 'game', 'win', 'player', 'championship', 'season', 'coach', 'match'],
                'typical_variables': ['team_name', 'current_record', 'recent_performance', 'key_players', 'injuries', 'home_away', 'weather', 'historical_matchup'],
                'questions': [
                    "What is the team's current win-loss record?",
                    "Are there any key player injuries?",
                    "Is this a home or away game?",
                    "How have they performed in recent games?",
                    "What is their historical head-to-head record?"
                ]
            },
            'products': {
                'keywords': ['product', 'feature', 'price', 'quality', 'design', 'technology', 'value'],
                'typical_variables': ['price_point', 'target_audience', 'key_features', 'brand_reputation', 'warranty', 'availability'],
                'questions': [
                    "What is the price range?",
                    "Who is the target customer?",
                    "What are the standout features?",
                    "How does it compare to competitors?",
                    "What's the warranty and support like?"
                ]
            },
            'profiles': {
                'keywords': ['person', 'personality', 'interest', 'goal', 'value', 'experience', 'looking'],
                'typical_variables': ['age', 'location', 'interests', 'goals', 'values', 'experience_level', 'relationship_intent'],
                'questions': [
                    "What are their core values?",
                    "What are they looking for?",
                    "What's their background/experience?",
                    "What are their long-term goals?",
                    "What makes them unique?"
                ]
            },
            'brands': {
                'keywords': ['mission', 'company', 'brand', 'business', 'customer', 'innovation', 'commitment'],
                'typical_variables': ['mission_statement', 'target_market', 'unique_value_prop', 'company_size', 'history', 'values'],
                'questions': [
                    "What is their mission statement?",
                    "Who is their target market?",
                    "What's their unique value proposition?",
                    "How long have they been in business?",
                    "What values do they prioritize?"
                ]
            }
        }
    
    def analyze_with_context(self, text: str, domain: str = 'general', known_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze text and identify missing contextual variables.
        
        Parameters
        ----------
        text : str
            The narrative text to analyze
        domain : str
            Domain type ('sports', 'products', 'profiles', 'brands', 'general')
        known_context : dict, optional
            Context variables already known
        
        Returns
        -------
        analysis : dict
            Contains missing_variables, follow_up_questions, and recommendations
        """
        known_context = known_context or {}
        
        # Detect domain if not specified
        if domain == 'general':
            domain = self._detect_domain(text)
        
        # Get expected variables for this domain
        domain_config = self.domain_patterns.get(domain, {})
        expected_variables = domain_config.get('typical_variables', [])
        
        # Identify what's missing
        missing_variables = []
        for var in expected_variables:
            if var not in known_context and not self._variable_present_in_text(text, var):
                missing_variables.append(var)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(
            text, domain, missing_variables
        )
        
        # Generate contextual recommendations
        recommendations = self._generate_recommendations(
            text, domain, missing_variables
        )
        
        return {
            'detected_domain': domain,
            'missing_variables': missing_variables,
            'follow_up_questions': follow_up_questions,
            'contextual_recommendations': recommendations,
            'completeness_score': 1 - (len(missing_variables) / max(len(expected_variables), 1))
        }
    
    def _detect_domain(self, text: str) -> str:
        """Detect the most likely domain from text content."""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, config in self.domain_patterns.items():
            score = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            domain_scores[domain] = score
        
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    def _variable_present_in_text(self, text: str, variable: str) -> bool:
        """Check if a variable is implicitly present in the text."""
        text_lower = text.lower()
        variable_keywords = {
            'team_name': ['team', 'name'],
            'current_record': ['record', 'wins', 'losses', 'season'],
            'recent_performance': ['recent', 'last', 'previous', 'lately'],
            'key_players': ['player', 'star', 'roster'],
            'injuries': ['injury', 'injured', 'out', 'questionable'],
            'home_away': ['home', 'away', 'road'],
            'price_point': ['price', 'cost', '$', 'expensive', 'cheap'],
            'target_audience': ['for', 'designed', 'ideal', 'perfect'],
            'age': ['age', 'old', 'young', 'years'],
            'location': ['from', 'live', 'based', 'located'],
            'interests': ['love', 'enjoy', 'passion', 'hobby'],
            'goals': ['want', 'looking', 'seeking', 'goal', 'aim']
        }
        
        keywords = variable_keywords.get(variable, [variable.replace('_', ' ')])
        return any(keyword in text_lower for keyword in keywords)
    
    def _generate_follow_up_questions(self, text: str, domain: str, missing: List[str]) -> List[str]:
        """Generate intelligent follow-up questions based on missing variables."""
        domain_config = self.domain_patterns.get(domain, {})
        base_questions = domain_config.get('questions', [])
        
        # Filter to questions relevant to missing variables
        relevant_questions = []
        for question in base_questions[:5]:  # Limit to 5 questions
            question_lower = question.lower()
            for var in missing:
                if any(keyword in question_lower for keyword in var.split('_')):
                    if question not in relevant_questions:
                        relevant_questions.append(question)
                        break
        
        # Add generic questions if we have few
        if len(relevant_questions) < 3:
            relevant_questions.extend([
                "Can you provide more background context?",
                "What additional details are relevant?",
                "What outcome are you trying to predict?"
            ][:3 - len(relevant_questions)])
        
        return relevant_questions
    
    def _generate_recommendations(self, text: str, domain: str, missing: List[str]) -> List[str]:
        """Generate recommendations for improving the analysis."""
        recommendations = []
        
        if len(missing) > 5:
            recommendations.append(
                f"Consider providing more context. {len(missing)} relevant variables are missing for complete {domain} analysis."
            )
        
        if domain == 'sports':
            if 'current_record' in missing:
                recommendations.append("Including current season record would strengthen competitive analysis.")
            if 'recent_performance' in missing:
                recommendations.append("Recent game results would help assess momentum.")
            if 'injuries' in missing:
                recommendations.append("Key player injury information impacts predictive accuracy.")
        
        elif domain == 'products':
            if 'price_point' in missing:
                recommendations.append("Price information enables value comparison analysis.")
            if 'target_audience' in missing:
                recommendations.append("Knowing target audience helps assess market positioning.")
        
        elif domain == 'profiles':
            if 'goals' in missing:
                recommendations.append("Understanding goals and intentions improves compatibility analysis.")
            if 'interests' in missing:
                recommendations.append("Interest details strengthen ensemble diversity analysis.")
        
        if not recommendations:
            recommendations.append(
                "Text provides good narrative coverage. Analysis captures available patterns comprehensively."
            )
        
        return recommendations


    def suggest_additional_transformers(self, text: str, domain: str) -> List[Dict[str, str]]:
        """Suggest which additional transformers would be most valuable."""
        suggestions = []
        
        # Analyze text characteristics
        has_first_person = bool(re.search(r'\b(i|me|my|myself)\b', text.lower()))
        has_future_tense = bool(re.search(r'\b(will|shall|going to|gonna)\b', text.lower()))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', text))
        
        if has_first_person:
            suggestions.append({
                'transformer': 'SelfPerceptionTransformer',
                'reason': 'Text contains first-person narratives — identity patterns detectable'
            })
        
        if has_future_tense:
            suggestions.append({
                'transformer': 'NarrativePotentialTransformer',
                'reason': 'Future-oriented language present — potential and openness measurable'
            })
        
        if has_proper_nouns:
            suggestions.append({
                'transformer': 'NominativeAnalysisTransformer',
                'reason': 'Proper nouns detected — naming patterns and categorization analyzable'
            })
        
        if domain == 'sports':
            suggestions.append({
                'transformer': 'RelationalValueTransformer',
                'reason': 'Sports comparisons benefit from complementarity analysis'
            })
        
        # Always suggest linguistic for sufficient text
        if len(text.split()) > 20:
            suggestions.append({
                'transformer': 'LinguisticPatternsTransformer',
                'reason': 'Text length sufficient for linguistic pattern analysis'
            })
        
        return suggestions
