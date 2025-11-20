"""
Complexity Scorer

Calculates instance complexity scores that determine π_effective variation.

Complexity factors:
- Evidence ambiguity: How clear is the evidence?
- Precedent clarity: How well-established are the rules?
- Instance novelty: How unprecedented is this situation?
- Factual disputes: How much disagreement on facts?
- Outcome variance: How predictable is the result?

Simple instances (low complexity): Evidence/rules dominate, π_effective < π_base
Complex instances (high complexity): Narrative decides, π_effective > π_base

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.story_instance import StoryInstance


class ComplexityScorer:
    """
    Calculate complexity scores for story instances.
    
    Complexity determines how much π varies from domain baseline.
    """
    
    def __init__(self, domain: Optional[str] = None):
        """
        Initialize complexity scorer.
        
        Parameters
        ----------
        domain : str, optional
            Domain name for domain-specific scoring
        """
        self.domain = domain
        
        # Domain-specific complexity indicators
        self.domain_indicators = self._load_domain_indicators()
    
    def calculate_complexity(
        self,
        instance: StoryInstance,
        narrative_text: Optional[str] = None
    ) -> float:
        """
        Calculate overall complexity score for instance.
        
        Parameters
        ----------
        instance : StoryInstance
            Instance to score
        narrative_text : str, optional
            Narrative text (uses instance.narrative_text if not provided)
        
        Returns
        -------
        float
            Complexity score (0-1)
            - 0: Very simple, clear, straightforward
            - 1: Highly complex, ambiguous, contested
        """
        if narrative_text is None:
            narrative_text = instance.narrative_text
        
        # Calculate component complexities
        evidence_ambiguity = self._calculate_evidence_ambiguity(narrative_text, instance)
        precedent_clarity = self._calculate_precedent_clarity(narrative_text, instance)
        instance_novelty = self._calculate_instance_novelty(narrative_text, instance)
        factual_disputes = self._calculate_factual_disputes(narrative_text, instance)
        outcome_variance = self._calculate_outcome_variance(instance)
        
        # Store in instance
        instance.complexity_factors = {
            'evidence_ambiguity': evidence_ambiguity,
            'precedent_clarity_inverse': 1.0 - precedent_clarity,  # Higher = more complex
            'instance_novelty': instance_novelty,
            'factual_disputes': factual_disputes,
            'outcome_variance': outcome_variance
        }
        
        # Weight and combine
        weights = self._get_complexity_weights(instance.domain)
        
        complexity = (
            weights['evidence'] * evidence_ambiguity +
            weights['precedent'] * (1.0 - precedent_clarity) +
            weights['novelty'] * instance_novelty +
            weights['disputes'] * factual_disputes +
            weights['variance'] * outcome_variance
        )
        
        # Clip to [0, 1]
        complexity = np.clip(complexity, 0.0, 1.0)
        
        return complexity
    
    def _calculate_evidence_ambiguity(
        self,
        narrative_text: str,
        instance: StoryInstance
    ) -> float:
        """
        Calculate evidence ambiguity (0 = clear, 1 = ambiguous).
        
        Indicators of ambiguity:
        - Hedging language ("perhaps", "possibly", "unclear")
        - Multiple interpretations mentioned
        - Conflicting evidence
        - Lack of definitive statements
        """
        text_lower = narrative_text.lower()
        
        # Ambiguity indicators
        hedging_words = ['perhaps', 'possibly', 'maybe', 'unclear', 'uncertain',
                        'ambiguous', 'disputed', 'contested', 'debatable', 'questionable']
        
        conflict_phrases = ['on the other hand', 'however', 'but', 'although',
                           'conflicting', 'contradictory', 'mixed']
        
        interpretation_phrases = ['could be interpreted', 'one view', 'another view',
                                 'some argue', 'others suggest']
        
        # Count indicators
        hedging_count = sum(1 for word in hedging_words if word in text_lower)
        conflict_count = sum(1 for phrase in conflict_phrases if phrase in text_lower)
        interpretation_count = sum(1 for phrase in interpretation_phrases if phrase in text_lower)
        
        # Definitive language (reduces ambiguity)
        definitive_words = ['clearly', 'definitely', 'certainly', 'obviously',
                           'undoubtedly', 'indisputably']
        definitive_count = sum(1 for word in definitive_words if word in text_lower)
        
        # Calculate ambiguity score
        ambiguity_indicators = hedging_count + conflict_count * 2 + interpretation_count * 2
        clarity_indicators = definitive_count * 2
        
        word_count = len(text_lower.split())
        if word_count < 10:
            return 0.5  # Default for very short text
        
        ambiguity = (ambiguity_indicators - clarity_indicators) / (word_count / 10)
        ambiguity = np.clip(ambiguity, 0.0, 1.0)
        
        return ambiguity
    
    def _calculate_precedent_clarity(
        self,
        narrative_text: str,
        instance: StoryInstance
    ) -> float:
        """
        Calculate precedent/rule clarity (0 = unclear, 1 = very clear).
        
        Indicators of clarity:
        - Established rules mentioned
        - Prior examples referenced
        - Clear guidelines
        - Standard procedures
        """
        text_lower = narrative_text.lower()
        
        # Clarity indicators
        precedent_words = ['precedent', 'established', 'standard', 'traditional',
                          'conventional', 'typical', 'usual', 'normal', 'routine']
        
        rule_words = ['rule', 'law', 'regulation', 'guideline', 'procedure',
                     'protocol', 'policy', 'standard operating']
        
        reference_phrases = ['previously', 'in the past', 'historically',
                            'as before', 'according to', 'following']
        
        # Count indicators
        precedent_count = sum(1 for word in precedent_words if word in text_lower)
        rule_count = sum(1 for word in rule_words if word in text_lower)
        reference_count = sum(1 for phrase in reference_phrases if phrase in text_lower)
        
        # Novel/unprecedented indicators (reduce clarity)
        novelty_words = ['unprecedented', 'first time', 'never before', 'unique',
                        'novel', 'new', 'unusual', 'rare', 'exceptional']
        novelty_count = sum(1 for word in novelty_words if word in text_lower)
        
        # Calculate clarity
        clarity_indicators = precedent_count + rule_count * 2 + reference_count
        novelty_indicators = novelty_count * 2
        
        word_count = len(text_lower.split())
        if word_count < 10:
            return 0.5
        
        clarity = (clarity_indicators - novelty_indicators) / (word_count / 10)
        clarity = np.clip(clarity + 0.5, 0.0, 1.0)  # Baseline at 0.5
        
        return clarity
    
    def _calculate_instance_novelty(
        self,
        narrative_text: str,
        instance: StoryInstance
    ) -> float:
        """
        Calculate instance novelty (0 = routine, 1 = unprecedented).
        
        Indicators of novelty:
        - "First time" language
        - Unprecedented situations
        - New contexts
        - Unique circumstances
        """
        text_lower = narrative_text.lower()
        
        # Novelty indicators
        novelty_phrases = [
            'first time', 'never before', 'unprecedented', 'unique',
            'first ever', 'historic', 'groundbreaking', 'revolutionary',
            'novel', 'new', 'innovative', 'breakthrough'
        ]
        
        routine_words = [
            'routine', 'typical', 'standard', 'usual', 'normal',
            'regular', 'ordinary', 'conventional', 'common'
        ]
        
        novelty_count = sum(1 for phrase in novelty_phrases if phrase in text_lower)
        routine_count = sum(1 for word in routine_words if word in text_lower)
        
        word_count = len(text_lower.split())
        if word_count < 10:
            return 0.3  # Default low novelty
        
        novelty = (novelty_count * 3 - routine_count) / (word_count / 20)
        novelty = np.clip(novelty + 0.3, 0.0, 1.0)  # Baseline at 0.3
        
        return novelty
    
    def _calculate_factual_disputes(
        self,
        narrative_text: str,
        instance: StoryInstance
    ) -> float:
        """
        Calculate factual dispute level (0 = agreed facts, 1 = contested).
        
        Indicators:
        - Disagreement language
        - Multiple versions of events
        - Contested facts
        - Conflicting accounts
        """
        text_lower = narrative_text.lower()
        
        # Dispute indicators
        dispute_words = [
            'dispute', 'disagree', 'contest', 'challenge', 'deny',
            'refute', 'contradict', 'oppose', 'conflict'
        ]
        
        disagreement_phrases = [
            'not clear', 'in question', 'disputed', 'contested',
            'versions differ', 'accounts vary', 'conflicting'
        ]
        
        agreement_words = [
            'agree', 'undisputed', 'accepted', 'acknowledged',
            'confirmed', 'verified', 'established'
        ]
        
        dispute_count = sum(1 for word in dispute_words if word in text_lower)
        disagreement_count = sum(1 for phrase in disagreement_phrases if phrase in text_lower)
        agreement_count = sum(1 for word in agreement_words if word in text_lower)
        
        word_count = len(text_lower.split())
        if word_count < 10:
            return 0.2  # Default low disputes
        
        disputes = (dispute_count * 2 + disagreement_count * 3 - agreement_count * 2) / (word_count / 15)
        disputes = np.clip(disputes + 0.2, 0.0, 1.0)  # Baseline at 0.2
        
        return disputes
    
    def _calculate_outcome_variance(
        self,
        instance: StoryInstance
    ) -> float:
        """
        Calculate outcome variance/predictability (0 = predictable, 1 = variable).
        
        Uses instance metadata and domain characteristics.
        """
        # Domain-specific variance
        domain = instance.domain
        
        # High variance domains
        high_variance_domains = ['startups', 'movies', 'novels', 'music', 'supreme_court']
        if domain in high_variance_domains:
            base_variance = 0.7
        
        # Medium variance domains
        medium_variance_domains = ['nba', 'nfl', 'tennis', 'golf', 'oscars']
        elif domain in medium_variance_domains:
            base_variance = 0.5
        
        # Low variance domains
        else:
            base_variance = 0.3
        
        # Adjust based on instance context
        if instance.context:
            # High stakes increase variance
            stakes = instance.context.get('stakes', 'normal')
            if stakes in ['championship', 'high', 'critical']:
                base_variance += 0.2
            
            # Close competition increases variance
            closeness = instance.context.get('closeness', 'normal')
            if closeness in ['very_close', 'tied', 'contested']:
                base_variance += 0.1
        
        return np.clip(base_variance, 0.0, 1.0)
    
    def _get_complexity_weights(self, domain: str) -> Dict[str, float]:
        """
        Get domain-specific weights for complexity components.
        
        Parameters
        ----------
        domain : str
            Domain name
        
        Returns
        -------
        dict
            Weights for each complexity component
        """
        # Default weights
        default_weights = {
            'evidence': 0.30,
            'precedent': 0.25,
            'novelty': 0.20,
            'disputes': 0.15,
            'variance': 0.10
        }
        
        # Domain-specific adjustments
        if domain == 'supreme_court':
            # Legal domain: precedent and disputes matter most
            return {
                'evidence': 0.20,
                'precedent': 0.35,
                'novelty': 0.15,
                'disputes': 0.25,
                'variance': 0.05
            }
        elif domain in ['golf', 'tennis', 'nba', 'nfl']:
            # Sports: evidence (performance) dominates
            return {
                'evidence': 0.40,
                'precedent': 0.20,
                'novelty': 0.15,
                'disputes': 0.10,
                'variance': 0.15
            }
        elif domain in ['novels', 'movies', 'oscars']:
            # Creative/subjective: novelty and variance matter
            return {
                'evidence': 0.15,
                'precedent': 0.15,
                'novelty': 0.35,
                'disputes': 0.15,
                'variance': 0.20
            }
        else:
            return default_weights
    
    def _load_domain_indicators(self) -> Dict:
        """Load domain-specific complexity indicators."""
        # Placeholder for future enhancement
        # Could load from domain configs or learned patterns
        return {}
    
    def batch_calculate(
        self,
        instances: List[StoryInstance]
    ) -> Dict[str, float]:
        """
        Calculate complexity for multiple instances.
        
        Parameters
        ----------
        instances : list of StoryInstance
            Instances to score
        
        Returns
        -------
        dict
            {instance_id: complexity_score}
        """
        results = {}
        
        for instance in instances:
            complexity = self.calculate_complexity(instance)
            results[instance.instance_id] = complexity
        
        return results

