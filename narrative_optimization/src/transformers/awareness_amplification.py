"""
Awareness Amplification Transformer

Detects awareness of narrative potential energy and measures
amplification effects.

Key Insight: There are TWO types of awareness:
1. θ_resistance: Awareness of narrative determinism (suppresses effects)
2. θ_amplification: Awareness of narrative potential (amplifies outcomes)

This transformer extracts θ_amplification features.

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list


class AwarenessAmplificationTransformer(NarrativeTransformer):
    """
    Extract awareness amplification features (15 features).
    
    Distinguishes awareness OF POTENTIAL (amplifies) from
    awareness OF DETERMINISM (suppresses).
    
    Features (15 total):
    -------------------
    1. explicit_awareness: "I know this is...", "aware that..."
    2. meta_narrative: "story of my life", "couldn't write this"
    3. stakes_consciousness: "everything on the line"
    4. opportunity_recognition: "chance of a lifetime"
    5. position_awareness: "underdog role", "favorite burden"
    6. historical_consciousness: "legacy moment", "for history"
    7. observer_awareness: "everyone watching", "all eyes on"
    8. transformation_awareness: "becoming who I'm meant to be"
    9. temporal_awareness: "now or never", "this is the moment"
    10. dramatic_awareness: "you couldn't write this script"
    11. potential_recognition: "been building to this"
    12. convergence_awareness: "all threads coming together"
    13. structural_awareness: "third act", "climax", "defining"
    14. audience_consciousness: "for everyone who believed"
    15. amplification_score: aggregate (0-1)
    
    Parameters
    ----------
    domain_config : DomainConfig, optional
        Domain-specific configuration
    """
    
    def __init__(self, domain_config=None):
        super().__init__(
            narrative_id="awareness_amplification",
            description="Awareness amplification: detecting consciousness of narrative potential"
        )
        
        self.domain_config = domain_config
        
        # Pattern 1: Explicit Awareness
        self.explicit_awareness_patterns = [
            r'\bi know\b.*\bis\b',
            r'\baware that\b',
            r'\brealize\b.*\bthis\b',
            r'\bunderstand\b.*\bmoment\b',
            r'\bconscious of\b',
            r'\brecognize\b.*\bopportunity\b'
        ]
        
        # Pattern 2: Meta-Narrative
        self.meta_narrative_patterns = [
            'story of my life',
            'couldn\'t write this',
            'like a movie',
            'like a novel',
            'scripted',
            'storybook',
            'fairy tale',
            'narrative',
            'chapter',
            'plot twist'
        ]
        
        # Pattern 3: Stakes Consciousness
        self.stakes_patterns = [
            'everything on the line',
            'all or nothing',
            'everything riding on',
            'stakes are high',
            'high stakes',
            'life or death',
            'do or die'
        ]
        
        # Pattern 4: Opportunity Recognition
        self.opportunity_patterns = [
            'chance of a lifetime',
            'once in a lifetime',
            'opportunity of',
            'golden opportunity',
            'rare chance',
            'won\'t come again',
            'don\'t get many'
        ]
        
        # Pattern 5: Position Awareness
        self.position_patterns = [
            'underdog',
            'favorite',
            'expected to',
            'pressure of',
            'burden of',
            'weight of expectations',
            'role of'
        ]
        
        # Pattern 6: Historical Consciousness
        self.historical_patterns = [
            'legacy',
            'for history',
            'history books',
            'will be remembered',
            'legendary',
            'historic',
            'immortal',
            'all-time',
            'forever'
        ]
        
        # Pattern 7: Observer Awareness
        self.observer_patterns = [
            'everyone watching',
            'all eyes on',
            'world is watching',
            'spotlight',
            'center stage',
            'in front of',
            'audience'
        ]
        
        # Pattern 8: Transformation Awareness
        self.transformation_patterns = [
            'becoming',
            'transforming',
            'meant to be',
            'destiny',
            'destined',
            'fulfilling',
            'realizing potential',
            'coming into'
        ]
        
        # Pattern 9: Temporal Awareness
        self.temporal_patterns = [
            'now or never',
            'this is the moment',
            'time is now',
            'moment has arrived',
            'moment of truth',
            'critical moment',
            'pivotal moment'
        ]
        
        # Pattern 10: Dramatic Awareness
        self.dramatic_patterns = [
            'couldn\'t write',
            'can\'t make this up',
            'unbelievable',
            'dramatic',
            'incredible moment',
            'amazing',
            'spectacular'
        ]
        
        # Pattern 11: Potential Recognition
        self.potential_patterns = [
            'building to this',
            'led to this',
            'prepared for',
            'ready for',
            'culmination',
            'payoff'
        ]
        
        # Pattern 12: Convergence Awareness
        self.convergence_patterns = [
            'coming together',
            'all threads',
            'pieces falling',
            'aligning',
            'converging',
            'confluence'
        ]
        
        # Pattern 13: Structural Awareness
        self.structural_patterns = [
            'third act',
            'climax',
            'finale',
            'crescendo',
            'peak',
            'apex',
            'defining moment'
        ]
        
        # Pattern 14: Audience Consciousness
        self.audience_patterns = [
            'for everyone',
            'for those who',
            'for all who',
            'dedicated to',
            'in honor of',
            'for my'
        ]
    
    def fit(self, X, y=None):
        """
        Learn awareness amplification patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        X = ensure_string_list(X)
        
        # Corpus statistics
        corpus_stats = {
            'avg_explicit_awareness': 0,
            'avg_meta_narrative': 0,
            'avg_stakes_consciousness': 0,
            'avg_opportunity_recognition': 0,
            'avg_amplification': 0
        }
        
        for text in X:
            text_lower = text.lower()
            words = text_lower.split()
            n_words = len(words) + 1
            
            # Count patterns
            explicit_count = sum(len(re.findall(p, text_lower)) for p in self.explicit_awareness_patterns)
            meta_count = sum(1 for p in self.meta_narrative_patterns if p in text_lower)
            stakes_count = sum(1 for p in self.stakes_patterns if p in text_lower)
            opportunity_count = sum(1 for p in self.opportunity_patterns if p in text_lower)
            
            corpus_stats['avg_explicit_awareness'] += explicit_count / n_words
            corpus_stats['avg_meta_narrative'] += meta_count / n_words
            corpus_stats['avg_stakes_consciousness'] += stakes_count / n_words
            corpus_stats['avg_opportunity_recognition'] += opportunity_count / n_words
        
        # Average
        n_docs = len(X)
        for key in corpus_stats:
            corpus_stats[key] /= n_docs
        
        self.metadata['corpus_stats'] = corpus_stats
        self.metadata['n_documents'] = n_docs
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to awareness amplification features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array
            Awareness amplification feature matrix (n_documents, 15)
        """
        self._validate_fitted()
        X = ensure_string_list(X)
        
        features = []
        
        for text in X:
            text_lower = text.lower()
            words = text_lower.split()
            n_words = len(words) + 1
            
            doc_features = []
            
            # 1. Explicit awareness
            explicit_count = sum(len(re.findall(p, text_lower)) for p in self.explicit_awareness_patterns)
            doc_features.append(explicit_count / n_words)
            
            # 2. Meta-narrative
            meta_count = sum(1 for p in self.meta_narrative_patterns if p in text_lower)
            doc_features.append(meta_count / n_words)
            
            # 3. Stakes consciousness
            stakes_count = sum(1 for p in self.stakes_patterns if p in text_lower)
            doc_features.append(stakes_count / n_words)
            
            # 4. Opportunity recognition
            opportunity_count = sum(1 for p in self.opportunity_patterns if p in text_lower)
            doc_features.append(opportunity_count / n_words)
            
            # 5. Position awareness
            position_count = sum(1 for p in self.position_patterns if p in text_lower)
            doc_features.append(position_count / n_words)
            
            # 6. Historical consciousness
            historical_count = sum(1 for p in self.historical_patterns if p in text_lower)
            doc_features.append(historical_count / n_words)
            
            # 7. Observer awareness
            observer_count = sum(1 for p in self.observer_patterns if p in text_lower)
            doc_features.append(observer_count / n_words)
            
            # 8. Transformation awareness
            transformation_count = sum(1 for p in self.transformation_patterns if p in text_lower)
            doc_features.append(transformation_count / n_words)
            
            # 9. Temporal awareness
            temporal_count = sum(1 for p in self.temporal_patterns if p in text_lower)
            doc_features.append(temporal_count / n_words)
            
            # 10. Dramatic awareness
            dramatic_count = sum(1 for p in self.dramatic_patterns if p in text_lower)
            doc_features.append(dramatic_count / n_words)
            
            # 11. Potential recognition
            potential_count = sum(1 for p in self.potential_patterns if p in text_lower)
            doc_features.append(potential_count / n_words)
            
            # 12. Convergence awareness
            convergence_count = sum(1 for p in self.convergence_patterns if p in text_lower)
            doc_features.append(convergence_count / n_words)
            
            # 13. Structural awareness
            structural_count = sum(1 for p in self.structural_patterns if p in text_lower)
            doc_features.append(structural_count / n_words)
            
            # 14. Audience consciousness
            audience_count = sum(1 for p in self.audience_patterns if p in text_lower)
            doc_features.append(audience_count / n_words)
            
            # 15. Amplification score (aggregate)
            # Weight the components
            amplification_score = (
                0.15 * doc_features[0] +   # explicit awareness
                0.10 * doc_features[1] +   # meta-narrative
                0.12 * doc_features[2] +   # stakes consciousness
                0.12 * doc_features[3] +   # opportunity recognition
                0.08 * doc_features[4] +   # position awareness
                0.10 * doc_features[5] +   # historical consciousness
                0.08 * doc_features[6] +   # observer awareness
                0.08 * doc_features[7] +   # transformation awareness
                0.10 * doc_features[8] +   # temporal awareness
                0.05 * doc_features[9] +   # dramatic awareness
                0.02 * doc_features[10] +  # potential recognition
                0.02 * doc_features[11] +  # convergence awareness
                0.03 * doc_features[12] +  # structural awareness
                0.02 * doc_features[13]    # audience consciousness
            )
            
            # Normalize to [0, 1]
            amplification_score = min(1.0, amplification_score * 100)  # Scale up
            doc_features.append(amplification_score)
            
            features.append(doc_features)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        Returns
        -------
        list of str
            Feature names
        """
        return [
            'awareness_explicit_awareness',
            'awareness_meta_narrative',
            'awareness_stakes_consciousness',
            'awareness_opportunity_recognition',
            'awareness_position_awareness',
            'awareness_historical_consciousness',
            'awareness_observer_awareness',
            'awareness_transformation_awareness',
            'awareness_temporal_awareness',
            'awareness_dramatic_awareness',
            'awareness_potential_recognition',
            'awareness_convergence_awareness',
            'awareness_structural_awareness',
            'awareness_audience_consciousness',
            'awareness_amplification_score'
        ]
    
    def calculate_amplification_effect(
        self,
        amplification_score: float,
        potential_energy: float,
        consciousness: float = 1.0
    ) -> float:
        """
        Calculate amplification effect on outcome.
        
        Formula: amplification = 1 + θ_amp × potential × consciousness
        
        Parameters
        ----------
        amplification_score : float
            θ_amp from features (0-1)
        potential_energy : float
            Narrative potential energy (from NarrativePotentialTransformer)
        consciousness : float
            Did actor explicitly recognize potential? (0 or 1)
        
        Returns
        -------
        float
            Amplification multiplier (1.0 - 2.0+)
        """
        amplification = 1.0 + (amplification_score * potential_energy * consciousness)
        return amplification

