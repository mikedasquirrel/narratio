"""
Self-Perception Transformer

Analyzes self-referential patterns and how narratives reflect self-understanding.
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string


class SelfPerceptionTransformer(NarrativeTransformer):
    """
    Analyzes self-referential patterns in narratives.
    
    Tests the hypothesis that how people talk about themselves—their
    self-perception as reflected in language—predicts outcomes.
    
    Features extracted:
    - First-person narrative intensity
    - Self-attribution patterns (agency, traits)
    - Growth indicators (change language)
    - Aspirational vs descriptive balance
    - Identity coherence
    
    Parameters
    ----------
    track_attribution : bool
        Whether to track attribution patterns
    track_growth : bool
        Whether to track growth/change language
    track_coherence : bool
        Whether to measure identity coherence
    """
    
    def __init__(
        self,
        track_attribution: bool = True,
        track_growth: bool = True,
        track_coherence: bool = True
    ):
        super().__init__(
            narrative_id="self_perception",
            description="Self-perception: how self-reference reveals identity and potential"
        )
        
        self.track_attribution = track_attribution
        self.track_growth = track_growth
        self.track_coherence = track_coherence
        
        # Self-reference patterns
        self.first_person_singular = [r'\bi\b', r'\bme\b', r'\bmy\b', r'\bmine\b', r'\bmyself\b']
        self.first_person_plural = [r'\bwe\b', r'\bus\b', r'\bour\b', r'\bours\b', r'\bourselves\b']
        
        # Attribution patterns
        self.positive_traits = ['good', 'strong', 'smart', 'capable', 'confident', 'skilled', 'talented', 'successful', 'effective', 'competent']
        self.negative_traits = ['bad', 'weak', 'stupid', 'incapable', 'unsure', 'unskilled', 'incompetent', 'unsuccessful', 'ineffective']
        
        # Growth/change indicators
        self.growth_words = ['grow', 'learn', 'develop', 'improve', 'progress', 'advance', 'evolve', 'become', 'transform', 'change']
        self.stasis_words = ['stay', 'remain', 'continue', 'keep', 'maintain', 'stuck', 'same', 'static', 'fixed', 'unchanging']
        
        # Aspirational language
        self.aspirational_words = ['want', 'hope', 'wish', 'dream', 'aspire', 'goal', 'aim', 'desire', 'strive', 'seek']
        self.descriptive_words = ['am', 'is', 'are', 'have', 'has', 'do', 'does', 'been', 'being']
        
        # Agency indicators
        self.high_agency = [r'\bi\s+\w+ed\b', r'\bi\s+can\b', r'\bi\s+will\b', r'\bi\s+did\b', r'\bi\s+made\b', r'\bi\s+created\b']
        self.low_agency = [r'\bi\s+was\s+\w+ed\b', r'\bi\s+can\'t\b', r'\bi\s+couldn\'t\b', r'\bit\s+happened\b', r'\bit\s+was\b']
    
    def fit(self, X, y=None):
        """
        Learn self-perception patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Corpus statistics for normalization
        corpus_stats = {
            'avg_first_person': 0,
            'avg_attribution_positive': 0,
            'avg_growth_orientation': 0,
            'avg_aspirational': 0
        }
        
        # Ensure X is list of strings
        if isinstance(X, np.ndarray):
            X = [str(x) for x in X]
        elif not isinstance(X, (list, tuple)):
            X = [str(X)]
        
        for text in X:
            # Ensure text is string
            text = str(text) if not isinstance(text, str) else text
            text_lower = text.lower()
            words = text_lower.split()
            n_words = len(words) + 1
            
            # First person usage
            first_person_count = sum(len(re.findall(p, text_lower)) for p in self.first_person_singular)
            corpus_stats['avg_first_person'] += first_person_count / n_words
            
            # Positive attribution
            pos_traits = sum(1 for word in self.positive_traits if word in text_lower)
            corpus_stats['avg_attribution_positive'] += pos_traits / n_words
            
            # Growth orientation
            growth_count = sum(1 for word in self.growth_words if word in text_lower)
            corpus_stats['avg_growth_orientation'] += growth_count / n_words
            
            # Aspirational language
            asp_count = sum(1 for word in self.aspirational_words if word in text_lower)
            corpus_stats['avg_aspirational'] += asp_count / n_words
        
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
        Transform documents to self-perception features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array
            Self-perception feature matrix
        """
        self._validate_fitted()
        
        features_list = []
        
        # Ensure X is list of strings
        if isinstance(X, np.ndarray):
            X = [str(x) for x in X]
        elif not isinstance(X, (list, tuple)):
            X = [str(X)]
        
        for text in X:
            # Ensure text is string
            text = str(text) if not isinstance(text, str) else text
            doc_features = self._extract_self_perception_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_self_perception_features(self, text: str) -> List[float]:
        """Extract self-perception features from a document."""
        # Ensure text is string
        text = str(text) if not isinstance(text, str) else text
        features = []
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) + 1
        
        # 1. First-Person Intensity
        fp_singular_count = sum(len(re.findall(p, text_lower)) for p in self.first_person_singular)
        fp_plural_count = sum(len(re.findall(p, text_lower)) for p in self.first_person_plural)
        
        features.append(fp_singular_count / n_words)  # Individual self-focus
        features.append(fp_plural_count / n_words)  # Collective self-focus
        
        # Self-focus ratio (singular / total first person)
        total_fp = fp_singular_count + fp_plural_count + 1
        self_focus_ratio = fp_singular_count / total_fp
        features.append(self_focus_ratio)
        
        # 2. Self-Attribution Patterns
        if self.track_attribution:
            pos_trait_count = sum(1 for word in self.positive_traits if word in text_lower)
            neg_trait_count = sum(1 for word in self.negative_traits if word in text_lower)
            
            features.append(pos_trait_count / n_words)  # Positive self-attribution
            features.append(neg_trait_count / n_words)  # Negative self-attribution
            
            # Attribution balance (positive - negative)
            attribution_balance = (pos_trait_count - neg_trait_count) / n_words
            features.append(attribution_balance)
            
            # Self-attribution confidence (using positive traits)
            self_confidence = pos_trait_count / (pos_trait_count + neg_trait_count + 1)
            features.append(self_confidence)
        
        # 3. Growth Orientation
        if self.track_growth:
            growth_count = sum(1 for word in self.growth_words if word in text_lower)
            stasis_count = sum(1 for word in self.stasis_words if word in text_lower)
            
            features.append(growth_count / n_words)  # Growth orientation
            features.append(stasis_count / n_words)  # Stasis orientation
            
            # Growth mindset score
            growth_mindset = growth_count / (growth_count + stasis_count + 1)
            features.append(growth_mindset)
        
        # 4. Aspirational vs Descriptive Balance
        asp_count = sum(1 for word in self.aspirational_words if word in text_lower)
        desc_count = sum(1 for word in self.descriptive_words if word in text_lower)
        
        features.append(asp_count / n_words)  # Aspirational density
        features.append(desc_count / n_words)  # Descriptive density
        
        # Aspirational ratio (future vs present focus)
        aspirational_ratio = asp_count / (asp_count + desc_count + 1)
        features.append(aspirational_ratio)
        
        # 5. Agency Patterns
        high_agency_count = sum(len(re.findall(p, text_lower)) for p in self.high_agency)
        low_agency_count = sum(len(re.findall(p, text_lower)) for p in self.low_agency)
        
        features.append(high_agency_count / n_words)  # High agency
        features.append(low_agency_count / n_words)  # Low agency
        
        # Agency score
        agency_score = high_agency_count / (high_agency_count + low_agency_count + 1)
        features.append(agency_score)
        
        # 6. Identity Coherence
        if self.track_coherence:
            # Coherence measured through consistency of self-reference
            # If first-person is used, is it used consistently throughout?
            
            if fp_singular_count > 0:
                # Split into segments
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) > 2:
                    # Check first-person usage in each segment
                    segment_fp = []
                    for sentence in sentences:
                        sent_lower = sentence.lower()
                        sent_fp = sum(len(re.findall(p, sent_lower)) for p in self.first_person_singular)
                        sent_words = len(sent_lower.split()) + 1
                        segment_fp.append(sent_fp / sent_words)
                    
                    # Coherence = consistency (inverse of std dev)
                    if len(segment_fp) > 1:
                        coherence = 1 / (1 + np.std(segment_fp))
                        features.append(coherence)
                    else:
                        features.append(1.0)
                else:
                    features.append(1.0)
            else:
                features.append(0.0)  # No first-person = no coherence to measure
        
        # 7. Self-Complexity
        # Variety in how self is referenced and described
        self_descriptor_types = []
        if fp_singular_count > 0:
            self_descriptor_types.append('first_person')
        if pos_trait_count > 0:
            self_descriptor_types.append('positive_traits')
        if neg_trait_count > 0:
            self_descriptor_types.append('negative_traits')
        if growth_count > 0:
            self_descriptor_types.append('growth')
        if asp_count > 0:
            self_descriptor_types.append('aspirational')
        
        self_complexity = len(self_descriptor_types)
        features.append(self_complexity)
        
        # 8. Self-Awareness Indicators
        # Meta-cognitive language: "I think I", "I realize", "I know I"
        meta_patterns = [r'\bi\s+think\s+i\b', r'\bi\s+realize\b', r'\bi\s+know\s+i\b', r'\bi\s+feel\s+i\b', r'\bi\s+believe\s+i\b']
        meta_count = sum(len(re.findall(p, text_lower)) for p in meta_patterns)
        features.append(meta_count / n_words)
        
        # 9. Self-Transformation Language
        # Explicit change: "I became", "I changed", "I transformed"
        transformation_patterns = [r'\bi\s+became\b', r'\bi\s+changed\b', r'\bi\s+transformed\b', r'\bi\s+evolved\b', r'\bi\s+grew\b']
        transformation_count = sum(len(re.findall(p, text_lower)) for p in transformation_patterns)
        features.append(transformation_count / n_words)
        
        # 10. Self-Positioning
        # How does the self position relative to others/world?
        relational_patterns = [r'\bi\s+am\s+\w+\s+than\b', r'\bcompared\s+to\b', r'\blike\s+me\b', r'\bunlike\s+me\b']
        relational_count = sum(len(re.findall(p, text_lower)) for p in relational_patterns)
        features.append(relational_count / n_words)
        
        return features
    
    def _generate_interpretation(self):
        """Generate human-readable interpretation."""
        corpus_stats = self.metadata.get('corpus_stats', {})
        
        interpretation = (
            "Self-Perception Analysis: Analyzes how self-reference reveals identity. "
            f"Corpus averages - first person: {corpus_stats.get('avg_first_person', 0):.3f}, "
            f"positive attribution: {corpus_stats.get('avg_attribution_positive', 0):.3f}, "
            f"growth orientation: {corpus_stats.get('avg_growth_orientation', 0):.3f}. "
            "Features capture first-person intensity (individual vs collective), "
            "self-attribution patterns (positive/negative traits), growth orientation "
            "(change vs stasis language), aspirational vs descriptive balance, "
            "agency patterns (high vs low control), identity coherence (consistency), "
            "self-complexity (variety of self-description), self-awareness (meta-cognition), "
            "self-transformation (explicit change language), and self-positioning "
            "(relational comparison). If this predicts outcomes, it validates that "
            "how we perceive and present ourselves matters."
        )
        
        return interpretation

