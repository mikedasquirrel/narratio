"""
Advanced Linguistic Patterns Transformer

Deep analysis of narrative voice, agency, temporal orientation, and emotional trajectory.
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter, defaultdict
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list


class LinguisticPatternsTransformer(NarrativeTransformer):
    """
    Analyzes advanced linguistic patterns in narratives.
    
    Tests the hypothesis that how a story is told (voice, agency, temporality)
    matters as much as what is told.
    
    Features extracted (36 total):
    - Narrative voice consistency (POV stability)
    - Temporal markers (past/present/future orientation)
    - Agency patterns (active vs passive voice)
    - Emotional trajectory (sentiment evolution)
    - Linguistic complexity evolution
    - **NEW: Credibility/Authority markers (10)**: expert terminology, hedging, certainty, citations, credentials, jargon, precision, evidence
    
    Parameters
    ----------
    track_evolution : bool
        Whether to track feature evolution across document
    n_segments : int
        Number of segments to divide document into for evolution tracking
    """
    
    def __init__(
        self,
        track_evolution: bool = True,
        n_segments: int = 3
    ):
        super().__init__(
            narrative_id="linguistic_patterns",
            description="Advanced linguistic analysis: voice, agency, temporality, emotion"
        )
        
        self.track_evolution = track_evolution
        self.n_segments = n_segments
        
        # Pattern definitions
        self.first_person_patterns = [r'\bi\b', r'\bme\b', r'\bmy\b', r'\bmine\b', r'\bmyself\b', r'\bwe\b', r'\bus\b', r'\bour\b']
        self.second_person_patterns = [r'\byou\b', r'\byour\b', r'\byours\b', r'\byourself\b']
        self.third_person_patterns = [r'\bhe\b', r'\bshe\b', r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\bhis\b', r'\bher\b']
        
        self.past_tense_patterns = [r'\bed\b', r'\bwas\b', r'\bwere\b', r'\bhad\b', r'\bdid\b']
        self.present_tense_patterns = [r'\bis\b', r'\bare\b', r'\bam\b', r'\bdoes\b', r'\bdo\b']
        self.future_tense_patterns = [r'\bwill\b', r'\bshall\b', r'\bgoing to\b', r'\bgonna\b']
        
        self.passive_indicators = [r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b', r'\bbeen\s+\w+ed\b', r'\bbe\s+\w+ed\b']
        self.active_indicators = [r'\bi\s+\w+', r'\bwe\s+\w+', r'\byou\s+\w+', r'\bthey\s+\w+']
        
        self.positive_words = ['good', 'great', 'love', 'happy', 'excellent', 'wonderful', 'amazing', 'best', 'better', 'positive']
        self.negative_words = ['bad', 'hate', 'sad', 'terrible', 'awful', 'worst', 'worse', 'negative', 'poor', 'difficult']
        
        self.complexity_markers = {
            'subordinate_conj': [r'\balthough\b', r'\bthough\b', r'\bbecause\b', r'\bsince\b', r'\bunless\b', r'\bwhile\b'],
            'relative_pronouns': [r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b', r'\bwhose\b'],
            'modal_verbs': [r'\bcould\b', r'\bshould\b', r'\bwould\b', r'\bmight\b', r'\bmay\b', r'\bmust\b']
        }
    
    def fit(self, X, y=None):
        """
        Learn linguistic patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        # Compute corpus-level statistics for normalization
        corpus_stats = {
            'avg_first_person': 0,
            'avg_past_tense': 0,
            'avg_complexity': 0,
            'avg_sentiment': 0
        }
        
        for text in X:
            # Ensure text is string
            text = str(text) if not isinstance(text, str) else text
            text_lower = text.lower()
            words = text_lower.split()
            n_words = len(words) + 1
            
            # First person usage
            first_person_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.first_person_patterns)
            corpus_stats['avg_first_person'] += first_person_count / n_words
            
            # Past tense usage
            past_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.past_tense_patterns)
            corpus_stats['avg_past_tense'] += past_count / n_words
            
            # Complexity
            complexity_count = sum(
                sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
                for patterns in self.complexity_markers.values()
            )
            corpus_stats['avg_complexity'] += complexity_count / n_words
            
            # Sentiment
            pos_count = sum(1 for word in self.positive_words if word in text_lower)
            neg_count = sum(1 for word in self.negative_words if word in text_lower)
            corpus_stats['avg_sentiment'] += (pos_count - neg_count) / n_words
        
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
        Transform documents to linguistic pattern features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array
            Linguistic feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        features_list = []
        
        for text in X:
            # Ensure text is string
            text = str(text) if not isinstance(text, str) else text
            doc_features = self._extract_linguistic_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_linguistic_features(self, text: str) -> List[float]:
        """Extract linguistic features from a single document."""
        features = []
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) + 1  # Avoid division by zero
        
        # 1. Narrative Voice Features
        first_person_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.first_person_patterns)
        second_person_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.second_person_patterns)
        third_person_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.third_person_patterns)
        
        total_person = first_person_count + second_person_count + third_person_count + 1
        
        features.append(first_person_count / n_words)  # First person density
        features.append(second_person_count / n_words)  # Second person density
        features.append(third_person_count / n_words)  # Third person density
        
        # Voice consistency (entropy of person distribution)
        person_dist = np.array([first_person_count, second_person_count, third_person_count]) + 1
        person_dist = person_dist / person_dist.sum()
        voice_entropy = -np.sum(person_dist * np.log(person_dist + 1e-10))
        features.append(voice_entropy)  # Lower = more consistent voice
        
        # 2. Temporal Orientation
        past_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.past_tense_patterns)
        present_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.present_tense_patterns)
        future_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.future_tense_patterns)
        
        total_tense = past_count + present_count + future_count + 1
        
        features.append(past_count / n_words)  # Past orientation
        features.append(present_count / n_words)  # Present orientation
        features.append(future_count / n_words)  # Future orientation
        
        # Temporal balance
        tense_dist = np.array([past_count, present_count, future_count]) + 1
        tense_dist = tense_dist / tense_dist.sum()
        temporal_entropy = -np.sum(tense_dist * np.log(tense_dist + 1e-10))
        features.append(temporal_entropy)
        
        # 3. Agency Patterns
        passive_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.passive_indicators)
        active_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.active_indicators)
        
        features.append(active_count / n_words)  # Active voice density
        features.append(passive_count / n_words)  # Passive voice density
        
        # Agency ratio (active / (active + passive))
        agency_ratio = active_count / (active_count + passive_count + 1)
        features.append(agency_ratio)
        
        # 4. Emotional Trajectory
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        sentiment_score = (pos_count - neg_count) / n_words
        features.append(sentiment_score)
        
        # Emotional intensity
        emotional_intensity = (pos_count + neg_count) / n_words
        features.append(emotional_intensity)
        
        # 5. Linguistic Complexity
        subordinate_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.complexity_markers['subordinate_conj'])
        relative_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.complexity_markers['relative_pronouns'])
        modal_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.complexity_markers['modal_verbs'])
        
        features.append(subordinate_count / n_words)  # Subordination density
        features.append(relative_count / n_words)  # Relative clause density
        features.append(modal_count / n_words)  # Modality density
        
        # Overall complexity score
        complexity_score = (subordinate_count + relative_count + modal_count) / n_words
        features.append(complexity_score)
        
        # 6. Evolution Features (if enabled)
        if self.track_evolution:
            evolution_features = self._extract_evolution_features(text)
            features.extend(evolution_features)
        
        # === NEW: CREDIBILITY/AUTHORITY MARKERS (10 features) ===
        
        # Expert terminology density
        expert_words = ['research', 'study', 'analysis', 'data', 'evidence', 'findings', 'results', 'proven', 'demonstrated']
        expert_count = sum(1 for w in expert_words if w in text_lower)
        features.append(expert_count / n_words)
        
        # Hedging markers (uncertainty)
        hedging_words = ['maybe', 'possibly', 'might', 'could', 'perhaps', 'probably', 'likely', 'potentially']
        hedging_count = sum(1 for w in hedging_words if w in text_lower)
        features.append(hedging_count / n_words)
        
        # Certainty markers
        certainty_words = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously', 'undoubtedly', 'proven', 'established']
        certainty_count = sum(1 for w in certainty_words if w in text_lower)
        features.append(certainty_count / n_words)
        
        # Hedging-certainty balance
        total_epistemic = hedging_count + certainty_count
        certainty_ratio = certainty_count / total_epistemic if total_epistemic > 0 else 0.5
        features.append(certainty_ratio)
        
        # Citation patterns
        citation_patterns = [r'according to', r'research shows', r'studies indicate', r'data suggests', r'evidence demonstrates']
        citation_count = sum(len(re.findall(pattern, text_lower)) for pattern in citation_patterns)
        features.append(citation_count / n_words)
        
        # Credential references
        credential_patterns = [r'\bdr\.?\b', r'\bphd\b', r'\bprof', r'\bexpert', r'\bspecialist', r'\bresearcher']
        credential_count = sum(len(re.findall(pattern, text_lower)) for pattern in credential_patterns)
        features.append(credential_count / n_words)
        
        # Technical jargon density (words > 12 characters)
        long_words = [w for w in words if len(w) > 12]
        jargon_density = len(long_words) / n_words
        features.append(jargon_density)
        
        # Precision language (specific numbers)
        number_pattern = r'\b\d+\.?\d*\b'
        number_count = len(re.findall(number_pattern, text))
        features.append(number_count / n_words)
        
        # Evidence markers
        evidence_words = ['proof', 'demonstrate', 'show', 'indicate', 'reveal', 'confirm', 'validate', 'verify']
        evidence_count = sum(1 for w in evidence_words if w in text_lower)
        features.append(evidence_count / n_words)
        
        # Overall credibility score (composite)
        credibility_score = (
            expert_count + certainty_count + citation_count + 
            credential_count + evidence_count
        ) / (n_words * 5)
        features.append(credibility_score)
        
        return features
    
    def _extract_evolution_features(self, text: str) -> List[float]:
        """Track how linguistic features evolve across the document."""
        # Split document into segments
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < self.n_segments:
            # Not enough sentences, return zeros
            return [0.0] * (self.n_segments * 3)  # 3 tracked features
        
        segment_size = len(sentences) // self.n_segments
        segments = [
            ' '.join(sentences[i*segment_size:(i+1)*segment_size])
            for i in range(self.n_segments)
        ]
        
        evolution_features = []
        
        # Track voice evolution
        voice_trajectory = []
        for segment in segments:
            segment_lower = segment.lower()
            first_person = sum(len(re.findall(p, segment_lower)) for p in self.first_person_patterns)
            voice_trajectory.append(first_person / (len(segment.split()) + 1))
        
        # Trend (are they using more first person over time?)
        if len(voice_trajectory) > 1:
            voice_trend = voice_trajectory[-1] - voice_trajectory[0]
        else:
            voice_trend = 0
        evolution_features.append(voice_trend)
        
        # Track temporal evolution
        temporal_trajectory = []
        for segment in segments:
            segment_lower = segment.lower()
            future_count = sum(len(re.findall(p, segment_lower)) for p in self.future_tense_patterns)
            temporal_trajectory.append(future_count / (len(segment.split()) + 1))
        
        temporal_trend = temporal_trajectory[-1] - temporal_trajectory[0] if len(temporal_trajectory) > 1 else 0
        evolution_features.append(temporal_trend)
        
        # Track complexity evolution
        complexity_trajectory = []
        for segment in segments:
            segment_lower = segment.lower()
            complexity_count = sum(
                sum(len(re.findall(pattern, segment_lower)) for pattern in patterns)
                for patterns in self.complexity_markers.values()
            )
            complexity_trajectory.append(complexity_count / (len(segment.split()) + 1))
        
        complexity_trend = complexity_trajectory[-1] - complexity_trajectory[0] if len(complexity_trajectory) > 1 else 0
        evolution_features.append(complexity_trend)
        
        # Variability (standard deviation across segments)
        evolution_features.append(np.std(voice_trajectory))
        evolution_features.append(np.std(temporal_trajectory))
        evolution_features.append(np.std(complexity_trajectory))
        
        # Consistency (inverse of variability)
        voice_consistency = 1 / (1 + np.std(voice_trajectory))
        temporal_consistency = 1 / (1 + np.std(temporal_trajectory))
        complexity_consistency = 1 / (1 + np.std(complexity_trajectory))
        
        evolution_features.append(voice_consistency)
        evolution_features.append(temporal_consistency)
        evolution_features.append(complexity_consistency)
        
        return evolution_features
    
    def _generate_interpretation(self):
        """Generate human-readable interpretation."""
        corpus_stats = self.metadata.get('corpus_stats', {})
        
        interpretation = (
            "Linguistic Patterns Analysis: Captures how stories are told, not just what is told. "
            f"Corpus averages - first person: {corpus_stats.get('avg_first_person', 0):.3f}, "
            f"past tense: {corpus_stats.get('avg_past_tense', 0):.3f}, "
            f"complexity: {corpus_stats.get('avg_complexity', 0):.3f}. "
            "Features include narrative voice (POV density and consistency), "
            "temporal orientation (past/present/future), agency patterns (active vs passive), "
            "emotional trajectory (sentiment and intensity), linguistic complexity "
            "(subordination, relativization, modality), and evolution over document. "
            "If this outperforms simpler approaches, it validates that linguistic choices—"
            "voice, temporality, agency—are meaningful predictors."
        )
        
        return interpretation

