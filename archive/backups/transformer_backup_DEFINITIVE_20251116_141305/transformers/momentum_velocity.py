"""
Momentum and Velocity Transformer

Uses NLP to analyze narrative momentum and change velocity.
Temporal analysis and trajectory detection.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.input_validation import ensure_string_list

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class MomentumVelocityTransformer(BaseEstimator, TransformerMixin):
    """
    Analyzes narrative momentum and velocity of change.
    
    Features (5 total):
    1. Narrative momentum score
    2. Velocity of change
    3. Acceleration indicators
    4. Trajectory predictors
    5. Momentum sustainability
    
    Uses:
    - Temporal analysis of change markers
    - Sentiment trajectory analysis
    - Syntactic patterns for momentum
    - Event density acceleration
    """
    
    def __init__(self, use_spacy: bool = True):
        """Initialize momentum analyzer"""
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except:
                    self.use_spacy = False
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to momentum features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 5)
            Momentum and velocity features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_momentum_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_momentum_features(self, text: str) -> List[float]:
        """Extract all momentum features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            # 1. Narrative momentum
            momentum = self._compute_narrative_momentum(sentences)
            features.append(momentum)
            
            # 2. Velocity of change
            velocity = self._compute_velocity_of_change(sentences)
            features.append(velocity)
            
            # 3. Acceleration indicators
            acceleration = self._compute_acceleration(sentences)
            features.append(acceleration)
            
            # 4. Trajectory predictors
            trajectory = self._compute_trajectory(sentences)
            features.append(trajectory)
            
            # 5. Momentum sustainability
            sustainability = self._compute_sustainability(sentences)
            features.append(sustainability)
        else:
            # Fallback
            features = [0.5, 0.4, 0.3, 0.5, 0.4]
        
        return features
    
    def _compute_narrative_momentum(self, sentences: List) -> float:
        """
        Compute overall narrative momentum.
        Combines event density, temporal markers, and forward motion.
        """
        if not sentences:
            return 0.0
        
        momentum_score = 0.0
        
        # Event density (action verbs)
        action_verbs = sum(
            1 for sent in sentences
            for token in sent
            if token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'conj']
            and token.lemma_ not in {'be', 'have', 'do'}
        )
        
        momentum_score += min(0.4, action_verbs / len(sentences) / 3)
        
        # Forward temporal markers
        forward_lemmas = {'next', 'then', 'after', 'following', 'subsequently',
                         'soon', 'later', 'eventually', 'ultimately'}
        
        forward_count = sum(
            1 for sent in sentences
            for token in sent if token.lemma_ in forward_lemmas
        )
        
        momentum_score += min(0.3, forward_count / len(sentences) * 3)
        
        # Progressive aspects (ongoing action)
        progressive_verbs = sum(
            1 for sent in sentences
            for token in sent
            if token.pos_ == 'VERB' and token.tag_ == 'VBG'
        )
        
        momentum_score += min(0.3, progressive_verbs / len(sentences) * 2)
        
        return min(1.0, momentum_score)
    
    def _compute_velocity_of_change(self, sentences: List) -> float:
        """
        Measure rate of change over narrative.
        """
        if len(sentences) < 4:
            return 0.0
        
        # Track change markers over time
        change_lemmas = {'change', 'transform', 'become', 'turn', 'shift',
                        'evolve', 'develop', 'progress', 'advance', 'move'}
        
        # Divide into quarters
        quarter = len(sentences) // 4
        quarters = [
            sentences[i*quarter:(i+1)*quarter]
            for i in range(4)
        ]
        
        change_counts = []
        for quarter_sents in quarters:
            if quarter_sents:
                count = sum(
                    1 for sent in quarter_sents
                    for token in sent if token.lemma_ in change_lemmas
                )
                change_counts.append(count / len(quarter_sents))
            else:
                change_counts.append(0)
        
        # Velocity = average rate of change
        if change_counts:
            velocity = np.mean(change_counts)
            return min(1.0, velocity * 5)
        
        return 0.0
    
    def _compute_acceleration(self, sentences: List) -> float:
        """
        Detect increasing pace/intensity toward end.
        """
        if len(sentences) < 4:
            return 0.0
        
        # Compare first and last quarters
        quarter = len(sentences) // 4
        first_quarter = sentences[:quarter]
        last_quarter = sentences[-quarter:]
        
        # Measure event density in each
        first_events = sum(
            1 for sent in first_quarter
            for token in sent
            if token.pos_ == 'VERB' and token.dep_ != 'aux'
        )
        
        last_events = sum(
            1 for sent in last_quarter
            for token in sent
            if token.pos_ == 'VERB' and token.dep_ != 'aux'
        )
        
        # Acceleration = increase in density
        first_density = first_events / len(first_quarter) if first_quarter else 0
        last_density = last_events / len(last_quarter) if last_quarter else 0
        
        acceleration = (last_density - first_density) / (first_density + 0.1)
        
        return float(np.clip(acceleration, 0, 1))
    
    def _compute_trajectory(self, sentences: List) -> float:
        """
        Predict narrative trajectory (upward/forward vs stagnant).
        """
        if len(sentences) < 3:
            return 0.5
        
        # Analyze sentiment trajectory
        positive_lemmas = {'good', 'better', 'improve', 'win', 'succeed', 'achieve'}
        negative_lemmas = {'bad', 'worse', 'decline', 'lose', 'fail', 'struggle'}
        
        # Track sentiment over thirds
        third = len(sentences) // 3
        thirds = [
            sentences[i*third:(i+1)*third]
            for i in range(3)
        ]
        
        sentiments = []
        for third_sents in thirds:
            if third_sents:
                pos = sum(1 for s in third_sents for t in s if t.lemma_ in positive_lemmas)
                neg = sum(1 for s in third_sents for t in s if t.lemma_ in negative_lemmas)
                sentiments.append(pos - neg)
        
        # Trajectory = trend direction
        if len(sentiments) >= 2:
            trend = sentiments[-1] - sentiments[0]
            # Normalize to 0-1 (positive trajectory)
            return float(np.clip((trend + 3) / 6, 0, 1))
        
        return 0.5
    
    def _compute_sustainability(self, sentences: List) -> float:
        """
        Measure whether momentum is sustained or episodic.
        """
        if len(sentences) < 5:
            return 0.5
        
        # Track action density across document
        segments = 5
        segment_size = len(sentences) // segments
        
        action_densities = []
        for i in range(segments):
            segment = sentences[i*segment_size:(i+1)*segment_size]
            if segment:
                actions = sum(
                    1 for sent in segment
                    for token in sent
                    if token.pos_ == 'VERB' and token.dep_ == 'ROOT'
                )
                density = actions / len(segment)
                action_densities.append(density)
        
        # Sustainability = consistency (low variance)
        if action_densities:
            variance = np.var(action_densities)
            mean = np.mean(action_densities)
            
            # Low coefficient of variation = sustained
            if mean > 0:
                cv = (variance ** 0.5) / mean
                sustainability = 1 / (1 + cv)
                return float(sustainability)
        
        return 0.5
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'momentum_narrative_momentum',
            'momentum_velocity_of_change',
            'momentum_acceleration',
            'momentum_trajectory',
            'momentum_sustainability'
        ])

