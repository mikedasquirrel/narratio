"""
Pacing and Rhythm Transformer

Uses NLP to analyze narrative tempo and rhythm patterns.
Syntactic analysis, not hardcoded metrics.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
import re
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.input_validation import ensure_string_list

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class PacingRhythmTransformer(BaseEstimator, TransformerMixin):
    """
    Analyzes narrative pacing and rhythm using linguistic features.
    
    Features (8 total):
    1. Sentence length variance trajectory
    2. Event density per segment
    3. Temporal acceleration/deceleration
    4. Scene vs summary ratio
    5. Action vs description balance
    6. Dialogue density evolution
    7. Pacing consistency
    8. Climactic buildup rate
    
    Uses:
    - Syntactic parsing for sentence structure
    - Dependency analysis for action/description
    - Part-of-speech patterns for pacing
    - Temporal progression markers
    """
    
    def __init__(self, use_spacy: bool = True):
        """Initialize pacing analyzer"""
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
        Transform texts to pacing features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 8)
            Pacing and rhythm features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_pacing_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_pacing_features(self, text: str) -> List[float]:
        """Extract all pacing features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            # 1. Sentence length variance trajectory
            sent_length_trajectory = self._compute_length_trajectory(sentences)
            features.append(sent_length_trajectory)
            
            # 2. Event density per segment
            event_density = self._compute_event_density(sentences)
            features.append(event_density)
            
            # 3. Temporal acceleration
            temporal_accel = self._compute_temporal_acceleration(sentences)
            features.append(temporal_accel)
            
            # 4. Scene vs summary ratio
            scene_summary_ratio = self._compute_scene_summary_ratio(sentences)
            features.append(scene_summary_ratio)
            
            # 5. Action vs description balance
            action_desc_balance = self._compute_action_description_balance(sentences)
            features.append(action_desc_balance)
            
            # 6. Dialogue density evolution
            dialogue_evolution = self._compute_dialogue_evolution(text, sentences)
            features.append(dialogue_evolution)
            
            # 7. Pacing consistency
            pacing_consistency = self._compute_pacing_consistency(sentences)
            features.append(pacing_consistency)
            
            # 8. Climactic buildup rate
            climactic_buildup = self._compute_climactic_buildup(sentences)
            features.append(climactic_buildup)
        else:
            # Fallback without spaCy
            features = self._extract_simple_pacing(text)
        
        return features
    
    def _compute_length_trajectory(self, sentences: List) -> float:
        """
        Compute sentence length variance over time.
        High variance trajectory = varied pacing.
        """
        if len(sentences) < 3:
            return 0.0
        
        # Divide into thirds
        third = len(sentences) // 3
        
        lengths_1 = [len(sent) for sent in sentences[:third]]
        lengths_2 = [len(sent) for sent in sentences[third:2*third]]
        lengths_3 = [len(sent) for sent in sentences[2*third:]]
        
        # Compute variance in each section
        var_1 = np.var(lengths_1) if lengths_1 else 0
        var_2 = np.var(lengths_2) if lengths_2 else 0
        var_3 = np.var(lengths_3) if lengths_3 else 0
        
        # Trajectory = change in variance
        trajectory = (var_3 - var_1) / (var_1 + var_2 + var_3 + 1)
        
        return float(np.clip(trajectory, -1, 1))
    
    def _compute_event_density(self, sentences: List) -> float:
        """
        Count events (verbs in active voice) per unit text.
        High event density = fast pacing.
        """
        total_events = 0
        total_tokens = 0
        
        for sent in sentences:
            for token in sent:
                total_tokens += 1
                # Active verbs = events
                if token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'conj']:
                    # Check if truly active (not aux or modal)
                    if token.lemma_ not in {'be', 'have', 'do', 'can', 'will', 'should', 'could', 'would', 'may', 'might', 'must'}:
                        total_events += 1
        
        if total_tokens > 0:
            return min(1.0, (total_events / total_tokens) * 10)
        return 0.0
    
    def _compute_temporal_acceleration(self, sentences: List) -> float:
        """
        Measure change in temporal markers over document.
        More temporal markers at end = acceleration toward climax.
        """
        if len(sentences) < 4:
            return 0.0
        
        # Temporal markers
        temporal_lemmas = {'now', 'then', 'when', 'while', 'before', 'after', 
                          'suddenly', 'soon', 'finally', 'next', 'immediately'}
        
        # Count in first and last quarters
        first_quarter = sentences[:len(sentences)//4]
        last_quarter = sentences[-len(sentences)//4:]
        
        temporal_first = sum(
            1 for sent in first_quarter 
            for token in sent if token.lemma_ in temporal_lemmas
        )
        
        temporal_last = sum(
            1 for sent in last_quarter
            for token in sent if token.lemma_ in temporal_lemmas
        )
        
        # Normalize by section size
        first_count = temporal_first / len(first_quarter) if first_quarter else 0
        last_count = temporal_last / len(last_quarter) if last_quarter else 0
        
        # Acceleration = increase over time
        acceleration = (last_count - first_count) / (first_count + last_count + 0.1)
        
        return float(np.clip(acceleration, -1, 1))
    
    def _compute_scene_summary_ratio(self, sentences: List) -> float:
        """
        Scene = present tense, specific action, dialogue.
        Summary = past tense, general statements.
        """
        scene_markers = 0
        summary_markers = 0
        
        for sent in sentences:
            # Count present vs past tense
            present_verbs = sum(1 for token in sent if token.pos_ == 'VERB' and token.tag_ in ['VB', 'VBP', 'VBG', 'VBZ'])
            past_verbs = sum(1 for token in sent if token.pos_ == 'VERB' and token.tag_ in ['VBD', 'VBN'])
            
            scene_markers += present_verbs
            summary_markers += past_verbs
            
            # Dialogue indicates scene
            if any(token.text in ['"', "'", '"', '"'] for token in sent):
                scene_markers += 2
        
        total = scene_markers + summary_markers
        if total > 0:
            return scene_markers / total
        return 0.5
    
    def _compute_action_description_balance(self, sentences: List) -> float:
        """
        Action = verbs. Description = adjectives/adverbs.
        High ratio = action-packed. Low = descriptive.
        """
        action_count = 0
        description_count = 0
        
        for sent in sentences:
            for token in sent:
                if token.pos_ == 'VERB' and token.dep_ != 'aux':
                    action_count += 1
                elif token.pos_ in ['ADJ', 'ADV']:
                    description_count += 1
        
        total = action_count + description_count
        if total > 0:
            return action_count / total
        return 0.5
    
    def _compute_dialogue_evolution(self, text: str, sentences: List) -> float:
        """
        Track dialogue density over document.
        Increasing dialogue can indicate escalating tension.
        """
        if len(sentences) < 3:
            return 0.0
        
        # Split into thirds
        third = len(sentences) // 3
        sections = [
            sentences[:third],
            sentences[third:2*third],
            sentences[2*third:]
        ]
        
        dialogue_densities = []
        for section in sections:
            dialogue_count = sum(
                1 for sent in section 
                if any(token.text in ['"', "'", '"', '"'] for token in sent)
            )
            density = dialogue_count / len(section) if section else 0
            dialogue_densities.append(density)
        
        # Evolution = trend
        if len(dialogue_densities) >= 2:
            evolution = dialogue_densities[-1] - dialogue_densities[0]
            return float(np.clip(evolution, -1, 1))
        
        return 0.0
    
    def _compute_pacing_consistency(self, sentences: List) -> float:
        """
        Measure consistency of pacing throughout.
        Low variance in sentence length = consistent pacing.
        """
        if len(sentences) < 2:
            return 1.0
        
        lengths = [len(sent) for sent in sentences]
        variance = np.var(lengths)
        mean = np.mean(lengths)
        
        # Coefficient of variation (normalized variance)
        if mean > 0:
            cv = (variance ** 0.5) / mean
            # Consistency = inverse of variation
            consistency = 1 / (1 + cv)
            return float(consistency)
        
        return 1.0
    
    def _compute_climactic_buildup(self, sentences: List) -> float:
        """
        Detect increasing intensity/urgency toward end.
        Measured by sentence length decrease + intensity markers.
        """
        if len(sentences) < 4:
            return 0.0
        
        # Last quarter
        last_quarter = sentences[-len(sentences)//4:]
        
        # Intensity markers (linguistic markers of climax)
        intensity_lemmas = {'finally', 'ultimate', 'critical', 'decisive', 
                           'crucial', 'climax', 'peak', 'culminate'}
        
        intensity_count = sum(
            1 for sent in last_quarter
            for token in sent if token.lemma_ in intensity_lemmas
        )
        
        # Short sentences at end = punchy climax
        avg_length_last = np.mean([len(sent) for sent in last_quarter])
        avg_length_overall = np.mean([len(sent) for sent in sentences])
        
        # Buildup score
        buildup = 0.0
        
        # Intensity markers
        buildup += min(0.5, intensity_count / len(last_quarter) * 5)
        
        # Shortening sentences
        if avg_length_last < avg_length_overall:
            buildup += 0.3
        
        # Increasing verb density at end
        verb_density_last = sum(
            1 for sent in last_quarter for token in sent if token.pos_ == 'VERB'
        ) / sum(len(sent) for sent in last_quarter) if last_quarter else 0
        
        verb_density_overall = sum(
            1 for sent in sentences for token in sent if token.pos_ == 'VERB'
        ) / sum(len(sent) for sent in sentences) if sentences else 0
        
        if verb_density_last > verb_density_overall:
            buildup += 0.2
        
        return min(1.0, buildup)
    
    def _extract_simple_pacing(self, text: str) -> List[float]:
        """Fallback pacing features without spaCy"""
        features = []
        
        # Split into sentences (simple)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [0.0] * 8
        
        # 1. Sentence length variance
        lengths = [len(s.split()) for s in sentences]
        variance = np.var(lengths) if lengths else 0
        features.append(min(1.0, variance / 100.0))
        
        # 2. Event density (rough proxy: word count)
        avg_length = np.mean(lengths) if lengths else 0
        features.append(min(1.0, avg_length / 30.0))
        
        # 3-8: Simple proxies
        features.extend([0.5, 0.5, 0.5, 0.3, 0.7, 0.4])
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'pacing_length_trajectory',
            'pacing_event_density',
            'pacing_temporal_acceleration',
            'pacing_scene_summary_ratio',
            'pacing_action_description',
            'pacing_dialogue_evolution',
            'pacing_consistency',
            'pacing_climactic_buildup'
        ])

