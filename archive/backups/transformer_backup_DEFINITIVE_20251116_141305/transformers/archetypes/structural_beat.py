"""
Structural Beat Transformer

Detects 3-act structure, 5-act structure, and Save the Cat beats.
Measures timing, pacing, and structural adherence.

Based on:
- Aristotle's Poetics (335 BCE)
- Syd Field's Three-Act Structure (1979)
- Blake Snyder's Save the Cat (2005)

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
from typing import List, Dict
from ..base import NarrativeTransformer


class StructuralBeatTransformer(NarrativeTransformer):
    """
    Detect structural patterns and beat timing.
    Extracts ~35 features for pacing and structure analysis.
    """
    
    def __init__(self, use_learned_weights=False, learned_weights=None):
        super().__init__(
            narrative_id="structural_beat",
            description="3-act, 5-act, Save the Cat beats"
        )
        
        self.use_learned_weights = use_learned_weights
        self.learned_weights = learned_weights or {}
        
        # Save the Cat 15 beats with expected positions
        self.save_the_cat_beats = {
            'opening_image': {'position': 0.01, 'markers': ['first', 'opening', 'begin'], 'weight': 0.7},
            'theme_stated': {'position': 0.05, 'markers': ['point', 'lesson', 'about'], 'weight': 0.6},
            'setup': {'position': 0.08, 'markers': ['introduce', 'establish', 'world'], 'weight': 0.8},
            'catalyst': {'position': 0.12, 'markers': ['suddenly', 'then', 'happened', 'crisis'], 'weight': 1.0},
            'debate': {'position': 0.20, 'markers': ['should', 'could', 'hesitate', 'uncertain'], 'weight': 0.7},
            'break_into_two': {'position': 0.25, 'markers': ['decided', 'committed', 'began', 'entered'], 'weight': 1.0},
            'b_story': {'position': 0.30, 'markers': ['met', 'relationship', 'love', 'friend'], 'weight': 0.6},
            'fun_and_games': {'position': 0.42, 'markers': ['adventure', 'explore', 'experience'], 'weight': 0.8},
            'midpoint': {'position': 0.50, 'markers': ['turning point', 'everything changed', 'peak'], 'weight': 1.0},
            'bad_guys_close_in': {'position': 0.63, 'markers': ['worse', 'pressure', 'closing in', 'threat'], 'weight': 0.8},
            'all_is_lost': {'position': 0.75, 'markers': ['lost', 'defeat', 'darkest', 'hopeless'], 'weight': 1.0},
            'dark_night': {'position': 0.80, 'markers': ['despair', 'alone', 'doubt', 'give up'], 'weight': 0.7},
            'break_into_three': {'position': 0.85, 'markers': ['realized', 'idea', 'answer', 'solution'], 'weight': 1.0},
            'finale': {'position': 0.92, 'markers': ['final', 'ultimate', 'battle', 'confrontation'], 'weight': 1.0},
            'final_image': {'position': 0.99, 'markers': ['end', 'finally', 'last', 'concluded'], 'weight': 0.7}
        }
        
        # Three-act structure markers
        self.three_act_markers = {
            'act1': {'start': 0.0, 'end': 0.25, 'markers': ['introduce', 'establish', 'normal', 'before']},
            'act2': {'start': 0.25, 'end': 0.75, 'markers': ['conflict', 'struggle', 'challenge', 'obstacles']},
            'act3': {'start': 0.75, 'end': 1.0, 'markers': ['resolution', 'climax', 'final', 'resolve']}
        }
        
        # Plot points
        self.plot_points = {
            'inciting_incident': {'position': 0.12, 'markers': ['incident', 'event', 'happened', 'crisis']},
            'plot_point_1': {'position': 0.25, 'markers': ['point of no return', 'committed', 'began']},
            'midpoint': {'position': 0.50, 'markers': ['middle', 'turning point', 'changed']},
            'plot_point_2': {'position': 0.75, 'markers': ['all is lost', 'lowest point', 'darkest']},
            'climax': {'position': 0.90, 'markers': ['climax', 'final', 'ultimate', 'showdown']}
        }
    
    def fit(self, X: List[str], y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        features = []
        for text in X:
            features.append(self._extract_structural_features(text))
        return np.array(features)
    
    def _extract_structural_features(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        segments = self._split_text(text, 20)
        features = {}
        
        # 1. Save the Cat beats (15 features)
        beat_scores = []
        for beat_name, beat_info in self.save_the_cat_beats.items():
            score, timing_accuracy = self._detect_beat(segments, beat_info)
            beat_scores.append(score)
            features[f'beat_{beat_name}'] = score
        
        # 2. Beat adherence score (1 feature)
        features['beat_adherence_score'] = np.mean([
            s * self.save_the_cat_beats[list(self.save_the_cat_beats.keys())[i]]['weight']
            for i, s in enumerate(beat_scores)
        ])
        
        # 3. Three-act structure (3 features)
        for act_name, act_info in self.three_act_markers.items():
            features[f'{act_name}_strength'] = self._detect_act(segments, act_info)
        
        # 4. Act balance (1 feature)
        features['act_balance'] = 1 - np.std([features['act1_strength'], 
                                               features['act2_strength'],
                                               features['act3_strength']])
        
        # 5. Plot points (5 features)
        for pp_name, pp_info in self.plot_points.items():
            score, _ = self._detect_beat(segments, pp_info)
            features[f'plot_point_{pp_name}'] = score
        
        # 6. Pacing metrics (3 features)
        features['pacing_consistency'] = self._calculate_pacing(beat_scores)
        features['structural_completeness'] = sum([1 for s in beat_scores if s > 0.5]) / len(beat_scores)
        features['timing_accuracy'] = self._calculate_timing_accuracy(segments)
        
        # 7. Overall structure quality (1 feature)
        features['overall_structure_quality'] = (
            0.40 * features['beat_adherence_score'] +
            0.30 * features['structural_completeness'] +
            0.30 * features['act_balance']
        )
        
        return np.array(list(features.values()))
    
    def _split_text(self, text: str, n: int) -> List[str]:
        words = text.split()
        seg_size = max(1, len(words) // n)
        segments = []
        for i in range(n):
            start = i * seg_size
            end = start + seg_size if i < n-1 else len(words)
            segments.append(' '.join(words[start:end]).lower())
        return segments
    
    def _detect_beat(self, segments: List[str], beat_info: Dict) -> tuple:
        """Detect beat presence and timing accuracy."""
        expected_pos = beat_info['position']
        markers = beat_info['markers']
        
        # Find markers in segments
        segment_scores = []
        for seg in segments:
            score = sum([1 for m in markers if m in seg])
            segment_scores.append(score)
        
        if sum(segment_scores) == 0:
            return 0.0, 0.0
        
        # Find peak
        peak_idx = np.argmax(segment_scores)
        actual_pos = peak_idx / len(segments)
        
        # Timing accuracy
        timing_accuracy = max(0, 1 - abs(actual_pos - expected_pos) / 0.2)
        
        # Presence score
        presence = min(1.0, sum(segment_scores) / 3)
        
        return presence * 0.7 + timing_accuracy * 0.3, timing_accuracy
    
    def _detect_act(self, segments: List[str], act_info: Dict) -> float:
        """Detect act strength."""
        start_idx = int(act_info['start'] * len(segments))
        end_idx = int(act_info['end'] * len(segments))
        
        act_text = ' '.join(segments[start_idx:end_idx])
        marker_count = sum([act_text.count(m) for m in act_info['markers']])
        
        return min(1.0, marker_count / 3)
    
    def _calculate_pacing(self, beat_scores: List[float]) -> float:
        """Calculate pacing consistency."""
        if len(beat_scores) < 2:
            return 1.0
        
        # Low variance = consistent pacing
        variance = np.var(beat_scores)
        return max(0, 1 - variance)
    
    def _calculate_timing_accuracy(self, segments: List[str]) -> float:
        """Overall timing accuracy."""
        # Simple metric based on segment length variance
        lengths = [len(seg.split()) for seg in segments]
        variance = np.var(lengths)
        normalized_variance = variance / (np.mean(lengths) ** 2 + 1e-10)
        return max(0, 1 - normalized_variance)
    
    def get_feature_names(self) -> List[str]:
        names = []
        for beat in self.save_the_cat_beats.keys():
            names.append(f'beat_{beat}')
        names.append('beat_adherence_score')
        names.extend(['act1_strength', 'act2_strength', 'act3_strength', 'act_balance'])
        for pp in self.plot_points.keys():
            names.append(f'plot_point_{pp}')
        names.extend(['pacing_consistency', 'structural_completeness', 
                     'timing_accuracy', 'overall_structure_quality'])
        return names
    
    def learn_weights_from_data(self, X: List[str], y: np.ndarray, method='correlation') -> Dict[str, float]:
        """Learn empirical beat importance weights."""
        from scipy.stats import pearsonr
        
        features = self.transform(X)
        feature_names = self.get_feature_names()
        learned_weights = {}
        
        for i, name in enumerate(feature_names):
            if name.startswith('beat_') and name != 'beat_adherence_score':
                corr, _ = pearsonr(features[:, i], y)
                learned_weights[name] = abs(corr)
        
        self.learned_weights = learned_weights
        self.use_learned_weights = True
        return learned_weights

