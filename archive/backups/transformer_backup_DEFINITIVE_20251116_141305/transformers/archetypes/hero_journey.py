"""
Hero's Journey Transformer

Detects Campbell's 17-stage monomyth and Vogler's 12-stage adaptation.
Measures journey completion, transformation depth, and stage coherence.

Based on:
- Joseph Campbell's "The Hero with a Thousand Faces" (1949)
- Christopher Vogler's "The Writer's Journey" (1992)

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

from ..base import NarrativeTransformer


class HeroJourneyTransformer(NarrativeTransformer):
    """
    Detect and measure Hero's Journey pattern in narratives.
    
    Extracts ~60 features:
    - 17 Campbell stage detection scores
    - 12 Vogler stage detection scores
    - Journey completion percentage
    - Stage sequential coherence
    - Transformation depth
    - Mentor presence/quality
    - Threshold crossing detection
    - Death/rebirth pattern
    - Archetypal pairing strength
    
    Compatible with π/λ/θ framework:
    - High journey completion → High π (narrativity)
    - Stage coherence → Structural component
    - Transformation depth → Temporal component
    """
    
    def __init__(self, use_learned_weights=False, learned_weights=None):
        """
        Initialize Hero's Journey Transformer.
        
        Args:
            use_learned_weights: If True, use empirically learned weights instead of
                                 Campbell's theoretical weights. Enables discovery of
                                 domain-specific importance and validation of theory.
            learned_weights: Dict mapping stage names to learned importance weights.
                           If None and use_learned_weights=True, all stages weighted equally.
        """
        super().__init__(
            narrative_id="hero_journey",
            description="Campbell's Hero's Journey + Vogler's Writer's Journey detection"
        )
        
        self.use_learned_weights = use_learned_weights
        self.learned_weights = learned_weights or {}
        
        # Campbell's 17 stages with detection patterns
        # NOTE: 'weight' here is Campbell's THEORETICAL importance
        # Can be overridden by empirical weights learned from data
        self.campbell_stages = {
            'ordinary_world': {
                'markers': [
                    'normal', 'usual', 'everyday', 'routine', 'always', 'comfortable',
                    'home', 'family', 'familiar', 'safe', 'peaceful', 'before'
                ],
                'position': 0.05,
                'weight': 0.8
            },
            'call_to_adventure': {
                'markers': [
                    'suddenly', 'one day', 'then', 'message', 'summons', 'call',
                    'adventure', 'quest', 'problem', 'crisis', 'opportunity', 'discovered'
                ],
                'position': 0.12,
                'weight': 1.0
            },
            'refusal_of_call': {
                'markers': [
                    'refuse', 'no', 'cannot', 'impossible', 'afraid', 'fear',
                    'hesitate', 'doubt', 'reluctant', 'unsure', 'resist'
                ],
                'position': 0.15,
                'weight': 0.7
            },
            'meeting_mentor': {
                'markers': [
                    'mentor', 'teacher', 'guide', 'master', 'wise', 'elder',
                    'taught', 'learned', 'trained', 'prepared', 'showed', 'advice'
                ],
                'position': 0.18,
                'weight': 0.9
            },
            'crossing_threshold': {
                'markers': [
                    'crossed', 'entered', 'passed', 'beyond', 'no turning back',
                    'committed', 'decided', 'journey began', 'left behind', 'point of no return'
                ],
                'position': 0.25,
                'weight': 1.0
            },
            'tests_allies_enemies': {
                'markers': [
                    'test', 'trial', 'challenge', 'ally', 'friend', 'enemy',
                    'opponent', 'met', 'encountered', 'faced', 'learned', 'discovered'
                ],
                'position': 0.35,
                'weight': 0.9
            },
            'approach_inmost_cave': {
                'markers': [
                    'approach', 'near', 'prepare', 'plan', 'final', 'ready',
                    'before', 'edge', 'brink', 'threshold', 'gates', 'entrance'
                ],
                'position': 0.45,
                'weight': 0.8
            },
            'ordeal': {
                'markers': [
                    'ordeal', 'crisis', 'death', 'battle', 'confrontation', 'faced',
                    'survived', 'barely', 'nearly died', 'greatest fear', 'darkest moment'
                ],
                'position': 0.50,
                'weight': 1.0
            },
            'reward': {
                'markers': [
                    'reward', 'prize', 'treasure', 'gained', 'won', 'achieved',
                    'found', 'discovered', 'claimed', 'seized', 'earned'
                ],
                'position': 0.60,
                'weight': 0.9
            },
            'road_back': {
                'markers': [
                    'return', 'back', 'home', 'escape', 'flee', 'journey home',
                    'must go back', 'leaving', 'pursued', 'chase'
                ],
                'position': 0.70,
                'weight': 0.8
            },
            'resurrection': {
                'markers': [
                    'final', 'ultimate', 'last', 'climax', 'reborn', 'transformed',
                    'changed', 'different', 'new', 'death and rebirth', 'emerged'
                ],
                'position': 0.85,
                'weight': 1.0
            },
            'return_with_elixir': {
                'markers': [
                    'returned', 'brought back', 'shared', 'taught', 'wisdom',
                    'knowledge', 'gift', 'benefited', 'changed world', 'home at last'
                ],
                'position': 0.95,
                'weight': 0.9
            },
            # Extended Campbell stages
            'supernatural_aid': {
                'markers': [
                    'magic', 'magical', 'supernatural', 'divine', 'blessed',
                    'gift', 'power', 'aided', 'helped by', 'appeared'
                ],
                'position': 0.20,
                'weight': 0.6
            },
            'belly_of_whale': {
                'markers': [
                    'swallowed', 'consumed', 'trapped', 'enclosed', 'separated',
                    'alone', 'cut off', 'no escape', 'complete separation'
                ],
                'position': 0.28,
                'weight': 0.5
            },
            'meeting_goddess': {
                'markers': [
                    'goddess', 'love', 'unconditional', 'beauty', 'perfect',
                    'divine feminine', 'mother', 'complete love', 'acceptance'
                ],
                'position': 0.55,
                'weight': 0.4
            },
            'woman_as_temptress': {
                'markers': [
                    'temptation', 'tempted', 'distraction', 'seduced', 'lured',
                    'abandon', 'forget', 'give up', 'pleasure', 'desire'
                ],
                'position': 0.65,
                'weight': 0.4
            },
            'atonement_with_father': {
                'markers': [
                    'father', 'authority', 'confronted', 'faced', 'reconciled',
                    'understood', 'accepted', 'forgave', 'ultimate power'
                ],
                'position': 0.75,
                'weight': 0.5
            }
        }
        
        # Vogler's 12 simplified stages
        self.vogler_stages = {
            'ordinary_world': {'position': 0.05, 'weight': 1.0},
            'call_to_adventure': {'position': 0.12, 'weight': 1.0},
            'refusal_of_call': {'position': 0.15, 'weight': 0.8},
            'meeting_mentor': {'position': 0.18, 'weight': 1.0},
            'crossing_threshold': {'position': 0.25, 'weight': 1.0},
            'tests_allies_enemies': {'position': 0.40, 'weight': 1.0},
            'approach_inmost_cave': {'position': 0.47, 'weight': 0.9},
            'ordeal': {'position': 0.50, 'weight': 1.0},
            'reward': {'position': 0.60, 'weight': 1.0},
            'road_back': {'position': 0.70, 'weight': 1.0},
            'resurrection': {'position': 0.85, 'weight': 1.0},
            'return_with_elixir': {'position': 0.95, 'weight': 1.0}
        }
        
        # Transformation markers
        self.transformation_patterns = {
            'before_state': ['weak', 'afraid', 'ignorant', 'innocent', 'naive', 'untested'],
            'after_state': ['strong', 'brave', 'wise', 'experienced', 'changed', 'transformed'],
            'change_verbs': ['became', 'transformed', 'changed', 'grew', 'learned', 'emerged']
        }
        
        # Mentor quality markers
        self.mentor_patterns = {
            'wise_mentor': ['wisdom', 'knowledge', 'taught', 'guided', 'prepared'],
            'protective': ['protected', 'guarded', 'defended', 'saved'],
            'gift_giving': ['gave', 'provided', 'granted', 'bestowed', 'equipped'],
            'sacrificial': ['sacrificed', 'died', 'gave life', 'martyred']
        }
        
        # Threshold/gate patterns
        self.threshold_patterns = [
            'crossed', 'passed through', 'entered', 'gateway', 'portal',
            'threshold', 'boundary', 'no return', 'committed', 'decided'
        ]
        
        # Death/rebirth patterns
        self.death_rebirth_patterns = {
            'death': ['died', 'death', 'killed', 'perished', 'fell', 'lost'],
            'rebirth': ['reborn', 'resurrected', 'returned', 'emerged', 'rose', 'awakened']
        }
    
    def fit(self, X: List[str], y=None):
        """
        Fit transformer (no-op for this rule-based transformer).
        """
        self.is_fitted_ = True
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        Extract Hero's Journey features from narratives.
        
        Args:
            X: List of narrative texts
            
        Returns:
            Feature matrix of shape (n_samples, ~60)
        """
        features = []
        
        for text in X:
            features.append(self._extract_journey_features(text))
        
        return np.array(features)
    
    def _extract_journey_features(self, text: str) -> np.ndarray:
        """
        Extract complete Hero's Journey feature vector.
        """
        text_lower = text.lower()
        text_length = len(text)
        
        # Split text into segments for position analysis
        n_segments = 20
        segments = self._split_into_segments(text, n_segments)
        
        features = {}
        
        # 1. Campbell's 17 stages (17 features)
        campbell_scores = []
        campbell_positions = []
        
        for stage_name, stage_info in self.campbell_stages.items():
            score, position = self._detect_stage(
                text_lower, segments, stage_name, stage_info
            )
            campbell_scores.append(score)
            campbell_positions.append(position)
            features[f'campbell_{stage_name}'] = score
        
        # 2. Vogler's 12 stages (12 features)
        vogler_scores = []
        vogler_positions = []
        
        for stage_name, stage_info in self.vogler_stages.items():
            # Use Campbell detection for overlapping stages
            if stage_name in self.campbell_stages:
                score = campbell_scores[list(self.campbell_stages.keys()).index(stage_name)]
                position = campbell_positions[list(self.campbell_stages.keys()).index(stage_name)]
            else:
                score, position = self._detect_stage(
                    text_lower, segments, stage_name, stage_info
                )
            vogler_scores.append(score)
            vogler_positions.append(position)
            features[f'vogler_{stage_name}'] = score
        
        # 3. Journey completion scores (3 features)
        # Use learned weights if available, otherwise use theoretical weights
        if self.use_learned_weights and self.learned_weights:
            campbell_completion = np.mean([
                s * self.learned_weights.get(stage_name, 1.0)
                for s, stage_name in zip(campbell_scores, self.campbell_stages.keys())
            ])
            vogler_completion = np.mean([
                s * self.learned_weights.get(stage_name, 1.0)
                for s, stage_name in zip(vogler_scores, self.vogler_stages.keys())
            ])
        else:
            # Use Campbell/Vogler's theoretical weights
            campbell_completion = np.mean([s * self.campbell_stages[list(self.campbell_stages.keys())[i]]['weight'] 
                                           for i, s in enumerate(campbell_scores)])
            vogler_completion = np.mean([s * self.vogler_stages[list(self.vogler_stages.keys())[i]]['weight'] 
                                         for i, s in enumerate(vogler_scores)])
        
        features['campbell_journey_completion'] = campbell_completion
        features['vogler_journey_completion'] = vogler_completion
        features['journey_completion_mean'] = (campbell_completion + vogler_completion) / 2
        
        # 4. Stage sequential coherence (2 features)
        campbell_coherence = self._calculate_sequential_coherence(
            campbell_scores, campbell_positions, list(self.campbell_stages.values())
        )
        vogler_coherence = self._calculate_sequential_coherence(
            vogler_scores, vogler_positions, list(self.vogler_stages.values())
        )
        
        features['campbell_sequential_coherence'] = campbell_coherence
        features['vogler_sequential_coherence'] = vogler_coherence
        
        # 5. Transformation depth (4 features)
        transformation_features = self._analyze_transformation(text_lower, segments)
        features.update(transformation_features)
        
        # 6. Mentor presence and quality (5 features)
        mentor_features = self._analyze_mentor(text_lower)
        features.update(mentor_features)
        
        # 7. Threshold crossing strength (2 features)
        threshold_features = self._analyze_threshold_crossing(text_lower, segments)
        features.update(threshold_features)
        
        # 8. Death/rebirth pattern (3 features)
        death_rebirth_features = self._analyze_death_rebirth(text_lower, segments)
        features.update(death_rebirth_features)
        
        # 9. Key stage presence (5 features)
        features['has_call_to_adventure'] = float(campbell_scores[1] > 0.5)
        features['has_crossing_threshold'] = float(campbell_scores[4] > 0.5)
        features['has_ordeal'] = float(campbell_scores[7] > 0.5)
        features['has_resurrection'] = float(campbell_scores[10] > 0.5)
        features['has_return_with_elixir'] = float(campbell_scores[11] > 0.5)
        
        # 10. Core journey completeness (1 feature)
        core_stages = ['call_to_adventure', 'crossing_threshold', 'ordeal', 
                      'resurrection', 'return_with_elixir']
        core_present = sum([features[f'has_{stage}'] for stage in core_stages])
        features['core_journey_completeness'] = core_present / len(core_stages)
        
        # 11. Three-act alignment (3 features)
        features['act1_stages'] = np.mean(campbell_scores[0:4])  # Ordinary World → Mentor
        features['act2_stages'] = np.mean(campbell_scores[4:10])  # Threshold → Road Back
        features['act3_stages'] = np.mean(campbell_scores[10:12])  # Resurrection → Return
        
        # 12. Journey arc quality (1 feature)
        # Good arc: clear beginning, strong middle, satisfying end
        arc_quality = (
            0.25 * features['act1_stages'] +
            0.50 * features['act2_stages'] +
            0.25 * features['act3_stages']
        )
        features['journey_arc_quality'] = arc_quality
        
        # Convert to array
        feature_array = np.array(list(features.values()))
        
        return feature_array
    
    def _split_into_segments(self, text: str, n: int) -> List[str]:
        """Split text into n roughly equal segments."""
        words = text.split()
        segment_size = max(1, len(words) // n)
        
        segments = []
        for i in range(n):
            start = i * segment_size
            end = start + segment_size if i < n-1 else len(words)
            segments.append(' '.join(words[start:end]).lower())
        
        return segments
    
    def _detect_stage(self, text: str, segments: List[str], 
                     stage_name: str, stage_info: Dict) -> Tuple[float, float]:
        """
        Detect presence and position of a journey stage.
        
        Returns:
            (confidence_score, normalized_position)
        """
        markers = stage_info.get('markers', [])
        expected_position = stage_info['position']
        
        if not markers:
            return 0.0, expected_position
        
        # Count markers in each segment
        segment_scores = []
        for seg in segments:
            count = sum([1 for marker in markers if marker in seg])
            segment_scores.append(count)
        
        if sum(segment_scores) == 0:
            return 0.0, expected_position
        
        # Find peak segment
        peak_segment = np.argmax(segment_scores)
        actual_position = peak_segment / len(segments)
        
        # Calculate confidence based on:
        # 1. Number of markers found
        # 2. Concentration in expected region
        # 3. Position accuracy
        
        total_markers = sum(segment_scores)
        marker_density = total_markers / len(markers)
        
        # Check if peak is near expected position (within ±0.15)
        position_accuracy = max(0, 1 - abs(actual_position - expected_position) / 0.15)
        
        # Confidence score
        confidence = min(1.0, (
            0.50 * min(1.0, marker_density) +
            0.30 * (segment_scores[peak_segment] / (total_markers + 1)) +
            0.20 * position_accuracy
        ))
        
        return confidence, actual_position
    
    def _calculate_sequential_coherence(self, scores: List[float], 
                                       positions: List[float],
                                       stage_infos: List[Dict]) -> float:
        """
        Measure how well stages follow expected sequential order.
        """
        # Only consider stages that are detected (score > 0.3)
        present_stages = [(i, positions[i]) for i, score in enumerate(scores) if score > 0.3]
        
        if len(present_stages) < 2:
            return 1.0  # Can't measure with less than 2 stages
        
        # Count order violations
        violations = 0
        for i in range(len(present_stages) - 1):
            idx_a, pos_a = present_stages[i]
            idx_b, pos_b = present_stages[i + 1]
            
            # If stage A should come before stage B (by index) but doesn't (by position)
            if idx_a < idx_b and pos_a > pos_b:
                violations += 1
        
        max_violations = len(present_stages) * (len(present_stages) - 1) / 2
        coherence = 1 - (violations / max_violations) if max_violations > 0 else 1.0
        
        return coherence
    
    def _analyze_transformation(self, text: str, segments: List[str]) -> Dict[str, float]:
        """
        Analyze character transformation depth.
        """
        features = {}
        
        # Before state (early segments)
        early_text = ' '.join(segments[:4])
        before_count = sum([1 for marker in self.transformation_patterns['before_state'] 
                           if marker in early_text])
        
        # After state (late segments)
        late_text = ' '.join(segments[-4:])
        after_count = sum([1 for marker in self.transformation_patterns['after_state'] 
                          if marker in late_text])
        
        # Explicit change markers
        change_count = sum([text.count(verb) for verb in self.transformation_patterns['change_verbs']])
        
        features['transformation_before_state'] = min(1.0, before_count / 3)
        features['transformation_after_state'] = min(1.0, after_count / 3)
        features['transformation_explicit_change'] = min(1.0, change_count / 3)
        features['transformation_depth'] = (
            features['transformation_before_state'] +
            features['transformation_after_state'] +
            features['transformation_explicit_change']
        ) / 3
        
        return features
    
    def _analyze_mentor(self, text: str) -> Dict[str, float]:
        """
        Analyze mentor presence and quality.
        """
        features = {}
        
        # Overall mentor presence
        mentor_count = sum([text.count(marker) for marker in self.mentor_patterns['wise_mentor']])
        features['mentor_presence'] = min(1.0, mentor_count / 3)
        
        # Mentor qualities
        for quality, markers in self.mentor_patterns.items():
            count = sum([text.count(marker) for marker in markers])
            features[f'mentor_{quality}'] = min(1.0, count / 2)
        
        return features
    
    def _analyze_threshold_crossing(self, text: str, segments: List[str]) -> Dict[str, float]:
        """
        Analyze threshold crossing strength and position.
        """
        features = {}
        
        # Overall threshold presence
        threshold_count = sum([text.count(pattern) for pattern in self.threshold_patterns])
        features['threshold_crossing_strength'] = min(1.0, threshold_count / 2)
        
        # Position (should be around Act I/II boundary, ~25%)
        segment_counts = [sum([1 for pattern in self.threshold_patterns if pattern in seg]) 
                         for seg in segments]
        if sum(segment_counts) > 0:
            peak_segment = np.argmax(segment_counts)
            actual_position = peak_segment / len(segments)
            # Ideal position is 0.25 (25%)
            position_accuracy = max(0, 1 - abs(actual_position - 0.25) / 0.20)
            features['threshold_crossing_timing'] = position_accuracy
        else:
            features['threshold_crossing_timing'] = 0.0
        
        return features
    
    def _analyze_death_rebirth(self, text: str, segments: List[str]) -> Dict[str, float]:
        """
        Analyze death/rebirth pattern strength.
        """
        features = {}
        
        # Death markers
        death_count = sum([text.count(marker) for marker in self.death_rebirth_patterns['death']])
        features['death_pattern_strength'] = min(1.0, death_count / 2)
        
        # Rebirth markers
        rebirth_count = sum([text.count(marker) for marker in self.death_rebirth_patterns['rebirth']])
        features['rebirth_pattern_strength'] = min(1.0, rebirth_count / 2)
        
        # Death should come before rebirth (temporal check)
        death_positions = []
        rebirth_positions = []
        
        for i, seg in enumerate(segments):
            if any(marker in seg for marker in self.death_rebirth_patterns['death']):
                death_positions.append(i)
            if any(marker in seg for marker in self.death_rebirth_patterns['rebirth']):
                rebirth_positions.append(i)
        
        # Check if rebirth comes after death
        if death_positions and rebirth_positions:
            avg_death_pos = np.mean(death_positions)
            avg_rebirth_pos = np.mean(rebirth_positions)
            correct_order = float(avg_rebirth_pos > avg_death_pos)
        else:
            correct_order = 0.0
        
        features['death_rebirth_pattern'] = (
            features['death_pattern_strength'] +
            features['rebirth_pattern_strength'] +
            correct_order
        ) / 3
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all extracted features.
        """
        names = []
        
        # Campbell stages
        for stage in self.campbell_stages.keys():
            names.append(f'campbell_{stage}')
        
        # Vogler stages
        for stage in self.vogler_stages.keys():
            names.append(f'vogler_{stage}')
        
        # Aggregate features
        names.extend([
            'campbell_journey_completion',
            'vogler_journey_completion',
            'journey_completion_mean',
            'campbell_sequential_coherence',
            'vogler_sequential_coherence',
            'transformation_before_state',
            'transformation_after_state',
            'transformation_explicit_change',
            'transformation_depth',
            'mentor_presence',
            'mentor_wise_mentor',
            'mentor_protective',
            'mentor_gift_giving',
            'mentor_sacrificial',
            'threshold_crossing_strength',
            'threshold_crossing_timing',
            'death_pattern_strength',
            'rebirth_pattern_strength',
            'death_rebirth_pattern',
            'has_call_to_adventure',
            'has_crossing_threshold',
            'has_ordeal',
            'has_resurrection',
            'has_return_with_elixir',
            'core_journey_completeness',
            'act1_stages',
            'act2_stages',
            'act3_stages',
            'journey_arc_quality'
        ])
        
        return names
    
    def get_stage_analysis(self, text: str) -> Dict:
        """
        Get detailed analysis of Hero's Journey stages for a single narrative.
        
        Useful for visualization and interpretation.
        
        Returns:
            Dictionary with stage-by-stage breakdown
        """
        text_lower = text.lower()
        segments = self._split_into_segments(text, 20)
        
        analysis = {
            'campbell': {},
            'vogler': {},
            'summary': {}
        }
        
        # Analyze Campbell stages
        for stage_name, stage_info in self.campbell_stages.items():
            score, position = self._detect_stage(
                text_lower, segments, stage_name, stage_info
            )
            analysis['campbell'][stage_name] = {
                'present': score > 0.5,
                'confidence': score,
                'actual_position': position,
                'expected_position': stage_info['position'],
                'timing_accuracy': 1 - abs(position - stage_info['position'])
            }
        
        # Analyze Vogler stages
        for stage_name, stage_info in self.vogler_stages.items():
            if stage_name in self.campbell_stages:
                score, position = (
                    analysis['campbell'][stage_name]['confidence'],
                    analysis['campbell'][stage_name]['actual_position']
                )
            else:
                score, position = self._detect_stage(
                    text_lower, segments, stage_name, stage_info
                )
            
            analysis['vogler'][stage_name] = {
                'present': score > 0.5,
                'confidence': score,
                'actual_position': position,
                'expected_position': stage_info['position']
            }
        
        # Summary statistics
        campbell_completion = np.mean([
            s['confidence'] * self.campbell_stages[name]['weight']
            for name, s in analysis['campbell'].items()
        ])
        
        vogler_completion = np.mean([
            s['confidence'] * self.vogler_stages[name]['weight']
            for name, s in analysis['vogler'].items()
        ])
        
        analysis['summary'] = {
            'campbell_completion': campbell_completion,
            'vogler_completion': vogler_completion,
            'overall_completion': (campbell_completion + vogler_completion) / 2,
            'stages_present_campbell': sum([1 for s in analysis['campbell'].values() if s['present']]),
            'stages_present_vogler': sum([1 for s in analysis['vogler'].values() if s['present']]),
            'follows_hero_journey': campbell_completion > 0.60 or vogler_completion > 0.60
        }
        
        return analysis


    def learn_weights_from_data(self, X: List[str], y: np.ndarray, 
                                method='correlation') -> Dict[str, float]:
        """
        Learn empirical feature weights from data.
        
        This discovers which stages actually predict success in this domain,
        allowing validation of Campbell's theoretical weights and discovery
        of domain-specific patterns.
        
        Args:
            X: List of narrative texts
            y: Outcomes (1=success, 0=failure or continuous scores)
            method: 'correlation', 'mutual_info', or 'regression'
            
        Returns:
            Dictionary mapping stage names to learned importance weights
        """
        from scipy.stats import pearsonr
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.linear_model import Ridge
        
        # Extract features
        features = self.transform(X)
        
        # Get stage feature indices
        feature_names = self.get_feature_names()
        stage_indices = {
            name: i for i, name in enumerate(feature_names)
            if name.startswith('campbell_') and not name.startswith('campbell_journey')
        }
        
        learned_weights = {}
        
        if method == 'correlation':
            # Simple correlation-based weights
            for stage_name, idx in stage_indices.items():
                corr, pval = pearsonr(features[:, idx], y)
                # Absolute correlation as importance
                learned_weights[stage_name.replace('campbell_', '')] = abs(corr)
        
        elif method == 'mutual_info':
            # Mutual information (handles non-linear relationships)
            stage_features = features[:, list(stage_indices.values())]
            
            if len(np.unique(y)) <= 10:  # Classification
                mi_scores = mutual_info_classif(stage_features, y)
            else:  # Regression
                mi_scores = mutual_info_regression(stage_features, y)
            
            # Normalize to [0, 1]
            mi_scores = mi_scores / (mi_scores.max() + 1e-10)
            
            for i, (stage_name, idx) in enumerate(stage_indices.items()):
                learned_weights[stage_name.replace('campbell_', '')] = mi_scores[i]
        
        elif method == 'regression':
            # Regression coefficients as importance
            stage_features = features[:, list(stage_indices.values())]
            model = Ridge(alpha=1.0)
            model.fit(stage_features, y)
            
            # Absolute coefficients as importance
            coeffs = np.abs(model.coef_)
            # Normalize
            coeffs = coeffs / (coeffs.max() + 1e-10)
            
            for i, (stage_name, idx) in enumerate(stage_indices.items()):
                learned_weights[stage_name.replace('campbell_', '')] = coeffs[i]
        
        # Store learned weights
        self.learned_weights = learned_weights
        self.use_learned_weights = True
        
        return learned_weights
    
    def compare_theoretical_vs_empirical(self) -> Dict:
        """
        Compare Campbell's theoretical weights to empirically learned weights.
        
        Enables validation of classical theory and discovery of new insights.
        
        Returns:
            Dictionary with comparative analysis
        """
        if not self.learned_weights:
            raise ValueError("Must call learn_weights_from_data first")
        
        comparison = {
            'stages': {},
            'summary': {}
        }
        
        # Compare each stage
        for stage_name, stage_info in self.campbell_stages.items():
            theoretical_weight = stage_info['weight']
            empirical_weight = self.learned_weights.get(stage_name, 0.5)
            
            deviation = theoretical_weight - empirical_weight
            
            comparison['stages'][stage_name] = {
                'theoretical_weight': theoretical_weight,
                'empirical_weight': empirical_weight,
                'deviation': deviation,
                'campbell_overvalued': deviation > 0.2,
                'campbell_undervalued': deviation < -0.2,
                'agreement': abs(deviation) < 0.2
            }
        
        # Summary statistics
        deviations = [s['deviation'] for s in comparison['stages'].values()]
        comparison['summary'] = {
            'mean_absolute_deviation': np.mean(np.abs(deviations)),
            'correlation': np.corrcoef(
                [s['theoretical_weight'] for s in comparison['stages'].values()],
                [s['empirical_weight'] for s in comparison['stages'].values()]
            )[0, 1],
            'stages_agreeing': sum([s['agreement'] for s in comparison['stages'].values()]),
            'campbell_validated': np.mean(np.abs(deviations)) < 0.15,
            'most_overvalued': max(comparison['stages'].items(), 
                                  key=lambda x: x[1]['deviation'])[0],
            'most_undervalued': min(comparison['stages'].items(), 
                                   key=lambda x: x[1]['deviation'])[0]
        }
        
        return comparison


# Convenience functions
def analyze_hero_journey(text: str, use_learned_weights=False, 
                        learned_weights=None) -> Dict:
    """
    Quick analysis of Hero's Journey pattern in a narrative.
    
    Args:
        text: Narrative text
        use_learned_weights: Use empirical weights instead of theoretical
        learned_weights: Optional dict of learned weights
        
    Returns:
        Dictionary with stage analysis and completion scores
    """
    transformer = HeroJourneyTransformer(
        use_learned_weights=use_learned_weights,
        learned_weights=learned_weights
    )
    transformer.fit([text])
    return transformer.get_stage_analysis(text)


def discover_journey_patterns(texts: List[str], outcomes: np.ndarray,
                              method='correlation') -> Dict:
    """
    Discover which Hero's Journey stages actually matter in a dataset.
    
    This is the empirical discovery mode - learns from data rather than
    relying on Campbell's theoretical weights.
    
    Args:
        texts: List of narrative texts
        outcomes: Success outcomes (1/0 or continuous scores)
        method: Weight learning method ('correlation', 'mutual_info', 'regression')
        
    Returns:
        Dictionary with learned weights and validation of Campbell's theory
    """
    transformer = HeroJourneyTransformer()
    transformer.fit(texts)
    
    # Learn empirical weights
    learned_weights = transformer.learn_weights_from_data(texts, outcomes, method)
    
    # Compare to theory
    comparison = transformer.compare_theoretical_vs_empirical()
    
    return {
        'learned_weights': learned_weights,
        'theoretical_validation': comparison,
        'transformer': transformer  # Return configured transformer
    }

