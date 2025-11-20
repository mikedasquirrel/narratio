"""
Cross-Temporal Isomorphism Transformer

Tests structural equivalence across different temporal scales.

Core Hypothesis: Narratives at equivalent % completion show similar patterns
regardless of absolute duration.

Examples:
- NBA Game minute 35/48 (73%) ≈ Novel page 220/300 (73%)
- NFL Week 13/17 (76%) ≈ TV Episode 17/22 (77%)
- Symphony movement 3/4 (75%) ≈ Film act 2.5/3 (83%)

Key Insight: If validated, enables transfer learning across temporal domains.
Can train on novels and predict NBA games if structural patterns align.

Author: Narrative Optimization Framework - Phase 1 Implementation
Date: November 2025
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..base import NarrativeTransformer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from base import NarrativeTransformer


class CrossTemporalIsomorphismTransformer(NarrativeTransformer):
    """
    Extracts structural features at equivalent temporal positions.
    
    Features (40 total):
    - Position markers (10): Where in narrative (%, quartile, act)
    - Local temporal dynamics (10): ς, ρ at this position
    - Structural equivalents (10): Predicted stage based on position
    - Cross-domain mappings (10): Isomorphic position features
    
    Applications:
    - Transfer learning: Train on one temporal scale, predict another
    - Universal patterns: Find what's common at 25%, 50%, 75%, 90%
    - Domain comparison: Do novels at 73% match NBA at 73%?
    - Temporal archetypes: Universal structure independent of duration
    """
    
    def __init__(self, domain: str = 'general', reference_points: List[float] = None):
        """
        Initialize with domain and reference positions.
        
        Parameters
        ----------
        domain : str
            Domain for contextualization
        reference_points : list of float, optional
            Key positions to analyze (default: [0.25, 0.50, 0.75, 0.90])
        """
        super().__init__(
            narrative_id="cross_temporal_isomorphism",
            description="Structural equivalence across temporal scales"
        )
        
        self.domain = domain
        self.reference_points = reference_points or [0.10, 0.25, 0.50, 0.75, 0.90]
        
        # Universal narrative stages (Campbell's Journey simplified)
        self.universal_stages = {
            0.05: 'ordinary_world',
            0.10: 'call_to_adventure',
            0.15: 'refusal_crossing',
            0.25: 'tests_allies',
            0.40: 'approach',
            0.50: 'ordeal',
            0.60: 'reward',
            0.75: 'road_back',
            0.85: 'resurrection',
            0.95: 'return'
        }
        
        # Temporal archetypes by position
        self.position_archetypes = {
            'early': (0.0, 0.25),    # Setup, establishment
            'mid': (0.25, 0.75),      # Development, conflict
            'late': (0.75, 1.0)       # Resolution, denouement
        }
        
        self.position_patterns_ = None
        
    def fit(self, X, y=None, metadata=None):
        """Learn position-specific patterns from corpus."""
        position_data = {pos: [] for pos in self.reference_points}
        
        for idx, narrative in enumerate(X):
            beats = self._extract_beats(narrative)
            if len(beats) < 3:
                continue
            
            beat_times = [b['time'] for b in beats]
            duration = beat_times[-1] if beat_times else 1.0
            
            # Analyze patterns at each reference position
            for ref_pos in self.reference_points:
                features = self._analyze_position(beats, beat_times, ref_pos, duration)
                position_data[ref_pos].append(features)
        
        # Calculate statistics for each reference position
        self.position_patterns_ = {}
        for pos, features_list in position_data.items():
            if features_list:
                # Average patterns at this position across all narratives
                avg_features = {}
                for key in features_list[0].keys():
                    values = [f[key] for f in features_list if key in f]
                    if values:
                        avg_features[key] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'median': np.median(values)
                        }
                self.position_patterns_[pos] = avg_features
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        """Extract cross-temporal isomorphism features."""
        self._validate_fitted()
        
        features_list = []
        
        for idx, narrative in enumerate(X):
            doc_features = []
            
            # Extract beats and timing
            beats = self._extract_beats(narrative)
            
            if len(beats) < 2:
                features_list.append([0.5] * 40)
                continue
            
            beat_times = [b['time'] for b in beats]
            duration = beat_times[-1] if beat_times else 1.0
            
            # ============================================================
            # Position Markers (10 features)
            # ============================================================
            
            # Current narrative position (if this is incomplete)
            # For complete narratives, analyze distribution
            positions = [t / duration for t in beat_times]
            
            # Position coverage (how much of narrative is covered)
            coverage = (max(positions) - min(positions)) if positions else 1.0
            doc_features.append(coverage)
            
            # Position density in quarters
            for i in range(4):
                quarter_start = i * 0.25
                quarter_end = (i + 1) * 0.25
                beats_in_quarter = sum(1 for p in positions if quarter_start <= p < quarter_end)
                density = beats_in_quarter / len(positions) if positions else 0.25
                doc_features.append(density)
            
            # Asymmetry (more content early vs late)
            if len(positions) >= 2:
                first_half = sum(1 for p in positions if p < 0.5)
                second_half = len(positions) - first_half
                asymmetry = (first_half - second_half) / len(positions)
                asymmetry_norm = (asymmetry + 1) / 2  # Normalize to [0,1]
                doc_features.append(asymmetry_norm)
            else:
                doc_features.append(0.5)
            
            # Golden ratio position (0.618) beat density
            golden_ratio = 0.618
            golden_window = [p for p in positions if golden_ratio - 0.1 < p < golden_ratio + 0.1]
            golden_density = len(golden_window) / max(len(positions) * 0.2, 1)
            doc_features.append(min(golden_density, 2.0) / 2.0)
            
            # Midpoint emphasis
            midpoint_window = [p for p in positions if 0.45 < p < 0.55]
            midpoint_emphasis = len(midpoint_window) / max(len(positions) * 0.1, 1)
            doc_features.append(min(midpoint_emphasis, 2.0) / 2.0)
            
            # Opening strength (first 10%)
            opening_window = [p for p in positions if p < 0.1]
            opening_density = len(opening_window) / max(len(positions) * 0.1, 1)
            doc_features.append(min(opening_density, 2.0) / 2.0)
            
            # Closing strength (last 10%)
            closing_window = [p for p in positions if p > 0.9]
            closing_density = len(closing_window) / max(len(positions) * 0.1, 1)
            doc_features.append(min(closing_density, 2.0) / 2.0)
            
            # ============================================================
            # Local Temporal Dynamics at Reference Points (10 features)
            # ============================================================
            
            # Analyze dynamics at each reference point
            ref_features = []
            for ref_pos in self.reference_points[:2]:  # Use first 2 for features
                analysis = self._analyze_position(beats, beat_times, ref_pos, duration)
                
                # Local compression (ς at this position)
                local_sigma = analysis.get('local_sigma', 1.0)
                ref_features.append(min(local_sigma, 3.0) / 3.0)
                
                # Local rhythm (ρ at this position)
                local_rho = analysis.get('local_rho', 0.35)
                ref_features.append(min(local_rho, 1.0))
                
                # Local intensity
                local_intensity = analysis.get('local_intensity', 0.5)
                ref_features.append(local_intensity)
                
                # Change rate (how fast things are changing at this position)
                change_rate = analysis.get('change_rate', 0.5)
                ref_features.append(change_rate)
            
            # Pad if needed
            while len(ref_features) < 8:
                ref_features.append(0.5)
            doc_features.extend(ref_features[:8])
            
            # Position-averaged dynamics
            all_sigmas = []
            all_rhos = []
            for ref_pos in self.reference_points:
                analysis = self._analyze_position(beats, beat_times, ref_pos, duration)
                all_sigmas.append(analysis.get('local_sigma', 1.0))
                all_rhos.append(analysis.get('local_rho', 0.35))
            
            avg_sigma = np.mean(all_sigmas) if all_sigmas else 1.0
            avg_rho = np.mean(all_rhos) if all_rhos else 0.35
            doc_features.append(min(avg_sigma, 3.0) / 3.0)
            doc_features.append(min(avg_rho, 1.0))
            
            # ============================================================
            # Structural Equivalents (10 features)
            # ============================================================
            
            # Map to universal narrative stages
            stage_present = {stage: 0.0 for stage in set(self.universal_stages.values())}
            
            for beat_time in beat_times:
                position = beat_time / duration
                # Find closest stage
                closest_stage_pos = min(self.universal_stages.keys(), 
                                       key=lambda x: abs(x - position))
                if abs(closest_stage_pos - position) < 0.05:  # Within 5%
                    stage = self.universal_stages[closest_stage_pos]
                    stage_present[stage] += 1
            
            # Normalize
            total_beats = len(beat_times)
            stage_features = []
            for stage in ['ordinary_world', 'call_to_adventure', 'ordeal', 
                         'reward', 'resurrection', 'return']:
                stage_features.append(stage_present.get(stage, 0) / max(total_beats, 1))
            
            doc_features.extend(stage_features[:6])
            
            # Journey completeness (how many stages covered)
            stages_covered = sum(1 for v in stage_present.values() if v > 0)
            completeness = stages_covered / len(self.universal_stages)
            doc_features.append(completeness)
            
            # Stage sequence (are stages in order?)
            stage_positions = []
            for stage_pos in sorted(self.universal_stages.keys()):
                stage = self.universal_stages[stage_pos]
                if stage_present[stage] > 0:
                    stage_positions.append(stage_pos)
            
            if len(stage_positions) >= 2:
                # Check if monotonically increasing (proper order)
                in_order = all(stage_positions[i] < stage_positions[i+1] 
                             for i in range(len(stage_positions)-1))
                doc_features.append(1.0 if in_order else 0.0)
            else:
                doc_features.append(0.5)
            
            # Placeholders
            doc_features.extend([0.5] * 2)
            
            # ============================================================
            # Cross-Domain Mappings (10 features)
            # ============================================================
            
            # Compare to learned position patterns
            if self.position_patterns_:
                deviation_scores = []
                for ref_pos in self.reference_points[:3]:
                    if ref_pos not in self.position_patterns_:
                        continue
                    
                    analysis = self._analyze_position(beats, beat_times, ref_pos, duration)
                    patterns = self.position_patterns_[ref_pos]
                    
                    # Compare local_sigma
                    if 'local_sigma' in patterns and 'local_sigma' in analysis:
                        expected = patterns['local_sigma']['mean']
                        actual = analysis['local_sigma']
                        deviation = abs(actual - expected) / (expected + 0.1)
                        deviation_scores.append(min(deviation, 2.0) / 2.0)
                    
                    # Compare local_rho
                    if 'local_rho' in patterns and 'local_rho' in analysis:
                        expected = patterns['local_rho']['mean']
                        actual = analysis['local_rho']
                        deviation = abs(actual - expected) / (expected + 0.1)
                        deviation_scores.append(min(deviation, 2.0) / 2.0)
                
                # Pad and add
                while len(deviation_scores) < 6:
                    deviation_scores.append(0.5)
                doc_features.extend(deviation_scores[:6])
            else:
                doc_features.extend([0.5] * 6)
            
            # Pattern consistency across positions
            if len(all_sigmas) >= 2 and len(all_rhos) >= 2:
                sigma_consistency = 1.0 / (1.0 + np.std(all_sigmas) / (np.mean(all_sigmas) + 0.1))
                rho_consistency = 1.0 / (1.0 + np.std(all_rhos) / (np.mean(all_rhos) + 0.1))
                doc_features.append(sigma_consistency)
                doc_features.append(rho_consistency)
            else:
                doc_features.extend([0.5, 0.5])
            
            # Isomorphism quality (overall structural alignment)
            # High quality = follows universal patterns + consistent dynamics
            isomorphism_quality = (completeness + stage_features[0] + sigma_consistency + rho_consistency) / 4.0
            doc_features.append(isomorphism_quality)
            
            # Temporal archetype (early/mid/late dominance)
            early_density = sum(1 for p in positions if p < 0.25) / len(positions)
            doc_features.append(early_density)
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_beats(self, text: str) -> List[Dict]:
        """Extract narrative beats (reuse from other transformers)."""
        beats = []
        
        paragraphs = text.split('\n\n')
        words = len(text.split())
        time_per_word = 0.004  # ~250 wpm
        total_time = words * time_per_word
        
        current_time = 0
        for i, para in enumerate(paragraphs):
            if para.strip():
                para_words = len(para.split())
                para_time = para_words * time_per_word
                
                # Estimate intensity
                intensity = 0.5
                if any(word in para.lower() for word in ['climax', 'peak', 'crisis']):
                    intensity = 0.9
                elif any(word in para.lower() for word in ['action', 'conflict']):
                    intensity = 0.7
                
                beats.append({
                    'time': current_time + para_time / 2,
                    'type': 'structural',
                    'strength': intensity
                })
                
                current_time += para_time
        
        return beats if beats else [{'time': 0, 'type': 'default', 'strength': 0.5}]
    
    def _analyze_position(self, beats: List[Dict], beat_times: List[float], 
                         position: float, duration: float) -> Dict:
        """
        Analyze narrative dynamics at specific position (0-1).
        
        Returns
        -------
        analysis : dict
            Local dynamics: sigma, rho, intensity, change_rate
        """
        target_time = position * duration
        window_size = duration * 0.1  # 10% window around position
        
        # Find beats in window
        window_beats = [b for b, t in zip(beats, beat_times) 
                       if target_time - window_size/2 <= t <= target_time + window_size/2]
        
        if not window_beats:
            # No beats in window, use nearby
            closest_idx = min(range(len(beat_times)), 
                            key=lambda i: abs(beat_times[i] - target_time))
            window_beats = [beats[closest_idx]]
        
        # Local compression (beats per time in window)
        local_sigma = len(window_beats) / max(window_size / 60.0, 0.1)  # Per hour
        
        # Local rhythm (CV of intervals in window)
        if len(window_beats) >= 3:
            window_times = [b['time'] for b in window_beats]
            intervals = np.diff(window_times)
            local_rho = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0.0
        else:
            local_rho = 0.35  # Default
        
        # Local intensity
        local_intensity = np.mean([b.get('strength', 0.5) for b in window_beats])
        
        # Change rate (how different is this window from previous?)
        prev_window_start = target_time - window_size
        prev_window_end = target_time - window_size/2
        prev_beats = [b for b, t in zip(beats, beat_times)
                     if prev_window_start <= t <= prev_window_end]
        
        if prev_beats:
            prev_intensity = np.mean([b.get('strength', 0.5) for b in prev_beats])
            change_rate = abs(local_intensity - prev_intensity)
        else:
            change_rate = 0.0
        
        return {
            'local_sigma': local_sigma,
            'local_rho': local_rho,
            'local_intensity': local_intensity,
            'change_rate': change_rate
        }
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        return [
            # Position Markers (10)
            'position_coverage',
            'q1_density', 'q2_density', 'q3_density', 'q4_density',
            'asymmetry_first_second_half', 'golden_ratio_density',
            'midpoint_emphasis', 'opening_strength', 'closing_strength',
            
            # Local Dynamics at Reference Points (10)
            'ref1_local_sigma', 'ref1_local_rho', 'ref1_local_intensity', 'ref1_change_rate',
            'ref2_local_sigma', 'ref2_local_rho', 'ref2_local_intensity', 'ref2_change_rate',
            'avg_sigma_all_positions', 'avg_rho_all_positions',
            
            # Structural Equivalents (10)
            'stage_ordinary_world', 'stage_call_adventure', 'stage_ordeal',
            'stage_reward', 'stage_resurrection', 'stage_return',
            'journey_completeness', 'stages_in_order',
            'struct_placeholder_1', 'struct_placeholder_2',
            
            # Cross-Domain Mappings (10)
            'deviation_ref1_sigma', 'deviation_ref1_rho',
            'deviation_ref2_sigma', 'deviation_ref2_rho',
            'deviation_ref3_sigma', 'deviation_ref3_rho',
            'sigma_consistency_across_positions', 'rho_consistency_across_positions',
            'isomorphism_quality_score', 'early_archetype_density'
        ]

