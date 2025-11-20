"""
Pacing & Rhythm Transformer

Extracts optimal pacing features, beat placement, and rhythmic patterns.

Core Insights:
- Inter-beat intervals reveal rhythm regularity (ρ)
- Act length ratios (1:2:1 vs others) affect satisfaction
- Climax positioning (70% vs 85% vs 95%) is genre-specific
- Breathing room (pause density) affects engagement
- Momentum acceleration curves predict emotional impact

Key Relationships:
- Optimal ρ = k × π^0.5  (higher narrativity → more rhythm variation)
- ρ × ς ≈ R_control (rhythm control constant)
- Act ratios follow genre conventions

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


class PacingRhythmTransformer(NarrativeTransformer):
    """
    Extracts pacing and rhythm optimization features.
    
    Features (50 total):
    - Inter-beat intervals (10): Timing between narrative beats
    - Act structure (10): Length ratios, transitions
    - Climax positioning (10): Placement and build-up
    - Breathing room (10): Pause density and distribution
    - Momentum patterns (10): Acceleration/deceleration curves
    
    Applications:
    - Film: Does thriller maintain tension correctly?
    - Sports: Does game have natural momentum swings?
    - Music: Does song follow optimal verse-chorus structure?
    - Literature: Does novel pace appropriately across acts?
    """
    
    def __init__(self, domain: str = 'general'):
        """Initialize with domain-specific pacing expectations."""
        super().__init__(
            narrative_id="pacing_rhythm",
            description="Optimal pacing, beat placement, and rhythmic analysis"
        )
        
        self.domain = domain
        
        # Optimal act ratios by structure type
        self.act_ratios = {
            '3_act_classical': [0.25, 0.50, 0.25],
            '3_act_hollywood': [0.25, 0.50, 0.25],
            '5_act_freytag': [0.15, 0.20, 0.30, 0.20, 0.15],
            '4_act': [0.20, 0.30, 0.30, 0.20],
            '7_act': [0.10, 0.15, 0.15, 0.20, 0.15, 0.15, 0.10],
        }
        
        # Optimal climax positions (as % of total duration)
        self.climax_positions = {
            'action': 0.85,  # Late climax
            'thriller': 0.90,  # Very late climax
            'drama': 0.75,  # Mid-late climax
            'romance': 0.80,  # Fairly late
            'mystery': 0.85,  # Late reveal
            'tragedy': 0.70,  # Earlier climax, longer resolution
            'comedy': 0.75,  # Mid-late
        }
        
        # Breathing room requirements (pauses per hour)
        self.pause_density = {
            'action': 2.0,  # Fewer pauses
            'thriller': 2.5,
            'drama': 4.0,  # More pauses
            'contemplative': 6.0,  # Many pauses
            'experimental': 8.0,  # Lots of breathing room
        }
        
        self.pacing_stats_ = None
        
    def fit(self, X, y=None, metadata=None):
        """Learn pacing patterns from corpus."""
        patterns = []
        
        for idx, narrative in enumerate(X):
            beats = self._extract_beats(narrative)
            if len(beats) >= 3:
                intervals = np.diff([b['time'] for b in beats])
                patterns.append({
                    'mean_interval': np.mean(intervals),
                    'std_interval': np.std(intervals),
                    'rho': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0,
                    'num_beats': len(beats)
                })
        
        if patterns:
            self.pacing_stats_ = {
                'mean_interval': np.mean([p['mean_interval'] for p in patterns]),
                'mean_rho': np.mean([p['rho'] for p in patterns]),
                'mean_beats': np.mean([p['num_beats'] for p in patterns]),
            }
        else:
            self.pacing_stats_ = {'mean_interval': 5.0, 'mean_rho': 0.35, 'mean_beats': 50}
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        """Extract pacing and rhythm features."""
        self._validate_fitted()
        
        features_list = []
        
        for idx, narrative in enumerate(X):
            doc_features = []
            
            # Extract beats and timing
            beats = self._extract_beats(narrative)
            beat_times = [b['time'] for b in beats]
            
            if len(beats) < 2:
                # Not enough beats for analysis - use defaults
                features_list.append([0.5] * 50)
                continue
            
            duration = beat_times[-1] if beat_times else 60.0
            intervals = np.diff(beat_times)
            
            # ============================================================
            # Inter-Beat Intervals (10 features)
            # ============================================================
            
            # Basic interval statistics
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            doc_features.append(mean_interval / 10.0)  # Normalized
            doc_features.append(std_interval / 10.0)
            
            # Coefficient of variation (ρ)
            rho = std_interval / mean_interval if mean_interval > 0 else 0.0
            doc_features.append(min(rho, 2.0) / 2.0)
            
            # Interval distribution shape
            if len(intervals) >= 3:
                skewness = stats.skew(intervals)
                kurtosis = stats.kurtosis(intervals)
                doc_features.append((skewness + 3) / 6.0)  # Normalize to [0,1]
                doc_features.append((kurtosis + 3) / 6.0)
            else:
                doc_features.extend([0.5, 0.5])
            
            # Short vs long intervals
            short_intervals = sum(1 for i in intervals if i < mean_interval * 0.5)
            long_intervals = sum(1 for i in intervals if i > mean_interval * 1.5)
            doc_features.append(short_intervals / len(intervals))
            doc_features.append(long_intervals / len(intervals))
            
            # Interval trend (accelerating vs decelerating)
            if len(intervals) >= 5:
                # Linear regression of intervals over time
                x = np.arange(len(intervals))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, intervals)
                trend = slope / mean_interval  # Normalized slope
                doc_features.append((trend + 1) / 2)  # Map to [0,1]
                doc_features.append(abs(r_value))  # Trend strength
            else:
                doc_features.extend([0.5, 0.0])
            
            # Consistency across narrative thirds
            third = len(intervals) // 3
            if third > 0:
                rho_first = np.std(intervals[:third]) / (np.mean(intervals[:third]) + 0.01)
                rho_last = np.std(intervals[2*third:]) / (np.mean(intervals[2*third:]) + 0.01)
                rhythm_shift = (rho_last - rho_first + 2) / 4  # Normalize
                doc_features.append(min(rhythm_shift, 1.0))
            else:
                doc_features.append(0.5)
            
            # ============================================================
            # Act Structure (10 features)
            # ============================================================
            
            # Detect act boundaries (major beats with large gaps before)
            act_boundaries = [0]
            for i in range(1, len(beat_times)):
                if intervals[i-1] > mean_interval * 1.5:  # Significant pause
                    act_boundaries.append(i)
            act_boundaries.append(len(beat_times))
            
            num_acts = len(act_boundaries) - 1
            doc_features.append(min(num_acts, 7) / 7.0)  # Number of acts, normalized
            
            # Act length ratios
            act_lengths = []
            for i in range(len(act_boundaries) - 1):
                start_idx = act_boundaries[i]
                end_idx = act_boundaries[i+1]
                if end_idx > start_idx:
                    act_length = beat_times[end_idx-1] - beat_times[start_idx]
                    act_lengths.append(act_length)
            
            if len(act_lengths) >= 2:
                # Normalize to percentages
                total_length = sum(act_lengths)
                act_percentages = [l / total_length for l in act_lengths]
                
                # Compare to optimal ratios (3-act structure)
                optimal = self.act_ratios['3_act_classical']
                if len(act_percentages) == 3:
                    deviation = sum(abs(a - o) for a, o in zip(act_percentages, optimal))
                    structure_fit = max(0, 1.0 - deviation)
                    doc_features.append(structure_fit)
                else:
                    doc_features.append(0.5)
                
                # Act balance (variance in act lengths)
                act_balance = 1.0 - (np.std(act_percentages) / 0.5)  # Lower variance = better balance
                doc_features.append(max(0, act_balance))
                
                # Progressive acts (getting longer or shorter)
                if len(act_lengths) >= 3:
                    progressive = 1.0 if all(act_lengths[i] < act_lengths[i+1] for i in range(len(act_lengths)-1)) else 0.0
                    regressive = 1.0 if all(act_lengths[i] > act_lengths[i+1] for i in range(len(act_lengths)-1)) else 0.0
                    doc_features.append(progressive)
                    doc_features.append(regressive)
                else:
                    doc_features.extend([0.0, 0.0])
                
                # Middle act dominance (should be longest in 3-act)
                if len(act_lengths) == 3:
                    middle_dominant = 1.0 if act_lengths[1] > act_lengths[0] and act_lengths[1] > act_lengths[2] else 0.0
                    doc_features.append(middle_dominant)
                else:
                    doc_features.append(0.0)
            else:
                doc_features.extend([0.5, 0.5, 0.0, 0.0, 0.0])
            
            # Act transition smoothness
            if len(act_boundaries) > 2:
                transition_gaps = [intervals[b-1] for b in act_boundaries[1:-1] if b > 0 and b-1 < len(intervals)]
                if transition_gaps:
                    avg_transition = np.mean(transition_gaps)
                    transition_smoothness = 1.0 / (1.0 + avg_transition / mean_interval)
                    doc_features.append(transition_smoothness)
                else:
                    doc_features.append(0.5)
            else:
                doc_features.append(0.5)
            
            # Placeholders for remaining act features
            doc_features.extend([0.5] * 4)
            
            # ============================================================
            # Climax Positioning (10 features)
            # ============================================================
            
            # Find climax (highest intensity beat, typically late)
            intensities = [b.get('strength', 0.5) for b in beats]
            if intensities:
                climax_idx = np.argmax(intensities)
                climax_position = beat_times[climax_idx] / duration
                doc_features.append(climax_position)
                
                # Compare to optimal position
                optimal_climax = self.climax_positions.get(self.domain, 0.80)
                climax_deviation = abs(climax_position - optimal_climax)
                climax_quality = max(0, 1.0 - climax_deviation * 2)
                doc_features.append(climax_quality)
                
                # Climax in final third (expected for most genres)
                climax_in_final_third = 1.0 if climax_position > 0.66 else 0.0
                doc_features.append(climax_in_final_third)
                
                # Climax intensity relative to average
                avg_intensity = np.mean(intensities)
                climax_relative_intensity = intensities[climax_idx] / (avg_intensity + 0.01)
                doc_features.append(min(climax_relative_intensity, 3.0) / 3.0)
            else:
                doc_features.extend([0.80, 0.5, 1.0, 0.67])
            
            # Build-up to climax (intensity acceleration)
            if len(intensities) >= 5:
                # Measure intensity trend in second half
                half_point = len(intensities) // 2
                first_half_intensity = np.mean(intensities[:half_point])
                second_half_intensity = np.mean(intensities[half_point:])
                build_up = (second_half_intensity - first_half_intensity + 1) / 2
                doc_features.append(min(build_up, 1.0))
            else:
                doc_features.append(0.5)
            
            # Multiple climaxes (can be good or bad depending on genre)
            # Count local maxima in intensity
            local_maxima = 0
            for i in range(1, len(intensities) - 1):
                if intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1]:
                    if intensities[i] > np.mean(intensities) * 1.2:
                        local_maxima += 1
            doc_features.append(min(local_maxima, 5) / 5.0)
            
            # Climax resolution distance (time from climax to end)
            if intensities:
                resolution_time = duration - beat_times[climax_idx]
                resolution_ratio = resolution_time / duration
                doc_features.append(resolution_ratio)
                
                # Appropriate resolution length (should be ~10-20% of total)
                resolution_appropriate = 1.0 if 0.05 <= resolution_ratio <= 0.25 else 0.0
                doc_features.append(resolution_appropriate)
            else:
                doc_features.extend([0.15, 1.0])
            
            # Placeholders
            doc_features.extend([0.5] * 2)
            
            # ============================================================
            # Breathing Room (10 features)
            # ============================================================
            
            # Pause density (long intervals = breathing room)
            pauses = sum(1 for i in intervals if i > mean_interval * 1.5)
            pause_density = pauses / (duration / 60)  # Per hour
            doc_features.append(min(pause_density, 10.0) / 10.0)
            
            # Compare to optimal pause density
            optimal_pause_density = self.pause_density.get(self.domain, 4.0)
            pause_quality = max(0, 1.0 - abs(pause_density - optimal_pause_density) / optimal_pause_density)
            doc_features.append(pause_quality)
            
            # Pause distribution (should be somewhat regular)
            if pauses >= 2:
                # Find pause locations
                pause_positions = [i for i in range(len(intervals)) if intervals[i] > mean_interval * 1.5]
                pause_intervals = np.diff(pause_positions) if len(pause_positions) > 1 else [0]
                pause_regularity = 1.0 / (1.0 + np.std(pause_intervals) / (np.mean(pause_intervals) + 1))
                doc_features.append(pause_regularity)
            else:
                doc_features.append(0.5)
            
            # Early pauses (setup breathing room)
            early_pauses = sum(1 for i in range(min(len(intervals) // 3, len(intervals))) 
                             if intervals[i] > mean_interval * 1.5)
            doc_features.append(min(early_pauses, 3) / 3.0)
            
            # Mid pauses (development breathing room)
            mid_start = len(intervals) // 3
            mid_end = 2 * len(intervals) // 3
            mid_pauses = sum(1 for i in range(mid_start, min(mid_end, len(intervals))) 
                           if intervals[i] > mean_interval * 1.5)
            doc_features.append(min(mid_pauses, 3) / 3.0)
            
            # Late pauses (resolution breathing room)
            late_pauses = sum(1 for i in range(max(2 * len(intervals) // 3, 0), len(intervals)) 
                            if intervals[i] > mean_interval * 1.5)
            doc_features.append(min(late_pauses, 3) / 3.0)
            
            # Longest pause (maximum breathing room)
            longest_pause = max(intervals) if intervals.size > 0 else 0
            longest_pause_normalized = min(longest_pause / mean_interval, 5.0) / 5.0
            doc_features.append(longest_pause_normalized)
            
            # Pause variation (different length pauses)
            pause_lengths = [i for i in intervals if i > mean_interval * 1.5]
            if len(pause_lengths) >= 2:
                pause_variation = np.std(pause_lengths) / (np.mean(pause_lengths) + 0.01)
                doc_features.append(min(pause_variation, 2.0) / 2.0)
            else:
                doc_features.append(0.5)
            
            # Placeholders
            doc_features.extend([0.5] * 2)
            
            # ============================================================
            # Momentum Patterns (10 features)
            # ============================================================
            
            # Overall momentum (beat density increasing or decreasing)
            if len(beat_times) >= 5:
                # Calculate local beat density across narrative
                window_size = max(len(beat_times) // 5, 2)
                densities = []
                for i in range(0, len(beat_times) - window_size):
                    window_duration = beat_times[i + window_size] - beat_times[i]
                    density = window_size / window_duration if window_duration > 0 else 0
                    densities.append(density)
                
                if len(densities) >= 2:
                    # Momentum trend
                    x = np.arange(len(densities))
                    slope, _, r_value, _, _ = stats.linregress(x, densities)
                    momentum_trend = (slope / np.mean(densities) + 1) / 2 if np.mean(densities) > 0 else 0.5
                    doc_features.append(min(momentum_trend, 1.0))
                    doc_features.append(abs(r_value))  # Trend consistency
                else:
                    doc_features.extend([0.5, 0.0])
            else:
                doc_features.extend([0.5, 0.0])
            
            # Momentum acceleration (second derivative)
            if len(intervals) >= 3:
                acceleration = np.diff(np.diff(intervals))
                avg_acceleration = np.mean(acceleration)
                acceleration_score = (avg_acceleration / mean_interval + 1) / 2
                doc_features.append(min(acceleration_score, 1.0))
            else:
                doc_features.append(0.5)
            
            # Momentum shifts (direction changes)
            if len(intervals) >= 4:
                velocity = np.diff(intervals)  # Rate of change
                shifts = sum(1 for i in range(len(velocity) - 1) 
                           if velocity[i] * velocity[i+1] < 0)  # Sign change
                shift_rate = shifts / (len(velocity) - 1)
                doc_features.append(shift_rate)
            else:
                doc_features.append(0.0)
            
            # Sustained momentum (long runs of similar intervals)
            if len(intervals) >= 3:
                runs = 1
                max_run = 1
                for i in range(1, len(intervals)):
                    if abs(intervals[i] - intervals[i-1]) < mean_interval * 0.3:
                        runs += 1
                        max_run = max(max_run, runs)
                    else:
                        runs = 1
                sustained_momentum = max_run / len(intervals)
                doc_features.append(sustained_momentum)
            else:
                doc_features.append(0.0)
            
            # Final sprint (acceleration in last 20%)
            if len(beat_times) >= 5:
                final_section_start = int(len(beat_times) * 0.8)
                if final_section_start < len(intervals):
                    final_intervals = intervals[final_section_start:]
                    early_intervals = intervals[:final_section_start]
                    if len(final_intervals) > 0 and len(early_intervals) > 0:
                        final_density = len(final_intervals) / sum(final_intervals)
                        early_density = len(early_intervals) / sum(early_intervals)
                        final_sprint = (final_density / early_density - 0.5) / 1.5  # Normalize
                        doc_features.append(max(0, min(final_sprint, 1.0)))
                    else:
                        doc_features.append(0.5)
                else:
                    doc_features.append(0.5)
            else:
                doc_features.append(0.5)
            
            # Momentum consistency (low variance in rate of change)
            if len(intervals) >= 3:
                velocity = np.diff(intervals)
                momentum_consistency = 1.0 / (1.0 + np.std(velocity) / (np.mean(np.abs(velocity)) + 0.01))
                doc_features.append(momentum_consistency)
            else:
                doc_features.append(0.5)
            
            # Placeholders
            doc_features.extend([0.5] * 3)
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_beats(self, text: str) -> List[Dict]:
        """Extract narrative beats with timing and intensity."""
        # Reuse beat extraction from TemporalCompressionTransformer
        # For now, simplified version
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
                
                # Estimate intensity from markers
                intensity = 0.5
                if any(word in para.lower() for word in ['climax', 'peak', 'turning', 'crisis']):
                    intensity = 0.9
                elif any(word in para.lower() for word in ['action', 'fight', 'chase', 'explode']):
                    intensity = 0.8
                elif any(word in para.lower() for word in ['quiet', 'calm', 'peace', 'rest']):
                    intensity = 0.3
                
                beats.append({
                    'time': current_time + para_time / 2,
                    'type': 'structural',
                    'strength': intensity
                })
                
                current_time += para_time
        
        return beats if beats else [{'time': 0, 'type': 'default', 'strength': 0.5}]
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        return [
            # Inter-Beat Intervals (10)
            'mean_interval_norm', 'std_interval_norm', 'rho_coefficient',
            'interval_skewness', 'interval_kurtosis',
            'short_intervals_pct', 'long_intervals_pct',
            'interval_trend', 'trend_strength', 'rhythm_shift_thirds',
            
            # Act Structure (10)
            'num_acts_norm', 'act_structure_fit', 'act_balance',
            'acts_progressive', 'acts_regressive', 'middle_act_dominant',
            'transition_smoothness',
            'act_placeholder_1', 'act_placeholder_2', 'act_placeholder_3',
            
            # Climax Positioning (10)
            'climax_position', 'climax_quality', 'climax_in_final_third',
            'climax_relative_intensity', 'build_up_score',
            'multiple_climaxes_count', 'resolution_ratio', 'resolution_appropriate',
            'climax_placeholder_1', 'climax_placeholder_2',
            
            # Breathing Room (10)
            'pause_density_per_hour', 'pause_quality_score', 'pause_regularity',
            'early_pauses', 'mid_pauses', 'late_pauses',
            'longest_pause_norm', 'pause_variation',
            'pause_placeholder_1', 'pause_placeholder_2',
            
            # Momentum Patterns (10)
            'momentum_trend', 'momentum_trend_consistency', 'momentum_acceleration',
            'momentum_shift_rate', 'sustained_momentum', 'final_sprint',
            'momentum_consistency',
            'momentum_placeholder_1', 'momentum_placeholder_2', 'momentum_placeholder_3'
        ]

