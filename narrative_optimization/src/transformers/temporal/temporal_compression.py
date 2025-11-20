"""
Temporal Compression Transformer

Extracts τ (tau), ς (sigma), and ρ (rho) temporal dynamics features.

Formalization based on TEMPORAL_DYNAMICS_THEORY.md:
- τ (tau): Duration ratio (actual_time / natural_timescale)
- ς (sigma): Compression ratio (beat_density / baseline_density)
- ρ (rho): Temporal rhythm (CV of inter-beat intervals)

Core Insight: A 3-second knockout ≠ 12-round decision.
Compression fundamentally affects narrative power.

Author: Narrative Optimization Framework - Phase 1 Implementation
Date: November 2025
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..base import NarrativeTransformer
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from base import NarrativeTransformer


class TemporalCompressionTransformer(NarrativeTransformer):
    """
    Extracts temporal compression features: duration, beat density, rhythm.
    
    Features (60 total):
    - τ (Tau) Duration features (10)
    - ς (Sigma) Compression features (25)
    - ρ (Rho) Rhythm features (15)
    - Cross-variable interactions (10)
    
    Applications:
    - UFC: Early finish vs decision affects narrative
    - Film: 90min thriller vs 180min epic requires different pacing
    - Music: 3min pop song vs 20min prog epic needs different density
    - Sports: Blowout vs close game has different rhythm patterns
    """
    
    def __init__(self, domain: str = 'general'):
        """
        Initialize transformer with domain-specific parameters.
        
        Parameters
        ----------
        domain : str
            Domain type for natural timescale and baselines
            Options: 'ufc', 'nba', 'nfl', 'film', 'novel', 'music', 'general'
        """
        super().__init__(
            narrative_id="temporal_compression",
            description="Duration, compression ratio, and temporal rhythm analysis"
        )
        
        self.domain = domain
        
        # Natural timescales by domain (in minutes unless noted)
        self.natural_timescales = {
            'ufc': 15,  # 3 x 5-minute rounds
            'nba': 48,  # 4 x 12-minute quarters
            'nfl': 60,  # 4 x 15-minute quarters
            'mlb': 180,  # ~3 hours average
            'golf': 4 * 24 * 60,  # 4 days in minutes
            'tennis': 150,  # ~2.5 hours
            'film': 120,  # 2-hour standard
            'film_short': 30,
            'film_epic': 180,
            'tv_episode': 44,  # Hour-long minus ads
            'novel': 10 * 60,  # 10 hours in minutes
            'short_story': 30,
            'pop_song': 3.5,
            'symphony': 45,
            'album': 45,
            'podcast': 60,
            'general': 60
        }
        
        # Beat density baselines (beats per minute) by domain/genre
        self.beat_baselines = {
            'thriller': 1.5,
            'drama': 0.5,
            'action': 2.0,
            'romance': 0.6,
            'horror': 1.2,
            'comedy': 1.8,
            'documentary': 0.4,
            'sports_action': 2.0,  # UFC, NBA
            'sports_strategic': 0.5,  # Golf, baseball
            'pop_music': 4.0,
            'classical_music': 0.3,
            'novel': 0.05,  # 3 beats/hour
            'short_story': 0.15,
            'general': 1.0
        }
        
        # Optimal rhythm (ρ) by genre
        self.optimal_rho = {
            'sitcom': 0.15,
            'procedural': 0.25,
            'drama': 0.35,
            'thriller': 0.45,
            'action': 0.45,
            'art_film': 0.75,
            'experimental': 0.90,
            'sports_close': 0.40,
            'sports_blowout': 0.15,
            'general': 0.35
        }
        
        # Will learn from data during fit
        self.beat_patterns_ = None
        self.duration_distribution_ = None
        
    def fit(self, X, y=None, metadata=None):
        """
        Learn beat patterns and duration distributions from corpus.
        
        Parameters
        ----------
        X : list of str
            Narrative texts
        y : ignored
        metadata : dict, optional
            Domain-specific metadata (durations, genres, etc.)
        
        Returns
        -------
        self
        """
        beat_patterns = []
        durations = []
        
        for idx, narrative in enumerate(X):
            # Extract beats and duration
            beats = self._extract_beats(narrative)
            duration = self._estimate_duration(narrative, metadata, idx) if metadata else None
            
            if beats:
                beat_patterns.append(len(beats))
            if duration:
                durations.append(duration)
        
        self.beat_patterns_ = {
            'mean': np.mean(beat_patterns) if beat_patterns else 50,
            'std': np.std(beat_patterns) if beat_patterns else 20,
            'median': np.median(beat_patterns) if beat_patterns else 45
        }
        
        self.duration_distribution_ = {
            'mean': np.mean(durations) if durations else 60,
            'std': np.std(durations) if durations else 30,
            'median': np.median(durations) if durations else 60
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        """
        Extract temporal compression features from narratives.
        
        Parameters
        ----------
        X : list of str
            Narratives to transform
        metadata : dict, optional
            Domain-specific temporal data
        
        Returns
        -------
        features : ndarray, shape (n_samples, 60)
            Temporal compression features
        """
        self._validate_fitted()
        
        features_list = []
        
        for idx, narrative in enumerate(X):
            doc_features = []
            
            # Extract beats and timing
            beats = self._extract_beats(narrative)
            beat_times = [b['time'] for b in beats]
            
            # Estimate duration
            duration = self._estimate_duration(narrative, metadata, idx)
            
            # Calculate core temporal variables
            tau = self._calculate_tau(duration)
            sigma = self._calculate_sigma(beats, duration)
            rho = self._calculate_rho(beat_times)
            
            # ============================================================
            # TAU (Duration) Features (10)
            # ============================================================
            
            doc_features.append(tau)  # Raw duration ratio
            doc_features.append(np.log1p(tau))  # Log-scaled tau
            doc_features.append(tau ** 2)  # Quadratic (inverted-U tests)
            doc_features.append(min(tau, 3.0) / 3.0)  # Normalized, capped
            
            # Duration category indicators
            doc_features.append(1.0 if tau < 0.5 else 0.0)  # Compressed
            doc_features.append(1.0 if 0.5 <= tau <= 1.5 else 0.0)  # Standard
            doc_features.append(1.0 if tau > 1.5 else 0.0)  # Extended
            
            # Deviation from optimal
            doc_features.append(abs(tau - 1.0))  # Distance from standard
            doc_features.append((tau - 1.0) ** 2)  # Squared deviation
            
            # Accessibility penalty (extreme durations reduce effective narrativity)
            accessibility = max(0, 1.0 - abs(tau - 1.0) / 3.0)
            doc_features.append(accessibility)
            
            # ============================================================
            # SIGMA (Compression) Features (25)
            # ============================================================
            
            doc_features.append(sigma)  # Raw compression ratio
            doc_features.append(np.log1p(sigma))  # Log-scaled
            doc_features.append(min(sigma, 5.0) / 5.0)  # Normalized, capped
            
            # Beat density metrics
            beats_per_minute = len(beats) / max(duration, 1) if duration else 0
            doc_features.append(beats_per_minute)
            doc_features.append(np.log1p(beats_per_minute))
            
            # Compression category indicators
            doc_features.append(1.0 if sigma < 0.5 else 0.0)  # Very low density
            doc_features.append(1.0 if 0.5 <= sigma < 0.8 else 0.0)  # Low density
            doc_features.append(1.0 if 0.8 <= sigma <= 1.5 else 0.0)  # Optimal
            doc_features.append(1.0 if 1.5 < sigma <= 2.5 else 0.0)  # High density
            doc_features.append(1.0 if sigma > 2.5 else 0.0)  # Very high density
            
            # Compression patterns across narrative
            if len(beats) >= 3:
                # Divide into thirds, measure density variation
                third = len(beats) // 3
                density_first = third / (duration / 3) if duration else 0
                density_middle = third / (duration / 3) if duration else 0
                density_last = (len(beats) - 2 * third) / (duration / 3) if duration else 0
                
                doc_features.append(density_first / max(beats_per_minute, 0.01))
                doc_features.append(density_middle / max(beats_per_minute, 0.01))
                doc_features.append(density_last / max(beats_per_minute, 0.01))
                
                # Acceleration (density increasing toward end)
                acceleration = (density_last - density_first) / max(density_first, 0.01)
                doc_features.append(acceleration)
                
                # Density variance
                densities = [density_first, density_middle, density_last]
                density_variance = np.var(densities)
                doc_features.append(density_variance)
            else:
                doc_features.extend([0.5, 0.5, 0.5, 0.0, 0.0])
            
            # Narrative Density Constant (ς × τ should ≈ 0.3)
            narrative_density_constant = sigma * tau
            doc_features.append(narrative_density_constant)
            doc_features.append(abs(narrative_density_constant - 0.3))  # Deviation from universal
            
            # Compression quality (deviation from optimal)
            optimal_sigma = self.beat_baselines.get(self.domain, 1.0)
            sigma_deviation = abs(sigma - optimal_sigma) / optimal_sigma
            compression_quality = max(0, 1.0 - sigma_deviation)
            doc_features.append(compression_quality)
            
            # Sustainable compression (high sigma requires low tau)
            if tau > 0:
                sustainable = sigma * tau
                sustainability_score = 1.0 / (1.0 + abs(sustainable - 0.3))
            else:
                sustainability_score = 0.5
            doc_features.append(sustainability_score)
            
            # Peak compression moments (local maxima in beat density)
            if len(beat_times) >= 5:
                windows = []
                for i in range(len(beat_times) - 4):
                    window_density = 5 / (beat_times[i+4] - beat_times[i] + 0.01)
                    windows.append(window_density)
                peak_compression = max(windows) if windows else 0
                doc_features.append(peak_compression / max(beats_per_minute, 0.01))
            else:
                doc_features.append(1.0)
            
            # Lull periods (local minima - breathing room)
            if len(beat_times) >= 3:
                intervals = np.diff(beat_times)
                lull_indicator = max(intervals) / (np.mean(intervals) + 0.01)
                doc_features.append(min(lull_indicator, 5.0) / 5.0)
            else:
                doc_features.append(0.5)
            
            # Placeholders for remaining sigma features
            doc_features.extend([0.5] * 3)
            
            # ============================================================
            # RHO (Rhythm) Features (15)
            # ============================================================
            
            doc_features.append(rho)  # Raw rhythm coefficient of variation
            doc_features.append(min(rho, 2.0) / 2.0)  # Normalized, capped
            
            # Rhythm category indicators
            doc_features.append(1.0 if rho < 0.2 else 0.0)  # Metronomic
            doc_features.append(1.0 if 0.2 <= rho <= 0.5 else 0.0)  # Natural
            doc_features.append(1.0 if rho > 0.5 else 0.0)  # Chaotic
            
            # Rhythm quality (deviation from optimal)
            optimal_rho_domain = self.optimal_rho.get(self.domain, 0.35)
            rho_deviation = abs(rho - optimal_rho_domain) / optimal_rho_domain
            rhythm_quality = max(0, 1.0 - rho_deviation)
            doc_features.append(rhythm_quality)
            
            # Rhythm patterns
            if len(beat_times) >= 5:
                intervals = np.diff(beat_times)
                
                # Ascending rhythm (intervals getting longer - deceleration)
                ascending = sum(1 for i in range(len(intervals)-1) if intervals[i+1] > intervals[i])
                ascending_pct = ascending / (len(intervals) - 1) if len(intervals) > 1 else 0.5
                doc_features.append(ascending_pct)
                
                # Descending rhythm (intervals getting shorter - acceleration)
                descending_pct = 1.0 - ascending_pct
                doc_features.append(descending_pct)
                
                # Alternating rhythm (regular back-and-forth)
                alternations = sum(1 for i in range(len(intervals)-2) 
                                 if (intervals[i+1] > intervals[i]) != (intervals[i+2] > intervals[i+1]))
                alternation_pct = alternations / (len(intervals) - 2) if len(intervals) > 2 else 0
                doc_features.append(alternation_pct)
                
                # Rhythm consistency across thirds
                third = len(intervals) // 3
                if third > 0:
                    rho_first = np.std(intervals[:third]) / (np.mean(intervals[:third]) + 0.01)
                    rho_middle = np.std(intervals[third:2*third]) / (np.mean(intervals[third:2*third]) + 0.01)
                    rho_last = np.std(intervals[2*third:]) / (np.mean(intervals[2*third:]) + 0.01)
                    
                    rhythm_evolution = rho_last - rho_first
                    doc_features.append(rhythm_evolution)
                    
                    rhythm_variance = np.var([rho_first, rho_middle, rho_last])
                    doc_features.append(rhythm_variance)
                else:
                    doc_features.extend([0.0, 0.0])
            else:
                doc_features.extend([0.5, 0.5, 0.5, 0.0, 0.0])
            
            # Rhythm Control Constant (ρ × ς should be roughly constant)
            rhythm_control = rho * sigma
            doc_features.append(rhythm_control)
            
            # Syncopation (unexpected timing deviations)
            if len(beat_times) >= 4:
                expected_intervals = np.mean(np.diff(beat_times))
                surprises = sum(1 for interval in np.diff(beat_times) 
                              if abs(interval - expected_intervals) > expected_intervals * 0.5)
                syncopation = surprises / len(beat_times)
                doc_features.append(syncopation)
            else:
                doc_features.append(0.0)
            
            # Placeholders
            doc_features.extend([0.5] * 1)
            
            # ============================================================
            # Cross-Variable Interactions (10)
            # ============================================================
            
            # τ × ς (duration-compression interaction)
            doc_features.append(tau * sigma)
            
            # τ × ρ (duration-rhythm interaction)
            doc_features.append(tau * rho)
            
            # ς × ρ (compression-rhythm interaction)
            doc_features.append(sigma * rho)
            
            # Three-way interaction
            doc_features.append(tau * sigma * rho)
            
            # Temporal quality score (aggregate)
            temporal_quality = (accessibility * compression_quality * rhythm_quality) ** (1/3)
            doc_features.append(temporal_quality)
            
            # Temporal stress (extreme values indicate problems)
            temporal_stress = (tau - 1.0)**2 + (sigma - 1.0)**2 + (rho - 0.35)**2
            doc_features.append(temporal_stress)
            
            # Optimal balance indicator
            is_balanced = (0.7 <= tau <= 1.3) and (0.7 <= sigma <= 1.5) and (0.25 <= rho <= 0.50)
            doc_features.append(1.0 if is_balanced else 0.0)
            
            # Duration accessibility × compression sustainability
            doc_features.append(accessibility * sustainability_score)
            
            # Rhythm-compression harmony
            doc_features.append(rhythm_quality * compression_quality)
            
            # Overall temporal harmony (all three aligned)
            overall_harmony = (accessibility + compression_quality + rhythm_quality) / 3.0
            doc_features.append(overall_harmony)
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_beats(self, text: str) -> List[Dict]:
        """
        Extract narrative beats from text.
        
        Uses multiple methods:
        - Structural: Paragraph/section breaks
        - Emotional: Sentiment shifts
        - Semantic: Topic changes
        - Dialogue: Speech acts
        
        Returns
        -------
        beats : list of dict
            Each beat has {'time': float, 'type': str, 'strength': float}
        """
        beats = []
        
        # Structural beats (paragraphs, sections)
        paragraphs = text.split('\n\n')
        section_markers = re.findall(r'(Chapter|Act|Scene|Part)\s+\d+', text, re.I)
        
        # Estimate time per paragraph (rough)
        words = len(text.split())
        time_per_word = 0.004  # ~250 wpm = 0.004 min/word
        total_time = words * time_per_word
        
        current_time = 0
        for i, para in enumerate(paragraphs):
            if para.strip():
                para_words = len(para.split())
                para_time = para_words * time_per_word
                
                beats.append({
                    'time': current_time + para_time / 2,  # Midpoint
                    'type': 'structural',
                    'strength': 0.5
                })
                
                current_time += para_time
        
        # Emotional beats (sentiment shifts)
        emotional_markers = [
            (r'\b(love|joy|happy|delight|ecstatic)\b', 'positive'),
            (r'\b(hate|anger|furious|rage|despair)\b', 'negative'),
            (r'\b(fear|terror|dread|panic|horror)\b', 'fear'),
            (r'\b(surprise|shock|astonish|amaz)', 'surprise'),
        ]
        
        for pattern, emotion in emotional_markers:
            matches = re.finditer(pattern, text, re.I)
            for match in matches:
                # Estimate time based on position in text
                position = match.start() / len(text)
                time = position * total_time
                beats.append({
                    'time': time,
                    'type': 'emotional',
                    'strength': 0.7
                })
        
        # Dialogue beats
        dialogue_markers = re.finditer(r'["\'](.*?)["\']', text)
        for match in dialogue_markers:
            position = match.start() / len(text)
            time = position * total_time
            beats.append({
                'time': time,
                'type': 'dialogue',
                'strength': 0.4
            })
        
        # Action beats
        action_words = [
            'suddenly', 'quickly', 'rushed', 'ran', 'jumped', 'crashed',
            'exploded', 'attacked', 'fought', 'struck', 'burst'
        ]
        for word in action_words:
            pattern = r'\b' + word + r'\b'
            matches = re.finditer(pattern, text, re.I)
            for match in matches:
                position = match.start() / len(text)
                time = position * total_time
                beats.append({
                    'time': time,
                    'type': 'action',
                    'strength': 0.8
                })
        
        # Sort by time and merge nearby beats (within 0.5 time units)
        beats = sorted(beats, key=lambda b: b['time'])
        merged_beats = []
        
        i = 0
        while i < len(beats):
            current_beat = beats[i]
            # Merge beats within 0.5 minutes
            while i + 1 < len(beats) and beats[i+1]['time'] - current_beat['time'] < 0.5:
                i += 1
                # Keep strongest beat
                if beats[i]['strength'] > current_beat['strength']:
                    current_beat = beats[i]
            merged_beats.append(current_beat)
            i += 1
        
        return merged_beats if merged_beats else [{'time': 0, 'type': 'default', 'strength': 0.5}]
    
    def _estimate_duration(self, text: str, metadata: Optional[Dict] = None, idx: int = 0) -> float:
        """
        Estimate narrative duration in minutes.
        
        Uses metadata if available, otherwise estimates from text.
        """
        if metadata and 'durations' in metadata and idx < len(metadata['durations']):
            return metadata['durations'][idx]
        
        # Estimate from text length
        words = len(text.split())
        
        if self.domain in ['novel', 'short_story']:
            # Reading time: ~250 words/minute
            return words / 250
        elif self.domain in ['film', 'tv_episode']:
            # Screenplay: ~1 page = 1 minute, ~100 words/page
            return words / 100
        elif self.domain in ['pop_song', 'symphony']:
            # Lyrics/description length as proxy
            return min(words / 50, 20)  # Cap at 20 minutes
        else:
            # General: assume ~150 wpm
            return words / 150
    
    def _calculate_tau(self, duration: float) -> float:
        """Calculate τ (duration ratio)."""
        natural = self.natural_timescales.get(self.domain, 60)
        return duration / natural if natural > 0 else 1.0
    
    def _calculate_sigma(self, beats: List[Dict], duration: float) -> float:
        """Calculate ς (compression ratio)."""
        if duration <= 0:
            return 1.0
        
        beats_per_minute = len(beats) / duration
        baseline = self.beat_baselines.get(self.domain, 1.0)
        
        return beats_per_minute / baseline if baseline > 0 else 1.0
    
    def _calculate_rho(self, beat_times: List[float]) -> float:
        """Calculate ρ (temporal rhythm)."""
        if len(beat_times) < 2:
            return 0.0
        
        intervals = np.diff(beat_times)
        
        if len(intervals) == 0:
            return 0.0
        
        mu = np.mean(intervals)
        sigma = np.std(intervals)
        
        return sigma / mu if mu > 0 else 0.0
    
    def get_feature_names(self) -> List[str]:
        """Return names of extracted features."""
        return [
            # Tau (Duration) - 10 features
            'tau_raw', 'tau_log', 'tau_squared', 'tau_normalized',
            'tau_compressed', 'tau_standard', 'tau_extended',
            'tau_deviation_abs', 'tau_deviation_squared', 'tau_accessibility',
            
            # Sigma (Compression) - 25 features
            'sigma_raw', 'sigma_log', 'sigma_normalized',
            'beats_per_minute', 'beats_per_minute_log',
            'sigma_very_low', 'sigma_low', 'sigma_optimal', 'sigma_high', 'sigma_very_high',
            'density_first_third', 'density_middle_third', 'density_last_third',
            'density_acceleration', 'density_variance',
            'narrative_density_constant', 'ndc_deviation',
            'compression_quality', 'sustainability_score',
            'peak_compression', 'lull_indicator',
            'sigma_placeholder_1', 'sigma_placeholder_2', 'sigma_placeholder_3',
            
            # Rho (Rhythm) - 15 features
            'rho_raw', 'rho_normalized',
            'rho_metronomic', 'rho_natural', 'rho_chaotic',
            'rhythm_quality',
            'rhythm_ascending_pct', 'rhythm_descending_pct', 'rhythm_alternation_pct',
            'rhythm_evolution', 'rhythm_variance',
            'rhythm_control_constant', 'syncopation',
            'rho_placeholder_1',
            
            # Cross-Variable Interactions - 10 features
            'tau_x_sigma', 'tau_x_rho', 'sigma_x_rho', 'tau_x_sigma_x_rho',
            'temporal_quality_aggregate', 'temporal_stress',
            'temporal_balance_indicator',
            'accessibility_x_sustainability', 'rhythm_x_compression_harmony',
            'overall_temporal_harmony'
        ]

