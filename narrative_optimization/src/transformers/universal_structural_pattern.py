"""
Universal Structural Pattern Transformer

Extracts narrative arc shapes, tension geometry, and archetypal structures
WITHOUT semantic assumptions. Purely geometric/structural analysis.

Philosophy: Discover SHAPE, not CONTENT. Universal patterns across domains.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import warnings

from .base_transformer import TextNarrativeTransformer


class UniversalStructuralPatternTransformer(TextNarrativeTransformer):
    """
    Extract universal structural patterns from narratives.
    
    Analyzes the GEOMETRY of narrative features without semantic assumptions:
    - Arc shapes (rising/falling/oscillating)
    - Tension geometry (buildup/resolution)
    - Symmetry/asymmetry measures
    - Complexity (entropy, diversity)
    - Archetypal shapes (detected geometrically)
    
    Works across ALL domains - same structural patterns in:
    - Sports: Momentum building, comeback arcs
    - Business: Growth curves, disruption patterns
    - Entertainment: Dramatic tension, resolution
    
    Features Extracted (~45 total):
    
    Arc Shape (12 features):
    - Overall trend (linear regression slope)
    - Curvature (second derivative)
    - Inflection points count
    - Arc type (rising/falling/oscillating/flat)
    - Rise rate, fall rate
    - Peak position, valley position
    - Amplitude range
    
    Tension Geometry (10 features):
    - Tension buildup rate
    - Resolution rate
    - Tension peaks count
    - Average tension level
    - Tension variance
    - Max tension, min tension
    - Tension asymmetry
    
    Symmetry/Asymmetry (8 features):
    - First half vs second half difference
    - Mirror symmetry score
    - Rotational symmetry
    - Balance point position
    - Skewness, kurtosis
    
    Complexity (10 features):
    - Shannon entropy
    - Structural diversity (unique patterns)
    - Fractal dimension estimate
    - Autocorrelation strength
    - Frequency domain complexity
    - Dominant frequency
    - Spectral entropy
    
    Archetypal Shapes (5 features):
    - Hero's journey fit (U-shape)
    - Tragedy fit (inverted U)
    - Linear growth fit
    - Exponential growth fit
    - Oscillatory fit
    
    Parameters
    ----------
    n_sequence_points : int, default=50
        Number of points to standardize sequences to
    use_fft : bool, default=True
        Whether to use FFT for frequency analysis
    detect_changepoints : bool, default=True
        Whether to detect regime shifts
    
    Examples
    --------
    >>> transformer = UniversalStructuralPatternTransformer()
    >>> 
    >>> # Option 1: Text input (extracts basic features first)
    >>> features = transformer.fit_transform(texts)
    >>> 
    >>> # Option 2: Pre-computed feature sequences (preferred)
    >>> X = [
    ...     {'feature_sequence': np.array([0.3, 0.4, 0.5, ...]), 'text': "..."},
    ...     ...
    ... ]
    >>> features = transformer.fit_transform(X)
    """
    
    def __init__(
        self,
        n_sequence_points: int = 50,
        use_fft: bool = True,
        detect_changepoints: bool = True
    ):
        super().__init__(
            narrative_id='universal_structural_pattern',
            description='Geometric/structural narrative pattern extraction'
        )
        self.n_sequence_points = n_sequence_points
        self.use_fft = use_fft
        self.detect_changepoints = detect_changepoints
        
        # Will be populated during fit
        self.scaler_ = None
        
    def fit(self, X, y=None):
        """
        Fit transformer to data.
        
        Learns normalization parameters for structural features.
        
        Parameters
        ----------
        X : list
            Training data (texts or feature sequences)
        y : array-like, optional
            Target values (not used, for sklearn compatibility)
            
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Initialize scaler for feature normalization
        self.scaler_ = StandardScaler()
        
        # Extract sequences and fit scaler
        sequences = [self._extract_sequence(item) for item in X]
        
        # Fit scaler on flattened sequences
        if sequences and len(sequences) > 0:
            sample_features = self._extract_structural_features(sequences[0])
            self.scaler_.fit([sample_features])
        
        # Store metadata
        self.metadata['n_samples'] = len(X)
        self.metadata['n_features'] = 45
        self.metadata['feature_names'] = self._get_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform data to structural pattern features.
        
        Parameters
        ----------
        X : list
            Data to transform (same format as fit)
            
        Returns
        -------
        features : ndarray, shape (n_samples, 45)
            Structural pattern features
        """
        self._validate_fitted()
        self._validate_input(X)
        
        features = []
        for item in X:
            # Extract sequence from item
            sequence = self._extract_sequence(item)
            
            # Extract structural features
            feat_vector = self._extract_structural_features(sequence)
            
            features.append(feat_vector)
        
        return np.array(features)
    
    def _extract_sequence(self, item) -> np.ndarray:
        """
        Extract feature sequence from input item.
        
        Parameters
        ----------
        item : str, dict, or array
            Input item
            
        Returns
        -------
        sequence : ndarray
            Feature sequence normalized to n_sequence_points
        """
        if isinstance(item, dict):
            # Check for pre-computed sequence
            if 'feature_sequence' in item:
                sequence = np.array(item['feature_sequence'])
            elif 'text' in item:
                sequence = self._text_to_sequence(item['text'])
            else:
                raise ValueError("Dict must contain 'feature_sequence' or 'text'")
        elif isinstance(item, (list, np.ndarray)):
            sequence = np.array(item)
        else:
            # Assume string text
            sequence = self._text_to_sequence(str(item))
        
        # Normalize to standard length
        return self._normalize_sequence_length(sequence)
    
    def _text_to_sequence(self, text: str) -> np.ndarray:
        """
        Convert text to feature sequence using basic heuristics.
        
        This is a fallback when pre-computed sequences aren't available.
        Ideally, use pre-computed ж features for better accuracy.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        sequence : ndarray
            Feature sequence
        """
        if not text or len(text) == 0:
            return np.zeros(self.n_sequence_points)
        
        # Simple approach: word-level features
        words = text.split()
        if len(words) == 0:
            return np.zeros(self.n_sequence_points)
        
        # Create sequence based on word length variation (proxy for complexity)
        sequence = np.array([len(w) for w in words])
        
        # Add sentence-level features if possible
        sentences = text.split('.')
        if len(sentences) > 1:
            # Sentence length variation
            sent_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sent_lengths:
                # Interpolate to match word count
                sequence = sequence * (1 + np.repeat(sent_lengths, len(words) // len(sent_lengths) + 1)[:len(words)] / np.mean(sent_lengths))
        
        return sequence
    
    def _normalize_sequence_length(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize sequence to standard length via interpolation.
        
        Parameters
        ----------
        sequence : ndarray
            Input sequence of any length
            
        Returns
        -------
        normalized : ndarray
            Sequence of length n_sequence_points
        """
        if len(sequence) == 0:
            return np.zeros(self.n_sequence_points)
        
        if len(sequence) == self.n_sequence_points:
            return sequence
        
        # Interpolate to standard length
        x_old = np.linspace(0, 1, len(sequence))
        x_new = np.linspace(0, 1, self.n_sequence_points)
        
        interpolator = interp1d(x_old, sequence, kind='linear', fill_value='extrapolate')
        normalized = interpolator(x_new)
        
        return normalized
    
    def _extract_structural_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract all structural features from sequence.
        
        Parameters
        ----------
        sequence : ndarray
            Normalized feature sequence
            
        Returns
        -------
        features : ndarray, shape (45,)
            All structural features
        """
        features = []
        
        # 1. Arc Shape Features (12)
        features.extend(self._extract_arc_features(sequence))
        
        # 2. Tension Geometry Features (10)
        features.extend(self._extract_tension_features(sequence))
        
        # 3. Symmetry/Asymmetry Features (8)
        features.extend(self._extract_symmetry_features(sequence))
        
        # 4. Complexity Features (10)
        features.extend(self._extract_complexity_features(sequence))
        
        # 5. Archetypal Shape Fits (5)
        features.extend(self._extract_archetypal_fits(sequence))
        
        return np.array(features)
    
    def _extract_arc_features(self, seq: np.ndarray) -> List[float]:
        """Extract arc shape features (12)."""
        features = []
        
        # Overall trend (linear regression slope)
        x = np.arange(len(seq))
        if len(seq) > 1 and np.std(seq) > 0:
            slope, intercept = np.polyfit(x, seq, 1)
            features.append(slope)
        else:
            features.append(0.0)
        
        # Curvature (second derivative)
        if len(seq) > 2:
            first_deriv = np.diff(seq)
            second_deriv = np.diff(first_deriv)
            avg_curvature = np.mean(np.abs(second_deriv))
            features.append(avg_curvature)
        else:
            features.append(0.0)
        
        # Inflection points count
        if len(seq) > 2:
            second_deriv = np.diff(np.diff(seq))
            sign_changes = np.sum(np.diff(np.sign(second_deriv)) != 0)
            features.append(sign_changes)
        else:
            features.append(0.0)
        
        # Arc type classification (0=flat, 1=rising, 2=falling, 3=oscillating)
        if len(seq) > 1:
            slope_magnitude = abs(slope) if 'slope' in locals() else 0
            oscillation = np.sum(np.abs(np.diff(np.sign(np.diff(seq))))) if len(seq) > 2 else 0
            
            if oscillation > len(seq) * 0.3:
                arc_type = 3  # Oscillating
            elif slope_magnitude < 0.01:
                arc_type = 0  # Flat
            elif slope > 0:
                arc_type = 1  # Rising
            else:
                arc_type = 2  # Falling
            features.append(arc_type)
        else:
            features.append(0.0)
        
        # Rise rate and fall rate
        if len(seq) > 1:
            diffs = np.diff(seq)
            rises = diffs[diffs > 0]
            falls = diffs[diffs < 0]
            
            rise_rate = np.mean(rises) if len(rises) > 0 else 0.0
            fall_rate = np.mean(np.abs(falls)) if len(falls) > 0 else 0.0
            
            features.append(rise_rate)
            features.append(fall_rate)
        else:
            features.extend([0.0, 0.0])
        
        # Peak and valley positions (normalized)
        if len(seq) > 0:
            peak_pos = np.argmax(seq) / len(seq)
            valley_pos = np.argmin(seq) / len(seq)
            features.append(peak_pos)
            features.append(valley_pos)
        else:
            features.extend([0.5, 0.5])
        
        # Amplitude range
        amplitude = np.max(seq) - np.min(seq) if len(seq) > 0 else 0.0
        features.append(amplitude)
        
        # Average level
        avg_level = np.mean(seq) if len(seq) > 0 else 0.0
        features.append(avg_level)
        
        # Range normalized by mean
        range_norm = amplitude / (avg_level + 1e-10) if len(seq) > 0 else 0.0
        features.append(range_norm)
        
        return features
    
    def _extract_tension_features(self, seq: np.ndarray) -> List[float]:
        """Extract tension geometry features (10)."""
        features = []
        
        # Tension proxy: use variance and changes
        if len(seq) > 1:
            # Tension buildup: increasing variance over time
            half_point = len(seq) // 2
            first_half_var = np.var(seq[:half_point]) if half_point > 0 else 0
            second_half_var = np.var(seq[half_point:]) if half_point < len(seq) else 0
            
            buildup_rate = second_half_var - first_half_var
            features.append(buildup_rate)
            
            # Resolution rate: decreasing variance in final third
            third_point = len(seq) * 2 // 3
            middle_var = np.var(seq[half_point:third_point]) if third_point > half_point else 0
            final_var = np.var(seq[third_point:]) if third_point < len(seq) else 0
            
            resolution_rate = middle_var - final_var
            features.append(resolution_rate)
        else:
            features.extend([0.0, 0.0])
        
        # Tension peaks count (local maxima in rate of change)
        if len(seq) > 2:
            rate_of_change = np.abs(np.diff(seq))
            peaks, _ = signal.find_peaks(rate_of_change, distance=len(seq)//10)
            features.append(len(peaks))
        else:
            features.append(0.0)
        
        # Average tension level (std dev)
        avg_tension = np.std(seq) if len(seq) > 0 else 0.0
        features.append(avg_tension)
        
        # Tension variance (how much does tension vary)
        if len(seq) > 5:
            window_size = len(seq) // 5
            rolling_std = [np.std(seq[i:i+window_size]) for i in range(len(seq) - window_size)]
            tension_variance = np.var(rolling_std) if rolling_std else 0.0
            features.append(tension_variance)
        else:
            features.append(0.0)
        
        # Max and min tension
        max_tension = np.max(np.abs(np.diff(seq))) if len(seq) > 1 else 0.0
        min_tension = np.min(np.abs(np.diff(seq))) if len(seq) > 1 else 0.0
        features.append(max_tension)
        features.append(min_tension)
        
        # Tension asymmetry (first half vs second half)
        if len(seq) > 1:
            half = len(seq) // 2
            first_tension = np.std(seq[:half]) if half > 0 else 0
            second_tension = np.std(seq[half:]) if half < len(seq) else 0
            asymmetry = abs(first_tension - second_tension)
            features.append(asymmetry)
        else:
            features.append(0.0)
        
        # Tension concentration (where is tension highest - beginning, middle, end)
        if len(seq) > 2:
            thirds = len(seq) // 3
            begin_tension = np.std(seq[:thirds]) if thirds > 0 else 0
            middle_tension = np.std(seq[thirds:2*thirds]) if 2*thirds < len(seq) else 0
            end_tension = np.std(seq[2*thirds:]) if 2*thirds < len(seq) else 0
            
            max_tension_third = np.argmax([begin_tension, middle_tension, end_tension])
            features.append(max_tension_third)
        else:
            features.append(1.0)  # Middle by default
        
        # Overall tension trend (is tension increasing or decreasing)
        if len(seq) > 5:
            window_stds = [np.std(seq[i:i+5]) for i in range(len(seq)-5)]
            if len(window_stds) > 1:
                tension_slope, _ = np.polyfit(np.arange(len(window_stds)), window_stds, 1)
                features.append(tension_slope)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_symmetry_features(self, seq: np.ndarray) -> List[float]:
        """Extract symmetry/asymmetry features (8)."""
        features = []
        
        if len(seq) == 0:
            return [0.0] * 8
        
        # First half vs second half difference
        half = len(seq) // 2
        if half > 0:
            first_mean = np.mean(seq[:half])
            second_mean = np.mean(seq[half:])
            half_diff = abs(first_mean - second_mean)
            features.append(half_diff)
        else:
            features.append(0.0)
        
        # Mirror symmetry score (correlation with reversed)
        if len(seq) > 1:
            reversed_seq = seq[::-1]
            mirror_corr = np.corrcoef(seq, reversed_seq)[0, 1] if np.std(seq) > 0 else 0.0
            features.append(mirror_corr if not np.isnan(mirror_corr) else 0.0)
        else:
            features.append(0.0)
        
        # Rotational symmetry (180 degree)
        if len(seq) > 1:
            inverted = -seq + 2 * np.mean(seq)
            rotational_corr = np.corrcoef(seq, inverted)[0, 1] if np.std(seq) > 0 else 0.0
            features.append(rotational_corr if not np.isnan(rotational_corr) else 0.0)
        else:
            features.append(0.0)
        
        # Balance point position (center of mass)
        if len(seq) > 0 and np.sum(np.abs(seq)) > 0:
            weights = np.abs(seq)
            balance_point = np.average(np.arange(len(seq)), weights=weights) / len(seq)
            features.append(balance_point)
        else:
            features.append(0.5)
        
        # Skewness
        if len(seq) > 2 and np.std(seq) > 0:
            skew = stats.skew(seq)
            features.append(skew if not np.isnan(skew) else 0.0)
        else:
            features.append(0.0)
        
        # Kurtosis
        if len(seq) > 3 and np.std(seq) > 0:
            kurt = stats.kurtosis(seq)
            features.append(kurt if not np.isnan(kurt) else 0.0)
        else:
            features.append(0.0)
        
        # Left-right imbalance
        if half > 0:
            left_energy = np.sum(seq[:half] ** 2)
            right_energy = np.sum(seq[half:] ** 2)
            total_energy = left_energy + right_energy
            imbalance = abs(left_energy - right_energy) / (total_energy + 1e-10)
            features.append(imbalance)
        else:
            features.append(0.0)
        
        # Symmetry score (combined metric)
        symmetry_score = (features[1] + 1) / 2  # Mirror correlation normalized to 0-1
        features.append(symmetry_score)
        
        return features
    
    def _extract_complexity_features(self, seq: np.ndarray) -> List[float]:
        """Extract complexity features (10)."""
        features = []
        
        if len(seq) == 0:
            return [0.0] * 10
        
        # Shannon entropy
        if len(seq) > 0 and np.std(seq) > 0:
            # Discretize sequence into bins
            hist, _ = np.histogram(seq, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features.append(entropy)
        else:
            features.append(0.0)
        
        # Structural diversity (unique patterns)
        if len(seq) > 3:
            # Count unique 3-grams
            trigrams = [tuple(seq[i:i+3]) for i in range(len(seq)-2)]
            unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 0.0
            features.append(unique_ratio)
        else:
            features.append(0.0)
        
        # Fractal dimension estimate (box-counting)
        if len(seq) > 4:
            # Simplified Higuchi fractal dimension
            k_max = min(10, len(seq) // 2)
            lengths = []
            for k in range(1, k_max):
                Lk = 0
                for m in range(k):
                    subset = seq[m::k]
                    if len(subset) > 1:
                        Lk += np.sum(np.abs(np.diff(subset)))
                if Lk > 0:
                    lengths.append(Lk / k)
            
            if len(lengths) > 1:
                log_k = np.log(np.arange(1, len(lengths) + 1))
                log_l = np.log(lengths)
                fractal_dim = -np.polyfit(log_k, log_l, 1)[0]
                features.append(fractal_dim)
            else:
                features.append(1.0)
        else:
            features.append(1.0)
        
        # Autocorrelation strength
        if len(seq) > 1 and np.std(seq) > 0:
            # Lag-1 autocorrelation
            autocorr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
            features.append(autocorr if not np.isnan(autocorr) else 0.0)
        else:
            features.append(0.0)
        
        # Frequency domain complexity (if FFT enabled)
        if self.use_fft and len(seq) > 4:
            # FFT
            fft_vals = fft(seq - np.mean(seq))
            power_spectrum = np.abs(fft_vals[:len(fft_vals)//2]) ** 2
            
            # Spectral complexity (entropy of power spectrum)
            power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
            power_spectrum_norm = power_spectrum_norm[power_spectrum_norm > 0]
            spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-10))
            features.append(spectral_entropy)
            
            # Dominant frequency
            freqs = fftfreq(len(seq))[:len(seq)//2]
            dominant_freq_idx = np.argmax(power_spectrum)
            dominant_freq = abs(freqs[dominant_freq_idx]) if dominant_freq_idx < len(freqs) else 0.0
            features.append(dominant_freq)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-10)
            features.append(spectral_centroid)
            
            # Bandwidth (spectral spread)
            bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / (np.sum(power_spectrum) + 1e-10))
            features.append(bandwidth)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Change point density
        if len(seq) > 5 and self.detect_changepoints:
            # Simple change point detection: significant changes in local mean
            window = len(seq) // 10 + 1
            local_means = [np.mean(seq[max(0, i-window):min(len(seq), i+window)]) for i in range(len(seq))]
            changepoints = np.sum(np.abs(np.diff(local_means)) > np.std(seq) * 0.5)
            changepoint_density = changepoints / len(seq)
            features.append(changepoint_density)
        else:
            features.append(0.0)
        
        # Pattern repetition score
        if len(seq) > 10:
            # Check for repeating patterns using autocorrelation at different lags
            max_lag = min(len(seq) // 3, 20)
            autocorrs = []
            for lag in range(1, max_lag):
                if len(seq) - lag > 0 and np.std(seq[:-lag]) > 0 and np.std(seq[lag:]) > 0:
                    corr = np.corrcoef(seq[:-lag], seq[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))
            
            repetition_score = np.max(autocorrs) if autocorrs else 0.0
            features.append(repetition_score)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_archetypal_fits(self, seq: np.ndarray) -> List[float]:
        """Extract archetypal shape fit scores (5)."""
        features = []
        
        if len(seq) < 3:
            return [0.0] * 5
        
        x = np.linspace(0, 1, len(seq))
        
        # 1. Hero's journey fit (U-shape: high -> low -> high)
        u_shape = 1 - 4 * (x - 0.5) ** 2  # Inverted parabola
        hero_fit = np.corrcoef(seq, u_shape)[0, 1] if np.std(seq) > 0 else 0.0
        features.append(hero_fit if not np.isnan(hero_fit) else 0.0)
        
        # 2. Tragedy fit (inverted U: low -> high -> low)
        inv_u_shape = 4 * (x - 0.5) ** 2  # Parabola
        tragedy_fit = np.corrcoef(seq, inv_u_shape)[0, 1] if np.std(seq) > 0 else 0.0
        features.append(tragedy_fit if not np.isnan(tragedy_fit) else 0.0)
        
        # 3. Linear growth fit
        linear = x
        linear_fit = np.corrcoef(seq, linear)[0, 1] if np.std(seq) > 0 else 0.0
        features.append(linear_fit if not np.isnan(linear_fit) else 0.0)
        
        # 4. Exponential growth fit
        exp_curve = np.exp(3 * x) - 1  # Exponential
        exp_fit = np.corrcoef(seq, exp_curve)[0, 1] if np.std(seq) > 0 else 0.0
        features.append(exp_fit if not np.isnan(exp_fit) else 0.0)
        
        # 5. Oscillatory fit (sine wave)
        oscillation = np.sin(4 * np.pi * x)
        osc_fit = np.corrcoef(seq, oscillation)[0, 1] if np.std(seq) > 0 else 0.0
        features.append(osc_fit if not np.isnan(osc_fit) else 0.0)
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Arc shape (12)
        names.extend([
            'arc_overall_slope',
            'arc_avg_curvature',
            'arc_inflection_points',
            'arc_type',
            'arc_rise_rate',
            'arc_fall_rate',
            'arc_peak_position',
            'arc_valley_position',
            'arc_amplitude',
            'arc_avg_level',
            'arc_range_normalized',
            'arc_reserved'
        ])
        
        # Tension geometry (10)
        names.extend([
            'tension_buildup_rate',
            'tension_resolution_rate',
            'tension_peaks_count',
            'tension_avg_level',
            'tension_variance',
            'tension_max',
            'tension_min',
            'tension_asymmetry',
            'tension_concentration',
            'tension_trend'
        ])
        
        # Symmetry (8)
        names.extend([
            'symmetry_half_difference',
            'symmetry_mirror_correlation',
            'symmetry_rotational',
            'symmetry_balance_point',
            'symmetry_skewness',
            'symmetry_kurtosis',
            'symmetry_lr_imbalance',
            'symmetry_score'
        ])
        
        # Complexity (10)
        names.extend([
            'complexity_entropy',
            'complexity_diversity',
            'complexity_fractal_dim',
            'complexity_autocorrelation',
            'complexity_spectral_entropy',
            'complexity_dominant_freq',
            'complexity_spectral_centroid',
            'complexity_bandwidth',
            'complexity_changepoint_density',
            'complexity_repetition'
        ])
        
        # Archetypal fits (5)
        names.extend([
            'archetype_hero_journey_fit',
            'archetype_tragedy_fit',
            'archetype_linear_fit',
            'archetype_exponential_fit',
            'archetype_oscillatory_fit'
        ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of discovered patterns."""
        if not self.is_fitted_:
            return "Transformer not fitted yet."
        
        interpretation = f"""
Universal Structural Pattern Analysis

Extracted geometric/structural patterns WITHOUT semantic assumptions.

Features Extracted: {self.metadata['n_features']}
Samples Analyzed: {self.metadata.get('n_samples', 'Unknown')}

Pattern Categories:
1. Arc Shape (12 features): Overall trajectory, curvature, inflection points
2. Tension Geometry (10 features): Buildup, resolution, peaks
3. Symmetry/Asymmetry (8 features): Balance, mirror properties
4. Complexity (10 features): Entropy, fractal dimension, frequency analysis
5. Archetypal Fits (5 features): Hero's journey, tragedy, growth curves

These features enable DISCOVERY of narrative patterns across domains.
The learning system can identify which geometric patterns predict outcomes
without assuming domain-specific semantics.
"""
        return interpretation.strip()

