"""
Temporal Derivative Transformer

Captures rate-of-change in narrative features WITHOUT assuming which
temporal patterns matter.

Philosophy: Extract VELOCITY and ACCELERATION, discover which momentum
patterns predict outcomes.

Universal across domains with temporal data: sports seasons, startup growth,
narrative arcs over time, etc.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import signal
from scipy.stats import linregress
import warnings

from .base_transformer import FeatureNarrativeTransformer


class TemporalDerivativeTransformer(FeatureNarrativeTransformer):
    """
    Extract temporal derivative features (velocity, acceleration, momentum).
    
    Analyzes the DYNAMICS of narrative evolution without semantic assumptions:
    - First derivatives (velocity): How fast features change
    - Second derivatives (acceleration): How velocity changes
    - Trend consistency: Smooth vs erratic evolution
    - Regime shifts: Statistical breaks in patterns
    - Temporal autocorrelation: Momentum persistence
    
    Works across ALL domains with temporal sequences:
    - NBA: Season-long momentum building
    - Startups: Growth trajectory analysis
    - Narratives: Story development pace
    
    Discovers patterns like:
    - "Late season acceleration predicts outcomes"
    - "Consistent velocity matters more than current state"
    - "Regime shifts signal turning points"
    
    Features Extracted (~40 total):
    
    First Derivatives - Velocity (12 features):
    - Mean velocity (average rate of change)
    - Velocity magnitude (speed regardless of direction)
    - Positive velocity rate (% of time increasing)
    - Negative velocity rate (% of time decreasing)
    - Velocity variance (consistency of change)
    - Recent velocity (last N periods)
    - Velocity trend (is velocity increasing?)
    - Max velocity, min velocity
    - Velocity asymmetry (up vs down rates)
    - Velocity persistence (autocorrelation)
    
    Second Derivatives - Acceleration (10 features):
    - Mean acceleration
    - Acceleration magnitude
    - Positive acceleration rate
    - Acceleration variance
    - Recent acceleration
    - Max acceleration, min acceleration
    - Jerk (third derivative - rate of acceleration change)
    - Acceleration consistency
    - Momentum building score
    
    Trend Consistency (8 features):
    - Linear fit R² (how linear is evolution)
    - Trend strength (correlation with time)
    - Volatility (relative variance)
    - Smoothness (inverse of roughness)
    - Directionality (net direction consistency)
    - Path efficiency (direct distance / actual path)
    - Oscillation frequency
    - Drift magnitude
    
    Regime Shifts (5 features):
    - Number of detected changepoints
    - Changepoint density
    - Largest shift magnitude
    - Time since last shift
    - Regime stability
    
    Temporal Autocorrelation (5 features):
    - Lag-1 autocorrelation
    - Lag-N autocorrelation (longer term)
    - Momentum persistence score
    - Mean reversion tendency
    - Predictability (autocorrelation strength)
    
    Parameters
    ----------
    recent_window : int, default=5
        Number of recent periods for "recent" calculations
    detect_changepoints : bool, default=True
        Whether to detect regime shifts
    normalize_by_time : bool, default=True
        Whether to normalize derivatives by time intervals
    
    Examples
    --------
    >>> transformer = TemporalDerivativeTransformer()
    >>> 
    >>> # Temporal sequence data
    >>> X = [
    ...     {
    ...         'feature_history': np.array([[0.3, 0.4], [0.35, 0.45], [0.4, 0.5], ...]),
    ...         'timestamps': [1, 2, 3, ...],  # or datetime objects
    ...         'current_features': np.array([0.45, 0.55])
    ...     },
    ...     ...
    ... ]
    >>> features = transformer.fit_transform(X)
    >>> 
    >>> # Discover: Does recent acceleration predict outcomes?
    >>> recent_accel = features[:, 15]  # Recent acceleration feature
    """
    
    def __init__(
        self,
        recent_window: int = 5,
        detect_changepoints: bool = True,
        normalize_by_time: bool = True
    ):
        super().__init__(
            narrative_id='temporal_derivative',
            description='Temporal velocity and acceleration analysis'
        )
        self.recent_window = recent_window
        self.detect_changepoints = detect_changepoints
        self.normalize_by_time = normalize_by_time
    
    def _validate_input(self, X):
        """Override base validation - we accept list of dicts."""
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty")
        return True
        
    def fit(self, X, y=None):
        """
        Fit transformer to data.
        
        Parameters
        ----------
        X : list of dict
            Training temporal sequences
        y : array-like, optional
            Target values (not used, for sklearn compatibility)
            
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Store metadata
        self.metadata['n_samples'] = len(X)
        self.metadata['n_features'] = 40
        self.metadata['feature_names'] = self._get_feature_names()
        self.metadata['recent_window'] = self.recent_window
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform temporal sequences to derivative features.
        
        Parameters
        ----------
        X : list of dict
            Temporal sequences to transform
            
        Returns
        -------
        features : ndarray, shape (n_samples, 40)
            Temporal derivative features
        """
        self._validate_fitted()
        # Skip base class validation - we handle list of dicts
        
        features = []
        for item in X:
            # Extract temporal sequence
            history, timestamps = self._extract_temporal_sequence(item)
            
            # Extract derivative features
            feat_vector = self._extract_derivative_features(history, timestamps)
            
            features.append(feat_vector)
        
        return np.array(features)
    
    def _extract_temporal_sequence(self, item: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature history and timestamps.
        
        Parameters
        ----------
        item : dict
            Item with temporal data
            
        Returns
        -------
        history, timestamps : tuple of ndarrays
            Feature history matrix and corresponding timestamps
        """
        if not isinstance(item, dict):
            raise ValueError("Input must be dict with 'feature_history' and 'timestamps'")
        
        history = item.get('feature_history')
        timestamps = item.get('timestamps')
        
        if history is None:
            raise ValueError("Dict must contain 'feature_history'")
        
        history = np.array(history)
        
        # Handle timestamps
        if timestamps is None:
            # Assume uniform spacing
            timestamps = np.arange(len(history))
        else:
            timestamps = np.array(timestamps)
            # Convert to numeric if datetime objects
            if hasattr(timestamps[0], 'timestamp'):
                timestamps = np.array([t.timestamp() for t in timestamps])
        
        return history, timestamps
    
    def _extract_derivative_features(
        self,
        history: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Extract all derivative features from temporal sequence.
        
        Parameters
        ----------
        history : ndarray, shape (n_timesteps, n_features)
            Feature history
        timestamps : ndarray, shape (n_timesteps,)
            Corresponding timestamps
            
        Returns
        -------
        features : ndarray, shape (40,)
            Temporal derivative features
        """
        # Aggregate history across feature dimensions (use mean trajectory)
        if history.ndim == 2:
            trajectory = np.mean(history, axis=1)
        else:
            trajectory = history
        
        features = []
        
        # Compute derivatives
        dt = np.diff(timestamps)
        if self.normalize_by_time and len(dt) > 0 and np.min(dt) > 0:
            velocity = np.diff(trajectory) / dt
        else:
            velocity = np.diff(trajectory)
        
        if len(velocity) > 1:
            dt2 = dt[:-1] if self.normalize_by_time else np.ones(len(velocity) - 1)
            if self.normalize_by_time and len(dt2) > 0 and np.min(dt2) > 0:
                acceleration = np.diff(velocity) / dt2
            else:
                acceleration = np.diff(velocity)
        else:
            acceleration = np.array([0.0])
        
        # 1. First Derivatives - Velocity (12)
        features.extend(self._compute_velocity_features(velocity, trajectory))
        
        # 2. Second Derivatives - Acceleration (10)
        features.extend(self._compute_acceleration_features(acceleration, velocity))
        
        # 3. Trend Consistency (8)
        features.extend(self._compute_trend_features(trajectory, timestamps))
        
        # 4. Regime Shifts (5)
        features.extend(self._compute_regime_features(trajectory, timestamps))
        
        # 5. Temporal Autocorrelation (5)
        features.extend(self._compute_autocorrelation_features(trajectory))
        
        return np.array(features)
    
    def _compute_velocity_features(self, velocity: np.ndarray, trajectory: np.ndarray) -> List[float]:
        """Compute velocity features (12)."""
        features = []
        
        if len(velocity) == 0:
            return [0.0] * 12
        
        # Mean velocity
        mean_vel = np.mean(velocity)
        features.append(mean_vel)
        
        # Velocity magnitude (speed)
        magnitude = np.mean(np.abs(velocity))
        features.append(magnitude)
        
        # Positive and negative velocity rates
        pos_rate = np.mean(velocity > 0)
        neg_rate = np.mean(velocity < 0)
        features.append(pos_rate)
        features.append(neg_rate)
        
        # Velocity variance
        vel_variance = np.var(velocity)
        features.append(vel_variance)
        
        # Recent velocity (last N periods)
        recent_n = min(self.recent_window, len(velocity))
        recent_vel = np.mean(velocity[-recent_n:]) if recent_n > 0 else 0.0
        features.append(recent_vel)
        
        # Velocity trend (is velocity increasing?)
        if len(velocity) > 2:
            vel_trend = linregress(np.arange(len(velocity)), velocity).slope
            features.append(vel_trend)
        else:
            features.append(0.0)
        
        # Max and min velocity
        max_vel = np.max(velocity)
        min_vel = np.min(velocity)
        features.append(max_vel)
        features.append(min_vel)
        
        # Velocity asymmetry (up vs down)
        if pos_rate > 0 and neg_rate > 0:
            up_vel = np.mean(velocity[velocity > 0])
            down_vel = np.mean(np.abs(velocity[velocity < 0]))
            asymmetry = abs(up_vel - down_vel)
            features.append(asymmetry)
        else:
            features.append(magnitude)
        
        # Velocity persistence (lag-1 autocorrelation)
        if len(velocity) > 1:
            if np.std(velocity[:-1]) > 0 and np.std(velocity[1:]) > 0:
                persistence = np.corrcoef(velocity[:-1], velocity[1:])[0, 1]
                features.append(persistence if not np.isnan(persistence) else 0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Current vs historical velocity ratio
        if magnitude > 0:
            current_ratio = abs(recent_vel) / magnitude
            features.append(current_ratio)
        else:
            features.append(1.0)
        
        return features
    
    def _compute_acceleration_features(
        self,
        acceleration: np.ndarray,
        velocity: np.ndarray
    ) -> List[float]:
        """Compute acceleration features (10)."""
        features = []
        
        if len(acceleration) == 0:
            return [0.0] * 10
        
        # Mean acceleration
        mean_accel = np.mean(acceleration)
        features.append(mean_accel)
        
        # Acceleration magnitude
        accel_magnitude = np.mean(np.abs(acceleration))
        features.append(accel_magnitude)
        
        # Positive acceleration rate (speeding up)
        pos_accel_rate = np.mean(acceleration > 0)
        features.append(pos_accel_rate)
        
        # Acceleration variance
        accel_variance = np.var(acceleration)
        features.append(accel_variance)
        
        # Recent acceleration
        recent_n = min(self.recent_window, len(acceleration))
        recent_accel = np.mean(acceleration[-recent_n:]) if recent_n > 0 else 0.0
        features.append(recent_accel)
        
        # Max and min acceleration
        max_accel = np.max(acceleration)
        min_accel = np.min(acceleration)
        features.append(max_accel)
        features.append(min_accel)
        
        # Jerk (third derivative - rate of acceleration change)
        if len(acceleration) > 1:
            jerk = np.mean(np.abs(np.diff(acceleration)))
            features.append(jerk)
        else:
            features.append(0.0)
        
        # Acceleration consistency (low variance = consistent)
        if accel_magnitude > 0:
            consistency = 1.0 / (1.0 + accel_variance / (accel_magnitude + 1e-10))
            features.append(consistency)
        else:
            features.append(1.0)
        
        # Momentum building score (consistent positive acceleration)
        if len(acceleration) > 3:
            # Count sustained positive acceleration periods
            sustained = 0
            current_streak = 0
            for a in acceleration:
                if a > 0:
                    current_streak += 1
                else:
                    if current_streak >= 2:
                        sustained += 1
                    current_streak = 0
            momentum_score = sustained / len(acceleration)
            features.append(momentum_score)
        else:
            features.append(pos_accel_rate)
        
        return features
    
    def _compute_trend_features(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray
    ) -> List[float]:
        """Compute trend consistency features (8)."""
        features = []
        
        if len(trajectory) < 2:
            return [0.0] * 8
        
        # Linear fit R² (how linear is evolution)
        if np.std(trajectory) > 0:
            result = linregress(timestamps, trajectory)
            r_squared = result.rvalue ** 2
            trend_strength = abs(result.rvalue)
            features.append(r_squared)
            features.append(trend_strength)
        else:
            features.extend([1.0, 0.0])  # Perfectly flat = perfectly linear but no trend
        
        # Volatility (relative variance)
        mean_level = np.mean(np.abs(trajectory))
        if mean_level > 0:
            volatility = np.std(trajectory) / mean_level
            features.append(volatility)
        else:
            features.append(0.0)
        
        # Smoothness (inverse of roughness)
        if len(trajectory) > 2:
            second_diff = np.diff(np.diff(trajectory))
            roughness = np.mean(np.abs(second_diff))
            smoothness = 1.0 / (1.0 + roughness)
            features.append(smoothness)
        else:
            features.append(1.0)
        
        # Directionality (net direction consistency)
        if len(trajectory) > 1:
            changes = np.diff(trajectory)
            if len(changes) > 0:
                net_direction = np.sum(changes) / (np.sum(np.abs(changes)) + 1e-10)
                features.append(abs(net_direction))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Path efficiency (straight line distance / actual path length)
        straight_dist = abs(trajectory[-1] - trajectory[0])
        actual_path = np.sum(np.abs(np.diff(trajectory)))
        if actual_path > 0:
            efficiency = straight_dist / actual_path
            features.append(efficiency)
        else:
            features.append(1.0)
        
        # Oscillation frequency (via FFT)
        if len(trajectory) > 4:
            from scipy.fft import fft, fftfreq
            fft_vals = fft(trajectory - np.mean(trajectory))
            power = np.abs(fft_vals[:len(fft_vals)//2]) ** 2
            if np.sum(power) > 0:
                freqs = fftfreq(len(trajectory))[:len(trajectory)//2]
                # Dominant frequency
                dominant_freq = abs(freqs[np.argmax(power)])
                features.append(dominant_freq)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Drift magnitude (net change)
        drift = trajectory[-1] - trajectory[0]
        features.append(abs(drift))
        
        return features
    
    def _compute_regime_features(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray
    ) -> List[float]:
        """Compute regime shift features (5)."""
        features = []
        
        if len(trajectory) < 5 or not self.detect_changepoints:
            return [0.0] * 5
        
        # Simple changepoint detection: large changes in local mean
        window = max(3, len(trajectory) // 10)
        local_means = []
        for i in range(len(trajectory)):
            start = max(0, i - window)
            end = min(len(trajectory), i + window)
            local_means.append(np.mean(trajectory[start:end]))
        
        local_means = np.array(local_means)
        
        # Detect changepoints as large changes in local mean
        threshold = np.std(trajectory) * 0.75
        changes = np.abs(np.diff(local_means))
        changepoints = changes > threshold
        
        # Number of changepoints
        n_changepoints = np.sum(changepoints)
        features.append(n_changepoints)
        
        # Changepoint density
        density = n_changepoints / len(trajectory)
        features.append(density)
        
        # Largest shift magnitude
        if len(changes) > 0:
            largest_shift = np.max(changes)
            features.append(largest_shift)
        else:
            features.append(0.0)
        
        # Time since last changepoint (normalized)
        if n_changepoints > 0:
            last_changepoint_idx = np.where(changepoints)[0][-1]
            time_since = (len(trajectory) - last_changepoint_idx) / len(trajectory)
            features.append(time_since)
        else:
            features.append(1.0)  # No changepoints = full time
        
        # Regime stability (inverse of changepoint density)
        stability = 1.0 / (1.0 + density * 10)
        features.append(stability)
        
        return features
    
    def _compute_autocorrelation_features(self, trajectory: np.ndarray) -> List[float]:
        """Compute temporal autocorrelation features (5)."""
        features = []
        
        if len(trajectory) < 3:
            return [0.0] * 5
        
        # Lag-1 autocorrelation
        if np.std(trajectory[:-1]) > 0 and np.std(trajectory[1:]) > 0:
            lag1_autocorr = np.corrcoef(trajectory[:-1], trajectory[1:])[0, 1]
            features.append(lag1_autocorr if not np.isnan(lag1_autocorr) else 0.0)
        else:
            features.append(0.0)
        
        # Lag-N autocorrelation (longer term)
        lag_n = min(5, len(trajectory) // 3)
        if lag_n > 1 and len(trajectory) > lag_n:
            if np.std(trajectory[:-lag_n]) > 0 and np.std(trajectory[lag_n:]) > 0:
                lagn_autocorr = np.corrcoef(trajectory[:-lag_n], trajectory[lag_n:])[0, 1]
                features.append(lagn_autocorr if not np.isnan(lagn_autocorr) else 0.0)
            else:
                features.append(0.0)
        else:
            features.append(features[0])  # Use lag-1 as fallback
        
        # Momentum persistence (average of autocorrelations)
        persistence = (features[0] + features[1]) / 2
        features.append(persistence)
        
        # Mean reversion tendency (negative autocorrelation)
        mean_reversion = max(0, -features[0])
        features.append(mean_reversion)
        
        # Predictability (autocorrelation strength)
        predictability = abs(features[0])
        features.append(predictability)
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Velocity (12)
        names.extend([
            'velocity_mean',
            'velocity_magnitude',
            'velocity_positive_rate',
            'velocity_negative_rate',
            'velocity_variance',
            'velocity_recent',
            'velocity_trend',
            'velocity_max',
            'velocity_min',
            'velocity_asymmetry',
            'velocity_persistence',
            'velocity_current_ratio'
        ])
        
        # Acceleration (10)
        names.extend([
            'acceleration_mean',
            'acceleration_magnitude',
            'acceleration_positive_rate',
            'acceleration_variance',
            'acceleration_recent',
            'acceleration_max',
            'acceleration_min',
            'acceleration_jerk',
            'acceleration_consistency',
            'momentum_building_score'
        ])
        
        # Trend consistency (8)
        names.extend([
            'trend_r_squared',
            'trend_strength',
            'trend_volatility',
            'trend_smoothness',
            'trend_directionality',
            'trend_path_efficiency',
            'trend_oscillation_frequency',
            'trend_drift_magnitude'
        ])
        
        # Regime shifts (5)
        names.extend([
            'regime_changepoints_count',
            'regime_changepoint_density',
            'regime_largest_shift',
            'regime_time_since_last',
            'regime_stability'
        ])
        
        # Autocorrelation (5)
        names.extend([
            'autocorr_lag1',
            'autocorr_lag_n',
            'autocorr_momentum_persistence',
            'autocorr_mean_reversion',
            'autocorr_predictability'
        ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of discovered patterns."""
        if not self.is_fitted_:
            return "Transformer not fitted yet."
        
        interpretation = f"""
Temporal Derivative Analysis

Extracted velocity, acceleration, and momentum features WITHOUT
assuming which temporal patterns matter.

Features Extracted: {self.metadata['n_features']}
Sequences Analyzed: {self.metadata.get('n_samples', 'Unknown')}
Recent Window: {self.metadata.get('recent_window', 'Unknown')} periods

Feature Categories:
1. Velocity (12): Rate of change, direction, consistency
2. Acceleration (10): Rate of velocity change, momentum building
3. Trend Consistency (8): Linearity, volatility, smoothness
4. Regime Shifts (5): Changepoints, stability
5. Autocorrelation (5): Momentum persistence, predictability

These features enable DISCOVERY of temporal patterns that predict outcomes:
- Does recent acceleration matter more than current state?
- Does velocity consistency predict success?
- Do regime shifts signal turning points?
- Does momentum persist or mean-revert?

The learning system discovers which dynamics matter without assumptions.
"""
        return interpretation.strip()

