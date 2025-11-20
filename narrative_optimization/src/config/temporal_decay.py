"""
Temporal Decay Functions

Various decay functions for weighting historical narratives by recency.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import Callable, Optional
from enum import Enum


class DecayType(Enum):
    """Types of temporal decay functions."""
    EXPONENTIAL = 'exponential'
    LINEAR = 'linear'
    POWER = 'power'
    LOGARITHMIC = 'logarithmic'
    CUSTOM = 'custom'


class TemporalDecay:
    """
    Temporal decay functions for historial weighting.
    
    More recent narratives get higher weight.
    """
    
    @staticmethod
    def exponential_decay(
        timestamps: np.ndarray,
        half_life: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Exponential decay: weight = exp(-(max_time - t) / half_life)
        
        Parameters
        ----------
        timestamps : ndarray
            Timestamps (higher = more recent)
        half_life : float, optional
            Half-life in time units (if None, uses 1/3 of time range)
        max_time : float, optional
            Maximum timestamp (if None, uses max(timestamps))
        
        Returns
        -------
        ndarray
            Decay weights (0-1, higher = more recent)
        """
        if max_time is None:
            max_time = timestamps.max()
        
        if half_life is None:
            time_range = max_time - timestamps.min()
            half_life = time_range / 3.0 if time_range > 0 else 1.0
        
        if half_life <= 0:
            return np.ones(len(timestamps))
        
        decay = np.exp(-(max_time - timestamps) / half_life)
        return decay / decay.max()  # Normalize to [0, 1]
    
    @staticmethod
    def linear_decay(
        timestamps: np.ndarray,
        max_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Linear decay: weight = (t - min) / (max - min)
        
        Parameters
        ----------
        timestamps : ndarray
            Timestamps
        max_time : float, optional
            Maximum timestamp
        
        Returns
        -------
        ndarray
            Linear weights
        """
        if max_time is None:
            max_time = timestamps.max()
        
        min_time = timestamps.min()
        
        if max_time == min_time:
            return np.ones(len(timestamps))
        
        weights = (timestamps - min_time) / (max_time - min_time)
        return weights
    
    @staticmethod
    def power_decay(
        timestamps: np.ndarray,
        power: float = 2.0,
        max_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Power decay: weight = ((t - min) / (max - min))^power
        
        Parameters
        ----------
        timestamps : ndarray
            Timestamps
        power : float
            Power exponent (higher = steeper decay)
        max_time : float, optional
            Maximum timestamp
        
        Returns
        -------
        ndarray
            Power decay weights
        """
        linear_weights = TemporalDecay.linear_decay(timestamps, max_time)
        return np.power(linear_weights, power)
    
    @staticmethod
    def logarithmic_decay(
        timestamps: np.ndarray,
        max_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Logarithmic decay: weight = log(1 + (t - min) / scale)
        
        Parameters
        ----------
        timestamps : ndarray
            Timestamps
        max_time : float, optional
            Maximum timestamp
        
        Returns
        -------
        ndarray
            Logarithmic decay weights
        """
        if max_time is None:
            max_time = timestamps.max()
        
        min_time = timestamps.min()
        time_range = max_time - min_time
        
        if time_range == 0:
            return np.ones(len(timestamps))
        
        normalized = (timestamps - min_time) / time_range
        weights = np.log1p(normalized * 9) / np.log(10)  # Scale to [0, 1]
        return weights
    
    @staticmethod
    def custom_decay(
        timestamps: np.ndarray,
        decay_func: Callable[[np.ndarray], np.ndarray],
        max_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Custom decay function.
        
        Parameters
        ----------
        timestamps : ndarray
            Timestamps
        decay_func : callable
            Function that takes timestamps and returns weights
        max_time : float, optional
            Maximum timestamp
        
        Returns
        -------
        ndarray
            Custom decay weights
        """
        return decay_func(timestamps)
    
    @staticmethod
    def get_decay_function(decay_type: DecayType) -> Callable:
        """Get decay function by type."""
        functions = {
            DecayType.EXPONENTIAL: TemporalDecay.exponential_decay,
            DecayType.LINEAR: TemporalDecay.linear_decay,
            DecayType.POWER: TemporalDecay.power_decay,
            DecayType.LOGARITHMIC: TemporalDecay.logarithmic_decay
        }
        return functions.get(decay_type, TemporalDecay.exponential_decay)

