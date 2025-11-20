"""
Temporal Momentum Transformer (Enhanced Universal Version)

Universal temporal momentum analysis for ALL domains with time progression.
Handles streaks, trends, acceleration, cycles, peaks.

Used by: All sports, crypto, stocks, music charts, social media, startups, careers.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
from typing import List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.shared_models import SharedModelRegistry
from .utils.input_validation import ensure_string_list


class TemporalMomentumEnhancedTransformer(BaseEstimator, TransformerMixin):
    """
    Enhanced universal temporal momentum analysis.
    
    Works with:
    - Text (temporal language analysis)
    - Time-series data (when provided)
    - Historical records (streaks, trends)
    - Mixed data (text + metrics over time)
    
    Features (10 total):
    1. Streak analysis (winning/losing streaks)
    2. Trend direction (improving/declining/stable)
    3. Acceleration (rate of change increasing/decreasing)
    4. Cyclical patterns (seasonal, periodic)
    5. Peak detection (at peak/past peak/pre-peak)
    6. Decline detection (declining phase)
    7. Breakout identification (emerging/breakout)
    8. Consistency over time (volatile vs steady)
    9. Volatility measure (how much variation)
    10. Momentum sustainability (can it continue)
    
    Input Options:
    - Text only: Extracts temporal language
    - Dict with 'history' or 'recent_results': Analyzes time series
    - Dict with temporal metrics: Analyzes momentum from metrics
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize temporal momentum analyzer"""
        self.use_spacy = use_spacy
        self.use_embeddings = use_embeddings
        
        self.nlp = None
        self.embedder = None
    
    def fit(self, X, y=None):
        """Fit transformer (load shared models)"""
        X = ensure_string_list(X)
        
        # Load shared models
        if self.use_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        
        if self.use_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
        
        return self
    
    def transform(self, X):
        """
        Transform to temporal momentum features.
        
        Parameters
        ----------
        X : array-like of strings or dicts
            Text or data with temporal information
            
        Returns
        -------
        features : ndarray of shape (n_samples, 10)
            Temporal momentum features
        """
        features = []
        
        for item in X:
            feat = self._extract_temporal_features(item)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_temporal_features(self, item: Union[str, Dict]) -> List[float]:
        """Extract all temporal momentum features"""
        features = []
        
        # Determine if we have time-series data or just text
        has_time_series = isinstance(item, dict) and ('history' in item or 'recent_results' in item or 'time_series' in item)
        
        if has_time_series:
            # Analyze actual time-series data
            features = self._analyze_time_series(item)
        else:
            # Analyze temporal language
            text = item if isinstance(item, str) else str(item.get('text', item.get('narrative', str(item))))
            features = self._analyze_temporal_language(text)
        
        return features
    
    def _analyze_time_series(self, data: Dict) -> List[float]:
        """Analyze actual time-series data for momentum"""
        # Get time series
        time_series = data.get('history', data.get('recent_results', data.get('time_series', [])))
        
        if not time_series or len(time_series) < 2:
            return [0.5] * 10  # No data, return neutral
        
        # Convert to numpy array
        if isinstance(time_series[0], dict):
            # Extract values
            values = [item.get('value', item.get('result', item.get('score', 0))) for item in time_series]
        else:
            values = time_series
        
        values = np.array(values, dtype=float)
        
        features = []
        
        # 1. Streak analysis
        streak = self._compute_streak(values)
        features.append(streak)
        
        # 2. Trend direction
        trend = self._compute_trend(values)
        features.append(trend)
        
        # 3. Acceleration
        acceleration = self._compute_acceleration(values)
        features.append(acceleration)
        
        # 4. Cyclical patterns
        cyclical = self._detect_cyclical(values)
        features.append(cyclical)
        
        # 5. Peak detection
        at_peak = self._detect_peak(values)
        features.append(at_peak)
        
        # 6. Decline detection
        declining = self._detect_decline(values)
        features.append(declining)
        
        # 7. Breakout identification
        breakout = self._detect_breakout(values)
        features.append(breakout)
        
        # 8. Consistency
        consistency = self._compute_consistency(values)
        features.append(consistency)
        
        # 9. Volatility
        volatility = self._compute_volatility(values)
        features.append(volatility)
        
        # 10. Sustainability
        sustainability = self._compute_sustainability(values)
        features.append(sustainability)
        
        return features
    
    def _analyze_temporal_language(self, text: str) -> List[float]:
        """Analyze temporal language when no time-series data"""
        features = []
        
        if self.nlp:
            doc = self.nlp(text[:5000])
            n_words = len(doc)
        else:
            doc = None
            n_words = len(text.split()) + 1
        
        # 1. Streak language
        if doc:
            streak_lemmas = {'streak', 'consecutive', 'row', 'straight', 'unbeaten', 'undefeated'}
            streak_count = sum(1 for token in doc if token.lemma_ in streak_lemmas)
            features.append(min(1.0, streak_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 2. Trend direction from language
        if doc:
            improving_lemmas = {'improve', 'rise', 'grow', 'increase', 'ascend', 'advance', 'progress'}
            declining_lemmas = {'decline', 'fall', 'decrease', 'descend', 'regress', 'worsen'}
            
            improving = sum(1 for token in doc if token.lemma_ in improving_lemmas)
            declining = sum(1 for token in doc if token.lemma_ in declining_lemmas)
            
            total = improving + declining
            if total > 0:
                features.append(improving / total)  # 1 = improving, 0 = declining
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # 3. Acceleration language
        if doc:
            accel_lemmas = {'accelerate', 'rapidly', 'quickly', 'suddenly', 'explosive', 'surge'}
            accel_count = sum(1 for token in doc if token.lemma_ in accel_lemmas)
            features.append(min(1.0, accel_count / n_words * 10))
        else:
            features.append(0.2)
        
        # 4. Cyclical language
        if doc:
            cyclical_lemmas = {'cycle', 'season', 'pattern', 'recurring', 'periodic', 'regular'}
            cyclical_count = sum(1 for token in doc if token.lemma_ in cyclical_lemmas)
            features.append(min(1.0, cyclical_count / n_words * 10))
        else:
            features.append(0.2)
        
        # 5. Peak language
        if doc:
            peak_lemmas = {'peak', 'prime', 'best', 'top', 'highest', 'zenith', 'apex'}
            peak_count = sum(1 for token in doc if token.lemma_ in peak_lemmas)
            features.append(min(1.0, peak_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 6. Decline language
        if doc:
            decline_lemmas = {'decline', 'fading', 'waning', 'diminish', 'deteriorate', 'past prime'}
            decline_count = sum(1 for token in doc if token.lemma_ in decline_lemmas)
            features.append(min(1.0, decline_count / n_words * 10))
        else:
            features.append(0.2)
        
        # 7. Breakout/emerging language
        if doc:
            breakout_lemmas = {'breakout', 'emerging', 'rising', 'breakthrough', 'bursting', 'explode'}
            breakout_count = sum(1 for token in doc if token.lemma_ in breakout_lemmas)
            features.append(min(1.0, breakout_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 8. Consistency language
        if doc:
            consistent_lemmas = {'consistent', 'steady', 'reliable', 'stable', 'dependable', 'regular'}
            consistent_count = sum(1 for token in doc if token.lemma_ in consistent_lemmas)
            features.append(min(1.0, consistent_count / n_words * 10))
        else:
            features.append(0.5)
        
        # 9. Volatility language
        if doc:
            volatile_lemmas = {'volatile', 'unpredictable', 'erratic', 'inconsistent', 'variable', 'fluctuate'}
            volatile_count = sum(1 for token in doc if token.lemma_ in volatile_lemmas)
            features.append(min(1.0, volatile_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 10. Sustainability language
        if doc:
            sustainable_lemmas = {'sustainable', 'maintain', 'continue', 'endure', 'lasting', 'durable'}
            sustainable_count = sum(1 for token in doc if token.lemma_ in sustainable_lemmas)
            features.append(min(1.0, sustainable_count / n_words * 10))
        else:
            features.append(0.4)
        
        return features
    
    def _compute_streak(self, values: np.ndarray) -> float:
        """Compute current streak (positive or negative)"""
        if len(values) < 2:
            return 0.5
        
        # Binary values (wins/losses)
        if all(v in [0, 1] for v in values):
            current_streak = 0
            current_value = values[-1]
            
            for v in reversed(values):
                if v == current_value:
                    current_streak += 1
                else:
                    break
            
            # Normalize (5+ streak = max)
            normalized = min(1.0, current_streak / 5.0)
            # If losing streak, invert
            return normalized if current_value == 1 else (1.0 - normalized)
        else:
            # Continuous values - momentum direction
            recent_avg = np.mean(values[-5:])
            overall_avg = np.mean(values)
            return (recent_avg - overall_avg) / (np.std(values) + 0.1)
    
    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute trend direction (0 = declining, 0.5 = stable, 1 = improving)"""
        if len(values) < 3:
            return 0.5
        
        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize to 0-1
        return float(np.clip((slope + 1) / 2, 0, 1))
    
    def _compute_acceleration(self, values: np.ndarray) -> float:
        """Compute acceleration (rate of change of rate of change)"""
        if len(values) < 4:
            return 0.5
        
        # First derivative (velocity)
        velocity = np.diff(values)
        
        # Second derivative (acceleration)
        acceleration = np.diff(velocity)
        
        # Recent acceleration
        recent_accel = np.mean(acceleration[-3:]) if len(acceleration) >= 3 else 0
        
        return float(np.clip((recent_accel + 1) / 2, 0, 1))
    
    def _detect_cyclical(self, values: np.ndarray) -> float:
        """Detect cyclical patterns"""
        if len(values) < 8:
            return 0.0
        
        # Simple autocorrelation at lag 1 and lag len//2
        autocorr_1 = np.corrcoef(values[:-1], values[1:])[0, 1] if len(values) > 1 else 0
        
        # Cyclical = moderate autocorrelation (not too high, not too low)
        cyclical_score = 1.0 - abs(autocorr_1 - 0.5)
        
        return float(np.clip(cyclical_score, 0, 1))
    
    def _detect_peak(self, values: np.ndarray) -> float:
        """Detect if at peak performance"""
        if len(values) < 3:
            return 0.5
        
        # Is current value highest or near highest?
        current = values[-1]
        max_val = np.max(values)
        
        if max_val == 0:
            return 0.5
        
        # Normalize
        at_peak_score = current / max_val
        
        return float(at_peak_score)
    
    def _detect_decline(self, values: np.ndarray) -> float:
        """Detect if in decline phase"""
        if len(values) < 5:
            return 0.0
        
        # Compare recent to historical peak
        recent_avg = np.mean(values[-5:])
        historical_peak = np.max(values[:-5]) if len(values) > 5 else np.max(values)
        
        if historical_peak == 0:
            return 0.0
        
        decline_score = 1.0 - (recent_avg / historical_peak)
        
        return float(np.clip(decline_score, 0, 1))
    
    def _detect_breakout(self, values: np.ndarray) -> float:
        """Detect breakout/emergence"""
        if len(values) < 5:
            return 0.0
        
        # Recent performance dramatically better than historical
        recent_avg = np.mean(values[-3:])
        historical_avg = np.mean(values[:-3])
        
        if historical_avg == 0:
            return 0.5
        
        breakout_score = (recent_avg - historical_avg) / (historical_avg + 0.1)
        
        return float(np.clip(breakout_score, 0, 1))
    
    def _compute_consistency(self, values: np.ndarray) -> float:
        """Compute consistency (inverse of volatility)"""
        if len(values) < 2:
            return 1.0
        
        # Coefficient of variation
        mean = np.mean(values)
        std = np.std(values)
        
        if mean == 0:
            return 0.5
        
        cv = std / abs(mean)
        consistency = 1 / (1 + cv)
        
        return float(consistency)
    
    def _compute_volatility(self, values: np.ndarray) -> float:
        """Compute volatility"""
        if len(values) < 2:
            return 0.0
        
        # Standard deviation normalized
        std = np.std(values)
        mean = np.mean(values)
        
        if mean == 0:
            return float(np.clip(std, 0, 1))
        
        volatility = std / abs(mean)
        
        return float(np.clip(volatility, 0, 1))
    
    def _compute_sustainability(self, values: np.ndarray) -> float:
        """Compute momentum sustainability"""
        if len(values) < 5:
            return 0.5
        
        # Check if trend is consistent (not just lucky spike)
        # Use linear regression R²
        x = np.arange(len(values))
        
        # Fit linear model
        coeffs = np.polyfit(x, values, 1)
        trend_line = np.polyval(coeffs, x)
        
        # R² (how well trend explains data)
        ss_res = np.sum((values - trend_line) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        
        if ss_tot == 0:
            return 0.5
        
        r_squared = 1 - (ss_res / ss_tot)
        
        return float(np.clip(r_squared, 0, 1))
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'temporal_streak',
            'temporal_trend_direction',
            'temporal_acceleration',
            'temporal_cyclical_pattern',
            'temporal_at_peak',
            'temporal_decline_phase',
            'temporal_breakout',
            'temporal_consistency',
            'temporal_volatility',
            'temporal_sustainability'
        ])

