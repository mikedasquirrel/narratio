"""
Online Learning & Streaming

Real-time pattern updates from streaming data.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Deque
from collections import deque, defaultdict
from datetime import datetime


class OnlineLearner:
    """
    Online learning for streaming data.
    
    Features:
    - Incremental updates (no retraining)
    - Concept drift detection
    - Adaptive forgetting (outdated patterns fade)
    - Low-latency inference
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        forgetting_rate: float = 0.01
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.forgetting_rate = forgetting_rate
        
        # Sliding window
        self.window: Deque = deque(maxlen=window_size)
        
        # Pattern statistics
        self.pattern_counts = defaultdict(int)
        self.pattern_outcomes = defaultdict(list)
        self.pattern_timestamps = defaultdict(list)
        
        # Drift detection
        self.baseline_distribution = None
        self.drift_detected = False
        
    def update(self, text: str, outcome: float, timestamp: Optional[float] = None):
        """
        Incremental update with new sample.
        
        Parameters
        ----------
        text : str
            New text
        outcome : float
            Outcome
        timestamp : float, optional
            Timestamp
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        # Add to window
        self.window.append({
            'text': text,
            'outcome': outcome,
            'timestamp': timestamp
        })
        
        # Extract patterns from new sample
        patterns = self._extract_patterns(text)
        
        # Update statistics
        for pattern in patterns:
            self.pattern_counts[pattern] += 1
            self.pattern_outcomes[pattern].append(outcome)
            self.pattern_timestamps[pattern].append(timestamp)
        
        # Apply forgetting to old patterns
        self._apply_forgetting(timestamp)
        
        # Check for drift
        if len(self.window) >= self.window_size // 2:
            self.drift_detected = self._detect_drift()
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract patterns from text."""
        words = text.lower().split()
        
        # Bigrams
        patterns = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        
        # Keywords
        keywords = ['underdog', 'comeback', 'pressure', 'dominant', 'rivalry']
        patterns.extend([kw for kw in keywords if kw in text.lower()])
        
        return patterns
    
    def _apply_forgetting(self, current_time: float):
        """
        Apply exponential forgetting to old patterns.
        
        Parameters
        ----------
        current_time : float
            Current timestamp
        """
        for pattern in list(self.pattern_counts.keys()):
            timestamps = self.pattern_timestamps[pattern]
            
            if len(timestamps) == 0:
                continue
            
            # Time since last occurrence
            time_since_last = current_time - timestamps[-1]
            
            # Exponential decay
            decay_factor = np.exp(-self.forgetting_rate * time_since_last)
            
            # Reduce count
            old_count = self.pattern_counts[pattern]
            new_count = old_count * decay_factor
            
            if new_count < 1.0:
                # Remove pattern
                del self.pattern_counts[pattern]
                del self.pattern_outcomes[pattern]
                del self.pattern_timestamps[pattern]
            else:
                self.pattern_counts[pattern] = new_count
    
    def _detect_drift(self) -> bool:
        """
        Detect concept drift in data distribution.
        
        Returns
        -------
        bool
            True if drift detected
        """
        if len(self.window) < 100:
            return False
        
        # Split window into old and new
        split_point = len(self.window) // 2
        old_data = list(self.window)[:split_point]
        new_data = list(self.window)[split_point:]
        
        # Compare outcome distributions
        old_outcomes = np.array([d['outcome'] for d in old_data])
        new_outcomes = np.array([d['outcome'] for d in new_data])
        
        # KS test for distribution change
        from scipy import stats
        statistic, p_value = stats.ks_2samp(old_outcomes, new_outcomes)
        
        return p_value < 0.05  # Significant difference
    
    def predict(self, text: str) -> float:
        """
        Fast online prediction.
        
        Parameters
        ----------
        text : str
            Text to predict
        
        Returns
        -------
        float
            Prediction
        """
        patterns = self._extract_patterns(text)
        
        if len(patterns) == 0:
            return 0.5  # Default
        
        scores = []
        
        for pattern in patterns:
            if pattern in self.pattern_outcomes:
                outcomes = self.pattern_outcomes[pattern]
                
                if len(outcomes) > 0:
                    # Recent performance
                    score = np.mean(outcomes[-10:])  # Last 10
                    scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def get_active_patterns(self, min_count: float = 5.0) -> Dict[str, Dict]:
        """
        Get currently active patterns.
        
        Parameters
        ----------
        min_count : float
            Minimum count (after decay)
        
        Returns
        -------
        dict
            Active patterns with statistics
        """
        active = {}
        
        for pattern, count in self.pattern_counts.items():
            if count >= min_count:
                outcomes = self.pattern_outcomes[pattern]
                
                active[pattern] = {
                    'count': count,
                    'win_rate': np.mean(outcomes) if outcomes else 0.5,
                    'recency': self.pattern_timestamps[pattern][-1] if self.pattern_timestamps[pattern] else 0,
                    'stability': np.std(outcomes[-20:]) if len(outcomes) >= 20 else 0.5
                }
        
        return active
    
    def reset_after_drift(self):
        """Reset statistics after drift detected."""
        self.pattern_counts = defaultdict(int)
        self.pattern_outcomes = defaultdict(list)
        self.pattern_timestamps = defaultdict(list)
        self.drift_detected = False
        
        print("Drift detected - statistics reset")
    
    def get_statistics(self) -> Dict:
        """Get online learning statistics."""
        return {
            'window_size': len(self.window),
            'n_patterns': len(self.pattern_counts),
            'drift_detected': self.drift_detected,
            'avg_outcomes': np.mean([d['outcome'] for d in self.window]) if self.window else 0.5
        }
    
    def batch_update(self, texts: List[str], outcomes: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """
        Batch update (for initialization or catching up).
        
        Parameters
        ----------
        texts : list
            Batch of texts
        outcomes : ndarray
            Batch of outcomes
        timestamps : ndarray, optional
            Timestamps
        """
        if timestamps is None:
            timestamps = np.arange(len(texts))
        
        for text, outcome, timestamp in zip(texts, outcomes, timestamps):
            self.update(text, outcome, timestamp)

