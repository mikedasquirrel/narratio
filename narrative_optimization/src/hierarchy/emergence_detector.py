"""
Emergence Detector

Detects when higher-level narratives emerge from lower-level stories.

Key questions:
- After how many games does a series narrative crystallize?
- When does season narrative become clear?
- What triggers narrative emergence?
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats


class EmergenceDetector:
    """
    Detects emergence of higher-level narratives from component stories.
    
    Emergence = when pattern becomes clear enough to predict from
    
    Example: After 2 games in a series, can we predict series outcome?
             Emergence score tells us if series narrative exists yet.
    """
    
    def __init__(self):
        # Emergence thresholds (to be discovered)
        self.emergence_thresholds = {
            'series': 2.5,      # Series after ~3 games
            'season': 18.0,     # Season after ~18 games
            'era': 150.0        # Era after ~150 games
        }
        
        # Emergence steepness (how fast narrative locks in)
        self.emergence_steepness = {
            'series': 0.8,      # Steep (series narrative forms quickly)
            'season': 0.3,      # Moderate (gradual season narrative)
            'era': 0.1          # Slow (era takes time to establish)
        }
    
    def detect_emergence(
        self,
        component_narratives: List[Dict],
        level: str
    ) -> Dict[str, Any]:
        """
        Detect if higher-level narrative has emerged.
        
        Parameters
        ----------
        component_narratives : list
            Lower-level stories (e.g., games in a series)
        level : str
            Target level (series, season, era)
        
        Returns
        -------
        emergence_data : dict
            Score, status, confidence, trigger points
        """
        n_components = len(component_narratives)
        
        if n_components == 0:
            return {
                'score': 0.0,
                'status': 'not_started',
                'confidence': 0.0,
                'n_components': 0
            }
        
        # Get threshold and steepness for this level
        threshold = self.emergence_thresholds.get(level, 10.0)
        steepness = self.emergence_steepness.get(level, 0.5)
        
        # Sigmoid emergence function
        # score = 1 / (1 + e^(-steepness * (n - threshold)))
        score = 1 / (1 + np.exp(-steepness * (n_components - threshold)))
        
        # Classify status
        if score < 0.2:
            status = 'forming'
        elif score < 0.5:
            status = 'emerging'
        elif score < 0.8:
            status = 'establishing'
        else:
            status = 'emerged'
        
        # Confidence in using higher-level narrative
        confidence = score
        
        # Identify trigger points
        triggers = self._identify_triggers(component_narratives, level)
        
        return {
            'score': float(score),
            'status': status,
            'confidence': float(confidence),
            'n_components': n_components,
            'threshold': threshold,
            'triggers': triggers,
            'level': level
        }
    
    def _identify_triggers(
        self,
        narratives: List[Dict],
        level: str
    ) -> List[str]:
        """Identify what triggered narrative emergence."""
        triggers = []
        
        if not narratives:
            return triggers
        
        n = len(narratives)
        
        # Quantity trigger
        if n >= self.emergence_thresholds.get(level, 10):
            triggers.append('Sufficient games played')
        
        # Quality trigger (high-stakes games accelerate emergence)
        high_stakes_count = sum(
            1 for n in narratives 
            if n.get('context', {}).get('weight', 1.0) >= 1.5
        )
        
        if high_stakes_count >= 2:
            triggers.append('Multiple high-stakes games')
        
        # Pattern trigger (consistent narrative thread)
        if self._has_consistent_pattern(narratives):
            triggers.append('Consistent narrative pattern')
        
        # Climax trigger (decisive moment)
        if self._has_climax_moment(narratives):
            triggers.append('Climax moment occurred')
        
        return triggers
    
    def _has_consistent_pattern(self, narratives: List[Dict]) -> bool:
        """Check if narratives show consistent pattern."""
        if len(narratives) < 3:
            return False
        
        # Check momentum consistency
        momenta = [n.get('features', {}).get('momentum', 0) for n in narratives]
        
        # Low variance = consistent
        if len(momenta) >= 3 and np.std(momenta) < 0.15:
            return True
        
        return False
    
    def _has_climax_moment(self, narratives: List[Dict]) -> bool:
        """Check if there was a decisive climactic moment."""
        # Look for game with very high context weight
        for narrative in narratives:
            if narrative.get('context', {}).get('weight', 1.0) >= 2.5:
                return True
        
        return False
    
    def calculate_optimal_decay(
        self,
        narratives_by_level: Dict[str, List[Dict]],
        outcomes: Dict[str, List[int]]
    ) -> Dict[str, float]:
        """
        Discover optimal decay rates empirically.
        
        Tests different γ values to maximize prediction accuracy.
        """
        optimal_decays = {}
        
        for level in ['game', 'series', 'season', 'era']:
            if level not in narratives_by_level:
                continue
            
            best_gamma = 0.95
            best_accuracy = 0
            
            # Grid search
            for gamma in np.linspace(0.85, 0.99, 15):
                # Test this decay rate
                self.decay_rates[level] = gamma
                
                # Accumulate with this gamma
                # (Would need to run predictions and measure)
                # For now, use heuristic
                
                # Hypothesis: Optimal decay increases with level
                # (Longer timescales = slower decay)
                pass
            
            optimal_decays[level] = best_gamma
        
        return optimal_decays
    
    def detect_narrative_shift(
        self,
        narratives: List[Dict],
        window: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Detect when narrative shifts/reverses (turning points).
        
        Returns list of shift events with timestamps and descriptions.
        """
        shifts = []
        
        if len(narratives) < window:
            return shifts
        
        # Sliding window to detect momentum shifts
        for i in range(window, len(narratives)):
            before = narratives[i-window:i]
            current = narratives[i]
            
            # Calculate momentum before and after
            momentum_before = np.mean([
                n.get('features', {}).get('momentum', 0) 
                for n in before
            ])
            
            momentum_current = current.get('features', {}).get('momentum', 0)
            
            # Detect significant shift
            if abs(momentum_current - momentum_before) > 0.4:
                shift_type = 'Reversal' if momentum_current * momentum_before < 0 else 'Amplification'
                
                shifts.append({
                    'index': i,
                    'game_id': current.get('id'),
                    'type': shift_type,
                    'magnitude': float(abs(momentum_current - momentum_before)),
                    'before': float(momentum_before),
                    'after': float(momentum_current),
                    'timestamp': current.get('date')
                })
        
        return shifts
    
    def predict_emergence_point(
        self,
        current_components: List[Dict],
        level: str
    ) -> Dict[str, Any]:
        """
        Predict when narrative will emerge based on current trajectory.
        
        Returns estimated components needed for emergence.
        """
        current_score = self.detect_emergence(current_components, level)['score']
        threshold = self.emergence_thresholds[level]
        n_current = len(current_components)
        
        if current_score >= 0.8:
            return {
                'status': 'already_emerged',
                'current_components': n_current,
                'components_needed': 0
            }
        
        # Estimate components needed for emergence (score > 0.8)
        # Using sigmoid: solve for n when score = 0.8
        steepness = self.emergence_steepness[level]
        
        # 0.8 = 1/(1 + e^(-s*(n-t)))
        # Solve for n: n = t + ln(4)/s
        components_for_emergence = threshold + np.log(4) / steepness
        components_needed = max(0, int(np.ceil(components_for_emergence - n_current)))
        
        return {
            'status': 'emerging',
            'current_components': n_current,
            'current_score': current_score,
            'components_needed': components_needed,
            'estimated_total': int(components_for_emergence)
        }


def create_emergence_detector():
    """Factory function."""
    return EmergenceDetector()

