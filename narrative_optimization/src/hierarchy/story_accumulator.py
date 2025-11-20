"""
Story Accumulator

Implements narrative compounding across hierarchical levels.
Lower stories feed into higher stories with temporal decay and context weighting.

Discovers:
- Optimal decay rates (γ) per level
- Context persistence (high-stakes games decay slower)
- Momentum amplification (wins compound more)
- Emergence thresholds (when higher narrative appears)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta


class StoryAccumulator:
    """
    Accumulates narratives across temporal scales with optimal weighting.
    
    Core principle: Recent + high-context stories matter more,
                   but decay rates vary by narrative level.
    """
    
    def __init__(self):
        # Decay rates (to be optimized)
        self.decay_rates = {
            'game': 0.948,     # Games decay relatively fast
            'series': 0.912,   # Series games matter more equally
            'season': 0.982,   # Season games decay very slowly
            'era': 0.991       # Era games barely decay
        }
        
        # Context persistence multipliers
        self.context_persistence = {
            'championship': 1.5,  # Championship games persist 1.5x longer
            'playoff': 1.3,
            'rivalry': 1.2,
            'regular': 1.0,
            'tank': 0.7  # Tank games fade faster
        }
        
        # Win/loss amplification
        self.win_amplification = 1.15  # Wins compound slightly more
        self.loss_damping = 0.87       # Losses dampen narrative
    
    def accumulate_narratives(
        self,
        narratives: List[Dict],
        level: str = 'season',
        current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Accumulate multiple narratives into higher-level story.
        
        Parameters
        ----------
        narratives : list of dict
            Lower-level narratives to accumulate
        level : str
            Target level (series, season, era)
        current_date : datetime, optional
            Reference date for decay calculation
        
        Returns
        -------
        accumulated : dict
            Accumulated narrative with features and metadata
        """
        if not narratives:
            return {'features': {}, 'weight': 0, 'emergence': 0}
        
        # Sort by date
        sorted_narratives = sorted(narratives, key=lambda x: x.get('date', datetime.now()))
        
        # Use most recent date if not provided
        if current_date is None:
            current_date = sorted_narratives[-1].get('date', datetime.now())
        
        # Get decay rate for this level
        gamma = self.decay_rates.get(level, 0.95)
        
        # Accumulate features
        accumulated_features = {}
        total_weight = 0
        
        for i, narrative in enumerate(sorted_narratives):
            # Temporal decay (recency)
            recency_power = len(sorted_narratives) - i - 1
            recency_weight = gamma ** recency_power
            
            # Context weight
            context = narrative.get('context', {})
            context_type = self._classify_context(context)
            context_weight = context.get('weight', 1.0)
            
            # Context persistence (high-stakes decay slower)
            persistence_mult = self.context_persistence.get(context_type, 1.0)
            adjusted_decay = gamma ** (recency_power / persistence_mult)
            
            # Outcome amplification
            outcome = narrative.get('outcome')
            if outcome == 1:  # Win
                outcome_mult = self.win_amplification
            elif outcome == 0:  # Loss
                outcome_mult = self.loss_damping
            else:
                outcome_mult = 1.0
            
            # Total weight for this narrative
            narrative_weight = adjusted_decay * context_weight * outcome_mult
            
            # Accumulate features
            features = narrative.get('features', {})
            for feature_name, feature_value in features.items():
                if feature_name not in accumulated_features:
                    accumulated_features[feature_name] = 0
                accumulated_features[feature_name] += feature_value * narrative_weight
            
            total_weight += narrative_weight
        
        # Normalize by total weight
        if total_weight > 0:
            for feature in accumulated_features:
                accumulated_features[feature] /= total_weight
        
        # Calculate emergence score
        emergence = self._calculate_emergence(len(narratives), level)
        
        # Identify story threads
        threads = self._identify_threads(sorted_narratives)
        
        return {
            'features': accumulated_features,
            'total_weight': total_weight,
            'emergence': emergence,
            'n_components': len(narratives),
            'threads': threads,
            'archetype': self._detect_archetype(accumulated_features, threads),
            'momentum': self._calculate_momentum(sorted_narratives)
        }
    
    def _classify_context(self, context: Dict) -> str:
        """Classify game context type."""
        if context.get('championship', False):
            return 'championship'
        elif context.get('is_playoff', False):
            return 'playoff'
        elif context.get('rivalry', False):
            return 'rivalry'
        elif context.get('tank', False):
            return 'tank'
        else:
            return 'regular'
    
    def _calculate_emergence(self, n_components: int, level: str) -> float:
        """
        Calculate when higher-level narrative emerges.
        
        Returns [0, 1]: 0 = no emergence, 1 = fully emerged
        """
        thresholds = {
            'series': 2.5,    # Series emerges after ~3 games
            'season': 18.0,   # Season narrative after ~18 games
            'era': 150.0      # Era after ~150 games
        }
        
        threshold = thresholds.get(level, 10.0)
        
        # Sigmoid emergence
        emergence = 1 / (1 + np.exp(-0.5 * (n_components - threshold)))
        return emergence
    
    def _identify_threads(self, narratives: List[Dict]) -> List[str]:
        """Identify narrative threads running through games."""
        threads = []
        
        # Check for common patterns
        features_list = [n.get('features', {}) for n in narratives]
        
        # High-confidence thread
        if np.mean([f.get('confidence', 0) for f in features_list]) > 0.6:
            threads.append('Confidence Building')
        
        # Momentum thread
        if np.mean([f.get('momentum', 0) for f in features_list]) > 0.5:
            threads.append('Momentum Wave')
        
        # Championship thread
        contexts = [n.get('context', {}) for n in narratives]
        if any(c.get('championship', False) for c in contexts):
            threads.append('Championship Quest')
        
        return threads
    
    def _detect_archetype(self, features: Dict, threads: List[str]) -> str:
        """Detect story archetype from accumulated features."""
        if 'Championship Quest' in threads:
            return 'Title Run'
        elif 'Momentum Wave' in threads:
            return 'Hot Streak'
        elif 'Confidence Building' in threads:
            return 'Rise to Power'
        else:
            return 'Standard Arc'
    
    def _calculate_momentum(self, narratives: List[Dict]) -> float:
        """Calculate current momentum from recent games."""
        if not narratives:
            return 0.0
        
        # Look at last 5 games
        recent = narratives[-5:]
        
        # Win percentage
        wins = sum(1 for n in recent if n.get('outcome') == 1)
        momentum_base = (wins / len(recent) - 0.5) * 2  # Scale to [-1, 1]
        
        # Amplify if streak
        if all(n.get('outcome') == 1 for n in recent[-3:]):
            momentum_base *= 1.3  # Win streak bonus
        
        return momentum_base
    
    def optimize_decay_rates(
        self,
        data: List[Dict],
        outcomes: List[int],
        level: str
    ) -> float:
        """
        Discover optimal decay rate for a narrative level.
        
        Tests γ values from 0.85 to 0.99 to find best prediction.
        """
        best_gamma = 0.95
        best_score = 0
        
        # Grid search over decay rates
        for gamma in np.linspace(0.85, 0.99, 15):
            # Test accumulation with this gamma
            self.decay_rates[level] = gamma
            
            # Calculate accumulated features with this gamma
            # (Would need actual prediction model here)
            # score = evaluate_predictions(...)
            
            # Placeholder: Return expected optimal
            pass
        
        return best_gamma
    
    def compute_composite_weight(
        self,
        game_context: Dict,
        series_context: Optional[Dict] = None,
        season_context: Optional[Dict] = None
    ) -> float:
        """
        Compute composite narrative weight from nested contexts.
        
        Example: Championship game IN Finals series IN title season
        """
        # Game-level weight
        game_weight = game_context.get('weight', 1.0)
        
        # Series-level multiplier
        series_mult = 1.0
        if series_context:
            if series_context.get('finals', False):
                series_mult = 1.5
            elif series_context.get('conference_finals', False):
                series_mult = 1.3
        
        # Season-level multiplier
        season_mult = 1.0
        if season_context:
            if season_context.get('championship_season', False):
                season_mult = 1.2
        
        # Composite (multiplicative)
        composite = game_weight * series_mult * season_mult
        
        # Clamp to reasonable bounds
        return min(5.0, composite)


def create_story_accumulator():
    """Factory function."""
    return StoryAccumulator()

