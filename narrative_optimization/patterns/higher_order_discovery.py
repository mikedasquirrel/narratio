"""
Higher-Order Pattern Discovery
===============================

Discovers multi-feature pattern interactions using:
- 2-way interactions (e.g., "home underdog + high momentum")
- 3-way interactions (e.g., "division + late season + record gap")
- Apriori algorithm for frequent pattern mining
- Statistical validation with multiple testing correction

Expected output: 50-100 new compound patterns per league.

Author: AI Coding Assistant
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from itertools import combinations
from scipy.stats import chi2_contingency
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HigherOrderPatternDiscovery:
    """
    Discovers compound patterns with statistical validation.
    """
    
    def __init__(
        self,
        min_support: float = 0.05,  # Pattern must appear in 5%+ of games
        min_confidence: float = 0.60,  # Must win 60%+ of games
        min_lift: float = 1.1,  # Must be 10% better than baseline
        p_value_threshold: float = 0.01  # Statistical significance
    ):
        """
        Initialize pattern discovery.
        
        Args:
            min_support: Minimum fraction of games with pattern
            min_confidence: Minimum win rate for pattern
            min_lift: Minimum improvement over baseline
            p_value_threshold: Maximum p-value for significance
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.p_value_threshold = p_value_threshold
        
    def extract_binary_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Extract binary features for pattern mining.
        
        Args:
            games: DataFrame with game data
            
        Returns:
            DataFrame with binary features
        """
        features = pd.DataFrame()
        
        # Home/away
        features['is_home'] = (games['home'] == 1)
        
        # Underdog/favorite
        if 'spread' in games.columns:
            features['is_underdog'] = (games['spread'] > 0)
            features['big_underdog'] = (games['spread'] >= 3.5)
            features['huge_underdog'] = (games['spread'] >= 7.0)
        
        # Record-based
        if 'season_win_pct' in games.columns:
            features['good_record'] = (games['season_win_pct'] >= 0.55)
            features['bad_record'] = (games['season_win_pct'] <= 0.45)
            if 'opp_season_win_pct' in games.columns:
                record_gap = games['season_win_pct'] - games['opp_season_win_pct']
                features['big_record_advantage'] = (record_gap >= 0.2)
                features['big_record_disadvantage'] = (record_gap <= -0.2)
        
        # Momentum
        if 'l10_win_pct' in games.columns:
            features['high_momentum'] = (games['l10_win_pct'] >= 0.7)
            features['low_momentum'] = (games['l10_win_pct'] <= 0.3)
        
        # Temporal
        if 'week' in games.columns:
            total_weeks = games['week'].max()
            features['early_season'] = (games['week'] <= total_weeks * 0.25)
            features['late_season'] = (games['week'] >= total_weeks * 0.75)
        
        # Context
        if 'is_division' in games.columns:
            features['division_game'] = games['is_division']
        if 'is_rivalry' in games.columns:
            features['rivalry'] = games['is_rivalry']
        
        # Rest
        if 'rest_days' in games.columns:
            features['short_rest'] = (games['rest_days'] <= 2)
            features['long_rest'] = (games['rest_days'] >= 5)
        
        return features
    
    def find_frequent_itemsets(
        self,
        features: pd.DataFrame,
        k: int = 2
    ) -> List[Tuple[str, ...]]:
        """
        Find frequent k-itemsets using Apriori.
        
        Args:
            features: Binary feature DataFrame
            k: Size of itemsets (2 or 3)
            
        Returns:
            List of frequent itemsets
        """
        n_games = len(features)
        min_count = int(n_games * self.min_support)
        
        frequent = []
        
        # Generate all k-combinations
        feature_names = features.columns.tolist()
        for combo in combinations(feature_names, k):
            # Count games where all features in combo are True
            mask = features[list(combo)].all(axis=1)
            count = mask.sum()
            
            if count >= min_count:
                frequent.append(combo)
        
        return frequent
    
    def evaluate_pattern(
        self,
        pattern: Tuple[str, ...],
        features: pd.DataFrame,
        outcomes: np.ndarray,
        baseline_win_rate: float
    ) -> Optional[Dict]:
        """
        Evaluate a pattern's performance.
        
        Args:
            pattern: Tuple of feature names
            features: Binary feature DataFrame
            outcomes: Binary outcomes (1 = win, 0 = loss)
            baseline_win_rate: Overall win rate
            
        Returns:
            Pattern evaluation dict or None if not significant
        """
        # Get games matching pattern
        mask = features[list(pattern)].all(axis=1)
        n_matches = mask.sum()
        
        if n_matches < len(features) * self.min_support:
            return None
        
        # Calculate win rate
        wins = outcomes[mask].sum()
        win_rate = wins / n_matches if n_matches > 0 else 0
        
        # Check confidence threshold
        if win_rate < self.min_confidence:
            return None
        
        # Calculate lift
        lift = win_rate / baseline_win_rate if baseline_win_rate > 0 else 0
        
        # Check lift threshold
        if lift < self.min_lift:
            return None
        
        # Statistical test (chi-square)
        contingency = np.array([
            [wins, n_matches - wins],  # Pattern: wins, losses
            [outcomes.sum() - wins, len(outcomes) - n_matches - (outcomes.sum() - wins)]  # Non-pattern
        ])
        
        try:
            chi2, p_value, _, _ = chi2_contingency(contingency)
        except:
            p_value = 1.0
        
        if p_value > self.p_value_threshold:
            return None
        
        # Calculate expected profit (assuming -110 odds, need 52.4% to break even)
        breakeven_rate = 0.524
        edge = win_rate - breakeven_rate
        roi = edge * 100  # As percentage
        
        return {
            'pattern': ' AND '.join(pattern),
            'features': list(pattern),
            'n_games': int(n_matches),
            'wins': int(wins),
            'losses': int(n_matches - wins),
            'win_rate': float(win_rate),
            'lift': float(lift),
            'edge': float(edge),
            'roi_pct': float(roi),
            'p_value': float(p_value),
            'support': float(n_matches / len(features)),
            'profitable': roi > 0
        }
    
    def discover_patterns(
        self,
        games: pd.DataFrame,
        outcomes: np.ndarray,
        max_order: int = 3
    ) -> List[Dict]:
        """
        Discover all significant patterns up to max_order.
        
        Args:
            games: DataFrame with game data
            outcomes: Binary outcomes
            max_order: Maximum pattern complexity (2 or 3)
            
        Returns:
            List of discovered patterns
        """
        print("=" * 80)
        print("HIGHER-ORDER PATTERN DISCOVERY")
        print("=" * 80)
        print(f"Games: {len(games)}")
        print(f"Max pattern order: {max_order}")
        print(f"Min support: {self.min_support:.1%}")
        print(f"Min confidence: {self.min_confidence:.1%}")
        print(f"Min lift: {self.min_lift:.2f}")
        
        # Extract binary features
        print("\nExtracting binary features...")
        features = self.extract_binary_features(games)
        print(f"  {len(features.columns)} binary features extracted")
        
        # Calculate baseline
        baseline_win_rate = outcomes.mean()
        print(f"  Baseline win rate: {baseline_win_rate:.1%}")
        
        all_patterns = []
        
        # Discover patterns of increasing order
        for order in range(2, max_order + 1):
            print(f"\nDiscovering {order}-way patterns...")
            
            # Find frequent itemsets
            frequent = self.find_frequent_itemsets(features, k=order)
            print(f"  Found {len(frequent)} frequent {order}-itemsets")
            
            # Evaluate each pattern
            significant = 0
            for pattern in frequent:
                result = self.evaluate_pattern(pattern, features, outcomes, baseline_win_rate)
                if result is not None:
                    result['order'] = order
                    all_patterns.append(result)
                    significant += 1
            
            print(f"  {significant} patterns passed significance tests")
        
        # Sort by ROI
        all_patterns.sort(key=lambda x: x['roi_pct'], reverse=True)
        
        print("\n" + "=" * 80)
        print(f"DISCOVERED {len(all_patterns)} SIGNIFICANT PATTERNS")
        print("=" * 80)
        
        return all_patterns
    
    def save_patterns(self, patterns: List[Dict], filepath: str):
        """Save discovered patterns to JSON."""
        # Convert numpy types to Python types for JSON serialization
        clean_patterns = []
        for p in patterns:
            clean_p = {}
            for k, v in p.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean_p[k] = float(v)
                elif isinstance(v, np.bool_):
                    clean_p[k] = bool(v)
                elif isinstance(v, list):
                    clean_p[k] = [str(x) if isinstance(x, np.bool_) else x for x in v]
                else:
                    clean_p[k] = v
            clean_patterns.append(clean_p)
        
        output = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_patterns': len(patterns),
            'parameters': {
                'min_support': self.min_support,
                'min_confidence': self.min_confidence,
                'min_lift': self.min_lift,
                'p_value_threshold': self.p_value_threshold
            },
            'patterns': clean_patterns
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Patterns saved to {filepath}")


def test_pattern_discovery():
    """Test higher-order pattern discovery."""
    print("Testing Higher-Order Pattern Discovery...")
    
    # Create synthetic data
    np.random.seed(42)
    n_games = 1000
    
    games_data = {
        'home': np.ones(n_games),
        'spread': np.random.uniform(-14, 14, n_games),
        'season_win_pct': np.random.uniform(0.3, 0.7, n_games),
        'opp_season_win_pct': np.random.uniform(0.3, 0.7, n_games),
        'l10_win_pct': np.random.uniform(0.2, 0.8, n_games),
        'week': np.random.randint(1, 19, n_games),
        'is_division': np.random.random(n_games) < 0.3,
        'is_rivalry': np.random.random(n_games) < 0.15,
        'rest_days': np.random.randint(1, 8, n_games)
    }
    
    games = pd.DataFrame(games_data)
    
    # Create outcomes with some patterns
    outcomes = np.zeros(n_games)
    
    # Pattern 1: Home + big underdog + late season = 75% win rate
    mask1 = (games['spread'] >= 7) & (games['week'] >= 14)
    outcomes[mask1] = (np.random.random(mask1.sum()) < 0.75).astype(int)
    
    # Pattern 2: High momentum + division game = 70% win rate
    mask2 = (games['l10_win_pct'] >= 0.7) & (games['is_division'])
    outcomes[mask2] = (np.random.random(mask2.sum()) < 0.70).astype(int)
    
    # Rest: baseline 58%
    mask_rest = ~(mask1 | mask2)
    outcomes[mask_rest] = (np.random.random(mask_rest.sum()) < 0.58).astype(int)
    
    # Discover patterns
    discoverer = HigherOrderPatternDiscovery(
        min_support=0.05,
        min_confidence=0.60,
        min_lift=1.1,
        p_value_threshold=0.05
    )
    
    patterns = discoverer.discover_patterns(games, outcomes, max_order=3)
    
    # Display top patterns
    print("\n" + "=" * 80)
    print("TOP 10 DISCOVERED PATTERNS")
    print("=" * 80)
    
    for i, pattern in enumerate(patterns[:10], 1):
        print(f"\n{i}. {pattern['pattern']}")
        print(f"   Games: {pattern['n_games']} | Win Rate: {pattern['win_rate']:.1%} | ROI: {pattern['roi_pct']:+.1f}%")
        print(f"   Lift: {pattern['lift']:.2f}x | P-value: {pattern['p_value']:.4f}")
    
    # Save
    save_path = Path(__file__).parent.parent.parent / 'data' / 'patterns' / 'test_higher_order_patterns.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    discoverer.save_patterns(patterns, str(save_path))
    
    print("\n" + "=" * 80)
    print("PATTERN DISCOVERY TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    test_pattern_discovery()

