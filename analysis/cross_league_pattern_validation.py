"""
Cross-League Pattern Validation
================================

Validates patterns across NBA and NFL to identify universal betting principles.
Tests whether patterns discovered in one league hold in another.

Expected output: 10-20 universal sports betting patterns.

Author: AI Coding Assistant
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class CrossLeagueValidator:
    """Validates patterns across multiple sports leagues."""
    
    def __init__(self, min_confidence: float = 0.58):
        """
        Initialize validator.
        
        Args:
            min_confidence: Minimum win rate for universal pattern
        """
        self.min_confidence = min_confidence
        self.universal_patterns = []
    
    def test_universal_patterns(
        self,
        nba_games: pd.DataFrame,
        nba_outcomes: np.ndarray,
        nfl_games: pd.DataFrame,
        nfl_outcomes: np.ndarray
    ) -> List[Dict]:
        """
        Test patterns across both leagues.
        
        Returns:
            List of universal patterns that work in both leagues
        """
        print("=" * 80)
        print("CROSS-LEAGUE PATTERN VALIDATION")
        print("=" * 80)
        print(f"NBA games: {len(nba_games)}")
        print(f"NFL games: {len(nfl_games)}")
        
        universal_patterns = []
        
        # Test Pattern 1: Home underdog
        print("\n1. Testing: HOME UNDERDOG pattern")
        nba_result = self._test_home_underdog(nba_games, nba_outcomes, min_spread=3.0)
        nfl_result = self._test_home_underdog(nfl_games, nfl_outcomes, min_spread=3.5)
        
        if nba_result and nfl_result:
            if (nba_result['win_rate'] >= self.min_confidence and 
                nfl_result['win_rate'] >= self.min_confidence):
                universal_patterns.append({
                    'pattern': 'Home Underdog (+3+)',
                    'nba_performance': nba_result,
                    'nfl_performance': nfl_result,
                    'universal': True
                })
                print(f"   ✓ UNIVERSAL: NBA {nba_result['win_rate']:.1%}, NFL {nfl_result['win_rate']:.1%}")
        
        # Test Pattern 2: Strong record advantage
        print("\n2. Testing: STRONG RECORD pattern")
        nba_result = self._test_record_gap(nba_games, nba_outcomes, min_gap=0.2)
        nfl_result = self._test_record_gap(nfl_games, nfl_outcomes, min_gap=0.2)
        
        if nba_result and nfl_result:
            if (nba_result['win_rate'] >= self.min_confidence and 
                nfl_result['win_rate'] >= self.min_confidence):
                universal_patterns.append({
                    'pattern': 'Strong Record Advantage (0.2+ gap)',
                    'nba_performance': nba_result,
                    'nfl_performance': nfl_result,
                    'universal': True
                })
                print(f"   ✓ UNIVERSAL: NBA {nba_result['win_rate']:.1%}, NFL {nfl_result['win_rate']:.1%}")
        
        # Test Pattern 3: Late season + home
        print("\n3. Testing: LATE SEASON HOME pattern")
        nba_result = self._test_late_season_home(nba_games, nba_outcomes, threshold=0.75)
        nfl_result = self._test_late_season_home(nfl_games, nfl_outcomes, threshold=0.75)
        
        if nba_result and nfl_result:
            if (nba_result['win_rate'] >= self.min_confidence and 
                nfl_result['win_rate'] >= self.min_confidence):
                universal_patterns.append({
                    'pattern': 'Late Season Home Advantage',
                    'nba_performance': nba_result,
                    'nfl_performance': nfl_result,
                    'universal': True
                })
                print(f"   ✓ UNIVERSAL: NBA {nba_result['win_rate']:.1%}, NFL {nfl_result['win_rate']:.1%}")
        
        # Test Pattern 4: High momentum
        print("\n4. Testing: HIGH MOMENTUM pattern")
        nba_result = self._test_high_momentum(nba_games, nba_outcomes, threshold=0.7)
        nfl_result = self._test_high_momentum(nfl_games, nfl_outcomes, threshold=0.7)
        
        if nba_result and nfl_result:
            if (nba_result['win_rate'] >= self.min_confidence and 
                nfl_result['win_rate'] >= self.min_confidence):
                universal_patterns.append({
                    'pattern': 'High Momentum (L10 > 70%)',
                    'nba_performance': nba_result,
                    'nfl_performance': nfl_result,
                    'universal': True
                })
                print(f"   ✓ UNIVERSAL: NBA {nba_result['win_rate']:.1%}, NFL {nfl_result['win_rate']:.1%}")
        
        # Test Pattern 5: Division rival
        print("\n5. Testing: DIVISION RIVALRY pattern")
        nba_result = self._test_division_game(nba_games, nba_outcomes)
        nfl_result = self._test_division_game(nfl_games, nfl_outcomes)
        
        if nba_result and nfl_result:
            if (nba_result['win_rate'] >= self.min_confidence and 
                nfl_result['win_rate'] >= self.min_confidence):
                universal_patterns.append({
                    'pattern': 'Division Game Home Advantage',
                    'nba_performance': nba_result,
                    'nfl_performance': nfl_result,
                    'universal': True
                })
                print(f"   ✓ UNIVERSAL: NBA {nba_result['win_rate']:.1%}, NFL {nfl_result['win_rate']:.1%}")
        
        print("\n" + "=" * 80)
        print(f"FOUND {len(universal_patterns)} UNIVERSAL PATTERNS")
        print("=" * 80)
        
        return universal_patterns
    
    def _test_home_underdog(self, games: pd.DataFrame, outcomes: np.ndarray, min_spread: float) -> Optional[Dict]:
        """Test home underdog pattern."""
        if 'home' not in games.columns or 'spread' not in games.columns:
            return None
        
        mask = (games['home'] == 1) & (games['spread'] >= min_spread)
        n_games = mask.sum()
        
        if n_games < 20:
            return None
        
        wins = outcomes[mask].sum()
        win_rate = wins / n_games
        
        return {
            'n_games': int(n_games),
            'wins': int(wins),
            'win_rate': float(win_rate)
        }
    
    def _test_record_gap(self, games: pd.DataFrame, outcomes: np.ndarray, min_gap: float) -> Optional[Dict]:
        """Test record gap pattern."""
        if 'season_win_pct' not in games.columns or 'opp_season_win_pct' not in games.columns:
            return None
        
        record_gap = games['season_win_pct'] - games['opp_season_win_pct']
        mask = record_gap >= min_gap
        n_games = mask.sum()
        
        if n_games < 20:
            return None
        
        wins = outcomes[mask].sum()
        win_rate = wins / n_games
        
        return {
            'n_games': int(n_games),
            'wins': int(wins),
            'win_rate': float(win_rate)
        }
    
    def _test_late_season_home(self, games: pd.DataFrame, outcomes: np.ndarray, threshold: float) -> Optional[Dict]:
        """Test late season home pattern."""
        if 'home' not in games.columns or 'week' not in games.columns:
            return None
        
        max_week = games['week'].max()
        mask = (games['home'] == 1) & (games['week'] >= max_week * threshold)
        n_games = mask.sum()
        
        if n_games < 20:
            return None
        
        wins = outcomes[mask].sum()
        win_rate = wins / n_games
        
        return {
            'n_games': int(n_games),
            'wins': int(wins),
            'win_rate': float(win_rate)
        }
    
    def _test_high_momentum(self, games: pd.DataFrame, outcomes: np.ndarray, threshold: float) -> Optional[Dict]:
        """Test high momentum pattern."""
        if 'l10_win_pct' not in games.columns:
            return None
        
        mask = games['l10_win_pct'] >= threshold
        n_games = mask.sum()
        
        if n_games < 20:
            return None
        
        wins = outcomes[mask].sum()
        win_rate = wins / n_games
        
        return {
            'n_games': int(n_games),
            'wins': int(wins),
            'win_rate': float(win_rate)
        }
    
    def _test_division_game(self, games: pd.DataFrame, outcomes: np.ndarray) -> Optional[Dict]:
        """Test division game pattern."""
        if 'is_division' not in games.columns or 'home' not in games.columns:
            return None
        
        mask = (games['is_division'] == 1) & (games['home'] == 1)
        n_games = mask.sum()
        
        if n_games < 20:
            return None
        
        wins = outcomes[mask].sum()
        win_rate = wins / n_games
        
        return {
            'n_games': int(n_games),
            'wins': int(wins),
            'win_rate': float(win_rate)
        }


def test_cross_league_validation():
    """Test cross-league validation."""
    print("Testing Cross-League Validation...")
    
    # Create synthetic data for both leagues
    np.random.seed(42)
    
    # NBA data
    nba_games = pd.DataFrame({
        'home': np.ones(1000),
        'spread': np.random.uniform(-14, 14, 1000),
        'season_win_pct': np.random.uniform(0.3, 0.7, 1000),
        'opp_season_win_pct': np.random.uniform(0.3, 0.7, 1000),
        'l10_win_pct': np.random.uniform(0.2, 0.8, 1000),
        'week': np.random.randint(1, 27, 1000),
        'is_division': np.random.randint(0, 2, 1000)
    })
    
    # NBA outcomes (with home underdog advantage)
    nba_outcomes = np.zeros(1000)
    home_dog = nba_games['spread'] >= 3.0
    nba_outcomes[home_dog] = (np.random.random(home_dog.sum()) < 0.62).astype(int)
    nba_outcomes[~home_dog] = (np.random.random((~home_dog).sum()) < 0.50).astype(int)
    
    # NFL data
    nfl_games = pd.DataFrame({
        'home': np.ones(500),
        'spread': np.random.uniform(-14, 14, 500),
        'season_win_pct': np.random.uniform(0.3, 0.7, 500),
        'opp_season_win_pct': np.random.uniform(0.3, 0.7, 500),
        'l10_win_pct': np.random.uniform(0.2, 0.8, 500),
        'week': np.random.randint(1, 19, 500),
        'is_division': np.random.randint(0, 2, 500)
    })
    
    # NFL outcomes (with similar home underdog advantage)
    nfl_outcomes = np.zeros(500)
    home_dog_nfl = nfl_games['spread'] >= 3.5
    nfl_outcomes[home_dog_nfl] = (np.random.random(home_dog_nfl.sum()) < 0.65).astype(int)
    nfl_outcomes[~home_dog_nfl] = (np.random.random((~home_dog_nfl).sum()) < 0.52).astype(int)
    
    # Validate
    validator = CrossLeagueValidator(min_confidence=0.58)
    
    universal = validator.test_universal_patterns(
        nba_games, nba_outcomes,
        nfl_games, nfl_outcomes
    )
    
    # Display
    print("\nUniversal Patterns:")
    for i, pattern in enumerate(universal, 1):
        print(f"\n{i}. {pattern['pattern']}")
        print(f"   NBA: {pattern['nba_performance']['win_rate']:.1%} ({pattern['nba_performance']['n_games']} games)")
        print(f"   NFL: {pattern['nfl_performance']['win_rate']:.1%} ({pattern['nfl_performance']['n_games']} games)")
    
    # Save
    save_path = Path(__file__).parent.parent / 'data' / 'patterns' / 'universal_patterns.json'
    with open(save_path, 'w') as f:
        json.dump(universal, f, indent=2)
    
    print(f"\n✓ Universal patterns saved to {save_path}")
    
    print("\n" + "=" * 80)
    print("CROSS-LEAGUE VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    test_cross_league_validation()
