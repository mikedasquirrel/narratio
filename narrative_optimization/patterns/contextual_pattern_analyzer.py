"""
Context-Aware Pattern Analyzer
===============================

Analyzes patterns with contextual splits:
- Home/away performance
- Rest days impact (back-to-back, well-rested)
- Matchup-specific (vs top 5, vs bottom 5)
- Temporal (early/mid/late season, playoffs)
- Weather conditions (NFL outdoor games)

Expected output: 30-50 context variants per base pattern.

Author: AI Coding Assistant  
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import chi2_contingency
import json
from pathlib import Path


class ContextualPatternAnalyzer:
    """
    Analyzes patterns across different contextual dimensions.
    """
    
    def __init__(self, min_samples: int = 20, min_win_rate: float = 0.58):
        """
        Initialize analyzer.
        
        Args:
            min_samples: Minimum games needed for valid context
            min_win_rate: Minimum win rate for profitable context
        """
        self.min_samples = min_samples
        self.min_win_rate = min_win_rate
    
    def analyze_home_away_split(
        self,
        pattern_games: pd.DataFrame,
        outcomes: np.ndarray
    ) -> Dict:
        """Analyze pattern performance by home/away."""
        results = {}
        
        if 'home' not in pattern_games.columns:
            return results
        
        for location in [True, False]:
            mask = (pattern_games['home'] == location)
            n_games = mask.sum()
            
            if n_games >= self.min_samples:
                wins = outcomes[mask].sum()
                win_rate = wins / n_games
                
                results['home' if location else 'away'] = {
                    'n_games': int(n_games),
                    'wins': int(wins),
                    'win_rate': float(win_rate),
                    'profitable': win_rate >= self.min_win_rate
                }
        
        return results
    
    def analyze_rest_split(
        self,
        pattern_games: pd.DataFrame,
        outcomes: np.ndarray
    ) -> Dict:
        """Analyze pattern performance by rest days."""
        results = {}
        
        if 'rest_days' not in pattern_games.columns:
            return results
        
        # Define rest categories
        rest_categories = [
            ('back_to_back', lambda x: x <= 1),
            ('short_rest', lambda x: 1 < x <= 2),
            ('normal_rest', lambda x: 2 < x <= 4),
            ('well_rested', lambda x: x > 4)
        ]
        
        for cat_name, cat_filter in rest_categories:
            mask = pattern_games['rest_days'].apply(cat_filter)
            n_games = mask.sum()
            
            if n_games >= self.min_samples:
                wins = outcomes[mask].sum()
                win_rate = wins / n_games
                
                results[cat_name] = {
                    'n_games': int(n_games),
                    'wins': int(wins),
                    'win_rate': float(win_rate),
                    'profitable': win_rate >= self.min_win_rate
                }
        
        return results
    
    def analyze_matchup_split(
        self,
        pattern_games: pd.DataFrame,
        outcomes: np.ndarray
    ) -> Dict:
        """Analyze pattern performance by opponent strength."""
        results = {}
        
        if 'opp_season_win_pct' not in pattern_games.columns:
            return results
        
        # Define opponent categories
        matchup_categories = [
            ('vs_top5', lambda x: x >= 0.65),
            ('vs_good', lambda x: 0.55 <= x < 0.65),
            ('vs_average', lambda x: 0.45 < x < 0.55),
            ('vs_bad', lambda x: 0.35 < x <= 0.45),
            ('vs_bottom5', lambda x: x <= 0.35)
        ]
        
        for cat_name, cat_filter in matchup_categories:
            mask = pattern_games['opp_season_win_pct'].apply(cat_filter)
            n_games = mask.sum()
            
            if n_games >= self.min_samples:
                wins = outcomes[mask].sum()
                win_rate = wins / n_games
                
                results[cat_name] = {
                    'n_games': int(n_games),
                    'wins': int(wins),
                    'win_rate': float(win_rate),
                    'profitable': win_rate >= self.min_win_rate
                }
        
        return results
    
    def analyze_temporal_split(
        self,
        pattern_games: pd.DataFrame,
        outcomes: np.ndarray
    ) -> Dict:
        """Analyze pattern performance by season timing."""
        results = {}
        
        if 'week' not in pattern_games.columns:
            return results
        
        max_week = pattern_games['week'].max()
        
        # Define temporal categories
        temporal_categories = [
            ('early_season', lambda x: x <= max_week * 0.25),
            ('mid_season', lambda x: max_week * 0.25 < x <= max_week * 0.75),
            ('late_season', lambda x: x > max_week * 0.75)
        ]
        
        for cat_name, cat_filter in temporal_categories:
            mask = pattern_games['week'].apply(cat_filter)
            n_games = mask.sum()
            
            if n_games >= self.min_samples:
                wins = outcomes[mask].sum()
                win_rate = wins / n_games
                
                results[cat_name] = {
                    'n_games': int(n_games),
                    'wins': int(wins),
                    'win_rate': float(win_rate),
                    'profitable': win_rate >= self.min_win_rate
                }
        
        return results
    
    def analyze_weather_split(
        self,
        pattern_games: pd.DataFrame,
        outcomes: np.ndarray
    ) -> Dict:
        """Analyze pattern performance by weather (NFL only)."""
        results = {}
        
        if 'is_outdoor' not in pattern_games.columns:
            return results
        
        if 'temp_f' in pattern_games.columns:
            # Temperature-based analysis
            weather_categories = [
                ('cold', lambda x: x < 40),
                ('moderate', lambda x: 40 <= x < 70),
                ('hot', lambda x: x >= 70)
            ]
            
            for cat_name, cat_filter in weather_categories:
                mask = pattern_games['temp_f'].apply(cat_filter) & pattern_games['is_outdoor']
                n_games = mask.sum()
                
                if n_games >= self.min_samples:
                    wins = outcomes[mask].sum()
                    win_rate = wins / n_games
                    
                    results[cat_name] = {
                        'n_games': int(n_games),
                        'wins': int(wins),
                        'win_rate': float(win_rate),
                        'profitable': win_rate >= self.min_win_rate
                    }
        
        # Indoor vs outdoor
        for location in [True, False]:
            mask = (pattern_games['is_outdoor'] == location)
            n_games = mask.sum()
            
            if n_games >= self.min_samples:
                wins = outcomes[mask].sum()
                win_rate = wins / n_games
                
                results['outdoor' if location else 'indoor'] = {
                    'n_games': int(n_games),
                    'wins': int(wins),
                    'win_rate': float(win_rate),
                    'profitable': win_rate >= self.min_win_rate
                }
        
        return results
    
    def analyze_all_contexts(
        self,
        pattern_name: str,
        pattern_games: pd.DataFrame,
        outcomes: np.ndarray
    ) -> Dict:
        """
        Analyze pattern across all contextual dimensions.
        
        Args:
            pattern_name: Pattern identifier
            pattern_games: DataFrame with games matching pattern
            outcomes: Binary outcomes
            
        Returns:
            Dict with all contextual analyses
        """
        print(f"\nAnalyzing contexts for: {pattern_name}")
        print(f"Base pattern: {len(pattern_games)} games, {outcomes.mean():.1%} win rate")
        print("-" * 80)
        
        contexts = {}
        
        # Home/away
        home_away = self.analyze_home_away_split(pattern_games, outcomes)
        if home_away:
            contexts['home_away'] = home_away
            print(f"  Home/Away: {len(home_away)} splits found")
        
        # Rest
        rest = self.analyze_rest_split(pattern_games, outcomes)
        if rest:
            contexts['rest'] = rest
            print(f"  Rest: {len(rest)} categories found")
        
        # Matchup
        matchup = self.analyze_matchup_split(pattern_games, outcomes)
        if matchup:
            contexts['matchup'] = matchup
            print(f"  Matchup: {len(matchup)} categories found")
        
        # Temporal
        temporal = self.analyze_temporal_split(pattern_games, outcomes)
        if temporal:
            contexts['temporal'] = temporal
            print(f"  Temporal: {len(temporal)} categories found")
        
        # Weather (if applicable)
        weather = self.analyze_weather_split(pattern_games, outcomes)
        if weather:
            contexts['weather'] = weather
            print(f"  Weather: {len(weather)} categories found")
        
        return {
            'pattern_name': pattern_name,
            'base_performance': {
                'n_games': len(pattern_games),
                'win_rate': float(outcomes.mean())
            },
            'contexts': contexts,
            'n_total_contexts': sum(len(c) for c in contexts.values())
        }
    
    def find_best_contexts(self, analysis: Dict) -> List[Dict]:
        """Find best performing contexts for a pattern."""
        best_contexts = []
        
        for context_type, contexts in analysis['contexts'].items():
            for context_name, perf in contexts.items():
                if perf['profitable'] and perf['n_games'] >= self.min_samples:
                    best_contexts.append({
                        'pattern': analysis['pattern_name'],
                        'context_type': context_type,
                        'context_name': context_name,
                        'n_games': perf['n_games'],
                        'win_rate': perf['win_rate'],
                        'improvement': perf['win_rate'] - analysis['base_performance']['win_rate']
                    })
        
        # Sort by win rate
        best_contexts.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return best_contexts


def test_contextual_analyzer():
    """Test contextual pattern analyzer."""
    print("=" * 80)
    print("CONTEXTUAL PATTERN ANALYZER TEST")
    print("=" * 80)
    
    # Create synthetic data
    np.random.seed(42)
    n_games = 500
    
    games = pd.DataFrame({
        'game_id': [f'game_{i}' for i in range(n_games)],
        'home': np.random.choice([True, False], n_games),
        'rest_days': np.random.randint(0, 8, n_games),
        'opp_season_win_pct': np.random.uniform(0.3, 0.7, n_games),
        'week': np.random.randint(1, 27, n_games),
        'is_outdoor': np.random.choice([True, False], n_games),
        'temp_f': np.random.uniform(30, 90, n_games)
    })
    
    # Create outcomes with contextual effects
    outcomes = np.zeros(n_games)
    
    # Better at home
    home_mask = games['home'] == True
    outcomes[home_mask] = (np.random.random(home_mask.sum()) < 0.65).astype(int)
    outcomes[~home_mask] = (np.random.random((~home_mask).sum()) < 0.50).astype(int)
    
    # Better against weak opponents
    weak_opp = games['opp_season_win_pct'] < 0.4
    outcomes[weak_opp] = np.maximum(outcomes[weak_opp], (np.random.random(weak_opp.sum()) < 0.70).astype(int))
    
    # Better late season
    late_season = games['week'] > 20
    outcomes[late_season] = np.maximum(outcomes[late_season], (np.random.random(late_season.sum()) < 0.62).astype(int))
    
    # Analyze
    analyzer = ContextualPatternAnalyzer(min_samples=20, min_win_rate=0.58)
    
    analysis = analyzer.analyze_all_contexts('test_pattern', games, outcomes)
    
    # Find best contexts
    print("\n" + "=" * 80)
    print("BEST CONTEXTS")
    print("=" * 80)
    
    best = analyzer.find_best_contexts(analysis)
    
    for i, context in enumerate(best[:10], 1):
        print(f"\n{i}. {context['context_type']}: {context['context_name']}")
        print(f"   Games: {context['n_games']} | Win Rate: {context['win_rate']:.1%} | " +
              f"Improvement: {context['improvement']:+.1%}")
    
    # Save (convert numpy types for JSON)
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    save_path = Path(__file__).parent.parent.parent / 'data' / 'patterns' / 'test_contextual_analysis.json'
    with open(save_path, 'w') as f:
        json.dump(convert_to_json_serializable(analysis), f, indent=2)
    
    print(f"\n✓ Analysis saved to {save_path}")
    
    print("\n" + "=" * 80)
    print("CONTEXTUAL ANALYZER TEST COMPLETE")
    print("=" * 80)
    
    print(f"\nTotal contexts found: {analysis['n_total_contexts']}")
    print(f"Profitable contexts: {len(best)}")


if __name__ == '__main__':
    test_contextual_analyzer()
