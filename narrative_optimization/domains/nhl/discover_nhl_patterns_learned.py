"""
NHL Data-Driven Pattern Discovery

LEARNS patterns from the actual 79 features extracted by transformers.
No hardcoding - discovers what matters from the data itself.

Methodology:
1. Analyze feature correlations with outcomes
2. Find optimal thresholds via quartile analysis
3. Test feature combinations
4. Validate with random forest feature importance
5. Discover patterns the DATA reveals, not what we assume

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class DataDrivenPatternDiscoverer:
    """Discover patterns by learning from actual feature distributions"""
    
    def __init__(self, min_sample_size: int = 15, min_win_rate: float = 0.53, min_roi: float = 0.05):
        """Initialize discoverer"""
        self.min_sample_size = min_sample_size
        self.min_win_rate = min_win_rate
        self.min_roi = min_roi
        
        self.feature_names = self._get_feature_names()
        self.patterns = []
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names from our transformers"""
        
        # Performance features (50)
        perf_features = [
            # Offensive (10)
            'goals_per_game', 'shots_per_game', 'shooting_pct',
            'even_strength_goals', 'high_danger_chances', 'expected_goals',
            'offensive_zone_time', 'faceoff_win_pct', 'assists_per_game',
            'offensive_momentum',
            
            # Defensive (10)
            'goals_against', 'shots_against', 'save_percentage',
            'blocks_per_game', 'takeaways', 'hits_defensive',
            'high_danger_chances_against', 'expected_goals_against',
            'defensive_zone_time', 'defensive_consistency',
            
            # Goalie (10)
            'goalie_save_pct', 'goalie_gaa', 'goalie_shutouts',
            'goalie_wins', 'goalie_recent_form', 'goalie_vs_opponent',
            'goalie_home_road_split', 'goalie_rest', 'goalie_playoff_exp',
            'goalie_starter_quality',
            
            # Physical (5)
            'hits_per_game', 'penalty_minutes', 'fighting_majors',
            'playoff_toughness', 'rivalry_intensity',
            
            # Special Teams (5)
            'power_play_pct', 'penalty_kill_pct', 'pp_goals',
            'shorthanded_goals', 'special_teams_diff',
            
            # Contextual (10)
            'home_advantage', 'back_to_back_performance', 'rest_advantage',
            'division_performance', 'playoff_performance', 'l5_form',
            'l10_form', 'h2h_record', 'win_streak', 'season_phase'
        ]
        
        # Nominative features (29)
        nom_features = [
            'away_brand_weight', 'away_coach_prestige', 'away_cup_history',
            'away_goalie_name_length', 'away_goalie_phonetic', 'away_goalie_prestige',
            'away_is_expansion', 'away_name_density', 'away_star_power',
            'brand_differential', 'coach_prestige_diff', 'combined_brand_gravity',
            'cup_history_diff', 'goalie_prestige_diff', 'home_brand_weight',
            'home_coach_prestige', 'home_cup_history', 'home_goalie_name_length',
            'home_goalie_phonetic', 'home_goalie_prestige', 'home_is_expansion',
            'home_name_density', 'home_star_power', 'legendary_goalie_matchup',
            'original_six_matchup', 'playoff_nominative_amp', 'rivalry_nominative_boost',
            'star_power_diff', 'total_nominative_gravity'
        ]
        
        return perf_features + nom_features
    
    def analyze_feature_importance(self, features: np.ndarray, outcomes: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Use Random Forest to find which features actually matter.
        This is data-driven - not our assumptions.
        """
        print("\n🌲 ANALYZING FEATURE IMPORTANCE (Random Forest)")
        print("-"*80)
        
        # Train random forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, min_samples_leaf=5)
        rf.fit(features, outcomes)
        
        # Get feature importance
        importances = rf.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        print(f"\n   Top 20 Most Important Features:")
        for i in range(min(20, len(indices))):
            idx = indices[i]
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f'Feature_{idx}'
            print(f"   {i+1:2d}. {feat_name:35s} = {importances[idx]:.4f}")
        
        return importances, [self.feature_names[i] if i < len(self.feature_names) else f'Feature_{i}' for i in indices]
    
    def analyze_feature_correlations(self, features: np.ndarray, outcomes: np.ndarray) -> List[Tuple[int, float, float]]:
        """
        Find which features correlate with winning.
        Pure data analysis.
        """
        print("\n📊 ANALYZING FEATURE CORRELATIONS")
        print("-"*80)
        
        correlations = []
        
        for i in range(features.shape[1]):
            try:
                r, p = pearsonr(features[:, i], outcomes)
                if not np.isnan(r):
                    correlations.append((i, r, p))
            except:
                continue
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n   Top 20 Correlated Features:")
        for i in range(min(20, len(correlations))):
            feat_idx, r, p = correlations[i]
            feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f'Feature_{feat_idx}'
            direction = "positive" if r > 0 else "negative"
            print(f"   {i+1:2d}. {feat_name:35s} r={r:+.4f} (p={p:.4f}) [{direction}]")
        
        return correlations
    
    def discover_threshold_patterns(self, games: List[Dict], features: np.ndarray, 
                                   importances: np.ndarray, top_n: int = 15) -> List[Dict]:
        """
        For the most important features, find optimal thresholds by binning.
        This discovers patterns from the data distribution.
        """
        print("\n🔍 DISCOVERING THRESHOLD PATTERNS (Data-Driven)")
        print("-"*80)
        print(f"   Testing thresholds for top {top_n} features...")
        
        patterns = []
        outcomes = np.array([g.get('home_won', False) for g in games], dtype=int)
        
        # Get top features by importance
        top_indices = np.argsort(importances)[::-1][:top_n]
        
        for feat_idx in top_indices:
            feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f'Feature_{feat_idx}'
            feat_values = features[:, feat_idx]
            
            # Try quartile-based thresholds
            q25 = np.percentile(feat_values, 25)
            q50 = np.percentile(feat_values, 50)
            q75 = np.percentile(feat_values, 75)
            
            thresholds = [
                (f'> Q3 (75th percentile)', feat_values > q75),
                (f'< Q1 (25th percentile)', feat_values < q25),
                (f'> Median', feat_values > q50),
                (f'< Median', feat_values < q50),
            ]
            
            for threshold_desc, mask in thresholds:
                matching_games = [games[i] for i in range(len(games)) if mask[i]]
                
                if len(matching_games) >= self.min_sample_size:
                    wins = sum(1 for g in matching_games if g.get('home_won', False))
                    win_rate = wins / len(matching_games)
                    roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
                    
                    if win_rate >= self.min_win_rate and roi >= self.min_roi:
                        pattern = {
                            'name': f'{feat_name} {threshold_desc}',
                            'description': f'Games where {feat_name} is {threshold_desc.lower()}',
                            'feature': feat_name,
                            'feature_idx': int(feat_idx),
                            'threshold': threshold_desc,
                            'n_games': len(matching_games),
                            'wins': wins,
                            'losses': len(matching_games) - wins,
                            'win_rate': win_rate,
                            'win_rate_pct': win_rate * 100,
                            'roi': roi,
                            'roi_pct': roi * 100,
                            'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                            'unit_recommendation': 2 if win_rate > 0.57 else 1,
                            'pattern_type': 'threshold',
                        }
                        patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} threshold-based patterns")
        return patterns
    
    def discover_combination_patterns(self, games: List[Dict], features: np.ndarray, 
                                     correlations: List[Tuple[int, float, float]], 
                                     top_n: int = 10) -> List[Dict]:
        """
        Test combinations of top correlated features.
        Discover multi-factor patterns from the data.
        """
        print("\n🔬 DISCOVERING COMBINATION PATTERNS (Data-Driven)")
        print("-"*80)
        print(f"   Testing combinations of top {top_n} features...")
        
        patterns = []
        outcomes = np.array([g.get('home_won', False) for g in games], dtype=int)
        
        # Get top correlated features
        top_features = [c[0] for c in correlations[:top_n]]
        
        # Test pairwise combinations
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                feat1_idx = top_features[i]
                feat2_idx = top_features[j]
                
                feat1_name = self.feature_names[feat1_idx] if feat1_idx < len(self.feature_names) else f'Feature_{feat1_idx}'
                feat2_name = self.feature_names[feat2_idx] if feat2_idx < len(self.feature_names) else f'Feature_{feat2_idx}'
                
                # Get feature values
                feat1_values = features[:, feat1_idx]
                feat2_values = features[:, feat2_idx]
                
                # Try different combinations
                feat1_median = np.median(feat1_values)
                feat2_median = np.median(feat2_values)
                
                combinations = [
                    (f'High {feat1_name} + High {feat2_name}', 
                     (feat1_values > feat1_median) & (feat2_values > feat2_median)),
                    (f'Low {feat1_name} + High {feat2_name}',
                     (feat1_values < feat1_median) & (feat2_values > feat2_median)),
                    (f'High {feat1_name} + Low {feat2_name}',
                     (feat1_values > feat1_median) & (feat2_values < feat2_median)),
                ]
                
                for combo_desc, mask in combinations:
                    matching_games = [games[i] for i in range(len(games)) if mask[i]]
                    
                    if len(matching_games) >= self.min_sample_size:
                        wins = sum(1 for g in matching_games if g.get('home_won', False))
                        win_rate = wins / len(matching_games)
                        roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
                        
                        if win_rate >= self.min_win_rate and roi >= self.min_roi:
                            pattern = {
                                'name': combo_desc,
                                'description': f'Games with {combo_desc.lower()}',
                                'features': [feat1_name, feat2_name],
                                'feature_indices': [int(feat1_idx), int(feat2_idx)],
                                'n_games': len(matching_games),
                                'wins': wins,
                                'losses': len(matching_games) - wins,
                                'win_rate': win_rate,
                                'win_rate_pct': win_rate * 100,
                                'roi': roi,
                                'roi_pct': roi * 100,
                                'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                                'unit_recommendation': 2 if win_rate > 0.57 else 1,
                                'pattern_type': 'combination',
                            }
                            patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} combination patterns")
        return patterns
    
    def discover_gradient_boosting_patterns(self, games: List[Dict], features: np.ndarray) -> List[Dict]:
        """
        Use Gradient Boosting to find complex interactions.
        Let the algorithm discover patterns.
        """
        print("\n🚀 DISCOVERING GRADIENT BOOSTING PATTERNS")
        print("-"*80)
        
        outcomes = np.array([g.get('home_won', False) for g in games], dtype=int)
        
        # Train gradient boosting
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        gb.fit(features, outcomes)
        
        # Get predictions and probabilities
        probs = gb.predict_proba(features)[:, 1]  # Probability of home win
        
        # Find high-confidence predictions
        patterns = []
        
        confidence_thresholds = [0.60, 0.55, 0.50]
        
        for conf_threshold in confidence_thresholds:
            high_conf_mask = probs > conf_threshold
            matching_games = [games[i] for i in range(len(games)) if high_conf_mask[i]]
            
            if len(matching_games) >= self.min_sample_size:
                wins = sum(1 for g in matching_games if g.get('home_won', False))
                win_rate = wins / len(matching_games)
                roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
                
                if win_rate >= self.min_win_rate and roi >= self.min_roi:
                    pattern = {
                        'name': f'GBM Confidence ≥{conf_threshold:.0%}',
                        'description': f'Games where model predicts home win with {conf_threshold:.0%}+ confidence',
                        'n_games': len(matching_games),
                        'wins': wins,
                        'losses': len(matching_games) - wins,
                        'win_rate': win_rate,
                        'win_rate_pct': win_rate * 100,
                        'roi': roi,
                        'roi_pct': roi * 100,
                        'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                        'unit_recommendation': 2 if win_rate > 0.57 else 1,
                        'pattern_type': 'ml_confidence',
                        'model_threshold': conf_threshold,
                    }
                    patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} ML-based patterns")
        return patterns
    
    def discover_all_patterns(self, games: List[Dict], features: np.ndarray) -> List[Dict]:
        """Discover all patterns using data-driven methods"""
        
        print("\n" + "="*80)
        print("NHL DATA-DRIVEN PATTERN DISCOVERY")
        print("="*80)
        print(f"Analyzing {len(games)} games with {features.shape[1]} features...")
        print("Learning from actual transformer output - NO HARDCODING")
        
        outcomes = np.array([g.get('home_won', False) for g in games], dtype=int)
        
        all_patterns = []
        
        # 1. Feature importance analysis
        importances, sorted_names = self.analyze_feature_importance(features, outcomes)
        
        # 2. Feature correlation analysis
        correlations = self.analyze_feature_correlations(features, outcomes)
        
        # 3. Threshold-based patterns (top important features)
        all_patterns.extend(self.discover_threshold_patterns(games, features, importances, top_n=15))
        
        # 4. Combination patterns (top correlated features)
        all_patterns.extend(self.discover_combination_patterns(games, features, correlations, top_n=10))
        
        # 5. Gradient boosting patterns (complex interactions)
        all_patterns.extend(self.discover_gradient_boosting_patterns(games, features))
        
        # Sort by ROI
        all_patterns.sort(key=lambda x: x['roi'], reverse=True)
        
        # Remove near-duplicates (similar sample size and win rate)
        unique_patterns = []
        seen = set()
        for pattern in all_patterns:
            key = (pattern['n_games'] // 5, int(pattern['win_rate'] * 100))  # Group similar patterns
            if key not in seen:
                seen.add(key)
                unique_patterns.append(pattern)
        
        print("\n" + "="*80)
        print(f"✅ DISCOVERED {len(unique_patterns)} DATA-DRIVEN PATTERNS")
        print("="*80)
        
        # Print top 20
        for i, pattern in enumerate(unique_patterns[:20], 1):
            print(f"\n{i}. {pattern['name']}")
            print(f"   {pattern['description']}")
            print(f"   Games: {pattern['n_games']}, Win: {pattern['win_rate_pct']:.1f}%, ROI: {pattern['roi_pct']:.1f}%")
        
        if len(unique_patterns) > 20:
            print(f"\n... and {len(unique_patterns) - 20} more patterns")
        
        self.patterns = unique_patterns
        return unique_patterns


def main():
    """Main execution"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
    features_path = project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'nhl_features_complete.npz'
    output_dir = project_root / 'data' / 'domains'
    
    # Load data
    print(f"\n📂 Loading NHL data and features...")
    with open(data_path, 'r') as f:
        games = json.load(f)
    
    data = np.load(features_path)
    features = data['features']
    
    print(f"   ✓ Loaded {len(games)} games")
    print(f"   ✓ Loaded {features.shape[1]} features from transformers")
    
    # Discover patterns
    discoverer = DataDrivenPatternDiscoverer(
        min_sample_size=15,
        min_win_rate=0.53,
        min_roi=0.05
    )
    
    patterns = discoverer.discover_all_patterns(games, features)
    
    # Save
    output_path = output_dir / 'nhl_betting_patterns_learned.json'
    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"\n💾 PATTERNS SAVED: {output_path}")
    print(f"✅ Data-driven discovery complete!")
    print(f"\n📊 Summary:")
    print(f"   Total patterns: {len(patterns)}")
    print(f"   Learned from 79 transformer features")
    print(f"   No hardcoded assumptions")
    print(f"   Pure data-driven discovery")


if __name__ == "__main__":
    main()

