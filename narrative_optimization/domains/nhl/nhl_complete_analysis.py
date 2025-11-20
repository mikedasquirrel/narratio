"""
NHL Complete Analysis - Integrating ALL Learnings

This is the master analysis that combines:
1. Data-driven pattern discovery (26 patterns from features)
2. Cross-domain insights (NBA momentum, NFL prestige)
3. Transformer feature importance (what actually matters)
4. Ensemble meta-learning (multi-model approach)

Creates comprehensive betting strategy with confidence scores.

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class NHLCompleteAnalyzer:
    """Master analyzer integrating all learnings"""
    
    def __init__(self):
        """Initialize with all models"""
        self.min_sample = 15
        self.min_win_rate = 0.53
        self.min_roi = 0.05
        
        # Build ensemble of models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    max_iter=20000,
                    solver='saga',
                    penalty='l2',
                    C=1.0,
                    random_state=42,
                )),
            ]),
        }
        
        self.feature_names = self._load_feature_names()
    
    def _load_feature_names(self) -> List[str]:
        """Load the 79 feature names"""
        # Performance (50) + Nominative (29)
        perf = [
            'goals_per_game', 'shots_per_game', 'shooting_pct', 'even_strength_goals', 
            'high_danger_chances', 'expected_goals', 'offensive_zone_time', 'faceoff_win_pct',
            'assists_per_game', 'offensive_momentum', 'goals_against', 'shots_against',
            'save_percentage', 'blocks_per_game', 'takeaways', 'hits_defensive',
            'high_danger_chances_against', 'expected_goals_against', 'defensive_zone_time',
            'defensive_consistency', 'goalie_save_pct', 'goalie_gaa', 'goalie_shutouts',
            'goalie_wins', 'goalie_recent_form', 'goalie_vs_opponent', 'goalie_home_road_split',
            'goalie_rest', 'goalie_playoff_exp', 'goalie_starter_quality', 'hits_per_game',
            'penalty_minutes', 'fighting_majors', 'playoff_toughness', 'rivalry_intensity',
            'power_play_pct', 'penalty_kill_pct', 'pp_goals', 'shorthanded_goals',
            'special_teams_diff', 'home_advantage', 'back_to_back_performance', 'rest_advantage',
            'division_performance', 'playoff_performance', 'l5_form', 'l10_form', 'h2h_record',
            'win_streak', 'season_phase'
        ]
        
        nom = [
            'away_brand_weight', 'away_coach_prestige', 'away_cup_history', 'away_goalie_name_length',
            'away_goalie_phonetic', 'away_goalie_prestige', 'away_is_expansion', 'away_name_density',
            'away_star_power', 'brand_differential', 'coach_prestige_diff', 'combined_brand_gravity',
            'cup_history_diff', 'goalie_prestige_diff', 'home_brand_weight', 'home_coach_prestige',
            'home_cup_history', 'home_goalie_name_length', 'home_goalie_phonetic', 'home_goalie_prestige',
            'home_is_expansion', 'home_name_density', 'home_star_power', 'legendary_goalie_matchup',
            'original_six_matchup', 'playoff_nominative_amp', 'rivalry_nominative_boost',
            'star_power_diff', 'total_nominative_gravity'
        ]
        
        return perf + nom
    
    def comprehensive_feature_analysis(self, features: np.ndarray, outcomes: np.ndarray) -> Dict:
        """
        Comprehensive analysis of what features matter.
        Uses multiple methods to triangulate truth.
        """
        print("\n🔬 COMPREHENSIVE FEATURE ANALYSIS")
        print("="*80)
        
        results = {}
        
        # 1. Random Forest importance
        print("\n1️⃣  Random Forest Feature Importance")
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(features, outcomes)
        rf_importance = rf.feature_importances_
        
        # 2. Gradient Boosting importance  
        print("2️⃣  Gradient Boosting Feature Importance")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(features, outcomes)
        gb_importance = gb.feature_importances_
        
        # 3. Correlation analysis
        print("3️⃣  Correlation Analysis")
        correlations = []
        for i in range(features.shape[1]):
            try:
                r, p = pearsonr(features[:, i], outcomes)
                if not np.isnan(r):
                    correlations.append((i, r, p))
            except:
                correlations.append((i, 0, 1))
        
        # Combine importance scores (ensemble of methods)
        combined_importance = (rf_importance + gb_importance) / 2
        
        # Get top features
        top_indices = np.argsort(combined_importance)[::-1][:20]
        
        print("\n🏆 TOP 20 FEATURES (Ensemble Importance)")
        print("-"*80)
        
        top_features_list = []
        for i, idx in enumerate(top_indices, 1):
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f'Feature_{idx}'
            rf_imp = rf_importance[idx]
            gb_imp = gb_importance[idx]
            combined = combined_importance[idx]
            
            # Get correlation
            corr_data = [c for c in correlations if c[0] == idx]
            r = corr_data[0][1] if corr_data else 0
            
            print(f"   {i:2d}. {feat_name:40s} = {combined:.4f} (RF:{rf_imp:.4f}, GB:{gb_imp:.4f}, r:{r:+.3f})")
            
            top_features_list.append({
                'rank': i,
                'feature': feat_name,
                'feature_idx': int(idx),
                'rf_importance': float(rf_imp),
                'gb_importance': float(gb_imp),
                'combined_importance': float(combined),
                'correlation': float(r),
            })
        
        # Categorize by type
        nominative_count = sum(1 for f in top_features_list[:10] if any(
            nom in f['feature'] for nom in ['brand', 'cup', 'gravity', 'expansion', 'six', 'star', 'goalie_prestige', 'prestige']
        ))
        
        print(f"\n📊 Top 10 Feature Composition:")
        print(f"   Nominative: {nominative_count}/10 ({nominative_count*10}%)")
        print(f"   Performance: {10-nominative_count}/10 ({(10-nominative_count)*10}%)")
        
        results['top_features'] = top_features_list
        results['nominative_dominance'] = nominative_count / 10
        results['rf_importance'] = rf_importance
        results['gb_importance'] = gb_importance
        results['correlations'] = [(int(c[0]), float(c[1]), float(c[2])) for c in correlations]
        
        return results
    
    def build_meta_ensemble_model(self, features: np.ndarray, outcomes: np.ndarray) -> VotingClassifier:
        """
        Build meta-ensemble combining RF, GB, and Logistic.
        This learns from NBA's "Ensemble Narrative" success.
        """
        print("\n🎭 BUILDING META-ENSEMBLE MODEL")
        print("-"*80)
        
        # Create voting ensemble
        voting = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('gb', self.models['gradient_boosting']),
                ('lr', self.models['logistic']),
            ],
            voting='soft',  # Use probabilities
            weights=[2, 3, 1]  # GB gets highest weight (found best patterns)
        )
        
        # Fit ensemble
        voting.fit(features, outcomes)
        
        # Cross-validation
        cv_scores = cross_val_score(voting, features, outcomes, cv=5, scoring='accuracy')
        
        print(f"   ✓ Meta-ensemble trained")
        print(f"   Cross-val accuracy: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
        
        return voting
    
    def discover_ensemble_patterns(self, games: List[Dict], features: np.ndarray, 
                                  ensemble: VotingClassifier) -> List[Dict]:
        """Discover patterns using meta-ensemble predictions"""
        
        print("\n🎯 DISCOVERING META-ENSEMBLE PATTERNS")
        print("-"*80)
        
        patterns = []
        
        # Get ensemble predictions
        probs = ensemble.predict_proba(features)[:, 1]
        
        # Test multiple confidence thresholds
        for threshold in [0.65, 0.60, 0.55, 0.50, 0.45]:
            high_conf_games = [games[i] for i in range(len(games)) if probs[i] > threshold]
            
            if len(high_conf_games) >= self.min_sample:
                wins = sum(1 for g in high_conf_games if g.get('home_won', False))
                win_rate = wins / len(high_conf_games)
                roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
                
                if win_rate >= self.min_win_rate and roi >= self.min_roi:
                    pattern = {
                        'name': f'Meta-Ensemble Confidence ≥{threshold:.0%}',
                        'description': f'3-model voting ensemble predicts home win with {threshold:.0%}+ confidence',
                        'n_games': len(high_conf_games),
                        'wins': wins,
                        'losses': len(high_conf_games) - wins,
                        'win_rate': win_rate,
                        'win_rate_pct': win_rate * 100,
                        'roi': roi,
                        'roi_pct': roi * 100,
                        'confidence': 'VERY HIGH' if win_rate > 0.60 else 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                        'unit_recommendation': 3 if win_rate > 0.65 else 2 if win_rate > 0.57 else 1,
                        'pattern_type': 'meta_ensemble',
                        'model_threshold': threshold,
                    }
                    patterns.append(pattern)
                    print(f"   ✓ Threshold {threshold:.0%}: {len(high_conf_games)} games, {win_rate:.1%} win, {roi:.1%} ROI")
        
        return patterns
    
    def run_complete_analysis(self, games: List[Dict], features: np.ndarray) -> Dict:
        """Run complete analysis and generate all patterns"""
        
        print("\n" + "="*80)
        print("NHL COMPLETE INTEGRATED ANALYSIS")
        print("="*80)
        print(f"{len(games)} games × {features.shape[1]} features")
        
        outcomes = np.array([g.get('home_won', False) for g in games], dtype=int)
        
        # 1. Comprehensive feature analysis
        feature_analysis = self.comprehensive_feature_analysis(features, outcomes)
        
        # 2. Build meta-ensemble
        ensemble = self.build_meta_ensemble_model(features, outcomes)
        
        # 3. Discover ensemble patterns
        ensemble_patterns = self.discover_ensemble_patterns(games, features, ensemble)
        
        # 4. Load existing learned patterns
        project_root = Path(__file__).parent.parent.parent.parent
        learned_path = project_root / 'data' / 'domains' / 'nhl_betting_patterns_learned.json'
        with open(learned_path, 'r') as f:
            learned_patterns = json.load(f)
        
        # 5. Combine all patterns
        all_patterns = learned_patterns + ensemble_patterns
        
        # Sort by ROI
        all_patterns.sort(key=lambda x: x['roi'], reverse=True)
        
        # Create comprehensive results
        results = {
            'analysis_date': '2025-11-16',
            'n_games': len(games),
            'n_features': features.shape[1],
            'feature_analysis': feature_analysis,
            'patterns': all_patterns,
            'pattern_summary': {
                'total': len(all_patterns),
                'learned': len(learned_patterns),
                'ensemble': len(ensemble_patterns),
                'avg_win_rate': float(np.mean([p['win_rate'] for p in all_patterns])),
                'avg_roi': float(np.mean([p['roi'] for p in all_patterns])),
                'top_win_rate': float(all_patterns[0]['win_rate']) if all_patterns else 0,
                'top_roi': float(all_patterns[0]['roi']) if all_patterns else 0,
            }
        }
        
        return results


def main():
    """Main execution"""
    
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
    features_path = project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'nhl_features_complete.npz'
    output_dir = project_root / 'narrative_optimization' / 'domains' / 'nhl'
    
    # Load data
    print("\n📂 Loading data...")
    with open(data_path, 'r') as f:
        games = json.load(f)
    
    data = np.load(features_path)
    features = data['features']
    
    print(f"   ✓ {len(games)} games, {features.shape[1]} features")
    
    # Run complete analysis
    analyzer = NHLCompleteAnalyzer()
    results = analyzer.run_complete_analysis(games, features)
    
    # Save results
    output_path = output_dir / 'nhl_complete_analysis.json'
    
    # Convert numpy types for JSON
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    
    summary = results['pattern_summary']
    print(f"\nTotal patterns discovered: {summary['total']}")
    print(f"  - Data-driven: {summary['learned']}")
    print(f"  - Meta-ensemble: {summary['ensemble']}")
    
    print(f"\nPerformance:")
    print(f"  Average win rate: {summary['avg_win_rate']:.1%}")
    print(f"  Average ROI: {summary['avg_roi']:.1%}")
    print(f"  Top pattern win rate: {summary['top_win_rate']:.1%}")
    print(f"  Top pattern ROI: {summary['top_roi']:.1%}")
    
    print(f"\nFeature Analysis:")
    print(f"  Nominative dominance: {results['feature_analysis']['nominative_dominance']:.0%}")
    
    print(f"\n💾 Complete analysis saved: {output_path}")
    
    # Print top 10 patterns
    print("\n🏆 TOP 10 PATTERNS")
    print("-"*80)
    for i, p in enumerate(results['patterns'][:10], 1):
        print(f"{i:2d}. {p['name']:50s} | {p['n_games']:3d} games | {p['win_rate_pct']:5.1f}% | ROI: {p['roi_pct']:5.1f}%")
    
    print("\n✅ COMPLETE ANALYSIS FINISHED!")


if __name__ == "__main__":
    main()

