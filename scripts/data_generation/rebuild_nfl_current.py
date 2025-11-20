"""
NFL Model Rebuild with Current QB Data
========================================

Properly rebuilds the NFL betting model using CURRENT QB/coach statistics
from 2020-2023, then tests on 2024 holdout data.

This tests whether narrative patterns (QB prestige differential) are truly
transposable when measured correctly in both training and test periods.

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from datetime import datetime


class NFLModelRebuilder:
    """Rebuild NFL model with current data"""
    
    def __init__(self):
        """Initialize rebuilder"""
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / 'data' / 'domains'
        
    def calculate_qb_prestige_current(self, games: list, min_games: int = 5) -> Dict:
        """
        Calculate QB prestige from actual game results.
        
        Parameters
        ----------
        games : list
            Games to calculate from
        min_games : int
            Minimum games to include QB
            
        Returns
        -------
        qb_stats : dict
            QB name -> {'games': int, 'wins': int, 'win_rate': float}
        """
        print("\n[Calculating QB Prestige]")
        
        qb_records = defaultdict(lambda: {'games': 0, 'wins': 0})
        
        for game in games:
            # Extract QB names from roster or direct field
            if 'home_roster' in game and game['home_roster']:
                home_qb = game['home_roster'].get('starting_qb', {}).get('name', '')
            else:
                home_qb = game.get('home_qb_name', '')
            
            if 'away_roster' in game and game['away_roster']:
                away_qb = game['away_roster'].get('starting_qb', {}).get('name', '')
            else:
                away_qb = game.get('away_qb_name', '')
            
            # Home team result
            home_won = game.get('home_won', False)
            if not isinstance(home_won, bool):
                # Fallback to score differential
                home_score = game.get('home_score', 0)
                away_score = game.get('away_score', 0)
                home_won = home_score > away_score
            
            if home_qb:
                qb_records[home_qb]['games'] += 1
                if home_won:
                    qb_records[home_qb]['wins'] += 1
            
            if away_qb:
                qb_records[away_qb]['games'] += 1
                if not home_won:
                    qb_records[away_qb]['wins'] += 1
        
        # Calculate win rates and filter by min games
        qb_stats = {}
        for qb, record in qb_records.items():
            if record['games'] >= min_games:
                win_rate = record['wins'] / record['games']
                qb_stats[qb] = {
                    'games': record['games'],
                    'wins': record['wins'],
                    'win_rate': win_rate
                }
        
        print(f"  ✓ Calculated prestige for {len(qb_stats)} QBs")
        print(f"  Top 5 QBs by win rate:")
        top_qbs = sorted(qb_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:5]
        for qb, stats in top_qbs:
            print(f"    {qb:25s}: {stats['win_rate']:.3f} ({stats['wins']}/{stats['games']})")
        
        return qb_stats
    
    def calculate_coach_prestige_current(self, games: list, min_games: int = 10) -> Dict:
        """Calculate coach prestige from actual game results"""
        print("\n[Calculating Coach Prestige]")
        
        coach_records = defaultdict(lambda: {'games': 0, 'wins': 0})
        
        for game in games:
            # Extract coach names
            if 'home_coaches' in game and game['home_coaches']:
                home_coach = game['home_coaches'].get('head_coach', '')
            else:
                home_coach = game.get('home_coach', '')
            
            if 'away_coaches' in game and game['away_coaches']:
                away_coach = game['away_coaches'].get('head_coach', '')
            else:
                away_coach = game.get('away_coach', '')
            
            # Home team result
            home_won = game.get('home_won', False)
            if not isinstance(home_won, bool):
                home_score = game.get('home_score', 0)
                away_score = game.get('away_score', 0)
                home_won = home_score > away_score
            
            if home_coach:
                coach_records[home_coach]['games'] += 1
                if home_won:
                    coach_records[home_coach]['wins'] += 1
            
            if away_coach:
                coach_records[away_coach]['games'] += 1
                if not home_won:
                    coach_records[away_coach]['wins'] += 1
        
        coach_stats = {}
        for coach, record in coach_records.items():
            if record['games'] >= min_games:
                win_rate = record['wins'] / record['games']
                coach_stats[coach] = {
                    'games': record['games'],
                    'wins': record['wins'],
                    'win_rate': win_rate
                }
        
        print(f"  ✓ Calculated prestige for {len(coach_stats)} coaches")
        print(f"  Top 5 coaches by win rate:")
        top_coaches = sorted(coach_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:5]
        for coach, stats in top_coaches:
            print(f"    {coach:25s}: {stats['win_rate']:.3f} ({stats['wins']}/{stats['games']})")
        
        return coach_stats
    
    def extract_features(self, games: list, qb_stats: Dict, coach_stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 29 nominative features from games.
        
        Returns X (features) and y (outcomes)
        """
        features = []
        outcomes = []
        
        for game in games:
            # Get QB names
            if 'home_roster' in game and game['home_roster']:
                home_qb = game['home_roster'].get('starting_qb', {}).get('name', '')
            else:
                home_qb = game.get('home_qb_name', '')
            
            if 'away_roster' in game and game['away_roster']:
                away_qb = game['away_roster'].get('starting_qb', {}).get('name', '')
            else:
                away_qb = game.get('away_qb_name', '')
            
            # Get coach names
            if 'home_coaches' in game and game['home_coaches']:
                home_coach = game['home_coaches'].get('head_coach', '')
            else:
                home_coach = game.get('home_coach', '')
            
            if 'away_coaches' in game and game['away_coaches']:
                away_coach = game['away_coaches'].get('head_coach', '')
            else:
                away_coach = game.get('away_coach', '')
            
            # QB prestige (from current stats or default)
            if home_qb in qb_stats:
                qb_home_prestige = qb_stats[home_qb]['win_rate']
            else:
                qb_home_prestige = 0.5
            
            if away_qb in qb_stats:
                qb_away_prestige = qb_stats[away_qb]['win_rate']
            else:
                qb_away_prestige = 0.5
            
            # Coach prestige
            if home_coach in coach_stats:
                coach_home_prestige = coach_stats[home_coach]['win_rate']
            else:
                coach_home_prestige = 0.5
            
            if away_coach in coach_stats:
                coach_away_prestige = coach_stats[away_coach]['win_rate']
            else:
                coach_away_prestige = 0.5
            
            # Calculate derived features
            qb_diff = qb_home_prestige - qb_away_prestige
            qb_max = max(qb_home_prestige, qb_away_prestige)
            qb_min = min(qb_home_prestige, qb_away_prestige)
            qb_product = qb_home_prestige * qb_away_prestige
            
            coach_diff = coach_home_prestige - coach_away_prestige
            coach_max = max(coach_home_prestige, coach_away_prestige)
            coach_product = coach_home_prestige * coach_away_prestige
            coach_home_elite = 1.0 if coach_home_prestige > 0.7 else 0.0
            coach_away_elite = 1.0 if coach_away_prestige > 0.7 else 0.0
            coach_home_exp = coach_home_prestige
            coach_away_exp = coach_away_prestige
            
            # O-line and star features (defaults for now)
            oline_home = 0.5
            oline_away = 0.5
            oline_diff = 0.0
            oline_product = 0.25
            
            star_home = 0.5
            star_away = 0.5
            star_diff = 0.0
            star_product = 0.25
            
            # Interaction features
            qb_coach_interaction = qb_home_prestige * coach_home_prestige
            elite_qb_elite_coach = 1.0 if (qb_home_prestige > 0.7 and coach_home_prestige > 0.7) else 0.0
            experience_mismatch = abs(coach_home_exp - coach_away_exp)
            prestige_total = qb_home_prestige + qb_away_prestige + coach_home_prestige + coach_away_prestige
            prestige_variance = np.var([qb_home_prestige, qb_away_prestige, coach_home_prestige, coach_away_prestige])
            ensemble_quality = (qb_home_prestige + coach_home_prestige + oline_home) / 3.0
            
            # Build feature vector
            feat = [
                qb_home_prestige, qb_away_prestige, qb_diff, qb_max, qb_min, qb_product,
                coach_home_prestige, coach_away_prestige, coach_diff, coach_max, coach_product,
                coach_home_elite, coach_away_elite, coach_home_exp, coach_away_exp,
                oline_home, oline_away, oline_diff, oline_product,
                star_home, star_away, star_diff, star_product,
                qb_coach_interaction, elite_qb_elite_coach, experience_mismatch,
                prestige_total, prestige_variance, ensemble_quality
            ]
            
            features.append(feat)
            
            # Outcome (home covered spread)
            if 'betting_odds' in game and game['betting_odds']:
                home_covered = game['betting_odds'].get('home_covered_spread', False)
            else:
                # Fallback: calculate from scores and spread
                home_score = game.get('home_score', 0)
                away_score = game.get('away_score', 0)
                spread = game.get('betting_odds', {}).get('spread', 0)
                if spread == 0:
                    spread = game.get('spread_line', 0)
                result = home_score - away_score
                home_covered = (result + spread) > 0
            
            outcomes.append(1 if home_covered else 0)
        
        return np.array(features), np.array(outcomes)
    
    def rebuild(self):
        """Main rebuild process"""
        print(f"\n{'='*80}")
        print("NFL MODEL REBUILD - WITH CURRENT QB DATA")
        print('='*80)
        print("\nThis tests whether QB prestige patterns are truly transposable")
        print("when measured correctly in both training and test periods.")
        
        # Load complete dataset
        print(f"\n[1/6] Loading NFL data...")
        with open(self.data_path / 'nfl_complete_dataset.json') as f:
            all_games = json.load(f)
        
        print(f"  ✓ Loaded {len(all_games)} games (2014-2024)")
        
        # Split by season
        train_games = [g for g in all_games if 2020 <= g.get('season', 0) <= 2023]
        test_games = [g for g in all_games if g.get('season', 0) == 2024]
        
        print(f"\n[2/6] Split data:")
        print(f"  Training: 2020-2023 ({len(train_games)} games)")
        print(f"  Testing:  2024      ({len(test_games)} games)")
        
        # Calculate current QB/coach prestige from training data
        print(f"\n[3/6] Calculating prestige from 2020-2023 data...")
        qb_stats = self.calculate_qb_prestige_current(train_games, min_games=10)
        coach_stats = self.calculate_coach_prestige_current(train_games, min_games=15)
        
        # Extract features
        print(f"\n[4/6] Extracting features...")
        X_train, y_train = self.extract_features(train_games, qb_stats, coach_stats)
        X_test, y_test = self.extract_features(test_games, qb_stats, coach_stats)
        
        print(f"  ✓ Training features: {X_train.shape}")
        print(f"  ✓ Test features: {X_test.shape}")
        print(f"  ✓ Baseline (home covers): Train={y_train.mean():.1%}, Test={y_test.mean():.1%}")
        
        # Check feature distribution in test set
        print(f"\n[Feature Quality Check]")
        qb_diffs_train = X_train[:, 2]  # qb_diff column
        qb_diffs_test = X_test[:, 2]
        print(f"  QB differential (train): mean={qb_diffs_train.mean():.3f}, std={qb_diffs_train.std():.3f}")
        print(f"  QB differential (test):  mean={qb_diffs_test.mean():.3f}, std={qb_diffs_test.std():.3f}")
        
        if qb_diffs_test.std() < 0.05:
            print(f"  ⚠️  WARNING: Low variance in test set features!")
        else:
            print(f"  ✓ Good feature variance in test set")
        
        # Train model
        print(f"\n[5/6] Training model...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        print("  Training GradientBoostingClassifier...")
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation on training
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"  ✓ CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
        
        # Test on holdout 2024 season
        print(f"\n[6/6] Testing on 2024 season...")
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        test_acc = (y_pred == y_test).mean()
        print(f"  ✓ Test Accuracy: {test_acc:.1%}")
        
        # Test at different confidence thresholds
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        
        print(f"\n{'='*80}")
        print("RESULTS BY CONFIDENCE THRESHOLD")
        print('='*80)
        print(f"\n{'Threshold':<12} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'ROI%':>7} {'Avg Conf':>9}")
        print('-' * 80)
        
        results = []
        
        for threshold in thresholds:
            # Filter by confidence
            confident_mask = (y_proba >= threshold) | (y_proba <= (1 - threshold))
            conf_indices = np.where(confident_mask)[0]
            
            if len(conf_indices) == 0:
                continue
            
            conf_preds = y_pred[conf_indices]
            conf_actual = y_test[conf_indices]
            conf_proba = y_proba[conf_indices]
            
            wins = (conf_preds == conf_actual).sum()
            losses = len(conf_preds) - wins
            win_rate = wins / len(conf_preds)
            
            # Calculate ROI
            roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
            
            print(f"≥{threshold:<10.0%} {len(conf_preds):>6} {wins:>6} {win_rate:>6.1%} {roi:>6.1%} {conf_proba.mean():>8.1%}")
            
            results.append({
                'threshold': threshold,
                'bets': int(len(conf_preds)),
                'wins': int(wins),
                'losses': int(losses),
                'win_rate': float(win_rate),
                'roi': float(roi),
                'avg_confidence': float(conf_proba.mean())
            })
        
        # Feature importances
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE")
        print('='*80)
        
        feature_names = [
            'qb_home_prestige', 'qb_away_prestige', 'qb_diff', 'qb_max', 'qb_min', 'qb_product',
            'coach_home_prestige', 'coach_away_prestige', 'coach_diff', 'coach_max', 'coach_product',
            'coach_home_elite', 'coach_away_elite', 'coach_home_exp', 'coach_away_exp',
            'oline_home', 'oline_away', 'oline_diff', 'oline_product',
            'star_home', 'star_away', 'star_diff', 'star_product',
            'qb_coach_interaction', 'elite_qb_elite_coach', 'experience_mismatch',
            'prestige_total', 'prestige_variance', 'ensemble_quality'
        ]
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 Features:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"  {i+1:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
        
        # Save rebuilt model
        print(f"\n{'='*80}")
        print("SAVING REBUILT MODEL")
        print('='*80)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'qb_stats': qb_stats,
            'coach_stats': coach_stats,
            'feature_names': feature_names,
            'training_period': '2020-2023',
            'test_period': '2024',
            'test_accuracy': float(test_acc),
            'results': results,
            'rebuild_date': datetime.now().isoformat()
        }
        
        output_path = self.base_path / 'narrative_optimization' / 'nfl_rebuilt_current.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  ✓ Saved to: {output_path}")
        
        # Summary
        print(f"\n{'='*80}")
        print("REBUILD COMPLETE - VERDICT")
        print('='*80)
        
        best_result = max(results, key=lambda x: x['roi'])
        
        print(f"\nBest Performance:")
        print(f"  Threshold: ≥{best_result['threshold']:.0%}")
        print(f"  Win Rate: {best_result['win_rate']:.1%}")
        print(f"  ROI: {best_result['roi']:.1%}")
        print(f"  Bets: {best_result['bets']}")
        
        if best_result['roi'] > 0.05:
            print(f"\n✅ PROFITABLE - Pattern is transposable with current data")
            print(f"   QB prestige differential DOES predict outcomes")
        elif best_result['win_rate'] > 0.52:
            print(f"\n⚠️  MARGINALLY PROFITABLE - Weak edge")
            print(f"   Pattern exists but market is efficient")
        else:
            print(f"\n❌ NOT PROFITABLE - Pattern doesn't hold")
            print(f"   Even with current data, no edge found")
        
        return model_data


def main():
    """Run NFL rebuild"""
    rebuilder = NFLModelRebuilder()
    model_data = rebuilder.rebuild()
    
    print(f"\n{'='*80}\n")
    
    return model_data


if __name__ == "__main__":
    main()

