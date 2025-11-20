"""
NHL Focused Temporal Model Training

Trains models on baseline + 10 focused temporal features (910 total).

Strategy: Add ONLY non-redundant temporal features that capture underpriced narratives.

Expected improvement: 69.4% → 71-73% win rate, 32.5% → 36-40% ROI

Author: Focused Temporal Training System
Date: November 19, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.domains.nhl.focused_temporal_features import FocusedTemporalExtractor


def load_and_prepare_data():
    """Load baseline dataset and add focused temporal features"""
    print("\n[Data Preparation] Loading baseline dataset...")
    
    # Load baseline (900 features)
    baseline_df = pd.read_parquet('narrative_optimization/domains/nhl/nhl_narrative_betting_dataset.parquet')
    print(f"✓ Baseline: {baseline_df.shape}")
    
    # Convert to game dicts for temporal extraction
    print(f"\n[Temporal Extraction] Adding 10 focused features...")
    games = []
    for idx, row in baseline_df.iterrows():
        game = {
            'game_id': row.get('game_id'),
            'date': row.get('date'),
            'season': row.get('season'),
            'home_team': row.get('home_team'),
            'away_team': row.get('away_team'),
            'home_won': row.get('home_won'),
            'home_score': row.get('home_score', 0),
            'away_score': row.get('away_score', 0),
            'temporal_context': {
                'home_rest_days': row.get('ctx_home_rest_days', 1),
                'away_rest_days': row.get('ctx_away_rest_days', 1),
            }
        }
        games.append(game)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Converted {idx + 1:,} games...")
    
    # Group by season for context
    season_groups = {}
    for game in games:
        season = game['season']
        if season not in season_groups:
            season_groups[season] = []
        season_groups[season].append(game)
    
    print(f"✓ Grouped into {len(season_groups)} seasons")
    
    # Extract focused temporal features
    extractor = FocusedTemporalExtractor()
    focused_features = []
    
    print(f"\n[Extraction] Processing {len(games):,} games...")
    for idx, game in enumerate(games):
        season = game['season']
        game_date = game['date']
        
        season_context = [g for g in season_groups[season] if g['date'] < game_date]
        temporal = extractor.extract_features(game, season_context)
        focused_features.append(temporal)
        
        if (idx + 1) % 2000 == 0:
            print(f"  [{idx + 1:,}/{len(games):,}] {(idx + 1) / len(games) * 100:.1f}%")
    
    focused_df = pd.DataFrame(focused_features)
    print(f"\n✓ Extracted focused temporal features: {focused_df.shape}")
    
    # Combine with baseline
    combined_df = pd.concat([baseline_df, focused_df], axis=1)
    print(f"✓ Combined dataset: {combined_df.shape} (900 baseline + 10 focused temporal)")
    
    return combined_df


def train_and_evaluate(df: pd.DataFrame):
    """Train models and evaluate performance"""
    print(f"\n{'='*80}")
    print("MODEL TRAINING & EVALUATION")
    print(f"{'='*80}")
    
    # Temporal holdout
    df = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\nTemporal holdout:")
    print(f"  Train: {len(train_df):,} games")
    print(f"  Test:  {len(test_df):,} games")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['game_id', 'season', 'date', 'home_team', 'away_team', 'home_won', 'home_score', 'away_score', 'home_goalie', 'away_goalie']]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['home_won'].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['home_won'].astype(int)
    
    print(f"\nFeature matrix: {X_train.shape[1]:,} features")
    
    # Train
    print(f"\n[Training] Building models...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print(f"✓ All models trained")
    
    # Evaluate meta-ensemble
    print(f"\n[Evaluation] Testing meta-ensemble...")
    
    meta_proba = (
        lr.predict_proba(X_test_scaled)[:, 1] * 0.3 +
        gb.predict_proba(X_test)[:, 1] * 0.4 +
        rf.predict_proba(X_test)[:, 1] * 0.3
    )
    meta_pred = (meta_proba >= 0.5).astype(int)
    
    meta_acc = accuracy_score(y_test, meta_pred)
    meta_auc = roc_auc_score(y_test, meta_proba)
    
    print(f"  Overall Test Accuracy: {meta_acc:.1%}")
    print(f"  Overall Test AUC:      {meta_auc:.3f}")
    
    # Betting tiers
    print(f"\n{'='*80}")
    print("BETTING PERFORMANCE BY CONFIDENCE TIER")
    print(f"{'='*80}")
    
    results = {}
    
    for threshold, label in [(0.60, "High-Confidence"), (0.65, "Ultra-Confident"), (0.70, "Elite")]:
        confident = meta_proba >= threshold
        if confident.sum() > 0:
            conf_acc = accuracy_score(y_test[confident], meta_pred[confident])
            n_bets = confident.sum()
            wins = (meta_pred[confident] == y_test[confident]).sum()
            losses = n_bets - wins
            roi = ((wins * 0.91 - losses) / n_bets) * 100
            
            print(f"\n{label} (≥{threshold:.0%}):")
            print(f"  Bets:     {n_bets} ({n_bets/len(y_test)*100:.1f}% of test set)")
            print(f"  Win Rate: {conf_acc:.1%}")
            print(f"  ROI:      {roi:+.1f}%")
            print(f"  Record:   {wins}-{losses}")
            
            results[label] = {
                'threshold': threshold,
                'bets': int(n_bets),
                'win_rate': float(conf_acc),
                'roi': float(roi),
                'record': f"{wins}-{losses}"
            }
    
    # Save models
    print(f"\n[Saving] Writing models...")
    models_dir = Path('narrative_optimization/domains/nhl/models/focused_temporal')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, models_dir / 'scaler.pkl')
    joblib.dump(lr, models_dir / 'focused_logistic.pkl')
    joblib.dump(gb, models_dir / 'focused_gradient.pkl')
    joblib.dump(rf, models_dir / 'focused_forest.pkl')
    
    metadata = {
        'trained_date': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'baseline_features': 900,
        'focused_temporal_features': 10,
        'test_accuracy': float(meta_acc),
        'test_auc': float(meta_auc),
        'betting_performance': results,
        'feature_columns': feature_cols
    }
    
    with open(models_dir / 'models_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved to {models_dir}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON TO BASELINE")
    print(f"{'='*80}")
    print(f"\nBaseline (900 features, Nov 17):")
    print(f"  Ultra-Confident: 69.4% win rate, +32.5% ROI (59-26)")
    print(f"\nFocused Temporal (910 features, Nov 19):")
    if 'Ultra-Confident' in results:
        print(f"  Ultra-Confident: {results['Ultra-Confident']['win_rate']:.1%} win rate, {results['Ultra-Confident']['roi']:+.1f}% ROI ({results['Ultra-Confident']['record']})")
        improvement = results['Ultra-Confident']['win_rate'] - 0.694
        roi_improvement = results['Ultra-Confident']['roi'] - 32.5
        print(f"\nImprovement: {improvement:+.1%} win rate, {roi_improvement:+.1f}% ROI")
    
    print(f"\n{'='*80}")
    
    return results


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("NHL FOCUSED TEMPORAL MODEL TRAINING")
    print("="*80)
    
    # Prepare data
    df = load_and_prepare_data()
    
    # Train and evaluate
    results = train_and_evaluate(df)
    
    print("\nTRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

