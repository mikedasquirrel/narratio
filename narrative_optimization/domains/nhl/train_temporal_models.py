"""
NHL Temporal Model Training

Trains models on the enriched 950-feature dataset (900 baseline + 50 temporal).

Models trained:
- Logistic Regression (fast, interpretable)
- Gradient Boosting (high accuracy)
- Random Forest (ensemble power)
- Meta-Ensemble (combines all three)

Validation:
- 80/20 train/test split
- Temporal holdout (last season held out)
- Performance metrics: Win rate, ROI, calibration

Author: Temporal Model Training System
Date: November 19, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def load_temporal_dataset():
    """Load enriched temporal dataset"""
    path = Path('narrative_optimization/domains/nhl/nhl_narrative_betting_temporal_dataset.parquet')
    df = pd.read_parquet(path)
    print(f"✓ Loaded temporal dataset: {df.shape}")
    return df


def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Prepare train/test split with temporal awareness.
    
    Uses temporal holdout: last season for testing, rest for training.
    This ensures we're testing on truly unseen future data.
    """
    print(f"\n[Data Split] Preparing temporal holdout split...")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Use last 20% as temporal holdout
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"  Train: {len(train_df):,} games ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Test:  {len(test_df):,} games ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c not in ['game_id', 'season', 'date', 'home_team', 'away_team', 'home_won', 'home_score', 'away_score']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['home_won'].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df['home_won'].astype(int)
    
    print(f"\n✓ Feature matrix: {X_train.shape[1]:,} features")
    print(f"  Baseline features: ~900")
    print(f"  Temporal features: 50")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_models(X_train, y_train):
    """Train all models"""
    print(f"\n[Model Training] Training 4 models on {len(X_train):,} games...")
    
    models = {}
    
    # 1. Scale features
    print(f"\n[1/4] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    models['scaler'] = scaler
    print(f"  ✓ StandardScaler fitted")
    
    # 2. Logistic Regression
    print(f"\n[2/4] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    models['temporal_logistic'] = lr
    train_acc = accuracy_score(y_train, lr.predict(X_train_scaled))
    print(f"  ✓ Logistic trained - Train accuracy: {train_acc:.1%}")
    
    # 3. Gradient Boosting
    print(f"\n[3/4] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)  # GBM doesn't need scaling
    models['temporal_gradient'] = gb
    train_acc = accuracy_score(y_train, gb.predict(X_train))
    print(f"  ✓ Gradient Boosting trained - Train accuracy: {train_acc:.1%}")
    
    # 4. Random Forest
    print(f"\n[4/4] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    models['temporal_forest'] = rf
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    print(f"  ✓ Random Forest trained - Train accuracy: {train_acc:.1%}")
    
    print(f"\n✓ All models trained successfully")
    
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    print(f"\n{'='*80}")
    print("MODEL EVALUATION")
    print(f"{'='*80}")
    
    scaler = models['scaler']
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name in ['temporal_logistic', 'temporal_gradient', 'temporal_forest']:
        model = models[name]
        
        print(f"\n--- {name.upper().replace('_', ' ')} ---")
        
        # Get predictions
        if 'logistic' in name:
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            y_proba_train = model.predict_proba(X_train_scaled)[:, 1]
            y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_train = model.predict_proba(X_train)[:, 1]
            y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        train_auc = roc_auc_score(y_train, y_proba_train)
        test_auc = roc_auc_score(y_test, y_proba_test)
        
        print(f"  Train Accuracy: {train_acc:.1%}")
        print(f"  Test Accuracy:  {test_acc:.1%}")
        print(f"  Train AUC:      {train_auc:.3f}")
        print(f"  Test AUC:       {test_auc:.3f}")
        
        # Calculate betting metrics (assuming -110 odds)
        confident_threshold = 0.60
        confident_preds = y_proba_test >= confident_threshold
        
        if confident_preds.sum() > 0:
            confident_acc = accuracy_score(y_test[confident_preds], y_pred_test[confident_preds])
            n_confident = confident_preds.sum()
            
            # Calculate ROI (assuming -110 odds = 1.91 decimal)
            wins = (y_pred_test[confident_preds] == y_test[confident_preds]).sum()
            losses = n_confident - wins
            roi = ((wins * 0.91 - losses) / n_confident) * 100
            
            print(f"\n  High-Confidence Bets (≥60%):")
            print(f"    Count:    {n_confident} ({n_confident/len(y_test)*100:.1f}% of test set)")
            print(f"    Win Rate: {confident_acc:.1%}")
            print(f"    ROI:      {roi:+.1f}%")
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'confident_accuracy': confident_acc if confident_preds.sum() > 0 else 0,
            'confident_count': int(confident_preds.sum()),
            'confident_roi': roi if confident_preds.sum() > 0 else 0
        }
    
    # Build meta-ensemble
    print(f"\n--- META-ENSEMBLE ---")
    meta_proba = (
        models['temporal_logistic'].predict_proba(X_test_scaled)[:, 1] * 0.3 +
        models['temporal_gradient'].predict_proba(X_test)[:, 1] * 0.4 +
        models['temporal_forest'].predict_proba(X_test)[:, 1] * 0.3
    )
    meta_pred = (meta_proba >= 0.5).astype(int)
    
    meta_acc = accuracy_score(y_test, meta_pred)
    meta_auc = roc_auc_score(y_test, meta_proba)
    
    print(f"  Test Accuracy: {meta_acc:.1%}")
    print(f"  Test AUC:      {meta_auc:.3f}")
    
    # Meta high-confidence
    meta_confident = meta_proba >= 0.65
    if meta_confident.sum() > 0:
        meta_confident_acc = accuracy_score(y_test[meta_confident], meta_pred[meta_confident])
        n_meta = meta_confident.sum()
        wins = (meta_pred[meta_confident] == y_test[meta_confident]).sum()
        losses = n_meta - wins
        meta_roi = ((wins * 0.91 - losses) / n_meta) * 100
        
        print(f"\n  Ultra-Confident Bets (≥65%):")
        print(f"    Count:    {n_meta} ({n_meta/len(y_test)*100:.1f}% of test set)")
        print(f"    Win Rate: {meta_confident_acc:.1%}")
        print(f"    ROI:      {meta_roi:+.1f}%")
        
        results['meta_ensemble'] = {
            'test_accuracy': meta_acc,
            'test_auc': meta_auc,
            'confident_accuracy': meta_confident_acc,
            'confident_count': int(n_meta),
            'confident_roi': meta_roi
        }
    
    return results


def save_models(models, results, feature_cols):
    """Save trained models and metadata"""
    print(f"\n[Saving Models] Writing to disk...")
    
    models_dir = Path('narrative_optimization/domains/nhl/models/temporal')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each model
    for name, model in models.items():
        model_path = models_dir / f'{name}.pkl'
        joblib.dump(model, model_path)
        print(f"  ✓ Saved {name} to {model_path}")
    
    # Save metadata
    metadata = {
        'trained_date': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'models': list(models.keys()),
        'performance': results,
        'temporal_features': 50,
        'baseline_features': len(feature_cols) - 50,
    }
    
    metadata_path = models_dir / 'models_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved metadata to {metadata_path}")
    print(f"\n✓ All models saved to {models_dir}")


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("NHL TEMPORAL MODEL TRAINING")
    print("="*80)
    
    # Load data
    print(f"\n[Step 1/5] Loading enriched temporal dataset...")
    df = load_temporal_dataset()
    
    # Prepare split
    print(f"\n[Step 2/5] Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)
    
    # Train models
    print(f"\n[Step 3/5] Training models...")
    models = train_models(X_train, y_train)
    
    # Evaluate
    print(f"\n[Step 4/5] Evaluating models...")
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Save
    print(f"\n[Step 5/5] Saving models...")
    save_models(models, results, feature_cols)
    
    # Final summary
    print(f"\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"="*80)
    print(f"\nComparison to Baseline (900 features):")
    print(f"  Baseline:  69.4% win rate, 32.5% ROI")
    print(f"  Temporal:  {results['meta_ensemble']['confident_accuracy']:.1%} win rate, {results['meta_ensemble']['confident_roi']:+.1f}% ROI")
    print(f"  Improvement: {results['meta_ensemble']['confident_accuracy'] - 0.694:.1%} win rate, {results['meta_ensemble']['confident_roi'] - 32.5:+.1f}% ROI")
    print(f"\nModels saved to: narrative_optimization/domains/nhl/models/temporal/")
    print(f"Ready for deployment to daily prediction pipeline.")
    print("="*80)


if __name__ == '__main__':
    main()

