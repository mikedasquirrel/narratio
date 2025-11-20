"""
Build MLB Betting Model with Archetype Features

MLB has HIGHEST journey completion (13.5%) of any sport!
This makes it potentially the best narrative betting domain.

Builds model from scratch integrating archetype features.
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import json

def build_mlb_model():
    print("="*70)
    print("MLB BETTING MODEL - WITH ARCHETYPE FEATURES")
    print("="*70)
    
    # Load MLB archetype features
    features_dir = Path('narrative_optimization/data/archetype_features/mlb')
    
    if not features_dir.exists():
        print("❌ MLB features not found")
        return
    
    # Note: hero_journey had error, so we'll skip it
    character = np.load(features_dir / 'character_features.npz', allow_pickle=True)
    plot = np.load(features_dir / 'plot_features.npz', allow_pickle=True)
    structural = np.load(features_dir / 'structural_features.npz', allow_pickle=True)
    thematic = np.load(features_dir / 'thematic_features.npz', allow_pickle=True)
    
    print(f"\n✅ Loaded MLB features ({len(plot['outcomes'])} games)")
    
    # Extract numeric outcomes from dict
    outcomes_raw = plot['outcomes']
    if len(outcomes_raw) > 0 and isinstance(outcomes_raw[0], dict):
        # Extract winner as binary (1=home, 0=away)
        outcomes = np.array([1 if o.get('winner') == 'home' else 0 for o in outcomes_raw])
    else:
        outcomes = outcomes_raw
    
    print(f"   Outcomes: win rate={outcomes.mean():.1%}")
    
    # Select key features
    print("\nSelecting MLB-specific archetype features:")
    
    features = {
        'magician_archetype': character['features'][:, 10],  # Clutch magic
        'quest_completion': plot['features'][:, 7],
        'structure_quality': structural['features'][:, -1],
        'comedy_mythos': thematic['features'][:, 0],  # Baseball = comedy
        'ruler_archetype': character['features'][:, 11],  # Control/mastery
        'warrior_archetype': character['features'][:, 2],  # Competition
    }
    
    for name, feature in features.items():
        print(f"   {name}: mean={feature.mean():.3f}, std={feature.std():.3f}")
    
    X = np.column_stack(list(features.values()))
    print(f"\n   Total features: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, outcomes, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining MLB archetype model...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"\n   Train R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, outcomes, cv=5, scoring='r2', n_jobs=-1)
    print(f"   CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance
    print("\nFeature Importance:")
    for name, importance in sorted(zip(features.keys(), model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"   {name}: {importance:.3f}")
    
    # Save model components
    output = Path('narrative_optimization/results/mlb_archetype_model.npz')
    np.savez(
        output,
        feature_matrix=X,
        outcomes=outcomes,
        feature_names=list(features.keys()),
        feature_importances=model.feature_importances_,
        test_r2=test_r2,
        cv_r2_mean=cv_scores.mean()
    )
    
    print(f"\n✅ MLB model saved: {output}")
    print(f"\nMLB Archetype Model Performance:")
    print(f"   R²: {cv_scores.mean():.1%}")
    print(f"   Note: Archetype-only baseline - add traditional stats for full model")
    
    print("\n" + "="*70)
    print("MLB MODEL COMPLETE")
    print("="*70)

if __name__ == '__main__':
    build_mlb_model()

