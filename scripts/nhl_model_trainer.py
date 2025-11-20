"""
NHL Model Trainer - Production ML Models

Trains the Meta-Ensemble and GBM models on full dataset for deployment.
Saves trained models for fast daily predictions.

Models:
- Random Forest (200 trees)
- Gradient Boosting (100 estimators)
- Logistic Regression (regularized)
- Meta-Ensemble (voting)

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class NHLModelTrainer:
    """Train and save NHL prediction models"""
    
    def __init__(self):
        """Initialize trainer"""
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
    
    def load_data(self) -> tuple:
        """Load NHL games and features"""
        
        # Load games
        games_path = self.project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
        with open(games_path, 'r') as f:
            games = json.load(f)
        
        # Load features
        features_path = self.project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'nhl_features_complete.npz'
        data = np.load(features_path)
        features = data['features']
        
        # Extract outcomes
        outcomes = np.array([g.get('home_won', False) for g in games], dtype=int)
        
        print(f"📂 Loaded {len(games)} games with {features.shape[1]} features")
        
        return features, outcomes, games
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        
        print("\n🏋️  TRAINING MODELS")
        print("="*80)
        
        # 1. Random Forest
        print("\n1️⃣  Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   CV Accuracy: {cv_rf.mean():.1%} (±{cv_rf.std():.1%})")
        self.models['random_forest'] = rf
        
        # 2. Gradient Boosting
        print("\n2️⃣  Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42
        )
        gb.fit(X_train, y_train)
        cv_gb = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   CV Accuracy: {cv_gb.mean():.1%} (±{cv_gb.std():.1%})")
        self.models['gradient_boosting'] = gb
        
        # 3. Logistic Regression
        print("\n3️⃣  Logistic Regression...")
        lr = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            n_jobs=-1
        )
        lr.fit(X_train, y_train)
        cv_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   CV Accuracy: {cv_lr.mean():.1%} (±{cv_lr.std():.1%})")
        self.models['logistic'] = lr
        
        # 4. Meta-Ensemble
        print("\n4️⃣  Meta-Ensemble (Voting)...")
        meta = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr),
            ],
            voting='soft',
            weights=[2, 3, 1]  # GB gets most weight
        )
        meta.fit(X_train, y_train)
        cv_meta = cross_val_score(meta, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   CV Accuracy: {cv_meta.mean():.1%} (±{cv_meta.std():.1%})")
        self.models['meta_ensemble'] = meta
        
        print("\n✅ All models trained")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate on test set"""
        
        print("\n📊 TEST SET EVALUATION")
        print("="*80)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            print(f"\n{name}:")
            print(f"   Accuracy: {accuracy:.1%}")
            print(f"   AUC: {auc:.3f}")
            
            # High confidence predictions
            for threshold in [0.65, 0.60, 0.55]:
                high_conf_mask = y_proba > threshold
                if high_conf_mask.sum() > 0:
                    high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
                    print(f"   Confidence ≥{threshold:.0%}: {high_conf_acc:.1%} accuracy ({high_conf_mask.sum()} games)")
    
    def save_models(self):
        """Save trained models"""
        
        print("\n💾 SAVING MODELS")
        print("="*80)
        
        for name, model in self.models.items():
            model_path = self.models_dir / f'{name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✓ Saved {name} to {model_path.name}")
        
        # Save scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✓ Saved scaler")
        
        # Save metadata
        metadata = {
            'trained_date': datetime.now().isoformat(),
            'n_features': list(self.models.values())[0].n_features_in_,
            'models': list(self.models.keys()),
        }
        
        meta_path = self.models_dir / 'models_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Saved metadata")
    
    def train_full_pipeline(self):
        """Complete training pipeline"""
        
        print("\n" + "="*80)
        print("NHL MODEL TRAINING PIPELINE")
        print("="*80)
        
        # Load data
        X, y, games = self.load_data()
        
        # Scale features
        print("\n⚙️  Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Train: {len(X_train)} games")
        print(f"   Test: {len(X_test)} games")
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate
        self.evaluate_models(X_test, y_test)
        
        # Save
        self.save_models()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        print(f"\nModels saved to: {self.models_dir}")
        print("\nReady for daily predictions with trained models!")


def main():
    """Main execution"""
    
    trainer = NHLModelTrainer()
    trainer.train_full_pipeline()


if __name__ == "__main__":
    main()

