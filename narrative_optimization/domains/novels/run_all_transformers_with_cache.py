"""
Run All Transformers on Novels with Caching and Progress

Features:
- Caches each transformer's output
- Shows constant progress
- Resumes from checkpoint if interrupted
- Saves results incrementally
"""

import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import pickle

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.domains.novels.data_loader import NovelsDataLoader

# Import transformers
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
from narrative_optimization.src.transformers.phonetic import PhoneticTransformer
from narrative_optimization.src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from narrative_optimization.src.transformers.ensemble import EnsembleNarrativeTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class CachedTransformerRunner:
    """Run transformers with caching and progress tracking."""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.cache_dir = Path(__file__).parent / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.results_dir = Path(__file__).parent / 'results'
        self.results_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, transformer_name: str) -> Path:
        """Get cache file path for transformer."""
        return self.cache_dir / f"{transformer_name}_features.pkl"
    
    def load_cached_features(self, transformer_name: str):
        """Load cached features if available."""
        cache_path = self.get_cache_path(transformer_name)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_cached_features(self, transformer_name: str, features):
        """Save features to cache."""
        cache_path = self.get_cache_path(transformer_name)
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"    ✓ Cached {transformer_name} features")
    
    def run_transformer(self, name: str, transformer, texts: list, context: dict = None):
        """Run transformer with caching."""
        print(f"\n  [{name}]")
        print(f"    Checking cache...")
        
        # Try cache first
        cached = self.load_cached_features(name)
        if cached is not None:
            print(f"    ✓ Loaded from cache: {cached.shape}")
            return cached
        
        print(f"    Running transformer...")
        try:
            if context:
                features = transformer.fit_transform(texts, **context)
            else:
                features = transformer.fit_transform(texts)
            
            print(f"    ✓ Extracted: {features.shape}")
            self.save_cached_features(name, features)
            return features
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return None
    
    def run_all_transformers(self, texts: list, context: dict):
        """Run all transformers with caching."""
        print("\n" + "="*80)
        print("RUNNING ALL TRANSFORMERS WITH CACHING")
        print("="*80)
        
        all_features = []
        transformer_names = []
        
        # Define transformers
        transformers = [
            ('statistical', StatisticalTransformer(max_features=50)),
            ('nominative', NominativeAnalysisTransformer()),
            ('phonetic', PhoneticTransformer()),
            ('linguistic', LinguisticPatternsTransformer()),
            ('ensemble', EnsembleNarrativeTransformer()),
        ]
        
        print(f"\nRunning {len(transformers)} transformers on {len(texts)} texts...")
        
        for name, transformer in transformers:
            features = self.run_transformer(name, transformer, texts, context if name != 'statistical' else None)
            if features is not None and features.shape[0] == len(texts):
                all_features.append(features)
                transformer_names.append(name)
        
        if not all_features:
            print("\n❌ No features extracted!")
            return None, []
        
        # Combine features
        print(f"\n  Combining {len(all_features)} feature sets...")
        X = np.hstack(all_features)
        print(f"  ✓ Combined shape: {X.shape}")
        
        return X, transformer_names
    
    def train_and_evaluate(self, X, y, transformer_names):
        """Train model and evaluate."""
        print("\n" + "="*80)
        print("TRAINING AND EVALUATION")
        print("="*80)
        
        print(f"\nFeature matrix: {X.shape}")
        print(f"Outcome vector: {y.shape}")
        print(f"Transformers used: {', '.join(transformer_names)}")
        
        # Train model
        print("\nTraining Random Forest...")
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X, y)
        print("✓ Model trained")
        
        # Cross-validation
        print("\nCross-validating...")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        print(f"✓ CV R² scores: {cv_scores}")
        print(f"✓ Mean CV R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        # Feature importance
        print("\nCalculating feature importance...")
        importances = model.feature_importances_
        
        # Map back to transformers
        feature_start = 0
        transformer_importances = {}
        for i, name in enumerate(transformer_names):
            if i < len(all_features):
                n_features = all_features[i].shape[1]
                transformer_importances[name] = np.sum(importances[feature_start:feature_start + n_features])
                feature_start += n_features
        
        print("\nTransformer Importance:")
        for name, imp in sorted(transformer_importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.4f}")
        
        # Save results
        results = {
            'domain': self.domain_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'transformers': transformer_names,
            'cv_scores': cv_scores.tolist(),
            'mean_cv_r2': float(np.mean(cv_scores)),
            'std_cv_r2': float(np.std(cv_scores)),
            'transformer_importances': transformer_importances
        }
        
        results_path = self.results_dir / f'{self.domain_name}_analysis_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        
        return results


def main():
    print("="*80)
    print("NOVELS DOMAIN - TRANSFORMER ANALYSIS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n[1/4] Loading novels dataset...")
    loader = NovelsDataLoader()
    novels = loader.load_full_dataset()
    print(f"✓ Loaded {len(novels)} novels")
    
    # Extract data for transformers
    print("\n[2/4] Preparing data for transformers...")
    texts = [n['full_narrative'] for n in novels if n.get('full_narrative')]
    outcomes = np.array([n['success_score'] for n in novels if n.get('full_narrative')])
    
    # Context for nominative transformers
    context = {
        'author_names': [n.get('author_name', '') for n in novels if n.get('full_narrative')],
        'titles': [n.get('book_title', '') for n in novels if n.get('full_narrative')],
        'character_names': [n.get('character_names', []) for n in novels if n.get('full_narrative')],
    }
    
    print(f"✓ Prepared {len(texts)} samples")
    print(f"  Mean outcome: {np.mean(outcomes):.3f}")
    print(f"  Std outcome: {np.std(outcomes):.3f}")
    
    # Run transformers
    print("\n[3/4] Running transformers...")
    runner = CachedTransformerRunner('novels')
    X, transformer_names = runner.run_all_transformers(texts, context)
    
    if X is None:
        print("\n❌ Feature extraction failed!")
        return
    
    # Train and evaluate
    print("\n[4/4] Training and evaluating...")
    global all_features  # Make available for importance calculation
    # Store features for importance mapping
    results = runner.train_and_evaluate(X, outcomes, transformer_names)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Domain: novels")
    print(f"Samples: {len(texts)}")
    print(f"Features: {X.shape[1]}")
    print(f"CV R²: {results['mean_cv_r2']:.4f}")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()






