"""
Feature Extraction Pipeline

Unified feature extraction system that:
- Applies selected transformers
- Handles errors gracefully
- Caches results
- Tracks feature provenance
- Reports statistics

Author: Narrative Optimization Framework
Date: November 2025
"""

# FIX TENSORFLOW MUTEX DEADLOCK ON MACOS
# Must be set BEFORE any TensorFlow/transformers imports
import os
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import pickle
import hashlib
from datetime import datetime
import traceback
import sys

# Import transformer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# LAZY IMPORT - Don't import all transformers at module level
# Import only when needed via importlib to avoid mutex deadlock
import importlib


class FeatureExtractionPipeline:
    """
    Production-ready feature extraction with error handling and caching.
    """
    
    def __init__(
        self,
        transformer_names: List[str],
        domain_name: str,
        cache_dir: Optional[str] = None,
        enable_caching: bool = True,
        verbose: bool = True
    ):
        """
        Initialize feature extraction pipeline.
        
        Parameters
        ----------
        transformer_names : list of str
            Transformer class names to apply
        domain_name : str
            Domain identifier (for caching and logging)
        cache_dir : str, optional
            Directory for caching features
        enable_caching : bool
            Enable feature caching (default: True)
        verbose : bool
            Print progress information (default: True)
        """
        self.transformer_names = transformer_names
        self.domain_name = domain_name
        self.enable_caching = enable_caching
        self.verbose = verbose
        # Transformers that require labels/outcomes (y) to function properly
        self.supervised_only = {
            'AlphaTransformer',
            'GoldenNarratioTransformer',
            'ContextPatternTransformer',
            'MetaFeatureInteractionTransformer',
            'EnsembleMetaTransformer',
        }
        # Transformers that expect specialized genome inputs that aren't produced here
        self.genome_required = {
            'CrossDomainEmbeddingTransformer',
        }
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path('narrative_optimization/cache/features')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize transformers
        self.transformers = []
        self.transformer_status = {}
        self.feature_provenance = {}  # Maps feature names to transformers
        
        self._instantiate_transformers()
    
    def _instantiate_transformers(self):
        """Instantiate all transformers with error handling."""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"INITIALIZING TRANSFORMERS: {len(self.transformer_names)} requested")
            print(f"{'='*80}\n")
        
        transformers_module = importlib.import_module('transformers')

        for t_name in self.transformer_names:
            try:
                # Lazy attribute access triggers transformers.__getattr__
                transformer_class = getattr(transformers_module, t_name, None)
                if transformer_class is None:
                    self.transformer_status[t_name] = {
                        'status': 'error',
                        'error': f'Class {t_name} not found in transformers module',
                        'instance': None
                    }
                    if self.verbose:
                        print(f"  ✗ {t_name}: Class not found")
                    continue
                
                # Instantiate (some transformers need specific params)
                if t_name in self.supervised_only:
                    self.transformer_status[t_name] = {
                        'status': 'skipped',
                        'error': 'Requires supervisory labels (y) not available in unsupervised pipeline',
                        'instance': None
                    }
                    if self.verbose:
                        print(f"  ⊘ {t_name}: Skipped (requires labels)")
                    continue
                if t_name in self.genome_required:
                    self.transformer_status[t_name] = {
                        'status': 'skipped',
                        'error': 'Requires genome feature dict not produced in feature pipeline',
                        'instance': None
                    }
                    if self.verbose:
                        print(f"  ⊘ {t_name}: Skipped (requires genome inputs)")
                    continue
                if t_name == 'StatisticalTransformer':
                    instance = transformer_class(max_features=150)
                else:
                    instance = transformer_class()
                
                self.transformers.append((t_name, instance))
                self.transformer_status[t_name] = {
                    'status': 'initialized',
                    'error': None,
                    'instance': instance
                }
                
                if self.verbose:
                    print(f"  ✓ {t_name}: Initialized")
                    
            except Exception as e:
                self.transformer_status[t_name] = {
                    'status': 'error',
                    'error': str(e),
                    'instance': None
                }
                if self.verbose:
                    print(f"  ✗ {t_name}: Error - {str(e)}")
        
        if self.verbose:
            successful = len(self.transformers)
            total = len(self.transformer_names)
            print(f"\n✓ Initialized {successful}/{total} transformers successfully\n")
    
    def _get_cache_key(self, narratives: List[str]) -> str:
        """Generate cache key from narratives hash."""
        # Hash the narratives to create unique key
        sample_size = len(narratives)
        narrative_hash = hashlib.md5(
            (str(sample_size) + '||' + ''.join(narratives[:100])).encode()
        ).hexdigest()
        
        transformer_hash = hashlib.md5(
            ''.join(sorted(self.transformer_names)).encode()
        ).hexdigest()
        
        return f"{self.domain_name}_{sample_size}_{narrative_hash[:8]}_{transformer_hash[:8]}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load features from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
        
        if cache_file.exists() and metadata_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if self.verbose:
                    print(f"✓ Loaded features from cache: {cache_key}")
                    print(f"  Shape: {features.shape}")
                    print(f"  Cached: {metadata['timestamp']}\n")
                
                return features, metadata
            except Exception as e:
                if self.verbose:
                    print(f"✗ Cache load failed: {e}")
                return None
        
        return None
    
    def _save_to_cache(
        self,
        cache_key: str,
        features: np.ndarray,
        metadata: Dict
    ):
        """Save features to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            
            metadata['timestamp'] = datetime.now().isoformat()
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if self.verbose:
                print(f"✓ Saved features to cache: {cache_key}\n")
        except Exception as e:
            if self.verbose:
                print(f"✗ Cache save failed: {e}\n")
    
    def fit_transform(
        self,
        narratives: List[str],
        return_dataframe: bool = False
    ) -> np.ndarray:
        """
        Extract features from narratives.
        
        Parameters
        ----------
        narratives : list of str
            Narrative texts
        return_dataframe : bool
            Return pandas DataFrame with feature names (default: False)
        
        Returns
        -------
        features : ndarray or DataFrame
            Extracted features (n_samples, n_features)
        """
        if len(narratives) == 0:
            raise ValueError("Empty narratives list")
        
        # Check cache
        cache_key = None
        if self.enable_caching:
            cache_key = self._get_cache_key(narratives)
            cached = self._load_from_cache(cache_key)
            if cached:
                features, metadata = cached
                self.feature_provenance = metadata.get('feature_provenance', {})
                if return_dataframe:
                    feature_names = metadata.get('feature_names', [])
                    return pd.DataFrame(features, columns=feature_names)
                return features
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"EXTRACTING FEATURES")
            print(f"{'='*80}")
            print(f"Domain: {self.domain_name}")
            print(f"Narratives: {len(narratives):,}")
            print(f"Transformers: {len(self.transformers)}")
            print(f"{'='*80}\n")
        
        # Extract features from each transformer
        all_features = []
        all_feature_names = []
        extraction_stats = []
        
        for i, (t_name, transformer) in enumerate(self.transformers, 1):
            if self.verbose:
                print(f"[{i}/{len(self.transformers)}] Applying {t_name}...", end=' ')
            
            try:
                # Apply transformer
                t_features = transformer.fit_transform(narratives)

                # Convert sparse matrices
                if sparse.issparse(t_features):
                    t_features = t_features.toarray()
                
                t_features = np.asarray(t_features)
                
                # Handle different return types
                if isinstance(t_features, pd.DataFrame):
                    feature_names = list(t_features.columns)
                    t_features = t_features.values
                else:
                    n_features = t_features.shape[1] if t_features.ndim > 1 else 1
                    feature_names = [f"{t_name}_{j}" for j in range(n_features)]
                
                # Ensure 2D array for stacking
                if t_features.ndim == 0:
                    t_features = np.full((len(narratives), 1), t_features)
                elif t_features.ndim == 1:
                    if t_features.shape[0] == len(narratives):
                        t_features = t_features.reshape(len(narratives), 1)
                    else:
                        t_features = t_features.reshape(1, -1)
                elif t_features.ndim > 2:
                    t_features = t_features.reshape(t_features.shape[0], -1)

                # Ensure first dimension matches sample count
                if t_features.shape[0] != len(narratives):
                    if t_features.shape[1] == len(narratives):
                        t_features = t_features.T
                    else:
                        raise ValueError(
                            f"{t_name} returned shape {t_features.shape}; "
                            f"expected first dimension {len(narratives)}"
                        )

                # Align feature names with actual width
                n_features = t_features.shape[1]
                if len(feature_names) != n_features:
                    feature_names = [f"{t_name}_{j}" for j in range(n_features)]
                
                # Track provenance
                for fname in feature_names:
                    self.feature_provenance[fname] = t_name
                
                all_features.append(t_features)
                all_feature_names.extend(feature_names)
                
                extraction_stats.append({
                    'transformer': t_name,
                    'status': 'success',
                    'feature_count': len(feature_names),
                    'error': None
                })
                
                if self.verbose:
                    print(f"✓ {len(feature_names)} features")
                
            except Exception as e:
                extraction_stats.append({
                    'transformer': t_name,
                    'status': 'error',
                    'feature_count': 0,
                    'error': str(e)
                })
                
                if self.verbose:
                    print(f"✗ Error: {str(e)[:60]}")
                
                # Continue with other transformers
                continue
        
        if len(all_features) == 0:
            raise ValueError("No features extracted! All transformers failed.")
        
        # Combine features
        features = np.hstack(all_features)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"EXTRACTION COMPLETE")
            print(f"{'='*80}")
            print(f"Total features: {features.shape[1]:,}")
            print(f"Successful transformers: {len(all_features)}/{len(self.transformers)}")
            print(f"Feature matrix shape: {features.shape}")
            print(f"{'='*80}\n")
        
        # Save to cache
        if self.enable_caching and cache_key:
            metadata = {
                'domain': self.domain_name,
                'n_narratives': len(narratives),
                'n_features': features.shape[1],
                'feature_names': all_feature_names,
                'feature_provenance': self.feature_provenance,
                'extraction_stats': extraction_stats,
                'transformer_names': self.transformer_names
            }
            self._save_to_cache(cache_key, features, metadata)
        
        if return_dataframe:
            return pd.DataFrame(features, columns=all_feature_names)
        
        return features
    
    def get_extraction_report(self) -> Dict:
        """Get detailed extraction report."""
        successful = sum(1 for s in self.transformer_status.values() if s['status'] == 'initialized')
        failed = sum(1 for s in self.transformer_status.values() if s['status'] == 'error')
        skipped = sum(1 for s in self.transformer_status.values() if s['status'] == 'skipped')
        
        return {
            'domain': self.domain_name,
            'total_requested': len(self.transformer_names),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'transformer_status': self.transformer_status,
            'feature_provenance': self.feature_provenance
        }
    
    def print_extraction_report(self):
        """Print human-readable extraction report."""
        report = self.get_extraction_report()
        
        print(f"\n{'='*80}")
        print(f"FEATURE EXTRACTION REPORT: {report['domain'].upper()}")
        print(f"{'='*80}")
        print(f"Total Transformers Requested: {report['total_requested']}")
        print(f"  ✓ Successful: {report['successful']}")
        print(f"  ✗ Failed: {report['failed']}")
        print(f"  ⊘ Skipped: {report['skipped']}")
        
        if report['failed'] > 0:
            print(f"\nFailed Transformers:")
            for t_name, status in report['transformer_status'].items():
                if status['status'] == 'error':
                    print(f"  - {t_name}: {status['error']}")
        
        if report['skipped'] > 0:
            print(f"\nSkipped Transformers:")
            for t_name, status in report['transformer_status'].items():
                if status['status'] == 'skipped':
                    print(f"  - {t_name}: {status['error']}")
        
        print(f"\nTotal Features Extracted: {len(report['feature_provenance']):,}")
        print(f"{'='*80}\n")
    
    def get_features_by_transformer(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split features back by transformer for analysis.
        
        Parameters
        ----------
        features : ndarray
            Full feature matrix
        
        Returns
        -------
        features_by_transformer : dict
            {transformer_name: feature_subset}
        """
        result = {}
        feature_names = list(self.feature_provenance.keys())
        
        for t_name in set(self.feature_provenance.values()):
            # Find indices for this transformer
            indices = [i for i, fname in enumerate(feature_names) 
                      if self.feature_provenance[fname] == t_name]
            if indices:
                result[t_name] = features[:, indices]
        
        return result


def quick_extract(
    narratives: List[str],
    domain_name: str,
    pi_value: float,
    domain_type: Optional[str] = None,
    fast_mode: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Quick extraction helper function.
    
    Parameters
    ----------
    narratives : list of str
        Narrative texts
    domain_name : str
        Domain identifier
    pi_value : float
        Domain narrativity
    domain_type : str, optional
        Domain category
    fast_mode : bool
        Use fast subset of transformers (default: False)
    
    Returns
    -------
    features : ndarray
        Extracted features
    report : dict
        Extraction report
    """
    # Import selector
    from transformers.transformer_selector import TransformerSelector
    
    # Select transformers
    selector = TransformerSelector()
    transformer_names = selector.select_transformers(
        domain_name, pi_value, domain_type
    )
    
    if fast_mode:
        transformer_names = selector.get_fast_subset(transformer_names)
    
    # Extract features
    pipeline = FeatureExtractionPipeline(
        transformer_names,
        domain_name,
        verbose=True
    )
    
    features = pipeline.fit_transform(narratives)
    report = pipeline.get_extraction_report()
    
    return features, report


if __name__ == '__main__':
    # Test the pipeline
    print("\n" + "="*80)
    print("TESTING FEATURE EXTRACTION PIPELINE")
    print("="*80)
    
    # Sample narratives
    test_narratives = [
        "The hero embarks on a dangerous journey to save the kingdom from darkness.",
        "A young scientist discovers a revolutionary theory that challenges everything we know.",
        "Two rivals face off in an epic battle for supremacy."
    ] * 10  # 30 samples
    
    from transformers.transformer_selector import TransformerSelector
    
    # Test with NBA domain
    selector = TransformerSelector()
    transformer_names = selector.select_transformers('nba', pi_value=0.49, domain_type='sports')
    
    # Create pipeline
    pipeline = FeatureExtractionPipeline(
        transformer_names[:10],  # Use first 10 for quick test
        domain_name='nba_test',
        enable_caching=False,
        verbose=True
    )
    
    # Extract features
    try:
        features = pipeline.fit_transform(test_narratives)
        pipeline.print_extraction_report()
        
        print(f"Final feature matrix shape: {features.shape}")
        print(f"Features per sample: {features.shape[1]}")
        print(f"\nTest PASSED ✓")
    except Exception as e:
        print(f"\nTest FAILED ✗")
        print(f"Error: {e}")
        traceback.print_exc()

