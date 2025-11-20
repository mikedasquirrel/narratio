"""
Supervised Feature Extraction Pipeline
--------------------------------------

Extends the universal `FeatureExtractionPipeline` with transformers that
require supervised signals (labels/outcomes) or specialized inputs (genome
payloads, canonical feature matrices, downstream probability blocks).

This pipeline:
1. Runs the standard unsupervised pipeline to build the canonical feature
   matrix (ж) and provenance metadata.
2. Builds genome payloads (nominative + archetypal + historial + uniquity)
   for discovery-stage transformers.
3. Applies label-aware transformers (Ξ discovery, α measurement, context
   pattern detection, interaction mining, ensemble fusion, cross-domain
   embedding) in the same order returned by the transformer selector.
4. Caches the combined feature matrix with mode metadata so supervised runs
   never collide with unsupervised caches.
"""

from __future__ import annotations

# FIX TENSORFLOW MUTEX DEADLOCK ON MACOS (same guard as base pipeline)
import os

if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import hashlib
import importlib
import json
import pickle
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_extraction_pipeline import FeatureExtractionPipeline
from .genome_feature_adapter import GenomeFeatureAdapter


class SupervisedFeatureExtractionPipeline:
    """
    Production-grade supervised feature extraction.
    """

    LABEL_REQUIRED = {
        'AlphaTransformer',
        'GoldenNarratioTransformer',
        'ContextPatternTransformer',
        'MetaFeatureInteractionTransformer',
        'EnsembleMetaTransformer',
    }

    GENOME_REQUIRED = {
        'CrossDomainEmbeddingTransformer',
    }

    FEATURE_MATRIX_INPUTS = {
        'AlphaTransformer',
        'ContextPatternTransformer',
        'MetaFeatureInteractionTransformer',
        'EnsembleMetaTransformer',
    }

    TEXT_LABEL_INPUTS = {
        'GoldenNarratioTransformer',
    }

    PIPELINE_MODE = 'supervised'

    def __init__(
        self,
        transformer_names: List[str],
        domain_name: str,
        cache_dir: Optional[str] = None,
        enable_caching: bool = True,
        verbose: bool = True,
        domain_narrativity: Optional[float] = None,
    ):
        self.transformer_names = transformer_names
        self.domain_name = domain_name
        self.enable_caching = enable_caching
        self.verbose = verbose
        self.domain_narrativity = domain_narrativity

        # Partition transformers so the base pipeline only receives the
        # unsupervised subset.
        self.supervised_transformer_names = [
            name for name in transformer_names
            if name in self.LABEL_REQUIRED or name in self.GENOME_REQUIRED
        ]
        self.base_transformer_names = [
            name for name in transformer_names
            if name not in self.supervised_transformer_names
        ]

        if not self.base_transformer_names:
            raise ValueError(
                "Supervised pipeline requires at least one base transformer "
                "to construct the canonical feature matrix."
            )

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path('narrative_optimization/cache/features')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Unsupervised base pipeline (handles caching internally).
        self.base_pipeline = FeatureExtractionPipeline(
            transformer_names=self.base_transformer_names,
            domain_name=domain_name,
            cache_dir=str(self.cache_dir),
            enable_caching=enable_caching,
            verbose=verbose,
        )

        # Populated after fit_transform
        self.feature_provenance: Dict[str, str] = {}
        self.features_by_transformer: OrderedDict[str, np.ndarray] = OrderedDict()
        self.transformer_status: Dict[str, Dict] = {}
        self.extraction_stats: List[Dict] = []
        self.all_feature_names: List[str] = []

        self.supervised_transformers = self._instantiate_supervised_transformers()

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #

    def _instantiate_supervised_transformers(self) -> Dict[str, object]:
        """Instantiate supervised transformers with error handling."""
        supervised = {}
        transformers_module = importlib.import_module('transformers')

        for name in self.supervised_transformer_names:
            transformer_class = getattr(transformers_module, name, None)
            if transformer_class is None:
                self.transformer_status[name] = {
                    'status': 'error',
                    'error': f'Class {name} not found in transformers module',
                }
                if self.verbose:
                    print(f"  ✗ {name}: Class not found")
                continue

            kwargs = {}
            if name == 'AlphaTransformer' and self.domain_narrativity is not None:
                kwargs['narrativity'] = self.domain_narrativity

            try:
                instance = transformer_class(**kwargs)
                supervised[name] = instance
                self.transformer_status[name] = {
                    'status': 'initialized',
                    'error': None,
                }
                if self.verbose:
                    print(f"  ✓ {name}: Initialized (supervised pipeline)")
            except Exception as exc:
                self.transformer_status[name] = {
                    'status': 'error',
                    'error': str(exc),
                }
                if self.verbose:
                    print(f"  ✗ {name}: Error during initialization - {exc}")

        return supervised

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _hash_labels(self, labels: Optional[Sequence]) -> str:
        if labels is None:
            return 'nolabel'
        if isinstance(labels, np.ndarray):
            arr = labels
        else:
            arr = np.asarray(labels)
        if arr.dtype == object:
            payload = json.dumps(arr.tolist()).encode('utf-8')
        else:
            payload = arr.tobytes()
        return hashlib.md5(payload).hexdigest()[:8]

    def _get_cache_key(self, narratives: Sequence[str], labels: Optional[Sequence]) -> str:
        sample_size = len(narratives)
        narrative_hash = hashlib.md5(
            (str(sample_size) + '||' + ''.join(narratives[:100])).encode('utf-8')
        ).hexdigest()
        transformer_hash = hashlib.md5(
            ''.join(sorted(self.transformer_names)).encode('utf-8')
        ).hexdigest()
        label_hash = self._hash_labels(labels)
        return (
            f"{self.domain_name}_{self.PIPELINE_MODE}_{sample_size}_"
            f"{narrative_hash[:8]}_{transformer_hash[:8]}_{label_hash}"
        )

    def _load_from_cache(self, cache_key: str):
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

        if cache_file.exists() and metadata_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.feature_provenance = metadata.get('feature_provenance', {})
                self.all_feature_names = metadata.get('feature_names', [])
                self.transformer_status = metadata.get('transformer_status', {})
                self.extraction_stats = metadata.get('extraction_stats', [])
                if self.verbose:
                    print(f"✓ Loaded supervised features from cache: {cache_key}")
                return features, metadata
            except Exception as exc:
                if self.verbose:
                    print(f"✗ Supervised cache load failed: {exc}")
        return None

    def _save_to_cache(self, cache_key: str, features: np.ndarray, metadata: Dict):
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            metadata['timestamp'] = datetime.now().isoformat()
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            if self.verbose:
                print(f"✓ Saved supervised features to cache: {cache_key}")
        except Exception as exc:
            if self.verbose:
                print(f"✗ Supervised cache save failed: {exc}")

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def fit_transform(
        self,
        narratives: List[str],
        labels: Sequence,
        return_dataframe: bool = False
    ):
        if len(narratives) == 0:
            raise ValueError("Empty narratives list")
        labels_array = self._validate_labels(labels, len(narratives))

        cache_key = None
        if self.enable_caching:
            cache_key = self._get_cache_key(narratives, labels_array)
            cached = self._load_from_cache(cache_key)
            if cached:
                features, metadata = cached
                if return_dataframe:
                    return pd.DataFrame(features, columns=self.all_feature_names)
                return features

        if self.verbose:
            print(f"\n{'='*80}")
            print("SUPERVISED FEATURE EXTRACTION")
            print(f"{'='*80}")
            print(f"Domain: {self.domain_name}")
            print(f"Narratives: {len(narratives):,}")
            print(f"Transformers (total): {len(self.transformer_names)}")
            print(f"Supervised transformers: {len(self.supervised_transformer_names)}")
            print(f"{'='*80}\n")

        # Phase 1 – base feature extraction
        base_features = self.base_pipeline.fit_transform(narratives)
        base_report = self.base_pipeline.get_extraction_report()
        base_feature_names = [
            name for name in self.base_pipeline.feature_provenance.keys()
            if self.base_pipeline.feature_provenance.get(name)
        ]
        base_provenance = dict(self.base_pipeline.feature_provenance)
        base_blocks = self.base_pipeline.get_features_by_transformer(base_features)

        if not base_blocks:
            raise ValueError("Base pipeline produced no features; aborting supervised stage.")

        # Canonical matrix ordered by base transformer appearance
        base_feature_matrix_blocks = []
        ordered_base_names: List[str] = []
        for name in self.base_transformer_names:
            block = base_blocks.get(name)
            if block is None:
                continue
            names = [
                fname for fname in base_feature_names
                if base_provenance.get(fname) == name
            ]
            base_feature_matrix_blocks.append(block)
            ordered_base_names.extend(names)

        if not base_feature_matrix_blocks:
            raise ValueError("Unable to assemble canonical feature matrix for supervised transformers.")

        canonical_matrix = np.hstack(base_feature_matrix_blocks)

        genome_payload = None
        if any(name in self.GENOME_REQUIRED for name in self.supervised_transformer_names):
            adapter = GenomeFeatureAdapter(
                feature_names=ordered_base_names,
                feature_provenance=base_provenance,
                domain_name=self.domain_name,
            )
            genome_payload = adapter.build_payload(canonical_matrix, narratives)

        ordered_blocks: List[np.ndarray] = []
        ordered_names: List[List[str]] = []
        feature_provenance: Dict[str, str] = OrderedDict()
        features_by_transformer: OrderedDict[str, np.ndarray] = OrderedDict()
        extraction_stats: List[Dict] = []
        transformer_status = {}

        # Helper to add provenance entries
        def _append_block(name: str, block: np.ndarray, block_names: List[str]):
            ordered_blocks.append(block)
            ordered_names.append(block_names)
            for fname in block_names:
                feature_provenance[fname] = name
            features_by_transformer[name] = block

        # Iterate in the original selection order
        for t_name in self.transformer_names:
            if t_name in self.base_transformer_names:
                block = base_blocks.get(t_name)
                status = base_report['transformer_status'].get(t_name, {})
                transformer_status[t_name] = {
                    'status': status.get('status', 'unknown'),
                    'error': status.get('error'),
                }
                if block is None:
                    extraction_stats.append({
                        'transformer': t_name,
                        'status': status.get('status', 'skipped'),
                        'feature_count': 0,
                        'error': status.get('error')
                    })
                    continue

                names = [
                    fname for fname in base_feature_names
                    if base_provenance.get(fname) == t_name
                ]
                _append_block(t_name, block, names)
                extraction_stats.append({
                    'transformer': t_name,
                    'status': 'success',
                    'feature_count': block.shape[1],
                    'error': None
                })
                continue

            if t_name not in self.supervised_transformer_names:
                # Transformer was filtered out (e.g., invisible sports feeds)
                transformer_status[t_name] = {
                    'status': 'skipped',
                    'error': 'Transformer not applicable in supervised pipeline'
                }
                extraction_stats.append({
                    'transformer': t_name,
                    'status': 'skipped',
                    'feature_count': 0,
                    'error': 'Not included in supervised pipeline'
                })
                continue

            block, names, error = self._apply_supervised_transformer(
                t_name=t_name,
                canonical_features=canonical_matrix,
                canonical_names=ordered_base_names,
                narratives=narratives,
                labels=labels_array,
                genome_payload=genome_payload,
                base_blocks=base_blocks,
            )

            if block is None or names is None:
                transformer_status[t_name] = {
                    'status': 'error',
                    'error': error
                }
                extraction_stats.append({
                    'transformer': t_name,
                    'status': 'error',
                    'feature_count': 0,
                    'error': error
                })
                if self.verbose:
                    print(f"  ✗ {t_name}: {error}")
                continue

            transformer_status[t_name] = {
                'status': 'success',
                'error': None
            }
            extraction_stats.append({
                'transformer': t_name,
                'status': 'success',
                'feature_count': block.shape[1],
                'error': None
            })
            _append_block(t_name, block, names)
            if self.verbose:
                print(f"  ✓ {t_name}: {block.shape[1]} features")

        if not ordered_blocks:
            raise ValueError("No features extracted. All transformers failed.")

        final_features = np.hstack(ordered_blocks)
        final_feature_names: List[str] = []
        for block_names in ordered_names:
            final_feature_names.extend(block_names)

        self.feature_provenance = feature_provenance
        self.features_by_transformer = features_by_transformer
        self.transformer_status = transformer_status
        self.extraction_stats = extraction_stats
        self.all_feature_names = final_feature_names

        if self.verbose:
            print(f"\n{'='*80}")
            print("SUPERVISED EXTRACTION COMPLETE")
            print(f"{'='*80}")
            print(f"Total features: {final_features.shape[1]:,}")
            print(f"Successful transformers: "
                  f"{sum(1 for s in transformer_status.values() if s['status'] == 'success')}/"
                  f"{len(self.transformer_names)}")
            print(f"{'='*80}\n")

        if self.enable_caching and cache_key:
            metadata = {
                'domain': self.domain_name,
                'n_narratives': len(narratives),
                'n_features': final_features.shape[1],
                'feature_names': final_feature_names,
                'feature_provenance': feature_provenance,
                'extraction_stats': extraction_stats,
                'transformer_status': transformer_status,
                'pipeline_mode': self.PIPELINE_MODE,
                'supervised_transformers': self.supervised_transformer_names,
            }
            self._save_to_cache(cache_key, final_features, metadata)

        if return_dataframe:
            return pd.DataFrame(final_features, columns=final_feature_names)

        return final_features

    # ------------------------------------------------------------------ #
    # Transformer application
    # ------------------------------------------------------------------ #

    def _apply_supervised_transformer(
        self,
        t_name: str,
        canonical_features: np.ndarray,
        canonical_names: List[str],
        narratives: Sequence[str],
        labels: np.ndarray,
        genome_payload: Optional[List[Dict]],
        base_blocks: Dict[str, np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
        transformer = self.supervised_transformers.get(t_name)
        if transformer is None:
            return None, None, f"{t_name} not initialized"

        try:
            if t_name == 'GoldenNarratioTransformer':
                features = transformer.fit_transform(list(narratives), labels)
                names = transformer.get_feature_names()
                return np.asarray(features), names, None

            if t_name == 'AlphaTransformer':
                transformer.fit(canonical_features, labels, feature_names=canonical_names)
                features = transformer.transform(canonical_features)
                names = transformer.get_feature_names()
                return np.asarray(features), names, None

            if t_name == 'ContextPatternTransformer':
                df = pd.DataFrame(canonical_features, columns=canonical_names)
                transformer.fit(df, labels)
                features = transformer.transform(df)
                names = [f'ContextPattern_{i}' for i in range(features.shape[1])]
                return np.asarray(features, dtype=np.float32), names, None

            if t_name == 'MetaFeatureInteractionTransformer':
                structured = self._build_feature_dict_input(canonical_features, canonical_names)
                features = transformer.fit_transform(structured, labels)
                names = transformer.get_feature_names()
                return np.asarray(features), names, None

            if t_name == 'CrossDomainEmbeddingTransformer':
                if not genome_payload:
                    raise ValueError("Genome payload missing for CrossDomainEmbeddingTransformer")
                features = transformer.fit_transform(genome_payload, labels)
                names = transformer.get_feature_names()
                return np.asarray(features), names, None

            if t_name == 'EnsembleMetaTransformer':
                blocks = [
                    (name, block)
                    for name, block in base_blocks.items()
                    if block is not None and block.shape[1] > 0
                ]
                if not blocks:
                    raise ValueError("No base feature blocks available for EnsembleMetaTransformer")
                if hasattr(transformer, 'set_precomputed_blocks'):
                    transformer.set_precomputed_blocks(blocks)
                features = transformer.fit_transform(None, labels)
                names = getattr(transformer, 'get_feature_names', lambda: None)()
                if not names:
                    names = [f'EnsembleMeta_{i}' for i in range(features.shape[1])]
                return np.asarray(features), names, None

            return None, None, f"{t_name} is not supported in supervised mode yet"

        except Exception as exc:
            return None, None, str(exc)

    def _build_feature_dict_input(self, features: np.ndarray, feature_names: List[str]) -> List[Dict]:
        """Create list-of-dicts input for transformers that expect genome payloads."""
        structured = []
        for idx, row in enumerate(features):
            if idx == 0:
                structured.append({'genome_features': row, 'feature_names': feature_names})
            else:
                structured.append({'genome_features': row})
        return structured

    def _validate_labels(self, labels: Sequence, expected_len: int) -> np.ndarray:
        if labels is None:
            raise ValueError("Supervised pipeline requires labels (y).")
        labels_array = np.asarray(labels)
        if labels_array.shape[0] != expected_len:
            raise ValueError(
                f"Labels length mismatch: expected {expected_len}, got {labels_array.shape[0]}"
            )
        if labels_array.ndim > 1:
            labels_array = labels_array.reshape(labels_array.shape[0])
        return labels_array

    # ------------------------------------------------------------------ #
    # Reporting helpers
    # ------------------------------------------------------------------ #

    def get_extraction_report(self) -> Dict:
        successful = sum(1 for s in self.transformer_status.values() if s['status'] == 'success')
        failed = sum(1 for s in self.transformer_status.values() if s['status'] == 'error')
        skipped = sum(1 for s in self.transformer_status.values() if s['status'] not in {'success', 'error'})

        return {
            'domain': self.domain_name,
            'pipeline_mode': self.PIPELINE_MODE,
            'total_requested': len(self.transformer_names),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'transformer_status': self.transformer_status,
            'feature_provenance': self.feature_provenance,
            'extraction_stats': self.extraction_stats,
        }

    def print_extraction_report(self):
        report = self.get_extraction_report()
        print(f"\n{'='*80}")
        print(f"SUPERVISED FEATURE EXTRACTION REPORT: {report['domain'].upper()}")
        print(f"{'='*80}")
        print(f"Total Transformers Requested: {report['total_requested']}")
        print(f"  ✓ Successful: {report['successful']}")
        print(f"  ✗ Failed: {report['failed']}")
        print(f"  ⊘ Skipped: {report['skipped']}")

        failed_items = [
            (name, status['error'])
            for name, status in report['transformer_status'].items()
            if status['status'] == 'error'
        ]
        if failed_items:
            print("\nFailed Transformers:")
            for name, error in failed_items:
                print(f"  - {name}: {error}")

        print(f"\nTotal Features Extracted: {len(report['feature_provenance']):,}")
        print(f"{'='*80}\n")

    def get_features_by_transformer(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        result = {}
        start = 0
        for t_name, block in self.features_by_transformer.items():
            width = block.shape[1]
            result[t_name] = features[:, start:start + width]
            start += width
        return result


