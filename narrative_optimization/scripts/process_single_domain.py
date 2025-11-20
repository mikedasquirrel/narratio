"""
Process Single Domain - Apply All Transformers

Applies all applicable transformers to a single domain with:
- Timeout protection (30 minutes)
- Error recovery (skip-on-error)
- Force recomputation
- Detailed logging
- Feature matrix output

Usage:
    python process_single_domain.py --domain nba
    python process_single_domain.py --domain crypto --force-recompute

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import time
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all transformers from main workspace
from narrative_optimization.src.transformers import (
    # Core (6)
    NominativeAnalysisTransformer,
    SelfPerceptionTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    RelationalValueTransformer,
    EnsembleNarrativeTransformer,
    # Statistical
    StatisticalTransformer,
    # Nominative (7)
    PhoneticTransformer,
    SocialStatusTransformer,
    UniversalNominativeTransformer,
    HierarchicalNominativeTransformer,
    NominativeInteractionTransformer,
    PureNominativePredictorTransformer,
    NominativeRichnessTransformer,
    # Narrative Semantic (6)
    EmotionalResonanceTransformer,
    AuthenticityTransformer,
    ConflictTensionTransformer,
    ExpertiseAuthorityTransformer,
    CulturalContextTransformer,
    SuspenseMysteryTransformer,
    # Structural (2)
    OpticsTransformer,
    FramingTransformer,
    # Contextual (1)
    TemporalEvolutionTransformer,
    # Advanced (6)
    InformationTheoryTransformer,
    NamespaceEcologyTransformer,
    AnticipatoryCommunicationTransformer,
    CognitiveFluencyTransformer,
    QuantitativeTransformer,
    DiscoverabilityTransformer,
    # Multimodal (4)
    VisualMultimodalTransformer,
    CrossmodalTransformer,
    AudioTransformer,
    CrossLingualTransformer,
    # Fractal (3)
    MultiScaleTransformer,
    MultiPerspectiveTransformer,
    ScaleInteractionTransformer,
    # Theory-aligned (7)
    CouplingStrengthTransformer,
    NarrativeMassTransformer,
    GravitationalFeaturesTransformer,
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer,
    AlphaTransformer,
    GoldenNarratioTransformer,
)


def load_domain_config(domain_name, project_root):
    """Load domain configuration from BATCH_EXECUTION_STATUS.json"""
    status_file = project_root / 'narrative_optimization' / 'BATCH_EXECUTION_STATUS.json'
    
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    if domain_name not in status['domains']:
        raise ValueError(f"Domain '{domain_name}' not found in configuration")
    
    return status['domains'][domain_name]


def load_domain_data(domain_config, project_root):
    """Load data for a domain"""
    data_path = project_root / domain_config['data_path']
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    
    try:
        if data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
        elif data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        elif data_path.suffix == '.tsv':
            import pandas as pd
            data = pd.read_csv(data_path, sep='\t').to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Extract texts and outcomes
        if isinstance(data, dict) and 'texts' in data:
            # Format C (crypto)
            texts = data['texts']
            outcomes = data.get(domain_config['outcome_field'], None)
        elif isinstance(data, dict) and 'data' in data and 'texts' in data['data']:
            # Nested format C
            texts = data['data']['texts']
            outcomes = data['data'].get(domain_config['outcome_field'], None)
        elif isinstance(data, dict) and 'disorders' in data:
            # Mental health format
            disorder_list = data['disorders']
            text_field = domain_config.get('text_field', 'full_description')
            texts = [item.get(text_field, '') for item in disorder_list]
            outcomes = [item.get(domain_config['outcome_field'], 0) for item in disorder_list]
        elif isinstance(data, list):
            # List of dicts containing narratives
            text_fields = domain_config.get('text_fields')
            if not text_fields:
                primary_field = domain_config.get('text_field')
                text_fields = [primary_field] if primary_field else []
            
            # Auto-discover common narrative fields if necessary
            if not text_fields or all(f not in data[0] for f in text_fields):
                fallback_fields = [
                    'rich_narrative',
                    'pregame_narrative',
                    'narrative',
                    'full_narrative',
                    'description',
                    'full_description'
                ]
                text_fields = [f for f in fallback_fields if f in data[0]]
            
            if not text_fields:
                raise ValueError(
                    "No valid text field found. Available fields: "
                    f"{list(data[0].keys())}"
                )
            
            def extract_text(item):
                for field in text_fields:
                    value = item.get(field)
                    if isinstance(value, str) and value.strip():
                        return value
                # Return first non-empty even if not string
                for field in text_fields:
                    value = item.get(field)
                    if value:
                        return str(value)
                return ''
            
            texts = [extract_text(item) for item in data]
            outcomes = [item.get(domain_config['outcome_field'], 0) for item in data]
        else:
            raise ValueError(f"Unexpected data format. Type: {type(data)}, Keys: {data.keys() if hasattr(data, 'keys') else 'N/A'}")
        
        # Filter empty texts
        valid_indices = [i for i, text in enumerate(texts) if text and len(str(text).strip()) > 0]
        texts = [texts[i] for i in valid_indices]
        
        if outcomes is not None:
            outcomes = np.array([outcomes[i] for i in valid_indices])
        
        # Sample if needed
        if domain_config.get('sample_size') and len(texts) > domain_config['sample_size']:
            np.random.seed(42)
            sample_idx = np.random.choice(len(texts), domain_config['sample_size'], replace=False)
            texts = [texts[i] for i in sample_idx]
            if outcomes is not None:
                outcomes = outcomes[sample_idx]
        
        logger.info(f"Loaded {len(texts)} samples")
        return texts, outcomes
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def instantiate_transformers(domain_name, skip_requires_y=False):
    """
    Instantiate all applicable transformers for a domain.
    
    Parameters
    ----------
    domain_name : str
        Domain identifier
    skip_requires_y : bool
        If True, skip transformers that require outcome labels
    
    Returns
    -------
    transformers : list
        List of (name, transformer, category) tuples
    """
    transformers = []
    
    # Core transformers (always used)
    transformers.extend([
        ('nominative_analysis', NominativeAnalysisTransformer(), 'core'),
        ('self_perception', SelfPerceptionTransformer(), 'core'),
        ('narrative_potential', NarrativePotentialTransformer(), 'core'),
        ('linguistic_patterns', LinguisticPatternsTransformer(), 'core'),
        ('relational_value', RelationalValueTransformer(), 'core'),
        ('ensemble_narrative', EnsembleNarrativeTransformer(), 'core'),
    ])
    
    # Statistical baseline
    transformers.append(('statistical', StatisticalTransformer(), 'statistical'))
    
    # Nominative (all text-based)
    transformers.extend([
        ('phonetic', PhoneticTransformer(), 'nominative'),
        ('social_status', SocialStatusTransformer(), 'nominative'),
        ('nominative_richness', NominativeRichnessTransformer(), 'nominative'),
    ])
    
    # Narrative Semantic
    transformers.extend([
        ('emotional_resonance', EmotionalResonanceTransformer(), 'narrative_semantic'),
        ('authenticity', AuthenticityTransformer(), 'narrative_semantic'),
        ('conflict_tension', ConflictTensionTransformer(), 'narrative_semantic'),
        ('expertise_authority', ExpertiseAuthorityTransformer(), 'narrative_semantic'),
        ('cultural_context', CulturalContextTransformer(), 'narrative_semantic'),
        ('suspense_mystery', SuspenseMysteryTransformer(), 'narrative_semantic'),
    ])
    
    # Structural
    transformers.extend([
        ('optics', OpticsTransformer(), 'structural'),
        ('framing', FramingTransformer(), 'structural'),
    ])
    
    # Contextual
    transformers.append(('temporal_evolution', TemporalEvolutionTransformer(), 'contextual'))
    
    # Advanced
    transformers.extend([
        ('information_theory', InformationTheoryTransformer(), 'advanced'),
        ('namespace_ecology', NamespaceEcologyTransformer(), 'advanced'),
        ('anticipatory_communication', AnticipatoryCommunicationTransformer(), 'advanced'),
        ('cognitive_fluency', CognitiveFluencyTransformer(), 'advanced'),
        ('quantitative', QuantitativeTransformer(), 'advanced'),
        ('discoverability', DiscoverabilityTransformer(), 'advanced'),
    ])
    
    # Multimodal
    transformers.extend([
        ('visual_multimodal', VisualMultimodalTransformer(), 'multimodal'),
        ('crossmodal', CrossmodalTransformer(), 'multimodal'),
        ('audio', AudioTransformer(), 'multimodal'),
        ('crosslingual', CrossLingualTransformer(), 'multimodal'),
    ])
    
    # Fractal
    transformers.extend([
        ('multi_scale', MultiScaleTransformer(), 'fractal'),
        ('multi_perspective', MultiPerspectiveTransformer(), 'fractal'),
        ('scale_interaction', ScaleInteractionTransformer(), 'fractal'),
    ])
    
    # Theory-aligned
    transformers.extend([
        ('coupling_strength', CouplingStrengthTransformer(), 'theory_aligned'),
        ('narrative_mass', NarrativeMassTransformer(), 'theory_aligned'),
        ('gravitational_features', GravitationalFeaturesTransformer(), 'theory_aligned'),
        ('awareness_resistance', AwarenessResistanceTransformer(), 'theory_aligned'),
        ('fundamental_constraints', FundamentalConstraintsTransformer(), 'theory_aligned'),
    ])
    
    # Requires y (skip during initial feature extraction)
    if not skip_requires_y:
        transformers.extend([
            ('alpha', AlphaTransformer(), 'theory_aligned'),
            ('golden_narratio', GoldenNarratioTransformer(), 'theory_aligned'),
        ])
    
    logger.info(f"Instantiated {len(transformers)} transformers for {domain_name}")
    return transformers


def process_domain(domain_name, force_recompute=True, skip_on_error=True, timeout_minutes=30):
    """
    Process a single domain with all transformers.
    
    Parameters
    ----------
    domain_name : str
        Domain identifier
    force_recompute : bool
        If True, ignore cache and recompute
    skip_on_error : bool
        If True, skip failed transformers and continue
    timeout_minutes : int
        Maximum time to process domain
    
    Returns
    -------
    results : dict
        Processing results and statistics
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"PROCESSING DOMAIN: {domain_name.upper()}")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        domain_config = load_domain_config(domain_name, project_root)
        logger.info(f"π = {domain_config['pi']}")
        
        # Load data
        texts, outcomes = load_domain_data(domain_config, project_root)
        
        # Instantiate transformers (skip requires_y for now)
        transformers = instantiate_transformers(domain_name, skip_requires_y=True)
        
        # Process each transformer
        all_features = []
        feature_names = []
        transformer_stats = []
        successful_count = 0
        failed_count = 0
        
        for name, transformer, category in transformers:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > (timeout_minutes * 60):
                logger.warning(f"Timeout reached ({timeout_minutes} min), stopping")
                break
            
            logger.info(f"\nProcessing: {name} ({category})")
            
            try:
                # Fit and transform
                t_start = time.time()
                features = transformer.fit_transform(texts)
                
                # Convert sparse matrices to dense arrays for downstream concatenation
                if hasattr(features, 'toarray'):
                    features = features.toarray()
                elif hasattr(features, 'todense'):
                    features = np.asarray(features.todense())
                
                t_elapsed = time.time() - t_start
                
                # Validate features
                if features is None or len(features) == 0:
                    raise ValueError("Transformer returned no features")
                
                if len(features) != len(texts):
                    raise ValueError(f"Feature count mismatch: {len(features)} vs {len(texts)}")
                
                # Store features
                all_features.append(features)
                
                # Get feature names
                if hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out()
                else:
                    names = [f"{name}_{i}" for i in range(features.shape[1])]
                feature_names.extend(names)
                
                successful_count += 1
                logger.info(f"  ✓ Extracted {features.shape[1]} features in {t_elapsed:.1f}s")
                
                transformer_stats.append({
                    'name': name,
                    'category': category,
                    'status': 'success',
                    'features': features.shape[1],
                    'time_seconds': t_elapsed
                })
                
            except Exception as e:
                failed_count += 1
                logger.error(f"  ✗ Failed: {str(e)}")
                
                transformer_stats.append({
                    'name': name,
                    'category': category,
                    'status': 'failed',
                    'features': 0,
                    'time_seconds': 0,
                    'error': str(e)
                })
                
                if not skip_on_error:
                    raise
        
        # Concatenate all features
        if len(all_features) == 0:
            raise ValueError("No features extracted successfully")
        
        X_all = np.hstack(all_features)
        logger.info(f"\nTotal features extracted: {X_all.shape[1]} from {successful_count} transformers")
        
        # Save features
        output_dir = project_root / 'narrative_optimization' / 'data' / 'features'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'{domain_name}_all_features.npz'
        np.savez_compressed(
            output_file,
            features=X_all,
            outcomes=outcomes if outcomes is not None else np.array([]),
            feature_names=np.array(feature_names, dtype=object),
            texts=np.array(texts, dtype=object)[:100] if len(texts) <= 100 else np.array(texts[:100], dtype=object)  # Sample for reference
        )
        
        logger.info(f"✓ Saved: {output_file}")
        
        # Generate results
        total_time = time.time() - start_time
        results = {
            'domain': domain_name,
            'status': 'success',
            'pi': domain_config['pi'],
            'sample_size': len(texts),
            'transformers_completed': successful_count,
            'transformers_failed': failed_count,
            'total_features': X_all.shape[1],
            'duration_seconds': total_time,
            'output_file': str(output_file),
            'transformer_stats': transformer_stats,
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ DOMAIN COMPLETE: {domain_name}")
        logger.info(f"  Features: {X_all.shape[1]}")
        logger.info(f"  Success: {successful_count}, Failed: {failed_count}")
        logger.info(f"  Time: {total_time/60:.1f} minutes")
        logger.info(f"{'=' * 80}\n")
        
        return results
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"\n❌ DOMAIN FAILED: {domain_name}")
        logger.error(f"  Error: {str(e)}")
        logger.error(f"  Time: {total_time/60:.1f} minutes\n")
        
        return {
            'domain': domain_name,
            'status': 'failed',
            'error': str(e),
            'duration_seconds': total_time,
            'completed_at': datetime.now().isoformat()
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Process single domain with all transformers')
    parser.add_argument('--domain', type=str, required=True, help='Domain name to process')
    parser.add_argument('--force-recompute', action='store_true', default=True, help='Force recomputation')
    parser.add_argument('--skip-on-error', action='store_true', default=True, help='Skip failed transformers')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in minutes')
    
    args = parser.parse_args()
    
    # Process domain
    results = process_domain(
        args.domain,
        force_recompute=args.force_recompute,
        skip_on_error=args.skip_on_error,
        timeout_minutes=args.timeout
    )
    
    # Save results
    results_file = project_root / 'narrative_optimization' / 'data' / 'features' / f'{args.domain}_processing_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved: {results_file}")
    
    # Return exit code
    sys.exit(0 if results['status'] == 'success' else 1)


if __name__ == '__main__':
    main()

