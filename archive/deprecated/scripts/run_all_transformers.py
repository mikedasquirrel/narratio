"""
Run All Transformers Across All Domains

Executes all 34+ transformers across all 16 domains with caching.
This is the main execution script for comprehensive feature extraction.

Usage:
    python narrative_optimization/scripts/run_all_transformers.py

Output:
    - narrative_optimization/data/features/cache/ - Cached transformer outputs
    - narrative_optimization/data/features/{domain}_all_features.npz - Feature matrices
    - narrative_optimization/data/features/extraction_report.json - Statistics

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import transformers
from narrative_optimization.src.transformers import (
    # Core (6)
    NominativeAnalysisTransformer,
    SelfPerceptionTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    RelationalValueTransformer,
    EnsembleNarrativeTransformer,
    # Structural (3)
    ConflictTensionTransformer,
    SuspenseMysteryTransformer,
    FramingTransformer,
    # Credibility (2)
    AuthenticityTransformer,
    ExpertiseAuthorityTransformer,
    # Contextual (2)
    TemporalEvolutionTransformer,
    CulturalContextTransformer,
    # Nominative (5)
    PhoneticTransformer,
    SocialStatusTransformer,
    UniversalNominativeTransformer,
    # Advanced (6)
    InformationTheoryTransformer,
    NamespaceEcologyTransformer,
    AnticipatoryCommunicationTransformer,
    CognitiveFluencyTransformer,
    # Theory-aligned (4)
    NominativeRichnessTransformer,
    CouplingStrengthTransformer,
    NarrativeMassTransformer,
    GravitationalFeaturesTransformer,
    # Phase 7 - Missing framework variables (4)
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer,
    # Note: AlphaTransformer and GoldenNarratioTransformer require y, added during analysis
    # Statistical baseline
    StatisticalTransformer,
)

# Import caching infrastructure
from narrative_optimization.src.pipelines.cached_pipeline import CachedTransformerPipeline


# Domain configurations
DOMAIN_CONFIGS = {
    'lottery': {
        'pi': 0.04,
        'data_path': 'data/domains/lottery_draws.json',
        'text_field': 'description',
        'outcome_field': 'win',
        'description': 'Pure randomness control'
    },
    'aviation': {
        'pi': 0.12,
        'data_path': 'data/domains/aviation/aviation_incidents_narratives.json',
        'text_field': 'narrative',
        'outcome_field': 'severity_binary',
        'description': 'Engineering-dominated'
    },
    'nba': {
        'pi': 0.49,
        'data_path': 'data/domains/nba_games_with_nominative.json',
        'text_field': 'narrative',
        'outcome_field': 'home_win',
        'description': 'Team sport, physical skill'
    },
    'nfl': {
        'pi': 0.57,
        'data_path': 'data/domains/nfl_complete_dataset.json',
        'text_field': 'narrative',
        'outcome_field': 'home_win',
        'description': 'Team sport, fractal structure'
    },
    'mental_health': {
        'pi': 0.55,
        'data_path': 'mental_health_complete_200_disorders.json',
        'text_field': 'full_description',
        'outcome_field': 'stigma_high',
        'description': 'Medical/social stigma'
    },
    'imdb': {
        'pi': 0.65,
        'data_path': 'data/domains/imdb_movies_complete.json',
        'text_field': 'narrative',
        'outcome_field': 'rating_high',
        'description': 'Entertainment, mixed'
    },
    'golf': {
        'pi': 0.70,
        'data_path': 'data/domains/golf_player_tournaments.json',
        'text_field': 'narrative',
        'outcome_field': 'top_10_finish',
        'description': 'Individual sport (sparse nominatives)'
    },
    'golf_enhanced': {
        'pi': 0.70,
        'data_path': 'data/domains/golf_enhanced_player_tournaments.json',
        'text_field': 'narrative_enhanced',
        'outcome_field': 'top_10_finish',
        'description': 'Individual sport (rich nominatives)'
    },
    'ufc': {
        'pi': 0.722,
        'data_path': 'data/domains/ufc_with_narratives.json',
        'text_field': 'narrative',
        'outcome_field': 'fighter1_win',
        'description': 'Combat sport, performance-dominated'
    },
    'tennis': {
        'pi': 0.75,
        'data_path': 'data/domains/tennis_complete_dataset.json',
        'text_field': 'narrative',
        'outcome_field': 'player1_win',
        'description': 'Individual sport, mental game'
    },
    # 'crypto': {
    #     'pi': 0.76,
    #     'data_path': None,  # Removed during workspace cleanup
    #     'text_field': 'texts',
    #     'outcome_field': 'labels_binary',
    #     'description': 'Speculation, market performance'
    # },
    'startups': {
        'pi': 0.76,
        'data_path': 'data/domains/startups_verified.json',
        'text_field': 'narrative',
        'outcome_field': 'success',
        'description': 'Business, funding outcomes'
    },
    'oscars': {
        'pi': 0.75,
        'data_path': 'data/domains/oscar_nominees_complete.json',
        'text_field': 'narrative',
        'outcome_field': 'won',
        'description': 'Entertainment competition'
    },
    'housing': {
        'pi': 0.92,
        'data_path': 'Housing/data/housing_with_number13.json',
        'text_field': 'description',
        'outcome_field': 'number_13',
        'description': 'Pure nominative effect'
    },
    'self_rated': {
        'pi': 0.95,
        'data_path': 'data/domains/self_rated_narratives.json',
        'text_field': 'narrative',
        'outcome_field': 'self_rating',
        'description': 'Identity, perfect coupling'
    },
    'wwe': {
        'pi': 0.974,
        'data_path': 'wwe/data/wwe_entities.json',
        'text_field': 'narrative',
        'outcome_field': 'kayfabe_success',
        'description': 'Prestige domain, constructed reality'
    }
}


def load_domain_data(domain_name, config, project_root):
    """Load data for a domain."""
    data_path = project_root / config['data_path']
    
    if not data_path.exists():
        print(f"  ⚠️  Data file not found: {data_path}")
        return None, None
    
    try:
        if data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
        elif data_path.suffix == '.pkl':
            import pickle
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"  ⚠️  Unsupported file format: {data_path.suffix}")
            return None, None
        
        # Extract texts and outcomes
        if isinstance(data, dict) and 'texts' in data:
            # Format C (crypto)
            texts = data['texts']
            outcomes = data.get(config['outcome_field'], None)
        elif isinstance(data, list):
            # List of dicts
            texts = [item.get(config['text_field'], '') for item in data]
            outcomes = [item.get(config['outcome_field'], 0) for item in data]
        else:
            print(f"  ⚠️  Unexpected data format")
            return None, None
        
        # Filter empty texts
        valid_indices = [i for i, text in enumerate(texts) if text and len(text.strip()) > 0]
        texts = [texts[i] for i in valid_indices]
        
        if outcomes is not None:
            outcomes = [outcomes[i] for i in valid_indices]
        
        print(f"  ✓ Loaded {len(texts)} samples")
        return texts, outcomes
    
    except Exception as e:
        print(f"  ⚠️  Error loading data: {e}")
        return None, None


def instantiate_transformers(domain_pi):
    """
    Instantiate all transformers for a domain.
    
    Parameters
    ----------
    domain_pi : float
        Domain narrativity value
    
    Returns
    -------
    transformers : list
        List of transformer instances
    """
    transformers = [
        # Core transformers (always used)
        NominativeAnalysisTransformer(),
        SelfPerceptionTransformer(),
        NarrativePotentialTransformer(),
        LinguisticPatternsTransformer(),
        RelationalValueTransformer(),
        EnsembleNarrativeTransformer(),
        
        # Structural (for all domains)
        ConflictTensionTransformer(),
        SuspenseMysteryTransformer(),
        FramingTransformer(),
        
        # Credibility
        AuthenticityTransformer(),
        ExpertiseAuthorityTransformer(),
        
        # Contextual
        TemporalEvolutionTransformer(),
        CulturalContextTransformer(),
        
        # Nominative (critical for all)
        PhoneticTransformer(),
        SocialStatusTransformer(),
        UniversalNominativeTransformer(),
        
        # Advanced
        InformationTheoryTransformer(),
        NamespaceEcologyTransformer(),
        AnticipatoryCommunicationTransformer(),
        CognitiveFluencyTransformer(),
        
        # Theory-aligned (Phase 6 - critical variables)
        NominativeRichnessTransformer(),
        CouplingStrengthTransformer(),
        NarrativeMassTransformer(),
        GravitationalFeaturesTransformer(),
        
        # Phase 7 - Missing framework variables (NEW - Complete coverage)
        AwarenessResistanceTransformer(),
        FundamentalConstraintsTransformer(),
        # Note: AlphaTransformer and GoldenNarratioTransformer require outcomes (y)
        # These are added during domain-specific analysis, not here
        
        # Statistical baseline
        StatisticalTransformer(),
    ]
    
    return transformers


def run_domain(domain_name, config, pipeline, project_root):
    """
    Run all transformers on a single domain.
    
    Parameters
    ----------
    domain_name : str
        Domain identifier
    config : dict
        Domain configuration
    pipeline : CachedTransformerPipeline
        Cached pipeline instance
    project_root : Path
        Project root directory
    
    Returns
    -------
    stats : dict
        Execution statistics
    """
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain_name.upper()} (π = {config['pi']})")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    
    # Load data
    print("\nLoading data...")
    texts, outcomes = load_domain_data(domain_name, config, project_root)
    
    if texts is None:
        return {
            'domain': domain_name,
            'status': 'failed',
            'error': 'Data loading failed'
        }
    
    # Instantiate transformers
    print(f"\nInstantiating transformers...")
    transformers = instantiate_transformers(config['pi'])
    print(f"  • {len(transformers)} transformers ready")
    
    # Execute with caching
    try:
        features, stats = pipeline.execute_transformers(
            domain=domain_name,
            transformers=transformers,
            data=texts,
            y=outcomes,
            force_recompute=False,
            skip_on_error=True
        )
        
        # Save features
        output_path = project_root / 'narrative_optimization' / 'data' / 'features' / f'{domain_name}_all_features.npz'
        pipeline.save_features(
            features=features,
            stats=stats,
            output_path=output_path,
            feature_names=stats.get('feature_names')
        )
        
        stats['status'] = 'success'
        stats['output_file'] = str(output_path)
        
        return stats
    
    except Exception as e:
        print(f"\n❌ Error processing {domain_name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'domain': domain_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("UNIFIED TRANSFORMER EXECUTION")
    print("Running all 34+ transformers across all 16 domains")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize cached pipeline
    cache_dir = project_root / 'narrative_optimization' / 'data' / 'features' / 'cache'
    pipeline = CachedTransformerPipeline(cache_dir=str(cache_dir), verbose=True)
    
    # Execute for each domain
    all_stats = []
    
    for domain_name, config in DOMAIN_CONFIGS.items():
        domain_stats = run_domain(domain_name, config, pipeline, project_root)
        all_stats.append(domain_stats)
    
    # Generate summary report
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    
    # Count successes/failures
    successes = sum(1 for s in all_stats if s['status'] == 'success')
    failures = len(all_stats) - successes
    
    print(f"\nSummary:")
    print(f"  • Domains processed: {len(all_stats)}")
    print(f"  • Successful: {successes}")
    print(f"  • Failed: {failures}")
    print(f"  • Total time: {total_time/60:.1f} minutes")
    
    # Cache statistics
    cache_stats = pipeline.get_cache_stats()
    print(f"\nCache Performance:")
    print(f"  • Hit rate: {cache_stats['cache']['hit_rate']}")
    print(f"  • Total entries: {cache_stats['cache']['total_entries']}")
    
    # Save execution report
    report = {
        'execution_date': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'domains_processed': len(all_stats),
        'successful': successes,
        'failed': failures,
        'cache_stats': cache_stats,
        'domain_results': all_stats
    }
    
    report_path = project_root / 'narrative_optimization' / 'data' / 'features' / 'extraction_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report saved: {report_path}")
    
    # Print per-domain summary
    print("\nPer-Domain Results:")
    print("-" * 70)
    for stats in all_stats:
        if stats['status'] == 'success':
            print(f"  ✅ {stats['domain']:20s} | "
                  f"Features: {stats['total_features']:4d} | "
                  f"Cache: {stats['cache_hit_rate']:6s} | "
                  f"Time: {stats['total_time']:5.1f}s")
        else:
            print(f"  ❌ {stats['domain']:20s} | Error: {stats.get('error', 'Unknown')}")
    
    print("\n" + "="*70)
    print("✅ ALL TRANSFORMERS EXECUTED ACROSS ALL DOMAINS")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

