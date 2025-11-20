"""
System Initialization

Initializes and validates the complete narrative optimization system.

Run this first to ensure everything is properly set up.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline_config import PipelineConfig, set_config
from src.learning import LearningPipeline
from src.registry import get_domain_registry
from src.optimization import get_global_cache


def initialize_system():
    """Initialize complete system."""
    print("="*80)
    print("INITIALIZING NARRATIVE OPTIMIZATION SYSTEM")
    print("="*80)
    
    success_count = 0
    total_checks = 8
    
    # 1. Initialize configuration
    print("\n[1/8] Initializing configuration...")
    try:
        config = PipelineConfig()
        set_config(config)
        print(f"  ✓ Configuration initialized")
        print(f"    Data dir: {config.data_dir}")
        print(f"    Output dir: {config.output_dir}")
        print(f"    Cache dir: {config.cache_dir}")
        success_count += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 2. Initialize learning pipeline
    print("\n[2/8] Initializing learning pipeline...")
    try:
        pipeline = LearningPipeline(config=config)
        print(f"  ✓ Learning pipeline initialized")
        print(f"    Universal learner: Ready")
        print(f"    Validator: Ready")
        print(f"    Registry: Ready")
        success_count += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 3. Initialize domain registry
    print("\n[3/8] Initializing domain registry...")
    try:
        registry = get_domain_registry()
        print(f"  ✓ Domain registry initialized")
        print(f"    Registered domains: {len(registry.domains)}")
        success_count += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 4. Initialize cache
    print("\n[4/8] Initializing cache system...")
    try:
        cache = get_global_cache()
        cache_stats = cache.get_stats()
        print(f"  ✓ Cache system initialized")
        print(f"    Cache dir: {cache.cache_dir}")
        print(f"    Current items: {cache_stats.get('n_items', 0)}")
        success_count += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 5. Verify data directory
    print("\n[5/8] Verifying data directory...")
    try:
        data_dir = config.data_dir
        if data_dir.exists():
            data_files = list(data_dir.glob('*.json'))
            print(f"  ✓ Data directory exists")
            print(f"    Found {len(data_files)} data files")
            success_count += 1
        else:
            print(f"  ⚠ Data directory doesn't exist (will be created)")
            data_dir.mkdir(parents=True, exist_ok=True)
            success_count += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 6. Verify transformers
    print("\n[6/8] Verifying transformers...")
    try:
        transformers_dir = Path(__file__).parent / 'src' / 'transformers'
        archetypes_dir = transformers_dir / 'archetypes'
        
        transformer_files = list(transformers_dir.glob('*.py'))
        archetype_files = list(archetypes_dir.glob('*.py'))
        
        print(f"  ✓ Transformers directory verified")
        print(f"    Base transformers: {len(transformer_files)}")
        print(f"    Domain archetypes: {len(archetype_files) - 1}")  # Exclude __init__
        success_count += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 7. Verify learning modules
    print("\n[7/8] Verifying learning modules...")
    try:
        learning_dir = Path(__file__).parent / 'src' / 'learning'
        learning_files = list(learning_dir.glob('*.py'))
        
        required_modules = [
            'learning_pipeline.py',
            'universal_learner.py',
            'domain_learner.py',
            'validation_engine.py',
            'registry_versioned.py'
        ]
        
        missing = [m for m in required_modules if not (learning_dir / m).exists()]
        
        if len(missing) == 0:
            print(f"  ✓ All learning modules present")
            print(f"    Total modules: {len(learning_files) - 1}")  # Exclude __init__
            success_count += 1
        else:
            print(f"  ✗ Missing modules: {', '.join(missing)}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 8. Test imports
    print("\n[8/8] Testing critical imports...")
    try:
        from src.learning import LearningPipeline, UniversalArchetypeLearner
        from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer
        from src.data import DataLoader
        from src.config import DomainConfig
        from MASTER_INTEGRATION import MasterDomainIntegration
        
        print(f"  ✓ All critical imports successful")
        success_count += 1
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"INITIALIZATION COMPLETE: {success_count}/{total_checks} checks passed")
    print(f"{'='*80}\n")
    
    if success_count == total_checks:
        print("✓ System fully operational and ready for use\n")
        print("Next steps:")
        print("  1. Run demo: python examples/learning_pipeline_demo.py")
        print("  2. Add domain: python MASTER_INTEGRATION.py DOMAIN data/domains/DOMAIN.json")
        print("  3. List domains: python LIST_DOMAINS.py --list")
        print("  4. Run tests: pytest tests/ -v")
        return True
    else:
        print(f"⚠ {total_checks - success_count} checks failed - see errors above\n")
        return False


if __name__ == '__main__':
    success = initialize_system()
    sys.exit(0 if success else 1)

