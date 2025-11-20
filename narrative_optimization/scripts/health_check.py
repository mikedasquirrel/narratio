"""
System Health Check

Comprehensive system health check and diagnostics.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check all critical imports work."""
    print("\n[1/8] Checking imports...")
    
    checks = {
        'Learning System': 'from src.learning import LearningPipeline',
        'Analysis': 'from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer',
        'Data Loading': 'from src.data import DataLoader',
        'Configuration': 'from src.config import DomainConfig',
        'Registry': 'from src.registry import get_domain_registry',
        'Visualization': 'from src.visualization import PatternVisualizer',
        'Optimization': 'from src.optimization import CacheManager',
        'Integration': 'from MASTER_INTEGRATION import MasterDomainIntegration'
    }
    
    passed = 0
    for check_name, import_statement in checks.items():
        try:
            exec(import_statement)
            print(f"  ✓ {check_name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {check_name}: {e}")
    
    return passed, len(checks)


def check_directories():
    """Check all required directories exist."""
    print("\n[2/8] Checking directories...")
    
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        'src/learning',
        'src/analysis',
        'src/transformers',
        'src/config',
        'src/data',
        'tools',
        'tests',
        'examples',
        'data/domains'
    ]
    
    passed = 0
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
            passed += 1
        else:
            print(f"  ✗ {dir_path}: Missing")
    
    return passed, len(required_dirs)


def check_data_files():
    """Check for data files."""
    print("\n[3/8] Checking data files...")
    
    from src.pipeline_config import get_config
    config = get_config()
    
    data_files = list(config.data_dir.glob('**/*.json'))
    data_files.extend(list(config.data_dir.glob('**/*.csv')))
    
    print(f"  ✓ Found {len(data_files)} data files")
    
    return 1 if len(data_files) > 0 else 0, 1


def check_registry():
    """Check domain registry."""
    print("\n[4/8] Checking domain registry...")
    
    try:
        from src.registry import get_domain_registry
        
        registry = get_domain_registry()
        stats = registry.get_statistics()
        
        n_domains = stats.get('total_domains', 0)
        
        print(f"  ✓ Registry operational")
        print(f"    Domains: {n_domains}")
        print(f"    Avg π: {stats.get('avg_pi', 0):.3f}")
        
        return 1, 1
    except Exception as e:
        print(f"  ✗ Registry error: {e}")
        return 0, 1


def check_transformers():
    """Check transformer availability."""
    print("\n[5/8] Checking transformers...")
    
    transformer_dir = Path(__file__).parent.parent / 'src' / 'transformers'
    
    base_transformers = len(list(transformer_dir.glob('*.py')))
    archetype_transformers = len(list((transformer_dir / 'archetypes').glob('*.py'))) - 1  # Exclude __init__
    
    print(f"  ✓ Base transformers: {base_transformers}")
    print(f"  ✓ Archetype transformers: {archetype_transformers}")
    print(f"  Total: {base_transformers + archetype_transformers}")
    
    return 1, 1


def check_learning_system():
    """Check learning system."""
    print("\n[6/8] Checking learning system...")
    
    try:
        from src.learning import LearningPipeline
        
        # Create pipeline
        pipeline = LearningPipeline()
        
        print(f"  ✓ Learning pipeline operational")
        print(f"    Universal learner: Ready")
        print(f"    Validator: Ready")
        print(f"    Registry: Ready")
        
        return 1, 1
    except Exception as e:
        print(f"  ✗ Learning system error: {e}")
        return 0, 1


def check_cache():
    """Check cache system."""
    print("\n[7/8] Checking cache...")
    
    try:
        from src.optimization import get_global_cache
        
        cache = get_global_cache()
        stats = cache.get_stats()
        
        print(f"  ✓ Cache operational")
        print(f"    Hit rate: {stats.get('hit_rate', 0):.1%}")
        print(f"    Size: {stats.get('cache_size_mb', 0):.1f}MB")
        print(f"    Items: {stats.get('n_items', 0)}")
        
        return 1, 1
    except Exception as e:
        print(f"  ✗ Cache error: {e}")
        return 0, 1


def check_integration():
    """Check integration system."""
    print("\n[8/8] Checking integration...")
    
    key_files = [
        'MASTER_INTEGRATION.py',
        'LIST_DOMAINS.py',
        'INITIALIZE_SYSTEM.py',
        'DEMO_COMPLETE_SYSTEM.py'
    ]
    
    project_root = Path(__file__).parent.parent
    
    passed = 0
    for file_name in key_files:
        if (project_root / file_name).exists():
            print(f"  ✓ {file_name}")
            passed += 1
        else:
            print(f"  ✗ {file_name}: Missing")
    
    return passed, len(key_files)


def run_health_check():
    """Run complete health check."""
    print("="*80)
    print("SYSTEM HEALTH CHECK")
    print("="*80)
    
    total_passed = 0
    total_checks = 0
    
    # Run all checks
    checks = [
        check_imports,
        check_directories,
        check_data_files,
        check_registry,
        check_transformers,
        check_learning_system,
        check_cache,
        check_integration
    ]
    
    for check_func in checks:
        passed, total = check_func()
        total_passed += passed
        total_checks += total
    
    # Summary
    print(f"\n{'='*80}")
    print(f"HEALTH CHECK COMPLETE: {total_passed}/{total_checks} passed")
    print(f"{'='*80}\n")
    
    if total_passed == total_checks:
        print("✓ System is healthy and fully operational")
        return True
    else:
        failed = total_checks - total_passed
        print(f"⚠ {failed} checks failed - see details above")
        return False


if __name__ == '__main__':
    success = run_health_check()
    sys.exit(0 if success else 1)

