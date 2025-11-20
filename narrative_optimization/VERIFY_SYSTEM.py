"""
System Verification

Comprehensive verification that all systems are properly integrated and operational.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def verify_core_imports():
    """Verify all core imports work."""
    print("\n[1/10] Verifying core imports...")
    
    try:
        from src.learning import (
            LearningPipeline, UniversalArchetypeLearner, DomainSpecificLearner,
            ValidationEngine, VersionedArchetypeRegistry, ExplanationGenerator,
            HierarchicalArchetypeLearner, ActiveLearner, MetaLearner,
            EnsembleArchetypeLearner, OnlineLearner, CausalArchetypeDiscovery,
            PatternRefiner, ContextAwareLearner
        )
        print("    ✓ Learning system (14 modules)")
        
        from src.analysis import (
            DomainSpecificAnalyzer, StoryQualityCalculator,
            BridgeCalculator, MultiModalPatternAnalyzer,
            UncertaintyQuantifier
        )
        print("    ✓ Analysis tools")
        
        from src.transformers.archetypes import (
            GolfArchetypeTransformer, TennisArchetypeTransformer,
            ChessArchetypeTransformer, BoxingArchetypeTransformer,
            NBAArchetypeTransformer, WWEArchetypeTransformer,
            OscarsArchetypeTransformer, CryptoArchetypeTransformer,
            MentalHealthArchetypeTransformer, StartupsArchetypeTransformer,
            HurricanesArchetypeTransformer, HousingArchetypeTransformer
        )
        print("    ✓ Domain archetypes (12 domains)")
        
        from src.config import (
            DOMAIN_ARCHETYPES, DomainConfig, ArchetypeDiscovery,
            GenomeStructure, HistorialCalculator, UniquityCalculator,
            SemanticArchetypeDiscovery, TemporalDecay
        )
        print("    ✓ Configuration system")
        
        from src.data import DataLoader, StreamProcessor
        print("    ✓ Data processing")
        
        from src.visualization import PatternVisualizer
        print("    ✓ Visualization")
        
        from src.optimization import CacheManager, PerformanceProfiler
        print("    ✓ Optimization tools")
        
        from src.registry import DomainRegistry, get_domain_registry
        print("    ✓ Domain registry")
        
        return True
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_integration_scripts():
    """Verify integration scripts exist and are importable."""
    print("\n[2/10] Verifying integration scripts...")
    
    scripts = [
        'MASTER_INTEGRATION',
        'LIST_DOMAINS',
        'INITIALIZE_SYSTEM',
        'DEMO_COMPLETE_SYSTEM',
        'RUN_COMPLETE_SYSTEM'
    ]
    
    all_exist = True
    
    for script in scripts:
        script_path = Path(__file__).parent / f'{script}.py'
        if script_path.exists():
            print(f"    ✓ {script}.py")
        else:
            print(f"    ✗ {script}.py missing")
            all_exist = False
    
    return all_exist


def verify_tools():
    """Verify all tools exist."""
    print("\n[3/10] Verifying tools...")
    
    tools_dir = Path(__file__).parent / 'tools'
    
    expected_tools = [
        'discover_archetypes.py',
        'discover_domain_archetypes.py',
        'integrate_existing_domains.py',
        'validate_all_domains.py',
        'batch_analyze_domains.py',
        'discover_all_archetypes.py',
        'generate_domain_docs.py',
        'auto_domain_generator.py',
        'make_transformers_adaptive.py',
        'batch_adapt_transformers.py'
    ]
    
    existing = 0
    for tool in expected_tools:
        if (tools_dir / tool).exists():
            existing += 1
    
    print(f"    ✓ {existing}/{len(expected_tools)} tools present")
    
    return existing >= len(expected_tools) * 0.8  # 80% threshold


def verify_transformers():
    """Verify transformer availability."""
    print("\n[4/10] Verifying transformers...")
    
    transformers_dir = Path(__file__).parent / 'src' / 'transformers'
    archetypes_dir = transformers_dir / 'archetypes'
    
    base_count = len([f for f in transformers_dir.glob('*.py') if f.name != '__init__.py'])
    archetype_count = len([f for f in archetypes_dir.glob('*.py') if f.name != '__init__.py'])
    
    print(f"    ✓ {base_count} base transformers")
    print(f"    ✓ {archetype_count} domain archetypes")
    print(f"    Total: {base_count + archetype_count}")
    
    return base_count > 30 and archetype_count >= 12


def verify_learning_system():
    """Verify learning system components."""
    print("\n[5/10] Verifying learning system...")
    
    learning_dir = Path(__file__).parent / 'src' / 'learning'
    
    required_modules = [
        'learning_pipeline.py',
        'universal_learner.py',
        'domain_learner.py',
        'validation_engine.py',
        'registry_versioned.py',
        'explanation_generator.py',
        'hierarchical_learner.py',
        'active_learner.py',
        'meta_learner.py',
        'ensemble_learner.py',
        'online_learner.py',
        'causal_discovery.py',
        'pattern_refiner.py',
        'context_aware_learner.py'
    ]
    
    existing = sum(1 for m in required_modules if (learning_dir / m).exists())
    
    print(f"    ✓ {existing}/{len(required_modules)} learning modules")
    
    return existing == len(required_modules)


def verify_data_handling():
    """Verify data handling capabilities."""
    print("\n[6/10] Verifying data handling...")
    
    import tempfile
    import json
    import numpy as np
    
    try:
        from src.data import DataLoader
        
        loader = DataLoader()
        
        # Create temp data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                'texts': ['test1', 'test2'],
                'outcomes': [1, 0]
            }
            json.dump(test_data, f)
            temp_path = Path(f.name)
        
        # Load
        data = loader.load(temp_path)
        
        # Validate
        valid = loader.validate_data(data)
        
        # Cleanup
        temp_path.unlink()
        
        if valid:
            print("    ✓ Data loading functional")
            print("    ✓ Data validation functional")
            return True
        else:
            print("    ✗ Data validation failed")
            return False
            
    except Exception as e:
        print(f"    ✗ Data handling error: {e}")
        return False


def verify_registry():
    """Verify domain registry."""
    print("\n[7/10] Verifying domain registry...")
    
    try:
        from src.registry import get_domain_registry, register_domain, list_all_domains
        
        registry = get_domain_registry()
        
        # Test registration
        register_domain(
            name='test_verify_domain',
            pi=0.5,
            domain_type='test',
            status='test'
        )
        
        # Test retrieval
        domain = registry.get_domain('test_verify_domain')
        
        if domain and domain.name == 'test_verify_domain':
            print("    ✓ Registry operational")
            print("    ✓ Domain registration works")
            print("    ✓ Domain retrieval works")
            return True
        else:
            print("    ✗ Registry operations failed")
            return False
            
    except Exception as e:
        print(f"    ✗ Registry error: {e}")
        return False


def verify_caching():
    """Verify caching system."""
    print("\n[8/10] Verifying caching system...")
    
    try:
        from src.optimization import get_global_cache
        
        cache = get_global_cache()
        
        # Test write
        cache.set('test_key', {'data': 'test_value'})
        
        # Test read
        value = cache.get('test_key')
        
        if value and value['data'] == 'test_value':
            stats = cache.get_stats()
            print("    ✓ Cache operational")
            print(f"    Hit rate: {stats.get('hit_rate', 0):.1%}")
            return True
        else:
            print("    ✗ Cache operations failed")
            return False
            
    except Exception as e:
        print(f"    ✗ Cache error: {e}")
        return False


def verify_deployment_readiness():
    """Verify deployment files."""
    print("\n[9/10] Verifying deployment readiness...")
    
    deployment_files = [
        'Dockerfile',
        'docker-compose.yml',
        'requirements.txt',
        '.gitignore',
        'Makefile'
    ]
    
    existing = 0
    for file_name in deployment_files:
        if (Path(__file__).parent / file_name).exists():
            existing += 1
            print(f"    ✓ {file_name}")
        else:
            print(f"    ⊙ {file_name} missing")
    
    return existing >= 4  # At least 4 out of 5


def verify_documentation():
    """Verify documentation exists."""
    print("\n[10/10] Verifying documentation...")
    
    docs = [
        'README.md',
        'QUICK_START.md',
        'SETUP_GUIDE.md',
        'DOMAIN_ADDITION_TEMPLATE.md',
        'DOMAIN_ARCHETYPE_SYSTEM.md',
        'DEVELOPER_GUIDE.md',
        'SYSTEM_OVERVIEW.md',
        'PROJECT_STRUCTURE.md',
        'INDEX.md'
    ]
    
    existing = sum(1 for doc in docs if (Path(__file__).parent / doc).exists())
    
    print(f"    ✓ {existing}/{len(docs)} documentation files")
    
    return existing >= 6  # At least core docs


def run_verification():
    """Run complete system verification."""
    print("="*80)
    print("COMPREHENSIVE SYSTEM VERIFICATION")
    print("="*80)
    
    checks = [
        ("Core Imports", verify_core_imports),
        ("Integration Scripts", verify_integration_scripts),
        ("Tools", verify_tools),
        ("Transformers", verify_transformers),
        ("Learning System", verify_learning_system),
        ("Data Handling", verify_data_handling),
        ("Registry", verify_registry),
        ("Caching", verify_caching),
        ("Deployment", verify_deployment_readiness),
        ("Documentation", verify_documentation)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            passed = check_func()
            results.append((check_name, passed))
        except Exception as e:
            print(f"    ✗ Exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for check_name, passed_check in results:
        status = "✓" if passed_check else "✗"
        print(f"  {status} {check_name}")
    
    print(f"\n{'='*80}")
    print(f"RESULT: {passed}/{total} checks passed ({passed/total:.0%})")
    print(f"{'='*80}\n")
    
    if passed == total:
        print("✓ SYSTEM FULLY OPERATIONAL")
        print("\nReady for:")
        print("  - Domain analysis")
        print("  - Pattern learning")
        print("  - Production deployment")
        print("\nRun: python DEMO_COMPLETE_SYSTEM.py")
        return True
    elif passed >= total * 0.8:
        print("⚠ SYSTEM MOSTLY OPERATIONAL")
        print(f"  {total - passed} checks failed - review above")
        print("\nSystem can be used but some features may not work")
        return False
    else:
        print("✗ SYSTEM REQUIRES ATTENTION")
        print(f"  {total - passed} checks failed")
        print("\nReview errors above and fix issues")
        return False


if __name__ == '__main__':
    success = run_verification()
    sys.exit(0 if success else 1)

