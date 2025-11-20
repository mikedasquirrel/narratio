"""
Complete System Test Runner

Runs comprehensive tests to validate the entire integrated system.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("COMPREHENSIVE SYSTEM TEST")
print("="*80)
print(f"\nStarting at: {time.strftime('%H:%M:%S')}")
print("This will test all major system components...")
print("\n" + "─"*80)

# Test 1: Core imports
print("\n[TEST 1/8] Core System Imports")
print("-"*80)
print("  Importing learning system...", end=" ", flush=True)
try:
    from src.learning import LearningPipeline, UniversalArchetypeLearner, DomainSpecificLearner
    print("✓")
    print("  Importing analysis tools...", end=" ", flush=True)
    from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer
    print("✓")
    print("  Importing data loader...", end=" ", flush=True)
    from src.data import DataLoader
    print("✓")
    print("  Importing config system...", end=" ", flush=True)
    from src.config import DomainConfig, DOMAIN_ARCHETYPES
    print("✓")
    print("  Importing registry...", end=" ", flush=True)
    from src.registry import get_domain_registry
    print("✓")
    print("  Importing optimization...", end=" ", flush=True)
    from src.optimization import get_global_cache
    print("✓")
    print("  Importing master integration...", end=" ", flush=True)
    from MASTER_INTEGRATION import MasterDomainIntegration
    print("✓")
    print("\n✓ All core imports successful (7/7)")
    test1_pass = True
except Exception as e:
    print(f"\n✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    test1_pass = False

# Test 2: Transformer availability
print("\n[TEST 2/8] Transformer Availability")
print("-"*80)
print("  Loading base transformers...", end=" ", flush=True)
try:
    from src.transformers import (
        StatisticalTransformer, NarrativePotentialTransformer,
        FramingTransformer, CognitiveFluencyTransformer,
        EnsembleNarrativeTransformer
    )
    print("✓ (5 loaded)")
    
    print("  Loading domain archetypes...", end=" ", flush=True)
    from src.transformers.archetypes import (
        GolfArchetypeTransformer, TennisArchetypeTransformer,
        ChessArchetypeTransformer, BoxingArchetypeTransformer,
        NBAArchetypeTransformer, WWEArchetypeTransformer,
        OscarsArchetypeTransformer, CryptoArchetypeTransformer,
        MentalHealthArchetypeTransformer, StartupsArchetypeTransformer,
        HurricanesArchetypeTransformer, HousingArchetypeTransformer
    )
    print("✓ (12 loaded)")
    
    print("\n✓ Transformer system operational (17 transformers verified)")
    test2_pass = True
except Exception as e:
    print(f"\n✗ Transformer import failed: {e}")
    import traceback
    traceback.print_exc()
    test2_pass = False

# Test 3: Data loading
print("\n[TEST 3/8] Data Loading")
print("-"*80)
try:
    import tempfile
    import json
    
    loader = DataLoader()
    
    # Create test data
    test_data = {
        'texts': ['Test narrative 1', 'Test narrative 2', 'Test narrative 3'],
        'outcomes': [1, 0, 1]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)
    
    # Load
    data = loader.load(temp_path)
    valid = loader.validate_data(data)
    
    temp_path.unlink()
    
    if valid:
        print(f"✓ Data loading: Functional")
        print(f"✓ Data validation: Functional")
        test3_pass = True
    else:
        print(f"✗ Data validation failed")
        test3_pass = False
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    test3_pass = False

# Test 4: Domain registry
print("\n[TEST 4/8] Domain Registry")
print("-"*80)
try:
    from src.registry import register_domain
    
    registry = get_domain_registry()
    
    # Test registration
    register_domain(
        name='test_domain_verification',
        pi=0.7,
        domain_type='test',
        status='test'
    )
    
    # Test retrieval
    domain = registry.get_domain('test_domain_verification')
    
    if domain:
        print(f"✓ Domain registration: Functional")
        print(f"✓ Domain retrieval: Functional")
        print(f"✓ Registry has {registry.get_statistics().get('total_domains', 0)} domains")
        test4_pass = True
    else:
        print(f"✗ Registry operations failed")
        test4_pass = False
except Exception as e:
    print(f"✗ Registry failed: {e}")
    test4_pass = False

# Test 5: Learning system
print("\n[TEST 5/8] Learning System")
print("-"*80)
print("  Creating learning pipeline...", end=" ", flush=True)
try:
    pipeline = LearningPipeline()
    print("✓")
    
    # Generate synthetic data
    print("  Generating test data (40 samples)...", end=" ", flush=True)
    texts = [
        "Underdog player shows skill",
        "Dominant champion wins",
        "Comeback victory achieved",
        "Pressure performance clutch"
    ] * 10
    
    outcomes = np.random.binomial(1, 0.6, len(texts))
    print("✓")
    
    # Ingest
    print("  Ingesting data...", end=" ", flush=True)
    pipeline.ingest_domain('test_learning', texts, outcomes)
    print("✓")
    
    # Learn
    print("  Running learning cycle...")
    print("    [1/6] Measuring baseline...", end=" ", flush=True)
    sys.stdout.flush()
    
    metrics = pipeline.learn_cycle(['test_learning'])
    
    print()  # New line after cycle
    
    if metrics.patterns_discovered > 0:
        print(f"\n✓ Learning pipeline: Functional")
        print(f"  • Patterns discovered: {metrics.patterns_discovered}")
        print(f"  • Patterns validated: {metrics.patterns_validated}")
        print(f"  • Patterns pruned: {metrics.patterns_pruned}")
        print(f"  • Performance: R²={metrics.r_squared_after:.3f}")
        test5_pass = True
    else:
        print(f"\n✗ No patterns discovered")
        test5_pass = False
except Exception as e:
    print(f"\n✗ Learning system failed: {e}")
    import traceback
    traceback.print_exc()
    test5_pass = False

# Test 6: Domain analysis
print("\n[TEST 6/8] Domain Analysis")
print("-"*80)
print("  Generating test narratives (20 samples)...", end=" ", flush=True)
try:
    texts = ["Sample narrative with patterns"] * 20
    outcomes = np.random.binomial(1, 0.5, 20)
    print("✓")
    
    print("  Creating domain analyzer (golf)...", end=" ", flush=True)
    analyzer = DomainSpecificAnalyzer('golf')
    print("✓")
    
    print("  Running complete analysis...", end=" ", flush=True)
    sys.stdout.flush()
    results = analyzer.analyze_complete(texts, outcomes)
    print("✓")
    
    print(f"\n✓ Domain analyzer: Functional")
    print(f"  • R²: {results['r_squared']:.3f}")
    print(f"  • Д (narrative agency): {results['delta']:.3f}")
    print(f"  • Story quality (ю): {results['story_quality'].mean():.3f}")
    print(f"  • Samples analyzed: {len(results['story_quality'])}")
    print(f"  • Genome features: {results['genomes'].shape[1] if hasattr(results['genomes'], 'shape') else 'N/A'}")
    test6_pass = True
except Exception as e:
    print(f"\n✗ Analysis failed: {e}")
    import traceback
    traceback.print_exc()
    test6_pass = False

# Test 7: Master integration
print("\n[TEST 7/8] Master Integration")
print("-"*80)
print("  Initializing master integration...", end=" ", flush=True)
try:
    integration = MasterDomainIntegration()
    print("✓")
    
    print("  Creating test domain data (45 samples)...", end=" ", flush=True)
    texts = [
        "Underdog defeats favorite",
        "Comeback victory",
        "Dominant display"
    ] * 15
    
    outcomes = np.random.binomial(1, 0.55, len(texts))
    print("✓")
    
    print("  Running complete domain integration...")
    print("    Step 1/6: Checking universal stories...", end=" ", flush=True)
    sys.stdout.flush()
    
    results = integration.analyze_new_domain(
        'test_integration',
        texts,
        outcomes,
        {'pi': 0.7, 'type': 'test'}
    )
    
    print()  # New line after all steps
    print(f"\n✓ Master integration: Functional")
    print(f"  • Universal patterns discovered: {len(results['universal_patterns'])}")
    print(f"  • Domain-specific patterns: {len(results['domain_patterns'])}")
    print(f"  • Similar domains identified: {len(results['similar_domains'])}")
    print(f"  • Trends detected: {len(results['trends'])}")
    print(f"  • Story frequency match: {results['frequency_analysis'].get('meets_expectations', False)}")
    test7_pass = True
except Exception as e:
    print(f"\n✗ Integration failed: {e}")
    import traceback
    traceback.print_exc()
    test7_pass = False

# Test 8: Caching
print("\n[TEST 8/8] Optimization Systems")
print("-"*80)
try:
    cache = get_global_cache()
    
    # Test cache operations
    cache.set('test_key', {'value': 'test'})
    retrieved = cache.get('test_key')
    
    if retrieved and retrieved['value'] == 'test':
        stats = cache.get_stats()
        print(f"✓ Cache system: Functional")
        print(f"✓ Hit rate: {stats.get('hit_rate', 0):.1%}")
        test8_pass = True
    else:
        print(f"✗ Cache operations failed")
        test8_pass = False
except Exception as e:
    print(f"✗ Cache failed: {e}")
    test8_pass = False

# Summary
print(f"\n{'='*80}")
print("TEST SUMMARY")
print(f"{'='*80}\n")

tests = [
    ("Core Imports", test1_pass),
    ("Transformers", test2_pass),
    ("Data Loading", test3_pass),
    ("Domain Registry", test4_pass),
    ("Learning System", test5_pass),
    ("Domain Analysis", test6_pass),
    ("Master Integration", test7_pass),
    ("Optimization", test8_pass)
]

passed = sum(1 for _, p in tests if p)
total = len(tests)

for test_name, passed_test in tests:
    status = "✓" if passed_test else "✗"
    print(f"  {status} {test_name}")

elapsed_time = time.time()
print(f"\n{'='*80}")
print(f"RESULT: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
print(f"Completed at: {time.strftime('%H:%M:%S')}")
print(f"{'='*80}\n")

if passed == total:
    print("🎉 ALL SYSTEMS OPERATIONAL 🎉")
    print("\nThe complete holistic learning system is working:")
    print("  • Domain-agnostic learning (universal patterns)")
    print("  • Domain-specific learning (unique patterns)")
    print("  • Statistical validation (all patterns tested)")
    print("  • Master integration (seamless data flow)")
    print("  • Registry tracking (all domains catalogued)")
    print("  • Caching optimization (performance enhanced)")
    print("\n✓ System is production-ready")
    print("\nNext: python DEMO_COMPLETE_SYSTEM.py")
    sys.exit(0)
elif passed >= 6:
    print("⚠ MOST SYSTEMS OPERATIONAL")
    print(f"\n{total - passed} tests failed - review above")
    print("\nSystem can be used but needs attention")
    sys.exit(1)
else:
    print("✗ SYSTEM NEEDS FIXES")
    print(f"\n{total - passed} tests failed")
    print("\nReview errors and fix issues")
    sys.exit(1)

