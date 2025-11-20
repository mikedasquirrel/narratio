"""
Framework 2.0 Validation Tests

Tests all new components to ensure everything works correctly.

Run this to validate your installation:
    python test_framework_2_0.py

Author: Narrative Optimization Framework
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_story_instance():
    """Test StoryInstance class."""
    print("\n[TEST 1/7] StoryInstance Class")
    print("-" * 50)
    
    try:
        from core.story_instance import StoryInstance
        
        # Create instance
        instance = StoryInstance(
            instance_id="test_instance",
            domain="test_domain",
            narrative_text="This is a test narrative.",
            outcome=1.0
        )
        
        # Set genome
        instance.set_genome_components(
            nominative=np.random.rand(10),
            archetypal=np.random.rand(15),
            historial=np.random.rand(10),
            uniquity=np.random.rand(5)
        )
        
        # Calculate properties
        instance.importance_score = 1.5
        instance.stakes_multiplier = 2.0
        mass = instance.calculate_mass()
        
        # Test serialization
        instance.save('/tmp/test_instance.json')
        loaded = StoryInstance.load('/tmp/test_instance.json')
        
        assert loaded.instance_id == instance.instance_id
        assert loaded.mass == mass
        
        print("✓ StoryInstance: PASS")
        return True
        
    except Exception as e:
        print(f"✗ StoryInstance: FAIL - {e}")
        return False

def test_instance_repository():
    """Test InstanceRepository."""
    print("\n[TEST 2/7] InstanceRepository")
    print("-" * 50)
    
    try:
        from core.story_instance import StoryInstance
        from data.instance_repository import InstanceRepository
        
        # Create repository
        repo = InstanceRepository(storage_path='/tmp/test_repo')
        repo.clear()
        
        # Create test instances
        instances = []
        for i in range(10):
            inst = StoryInstance(
                instance_id=f"test_{i}",
                domain="test_domain",
                narrative_text=f"Narrative {i}",
                outcome=float(i % 2)
            )
            inst.pi_effective = 0.5 + i * 0.05
            instances.append(inst)
        
        # Add to repository
        repo.add_instances_bulk(instances)
        
        # Test queries
        domain_instances = repo.get_instances_by_domain("test_domain")
        assert len(domain_instances) == 10
        
        # Test structural query
        similar = repo.query_by_structure(pi_range=(0.6, 0.8))
        assert len(similar) > 0
        
        # Test statistics
        stats = repo.get_domain_statistics("test_domain")
        assert stats['count'] == 10
        
        print(f"✓ InstanceRepository: PASS (10 instances)")
        return True
        
    except Exception as e:
        print(f"✗ InstanceRepository: FAIL - {e}")
        return False

def test_domain_config_extensions():
    """Test DomainConfig new methods."""
    print("\n[TEST 3/7] DomainConfig Extensions")
    print("-" * 50)
    
    try:
        from config.domain_config import DomainConfig
        
        config = DomainConfig('golf')
        
        # Test new methods
        blind_narratio = config.get_blind_narratio()  # May be None
        pi_sensitivity = config.get_pi_sensitivity()
        assert pi_sensitivity >= 0
        
        pi_eff = config.calculate_effective_pi(complexity=0.5)
        assert 0 <= pi_eff <= 1
        
        theta_amp_range = config.get_awareness_amplification_range()
        assert len(theta_amp_range) == 2
        
        neighbors = config.get_imperative_gravity_neighbors()
        assert isinstance(neighbors, list)
        
        print(f"✓ DomainConfig: PASS")
        print(f"  π_base: {config.get_pi():.3f}")
        print(f"  π_effective(0.5): {pi_eff:.3f}")
        print(f"  β: {pi_sensitivity:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ DomainConfig: FAIL - {e}")
        return False

def test_awareness_amplification():
    """Test AwarenessAmplificationTransformer."""
    print("\n[TEST 4/7] AwarenessAmplificationTransformer")
    print("-" * 50)
    
    try:
        from transformers.awareness_amplification import AwarenessAmplificationTransformer
        
        transformer = AwarenessAmplificationTransformer()
        
        # Fit on test data
        test_texts = [
            "I know this is my moment. Everything is on the line.",
            "Just another day at work.",
            "The story of my life culminates here. For everyone who believed.",
            "Normal performance expected."
        ]
        
        transformer.fit(test_texts)
        
        # Transform
        features = transformer.transform(test_texts)
        
        assert features.shape == (4, 15)
        assert features[0, -1] > features[1, -1]  # First text more aware
        
        # Test amplification calculation
        amp = transformer.calculate_amplification_effect(
            amplification_score=0.8,
            potential_energy=0.7,
            consciousness=1.0
        )
        assert amp > 1.0  # Should amplify
        
        print(f"✓ AwarenessAmplification: PASS")
        print(f"  Features: {features.shape}")
        print(f"  Amplification range: {features[:, -1].min():.3f} - {features[:, -1].max():.3f}")
        return True
        
    except Exception as e:
        print(f"✗ AwarenessAmplification: FAIL - {e}")
        return False

def test_blind_narratio_calculator():
    """Test BlindNarratioCalculator."""
    print("\n[TEST 5/7] BlindNarratioCalculator")
    print("-" * 50)
    
    try:
        from core.story_instance import StoryInstance
        from analysis.blind_narratio_calculator import BlindNarratioCalculator
        
        calculator = BlindNarratioCalculator()
        
        # Create test instances
        instances = []
        for i in range(20):
            inst = StoryInstance(
                instance_id=f"test_{i}",
                domain="test_domain",
                narrative_text=f"Narrative {i}",
                outcome=float(i % 2)
            )
            inst.theta_resistance = 0.4 + i * 0.01
            instances.append(inst)
        
        # Calculate domain Β
        result = calculator.calculate_domain_blind_narratio(
            instances=instances,
            domain_name="test_domain"
        )
        
        assert 'Β' in result
        assert result['Β'] > 0
        assert result['n_instances'] == 20
        
        print(f"✓ BlindNarratioCalculator: PASS")
        print(f"  Β: {result['Β']:.3f}")
        print(f"  Stability: {result['stability']:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ BlindNarratioCalculator: FAIL - {e}")
        return False

def test_imperative_gravity():
    """Test ImperativeGravityCalculator."""
    print("\n[TEST 6/7] ImperativeGravityCalculator")
    print("-" * 50)
    
    try:
        from physics.imperative_gravity import ImperativeGravityCalculator
        from config.domain_config import DomainConfig
        from core.story_instance import StoryInstance
        
        # Load configs
        configs = {
            'golf': DomainConfig('golf'),
            'tennis': DomainConfig('tennis'),
            'chess': DomainConfig('chess')
        }
        
        calculator = ImperativeGravityCalculator(configs)
        
        # Create test instance
        instance = StoryInstance(
            instance_id="test_golf",
            domain="golf",
            narrative_text="Test narrative"
        )
        instance.mass = 1.5
        instance.pi_effective = 0.72
        
        # Calculate forces
        forces = calculator.calculate_cross_domain_forces(
            instance,
            ['tennis', 'chess'],
            exclude_same_domain=True
        )
        
        assert 'tennis' in forces
        assert 'chess' in forces
        
        # Find neighbors
        neighbors = calculator.find_gravitational_neighbors(
            instance,
            ['tennis', 'chess'],
            n_neighbors=2
        )
        
        assert len(neighbors) == 2
        
        # Calculate similarity matrix
        matrix = calculator.calculate_domain_similarity_matrix(['golf', 'tennis', 'chess'])
        assert matrix.shape == (3, 3)
        
        print(f"✓ ImperativeGravity: PASS")
        print(f"  Forces calculated: {len(forces)}")
        print(f"  Top neighbor: {neighbors[0][0]} (force={neighbors[0][1]:.2f})")
        return True
        
    except Exception as e:
        print(f"✗ ImperativeGravity: FAIL - {e}")
        return False

def test_dynamic_narrativity():
    """Test DynamicNarrativityAnalyzer and ComplexityScorer."""
    print("\n[TEST 7/7] Dynamic Narrativity System")
    print("-" * 50)
    
    try:
        from core.story_instance import StoryInstance
        from analysis.complexity_scorer import ComplexityScorer
        from analysis.dynamic_narrativity import DynamicNarrativityAnalyzer
        from config.domain_config import DomainConfig
        
        config = DomainConfig('golf')
        scorer = ComplexityScorer(domain='golf')
        analyzer = DynamicNarrativityAnalyzer(config)
        
        # Create test instances with varying complexity
        instances = []
        for i in range(15):
            text = "Simple clear narrative." if i < 5 else \
                   "Ambiguous contested uncertain disputed narrative." if i >= 10 else \
                   "Moderate narrative with some complexity."
            
            inst = StoryInstance(
                instance_id=f"test_{i}",
                domain="golf",
                narrative_text=text,
                outcome=float(i % 2)
            )
            
            # Calculate complexity
            complexity = scorer.calculate_complexity(inst, text)
            inst.pi_effective = config.calculate_effective_pi(complexity)
            inst.pi_domain_base = config.get_pi()
            
            instances.append(inst)
        
        # Analyze π variance
        result = analyzer.analyze_pi_variance(
            instances=instances,
            domain_name="golf"
        )
        
        assert 'pi_variance_significant' in result
        assert 'pi_range' in result
        assert result['instances_analyzed'] == 15
        
        # Check that we have π variance
        pi_range = result['pi_range']
        assert pi_range[1] > pi_range[0]
        
        print(f"✓ DynamicNarrativity: PASS")
        print(f"  π range: [{pi_range[0]:.3f}, {pi_range[1]:.3f}]")
        print(f"  Variance significant: {result['pi_variance_significant']}")
        return True
        
    except Exception as e:
        print(f"✗ DynamicNarrativity: FAIL - {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    print("=" * 50)
    print("FRAMEWORK 2.0 VALIDATION TESTS")
    print("=" * 50)
    
    tests = [
        ("StoryInstance", test_story_instance),
        ("InstanceRepository", test_instance_repository),
        ("DomainConfig", test_domain_config_extensions),
        ("AwarenessAmplification", test_awareness_amplification),
        ("BlindNarratioCalculator", test_blind_narratio_calculator),
        ("ImperativeGravity", test_imperative_gravity),
        ("DynamicNarrativity", test_dynamic_narrativity)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name}: EXCEPTION - {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:30s} {status}")
    
    print(f"\n{'=' * 50}")
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'=' * 50}\n")
    
    if passed == total:
        print("🎉 All tests passed! Framework 2.0 is operational.")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check errors above.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

