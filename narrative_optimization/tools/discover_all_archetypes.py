"""
Discover Archetypes for All Domains

Runs archetype discovery on all registered domains.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_domain_registry
from src.pipeline_config import get_config
from src.data import DataLoader
from src.config import ArchetypeDiscovery, ArchetypeValidator


def discover_for_domain(domain_name: str) -> dict:
    """Discover archetypes for a single domain."""
    try:
        # Load data
        config = get_config()
        loader = DataLoader()
        
        data_path = config.get_domain_data_path(domain_name)
        
        if not data_path or not data_path.exists():
            return {'status': 'no_data'}
        
        data = loader.load(data_path)
        
        if not loader.validate_data(data):
            return {'status': 'invalid_data'}
        
        # Identify winners
        outcomes = data['outcomes']
        if len(np.unique(outcomes)) > 2:
            threshold = np.percentile(outcomes, 75)
            winner_mask = outcomes >= threshold
        else:
            winner_mask = outcomes == 1
        
        winner_texts = [data['texts'][i] for i in range(len(data['texts'])) if winner_mask[i]]
        
        if len(winner_texts) < 5:
            return {'status': 'insufficient_winners', 'n_winners': len(winner_texts)}
        
        # Discover
        discovery = ArchetypeDiscovery(min_pattern_frequency=0.05)
        archetypes = discovery.discover_archetypes(winner_texts, n_archetypes=5)
        
        # Validate
        validator = ArchetypeValidator()
        validated = {}
        
        for arch_name, arch_data in archetypes.items():
            validation = validator.validate_archetype(
                arch_data['patterns'],
                data['texts'],
                outcomes
            )
            
            if validation['is_significant']:
                validated[arch_name] = {
                    'patterns': arch_data['patterns'],
                    'correlation': validation['correlation'],
                    'p_value': validation['p_value'],
                    'coherence': arch_data['coherence']
                }
        
        return {
            'status': 'success',
            'discovered': len(archetypes),
            'validated': len(validated),
            'archetypes': validated
        }
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def discover_all():
    """Discover archetypes for all domains."""
    print("="*80)
    print("DISCOVERING ARCHETYPES FOR ALL DOMAINS")
    print("="*80)
    
    registry = get_domain_registry()
    domains = [d.name for d in registry.get_all_domains()]
    
    print(f"\nProcessing {len(domains)} domains...\n")
    
    all_results = {}
    success = 0
    
    for domain in domains:
        print(f"  {domain:20s} ", end="", flush=True)
        
        result = discover_for_domain(domain)
        all_results[domain] = result
        
        if result['status'] == 'success':
            print(f"✓ Discovered: {result['discovered']}, Validated: {result['validated']}")
            success += 1
            
            # Save archetypes
            output_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / domain
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'discovered_archetypes.json', 'w') as f:
                json.dump(result['archetypes'], f, indent=2)
        else:
            print(f"⊙ {result['status']}")
    
    # Summary
    print(f"\n{'='*80}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"  Success: {success}/{len(domains)}")
    
    # Save all results
    results_path = Path(__file__).parent.parent / 'all_discovered_archetypes.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"  All results saved: {results_path}")


if __name__ == '__main__':
    discover_all()

