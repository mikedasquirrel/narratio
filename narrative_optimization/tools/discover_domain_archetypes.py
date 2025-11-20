"""
Domain Archetype Discovery Tool

Automatically discovers archetypes from domain winner data and validates them.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ArchetypeDiscovery, ArchetypeRegistry, ArchetypeValidator, DomainConfig
from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer


def discover_and_validate_domain_archetypes(domain_name: str, data_path: Optional[Path] = None):
    """
    Discover archetypes from domain data and validate them.
    
    Parameters
    ----------
    domain_name : str
        Domain name
    data_path : Path, optional
        Path to domain data (if None, tries to find automatically)
    """
    print(f"\n{'='*80}")
    print(f"DISCOVERING ARCHETYPES FOR: {domain_name.upper()}")
    print(f"{'='*80}")
    
    # Load data
    if data_path is None:
        project_root = Path(__file__).parent.parent.parent
        possible_paths = [
            project_root / 'data' / 'domains' / f'{domain_name}*.json',
            project_root / 'narrative_optimization' / 'domains' / domain_name / '*.json'
        ]
        # Find first existing file
        for pattern in possible_paths:
            matches = list(pattern.parent.glob(pattern.name))
            if matches:
                data_path = matches[0]
                break
    
    if data_path is None or not data_path.exists():
        print(f"  ✗ No data file found for {domain_name}")
        return None
    
    print(f"\n[1/5] Loading data from {data_path.name}...")
    
    # Load data (simplified - adjust to your format)
    with open(data_path) as f:
        data = json.load(f)
    
    # Extract texts and outcomes
    if isinstance(data, dict):
        texts = data.get('texts', data.get('narratives', []))
        outcomes = np.array(data.get('outcomes', data.get('results', [])))
    elif isinstance(data, list):
        texts = [item.get('narrative', item.get('text', str(item))) for item in data]
        outcomes = np.array([item.get('outcome', item.get('result', 0)) for item in data])
    else:
        print(f"  ✗ Unknown data format")
        return None
    
    if len(texts) < 20:
        print(f"  ✗ Insufficient data ({len(texts)} samples, need 20+)")
        return None
    
    print(f"  ✓ Loaded {len(texts)} samples")
    
    # Identify winners
    print(f"\n[2/5] Identifying winners...")
    if len(np.unique(outcomes)) > 2:
        # Continuous - top 25%
        threshold = np.percentile(outcomes, 75)
        winner_mask = outcomes >= threshold
    else:
        # Binary
        winner_mask = outcomes == 1
    
    winner_texts = [texts[i] for i in range(len(texts)) if winner_mask[i]]
    
    print(f"  ✓ Identified {len(winner_texts)} winners ({len(winner_texts)/len(texts):.1%})")
    
    if len(winner_texts) < 5:
        print(f"  ✗ Too few winners ({len(winner_texts)}, need 5+)")
        return None
    
    # Discover archetypes
    print(f"\n[3/5] Discovering archetypes...")
    discovery = ArchetypeDiscovery(min_pattern_frequency=0.05)
    archetypes = discovery.discover_archetypes(winner_texts, n_archetypes=5)
    
    print(f"  ✓ Discovered {len(archetypes)} archetypes")
    
    # Validate each archetype
    print(f"\n[4/5] Validating discovered archetypes...")
    validator = ArchetypeValidator()
    validated_archetypes = {}
    
    for arch_name, arch_data in archetypes.items():
        patterns = arch_data['patterns']
        
        validation = validator.validate_archetype(
            patterns, texts, outcomes
        )
        
        if validation['is_significant']:
            validated_archetypes[arch_name] = {
                'patterns': patterns,
                'correlation': validation['correlation'],
                'p_value': validation['p_value'],
                'coherence': arch_data['coherence'],
                'sample_count': arch_data['sample_count']
            }
            print(f"  ✓ {arch_name}: r={validation['correlation']:.3f}, p={validation['p_value']:.4f}")
        else:
            print(f"  ✗ {arch_name}: Not significant (p={validation['p_value']:.4f})")
    
    # Discover contextual boosters for best archetype
    print(f"\n[5/5] Discovering contextual boosters...")
    if validated_archetypes:
        best_archetype = max(validated_archetypes.items(), key=lambda x: x[1]['correlation'])
        best_patterns = best_archetype[1]['patterns']
        
        boosters = discovery.discover_contextual_boosters(
            texts, outcomes, best_patterns
        )
        
        print(f"  ✓ Found {len(boosters)} contextual boosters")
        for context, multiplier in list(boosters.items())[:5]:
            boost_pct = (multiplier - 1) * 100
            print(f"    '{context}': {multiplier}x (+{boost_pct:.1f}%)")
        
        validated_archetypes[best_archetype[0]]['contextual_boosters'] = boosters
    
    # Save results
    output_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / domain_name / 'discovered_archetypes.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'domain': domain_name,
            'discovered_archetypes': validated_archetypes,
            'total_discovered': len(archetypes),
            'validated': len(validated_archetypes)
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    return validated_archetypes


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover archetypes from domain data')
    parser.add_argument('--domain', type=str, required=True, help='Domain name')
    parser.add_argument('--file', type=str, help='Path to data file')
    
    args = parser.parse_args()
    
    data_path = Path(args.file) if args.file else None
    discover_and_validate_domain_archetypes(args.domain, data_path)

