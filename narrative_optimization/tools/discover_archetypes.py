"""
Archetype Discovery Tool

Easy-to-use command-line tool for discovering and testing new archetypes.

Usage:
    python discover_archetypes.py --domain golf --winners golf_winners.txt
    python discover_archetypes.py --discover-sub mental_game --domain golf
    python discover_archetypes.py --test new_patterns.json --domain golf

Author: Narrative Integration System
Date: November 2025
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.archetype_discovery import ArchetypeDiscovery, ArchetypeRegistry, ArchetypeValidator


def discover_main_archetypes(winner_file: str, n_archetypes: int = 5):
    """
    Discover main archetypes from winner texts.
    
    Parameters
    ----------
    winner_file : str
        Path to file with winner narratives (one per line)
    n_archetypes : int
        Number of archetypes to discover
    """
    print(f"\n{'='*80}")
    print(f"DISCOVERING {n_archetypes} MAIN ARCHETYPES")
    print(f"{'='*80}\n")
    
    # Load texts
    with open(winner_file) as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Loaded {len(texts)} winner narratives")
    
    # Discover archetypes
    discovery = ArchetypeDiscovery(min_pattern_frequency=0.1)
    archetypes = discovery.discover_archetypes(texts, n_archetypes=n_archetypes)
    
    # Display results
    print(f"\n{'='*80}")
    print("DISCOVERED ARCHETYPES")
    print(f"{'='*80}\n")
    
    for arch_name, arch_data in archetypes.items():
        print(f"\n{arch_name.upper()}:")
        print(f"  Patterns ({len(arch_data['patterns'])}):")
        for i, pattern in enumerate(arch_data['patterns'][:10], 1):
            weight = arch_data['weights'].get(pattern, 0)
            print(f"    {i:2d}. '{pattern}' (weight: {weight:.3f})")
        print(f"  Sample count: {arch_data['sample_count']}")
        print(f"  Coherence: {arch_data['coherence']:.3f}")
    
    # Save results
    output_file = winner_file.replace('.txt', '_archetypes_discovered.json')
    with open(output_file, 'w') as f:
        json.dump(archetypes, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    
    return archetypes


def discover_sub_archetypes(
    winner_file: str,
    parent_archetype: str,
    parent_patterns: list,
    n_sub: int = 3
):
    """
    Discover sub-archetypes within a parent archetype.
    
    Parameters
    ----------
    winner_file : str
        Path to file with narratives
    parent_archetype : str
        Name of parent archetype
    parent_patterns : list
        Patterns defining parent
    n_sub : int
        Number of sub-archetypes
    """
    print(f"\n{'='*80}")
    print(f"DISCOVERING SUB-ARCHETYPES FOR: {parent_archetype.upper()}")
    print(f"{'='*80}\n")
    
    # Load texts
    with open(winner_file) as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Loaded {len(texts)} narratives")
    print(f"  Parent patterns: {', '.join(parent_patterns)}")
    
    # Discover sub-archetypes
    discovery = ArchetypeDiscovery()
    sub_archetypes = discovery.discover_sub_archetypes(
        texts, parent_patterns, n_sub_archetypes=n_sub
    )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"DISCOVERED SUB-ARCHETYPES")
    print(f"{'='*80}\n")
    
    for sub_name, sub_data in sub_archetypes.items():
        print(f"\n{sub_name.upper()}:")
        print(f"  Patterns ({len(sub_data['patterns'])}):")
        for i, pattern in enumerate(sub_data['patterns'][:10], 1):
            weight = sub_data['weights'].get(pattern, 0)
            print(f"    {i:2d}. '{pattern}' (weight: {weight:.3f})")
    
    # Save results
    output_file = winner_file.replace('.txt', f'_sub_{parent_archetype}_discovered.json')
    with open(output_file, 'w') as f:
        json.dump(sub_archetypes, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    
    return sub_archetypes


def discover_contextual_boosters(
    data_file: str,
    archetype_patterns: list,
    archetype_name: str
):
    """
    Discover contextual features that boost archetype effectiveness.
    
    Parameters
    ----------
    data_file : str
        Path to file with texts and outcomes (JSON format)
    archetype_patterns : list
        Patterns defining archetype
    archetype_name : str
        Name of archetype
    """
    print(f"\n{'='*80}")
    print(f"DISCOVERING CONTEXTUAL BOOSTERS FOR: {archetype_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load data
    with open(data_file) as f:
        data = json.load(f)
    
    texts = data['texts']
    outcomes = np.array(data['outcomes'])
    
    print(f"✓ Loaded {len(texts)} narratives with outcomes")
    print(f"  Archetype patterns: {', '.join(archetype_patterns)}")
    
    # Discover boosters
    discovery = ArchetypeDiscovery()
    boosters = discovery.discover_contextual_boosters(
        texts, outcomes, archetype_patterns
    )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"DISCOVERED CONTEXTUAL BOOSTERS")
    print(f"{'='*80}\n")
    
    for context, multiplier in boosters.items():
        effect_pct = (multiplier - 1) * 100
        print(f"  '{context}': {multiplier}x boost (+{effect_pct:.1f}%)")
    
    # Save results
    output_file = data_file.replace('.json', f'_boosters_{archetype_name}.json')
    with open(output_file, 'w') as f:
        json.dump(boosters, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    
    return boosters


def test_archetype(
    patterns: list,
    data_file: str,
    archetype_name: str
):
    """
    Test if a proposed archetype is valid and predictive.
    
    Parameters
    ----------
    patterns : list
        Patterns defining proposed archetype
    data_file : str
        Path to data file with texts and outcomes
    archetype_name : str
        Name of proposed archetype
    """
    print(f"\n{'='*80}")
    print(f"TESTING ARCHETYPE: {archetype_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load data
    with open(data_file) as f:
        data = json.load(f)
    
    texts = data['texts']
    outcomes = np.array(data['outcomes'])
    
    print(f"✓ Loaded {len(texts)} narratives")
    print(f"  Patterns to test: {', '.join(patterns)}")
    
    # Validate
    validator = ArchetypeValidator(significance_threshold=0.05)
    results = validator.validate_archetype(patterns, texts, outcomes)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*80}\n")
    
    print(f"  Correlation: {results['correlation']:.3f}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Significant: {'YES ✓' if results['is_significant'] else 'NO ✗'}")
    print(f"  Sample size: {results['sample_size']}")
    
    if results['is_significant']:
        print(f"\n✓ ARCHETYPE IS VALID - Can be registered!")
    else:
        print(f"\n✗ ARCHETYPE IS NOT SIGNIFICANT - Consider refining patterns")
    
    # Save results
    output_file = data_file.replace('.json', f'_validation_{archetype_name}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    
    return results


def register_archetype_interactive():
    """Interactive archetype registration."""
    print(f"\n{'='*80}")
    print("ARCHETYPE REGISTRATION")
    print(f"{'='*80}\n")
    
    registry = ArchetypeRegistry()
    
    # Get archetype details
    name = input("Archetype name: ").strip()
    domain = input("Domain (or 'agnostic'): ").strip()
    domain = None if domain.lower() == 'agnostic' else domain
    
    patterns_str = input("Patterns (comma-separated): ").strip()
    patterns = [p.strip() for p in patterns_str.split(',')]
    
    weight = float(input("Weight (0-1): ").strip() or "1.0")
    description = input("Description: ").strip()
    
    # Register
    registry.register_archetype(
        name=name,
        patterns=patterns,
        domain=domain,
        weight=weight,
        description=description
    )
    
    print(f"\n✓ Registered archetype: {name}")
    
    # Save registry
    registry_file = 'archetype_registry.json'
    with open(registry_file, 'w') as f:
        json.dump(registry.export_to_dict(), f, indent=2)
    
    print(f"✓ Saved to: {registry_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Discover and test narrative archetypes')
    
    parser.add_argument('--discover', action='store_true',
                       help='Discover main archetypes from winner texts')
    parser.add_argument('--discover-sub', type=str,
                       help='Discover sub-archetypes for parent archetype')
    parser.add_argument('--discover-boosters', type=str,
                       help='Discover contextual boosters for archetype')
    parser.add_argument('--test', type=str, nargs='+',
                       help='Test archetype patterns')
    parser.add_argument('--register', action='store_true',
                       help='Register a new archetype interactively')
    
    parser.add_argument('--file', type=str, help='Data file path')
    parser.add_argument('--domain', type=str, help='Domain name')
    parser.add_argument('--n-archetypes', type=int, default=5,
                       help='Number of archetypes to discover')
    parser.add_argument('--name', type=str, help='Archetype name')
    
    args = parser.parse_args()
    
    if args.discover:
        if not args.file:
            print("Error: --file required for discovery")
            sys.exit(1)
        discover_main_archetypes(args.file, args.n_archetypes)
    
    elif args.discover_sub:
        if not args.file:
            print("Error: --file required")
            sys.exit(1)
        # Load parent patterns from previous discovery
        parent_file = args.file.replace('.txt', '_archetypes_discovered.json')
        with open(parent_file) as f:
            parent_data = json.load(f)
        parent_patterns = parent_data.get(args.discover_sub, {}).get('patterns', [])
        
        discover_sub_archetypes(args.file, args.discover_sub, parent_patterns)
    
    elif args.discover_boosters:
        if not args.file or not args.test:
            print("Error: --file and --test patterns required")
            sys.exit(1)
        discover_contextual_boosters(args.file, args.test, args.discover_boosters)
    
    elif args.test:
        if not args.file or not args.name:
            print("Error: --file and --name required")
            sys.exit(1)
        test_archetype(args.test, args.file, args.name)
    
    elif args.register:
        register_archetype_interactive()
    
    else:
        parser.print_help()

