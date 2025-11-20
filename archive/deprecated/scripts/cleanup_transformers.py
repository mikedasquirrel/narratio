"""
Transformer Cleanup Script
==========================

Cleans up transformer directory to keep only the canonical 33 production-ready transformers.
Removes duplicates, experimental versions, and outdated files.

Canonical list from: docs/technical/TRANSFORMER_CATALOG.md

Author: AI Coding Assistant
Date: November 16, 2025
"""

import shutil
from pathlib import Path
from datetime import datetime

# CANONICAL 33 PRODUCTION-READY TRANSFORMERS
# Based on docs/technical/TRANSFORMER_CATALOG.md

CANONICAL_TRANSFORMERS = {
    # ==== CORE (6) - Always recommended ====
    'nominative.py': 'NominativeAnalysisTransformer - 51 features',
    'self_perception.py': 'SelfPerceptionTransformer - 21 features',
    'narrative_potential.py': 'NarrativePotentialTransformer - 35 features',
    'linguistic_advanced.py': 'LinguisticPatternsTransformer - 36 features',
    'ensemble.py': 'EnsembleNarrativeTransformer - 25 features',
    'relational.py': 'RelationalValueTransformer - 17 features',
    
    # ==== STRUCTURAL (3) ====
    'conflict_tension.py': 'ConflictTensionTransformer - 28 features',
    'suspense_mystery.py': 'SuspenseMysteryTransformer - 25 features',
    'framing.py': 'FramingTransformer - 24 features',
    
    # ==== CREDIBILITY (2) ====
    'authenticity.py': 'AuthenticityTransformer - 30 features',
    'expertise_authority.py': 'ExpertiseAuthorityTransformer - 32 features',
    
    # ==== CONTEXTUAL (2) ====
    'temporal_evolution.py': 'TemporalEvolutionTransformer - 18 features',
    'cultural_context.py': 'CulturalContextTransformer - 22 features',
    
    # ==== NOMINATIVE (5) ====
    'phonetic.py': 'PhoneticTransformer - 15 features',
    'social_status.py': 'SocialStatusTransformer - 12 features',
    'universal_nominative.py': 'UniversalNominativeTransformer - 40 features',
    'hierarchical_nominative.py': 'HierarchicalNominativeTransformer + related - 35 features',
    'nominative_richness.py': 'NominativeRichnessTransformer - 15 features (BREAKTHROUGH)',
    
    # ==== ADVANCED (6) ====
    'information_theory.py': 'InformationTheoryTransformer - 25 features',
    'namespace_ecology.py': 'NamespaceEcologyTransformer - 35 features',
    'anticipatory_commitment.py': 'AnticipatoryCommunicationTransformer - 25 features',
    'cognitive_fluency.py': 'CognitiveFluencyTransformer - 16 features',
    'discoverability.py': 'DiscoverabilityTransformer - 12 features',
    'multi_scale.py': 'MultiScaleTransformer + related - 70 features',
    
    # ==== THEORY-ALIGNED / PHASE 7 (8) ====
    'coupling_strength.py': 'CouplingStrengthTransformer (κ) - 12 features',
    'narrative_mass.py': 'NarrativeMassTransformer (μ) - 10 features',
    'gravitational_features.py': 'GravitationalFeaturesTransformer (φ & ة) - 20 features',
    'awareness_resistance.py': 'AwarenessResistanceTransformer (θ) - 15 features',
    'fundamental_constraints.py': 'FundamentalConstraintsTransformer (λ) - 28 features',
    'alpha.py': 'AlphaTransformer (α) - 8 features',
    'quantitative.py': 'QuantitativeTransformer - 10 features',
    'optics.py': 'OpticsTransformer - 15 features',
    
    # ==== PATTERN / MISC (1) ====
    'context_pattern.py': 'ContextPatternTransformer - pattern discovery',
    
    # ==== STATISTICAL BASELINE (1) ====
    'statistical.py': 'StatisticalTransformer - TF-IDF baseline',
    
    # ==== EMOTIONAL (1) - Should be here ====
    'emotional_resonance.py': 'EmotionalResonanceTransformer - 34 features',
}

# KEEP THESE UTILITY/INFRASTRUCTURE FILES
KEEP_UTILITIES = {
    'base.py',
    'base_transformer.py',
    'domain_adaptive_base.py',
    'transformer_factory.py',
    'transformer_library.py',
    'transformer_selector.py',
    '__init__.py',
}

# KEEP THESE DIRECTORIES (infrastructure)
KEEP_DIRECTORIES = {
    'utils',
    'caching',
}

# VERSION 2 HANDLING: Keep V2 if it exists, otherwise keep V1
VERSION_PREFERENCE = {
    'emotional_resonance.py': 'emotional_resonance_v2.py',  # Prefer V2
    'linguistic_advanced.py': 'linguistic_v2.py',  # Prefer V2
    'narrative_potential.py': 'narrative_potential_v2.py',  # Prefer V2
    'nominative.py': 'nominative_v2.py',  # Prefer V2
    'self_perception.py': 'self_perception_v2.py',  # Prefer V2
}


def print_header(text, char='='):
    print()
    print(char * 80)
    print(text)
    print(char * 80)


def main():
    """Clean up transformers directory"""
    
    print_header("TRANSFORMER CLEANUP - KEEPING ONLY CANONICAL 33", "█")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Based on: docs/technical/TRANSFORMER_CATALOG.md")
    print(f"Canonical transformers: {len(CANONICAL_TRANSFORMERS)}")
    print()
    
    transformers_dir = Path('narrative_optimization/src/transformers')
    
    if not transformers_dir.exists():
        print(f"✗ Transformers directory not found: {transformers_dir}")
        return
    
    # Create backup directory
    backup_dir = Path('transformer_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup_dir.mkdir(exist_ok=True)
    
    print(f"✓ Created backup directory: {backup_dir}")
    print()
    
    # Scan all transformer files
    print_header("SCANNING TRANSFORMER DIRECTORY", "-")
    
    all_files = list(transformers_dir.rglob('*.py'))
    all_files = [f for f in all_files if '__pycache__' not in str(f)]
    
    print(f"Total Python files found: {len(all_files)}")
    print()
    
    # Categorize files
    to_keep = []
    to_remove = []
    to_check = []
    
    for file_path in all_files:
        rel_path = file_path.relative_to(transformers_dir)
        filename = file_path.name
        parent_dir = file_path.parent.name
        
        # Keep utilities
        if filename in KEEP_UTILITIES:
            to_keep.append((file_path, 'utility/infrastructure'))
            continue
        
        # Keep utility directories
        if any(keep_dir in str(rel_path) for keep_dir in KEEP_DIRECTORIES):
            to_keep.append((file_path, 'utility directory'))
            continue
        
        # Check if it's a canonical transformer (in root transformers dir)
        if file_path.parent == transformers_dir:
            if filename in CANONICAL_TRANSFORMERS:
                # Check if V2 version exists and should be preferred
                if filename in VERSION_PREFERENCE:
                    v2_file = VERSION_PREFERENCE[filename]
                    v2_path = transformers_dir / v2_file
                    if v2_path.exists():
                        to_remove.append((file_path, f'replaced by {v2_file}'))
                    else:
                        to_keep.append((file_path, 'canonical'))
                else:
                    to_keep.append((file_path, 'canonical'))
            elif filename in VERSION_PREFERENCE.values():
                # This IS a V2 file - keep it
                to_keep.append((file_path, 'canonical V2'))
            else:
                # Not in canonical list
                to_remove.append((file_path, 'not in canonical list'))
        else:
            # Subdirectory - check if it's a special domain
            if parent_dir in ['archetypes', 'mental_health', 'sports', 'hurricanes', 'ships',
                            'linguistic', 'temporal', 'cognitive', 'anthropological', 
                            'cross_cultural', 'semantic', 'credibility', 'structural',
                            'contextual', 'core', 'nominative', 'specialized']:
                # These are organized special transformers - review case by case
                to_check.append((file_path, f'special domain: {parent_dir}'))
            else:
                to_remove.append((file_path, 'experimental/old subdirectory'))
    
    # Print summary
    print_header("CLASSIFICATION RESULTS", "-")
    print(f"✓ Keep: {len(to_keep)} files")
    print(f"✗ Remove: {len(to_remove)} files")
    print(f"⚠ Review: {len(to_check)} files")
    print()
    
    # Show what will be removed
    print_header("FILES TO REMOVE", "-")
    print(f"Total: {len(to_remove)} files\n")
    
    by_reason = {}
    for filepath, reason in to_remove:
        if reason not in by_reason:
            by_reason[reason] = []
        by_reason[reason].append(filepath.name)
    
    for reason, files in sorted(by_reason.items()):
        print(f"{reason}: {len(files)} files")
        if len(files) <= 10:
            for f in sorted(files):
                print(f"  - {f}")
        else:
            for f in sorted(files)[:5]:
                print(f"  - {f}")
            print(f"  ... and {len(files)-5} more")
        print()
    
    # Show what will be kept
    print_header("FILES TO KEEP (CANONICAL 33 + UTILS)", "-")
    print(f"Total: {len(to_keep)} files\n")
    
    canonical_count = sum(1 for _, r in to_keep if 'canonical' in r)
    utility_count = sum(1 for _, r in to_keep if 'utility' in r)
    
    print(f"Canonical transformers: {canonical_count}")
    print(f"Utility/infrastructure: {utility_count}")
    print()
    
    canonical_files = [(f.name, r) for f, r in to_keep if 'canonical' in r]
    for filename, reason in sorted(canonical_files):
        desc = CANONICAL_TRANSFORMERS.get(filename, 'V2 enhanced')
        print(f"  ✓ {filename:<40} {desc}")
    
    # Show special domains to review
    if to_check:
        print_header("SPECIAL DOMAIN TRANSFORMERS (REVIEW NEEDED)", "-")
        print(f"Total: {len(to_check)} files\n")
        
        by_domain = {}
        for filepath, reason in to_check:
            domain = filepath.parent.name
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(filepath.name)
        
        for domain, files in sorted(by_domain.items()):
            print(f"{domain}/: {len(files)} files")
            for f in sorted(files):
                print(f"  - {f}")
            print()
        
        print("These are domain-specific transformers (archetypes, sports, etc.)")
        print("Recommendation: KEEP if actively used, REMOVE if experimental")
    
    # Confirm before proceeding
    print_header("READY TO CLEAN", "=")
    print(f"Will remove {len(to_remove)} files")
    print(f"Will backup to: {backup_dir}")
    print()
    
    response = input("Proceed with cleanup? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\n✗ Cleanup cancelled")
        return
    
    # Perform cleanup
    print()
    print_header("EXECUTING CLEANUP", "=")
    
    removed_count = 0
    for filepath, reason in to_remove:
        try:
            # Create backup
            rel_path = filepath.relative_to(transformers_dir)
            backup_path = backup_dir / 'transformers' / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(filepath, backup_path)
            
            # Remove original
            filepath.unlink()
            removed_count += 1
            
            if removed_count % 10 == 0:
                print(f"  Processed {removed_count}/{len(to_remove)} files...")
        
        except Exception as e:
            print(f"  ✗ Error removing {filepath.name}: {e}")
    
    print(f"\n✓ Removed {removed_count} files")
    print(f"✓ Backed up to: {backup_dir}/transformers/")
    
    # Clean up empty directories
    print("\nCleaning up empty directories...")
    for dirpath in sorted(transformers_dir.rglob('*'), reverse=True):
        if dirpath.is_dir() and dirpath.name != '__pycache__':
            try:
                if not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    print(f"  Removed empty: {dirpath.name}/")
            except:
                pass
    
    # Final summary
    print()
    print_header("CLEANUP COMPLETE", "█")
    
    remaining_files = list(transformers_dir.rglob('*.py'))
    remaining_files = [f for f in remaining_files if '__pycache__' not in str(f)]
    
    print(f"✓ Removed: {removed_count} files")
    print(f"✓ Remaining: {len(remaining_files)} files")
    print(f"✓ Canonical transformers: {canonical_count}")
    print(f"✓ Infrastructure files: {utility_count}")
    print()
    print(f"📁 Backup location: {backup_dir}/")
    print()
    print("Your transformer library is now clean and production-ready! 🎉")
    print()


if __name__ == "__main__":
    main()

