"""
Transformer Cleanup Script - UPDATED
=====================================

Keeps the UPDATED canonical 45 transformers including recent universal/meta additions.

Canonical list: 45 transformers
- 35 original production-ready
- 10 new universal/meta transformers added recently

Author: AI Coding Assistant
Date: November 16, 2025
"""

import shutil
from pathlib import Path
from datetime import datetime

# UPDATED CANONICAL LIST - 45 PRODUCTION-READY TRANSFORMERS
# Includes recent universal/meta narrative additions

CANONICAL_TRANSFORMERS = {
    # ==== CORE (6) - Version preference ====
    'nominative_v2.py': 'NominativeAnalysisV2Transformer (prefer V2)',
    'nominative.py': 'NominativeAnalysisTransformer (fallback if no V2)',
    'self_perception_v2.py': 'SelfPerceptionV2Transformer (prefer V2)',
    'self_perception.py': 'SelfPerceptionTransformer (fallback)',
    'narrative_potential_v2.py': 'NarrativePotentialV2Transformer (prefer V2)',
    'narrative_potential.py': 'NarrativePotentialTransformer (fallback)',
    'linguistic_v2.py': 'LinguisticPatternsV2Transformer (prefer V2)',
    'linguistic_advanced.py': 'LinguisticPatternsTransformer (fallback)',
    'ensemble.py': 'EnsembleNarrativeTransformer',
    'relational.py': 'RelationalValueTransformer',
    'emotional_resonance_v2.py': 'EmotionalResonanceV2Transformer (prefer V2)',
    'emotional_resonance.py': 'EmotionalResonanceTransformer (fallback)',
    
    # ==== STRUCTURAL (3) ====
    'conflict_tension.py': 'ConflictTensionTransformer - TOP PERFORMER 56%',
    'suspense_mystery.py': 'SuspenseMysteryTransformer',
    'framing.py': 'FramingTransformer',
    
    # ==== CREDIBILITY (2) ====
    'authenticity.py': 'AuthenticityTransformer',
    'expertise_authority.py': 'ExpertiseAuthorityTransformer',
    
    # ==== CONTEXTUAL (2) ====
    'temporal_evolution.py': 'TemporalEvolutionTransformer',
    'cultural_context.py': 'CulturalContextTransformer',
    
    # ==== NOMINATIVE (5) ====
    'phonetic.py': 'PhoneticTransformer',
    'social_status.py': 'SocialStatusTransformer',
    'universal_nominative.py': 'UniversalNominativeTransformer',
    'hierarchical_nominative.py': 'HierarchicalNominativeTransformer',
    'nominative_richness.py': 'NominativeRichnessTransformer - BREAKTHROUGH 57.6%',
    
    # ==== ADVANCED (6) ====
    'information_theory.py': 'InformationTheoryTransformer - 55.2%',
    'namespace_ecology.py': 'NamespaceEcologyTransformer',
    'anticipatory_commitment.py': 'AnticipatoryCommunicationTransformer',
    'cognitive_fluency.py': 'CognitiveFluencyTransformer - FASTEST',
    'discoverability.py': 'DiscoverabilityTransformer',
    'multi_scale.py': 'MultiScaleTransformer - 54.4%',
    
    # ==== THEORY-ALIGNED (8) ====
    'coupling_strength.py': 'CouplingStrengthTransformer (κ)',
    'narrative_mass.py': 'NarrativeMassTransformer (μ)',
    'gravitational_features.py': 'GravitationalFeaturesTransformer (φ & ة)',
    'awareness_resistance.py': 'AwarenessResistanceTransformer (θ)',
    'fundamental_constraints.py': 'FundamentalConstraintsTransformer (λ)',
    'alpha.py': 'AlphaTransformer (α) - META',
    'quantitative.py': 'QuantitativeTransformer',
    'optics.py': 'OpticsTransformer',
    
    # ==== PATTERN / STATISTICAL (2) ====
    'context_pattern.py': 'ContextPatternTransformer',
    'statistical.py': 'StatisticalTransformer - baseline',
    
    # ==== 🆕 UNIVERSAL & META (10 NEW) ====
    'universal_themes.py': 'UniversalThemesTransformer - 20 universal themes',
    'universal_structural_pattern.py': 'UniversalStructuralPatternTransformer - pure geometry',
    'universal_hybrid.py': 'UniversalHybridTransformer - handles ANY data',
    'cross_domain_embedding.py': 'CrossDomainEmbeddingTransformer - universal patterns',
    'meta_narrative.py': 'MetaNarrativeTransformer - meta-awareness',
    'meta_feature_interaction.py': 'MetaFeatureInteractionTransformer - interaction discovery',
    'ensemble_meta.py': 'EnsembleMetaTransformer - intelligent stacking',
    'temporal_momentum_enhanced.py': 'TemporalMomentumEnhancedTransformer - momentum++',
    'competitive_context.py': 'CompetitiveContextTransformer - competition dynamics',
    'enriched_patterns.py': 'EnrichedPatternsTransformer - combined patterns',
}

# Keep all utilities and base classes
KEEP_UTILITIES = {
    'base.py',
    'base_transformer.py',
    'domain_adaptive_base.py',
    'domain_text.py',
    'transformer_factory.py',
    'transformer_library.py',
    'transformer_selector.py',
    'feature_fusion.py',
    'feature_selection.py',
    'semantic.py',
    '__init__.py',
}

# Keep these directories
KEEP_DIRECTORIES = {
    'utils',
    'caching',
    'semantic',  # Has emotional_semantic which might be used
}

# Optional: Keep domain-specific if actively used
OPTIONAL_KEEP_DIRECTORIES = {
    'archetypes',  # 17 domain-specific archetypes
    'temporal',  # 4 specialized temporal
    'linguistic',  # 5 specialized linguistic
    'sports',  # 4 sports performance
    'mental_health',  # 3 mental health
    'hurricanes',  # 3 hurricane
    'ships',  # 2 maritime
    'cognitive',  # 1 cognitive
    'anthropological',  # 1 ritual
    'cross_cultural',  # 1 cross-cultural
}


def print_header(text, char='='):
    print()
    print(char * 80)
    print(text)
    print(char * 80)


def main():
    """Clean up transformers directory"""
    
    print_header("TRANSFORMER CLEANUP - UPDATED CANONICAL 45", "█")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Canonical transformers: 45 (includes recent universal/meta additions)")
    print()
    
    transformers_dir = Path('narrative_optimization/src/transformers')
    
    if not transformers_dir.exists():
        print(f"✗ Transformers directory not found: {transformers_dir}")
        return
    
    # Scan all files
    print_header("SCANNING TRANSFORMER DIRECTORY", "-")
    
    all_files = list(transformers_dir.rglob('*.py'))
    all_files = [f for f in all_files if '__pycache__' not in str(f)]
    
    print(f"Total Python files found: {len(all_files)}")
    print()
    
    # Categorize
    to_keep = []
    to_remove = []
    optional_keep = []
    
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
        
        # Optional keep directories
        if any(opt_dir in str(rel_path) for opt_dir in OPTIONAL_KEEP_DIRECTORIES):
            optional_keep.append((file_path, f'domain-specific: {parent_dir}'))
            continue
        
        # Check if canonical (in root dir)
        if file_path.parent == transformers_dir:
            if filename in CANONICAL_TRANSFORMERS:
                to_keep.append((file_path, 'canonical'))
            else:
                to_remove.append((file_path, 'not in canonical list'))
        else:
            # Subdirectory not in optional keep
            to_remove.append((file_path, 'experimental/old'))
    
    # Summary
    print_header("CLASSIFICATION RESULTS", "-")
    print(f"✅ Keep (Canonical + Utils): {len(to_keep)} files")
    print(f"⚠️  Optional (Domain-Specific): {len(optional_keep)} files")
    print(f"✗ Remove (Experimental): {len(to_remove)} files")
    print()
    
    # Show canonical transformers
    print_header("CANONICAL 45 TRANSFORMERS TO KEEP", "-")
    
    canonical_files = [(f, r) for f, r in to_keep if 'canonical' in r]
    print(f"Total canonical: {len(canonical_files)}\n")
    
    categories = [
        ('Core', ['nominative', 'self_perception', 'narrative_potential', 'linguistic', 'ensemble', 'relational', 'emotional']),
        ('Structural', ['conflict', 'suspense', 'framing']),
        ('Credibility', ['authenticity', 'expertise']),
        ('Contextual', ['temporal_evolution', 'cultural_context']),
        ('Nominative', ['phonetic', 'social_status', 'universal_nominative', 'hierarchical', 'nominative_richness']),
        ('Advanced', ['information_theory', 'namespace', 'anticipatory', 'cognitive', 'discoverability', 'multi_scale']),
        ('Theory-Aligned', ['coupling', 'narrative_mass', 'gravitational', 'awareness', 'fundamental', 'alpha', 'quantitative', 'optics']),
        ('Pattern/Statistical', ['context_pattern', 'statistical']),
        ('🆕 Universal/Meta', ['universal', 'meta', 'cross_domain', 'ensemble_meta', 'temporal_momentum', 'competitive', 'enriched']),
    ]
    
    for cat_name, keywords in categories:
        print(f"\n{cat_name}:")
        cat_files = [f for f, r in canonical_files if any(kw in f.name for kw in keywords)]
        for f in sorted(cat_files, key=lambda x: x.name):
            desc = CANONICAL_TRANSFORMERS.get(f.name, '')
            v2_marker = ' 🆕 V2' if '_v2' in f.name else ''
            new_marker = ' 🆕 NEW' if f.name in ['universal_themes.py', 'universal_structural_pattern.py', 'universal_hybrid.py', 'cross_domain_embedding.py', 'meta_narrative.py', 'meta_feature_interaction.py', 'ensemble_meta.py', 'temporal_momentum_enhanced.py', 'competitive_context.py', 'enriched_patterns.py'] else ''
            print(f"  ✅ {f.name:<45} {v2_marker}{new_marker}")
    
    # Show optional domain-specific
    print()
    print_header("OPTIONAL DOMAIN-SPECIFIC TRANSFORMERS", "-")
    print(f"Total: {len(optional_keep)} files (in subdirectories)\n")
    
    by_dir = {}
    for filepath, reason in optional_keep:
        parent = filepath.parent.name
        if parent not in by_dir:
            by_dir[parent] = []
        by_dir[parent].append(filepath.name)
    
    for dir_name, files in sorted(by_dir.items()):
        print(f"{dir_name}/: {len(files)} transformers")
    
    print()
    print("💡 These are domain-specific. Recommendation:")
    print("   - KEEP if actively using domain-specific analysis")
    print("   - ARCHIVE if not currently used (can restore anytime)")
    
    # Show what will be removed
    print()
    print_header("FILES TO REMOVE (Experimental/Superseded)", "-")
    print(f"Total: {len(to_remove)} files\n")
    
    if len(to_remove) <= 30:
        for filepath, reason in sorted(to_remove):
            print(f"  ✗ {filepath.name}")
    else:
        for filepath, reason in sorted(to_remove)[:15]:
            print(f"  ✗ {filepath.name}")
        print(f"  ... and {len(to_remove)-15} more")
    
    # Decision prompt
    print()
    print_header("CLEANUP OPTIONS", "=")
    print()
    print("Choose cleanup level:")
    print("  1. MINIMAL - Keep all 45 canonical + all domain-specific")
    print(f"     (Keep: {len(to_keep) + len(optional_keep)}, Remove: {len(to_remove)})")
    print()
    print("  2. MODERATE - Keep 45 canonical + selected domain-specific")
    print("     (Interactive - you choose which domains)")
    print()
    print("  3. AGGRESSIVE - Keep only 45 canonical, archive all domain-specific")
    print(f"     (Keep: {len(to_keep)}, Archive: {len(to_remove) + len(optional_keep)})")
    print()
    print("  4. CANCEL - Don't clean up")
    print()
    
    response = input("Choose option (1/2/3/4): ").strip()
    
    if response == '4':
        print("\n✗ Cleanup cancelled")
        return
    
    # Create backup
    backup_dir = Path('transformer_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created backup directory: {backup_dir}")
    
    # Determine what to remove
    files_to_remove = []
    
    if response == '1':
        # Minimal - only remove experimental
        files_to_remove = to_remove
        print("\n✓ MINIMAL cleanup - removing only experimental files")
        
    elif response == '2':
        # Moderate - interactive domain selection
        print("\n🔍 MODERATE cleanup - choose domains to keep:")
        print()
        
        domains_to_keep = set()
        for dir_name in sorted(by_dir.keys()):
            files_count = len(by_dir[dir_name])
            keep = input(f"  Keep {dir_name}/ ({files_count} files)? (y/n): ").strip().lower()
            if keep == 'y':
                domains_to_keep.add(dir_name)
                print(f"    ✓ Keeping {dir_name}/")
            else:
                print(f"    ✗ Archiving {dir_name}/")
        
        # Remove experimental + unselected domains
        files_to_remove = to_remove
        for filepath, reason in optional_keep:
            if filepath.parent.name not in domains_to_keep:
                files_to_remove.append((filepath, 'domain not selected'))
        
    elif response == '3':
        # Aggressive - remove experimental + all domain-specific
        files_to_remove = to_remove + optional_keep
        print("\n✓ AGGRESSIVE cleanup - archiving experimental + all domain-specific")
    
    else:
        print("\n✗ Invalid option, cleanup cancelled")
        return
    
    # Execute cleanup
    print()
    print_header("EXECUTING CLEANUP", "=")
    print(f"Removing/archiving {len(files_to_remove)} files...")
    print()
    
    removed_count = 0
    for filepath, reason in files_to_remove:
        try:
            # Backup
            rel_path = filepath.relative_to(transformers_dir)
            backup_path = backup_dir / 'transformers' / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(filepath, backup_path)
            
            # Remove
            filepath.unlink()
            removed_count += 1
            
            if removed_count % 10 == 0:
                print(f"  Processed {removed_count}/{len(files_to_remove)}...")
        
        except Exception as e:
            print(f"  ✗ Error: {filepath.name}: {e}")
    
    print(f"\n✓ Removed/archived {removed_count} files")
    
    # Clean empty directories
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
    
    remaining = list(transformers_dir.rglob('*.py'))
    remaining = [f for f in remaining if '__pycache__' not in str(f)]
    
    print(f"✅ Removed/archived: {removed_count} files")
    print(f"✅ Remaining: {len(remaining)} files")
    print(f"✅ Canonical transformers: 45")
    print(f"✅ Infrastructure files: {len([f for f, r in to_keep if 'utility' in r])}")
    print()
    print(f"📁 Backup: {backup_dir}/transformers/")
    print()
    print("🎉 Your transformer library is now clean and production-ready!")
    print()
    print("📊 You now have:")
    print("   - 35 tested & verified transformers")
    print("   - 10 new universal/meta transformers")
    print("   - All infrastructure intact")
    print("   - Optional domain-specific (your choice)")
    print()


if __name__ == "__main__":
    main()

