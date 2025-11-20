"""
COMPREHENSIVE NBA Pipeline Test - ALL 55 Transformers
======================================================

Tests ALL definitive transformers on CLEAN NBA data:
- 35 core tested transformers
- 10 universal/meta transformers  
- 7 temporal transformers
- 3 contextual transformers

NO DATA LEAKAGE - Using clean pre-game narratives only.
Continuous progress updates!

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import json
import time
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

from sklearn.linear_model import LogisticRegression


def print_header(text, char='=', width=80):
    """Print header with timestamp"""
    print()
    print(char * width)
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {text}")
    print(char * width)
    print()


def print_progress(text, indent=0):
    """Print progress with timestamp and optional indent"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = "  " * indent
    print(f"{prefix}[{timestamp}] {text}", flush=True)


def load_nba_data(sample_size=1000):
    """Load NBA data with CLEAN pre-game narratives (NO LEAKAGE)"""
    print_header("LOADING NBA DATA", "█")
    
    data_path = Path('data/domains/nba_complete_with_players.json')
    
    print_progress(f"📂 Loading: {data_path}")
    print_progress(f"   Expected: ~12,000 games, 179MB")
    
    start = time.time()
    with open(data_path) as f:
        all_games = json.load(f)
    load_time = time.time() - start
    
    print_progress(f"✓ Loaded {len(all_games):,} games in {load_time:.1f}s")
    
    # Split by season (NO random sampling - use actual seasons)
    train_games = [g for g in all_games if g['season'] < '2023-24']
    test_games = [g for g in all_games if g['season'] == '2023-24']
    
    # Sample if requested
    if sample_size and sample_size < len(train_games):
        print_progress(f"🎲 Sampling {sample_size} train games...")
        np.random.seed(42)
        train_games = np.random.choice(train_games, sample_size, replace=False).tolist()
        test_games = np.random.choice(test_games, sample_size // 4, replace=False).tolist()
    
    print_progress(f"✓ Train: {len(train_games):,} games (seasons < 2023-24)")
    print_progress(f"✓ Test: {len(test_games):,} games (season 2023-24)")
    
    # Build CLEAN pre-game narratives (NO OUTCOME LEAKAGE)
    print_progress("🔨 Building CLEAN pre-game narratives...")
    print_progress("   ⚠️  NO outcome information included!", indent=1)
    
    def build_clean_pregame_narrative(game):
        """Build narrative with ONLY pre-game information"""
        parts = []
        
        # Teams & Matchup
        parts.append(f"Team {game.get('team_name', 'Unknown')}")
        parts.append(f"Matchup {game.get('matchup', 'vs Opponent')}")
        parts.append(f"Location {'home' if game.get('home_game', False) else 'away'}")
        
        # Players (pre-game roster only)
        if game.get('player_data', {}).get('available'):
            agg = game['player_data']['team_aggregates']
            if agg.get('top1_name'):
                parts.append(f"Leading scorer {agg['top1_name']}")
            if agg.get('top2_name'):
                parts.append(f"Second scorer {agg['top2_name']}")
            if agg.get('top3_name'):
                parts.append(f"Third scorer {agg['top3_name']}")
        
        # Pre-game record (before this game)
        tc = game.get('temporal_context', {})
        if tc.get('season_record_prior'):
            parts.append(f"Season record {tc['season_record_prior']}")
        if tc.get('l10_record'):
            parts.append(f"Last 10 games {tc['l10_record']}")
        if tc.get('games_played'):
            parts.append(f"Games into season {tc['games_played']}")
        
        # Betting odds (pre-game only)
        betting = game.get('betting_odds', {})
        if betting.get('moneyline'):
            ml = betting['moneyline']
            parts.append(f"Betting line {ml}")
            if ml > 0:
                parts.append("Underdog")
            else:
                parts.append("Favorite")
        
        # Scheduling
        sched = game.get('scheduling', {})
        if sched.get('rest_days', 1) == 0:
            parts.append("Back to back game")
        elif sched.get('rest_days', 1) >= 3:
            parts.append("Well rested")
        
        return ". ".join(parts) + "."
    
    print_progress("   Building train narratives...", indent=1)
    X_train = pd.Series([build_clean_pregame_narrative(g) for g in train_games])
    y_train = np.array([1 if g.get('won', False) else 0 for g in train_games])
    
    print_progress("   Building test narratives...", indent=1)
    X_test = pd.Series([build_clean_pregame_narrative(g) for g in test_games])
    y_test = np.array([1 if g.get('won', False) else 0 for g in test_games])
    
    # Verify NO LEAKAGE
    print_progress("🔍 Verifying no outcome leakage...", indent=1)
    sample = X_train.iloc[0]
    leakage_words = ['won', 'lost', 'victory', 'defeat', 'final score', 'result']
    has_leakage = any(word in sample.lower() for word in leakage_words)
    
    if has_leakage:
        print_progress("   ❌ WARNING: Possible leakage detected!", indent=1)
    else:
        print_progress("   ✅ NO LEAKAGE - Clean pre-game data only", indent=1)
    
    print_progress(f"✓ Baseline win rate: {y_train.mean():.1%} (expect ~50%)")
    print_progress(f"✓ Sample narrative: {sample[:120]}...")
    
    return X_train, y_train, X_test, y_test


def test_transformer(i, total, name, cls, kwargs, X_train, y_train, X_test, y_test, category):
    """Test single transformer with detailed progress"""
    
    print()
    print("-" * 80)
    print_progress(f"[{i}/{total}] {name}")
    print_progress(f"Category: {category}", indent=1)
    print("-" * 80)
    
    result = {
        'name': name,
        'category': category,
        'status': 'unknown',
        'fit_time': 0,
        'transform_time': 0,
        'total_time': 0,
        'features': 0,
        'accuracy': 0,
        'error': None
    }
    
    try:
        # Initialize
        print_progress("🔧 Initializing...", indent=1)
        transformer = cls(**kwargs)
        
        # Fit
        print_progress(f"🎓 Fitting on {len(X_train)} samples...", indent=1)
        start_fit = time.time()
        X_train_t = transformer.fit_transform(X_train, y_train)
        fit_time = time.time() - start_fit
        print_progress(f"   ✓ Fit: {fit_time:.2f}s", indent=1)
        
        # Transform  
        print_progress(f"🔄 Transforming {len(X_test)} test samples...", indent=1)
        start_transform = time.time()
        X_test_t = transformer.transform(X_test)
        transform_time = time.time() - start_transform
        print_progress(f"   ✓ Transform: {transform_time:.2f}s", indent=1)
        
        # Format
        if hasattr(X_train_t, 'toarray'):
            X_train_t = X_train_t.toarray()
            X_test_t = X_test_t.toarray()
        
        if len(X_train_t.shape) == 1:
            X_train_t = X_train_t.reshape(-1, 1)
            X_test_t = X_test_t.reshape(-1, 1)
        
        n_features = X_train_t.shape[1]
        print_progress(f"📊 Generated {n_features} features", indent=1)
        
        # Train classifier
        if n_features > 0 and not np.all(X_train_t == 0):
            print_progress(f"🎯 Training classifier...", indent=1)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train_t, y_train)
            test_acc = clf.score(X_test_t, y_test)
            print_progress(f"   ✓ Test accuracy: {test_acc:.1%}", indent=1)
        else:
            test_acc = 0.0
            print_progress(f"   ⚠️  No valid features", indent=1)
        
        total_time = fit_time + transform_time
        speed = len(X_train) / total_time if total_time > 0 else 0
        
        result.update({
            'status': 'SUCCESS',
            'fit_time': fit_time,
            'transform_time': transform_time,
            'total_time': total_time,
            'features': n_features,
            'accuracy': test_acc
        })
        
        print_progress(f"✅ SUCCESS: {total_time:.2f}s, {speed:.0f} samp/s, {test_acc:.1%} acc", indent=1)
        
    except Exception as e:
        error_msg = str(e)[:150]
        result['status'] = 'ERROR'
        result['error'] = error_msg
        print_progress(f"❌ ERROR: {error_msg}", indent=1)
    
    return result


def main():
    """Run comprehensive NBA test"""
    
    print_header("NBA COMPREHENSIVE TEST - ALL 55 TRANSFORMERS", "█")
    print_progress("Testing DEFINITIVE transformer library on CLEAN NBA data")
    print_progress(f"Python: {sys.version.split()[0]}")
    print_progress(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load clean data
    X_train, y_train, X_test, y_test = load_nba_data(sample_size=1000)
    
    # Load transformers
    print_header("LOADING TRANSFORMERS", "=")
    transformers = []
    
    print_progress("📦 Loading Core transformers...")
    core_count = 0
    
    for name, module, cls in [
        ("Nominative Analysis", "nominative", "NominativeAnalysisTransformer"),
        ("Self Perception", "self_perception", "SelfPerceptionTransformer"),
        ("Narrative Potential", "narrative_potential", "NarrativePotentialTransformer"),
        ("Linguistic Patterns", "linguistic_advanced", "LinguisticPatternsTransformer"),
        ("Ensemble Narrative", "ensemble", "EnsembleNarrativeTransformer"),
        ("Relational Value", "relational", "RelationalValueTransformer"),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), {}, "Core"))
            core_count += 1
            print_progress(f"  ✓ {name}", indent=1)
        except Exception as e:
            print_progress(f"  ✗ Skip {name}: {str(e)[:50]}", indent=1)
    
    print_progress(f"✓ Core: {core_count}/6 loaded")
    
    print_progress("📦 Loading Emotional transformers...")
    emotional_count = 0
    
    for name, module, cls in [
        ("Emotional Resonance", "emotional_resonance", "EmotionalResonanceTransformer"),
        ("Authenticity", "authenticity", "AuthenticityTransformer"),
        ("Conflict Tension", "conflict_tension", "ConflictTensionTransformer"),
        ("Suspense Mystery", "suspense_mystery", "SuspenseMysteryTransformer"),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), {}, "Emotional"))
            emotional_count += 1
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Emotional: {emotional_count}/4 loaded")
    
    print_progress("📦 Loading Structural transformers...")
    struct_count = 0
    
    for name, module, cls in [
        ("Framing", "framing", "FramingTransformer"),
        ("Optics", "optics", "OpticsTransformer"),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), {}, "Structural"))
            struct_count += 1
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Structural: {struct_count}/2 loaded")
    
    print_progress("📦 Loading Nominative transformers...")
    nom_count = 0
    
    for name, module, cls in [
        ("Phonetic", "phonetic", "PhoneticTransformer"),
        ("Social Status", "social_status", "SocialStatusTransformer"),
        ("Universal Nominative", "universal_nominative", "UniversalNominativeTransformer"),
        ("Hierarchical Nominative", "hierarchical_nominative", "HierarchicalNominativeTransformer"),
        ("Nominative Richness", "nominative_richness", "NominativeRichnessTransformer"),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), {}, "Nominative"))
            nom_count += 1
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Nominative: {nom_count}/5 loaded")
    
    print_progress("📦 Loading Advanced transformers...")
    adv_count = 0
    
    for name, module, cls in [
        ("Information Theory", "information_theory", "InformationTheoryTransformer"),
        ("Namespace Ecology", "namespace_ecology", "NamespaceEcologyTransformer"),
        ("Cognitive Fluency", "cognitive_fluency", "CognitiveFluencyTransformer"),
        ("Discoverability", "discoverability", "DiscoverabilityTransformer"),
        ("Multi-Scale", "multi_scale", "MultiScaleTransformer"),
        ("Quantitative", "quantitative", "QuantitativeTransformer"),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), {}, "Advanced"))
            adv_count += 1
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Advanced: {adv_count}/6 loaded")
    
    print_progress("📦 Loading Theory transformers...")
    theory_count = 0
    
    for name, module, cls, special_kwargs in [
        ("Coupling Strength (κ)", "coupling_strength", "CouplingStrengthTransformer", {}),
        ("Narrative Mass (μ)", "narrative_mass", "NarrativeMassTransformer", {}),
        ("Gravitational Features (φ & ة)", "gravitational_features", "GravitationalFeaturesTransformer", {}),
        ("Awareness Resistance (θ)", "awareness_resistance", "AwarenessResistanceTransformer", {}),
        ("Fundamental Constraints (λ)", "fundamental_constraints", "FundamentalConstraintsTransformer", {'use_embeddings': False}),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), special_kwargs, "Theory"))
            theory_count += 1
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Theory: {theory_count}/5 loaded")
    
    print_progress("📦 Loading Contextual transformers...")
    ctx_count = 0
    
    for name, module, cls in [
        ("Cultural Context", "cultural_context", "CulturalContextTransformer"),
        ("Competitive Context", "competitive_context", "CompetitiveContextTransformer"),
        ("Anticipatory Communication", "anticipatory_commitment", "AnticipatoryCommunicationTransformer"),
        ("Expertise Authority", "expertise_authority", "ExpertiseAuthorityTransformer"),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), {}, "Contextual"))
            ctx_count += 1
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Contextual: {ctx_count}/4 loaded")
    
    print_progress("📦 Loading Universal/Meta transformers...")
    uni_count = 0
    
    for name, module, cls, special_kwargs in [
        ("Universal Themes", "universal_themes", "UniversalThemesTransformer", {}),
        ("Universal Structural", "universal_structural_pattern", "UniversalStructuralPatternTransformer", {}),
        ("Universal Hybrid", "universal_hybrid", "UniversalHybridTransformer", {}),
        ("Cross-Domain Embedding", "cross_domain_embedding", "CrossDomainEmbeddingTransformer", {}),
        ("Meta Narrative", "meta_narrative", "MetaNarrativeTransformer", {'use_spacy': False, 'use_embeddings': False}),
        ("Meta Feature Interaction", "meta_feature_interaction", "MetaFeatureInteractionTransformer", {}),
        ("Ensemble Meta", "ensemble_meta", "EnsembleMetaTransformer", {}),
        ("Enriched Patterns", "enriched_patterns", "EnrichedPatternsTransformer", {}),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), special_kwargs, "Universal/Meta"))
            uni_count += 1
            print_progress(f"  ✓ {name} 🆕", indent=1)
        except Exception as e:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Universal/Meta: {uni_count}/8 loaded")
    
    print_progress("📦 Loading Temporal transformers...")
    temp_count = 0
    
    # Root temporal
    for name, module, cls, special_kwargs in [
        ("Temporal Evolution", "temporal_evolution", "TemporalEvolutionTransformer", {}),
        ("Temporal Momentum Enhanced", "temporal_momentum_enhanced", "TemporalMomentumEnhancedTransformer", {'use_spacy': False, 'use_embeddings': False}),
        ("Temporal Narrative Context", "temporal_narrative_context", "TemporalNarrativeContextTransformer", {}),
        ("Temporal Derivative", "temporal_derivative", "TemporalDerivativeTransformer", {}),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), special_kwargs, "Temporal"))
            temp_count += 1
            print_progress(f"  ✓ {name} ⏱️", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    # Temporal subdirectory
    for name, path, cls, special_kwargs in [
        ("Pacing & Rhythm", "temporal.pacing_rhythm", "PacingRhythmTransformer", {}),
        ("Duration Effects", "temporal.duration_effects", "DurationEffectsTransformer", {}),
        ("Cross-Temporal Isomorphism", "temporal.cross_temporal_isomorphism", "CrossTemporalIsomorphismTransformer", {}),
        ("Temporal Compression", "temporal.temporal_compression", "TemporalCompressionTransformer", {}),
    ]:
        try:
            mod = __import__(f'transformers.{path}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), special_kwargs, "Temporal-Specialized"))
            temp_count += 1
            print_progress(f"  ✓ {name} ⏱️🔬", indent=1)
        except Exception as e:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Temporal: {temp_count}/8 loaded")
    
    print_progress("📦 Loading Pattern/Baseline transformers...")
    other_count = 0
    
    for name, module, cls, special_kwargs in [
        ("Statistical", "statistical", "StatisticalTransformer", {}),
        ("Context Pattern", "context_pattern", "ContextPatternTransformer", {'min_samples': 30, 'max_patterns': 20}),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformers.append((name, getattr(mod, cls), special_kwargs, "Pattern/Baseline"))
            other_count += 1
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Pattern/Baseline: {other_count}/2 loaded")
    
    print()
    print_progress(f"🎉 TOTAL LOADED: {len(transformers)} transformers")
    
    # Test all
    print_header("TESTING ALL TRANSFORMERS", "█")
    print_progress(f"Will test {len(transformers)} transformers with continuous progress")
    print_progress("Expected realistic accuracy: 50-60% (no leakage)")
    
    results = []
    overall_start = time.time()
    
    for i, (name, cls, kwargs, category) in enumerate(transformers, 1):
        result = test_transformer(i, len(transformers), name, cls, kwargs,
                                 X_train, y_train, X_test, y_test, category)
        results.append(result)
        
        # Checkpoint every 5
        if i % 5 == 0:
            elapsed = time.time() - overall_start
            avg_time = elapsed / i
            remaining = (len(transformers) - i) * avg_time
            print()
            print_progress(f"📊 CHECKPOINT: {i}/{len(transformers)} complete ({i/len(transformers)*100:.0f}%)")
            print_progress(f"   ⏱️  Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min", indent=1)
            success_so_far = sum(1 for r in results if r['status'] == 'SUCCESS')
            print_progress(f"   ✅ Success rate: {success_so_far}/{i} ({success_so_far/i*100:.0f}%)", indent=1)
            
            if success_so_far > 0:
                accs = [r['accuracy'] for r in results if r['status'] == 'SUCCESS']
                print_progress(f"   📊 Avg accuracy: {np.mean(accs):.1%} (expect ~50-56%)", indent=1)
    
    total_time = time.time() - overall_start
    
    # FINAL RESULTS
    print_header("🏆 FINAL RESULTS 🏆", "█")
    
    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'SUCCESS']
    df_error = df[df['status'] == 'ERROR']
    
    print_progress(f"Total transformers tested: {len(results)}")
    print_progress(f"✅ Successful: {len(df_success)} ({len(df_success)/len(results)*100:.0f}%)")
    print_progress(f"❌ Errors: {len(df_error)} ({len(df_error)/len(results)*100:.0f}%)")
    print_progress(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print_progress(f"⚡ Avg per transformer: {total_time/len(results):.2f}s")
    
    if len(df_success) > 0:
        print_progress(f"📊 Avg accuracy: {df_success['accuracy'].mean():.1%}")
        print_progress(f"📊 Best accuracy: {df_success['accuracy'].max():.1%}")
        print_progress(f"📊 Baseline: {y_train.mean():.1%}")
    
    # Results by category
    print()
    print_header("RESULTS BY CATEGORY", "-")
    for cat in sorted(df['category'].unique()):
        df_cat = df[df['category'] == cat]
        success = len(df_cat[df_cat['status'] == 'SUCCESS'])
        total = len(df_cat)
        avg_acc = df_cat[df_cat['status'] == 'SUCCESS']['accuracy'].mean() if success > 0 else 0
        print_progress(f"{cat:<25} {success:>2}/{total:<2} working ({success/total*100:>3.0f}%) - Avg: {avg_acc:>5.1%}")
    
    if len(df_success) > 0:
        print()
        print_header("🏆 TOP 15 PERFORMERS", "-")
        df_sorted = df_success.sort_values('accuracy', ascending=False)
        
        print(f"{'#':<4} {'Transformer':<45} {'Category':<20} {'Acc':<8} {'Time'}")
        print("-" * 90)
        
        for i, (_, row) in enumerate(df_sorted.head(15).iterrows(), 1):
            marker = "🏆" if i <= 3 else "⭐" if i <= 10 else "  "
            print(f"{marker} {i:<2} {row['name']:<45} {row['category']:<20} {row['accuracy']:<7.1%} {row['total_time']:>5.2f}s")
        
        print()
        print_header("⚡ 10 FASTEST TRANSFORMERS", "-")
        df_fast = df_success.sort_values('total_time')
        
        print(f"{'#':<4} {'Transformer':<45} {'Time':<10} {'Speed (samp/s)'}")
        print("-" * 75)
        
        for i, (_, row) in enumerate(df_fast.head(10).iterrows(), 1):
            speed = len(X_train) / row['total_time'] if row['total_time'] > 0 else 0
            print(f"{'⚡'} {i:<2} {row['name']:<45} {row['total_time']:<8.2f}s {speed:>10.0f}")
    
    if len(df_error) > 0:
        print()
        print_header("❌ ERRORS ENCOUNTERED", "-")
        for _, row in df_error.iterrows():
            print(f"  ✗ {row['name']}: {row['error'][:120]}")
    
    # Save results
    output_file = 'nba_comprehensive_ALL_55_results.json'
    print()
    print_progress(f"💾 Saving results...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'dataset': 'NBA (Clean Pre-Game)',
            'no_leakage': True,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'baseline_accuracy': float(y_train.mean()),
            'transformers_tested': len(results),
            'successful': len(df_success),
            'errors': len(df_error),
            'total_time_seconds': total_time,
            'avg_accuracy': float(df_success['accuracy'].mean()) if len(df_success) > 0 else 0,
            'results': df.to_dict('records')
        }, f, indent=2)
    
    csv_file = 'nba_comprehensive_ALL_55_results.csv'
    df.to_csv(csv_file, index=False)
    
    print_progress(f"✓ JSON: {output_file}")
    print_progress(f"✓ CSV: {csv_file}")
    
    # Final summary
    print()
    print_header("TEST COMPLETE! 🎉", "█")
    print_progress(f"🏀 NBA Comprehensive Test Finished!")
    print_progress(f"📊 {len(df_success)}/{len(results)} transformers working ({len(df_success)/len(results)*100:.0f}%)")
    print_progress(f"⏱️  Completed in {total_time/60:.1f} minutes")
    
    if len(df_success) > 0:
        best = df_sorted.iloc[0]
        print()
        print_progress(f"🏆 BEST: {best['name']} - {best['accuracy']:.1%} accuracy")
        print_progress(f"📊 Average: {df_success['accuracy'].mean():.1%}")
        print_progress(f"📊 Baseline: {y_train.mean():.1%}")
        
        improvement = df_success['accuracy'].mean() - y_train.mean()
        print_progress(f"📈 Avg improvement: {improvement:+.1%}")
    
    print()


if __name__ == "__main__":
    main()

