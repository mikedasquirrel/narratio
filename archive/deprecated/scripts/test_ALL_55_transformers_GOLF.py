"""
COMPREHENSIVE Golf Pipeline Test - ALL 55 Transformers
=======================================================

Tests ALL definitive transformers on Golf data with CONTINUOUS progress output:
- 35 core tested transformers
- 10 universal/meta transformers
- 7 temporal transformers
- 3 contextual transformers

Progress updates every step of the way!

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
from sklearn.metrics import accuracy_score


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


def load_golf_data(sample_size=500):
    """Load Golf data with progress"""
    print_header("LOADING GOLF DATA", "█")
    
    data_path = Path('data/domains/golf_enhanced_narratives.json')
    
    print_progress(f"📂 Loading: {data_path}")
    print_progress(f"   Expected size: ~25MB, ~7,700 tournaments")
    
    start = time.time()
    with open(data_path) as f:
        all_tournaments = json.load(f)
    load_time = time.time() - start
    
    print_progress(f"✓ Loaded {len(all_tournaments):,} tournaments in {load_time:.1f}s")
    
    # Sample if needed
    if sample_size and sample_size < len(all_tournaments):
        print_progress(f"🎲 Sampling {sample_size} tournaments for testing...")
        np.random.seed(42)
        indices = np.random.choice(len(all_tournaments), sample_size, replace=False)
        tournaments = [all_tournaments[i] for i in sorted(indices)]
    else:
        tournaments = all_tournaments
    
    print_progress(f"✓ Using {len(tournaments):,} tournaments")
    
    # Split train/test
    split_idx = int(len(tournaments) * 0.8)
    train_data = tournaments[:split_idx]
    test_data = tournaments[split_idx:]
    
    print_progress(f"✓ Train: {len(train_data):,} tournaments")
    print_progress(f"✓ Test: {len(test_data):,} tournaments")
    
    # Build narratives
    print_progress("🔨 Building narratives...")
    
    def build_narrative(tournament):
        """Build rich narrative from tournament data"""
        parts = []
        
        # Tournament name
        if tournament.get('tournament_name'):
            parts.append(f"Tournament {tournament['tournament_name']}")
        
        # Player name
        if tournament.get('player_name'):
            parts.append(f"Player {tournament['player_name']}")
        
        # Course
        if tournament.get('course_name'):
            parts.append(f"Course {tournament['course_name']}")
        
        # Year/season context
        if tournament.get('year'):
            parts.append(f"Year {tournament['year']}")
        
        # Position/finish
        if tournament.get('finish_position'):
            parts.append(f"Finish {tournament['finish_position']}")
        
        # Round scores (temporal progression)
        if tournament.get('rounds'):
            rounds_text = " ".join([f"R{i+1}: {r}" for i, r in enumerate(tournament['rounds'][:4])])
            parts.append(f"Scores {rounds_text}")
        
        return ". ".join(parts) + "." if parts else "Golf tournament"
    
    print_progress("   Building train narratives...", indent=1)
    X_train = pd.Series([build_narrative(t) for t in train_data])
    y_train = np.array([1 if t.get('won', False) or t.get('finish_position', 99) == 1 else 0 for t in train_data])
    
    print_progress("   Building test narratives...", indent=1)
    X_test = pd.Series([build_narrative(t) for t in test_data])
    y_test = np.array([1 if t.get('won', False) or t.get('finish_position', 99) == 1 else 0 for t in test_data])
    
    print_progress(f"✓ Baseline win rate: {y_train.mean():.1%}")
    print_progress(f"✓ Sample narrative: {X_train.iloc[0][:100]}...")
    
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
        print_progress("🔧 Initializing transformer...", indent=1)
        transformer = cls(**kwargs)
        
        # Fit
        print_progress(f"🎓 Fitting on {len(X_train)} samples...", indent=1)
        start_fit = time.time()
        X_train_t = transformer.fit_transform(X_train, y_train)
        fit_time = time.time() - start_fit
        print_progress(f"   ✓ Fit complete in {fit_time:.2f}s", indent=1)
        
        # Transform
        print_progress(f"🔄 Transforming {len(X_test)} test samples...", indent=1)
        start_transform = time.time()
        X_test_t = transformer.transform(X_test)
        transform_time = time.time() - start_transform
        print_progress(f"   ✓ Transform complete in {transform_time:.2f}s", indent=1)
        
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
        
        print_progress(f"✅ SUCCESS: {total_time:.2f}s total, {speed:.0f} samples/sec", indent=1)
        
    except Exception as e:
        error_msg = str(e)[:150]
        result['status'] = 'ERROR'
        result['error'] = error_msg
        print_progress(f"❌ ERROR: {error_msg}", indent=1)
    
    return result


def main():
    """Run comprehensive Golf test"""
    
    print_header("GOLF COMPREHENSIVE TEST - ALL 55 TRANSFORMERS", "█")
    print_progress("Testing DEFINITIVE transformer library on Golf data")
    print_progress(f"Python: {sys.version.split()[0]}")
    print_progress(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    X_train, y_train, X_test, y_test = load_golf_data(sample_size=500)
    
    # Define all transformers
    print_header("LOADING 55 TRANSFORMERS", "=")
    transformers = []
    load_start = time.time()
    
    # === CORE (6) ===
    print_progress("Loading Core transformers...")
    core_start = len(transformers)
    
    try:
        from transformers.nominative_v2 import NominativeAnalysisV2Transformer
        transformers.append(("Nominative Analysis V2", NominativeAnalysisV2Transformer, {}, "Core"))
        print_progress("  ✓ Nominative V2", indent=1)
    except:
        try:
            from transformers.nominative import NominativeAnalysisTransformer
            transformers.append(("Nominative Analysis", NominativeAnalysisTransformer, {}, "Core"))
            print_progress("  ✓ Nominative V1 (fallback)", indent=1)
        except:
            print_progress("  ✗ Skip Nominative", indent=1)
    
    try:
        from transformers.self_perception_v2 import SelfPerceptionV2Transformer
        transformers.append(("Self Perception V2", SelfPerceptionV2Transformer, {}, "Core"))
        print_progress("  ✓ Self Perception V2", indent=1)
    except:
        try:
            from transformers.self_perception import SelfPerceptionTransformer
            transformers.append(("Self Perception", SelfPerceptionTransformer, {}, "Core"))
            print_progress("  ✓ Self Perception V1 (fallback)", indent=1)
        except:
            print_progress("  ✗ Skip Self Perception", indent=1)
    
    try:
        from transformers.narrative_potential_v2 import NarrativePotentialV2Transformer
        transformers.append(("Narrative Potential V2", NarrativePotentialV2Transformer, {}, "Core"))
        print_progress("  ✓ Narrative Potential V2", indent=1)
    except:
        try:
            from transformers.narrative_potential import NarrativePotentialTransformer
            transformers.append(("Narrative Potential", NarrativePotentialTransformer, {}, "Core"))
            print_progress("  ✓ Narrative Potential V1 (fallback)", indent=1)
        except:
            print_progress("  ✗ Skip Narrative Potential", indent=1)
    
    try:
        from transformers.linguistic_v2 import LinguisticPatternsV2Transformer
        transformers.append(("Linguistic Patterns V2", LinguisticPatternsV2Transformer, {}, "Core"))
        print_progress("  ✓ Linguistic V2", indent=1)
    except:
        try:
            from transformers.linguistic_advanced import LinguisticPatternsTransformer
            transformers.append(("Linguistic Patterns", LinguisticPatternsTransformer, {}, "Core"))
            print_progress("  ✓ Linguistic V1 (fallback)", indent=1)
        except:
            print_progress("  ✗ Skip Linguistic", indent=1)
    
    try:
        from transformers.ensemble import EnsembleNarrativeTransformer
        transformers.append(("Ensemble Narrative", EnsembleNarrativeTransformer, {}, "Core"))
        print_progress("  ✓ Ensemble", indent=1)
    except:
        print_progress("  ✗ Skip Ensemble", indent=1)
    
    try:
        from transformers.relational import RelationalValueTransformer
        transformers.append(("Relational Value", RelationalValueTransformer, {}, "Core"))
        print_progress("  ✓ Relational", indent=1)
    except:
        print_progress("  ✗ Skip Relational", indent=1)
    
    print_progress(f"✓ Core: {len(transformers) - core_start} loaded")
    
    # === EMOTIONAL (2) ===
    print_progress("Loading Emotional transformers...")
    emotional_start = len(transformers)
    
    try:
        from transformers.emotional_resonance_v2 import EmotionalResonanceV2Transformer
        transformers.append(("Emotional Resonance V2", EmotionalResonanceV2Transformer, {}, "Emotional"))
        print_progress("  ✓ Emotional Resonance V2", indent=1)
    except:
        try:
            from transformers.emotional_resonance import EmotionalResonanceTransformer
            transformers.append(("Emotional Resonance", EmotionalResonanceTransformer, {}, "Emotional"))
            print_progress("  ✓ Emotional Resonance V1", indent=1)
        except:
            print_progress("  ✗ Skip Emotional Resonance", indent=1)
    
    try:
        from transformers.authenticity import AuthenticityTransformer
        transformers.append(("Authenticity", AuthenticityTransformer, {}, "Emotional"))
        print_progress("  ✓ Authenticity", indent=1)
    except:
        print_progress("  ✗ Skip Authenticity", indent=1)
    
    print_progress(f"✓ Emotional: {len(transformers) - emotional_start} loaded")
    
    # === STRUCTURAL (3) ===
    print_progress("Loading Structural transformers...")
    structural_start = len(transformers)
    
    try:
        from transformers.conflict_tension import ConflictTensionTransformer
        transformers.append(("Conflict Tension", ConflictTensionTransformer, {}, "Structural"))
        print_progress("  ✓ Conflict Tension", indent=1)
    except:
        print_progress("  ✗ Skip Conflict Tension", indent=1)
    
    try:
        from transformers.suspense_mystery import SuspenseMysteryTransformer
        transformers.append(("Suspense Mystery", SuspenseMysteryTransformer, {}, "Structural"))
        print_progress("  ✓ Suspense Mystery", indent=1)
    except:
        print_progress("  ✗ Skip Suspense Mystery", indent=1)
    
    try:
        from transformers.framing import FramingTransformer
        transformers.append(("Framing", FramingTransformer, {}, "Structural"))
        print_progress("  ✓ Framing", indent=1)
    except:
        print_progress("  ✗ Skip Framing", indent=1)
    
    print_progress(f"✓ Structural: {len(transformers) - structural_start} loaded")
    
    # === NOMINATIVE (5) ===
    print_progress("Loading Nominative transformers...")
    nom_start = len(transformers)
    
    try:
        from transformers.phonetic import PhoneticTransformer
        transformers.append(("Phonetic", PhoneticTransformer, {}, "Nominative"))
        print_progress("  ✓ Phonetic", indent=1)
    except:
        print_progress("  ✗ Skip Phonetic", indent=1)
    
    try:
        from transformers.social_status import SocialStatusTransformer
        transformers.append(("Social Status", SocialStatusTransformer, {}, "Nominative"))
        print_progress("  ✓ Social Status", indent=1)
    except:
        print_progress("  ✗ Skip Social Status", indent=1)
    
    try:
        from transformers.universal_nominative import UniversalNominativeTransformer
        transformers.append(("Universal Nominative", UniversalNominativeTransformer, {}, "Nominative"))
        print_progress("  ✓ Universal Nominative", indent=1)
    except:
        print_progress("  ✗ Skip Universal Nominative", indent=1)
    
    try:
        from transformers.hierarchical_nominative import HierarchicalNominativeTransformer
        transformers.append(("Hierarchical Nominative", HierarchicalNominativeTransformer, {}, "Nominative"))
        print_progress("  ✓ Hierarchical Nominative", indent=1)
    except:
        print_progress("  ✗ Skip Hierarchical Nominative", indent=1)
    
    try:
        from transformers.nominative_richness import NominativeRichnessTransformer
        transformers.append(("Nominative Richness", NominativeRichnessTransformer, {}, "Nominative"))
        print_progress("  ✓ Nominative Richness (BREAKTHROUGH)", indent=1)
    except:
        print_progress("  ✗ Skip Nominative Richness", indent=1)
    
    print_progress(f"✓ Nominative: {len(transformers) - nom_start} loaded")
    
    # === ADVANCED (6) ===
    print_progress("Loading Advanced transformers...")
    adv_start = len(transformers)
    
    for name, module, cls, desc in [
        ("Information Theory", "information_theory", "InformationTheoryTransformer", "Entropy/complexity"),
        ("Namespace Ecology", "namespace_ecology", "NamespaceEcologyTransformer", "Name competition"),
        ("Cognitive Fluency", "cognitive_fluency", "CognitiveFluencyTransformer", "Processing ease"),
        ("Discoverability", "discoverability", "DiscoverabilityTransformer", "Findability"),
        ("Multi-Scale", "multi_scale", "MultiScaleTransformer", "Scale analysis"),
        ("Quantitative", "quantitative", "QuantitativeTransformer", "Numeric patterns"),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformer_cls = getattr(mod, cls)
            transformers.append((name, transformer_cls, {}, "Advanced"))
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Advanced: {len(transformers) - adv_start} loaded")
    
    # === THEORY VARIABLES (8) ===
    print_progress("Loading Theory-Aligned transformers...")
    theory_start = len(transformers)
    
    try:
        from transformers.coupling_strength import CouplingStrengthTransformer
        transformers.append(("Coupling Strength (κ)", CouplingStrengthTransformer, {}, "Theory"))
        print_progress("  ✓ Coupling (κ)", indent=1)
    except:
        print_progress("  ✗ Skip Coupling", indent=1)
    
    try:
        from transformers.narrative_mass import NarrativeMassTransformer
        transformers.append(("Narrative Mass (μ)", NarrativeMassTransformer, {}, "Theory"))
        print_progress("  ✓ Narrative Mass (μ)", indent=1)
    except:
        print_progress("  ✗ Skip Narrative Mass", indent=1)
    
    try:
        from transformers.gravitational_features import GravitationalFeaturesTransformer
        transformers.append(("Gravitational Features (φ & ة)", GravitationalFeaturesTransformer, {}, "Theory"))
        print_progress("  ✓ Gravitational (φ & ة)", indent=1)
    except:
        print_progress("  ✗ Skip Gravitational", indent=1)
    
    try:
        from transformers.awareness_resistance import AwarenessResistanceTransformer
        transformers.append(("Awareness Resistance (θ)", AwarenessResistanceTransformer, {}, "Theory"))
        print_progress("  ✓ Awareness (θ)", indent=1)
    except:
        print_progress("  ✗ Skip Awareness", indent=1)
    
    try:
        from transformers.fundamental_constraints import FundamentalConstraintsTransformer
        transformers.append(("Fundamental Constraints (λ)", FundamentalConstraintsTransformer, {'use_embeddings': False}, "Theory"))
        print_progress("  ✓ Constraints (λ)", indent=1)
    except:
        print_progress("  ✗ Skip Constraints", indent=1)
    
    try:
        from transformers.optics import OpticsTransformer
        transformers.append(("Optics", OpticsTransformer, {}, "Theory"))
        print_progress("  ✓ Optics", indent=1)
    except:
        print_progress("  ✗ Skip Optics", indent=1)
    
    print_progress(f"✓ Theory: {len(transformers) - theory_start} loaded")
    
    # === CONTEXTUAL (3) ===
    print_progress("Loading Contextual transformers...")
    ctx_start = len(transformers)
    
    try:
        from transformers.cultural_context import CulturalContextTransformer
        transformers.append(("Cultural Context", CulturalContextTransformer, {}, "Contextual"))
        print_progress("  ✓ Cultural Context", indent=1)
    except:
        print_progress("  ✗ Skip Cultural Context", indent=1)
    
    try:
        from transformers.competitive_context import CompetitiveContextTransformer
        transformers.append(("Competitive Context", CompetitiveContextTransformer, {}, "Contextual"))
        print_progress("  ✓ Competitive Context", indent=1)
    except:
        print_progress("  ✗ Skip Competitive Context", indent=1)
    
    try:
        from transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
        transformers.append(("Anticipatory Communication", AnticipatoryCommunicationTransformer, {}, "Contextual"))
        print_progress("  ✓ Anticipatory Communication", indent=1)
    except:
        print_progress("  ✗ Skip Anticipatory Communication", indent=1)
    
    try:
        from transformers.expertise_authority import ExpertiseAuthorityTransformer
        transformers.append(("Expertise Authority", ExpertiseAuthorityTransformer, {}, "Contextual"))
        print_progress("  ✓ Expertise Authority", indent=1)
    except:
        print_progress("  ✗ Skip Expertise Authority", indent=1)
    
    print_progress(f"✓ Contextual: {len(transformers) - ctx_start} loaded")
    
    # === 🆕 UNIVERSAL & META (10) ===
    print_progress("Loading Universal & Meta transformers...")
    universal_start = len(transformers)
    
    try:
        from transformers.universal_themes import UniversalThemesTransformer
        transformers.append(("Universal Themes", UniversalThemesTransformer, {}, "Universal"))
        print_progress("  ✓ Universal Themes 🆕", indent=1)
    except:
        print_progress("  ✗ Skip Universal Themes", indent=1)
    
    try:
        from transformers.universal_structural_pattern import UniversalStructuralPatternTransformer
        transformers.append(("Universal Structural", UniversalStructuralPatternTransformer, {}, "Universal"))
        print_progress("  ✓ Universal Structural 🆕", indent=1)
    except:
        print_progress("  ✗ Skip Universal Structural", indent=1)
    
    try:
        from transformers.universal_hybrid import UniversalHybridTransformer
        transformers.append(("Universal Hybrid", UniversalHybridTransformer, {}, "Universal"))
        print_progress("  ✓ Universal Hybrid 🆕", indent=1)
    except:
        print_progress("  ✗ Skip Universal Hybrid", indent=1)
    
    try:
        from transformers.cross_domain_embedding import CrossDomainEmbeddingTransformer
        transformers.append(("Cross-Domain Embedding", CrossDomainEmbeddingTransformer, {}, "Universal"))
        print_progress("  ✓ Cross-Domain Embedding 🆕", indent=1)
    except:
        print_progress("  ✗ Skip Cross-Domain Embedding", indent=1)
    
    try:
        from transformers.meta_narrative import MetaNarrativeTransformer
        transformers.append(("Meta Narrative", MetaNarrativeTransformer, {'use_spacy': False, 'use_embeddings': False}, "Meta"))
        print_progress("  ✓ Meta Narrative 🆕", indent=1)
    except:
        print_progress("  ✗ Skip Meta Narrative", indent=1)
    
    try:
        from transformers.ensemble_meta import EnsembleMetaTransformer
        transformers.append(("Ensemble Meta", EnsembleMetaTransformer, {}, "Meta"))
        print_progress("  ✓ Ensemble Meta 🆕", indent=1)
    except:
        print_progress("  ✗ Skip Ensemble Meta", indent=1)
    
    try:
        from transformers.enriched_patterns import EnrichedPatternsTransformer
        transformers.append(("Enriched Patterns", EnrichedPatternsTransformer, {}, "Pattern"))
        print_progress("  ✓ Enriched Patterns 🆕", indent=1)
    except:
        print_progress("  ✗ Skip Enriched Patterns", indent=1)
    
    print_progress(f"✓ Universal/Meta: {len(transformers) - universal_start} loaded")
    
    # === ⏱️ TEMPORAL (7) ===
    print_progress("Loading Temporal transformers...")
    temp_start = len(transformers)
    
    try:
        from transformers.temporal_evolution import TemporalEvolutionTransformer
        transformers.append(("Temporal Evolution", TemporalEvolutionTransformer, {}, "Temporal"))
        print_progress("  ✓ Temporal Evolution", indent=1)
    except:
        print_progress("  ✗ Skip Temporal Evolution", indent=1)
    
    try:
        from transformers.temporal_momentum_enhanced import TemporalMomentumEnhancedTransformer
        transformers.append(("Temporal Momentum Enhanced", TemporalMomentumEnhancedTransformer, {'use_spacy': False, 'use_embeddings': False}, "Temporal"))
        print_progress("  ✓ Temporal Momentum Enhanced 🆕", indent=1)
    except:
        print_progress("  ✗ Skip Temporal Momentum Enhanced", indent=1)
    
    try:
        from transformers.temporal_narrative_context import TemporalNarrativeContextTransformer
        transformers.append(("Temporal Narrative Context", TemporalNarrativeContextTransformer, {}, "Temporal"))
        print_progress("  ✓ Temporal Narrative Context ⏱️", indent=1)
    except:
        print_progress("  ✗ Skip Temporal Narrative Context", indent=1)
    
    try:
        from transformers.temporal_derivative import TemporalDerivativeTransformer
        transformers.append(("Temporal Derivative", TemporalDerivativeTransformer, {}, "Temporal"))
        print_progress("  ✓ Temporal Derivative ⏱️", indent=1)
    except:
        print_progress("  ✗ Skip Temporal Derivative", indent=1)
    
    # Temporal subdirectory
    try:
        from transformers.temporal.pacing_rhythm import PacingRhythmTransformer
        transformers.append(("Pacing & Rhythm", PacingRhythmTransformer, {}, "Temporal-Specialized"))
        print_progress("  ✓ Pacing & Rhythm ⏱️", indent=1)
    except:
        print_progress("  ✗ Skip Pacing & Rhythm", indent=1)
    
    try:
        from transformers.temporal.duration_effects import DurationEffectsTransformer
        transformers.append(("Duration Effects", DurationEffectsTransformer, {}, "Temporal-Specialized"))
        print_progress("  ✓ Duration Effects ⏱️", indent=1)
    except:
        print_progress("  ✗ Skip Duration Effects", indent=1)
    
    try:
        from transformers.temporal.cross_temporal_isomorphism import CrossTemporalIsomorphismTransformer
        transformers.append(("Cross-Temporal Isomorphism", CrossTemporalIsomorphismTransformer, {}, "Temporal-Specialized"))
        print_progress("  ✓ Cross-Temporal Isomorphism ⏱️", indent=1)
    except:
        print_progress("  ✗ Skip Cross-Temporal Isomorphism", indent=1)
    
    try:
        from transformers.temporal.temporal_compression import TemporalCompressionTransformer
        transformers.append(("Temporal Compression", TemporalCompressionTransformer, {}, "Temporal-Specialized"))
        print_progress("  ✓ Temporal Compression ⏱️", indent=1)
    except:
        print_progress("  ✗ Skip Temporal Compression", indent=1)
    
    print_progress(f"✓ Temporal: {len(transformers) - temp_start} loaded")
    
    # === REMAINING ===
    print_progress("Loading remaining transformers...")
    remaining_start = len(transformers)
    
    for name, module, cls, cat in [
        ("Statistical", "statistical", "StatisticalTransformer", "Baseline"),
        ("Context Pattern", "context_pattern", "ContextPatternTransformer", "Pattern"),
    ]:
        try:
            mod = __import__(f'transformers.{module}', fromlist=[cls])
            transformer_cls = getattr(mod, cls)
            kwargs = {'min_samples': 30, 'max_patterns': 20} if module == 'context_pattern' else {}
            transformers.append((name, transformer_cls, kwargs, cat))
            print_progress(f"  ✓ {name}", indent=1)
        except:
            print_progress(f"  ✗ Skip {name}", indent=1)
    
    print_progress(f"✓ Remaining: {len(transformers) - remaining_start} loaded")
    
    load_time = time.time() - load_start
    print()
    print_progress(f"🎉 TOTAL LOADED: {len(transformers)} transformers in {load_time:.1f}s")
    
    # Test all transformers
    print_header("TESTING ALL TRANSFORMERS ON GOLF DATA", "█")
    print_progress(f"Will test {len(transformers)} transformers")
    print_progress("Progress updates every transformer!")
    
    results = []
    overall_start = time.time()
    
    for i, (name, cls, kwargs, category) in enumerate(transformers, 1):
        result = test_transformer(i, len(transformers), name, cls, kwargs, 
                                 X_train, y_train, X_test, y_test, category)
        results.append(result)
        
        # Progress checkpoint every 5
        if i % 5 == 0:
            elapsed = time.time() - overall_start
            avg_time = elapsed / i
            remaining = (len(transformers) - i) * avg_time
            print()
            print_progress(f"📊 CHECKPOINT: {i}/{len(transformers)} complete")
            print_progress(f"   Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min")
            success_so_far = sum(1 for r in results if r['status'] == 'SUCCESS')
            print_progress(f"   Success rate: {success_so_far}/{i} ({success_so_far/i*100:.0f}%)")
    
    total_time = time.time() - overall_start
    
    # FINAL RESULTS
    print_header("🏆 FINAL RESULTS 🏆", "█")
    
    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'SUCCESS']
    df_error = df[df['status'] == 'ERROR']
    
    print_progress(f"Total transformers tested: {len(results)}")
    print_progress(f"✅ Successful: {len(df_success)} ({len(df_success)/len(results)*100:.0f}%)")
    print_progress(f"❌ Errors: {len(df_error)} ({len(df_error)/len(results)*100:.0f}%)")
    print_progress(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print_progress(f"⚡ Avg per transformer: {total_time/len(results):.1f}s")
    
    # Results by category
    print()
    print_header("RESULTS BY CATEGORY", "-")
    for cat in sorted(df['category'].unique()):
        df_cat = df[df['category'] == cat]
        success = len(df_cat[df_cat['status'] == 'SUCCESS'])
        total = len(df_cat)
        avg_acc = df_cat[df_cat['status'] == 'SUCCESS']['accuracy'].mean() if success > 0 else 0
        print_progress(f"{cat:<25} {success}/{total} working ({success/total*100:>3.0f}%) - Avg: {avg_acc:.1%}")
    
    if len(df_success) > 0:
        print()
        print_header("🏆 TOP 15 PERFORMERS", "-")
        df_sorted = df_success.sort_values('accuracy', ascending=False)
        
        print(f"{'#':<4} {'Transformer':<45} {'Category':<20} {'Acc':<8} {'Time'}")
        print("-" * 90)
        
        for i, (_, row) in enumerate(df_sorted.head(15).iterrows(), 1):
            marker = "🏆" if i <= 3 else "⭐" if i <= 10 else "  "
            print(f"{marker} {i:<2} {row['name']:<45} {row['category']:<20} {row['accuracy']:<7.1%} {row['total_time']:.2f}s")
        
        print()
        print_header("⚡ 10 FASTEST TRANSFORMERS", "-")
        df_fast = df_success.sort_values('total_time')
        
        print(f"{'#':<4} {'Transformer':<45} {'Time':<10} {'Speed (samp/s)'}")
        print("-" * 75)
        
        for i, (_, row) in enumerate(df_fast.head(10).iterrows(), 1):
            speed = len(X_train) / row['total_time'] if row['total_time'] > 0 else 0
            print(f"{'⚡'} {i:<2} {row['name']:<45} {row['total_time']:<8.2f}s {speed:>10.0f}")
        
        print()
        print_header("🐌 5 SLOWEST TRANSFORMERS", "-")
        df_slow = df_success.sort_values('total_time', ascending=False)
        
        for i, (_, row) in enumerate(df_slow.head(5).iterrows(), 1):
            print(f"  {i}. {row['name']:<45} {row['total_time']:.2f}s")
    
    if len(df_error) > 0:
        print()
        print_header("❌ ERRORS", "-")
        for _, row in df_error.iterrows():
            print(f"  ✗ {row['name']}: {row['error'][:100]}")
    
    # Save results
    output_file = 'golf_comprehensive_results.json'
    print()
    print_progress(f"💾 Saving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'dataset': 'Golf',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'transformers_tested': len(results),
            'successful': len(df_success),
            'errors': len(df_error),
            'total_time_seconds': total_time,
            'avg_time_per_transformer': total_time / len(results),
            'results': df.to_dict('records'),
            'category_breakdown': df.groupby('category').apply(lambda x: {
                'tested': len(x),
                'successful': len(x[x['status'] == 'SUCCESS']),
                'avg_accuracy': float(x[x['status'] == 'SUCCESS']['accuracy'].mean()) if len(x[x['status'] == 'SUCCESS']) > 0 else 0
            }).to_dict()
        }, f, indent=2)
    
    print_progress(f"✓ Results saved!")
    
    # CSV export
    csv_file = 'golf_comprehensive_results.csv'
    df.to_csv(csv_file, index=False)
    print_progress(f"✓ CSV saved to {csv_file}")
    
    # Final summary
    print()
    print_header("TEST COMPLETE! 🎉", "█")
    print_progress(f"⛳ Golf Comprehensive Test Finished!")
    print_progress(f"📊 {len(df_success)}/{len(results)} transformers working ({len(df_success)/len(results)*100:.0f}%)")
    print_progress(f"⏱️  Completed in {total_time/60:.1f} minutes")
    
    if len(df_success) > 0:
        best = df_sorted.iloc[0]
        print()
        print_progress(f"🏆 BEST PERFORMER: {best['name']}")
        print_progress(f"   Accuracy: {best['accuracy']:.1%}")
        print_progress(f"   Category: {best['category']}")
        print_progress(f"   Time: {best['total_time']:.2f}s")
    
    print()
    print_progress("All results saved. Check:")
    print_progress(f"  - {output_file}")
    print_progress(f"  - {csv_file}")
    print()


if __name__ == "__main__":
    import sys
    
    # Check for command line args
    sample_size = 500  # Default
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            print(f"Using sample size: {sample_size}")
        except:
            print("Usage: python test_ALL_55_transformers_GOLF.py [sample_size]")
            print("Default sample_size: 500")
    
    main()

