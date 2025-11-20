"""
Centralized Transformer Selection System

Intelligently selects appropriate transformers based on:
- π value (narrativity)
- Domain type (sports, entertainment, nominative, etc.)
- Data characteristics (text quality, length, structure)
- Renovation requirements (always include temporal, multi-stream, etc.)

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path


class TransformerSelector:
    """
    Intelligent transformer selection based on domain characteristics.
    
    Ensures all domains benefit from November 2025 renovation transformers
    while adapting to specific domain needs.
    
    The selector now cross-checks the transformer registry on import. Any
    newly created transformer MUST be added to one of the selection pools
    below, otherwise a warning will be emitted to alert future agents.
    """
    
    # Renovation transformers - ALWAYS included for consistency
    RENOVATION_TRANSFORMERS = {
        'temporal_dynamics': [
            'TemporalCompressionTransformer',
            'DurationEffectsTransformer',
            'PacingRhythmTransformer',
            'CrossTemporalIsomorphismTransformer',
            'TemporalEvolutionTransformer',
            'TemporalDerivativeTransformer',
            'TemporalNarrativeContextTransformer'
        ],
        'cross_cultural': [
            'CrossCulturalArchetypeTransformer'
        ],
        'linguistic': [
            'ActantialStructureTransformer',
            'ConceptualMetaphorTransformer',
            'LabovianNarrativeTransformer',
            'DiscourseAnalysisTransformer',
            'NarrativeSemioticsTransformer'
        ],
        'cognitive': [
            'CognitiveLoadTransformer',
            'EmbodiedMetaphorTransformer',
            'ScriptDeviationTransformer',
            'AttentionalStructureTransformer',
            'MemorabilityTransformer'
        ],
        'anthropological': [
            'RitualStructureTransformer'
        ]
    }
    
    # Core transformers - used across all domains
    CORE_TRANSFORMERS = [
        'StatisticalTransformer',
        'NominativeAnalysisTransformer',
        'SelfPerceptionTransformer',
        'NarrativePotentialTransformer',
        'LinguisticPatternsTransformer',
        'RelationalValueTransformer',
        'EnsembleNarrativeTransformer'
    ]
    # Next-gen upgrades that should travel with the core set
    NEXT_GEN_CORE_TRANSFORMERS = [
        'NominativeAnalysisV2Transformer',
        'SelfPerceptionV2Transformer',
        'NarrativePotentialV2Transformer',
        'LinguisticPatternsV2Transformer',
        'EmotionalResonanceV2Transformer'
    ]
    
    # π-dependent transformers
    LOW_PI_TRANSFORMERS = [
        'QuantitativeTransformer',
        'FundamentalConstraintsTransformer',
        'InformationTheoryTransformer',
        'CognitiveFluencyTransformer'
    ]
    
    MEDIUM_PI_TRANSFORMERS = [
        'OpticsTransformer',
        'FramingTransformer',
        'EmotionalResonanceTransformer',
        'ConflictTensionTransformer',
        'ExpertiseAuthorityTransformer'
    ]
    
    HIGH_PI_TRANSFORMERS = [
        'PhoneticTransformer',
        'SocialStatusTransformer',
        'AuthenticityTransformer',
        'CulturalContextTransformer',
        'SuspenseMysteryTransformer',
        'CouplingStrengthTransformer',
        'NarrativeMassTransformer',
        # Enhanced narrative transformers
        'LinguisticResonanceTransformer',
        'CulturalZeitgeistTransformer',
        'MetaNarrativeAwarenessTransformer',
        # Invisible narrative context transformer
        'NarrativeInterferenceTransformer'
    ]
    
    # Full nominative suite for high-nominative domains
    NOMINATIVE_SUITE = [
        'PhoneticTransformer',
        'SocialStatusTransformer',
        'UniversalNominativeTransformer',
        'HierarchicalNominativeTransformer',
        'NominativeInteractionTransformer',
        'PureNominativePredictorTransformer',
        'NominativeRichnessTransformer',
        'NominativeAnalysisV2Transformer'
    ]
    
    # Domain-specific transformers
    SPORTS_TRANSFORMERS = [
        'CompetitiveContextTransformer',
        'TemporalMomentumEnhancedTransformer',
        'MatchupAdvantageTransformer',
        'ReputationPrestigeTransformer',
        'MomentumVelocityTransformer',
        # Enhanced narrative transformers
        'DeepArchetypeTransformer',
        'NarrativeCompletionPressureTransformer',
        'TemporalNarrativeCyclesTransformer',
        'RitualCeremonyTransformer',
        'GeographicNarrativeTransformer',
        # Invisible narrative context transformers
        'ScheduleNarrativeTransformer',
        'MilestoneProximityTransformer',
        'CalendarRhythmTransformer',
        'BroadcastNarrativeTransformer',
        'OpponentContextTransformer',
        'SeasonSeriesNarrativeTransformer',
        'EliminationProximityTransformer'
    ]
    # Multi-modal / semantic expansion set
    ADVANCED_SEMANTIC_TRANSFORMERS = [
        'CrossmodalTransformer',
        'CrossLingualTransformer',
        'AudioTransformer',
        'EmotionalSemanticTransformer'
    ]
    
    ENTERTAINMENT_TRANSFORMERS = [
        'EmotionalResonanceTransformer',
        'VisualMultimodalTransformer',
        'SuspenseMysteryTransformer',
        'CulturalResonanceTransformer',
        'CharacterComplexityTransformer',
        'NarrativeDevicesTransformer'
    ]
    
    BUSINESS_TRANSFORMERS = [
        'NarrativePotentialTransformer',
        'NamespaceEcologyTransformer',
        'AnticipatoryCommunicationTransformer',
        'DiscoverabilityTransformer',
        'OriginStoryTransformer',
        'CommunityNetworkTransformer',
        'ScarcityExclusivityTransformer'
    ]
    
    # Theory-aligned transformers for all domains
    THEORY_ALIGNED = [
        'GravitationalFeaturesTransformer',
        'AwarenessResistanceTransformer',
        'ContextPatternTransformer'
    ]
    
    # Multi-scale transformers for hierarchical domains
    MULTI_SCALE = [
        'MultiScaleTransformer',
        'MultiPerspectiveTransformer',
        'ScaleInteractionTransformer'
    ]
    
    UNIVERSAL_META_TRANSFORMERS = [
        'MetaNarrativeTransformer',
        'MetaFeatureInteractionTransformer',
        'EnsembleMetaTransformer',
        'EnrichedPatternsTransformer',
        'CrossDomainEmbeddingTransformer',
        'UniversalHybridTransformer',
        'UniversalStructuralPatternTransformer',
        'UniversalThemesTransformer'
    ]
    
    OUTCOME_AWARE_TRANSFORMERS = [
        'AlphaTransformer',
        'GoldenNarratioTransformer'
    ]
    
    def __init__(self):
        """Initialize selector."""
        self.selection_log = []
    
    def select_transformers(
        self,
        domain_name: str,
        pi_value: float,
        domain_type: Optional[str] = None,
        data_sample: Optional[List[str]] = None,
        include_renovation: bool = True,
        include_expensive: bool = True
    ) -> List[str]:
        """
        Select appropriate transformers for a domain.
        
        Parameters
        ----------
        domain_name : str
            Domain identifier
        pi_value : float
            Domain narrativity (0-1)
        domain_type : str, optional
            Domain category: 'sports', 'entertainment', 'business', 'nominative', etc.
        data_sample : list of str, optional
            Sample narratives for adaptive selection
        include_renovation : bool
            Include all renovation transformers (default: True)
        include_expensive : bool
            Include computationally expensive transformers (default: True)
        
        Returns
        -------
        transformer_names : list of str
            Ordered list of transformer class names to apply
        """
        selected = []
        selection_reasoning = []
        
        # 1. ALWAYS: Core transformers
        selected.extend(self.CORE_TRANSFORMERS)
        selection_reasoning.append(f"Added {len(self.CORE_TRANSFORMERS)} core transformers (universal)")
        selected.extend(self.NEXT_GEN_CORE_TRANSFORMERS)
        selection_reasoning.append(
            f"Added {len(self.NEXT_GEN_CORE_TRANSFORMERS)} next-gen core transformers"
        )
        
        # 2. ALWAYS: Renovation transformers (unless explicitly disabled)
        if include_renovation:
            renovation_count = 0
            for category, transformers in self.RENOVATION_TRANSFORMERS.items():
                selected.extend(transformers)
                renovation_count += len(transformers)
            selection_reasoning.append(
                f"Added {renovation_count} renovation transformers (Nov 2025 framework)"
            )
        
        # 3. π-dependent selection
        if pi_value < 0.3:
            selected.extend(self.LOW_PI_TRANSFORMERS)
            selection_reasoning.append(
                f"π={pi_value:.2f} (LOW): Added {len(self.LOW_PI_TRANSFORMERS)} "
                "constrained-domain transformers"
            )
        elif pi_value < 0.7:
            selected.extend(self.MEDIUM_PI_TRANSFORMERS)
            selection_reasoning.append(
                f"π={pi_value:.2f} (MEDIUM): Added {len(self.MEDIUM_PI_TRANSFORMERS)} "
                "balanced transformers"
            )
        else:
            selected.extend(self.HIGH_PI_TRANSFORMERS)
            selection_reasoning.append(
                f"π={pi_value:.2f} (HIGH): Added {len(self.HIGH_PI_TRANSFORMERS)} "
                "narrative-intensive transformers"
            )
        if pi_value >= 0.3:
            selected.extend(self.ADVANCED_SEMANTIC_TRANSFORMERS)
            selection_reasoning.append(
                f"π={pi_value:.2f}: Added {len(self.ADVANCED_SEMANTIC_TRANSFORMERS)} advanced semantic transformers"
            )
        
        # 4. Theory-aligned transformers (all domains)
        selected.extend(self.THEORY_ALIGNED)
        selected.extend(self.UNIVERSAL_META_TRANSFORMERS)
        selection_reasoning.append(
            f"Added {len(self.THEORY_ALIGNED) + len(self.UNIVERSAL_META_TRANSFORMERS)} theory/meta transformers"
        )
        
        # 5. Domain-specific transformers
        if domain_type:
            domain_type_lower = domain_type.lower()
            
            if 'sport' in domain_type_lower or domain_name.lower() in [
                'nba', 'nfl', 'tennis', 'golf', 'ufc', 'mlb', 'boxing', 'poker', 'wwe'
            ]:
                selected.extend(self.SPORTS_TRANSFORMERS)
                selection_reasoning.append(
                    f"Domain type=SPORTS: Added {len(self.SPORTS_TRANSFORMERS)} sports transformers"
                )
                # Sports are hierarchical (game > season > career)
                selected.extend(self.MULTI_SCALE)
                selection_reasoning.append("Added multi-scale transformers (hierarchical sports structure)")
            
            elif 'entertainment' in domain_type_lower or domain_name.lower() in [
                'movies', 'imdb', 'oscars', 'music', 'novels', 'wwe'
            ]:
                selected.extend(self.ENTERTAINMENT_TRANSFORMERS)
                selection_reasoning.append(
                    f"Domain type=ENTERTAINMENT: Added {len(self.ENTERTAINMENT_TRANSFORMERS)} "
                    "entertainment transformers"
                )
            
            elif 'business' in domain_type_lower or domain_name.lower() in [
                'startups', 'crypto'
            ]:
                selected.extend(self.BUSINESS_TRANSFORMERS)
                selection_reasoning.append(
                    f"Domain type=BUSINESS: Added {len(self.BUSINESS_TRANSFORMERS)} "
                    "business/market transformers"
                )
            
            elif 'nominative' in domain_type_lower or domain_name.lower() in [
                'housing', 'aviation', 'ships', 'meta_nominative'
            ]:
                # Use FULL nominative suite
                selected.extend(self.NOMINATIVE_SUITE)
                selection_reasoning.append(
                    f"Domain type=NOMINATIVE: Added {len(self.NOMINATIVE_SUITE)} "
                    "nominative transformers (full suite)"
                )
        
        # 6. High π always gets nominative richness
        if pi_value > 0.7:
            if 'NominativeRichnessTransformer' not in selected:
                selected.append('NominativeRichnessTransformer')
                selection_reasoning.append("Added nominative richness (π > 0.7)")
        
        # 7. Multi-scale for complex domains
        if pi_value > 0.5 and include_expensive:
            if not any(t in selected for t in self.MULTI_SCALE):
                selected.extend(self.MULTI_SCALE)
                selection_reasoning.append("Added multi-scale transformers (π > 0.5, complex narratives)")
        
        # 8. Data-adaptive selection (if sample provided)
        if data_sample:
            avg_length = np.mean([len(text) for text in data_sample[:100]])
            
            if avg_length > 1000:
                # Long-form narratives
                if 'TemporalEvolutionTransformer' not in selected:
                    selected.append('TemporalEvolutionTransformer')
                selection_reasoning.append(f"Added temporal evolution (avg length={avg_length:.0f} chars)")
            
            if avg_length < 200:
                # Short-form - focus on compression
                if 'CognitiveFluencyTransformer' not in selected:
                    selected.append('CognitiveFluencyTransformer')
                selection_reasoning.append(f"Added cognitive fluency (short text, avg={avg_length:.0f})")
        
        # 9. Surface outcome-aware transformers (skipped during extraction if unsupported)
        selected.extend(self.OUTCOME_AWARE_TRANSFORMERS)
        selection_reasoning.append("Ensured outcome-aware transformers are listed for downstream tooling")
        
        # Remove duplicates while preserving order
        seen = set()
        selected_unique = []
        for t in selected:
            if t not in seen:
                seen.add(t)
                selected_unique.append(t)
        
        # Log selection
        self.selection_log.append({
            'domain': domain_name,
            'pi': pi_value,
            'domain_type': domain_type,
            'transformer_count': len(selected_unique),
            'transformers': selected_unique,
            'reasoning': selection_reasoning
        })
        
        return selected_unique
    
    @classmethod
    def _configured_transformers(cls) -> set:
        """Return the set of transformer names referenced by selector constants."""
        names: set = set()
        for attr in dir(cls):
            if not attr.isupper():
                continue
            value = getattr(cls, attr)
            if isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, (list, tuple, set)):
                        names.update(v)
            elif isinstance(value, (list, tuple, set)):
                names.update(value)
        return names
    
    def get_selection_summary(self, domain_name: str) -> Dict:
        """Get selection summary for a domain."""
        for entry in self.selection_log:
            if entry['domain'] == domain_name:
                return entry
        return None
    
    def print_selection_summary(self, domain_name: str):
        """Print human-readable selection summary."""
        summary = self.get_selection_summary(domain_name)
        if not summary:
            print(f"No selection recorded for {domain_name}")
            return
        
        print(f"\n{'='*80}")
        print(f"TRANSFORMER SELECTION: {domain_name.upper()}")
        print(f"{'='*80}")
        print(f"π (Narrativity): {summary['pi']:.3f}")
        print(f"Domain Type: {summary['domain_type'] or 'auto-detected'}")
        print(f"Total Transformers: {summary['transformer_count']}")
        print(f"\nSelection Reasoning:")
        for i, reason in enumerate(summary['reasoning'], 1):
            print(f"  {i}. {reason}")
        print(f"\nSelected Transformers ({len(summary['transformers'])}):")
        for i, t in enumerate(summary['transformers'], 1):
            print(f"  {i:2d}. {t}")
        print(f"{'='*80}\n")
    
    def get_fast_subset(self, transformer_names: List[str], max_count: int = 20) -> List[str]:
        """
        Get fastest subset of transformers for production use.
        
        Prioritizes high-value, low-cost transformers.
        """
        # Priority order (fastest + highest value)
        priority = [
            'StatisticalTransformer',
            'NominativeAnalysisTransformer',
            'PhoneticTransformer',
            'InformationTheoryTransformer',
            'SelfPerceptionTransformer',
            'NarrativePotentialTransformer',
            'CouplingStrengthTransformer',
            'NominativeRichnessTransformer',
            'ConflictTensionTransformer',
            'SuspenseMysteryTransformer',
            'FramingTransformer',
            'QuantitativeTransformer',
            'CognitiveFluencyTransformer',
            'EmotionalResonanceTransformer',
            'GravitationalFeaturesTransformer',
            'AwarenessResistanceTransformer',
            'AuthenticityTransformer',
            'TemporalEvolutionTransformer',
            'RelationalValueTransformer',
            'EnsembleNarrativeTransformer'
        ]
        
        # Select based on priority + availability
        fast_subset = []
        for t in priority:
            if t in transformer_names:
                fast_subset.append(t)
                if len(fast_subset) >= max_count:
                    break
        
        # Add any remaining up to max_count
        for t in transformer_names:
            if t not in fast_subset:
                fast_subset.append(t)
                if len(fast_subset) >= max_count:
                    break
        
        return fast_subset

# Validate coverage against registered transformer modules
try:
    from narrative_optimization.src.transformers import _TRANSFORMER_MODULES
    _MISSING_TRANSFORMERS = sorted(
        set(_TRANSFORMER_MODULES.keys()) - TransformerSelector._configured_transformers()
    )
    if _MISSING_TRANSFORMERS:
        import warnings
        warnings.warn(
            "TransformerSelector is missing "
            f"{len(_MISSING_TRANSFORMERS)} transformers: "
            + ", ".join(_MISSING_TRANSFORMERS)
        )
except Exception:
    # Avoid hard failure during early imports; the warning will surface when possible.
    pass


def create_transformer_profile(domain_name: str, pi_value: float, domain_type: str = None) -> Dict:
    """
    Create complete transformer profile for a domain.
    
    Helper function for domain configuration.
    """
    selector = TransformerSelector()
    transformers = selector.select_transformers(domain_name, pi_value, domain_type)
    fast_subset = selector.get_fast_subset(transformers)
    
    return {
        'full_suite': transformers,
        'fast_subset': fast_subset,
        'transformer_count': len(transformers),
        'fast_count': len(fast_subset),
        'expected_features': estimate_feature_count(transformers)
    }


def estimate_feature_count(transformer_names: List[str]) -> int:
    """Estimate total feature count from transformer list."""
    # Rough estimates per transformer
    feature_estimates = {
        'StatisticalTransformer': 150,
        'NominativeAnalysisTransformer': 51,
        'SelfPerceptionTransformer': 21,
        'NarrativePotentialTransformer': 35,
        'LinguisticPatternsTransformer': 36,
        'RelationalValueTransformer': 17,
        'EnsembleNarrativeTransformer': 25,
        'PhoneticTransformer': 91,
        'SocialStatusTransformer': 45,
        'UniversalNominativeTransformer': 116,
        'HierarchicalNominativeTransformer': 41,
        'NominativeInteractionTransformer': 30,
        'PureNominativePredictorTransformer': 20,
        'NominativeRichnessTransformer': 25,
        'EmotionalResonanceTransformer': 34,
        'AuthenticityTransformer': 30,
        'ConflictTensionTransformer': 28,
        'ExpertiseAuthorityTransformer': 32,
        'CulturalContextTransformer': 34,
        'SuspenseMysteryTransformer': 25,
        'VisualMultimodalTransformer': 39,
        'OpticsTransformer': 22,
        'FramingTransformer': 28,
        'TemporalEvolutionTransformer': 40,
        'InformationTheoryTransformer': 35,
        'QuantitativeTransformer': 50,
        'CognitiveFluencyTransformer': 32,
        'MultiScaleTransformer': 28,
        'MultiPerspectiveTransformer': 22,
        'ScaleInteractionTransformer': 20,
        'CouplingStrengthTransformer': 18,
        'NarrativeMassTransformer': 22,
        'GravitationalFeaturesTransformer': 30,
        'AwarenessResistanceTransformer': 26,
        'FundamentalConstraintsTransformer': 28,
        'CompetitiveContextTransformer': 35,
        'TemporalMomentumEnhancedTransformer': 30,
        'MatchupAdvantageTransformer': 25,
        'ReputationPrestigeTransformer': 28,
        'NamespaceEcologyTransformer': 45,
        'AnticipatoryCommunicationTransformer': 38,
        'DiscoverabilityTransformer': 28,
        'OriginStoryTransformer': 30,
        'CommunityNetworkTransformer': 32,
        'ScarcityExclusivityTransformer': 25,
        'MomentumVelocityTransformer': 20,
        'CharacterComplexityTransformer': 35,
        'NarrativeDevicesTransformer': 28,
        'CulturalResonanceTransformer': 25
    }
    
    # Renovation transformers (not in main catalog yet)
    renovation_estimates = {
        'TemporalCompressionTransformer': 60,
        'DurationEffectsTransformer': 45,
        'PacingRhythmTransformer': 50,
        'CrossTemporalIsomorphismTransformer': 40,
        'CrossCulturalArchetypeTransformer': 80,
        'ActantialStructureTransformer': 45,
        'ConceptualMetaphorTransformer': 55,
        'LabovianNarrativeTransformer': 40,
        'DiscourseAnalysisTransformer': 45,
        'NarrativeSemioticsTransformer': 40,
        'CognitiveLoadTransformer': 40,
        'EmbodiedMetaphorTransformer': 35,
        'ScriptDeviationTransformer': 30,
        'AttentionalStructureTransformer': 40,
        'MemorabilityTransformer': 45,
        'RitualStructureTransformer': 35
    }
    
    # Enhanced narrative transformers
    narrative_enhancement_estimates = {
        'DeepArchetypeTransformer': 40,
        'LinguisticResonanceTransformer': 35,
        'NarrativeCompletionPressureTransformer': 45,
        'TemporalNarrativeCyclesTransformer': 40,
        'CulturalZeitgeistTransformer': 35,
        'RitualCeremonyTransformer': 30,
        'MetaNarrativeAwarenessTransformer': 35,
        'GeographicNarrativeTransformer': 35
    }
    
    # Invisible narrative context transformers
    invisible_narrative_estimates = {
        'ScheduleNarrativeTransformer': 40,
        'MilestoneProximityTransformer': 35,
        'CalendarRhythmTransformer': 30,
        'BroadcastNarrativeTransformer': 25,
        'NarrativeInterferenceTransformer': 30,
        'OpponentContextTransformer': 25,
        'SeasonSeriesNarrativeTransformer': 20,
        'EliminationProximityTransformer': 25
    }
    
    all_estimates = {**feature_estimates, **renovation_estimates, **narrative_enhancement_estimates, **invisible_narrative_estimates}
    
    total = 0
    for t_name in transformer_names:
        total += all_estimates.get(t_name, 30)  # Default 30 if unknown
    
    return total


if __name__ == '__main__':
    # Test the selector
    selector = TransformerSelector()
    
    # Test low π domain
    print("\nTest 1: Low π domain (Coin Flips)")
    transformers = selector.select_transformers('coin_flips', pi_value=0.12, domain_type='benchmark')
    selector.print_selection_summary('coin_flips')
    
    # Test medium π sports domain
    print("\nTest 2: Medium π sports domain (NBA)")
    transformers = selector.select_transformers('nba', pi_value=0.49, domain_type='sports')
    selector.print_selection_summary('nba')
    
    # Test high π entertainment domain
    print("\nTest 3: High π entertainment (Movies)")
    transformers = selector.select_transformers('movies', pi_value=0.65, domain_type='entertainment')
    selector.print_selection_summary('movies')
    
    # Test very high π nominative domain
    print("\nTest 4: Very high π nominative (Housing)")
    transformers = selector.select_transformers('housing', pi_value=0.85, domain_type='nominative')
    selector.print_selection_summary('housing')
    
    print("\n" + "="*80)
    print("TRANSFORMER SELECTOR TESTS COMPLETE")
    print("="*80)

