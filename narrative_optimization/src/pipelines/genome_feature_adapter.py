"""
Genome Feature Adapter
----------------------

Utility to assemble canonical genome vectors (ж) from the heterogeneous feature
space produced by the base transformer pipeline. The adapter maps transformer
outputs into the four canonical genome components so downstream supervised
transformers can consume a consistent payload.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class GenomeFeatureAdapter:
    """Map transformer-level features into canonical genome components."""

    COMPONENT_TRANSFORMERS: Dict[str, Sequence[str]] = {
        "nominative": {
            "NominativeAnalysisTransformer",
            "NominativeAnalysisV2Transformer",
            "NominativeRichnessTransformer",
            "UniversalNominativeTransformer",
            "HierarchicalNominativeTransformer",
            "NominativeInteractionTransformer",
            "PureNominativePredictorTransformer",
            "PhoneticTransformer",
            "SocialStatusTransformer",
        },
        "archetypal": {
            "GoldenNarratioTransformer",
            "DeepArchetypeTransformer",
            "UniversalStructuralPatternTransformer",
            "UniversalHybridTransformer",
            "UniversalThemesTransformer",
            "MetaNarrativeTransformer",
            "MetaNarrativeAwarenessTransformer",
            "ConceptualMetaphorTransformer",
            "ActantialStructureTransformer",
            "NarrativeSemioticsTransformer",
            "CrossCulturalArchetypeTransformer",
            "LabovianNarrativeTransformer",
        },
        "historial": {
            "TemporalEvolutionTransformer",
            "TemporalMomentumEnhancedTransformer",
            "TemporalNarrativeContextTransformer",
            "TemporalDerivativeTransformer",
            "TemporalCompressionTransformer",
            "TemporalNarrativeCyclesTransformer",
            "SeasonSeriesNarrativeTransformer",
            "ScheduleNarrativeTransformer",
            "MilestoneProximityTransformer",
            "CalendarRhythmTransformer",
            "BroadcastNarrativeTransformer",
            "EliminationProximityTransformer",
            "OpponentContextTransformer",
        },
        "uniquity": {
            "MemorabilityTransformer",
            "CognitiveLoadTransformer",
            "ScriptDeviationTransformer",
            "AttentionalStructureTransformer",
            "EmbodiedMetaphorTransformer",
            "NarrativeInterferenceTransformer",
            "CulturalZeitgeistTransformer",
            "RitualStructureTransformer",
            "RitualCeremonyTransformer",
            "DiscoverabilityTransformer",
            "AnomalyUniquityTransformer",
        },
    }

    def __init__(
        self,
        feature_names: List[str],
        feature_provenance: Dict[str, str],
        domain_name: str,
        default_component_size: int = 1,
    ):
        self.feature_names = feature_names
        self.feature_provenance = feature_provenance
        self.domain_name = domain_name
        self.default_component_size = default_component_size

        self.component_indices: Dict[str, List[int]] = {
            key: [] for key in self.COMPONENT_TRANSFORMERS.keys()
        }
        self.component_sizes: Dict[str, int] = {}

        self._map_components()

    def _map_components(self) -> None:
        """Build index lists for each genome component."""
        for idx, fname in enumerate(self.feature_names):
            transformer = self.feature_provenance.get(fname)
            if not transformer:
                continue

            for component, transformer_names in self.COMPONENT_TRANSFORMERS.items():
                if transformer in transformer_names:
                    self.component_indices[component].append(idx)
                    break

        for component, indices in self.component_indices.items():
            size = len(indices) if indices else self.default_component_size
            self.component_sizes[component] = size

    def build_payload(
        self,
        feature_matrix: np.ndarray,
        narratives: Sequence[str],
        include_text: bool = True,
    ) -> List[Dict]:
        """
        Assemble the genome payload expected by discovery transformers.
        """
        if feature_matrix.shape[0] != len(narratives):
            raise ValueError(
                "Feature matrix and narratives length mismatch: "
                f"{feature_matrix.shape[0]} vs {len(narratives)}"
            )

        payload: List[Dict] = []
        for i in range(feature_matrix.shape[0]):
            genome_vector = self._assemble_genome_vector(feature_matrix[i])
            entry = {
                "genome_features": genome_vector,
                "domain": self.domain_name,
            }
            if include_text:
                entry["text"] = narratives[i]
            payload.append(entry)

        return payload

    def _assemble_genome_vector(self, feature_row: np.ndarray) -> np.ndarray:
        """Concatenate component slices into a single genome vector."""
        segments: List[np.ndarray] = []
        for component in ["nominative", "archetypal", "historial", "uniquity"]:
            indices = self.component_indices.get(component, [])
            if indices:
                segments.append(feature_row[indices])
            else:
                segments.append(np.zeros(self.component_sizes[component], dtype=float))

        if not segments:
            return np.zeros(self.default_component_size, dtype=float)

        return np.concatenate(segments, axis=0)


