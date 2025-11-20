"""
Integration Tests for Discovery Pipeline

Tests complete pipeline with discovery transformers on real/realistic data:
- Multi-domain data loading
- Base genome extraction
- Discovery transformer application
- Cross-domain pattern detection
- Archetype transfer learning
- End-to-end prediction

Author: Narrative Optimization Framework
Date: November 2025
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from transformers.universal_structural_pattern import UniversalStructuralPatternTransformer
from transformers.relational_topology import RelationalTopologyTransformer
from transformers.cross_domain_embedding import CrossDomainEmbeddingTransformer
from transformers.temporal_derivative import TemporalDerivativeTransformer
from transformers.meta_feature_interaction import MetaFeatureInteractionTransformer
from transformers.outcome_conditioned_archetype import OutcomeConditionedArchetypeTransformer
from transformers.anomaly_uniquity import AnomalyUniquityTransformer

# Base transformers for genome extraction
from transformers.nominative import NominativeAnalysisTransformer
from transformers.linguistic_advanced import LinguisticPatternsTransformer
from transformers.narrative_potential import NarrativePotentialTransformer


# ============================================================================
# Test Data Creation
# ============================================================================

@pytest.fixture
def multi_domain_data():
    """Create multi-domain dataset for testing."""
    np.random.seed(42)
    
    # NBA domain
    nba_texts = [
        f"NBA game {i}: team momentum and scoring run" 
        for i in range(30)
    ]
    nba_outcomes = np.random.randint(0, 2, size=30)
    
    # NFL domain
    nfl_texts = [
        f"NFL matchup {i}: quarterback duel and defensive battle"
        for i in range(30)
    ]
    nfl_outcomes = np.random.randint(0, 2, size=30)
    
    return {
        'nba': {
            'texts': nba_texts,
            'outcomes': nba_outcomes,
            'domain': 'nba'
        },
        'nfl': {
            'texts': nfl_texts,
            'outcomes': nfl_outcomes,
            'domain': 'nfl'
        }
    }


@pytest.fixture
def genome_extractor():
    """Create genome feature extractor pipeline."""
    transformers = [
        NominativeAnalysisTransformer(),
        LinguisticPatternsTransformer(),
        NarrativePotentialTransformer()
    ]
    return transformers


# ============================================================================
# Integration Tests
# ============================================================================

class TestDiscoveryPipelineIntegration:
    """Integration tests for complete discovery pipeline."""
    
    def test_end_to_end_single_domain(self, multi_domain_data, genome_extractor):
        """Test end-to-end pipeline on single domain."""
        data = multi_domain_data['nba']
        
        # Step 1: Extract genome
        genome_features = []
        for transformer in genome_extractor:
            feat = transformer.fit_transform(data['texts'])
            genome_features.append(feat)
        
        genome = np.hstack(genome_features)
        
        # Step 2: Apply discovery transformers
        structural = UniversalStructuralPatternTransformer()
        structural_feat = structural.fit_transform(data['texts'])
        
        interaction = MetaFeatureInteractionTransformer(max_features=20)
        interaction_feat = interaction.fit_transform(genome, y=data['outcomes'])
        
        anomaly = AnomalyUniquityTransformer()
        anomaly_feat = anomaly.fit_transform(genome)
        
        # All should succeed
        assert structural_feat.shape[0] == len(data['texts'])
        assert interaction_feat.shape[0] == len(data['texts'])
        assert anomaly_feat.shape[0] == len(data['texts'])
        
        # All should be finite
        assert np.isfinite(structural_feat).all()
        assert np.isfinite(interaction_feat).all()
        assert np.isfinite(anomaly_feat).all()
    
    def test_end_to_end_multi_domain(self, multi_domain_data, genome_extractor):
        """Test end-to-end pipeline with multiple domains."""
        # Extract genomes for both domains
        nba_data = multi_domain_data['nba']
        nfl_data = multi_domain_data['nfl']
        
        # Extract genomes
        nba_genome = []
        nfl_genome = []
        
        for transformer in genome_extractor:
            nba_feat = transformer.fit_transform(nba_data['texts'])
            nfl_feat = transformer.transform(nfl_data['texts'])
            nba_genome.append(nba_feat)
            nfl_genome.append(nfl_feat)
        
        nba_genome = np.hstack(nba_genome)
        nfl_genome = np.hstack(nfl_genome)
        
        # Create embedding data
        embedding_data = (
            [{'genome_features': nba_genome[i], 'domain': 'nba', 'text': nba_data['texts'][i]}
             for i in range(len(nba_genome))] +
            [{'genome_features': nfl_genome[i], 'domain': 'nfl', 'text': nfl_data['texts'][i]}
             for i in range(len(nfl_genome))]
        )
        
        all_outcomes = np.concatenate([nba_data['outcomes'], nfl_data['outcomes']])
        
        # Test cross-domain embedding
        embedding = CrossDomainEmbeddingTransformer(
            n_clusters=3,
            embedding_method='pca',
            track_domains=True
        )
        embedding_feat = embedding.fit_transform(embedding_data, y=all_outcomes)
        
        assert embedding_feat.shape[0] == len(embedding_data)
        assert embedding_feat.shape[1] == 30
        assert np.isfinite(embedding_feat).all()
        
        # Should track domains
        assert embedding.domain_cluster_map_ is not None
        assert 'nba' in embedding.domain_cluster_map_
        assert 'nfl' in embedding.domain_cluster_map_
    
    def test_archetype_transfer_learning(self, multi_domain_data, genome_extractor):
        """Test archetype discovery and transfer learning."""
        # Extract genomes
        nba_data = multi_domain_data['nba']
        nfl_data = multi_domain_data['nfl']
        
        nba_genome = []
        for transformer in genome_extractor:
            feat = transformer.fit_transform(nba_data['texts'])
            nba_genome.append(feat)
        nba_genome = np.hstack(nba_genome)
        
        # Prepare embedding data
        embedding_data = [
            {'genome_features': nba_genome[i], 'feature_names': [f'f{j}' for j in range(nba_genome.shape[1])], 'domain': 'nba'}
            for i in range(len(nba_genome))
        ]
        
        # Discover archetypes
        archetype = OutcomeConditionedArchetypeTransformer(
            n_winner_clusters=2,
            enable_transfer=True
        )
        features = archetype.fit_transform(embedding_data, y=nba_data['outcomes'])
        
        # Should discover Ξ
        assert archetype.xi_vector_ is not None
        assert archetype.optimal_alpha_ is not None
        
        # Should enable transfer
        assert archetype.archetype_library_ is not None
        assert 'nba' in archetype.archetype_library_
    
    def test_temporal_analysis_pipeline(self, multi_domain_data, genome_extractor):
        """Test temporal derivative transformer in pipeline."""
        data = multi_domain_data['nba']
        
        # Extract genome features over time (simulate)
        all_genomes = []
        for transformer in genome_extractor:
            feat = transformer.fit_transform(data['texts'])
            all_genomes.append(feat)
        
        genome = np.hstack(all_genomes)
        
        # Create temporal sequences (every 5 samples = one sequence)
        n_seq = len(genome) // 10
        temporal_data = [
            {
                'feature_history': genome[i*10:(i+1)*10],
                'timestamps': list(range(10)),
                'current_features': genome[(i+1)*10-1]
            }
            for i in range(n_seq)
        ]
        
        if len(temporal_data) > 0:
            temporal = TemporalDerivativeTransformer()
            features = temporal.fit_transform(temporal_data)
            
            assert features.shape[0] == len(temporal_data)
            assert features.shape[1] == 40
            assert np.isfinite(features).all()
    
    def test_complete_feature_pipeline(self, multi_domain_data, genome_extractor):
        """Test complete feature extraction pipeline."""
        data = multi_domain_data['nba']
        
        # Stage 1: Extract genome
        genome_features = []
        for transformer in genome_extractor:
            feat = transformer.fit_transform(data['texts'])
            genome_features.append(feat)
        
        genome = np.hstack(genome_features)
        
        # Stage 2: Discovery transformers
        # Structural
        structural = UniversalStructuralPatternTransformer()
        structural_feat = structural.fit_transform(data['texts'])
        
        # Interactions
        interaction = MetaFeatureInteractionTransformer(max_features=30)
        interaction_feat = interaction.fit_transform(genome, y=data['outcomes'])
        
        # Anomaly
        anomaly = AnomalyUniquityTransformer()
        anomaly_feat = anomaly.fit_transform(genome)
        
        # Archetypes
        embedding_data = [
            {'genome_features': genome[i], 'domain': 'nba', 'feature_names': [f'f{j}' for j in range(genome.shape[1])]}
            for i in range(len(genome))
        ]
        archetype = OutcomeConditionedArchetypeTransformer(n_winner_clusters=2)
        archetype_feat = archetype.fit_transform(embedding_data, y=data['outcomes'])
        
        # Combine all features
        all_features = np.hstack([
            genome,
            structural_feat,
            interaction_feat,
            anomaly_feat,
            archetype_feat
        ])
        
        # Final feature matrix should be valid
        assert all_features.shape[0] == len(data['texts'])
        assert all_features.shape[1] > 0
        assert np.isfinite(all_features).all()
        
        # Should be usable for prediction
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(model, all_features, data['outcomes'], cv=3)
        
        # Should produce valid scores (may not be good, but should run)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))


class TestCrossDomainPatternDiscovery:
    """Test cross-domain pattern discovery capabilities."""
    
    def test_structural_isomorphism_detection(self, multi_domain_data, genome_extractor):
        """Test detection of structurally similar patterns across domains."""
        # Extract genomes from both domains
        nba_data = multi_domain_data['nba']
        nfl_data = multi_domain_data['nfl']
        
        nba_genome = []
        nfl_genome = []
        
        for transformer in genome_extractor:
            nba_feat = transformer.fit_transform(nba_data['texts'])
            nfl_feat = transformer.transform(nfl_data['texts'])
            nba_genome.append(nba_feat)
            nfl_genome.append(nfl_feat)
        
        nba_genome = np.hstack(nba_genome)
        nfl_genome = np.hstack(nfl_genome)
        
        # Create multi-domain embedding
        embedding_data = (
            [{'genome_features': nba_genome[i], 'domain': 'nba'} for i in range(len(nba_genome))] +
            [{'genome_features': nfl_genome[i], 'domain': 'nfl'} for i in range(len(nfl_genome))]
        )
        
        embedding = CrossDomainEmbeddingTransformer(
            n_clusters=4,
            embedding_method='pca',
            track_domains=True
        )
        features = embedding.fit_transform(embedding_data)
        
        # Extract cluster assignments
        nba_clusters = features[:len(nba_genome), 0]
        nfl_clusters = features[len(nba_genome):, 0]
        
        # Check for shared clusters (structural isomorphism)
        nba_cluster_set = set(nba_clusters)
        nfl_cluster_set = set(nfl_clusters)
        shared_clusters = nba_cluster_set.intersection(nfl_cluster_set)
        
        # Should have at least one shared cluster
        assert len(shared_clusters) > 0
        
        # Domain cluster maps should exist
        assert 'nba' in embedding.domain_cluster_map_
        assert 'nfl' in embedding.domain_cluster_map_
    
    def test_archetype_transfer_across_domains(self, multi_domain_data, genome_extractor):
        """Test archetype transfer from domain A to domain B."""
        nba_data = multi_domain_data['nba']
        nfl_data = multi_domain_data['nfl']
        
        # Extract genomes
        nba_genome = []
        nfl_genome = []
        
        for transformer in genome_extractor:
            nba_feat = transformer.fit_transform(nba_data['texts'])
            nfl_feat = transformer.transform(nfl_data['texts'])
            nba_genome.append(nba_feat)
            nfl_genome.append(nfl_feat)
        
        nba_genome = np.hstack(nba_genome)
        nfl_genome = np.hstack(nfl_genome)
        
        # Learn archetype from NBA
        nba_embedding_data = [
            {'genome_features': nba_genome[i], 'domain': 'nba'}
            for i in range(len(nba_genome))
        ]
        
        archetype = OutcomeConditionedArchetypeTransformer(
            n_winner_clusters=2,
            enable_transfer=True
        )
        archetype.fit(nba_embedding_data, y=nba_data['outcomes'])
        
        # Apply to NFL
        nfl_embedding_data = [
            {'genome_features': nfl_genome[i], 'domain': 'nfl'}
            for i in range(len(nfl_genome))
        ]
        
        nfl_features = archetype.transform(nfl_embedding_data)
        
        # Should compute distance to NBA's Ξ for NFL data
        assert nfl_features.shape[0] == len(nfl_genome)
        assert nfl_features.shape[1] == 25
        assert np.isfinite(nfl_features).all()
        
        # Distance to Ξ should vary (some NFL games closer to NBA winner pattern)
        dist_to_xi = nfl_features[:, 0]
        assert np.std(dist_to_xi) > 0


class TestPerformanceAndScaling:
    """Test performance and scaling of discovery transformers."""
    
    def test_large_dataset_handling(self):
        """Test transformers handle large datasets."""
        np.random.seed(42)
        
        # Large synthetic dataset
        n_samples = 1000
        texts = [f"Narrative {i}" for i in range(n_samples)]
        genome = np.random.rand(n_samples, 50)
        outcomes = np.random.randint(0, 2, size=n_samples)
        
        # Test anomaly transformer (fast)
        anomaly = AnomalyUniquityTransformer()
        features = anomaly.fit_transform(genome)
        
        assert features.shape[0] == n_samples
        assert np.isfinite(features).all()
    
    def test_small_dataset_handling(self):
        """Test transformers handle small datasets gracefully."""
        np.random.seed(42)
        
        # Small dataset
        n_samples = 5
        texts = [f"Narrative {i}" for i in range(n_samples)]
        genome = np.random.rand(n_samples, 10)
        outcomes = np.array([1, 0, 1, 0, 1])
        
        # Most transformers should handle small data
        structural = UniversalStructuralPatternTransformer()
        structural_feat = structural.fit_transform(texts)
        
        anomaly = AnomalyUniquityTransformer()
        anomaly_feat = anomaly.fit_transform(genome)
        
        assert structural_feat.shape[0] == n_samples
        assert anomaly_feat.shape[0] == n_samples


class TestDiscoveredPatternValidation:
    """Test that discovered patterns are valid and interpretable."""
    
    def test_structural_patterns_consistency(self):
        """Test structural patterns are consistent."""
        np.random.seed(42)
        
        # Create narratives with known structures
        # Rising arc
        rising_seq = np.linspace(0, 1, 50)
        # Falling arc
        falling_seq = np.linspace(1, 0, 50)
        # U-shape (hero's journey)
        u_shape = 1 - 4 * (np.linspace(0, 1, 50) - 0.5) ** 2
        
        X = [
            {'feature_sequence': rising_seq, 'text': "rising"},
            {'feature_sequence': falling_seq, 'text': "falling"},
            {'feature_sequence': u_shape, 'text': "hero"}
        ]
        
        transformer = UniversalStructuralPatternTransformer()
        features = transformer.fit_transform(X)
        
        # Rising should have positive slope
        assert features[0, 0] > 0  # Arc slope
        
        # Falling should have negative slope
        assert features[1, 0] < 0
        
        # U-shape should fit hero's journey pattern
        hero_fit = features[2, -5]  # Hero's journey fit
        assert hero_fit > 0.5  # Should correlate with U-shape
    
    def test_relational_topology_patterns(self):
        """Test relational patterns are meaningful."""
        np.random.seed(42)
        
        # Create known relationship patterns
        # Symmetric matchup
        symmetric_a = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        symmetric_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Asymmetric matchup
        asymmetric_a = np.array([0.9, 0.9, 0.9, 0.1, 0.1])
        asymmetric_b = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        
        # Dominant matchup
        dominant_a = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        dominant_b = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        
        X = [
            {'entity_a_features': symmetric_a, 'entity_b_features': symmetric_b},
            {'entity_a_features': asymmetric_a, 'entity_b_features': asymmetric_b},
            {'entity_a_features': dominant_a, 'entity_b_features': dominant_b}
        ]
        
        transformer = RelationalTopologyTransformer()
        features = transformer.fit_transform(X)
        
        # Symmetric should have low distance
        assert features[0, 0] < features[2, 0]  # Distance
        
        # Asymmetric should have high complementarity
        asymmetric_complementarity = features[1, 9]
        symmetric_complementarity = features[0, 9]
        assert asymmetric_complementarity > symmetric_complementarity
        
        # Dominant should have high dominance ratio
        dominant_ratio = features[2, 21]
        assert dominant_ratio > 0.7  # A dominates B in most dimensions


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        transformer = UniversalStructuralPatternTransformer()
        
        # Empty text should not crash
        features = transformer.fit_transform([""])
        assert features.shape == (1, 45)
    
    def test_single_sample_handling(self):
        """Test handling of single samples."""
        np.random.seed(42)
        
        genome = np.random.rand(1, 10)
        outcome = np.array([1])
        
        # Should handle single sample
        anomaly = AnomalyUniquityTransformer()
        features = anomaly.fit_transform(genome)
        
        assert features.shape == (1, 20)
    
    def test_mismatched_dimensions(self):
        """Test error handling for mismatched dimensions."""
        transformer = RelationalTopologyTransformer()
        
        X = [
            {
                'entity_a_features': np.array([1, 2, 3]),
                'entity_b_features': np.array([1, 2])  # Different length!
            }
        ]
        
        with pytest.raises(ValueError):
            transformer.fit_transform(X)
    
    def test_missing_required_fields(self):
        """Test error handling for missing required fields."""
        # Outcome-conditioned without outcomes
        transformer = OutcomeConditionedArchetypeTransformer()
        X = [{'genome_features': np.random.rand(10)}]
        
        with pytest.raises(ValueError, match="REQUIRES outcomes"):
            transformer.fit(X, y=None)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

