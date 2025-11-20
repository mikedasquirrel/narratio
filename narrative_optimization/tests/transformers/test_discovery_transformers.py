"""
Unit Tests for Discovery Transformers

Tests all 7 discovery-enabled transformers:
1. UniversalStructuralPatternTransformer
2. RelationalTopologyTransformer
3. CrossDomainEmbeddingTransformer
4. TemporalDerivativeTransformer
5. MetaFeatureInteractionTransformer
6. OutcomeConditionedArchetypeTransformer
7. AnomalyUniquityTransformer

Author: Narrative Optimization Framework
Date: November 2025
"""

import pytest
import numpy as np
from sklearn.utils.validation import check_is_fitted

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


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_texts():
    """Sample narrative texts for testing."""
    return [
        "The underdog team fought bravely and won in the end.",
        "A dominant performance from start to finish.",
        "The game was close throughout with multiple lead changes.",
        "An unexpected upset shocked the crowd.",
        "The favorite cruised to an easy victory."
    ]


@pytest.fixture
def sample_genome_features():
    """Sample genome feature matrix."""
    np.random.seed(42)
    return np.random.rand(20, 10)


@pytest.fixture
def sample_matchup_pairs():
    """Sample entity pairs for matchup analysis."""
    np.random.seed(42)
    return [
        {
            'entity_a_features': np.random.rand(10),
            'entity_b_features': np.random.rand(10),
            'entity_a_text': "Entity A narrative",
            'entity_b_text': "Entity B narrative"
        }
        for _ in range(15)
    ]


@pytest.fixture
def sample_temporal_data():
    """Sample temporal sequences."""
    np.random.seed(42)
    return [
        {
            'feature_history': np.random.rand(10, 5),  # 10 time points, 5 features
            'timestamps': list(range(10)),
            'current_features': np.random.rand(5)
        }
        for _ in range(15)
    ]


@pytest.fixture
def sample_embedding_data():
    """Sample data for cross-domain embedding."""
    np.random.seed(42)
    return [
        {
            'genome_features': np.random.rand(10),
            'domain': 'domain_a' if i < 10 else 'domain_b',
            'text': f"Narrative {i}"
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_outcomes():
    """Sample binary outcomes."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=20)


# ============================================================================
# Test UniversalStructuralPatternTransformer
# ============================================================================

class TestUniversalStructuralPatternTransformer:
    """Tests for Universal Structural Pattern Transformer."""
    
    def test_initialization(self):
        """Test transformer initializes correctly."""
        transformer = UniversalStructuralPatternTransformer()
        assert transformer.n_sequence_points == 50
        assert transformer.use_fft == True
        assert transformer.detect_changepoints == True
    
    def test_fit_transform_text_input(self, sample_texts):
        """Test fit_transform with text input."""
        transformer = UniversalStructuralPatternTransformer()
        features = transformer.fit_transform(sample_texts)
        
        assert features.shape == (len(sample_texts), 45)
        assert np.isfinite(features).all()
        check_is_fitted(transformer)
    
    def test_fit_transform_sequence_input(self):
        """Test fit_transform with pre-computed sequences."""
        np.random.seed(42)
        X = [
            {'feature_sequence': np.random.rand(30), 'text': "text"}
            for _ in range(10)
        ]
        
        transformer = UniversalStructuralPatternTransformer()
        features = transformer.fit_transform(X)
        
        assert features.shape == (10, 45)
        assert np.isfinite(features).all()
    
    def test_feature_ranges(self, sample_texts):
        """Test that features are in reasonable ranges."""
        transformer = UniversalStructuralPatternTransformer()
        features = transformer.fit_transform(sample_texts)
        
        # Arc type should be 0-3
        arc_type = features[:, 3]
        assert np.all((arc_type >= 0) & (arc_type <= 3))
        
        # Symmetry score should be 0-1
        symmetry = features[:, -1]
        assert np.all((symmetry >= 0) & (symmetry <= 1))
    
    def test_feature_names(self, sample_texts):
        """Test feature names are generated."""
        transformer = UniversalStructuralPatternTransformer()
        transformer.fit(sample_texts)
        
        names = transformer._get_feature_names()
        assert len(names) == 45
        assert 'arc_overall_slope' in names
        assert 'tension_buildup_rate' in names


# ============================================================================
# Test RelationalTopologyTransformer
# ============================================================================

class TestRelationalTopologyTransformer:
    """Tests for Relational Topology Transformer."""
    
    def test_initialization(self):
        """Test transformer initializes correctly."""
        transformer = RelationalTopologyTransformer()
        assert transformer.normalize_features == True
        assert transformer.compute_pca_topology == True
    
    def test_fit_transform(self, sample_matchup_pairs):
        """Test fit_transform with matchup pairs."""
        transformer = RelationalTopologyTransformer()
        features = transformer.fit_transform(sample_matchup_pairs)
        
        assert features.shape == (len(sample_matchup_pairs), 35)
        assert np.isfinite(features).all()
        check_is_fitted(transformer)
    
    def test_distance_metrics(self, sample_matchup_pairs):
        """Test distance metrics are computed."""
        transformer = RelationalTopologyTransformer()
        features = transformer.fit_transform(sample_matchup_pairs)
        
        # Euclidean distance (column 0)
        euclidean = features[:, 0]
        assert np.all(euclidean >= 0)
        
        # Cosine distance (column 1)
        cosine = features[:, 1]
        assert np.all((cosine >= 0) & (cosine <= 2))
    
    def test_invalid_input(self):
        """Test error handling for invalid input."""
        transformer = RelationalTopologyTransformer()
        
        with pytest.raises(ValueError):
            transformer.fit_transform([{"entity_a_features": np.array([1, 2])}])


# ============================================================================
# Test CrossDomainEmbeddingTransformer
# ============================================================================

class TestCrossDomainEmbeddingTransformer:
    """Tests for Cross-Domain Embedding Transformer."""
    
    def test_initialization(self):
        """Test transformer initializes correctly."""
        transformer = CrossDomainEmbeddingTransformer()
        assert transformer.n_embedding_dims == 10
        assert transformer.n_clusters == 8
        assert transformer.track_domains == True
    
    def test_fit_transform(self, sample_embedding_data, sample_outcomes):
        """Test fit_transform with embedding data."""
        transformer = CrossDomainEmbeddingTransformer(
            n_clusters=3,  # Reduce for small dataset
            embedding_method='pca'  # Use PCA for speed
        )
        features = transformer.fit_transform(sample_embedding_data, y=sample_outcomes)
        
        assert features.shape == (len(sample_embedding_data), 30)
        assert np.isfinite(features).all()
        check_is_fitted(transformer)
    
    def test_cluster_discovery(self, sample_embedding_data, sample_outcomes):
        """Test that clusters are discovered."""
        transformer = CrossDomainEmbeddingTransformer(
            n_clusters=3,
            embedding_method='pca'
        )
        transformer.fit(sample_embedding_data, y=sample_outcomes)
        
        assert transformer.cluster_centroids_ is not None
        assert len(transformer.cluster_centroids_) == 3
    
    def test_domain_tracking(self, sample_embedding_data):
        """Test domain label tracking."""
        transformer = CrossDomainEmbeddingTransformer(
            embedding_method='pca',
            track_domains=True
        )
        transformer.fit(sample_embedding_data)
        
        assert transformer.domain_cluster_map_ is not None
        assert 'domain_a' in transformer.domain_cluster_map_
        assert 'domain_b' in transformer.domain_cluster_map_


# ============================================================================
# Test TemporalDerivativeTransformer
# ============================================================================

class TestTemporalDerivativeTransformer:
    """Tests for Temporal Derivative Transformer."""
    
    def test_initialization(self):
        """Test transformer initializes correctly."""
        transformer = TemporalDerivativeTransformer()
        assert transformer.recent_window == 5
        assert transformer.detect_changepoints == True
    
    def test_fit_transform(self, sample_temporal_data):
        """Test fit_transform with temporal sequences."""
        transformer = TemporalDerivativeTransformer()
        features = transformer.fit_transform(sample_temporal_data)
        
        assert features.shape == (len(sample_temporal_data), 40)
        assert np.isfinite(features).all()
        check_is_fitted(transformer)
    
    def test_velocity_features(self, sample_temporal_data):
        """Test velocity features are computed."""
        transformer = TemporalDerivativeTransformer()
        features = transformer.fit_transform(sample_temporal_data)
        
        # Mean velocity (column 0)
        mean_vel = features[:, 0]
        assert np.isfinite(mean_vel).all()
        
        # Velocity magnitude (column 1)
        vel_mag = features[:, 1]
        assert np.all(vel_mag >= 0)


# ============================================================================
# Test MetaFeatureInteractionTransformer
# ============================================================================

class TestMetaFeatureInteractionTransformer:
    """Tests for Meta-Feature Interaction Transformer."""
    
    def test_initialization(self):
        """Test transformer initializes correctly."""
        transformer = MetaFeatureInteractionTransformer()
        assert transformer.interaction_degree == 2
        assert transformer.include_ratios == True
        assert transformer.include_synergy == True
    
    def test_fit_transform_array(self, sample_genome_features, sample_outcomes):
        """Test fit_transform with array input."""
        transformer = MetaFeatureInteractionTransformer(max_features=50)
        features = transformer.fit_transform(sample_genome_features, y=sample_outcomes)
        
        assert features.shape[0] == len(sample_genome_features)
        assert features.shape[1] <= 50
        assert np.isfinite(features).all()
        check_is_fitted(transformer)
    
    def test_fit_transform_dict(self, sample_genome_features, sample_outcomes):
        """Test fit_transform with dict input."""
        X = [
            {
                'genome_features': sample_genome_features[i],
                'feature_names': [f'feat_{j}' for j in range(10)]
            }
            for i in range(len(sample_genome_features))
        ]
        
        transformer = MetaFeatureInteractionTransformer(max_features=30)
        features = transformer.fit_transform(X, y=sample_outcomes)
        
        assert features.shape[0] == len(X)
        assert np.isfinite(features).all()
    
    def test_interaction_generation(self, sample_genome_features):
        """Test that interactions are generated."""
        transformer = MetaFeatureInteractionTransformer(
            interaction_degree=2,
            max_features=20
        )
        transformer.fit(sample_genome_features)
        
        assert transformer.selected_interactions_ is not None
        assert len(transformer.selected_interactions_) > 0


# ============================================================================
# Test OutcomeConditionedArchetypeTransformer
# ============================================================================

class TestOutcomeConditionedArchetypeTransformer:
    """Tests for Outcome-Conditioned Archetype Transformer."""
    
    def test_initialization(self):
        """Test transformer initializes correctly."""
        transformer = OutcomeConditionedArchetypeTransformer()
        assert transformer.n_winner_clusters == 3
        assert transformer.min_winner_samples == 5
        assert transformer.enable_transfer == True
    
    def test_fit_requires_outcomes(self, sample_embedding_data):
        """Test that fit requires outcomes."""
        transformer = OutcomeConditionedArchetypeTransformer()
        
        with pytest.raises(ValueError, match="REQUIRES outcomes"):
            transformer.fit(sample_embedding_data, y=None)
    
    def test_fit_transform(self, sample_embedding_data, sample_outcomes):
        """Test fit_transform with outcomes."""
        transformer = OutcomeConditionedArchetypeTransformer(
            n_winner_clusters=2,
            use_pca=True
        )
        features = transformer.fit_transform(sample_embedding_data, y=sample_outcomes)
        
        assert features.shape == (len(sample_embedding_data), 25)
        assert np.isfinite(features).all()
        check_is_fitted(transformer)
    
    def test_golden_narratio_discovery(self, sample_embedding_data, sample_outcomes):
        """Test that Ξ (Golden Narratio) is discovered."""
        transformer = OutcomeConditionedArchetypeTransformer(n_winner_clusters=2)
        transformer.fit(sample_embedding_data, y=sample_outcomes)
        
        assert transformer.xi_vector_ is not None
        assert transformer.anti_xi_vector_ is not None
        assert len(transformer.xi_vector_) > 0
    
    def test_alpha_discovery(self, sample_embedding_data, sample_outcomes):
        """Test that α is discovered."""
        transformer = OutcomeConditionedArchetypeTransformer()
        transformer.fit(sample_embedding_data, y=sample_outcomes)
        
        assert transformer.optimal_alpha_ is not None
        assert 0 <= transformer.optimal_alpha_ <= 1


# ============================================================================
# Test AnomalyUniquityTransformer
# ============================================================================

class TestAnomalyUniquityTransformer:
    """Tests for Anomaly Uniquity Transformer."""
    
    def test_initialization(self):
        """Test transformer initializes correctly."""
        transformer = AnomalyUniquityTransformer()
        assert transformer.contamination == 0.1
        assert transformer.n_neighbors == 10
    
    def test_fit_transform(self, sample_genome_features):
        """Test fit_transform with genome features."""
        transformer = AnomalyUniquityTransformer()
        features = transformer.fit_transform(sample_genome_features)
        
        assert features.shape == (len(sample_genome_features), 20)
        assert np.isfinite(features).all()
        check_is_fitted(transformer)
    
    def test_novelty_scores(self, sample_genome_features):
        """Test novelty scores are computed."""
        transformer = AnomalyUniquityTransformer()
        features = transformer.fit_transform(sample_genome_features)
        
        # Overall novelty score (column -3)
        novelty = features[:, -3]
        assert np.all((novelty >= 0) & (novelty <= 2))  # Reasonable range
        
        # Conformity score (column -1)
        conformity = features[:, -1]
        assert np.all((conformity >= 0) & (conformity <= 1))
    
    def test_anomaly_detection(self):
        """Test that anomalies are detected."""
        np.random.seed(42)
        
        # Normal data
        normal = np.random.randn(50, 5)
        
        # Add anomalies
        anomalies = np.random.randn(5, 5) * 5 + 10
        
        X = np.vstack([normal, anomalies])
        
        transformer = AnomalyUniquityTransformer(contamination=0.1)
        features = transformer.fit_transform(X)
        
        # Novelty scores should be higher for anomalies
        normal_novelty = np.mean(features[:50, -3])
        anomaly_novelty = np.mean(features[50:, -3])
        
        assert anomaly_novelty > normal_novelty


# ============================================================================
# Integration Tests
# ============================================================================

class TestDiscoveryTransformersIntegration:
    """Integration tests for discovery transformers working together."""
    
    def test_pipeline_flow(self, sample_texts, sample_outcomes):
        """Test transformers work together in pipeline."""
        np.random.seed(42)
        
        # Extract structural patterns
        structural = UniversalStructuralPatternTransformer()
        structural_feat = structural.fit_transform(sample_texts)
        
        # Use as genome for embedding
        embedding_data = [
            {'genome_features': feat, 'domain': 'test', 'text': text}
            for feat, text in zip(structural_feat, sample_texts)
        ]
        
        embedding = CrossDomainEmbeddingTransformer(
            n_clusters=2,
            embedding_method='pca'
        )
        embedding_feat = embedding.fit_transform(embedding_data, y=sample_outcomes)
        
        # Discover archetypes
        archetype = OutcomeConditionedArchetypeTransformer(n_winner_clusters=2)
        archetype_feat = archetype.fit_transform(embedding_data, y=sample_outcomes)
        
        # All should produce valid features
        assert structural_feat.shape[0] == len(sample_texts)
        assert embedding_feat.shape[0] == len(sample_texts)
        assert archetype_feat.shape[0] == len(sample_texts)
    
    def test_cross_domain_transfer(self):
        """Test cross-domain pattern transfer."""
        np.random.seed(42)
        
        # Create two domains with shared structure
        domain_a = [{'genome_features': np.random.rand(10), 'domain': 'a'} for _ in range(15)]
        domain_b = [{'genome_features': np.random.rand(10), 'domain': 'b'} for _ in range(15)]
        
        X = domain_a + domain_b
        y = np.array([1, 0, 1, 0, 1] * 6)  # Alternating outcomes
        
        # Learn cross-domain embedding
        embedding = CrossDomainEmbeddingTransformer(
            n_clusters=2,
            embedding_method='pca',
            track_domains=True
        )
        features = embedding.fit_transform(X, y=y)
        
        # Should discover domain clusters
        assert embedding.domain_cluster_map_ is not None
        assert 'a' in embedding.domain_cluster_map_
        assert 'b' in embedding.domain_cluster_map_


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

