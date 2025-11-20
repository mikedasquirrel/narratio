"""
Gravitational Features Transformer (φ & ة Effects)

Computes narrative gravity (φ) and nominative gravity (ة) effects.
Tests if φ vs ة tension exists in real data.

Theory:
- φ (phi) = Narrative gravity: attraction based on story similarity
- ة (ta marbuta) = Nominative gravity: attraction based on name similarity
- These forces can pull in opposite directions (tension)

Author: Narrative Integration System
Date: November 2025
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings

from .base import NarrativeTransformer


class GravitationalFeaturesTransformer(NarrativeTransformer):
    """
    Extracts gravitational force features (φ and ة).
    
    Theory:
    φ(i,j) = (μᵢ × μⱼ × similarity(stories)) / distance(stories)²
    ة(i,j) = (μᵢ × μⱼ × similarity(names)) / distance(names)²
    
    These forces create clusters in narrative space. Sometimes they align
    (similar stories have similar names) and sometimes they conflict
    (similar stories have different names).
    
    This transformer:
    1. Clusters texts by story similarity (narrative space)
    2. Clusters texts by name similarity (nominative space)
    3. Computes gravitational pull from archetypal clusters
    4. Measures φ vs ة tension
    
    Features Extracted (20 total):
    
    Narrative Gravity (φ) - 8 features:
    1. Distance to winning story archetype
    2. Distance to losing story archetype
    3. Gravitational pull from winner cluster
    4. Gravitational pull from loser cluster
    5. Net narrative gravity (winner - loser pull)
    6. Narrative cluster membership probability
    7. Story similarity to domain centroid
    8. Narrative outlier score
    
    Nominative Gravity (ة) - 8 features:
    9. Distance to winning name archetype
    10. Distance to losing name archetype
    11. Gravitational pull from winner names
    12. Gravitational pull from loser names
    13. Net nominative gravity
    14. Name cluster membership probability
    15. Name similarity to domain centroid
    16. Nominative outlier score
    
    Tension & Interaction - 4 features:
    17. φ vs ة alignment (do forces agree?)
    18. Gravitational tension magnitude
    19. Dominant force (narrative vs nominative)
    20. Net gravitational pull (φ + ة)
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters (typically: winners, losers, middle)
    use_mass : bool, default=True
        Whether to weight by narrative mass (μ)
    min_cluster_size : int, default=5
        Minimum samples for clustering
    
    Examples
    --------
    >>> transformer = GravitationalFeaturesTransformer()
    >>> features = transformer.fit_transform(narratives, y=outcomes)
    >>> 
    >>> # Check gravitational patterns
    >>> phi_pull = features[:, 4]  # Net narrative gravity
    >>> ta_pull = features[:, 12]  # Net nominative gravity
    >>> tension = features[:, 17]  # Tension between forces
    >>> 
    >>> # Identify instances where forces conflict
    >>> conflicting = np.abs(tension) > 0.5
    >>> print(f"Conflicting forces: {conflicting.sum()} instances")
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        use_mass: bool = True,
        min_cluster_size: int = 5
    ):
        super().__init__(
            narrative_id="gravitational_features",
            description="Computes φ (narrative) and ة (nominative) gravitational effects"
        )
        
        self.n_clusters = n_clusters
        self.use_mass = use_mass
        self.min_cluster_size = min_cluster_size
        
        # Learned during fit
        self.story_clusters_ = None
        self.name_clusters_ = None
        self.winner_story_centroid_ = None
        self.loser_story_centroid_ = None
        self.winner_name_centroid_ = None
        self.loser_name_centroid_ = None
        self.domain_story_centroid_ = None
        self.domain_name_centroid_ = None
        self.mass_values_ = None
        
        # Story features (simple bag-of-words for clustering)
        self.story_vocab_ = None
        self.name_vocab_ = None
    
    def _extract_simple_story_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract simple story features for clustering.
        
        Uses basic bag-of-words on content words (not names).
        
        Parameters
        ----------
        texts : list of str
            Input texts
        
        Returns
        -------
        features : np.ndarray
            Story feature matrix
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if self.story_vocab_ is None:
            # Fit vocabulary during training
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            features = vectorizer.fit_transform(texts).toarray()
            self.story_vocab_ = vectorizer
        else:
            # Transform using learned vocabulary
            features = self.story_vocab_.transform(texts).toarray()
        
        return features
    
    def _extract_name_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract name-based features for nominative clustering.
        
        Focuses on proper nouns and capitalized terms.
        
        Parameters
        ----------
        texts : list of str
            Input texts
        
        Returns
        -------
        features : np.ndarray
            Name feature matrix
        """
        # Extract proper nouns (capitalized words not at sentence start)
        name_texts = []
        
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            proper_nouns = []
            
            for sentence in sentences:
                words = sentence.strip().split()
                for i, word in enumerate(words):
                    clean_word = re.sub(r'[^\w\s-]', '', word)
                    if i > 0 and clean_word and clean_word[0].isupper():
                        proper_nouns.append(clean_word.lower())
            
            # Create "name text" from proper nouns
            name_text = ' '.join(proper_nouns) if proper_nouns else 'none'
            name_texts.append(name_text)
        
        # Vectorize name texts
        from sklearn.feature_extraction.text import CountVectorizer
        
        if self.name_vocab_ is None:
            vectorizer = CountVectorizer(
                max_features=50,
                ngram_range=(1, 1),
                min_df=1
            )
            features = vectorizer.fit_transform(name_texts).toarray()
            self.name_vocab_ = vectorizer
        else:
            features = self.name_vocab_.transform(name_texts).toarray()
        
        return features
    
    def _compute_gravitational_pull(
        self,
        point: np.ndarray,
        centroid: np.ndarray,
        mass_point: float = 1.0,
        mass_centroid: float = 1.0
    ) -> float:
        """
        Compute gravitational pull between point and centroid.
        
        Formula: F = (m₁ × m₂ × similarity) / distance²
        
        Parameters
        ----------
        point : np.ndarray
            Point in feature space
        centroid : np.ndarray
            Cluster centroid
        mass_point : float
            Mass of point
        mass_centroid : float
            Mass of centroid
        
        Returns
        -------
        pull : float
            Gravitational pull strength
        """
        # Compute distance
        distance = np.linalg.norm(point - centroid)
        
        # Avoid division by zero
        if distance < 0.001:
            distance = 0.001
        
        # Compute similarity (inverse of distance, normalized)
        similarity = 1.0 / (1.0 + distance)
        
        # Gravitational force
        pull = (mass_point * mass_centroid * similarity) / (distance ** 2)
        
        return pull
    
    def fit(self, X, y=None):
        """
        Learn archetypal clusters from training data.
        
        Parameters
        ----------
        X : list of str
            Training texts
        y : array, optional
            Outcomes (0/1 for winner/loser)
        
        Returns
        -------
        self
        """
        if len(X) < self.min_cluster_size:
            warnings.warn(
                f"Too few samples ({len(X)}) for clustering. "
                f"Minimum {self.min_cluster_size} required. "
                f"Gravitational features will be zeros."
            )
            self.is_fitted_ = True
            return self
        
        # Extract features
        story_features = self._extract_simple_story_features(X)
        name_features = self._extract_name_features(X)
        
        # Estimate mass values (use all 1.0 if no mass transformer)
        if self.use_mass:
            # Try to extract mass from narratives (simple heuristic)
            self.mass_values_ = np.ones(len(X))
            # Could integrate with NarrativeMassTransformer here
        else:
            self.mass_values_ = np.ones(len(X))
        
        # Cluster by story similarity
        if len(X) >= self.n_clusters:
            story_clusterer = KMeans(n_clusters=min(self.n_clusters, len(X)), random_state=42, n_init=10)
            story_clusterer.fit(story_features)
            self.story_clusters_ = story_clusterer
            self.domain_story_centroid_ = story_features.mean(axis=0)
        else:
            self.story_clusters_ = None
            self.domain_story_centroid_ = story_features.mean(axis=0)
        
        # Cluster by name similarity
        if len(X) >= self.n_clusters:
            name_clusterer = KMeans(n_clusters=min(self.n_clusters, len(X)), random_state=42, n_init=10)
            name_clusterer.fit(name_features)
            self.name_clusters_ = name_clusterer
            self.domain_name_centroid_ = name_features.mean(axis=0)
        else:
            self.name_clusters_ = None
            self.domain_name_centroid_ = name_features.mean(axis=0)
        
        # If outcomes provided, compute winner/loser centroids
        if y is not None and len(y) == len(X):
            y_array = np.array(y)
            
            # Winner centroids
            if y_array.sum() > 0:
                winner_indices = y_array == 1
                self.winner_story_centroid_ = story_features[winner_indices].mean(axis=0)
                self.winner_name_centroid_ = name_features[winner_indices].mean(axis=0)
            else:
                self.winner_story_centroid_ = self.domain_story_centroid_
                self.winner_name_centroid_ = self.domain_name_centroid_
            
            # Loser centroids
            if (y_array == 0).sum() > 0:
                loser_indices = y_array == 0
                self.loser_story_centroid_ = story_features[loser_indices].mean(axis=0)
                self.loser_name_centroid_ = name_features[loser_indices].mean(axis=0)
            else:
                self.loser_story_centroid_ = self.domain_story_centroid_
                self.loser_name_centroid_ = self.domain_name_centroid_
        else:
            # No outcomes - use cluster centroids as proxies
            if self.story_clusters_ is not None:
                self.winner_story_centroid_ = self.story_clusters_.cluster_centers_[0]
                self.loser_story_centroid_ = self.story_clusters_.cluster_centers_[-1]
            else:
                self.winner_story_centroid_ = self.domain_story_centroid_
                self.loser_story_centroid_ = self.domain_story_centroid_
            
            if self.name_clusters_ is not None:
                self.winner_name_centroid_ = self.name_clusters_.cluster_centers_[0]
                self.loser_name_centroid_ = self.name_clusters_.cluster_centers_[-1]
            else:
                self.winner_name_centroid_ = self.domain_name_centroid_
                self.loser_name_centroid_ = self.domain_name_centroid_
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Extract gravitational features.
        
        Parameters
        ----------
        X : list of str
            Texts to transform
        
        Returns
        -------
        features : np.ndarray, shape (n_samples, 20)
            Gravitational features
        """
        self._validate_fitted()
        
        # Handle case where clustering failed
        if self.story_clusters_ is None or self.name_clusters_ is None:
            # Return zeros
            return np.zeros((len(X), 20))
        
        # Extract features
        story_features = self._extract_simple_story_features(X)
        name_features = self._extract_name_features(X)
        
        features = []
        
        for i in range(len(X)):
            story_vec = story_features[i]
            name_vec = name_features[i]
            mass = self.mass_values_[min(i, len(self.mass_values_) - 1)] if self.use_mass else 1.0
            
            # === Narrative Gravity (φ) - 8 features ===
            
            # 1-2. Distance to winner/loser story archetypes
            dist_winner_story = np.linalg.norm(story_vec - self.winner_story_centroid_)
            dist_loser_story = np.linalg.norm(story_vec - self.loser_story_centroid_)
            
            # 3-4. Gravitational pull from winner/loser clusters
            pull_winner_story = self._compute_gravitational_pull(
                story_vec, self.winner_story_centroid_, mass, 2.0
            )
            pull_loser_story = self._compute_gravitational_pull(
                story_vec, self.loser_story_centroid_, mass, 1.0
            )
            
            # 5. Net narrative gravity
            net_phi = pull_winner_story - pull_loser_story
            
            # 6. Narrative cluster membership probability
            story_distances = [
                np.linalg.norm(story_vec - centroid)
                for centroid in self.story_clusters_.cluster_centers_
            ]
            story_probs = np.exp(-np.array(story_distances))
            story_membership = story_probs.max() / (story_probs.sum() + 0.001)
            
            # 7. Story similarity to domain centroid
            story_centroid_sim = 1.0 / (1.0 + np.linalg.norm(story_vec - self.domain_story_centroid_))
            
            # 8. Narrative outlier score (high = far from all clusters)
            story_outlier = min(story_distances) / (np.mean(story_distances) + 0.001)
            
            # === Nominative Gravity (ة) - 8 features ===
            
            # 9-10. Distance to winner/loser name archetypes
            dist_winner_name = np.linalg.norm(name_vec - self.winner_name_centroid_)
            dist_loser_name = np.linalg.norm(name_vec - self.loser_name_centroid_)
            
            # 11-12. Gravitational pull from winner/loser names
            pull_winner_name = self._compute_gravitational_pull(
                name_vec, self.winner_name_centroid_, mass, 2.0
            )
            pull_loser_name = self._compute_gravitational_pull(
                name_vec, self.loser_name_centroid_, mass, 1.0
            )
            
            # 13. Net nominative gravity
            net_ta = pull_winner_name - pull_loser_name
            
            # 14. Name cluster membership probability
            name_distances = [
                np.linalg.norm(name_vec - centroid)
                for centroid in self.name_clusters_.cluster_centers_
            ]
            name_probs = np.exp(-np.array(name_distances))
            name_membership = name_probs.max() / (name_probs.sum() + 0.001)
            
            # 15. Name similarity to domain centroid
            name_centroid_sim = 1.0 / (1.0 + np.linalg.norm(name_vec - self.domain_name_centroid_))
            
            # 16. Nominative outlier score
            name_outlier = min(name_distances) / (np.mean(name_distances) + 0.001)
            
            # === Tension & Interaction - 4 features ===
            
            # 17. φ vs ة alignment (do they agree on winner/loser?)
            # Positive = both pull toward winner, Negative = they disagree
            alignment = np.sign(net_phi) * np.sign(net_ta)
            
            # 18. Gravitational tension magnitude
            tension = np.abs(net_phi - net_ta)
            
            # 19. Dominant force (which is stronger?)
            dominant = np.abs(net_phi) - np.abs(net_ta)  # Positive = φ dominates, negative = ة dominates
            
            # 20. Net gravitational pull
            net_pull = net_phi + net_ta
            
            # Assemble feature vector
            feature_vector = [
                dist_winner_story,
                dist_loser_story,
                pull_winner_story,
                pull_loser_story,
                net_phi,
                story_membership,
                story_centroid_sim,
                story_outlier,
                dist_winner_name,
                dist_loser_name,
                pull_winner_name,
                pull_loser_name,
                net_ta,
                name_membership,
                name_centroid_sim,
                name_outlier,
                alignment,
                tension,
                dominant,
                net_pull
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names."""
        return [
            'dist_winner_story',
            'dist_loser_story',
            'pull_winner_story_phi',
            'pull_loser_story_phi',
            'net_narrative_gravity_phi',
            'story_cluster_membership',
            'story_centroid_similarity',
            'narrative_outlier_score',
            'dist_winner_name',
            'dist_loser_name',
            'pull_winner_name_ta',
            'pull_loser_name_ta',
            'net_nominative_gravity_ta',
            'name_cluster_membership',
            'name_centroid_similarity',
            'nominative_outlier_score',
            'phi_ta_alignment',
            'gravitational_tension',
            'dominant_force',
            'net_gravitational_pull'
        ]
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of learned patterns."""
        return (
            f"Gravitational Features Analysis (φ & ة)\n"
            f"========================================\n\n"
            f"Learned Archetypes:\n"
            f"  • Story clusters: {self.n_clusters if self.story_clusters_ else 0}\n"
            f"  • Name clusters: {self.n_clusters if self.name_clusters_ else 0}\n"
            f"  • Using mass weighting: {self.use_mass}\n\n"
            f"Theory:\n"
            f"  φ (narrative gravity) = attraction based on story similarity\n"
            f"  ة (nominative gravity) = attraction based on name similarity\n\n"
            f"Key Findings:\n"
            f"  • Instances cluster in both narrative and nominative space\n"
            f"  • Forces can align (both pull toward winners)\n"
            f"  • Forces can conflict (story says winner, name says loser)\n"
            f"  • Tension between φ and ة reveals narrative complexity\n\n"
            f"Applications:\n"
            f"  • Identify instances with conflicting signals\n"
            f"  • Measure distance to archetypal winners/losers\n"
            f"  • Detect outliers in narrative space\n"
            f"  • Validate if names match stories\n"
        )

