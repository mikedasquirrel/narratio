"""
Golden Narratio Transformer (Ξ)

Measures how closely each narrative aligns with the archetypal "golden pattern"
of winners discovered directly from data. The transformer:

1. Vectorizes narratives with TF-IDF (lightweight + deterministic)
2. Projects into a dense latent space via Truncated SVD
3. Learns the Golden Narratio vector Ξ from the centroid of winners
4. Optionally discovers sub-archetypes (MiniBatch KMeans) for richer signals
5. Outputs 10 production-ready features capturing distance, alignment,
   winner membership probability, archetype confidence, and perfection gap

This implementation is text-native so it drops cleanly into existing pipelines
that operate on raw narratives (no precomputed genome required) while still
respecting the original definition: Ξ is empirically discovered from outcomes.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.special import expit
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_transformer import TextNarrativeTransformer


class GoldenNarratioTransformer(TextNarrativeTransformer):
    """
    Discover and quantify distance to Ξ (Golden Narratio) from narrative texts.

    Features (10 total):
        1. distance_to_xi            – Cosine distance to Ξ
        2. similarity_to_xi          – Cosine similarity to Ξ
        3. distance_to_anti_xi       – Distance to anti-Ξ (loser centroid)
        4. relative_alignment        – distance_to_anti_xi - distance_to_xi
        5. winner_membership_score   – Logistic membership probability
        6. primary_archetype_sim     – Max similarity to sub-archetype
        7. secondary_archetype_sim   – Second-best archetype similarity
        8. archetype_confidence      – primary - secondary similarity
        9. perfection_gap            – Fractional gap vs anti-Ξ
        10. xi_projection_score      – Direct projection onto Ξ vector
    """

    FEATURE_NAMES = [
        "distance_to_xi",
        "similarity_to_xi",
        "distance_to_anti_xi",
        "relative_alignment",
        "winner_membership_score",
        "primary_archetype_similarity",
        "secondary_archetype_similarity",
        "archetype_confidence",
        "perfection_gap",
        "xi_projection_score",
    ]

    def __init__(
        self,
        max_features: int = 4000,
        svd_components: int = 64,
        n_archetypes: int = 3,
        min_winner_samples: int = 40,
        winner_threshold: float = 0.5,
        random_state: int = 42,
    ):
        super().__init__(
            narrative_id="golden_narratio",
            description="Measures distance to Ξ (Golden Narratio) discovered from winners",
        )
        self.max_features = max_features
        self.svd_components = svd_components
        self.n_archetypes = n_archetypes
        self.min_winner_samples = min_winner_samples
        self.winner_threshold = winner_threshold
        self.random_state = random_state

        # Learned attributes
        self.vectorizer_: Optional[TfidfVectorizer] = None
        self.svd_: Optional[TruncatedSVD] = None
        self.xi_vector_: Optional[np.ndarray] = None
        self.anti_xi_vector_: Optional[np.ndarray] = None
        self.cluster_model_: Optional[MiniBatchKMeans] = None
        self.cluster_centroids_: Optional[np.ndarray] = None
        self.embedding_dim_: Optional[int] = None

    # --------------------------------------------------------------------- #
    # Fit / Transform
    # --------------------------------------------------------------------- #

    def fit(self, X: List[str], y: Optional[np.ndarray] = None):
        self._validate_input(X)

        if y is None:
            raise ValueError("GoldenNarratioTransformer requires outcome labels (y) to fit.")

        y = np.asarray(y).reshape(-1)
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of samples.")

        # Step 1: Vectorize text
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
        )
        tfidf_matrix = self.vectorizer_.fit_transform(X)

        # Step 2: Reduce dimensionality for stable centroids
        n_components = min(
            self.svd_components,
            max(2, tfidf_matrix.shape[1] - 1),
            max(2, tfidf_matrix.shape[0] - 1),
        )
        self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        embeddings = self.svd_.fit_transform(tfidf_matrix)
        self.embedding_dim_ = embeddings.shape[1]

        # Step 3: Separate winners vs losers
        winner_mask = y >= self.winner_threshold
        losers_mask = ~winner_mask

        if np.sum(winner_mask) < max(5, self.min_winner_samples // 2):
            raise ValueError(
                f"Not enough winners to establish Ξ (found {np.sum(winner_mask)})."
            )

        winners = embeddings[winner_mask]
        losers = embeddings[losers_mask] if np.any(losers_mask) else None

        self.xi_vector_ = winners.mean(axis=0)
        self.anti_xi_vector_ = (
            losers.mean(axis=0) if losers is not None else (-1.0 * self.xi_vector_)
        )

        # Step 4: Discover sub-archetypes if enough winners
        self.cluster_model_ = None
        self.cluster_centroids_ = None
        if winners.shape[0] >= self.min_winner_samples:
            n_clusters = min(self.n_archetypes, winners.shape[0] // self.min_winner_samples)
            if n_clusters >= 1:
                self.cluster_model_ = MiniBatchKMeans(
                    n_clusters=n_clusters or 1,
                    random_state=self.random_state,
                    batch_size=512,
                    n_init=10,
                )
                self.cluster_model_.fit(winners)
                self.cluster_centroids_ = self.cluster_model_.cluster_centers_

        self.metadata = {
            "n_winners": int(winners.shape[0]),
            "n_losers": int(losers.shape[0]) if losers is not None else 0,
            "embedding_dim": int(self.embedding_dim_),
            "n_archetypes": int(self.cluster_centroids_.shape[0])
            if self.cluster_centroids_ is not None
            else 1,
            "feature_names": self.FEATURE_NAMES,
            "n_features": len(self.FEATURE_NAMES),
        }

        self.is_fitted_ = True
        return self

    def transform(self, X: List[str]):
        self._validate_fitted()
        self._validate_input(X)

        tfidf_matrix = self.vectorizer_.transform(X)
        embeddings = self.svd_.transform(tfidf_matrix)

        xi_sim, xi_dist = self._similarity_and_distance(embeddings, self.xi_vector_)
        anti_sim, anti_dist = self._similarity_and_distance(embeddings, self.anti_xi_vector_)

        relative_alignment = anti_dist - xi_dist
        winner_membership = expit(relative_alignment)
        perfection_gap = xi_dist / (xi_dist + anti_dist + 1e-9)
        xi_projection = xi_sim.copy()

        primary_sim, secondary_sim, confidence = self._archetype_alignment(embeddings)

        features = np.column_stack(
            [
                xi_dist,
                xi_sim,
                anti_dist,
                relative_alignment,
                winner_membership,
                primary_sim,
                secondary_sim,
                confidence,
                perfection_gap,
                xi_projection,
            ]
        )

        return features

    # --------------------------------------------------------------------- #
    # Helper Methods
    # --------------------------------------------------------------------- #

    def _similarity_and_distance(
        self, embeddings: np.ndarray, reference: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_norm = reference / (np.linalg.norm(reference) + 1e-9)
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        similarity = np.clip(emb_norm @ ref_norm, -1.0, 1.0)
        distance = 1.0 - similarity
        return similarity, distance

    def _archetype_alignment(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cluster_centroids_ is None:
            xi_sim, _ = self._similarity_and_distance(embeddings, self.xi_vector_)
            zero = np.zeros_like(xi_sim)
            return xi_sim, zero, xi_sim

        centroids_norm = self.cluster_centroids_ / (
            np.linalg.norm(self.cluster_centroids_, axis=1, keepdims=True) + 1e-9
        )
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        sims = emb_norm @ centroids_norm.T

        primary_idx = np.argmax(sims, axis=1)
        primary_sim = sims[np.arange(len(sims)), primary_idx]

        sims[np.arange(len(sims)), primary_idx] = -np.inf
        secondary_sim = np.max(sims, axis=1)
        secondary_sim[~np.isfinite(secondary_sim)] = 0.0
        confidence = primary_sim - secondary_sim

        return primary_sim, secondary_sim, confidence

    def _generate_interpretation(self) -> str:
        if not self.is_fitted_:
            return "Transformer not fitted."

        n_archetypes = (
            self.cluster_centroids_.shape[0] if self.cluster_centroids_ is not None else 1
        )
        return (
            "Golden Narratio discovered from winners.\n"
            f"- Embedding dimension: {self.embedding_dim_}\n"
            f"- Winners analyzed: {self.metadata.get('n_winners', 0)}\n"
            f"- Archetypes learned: {n_archetypes}\n"
            "- Outputs distance and alignment metrics relative to Ξ."
        )


