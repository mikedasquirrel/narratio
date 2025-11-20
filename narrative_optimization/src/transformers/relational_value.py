"""
Relational Value Transformer

Captures value created through relationships between narrative elements.
Tests whether narrative power comes from complementarity and synergy.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string
from .utils.input_validation import ensure_string_list, ensure_string


class RelationalValueTransformer(NarrativeTransformer):
    """
    Analyzes value created through relationships in narratives.
    
    Tests the hypothesis that narrative elements create value through their
    relationships and complementarity, not just additive presence.
    
    Features extracted:
    - Complementarity scores: how elements complete each other
    - Synergy metrics: evidence that whole > sum of parts
    - Relational density: how interconnected the narrative is
    - Value attribution: which relationships contribute most
    
    Parameters
    ----------
    n_features : int
        Number of TF-IDF features for similarity computation
    complementarity_threshold : float
        Similarity threshold for complementarity (lower = more complementary)
    synergy_window : int
        Window size for detecting synergistic patterns
    """
    
    def __init__(
        self,
        n_features: int = 100,
        complementarity_threshold: float = 0.3,
        synergy_window: int = 3
    ):
        super().__init__(
            narrative_id="relational_value",
            description="Value through relationships: complementarity and synergy"
        )
        
        self.n_features = n_features
        self.complementarity_threshold = complementarity_threshold
        self.synergy_window = synergy_window
        
        self.vectorizer_ = None
        self.reference_vectors_ = None
        self.complementarity_matrix_ = None
    
    def fit(self, X, y=None):
        """
        Learn relational patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        # OPTIMIZATION: For large datasets (>10K), sample for fit to avoid O(n²) explosion
        # This makes it scalable while preserving learned patterns
        if len(X) > 10000:
            print(f" (sampling {min(5000, len(X))} for fit)...", end='', flush=True)
            import random
            sample_indices = random.sample(range(len(X)), min(5000, len(X)))
            X_sample = [X[i] for i in sample_indices]
        else:
            X_sample = X
        
        # Create TF-IDF representation with adaptive min_df
        n_docs = len(X_sample)
        adaptive_min_df = min(2, max(1, n_docs // 10))  # At least 1, at most 10% of docs
        
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.n_features,
            ngram_range=(1, 2),
            min_df=adaptive_min_df,
            max_df=0.95,  # Explicitly set max_df to avoid issues
            stop_words='english'
        )
        
        # Fit on sample, but store full for transform
        X_tfidf_sample = self.vectorizer_.fit_transform(X_sample)
        
        # Store sampled reference vectors for complementarity analysis
        self.reference_vectors_ = X_tfidf_sample
        
        # Compute pairwise similarities (now on sample, not full dataset!)
        similarities = cosine_similarity(X_tfidf_sample)
        
        # Complementarity = low similarity (different but related)
        # Convert similarity to complementarity score
        self.complementarity_matrix_ = 1 - similarities
        np.fill_diagonal(self.complementarity_matrix_, 0)
        
        # Metadata
        self.metadata['n_documents'] = len(X)
        self.metadata['avg_similarity'] = float(np.mean(similarities[np.triu_indices_from(similarities, k=1)]))
        self.metadata['avg_complementarity'] = float(np.mean(self.complementarity_matrix_[np.triu_indices_from(self.complementarity_matrix_, k=1)]))
        self.metadata['relationship_density'] = float(np.sum(self.complementarity_matrix_ > self.complementarity_threshold) / (len(X) * (len(X) - 1)))
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to relational value features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array
            Relational feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        
        X_tfidf = self.vectorizer_.transform(X)
        
        features_list = []
        
        for doc_idx, doc_vec in enumerate(X_tfidf):
            doc_features = []
            
            # 1. Self-complementarity: internal diversity
            # Split document into parts and measure their complementarity
            doc_text = X[doc_idx]
            sentences = doc_text.split('.')
            
            if len(sentences) > 1:
                sent_vecs = self.vectorizer_.transform(sentences)
                sent_sims = cosine_similarity(sent_vecs)
                # Low similarity = high complementarity
                internal_complementarity = 1 - np.mean(sent_sims[np.triu_indices_from(sent_sims, k=1)])
                doc_features.append(internal_complementarity)
            else:
                doc_features.append(0.0)
            
            # 2. Relational density: how many strong relationships
            doc_sim_to_corpus = cosine_similarity(doc_vec, self.reference_vectors_)[0]
            doc_complement_to_corpus = 1 - doc_sim_to_corpus
            
            # Count documents with high complementarity
            n_complementary = np.sum((doc_complement_to_corpus > self.complementarity_threshold) & 
                                    (doc_complement_to_corpus < 0.9))  # Not too different
            relational_density = n_complementary / len(self.reference_vectors_.toarray())
            doc_features.append(relational_density)
            
            # 3. Synergy score: non-linear interactions
            # Measured as variance in tf-idf weights (high variance = some terms dominate = synergy)
            doc_array = doc_vec.toarray().flatten()
            nonzero = doc_array[doc_array > 0]
            
            if len(nonzero) > 1:
                # Gini coefficient as measure of inequality (synergy)
                sorted_weights = np.sort(nonzero)
                n = len(sorted_weights)
                cumsum = np.cumsum(sorted_weights)
                gini = (2 * np.sum((np.arange(n) + 1) * sorted_weights)) / (n * cumsum[-1]) - (n + 1) / n
                doc_features.append(gini)
            else:
                doc_features.append(0.0)
            
            # 4. Complementarity potential: distance to nearest cluster
            nearest_dists = np.sort(doc_complement_to_corpus)[:5]
            complementarity_potential = np.mean(nearest_dists)
            doc_features.append(complementarity_potential)
            
            # 5. Value attribution: how much value from relationships vs individual terms
            # Sum of weights vs. their interactions
            individual_value = np.sum(doc_array)
            
            # Interaction value: product of co-occurring term weights
            interaction_value = 0
            nonzero_indices = np.where(doc_array > 0)[0]
            if len(nonzero_indices) > 1:
                for i in range(len(nonzero_indices)):
                    for j in range(i + 1, min(i + self.synergy_window, len(nonzero_indices))):
                        idx_i, idx_j = nonzero_indices[i], nonzero_indices[j]
                        interaction_value += doc_array[idx_i] * doc_array[idx_j]
            
            value_ratio = interaction_value / (individual_value + 1e-10)
            doc_features.append(value_ratio)
            
            # 6. Relational entropy: diversity of relationships
            if np.sum(doc_complement_to_corpus) > 0:
                rel_probs = doc_complement_to_corpus / np.sum(doc_complement_to_corpus)
                rel_entropy = entropy(rel_probs + 1e-10)
                doc_features.append(rel_entropy)
            else:
                doc_features.append(0.0)
            
            # 7. Complementarity balance: balance between similarity and difference
            sim_scores = doc_sim_to_corpus
            complement_scores = doc_complement_to_corpus
            
            # Optimal is moderate complementarity (not too similar, not too different)
            optimal_complement = 0.5
            balance_score = 1 - np.mean(np.abs(complement_scores - optimal_complement))
            doc_features.append(balance_score)
            
            # 8. Synergistic peaks: number of unusually high tf-idf weights
            if len(nonzero) > 0:
                mean_weight = np.mean(nonzero)
                std_weight = np.std(nonzero) + 1e-10
                n_peaks = np.sum(nonzero > (mean_weight + 2 * std_weight))
                doc_features.append(n_peaks)
            else:
                doc_features.append(0.0)
            
            # 9. Relational coherence: consistency of relationships
            if len(doc_complement_to_corpus) > 1:
                rel_coherence = 1 - np.std(doc_complement_to_corpus)
                doc_features.append(rel_coherence)
            else:
                doc_features.append(0.0)
            
            # === NEW: COMPETITIVE CONTEXT (8 features) ===
            
            # Market saturation proxy (use complementarity inverse)
            saturation = 1.0 - doc_features[0] if len(doc_features) > 0 else 0.5
            doc_features.append(saturation)
            
            # Relative positioning (use relational density as proxy)
            positioning = doc_features[1] if len(doc_features) > 1 else 0
            doc_features.append(positioning)
            
            # Category differentiation (inverse of value ratio)
            differentiation = 1.0 - doc_features[4] if len(doc_features) > 4 else 0.5
            doc_features.append(differentiation)
            
            # Competitive intensity (saturation × low differentiation)
            competitive_intensity = saturation * (1.0 - differentiation)
            doc_features.append(competitive_intensity)
            
            # Niche uniqueness (high differentiation + low saturation)
            niche_uniqueness = differentiation * (1.0 - saturation)
            doc_features.append(niche_uniqueness)
            
            # Competitive advantage (complementarity + differentiation)
            advantage = doc_features[0] * differentiation if len(doc_features) > 0 else 0
            doc_features.append(advantage)
            
            # Substitution threat (high saturation + low differentiation)
            substitution_threat = saturation * (1.0 - differentiation)
            doc_features.append(substitution_threat)
            
            # Market positioning score (composite)
            positioning_score = (niche_uniqueness + advantage) / 2.0
            doc_features.append(positioning_score)
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def get_complementary_documents(self, doc_index: int, n: int = 5):
        """
        Find most complementary documents to a given document.
        
        Parameters
        ----------
        doc_index : int
            Index of reference document
        n : int
            Number of complementary documents to return
        
        Returns
        -------
        indices : array
            Indices of complementary documents
        scores : array
            Complementarity scores
        """
        self._validate_fitted()
        
        if doc_index >= len(self.complementarity_matrix_):
            raise ValueError(f"doc_index {doc_index} out of range")
        
        scores = self.complementarity_matrix_[doc_index]
        
        # Get documents with high complementarity (but not too high = too different)
        valid_mask = (scores > self.complementarity_threshold) & (scores < 0.9)
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_indices]
        
        # Sort by score
        sorted_indices = np.argsort(valid_scores)[::-1][:n]
        
        return valid_indices[sorted_indices], valid_scores[sorted_indices]
    
    def _generate_interpretation(self):
        """Generate human-readable interpretation."""
        avg_sim = self.metadata.get('avg_similarity', 0)
        avg_comp = self.metadata.get('avg_complementarity', 0)
        rel_density = self.metadata.get('relationship_density', 0)
        
        interpretation = (
            f"Relational Value Analysis: Average document similarity: {avg_sim:.3f}, "
            f"average complementarity: {avg_comp:.3f}. "
            f"Relational density (high complementarity): {rel_density:.3f}. "
            "This narrative tests whether value comes from relationships between elements. "
            "Features capture internal complementarity (diversity within document), "
            "relational density (connections to other narratives), synergy (non-additive effects), "
            "complementarity potential, value attribution (relationships vs individuals), "
            "relational entropy, balance, peaks, and coherence. "
            "If this transformer performs well, it validates that narrative power comes "
            "from how elements relate, not just what elements are present."
        )
        
        return interpretation

