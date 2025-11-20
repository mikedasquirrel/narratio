"""
Semantic Similarity Utilities

Helper functions for semantic analysis without hardcoded word lists.
"""

import numpy as np
from typing import List, Union, Dict
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSimilarity:
    """
    Semantic similarity utilities for intelligent feature extraction.
    """
    
    @staticmethod
    def find_semantic_matches(
        texts: List[str],
        anchor_concepts: Dict[str, str],
        embedder,
        threshold=0.5
    ) -> Dict[str, np.ndarray]:
        """
        Find semantic matches to anchor concepts using embeddings.
        
        Instead of hardcoded word lists, use semantic similarity.
        
        Parameters
        ----------
        texts : list of str
            Texts to analyze
        anchor_concepts : dict
            {concept_name: concept_description}
            e.g., {'joy': 'happiness and delight'}
        embedder : EmbeddingManager
            Embedding manager
        threshold : float
            Minimum similarity threshold
            
        Returns
        -------
        matches : dict
            {concept_name: similarity_scores}
        """
        # Encode texts
        text_embeddings = embedder.encode(texts)
        
        # Encode anchor concepts
        anchor_embeddings = {
            concept: embedder.encode([description])[0]
            for concept, description in anchor_concepts.items()
        }
        
        # Compute similarities
        matches = {}
        for concept, anchor_emb in anchor_embeddings.items():
            similarities = cosine_similarity(
                text_embeddings,
                anchor_emb.reshape(1, -1)
            ).flatten()
            
            # Apply threshold
            similarities = np.maximum(0, similarities - threshold) / (1 - threshold)
            matches[concept] = similarities
        
        return matches
    
    @staticmethod
    def detect_categories(
        texts: List[str],
        categories: List[str],
        embedder,
        multi_label=True
    ) -> np.ndarray:
        """
        Detect categories using semantic similarity (zero-shot style).
        
        Parameters
        ----------
        texts : list of str
            Texts to classify
        categories : list of str
            Category names
        embedder : EmbeddingManager
            Embedding manager
        multi_label : bool
            Allow multiple categories per text
            
        Returns
        -------
        scores : ndarray
            Shape (n_texts, n_categories) with scores
        """
        # Encode texts and categories
        text_embeddings = embedder.encode(texts)
        category_embeddings = embedder.encode(categories)
        
        # Compute similarities
        scores = cosine_similarity(text_embeddings, category_embeddings)
        
        if not multi_label:
            # Convert to one-hot
            max_indices = scores.argmax(axis=1)
            one_hot = np.zeros_like(scores)
            one_hot[np.arange(len(scores)), max_indices] = 1
            return one_hot
        
        return scores
    
    @staticmethod
    def compute_semantic_density(
        embeddings: np.ndarray,
        method='variance'
    ) -> np.ndarray:
        """
        Compute how focused vs scattered semantic content is.
        
        Parameters
        ----------
        embeddings : ndarray
            Shape (n_samples, embedding_dim)
        method : str
            'variance' or 'entropy'
            
        Returns
        -------
        density : ndarray
            Per-sample density scores
        """
        if method == 'variance':
            # Low variance = focused, high variance = scattered
            density = 1.0 / (1.0 + embeddings.var(axis=1))
        
        elif method == 'entropy':
            # Normalize to probabilities
            abs_embeddings = np.abs(embeddings)
            probs = abs_embeddings / (abs_embeddings.sum(axis=1, keepdims=True) + 1e-8)
            
            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            
            # Convert to density (lower entropy = more focused)
            max_entropy = np.log(embeddings.shape[1])
            density = 1.0 - (entropy / max_entropy)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return density
    
    @staticmethod
    def detect_semantic_arc(
        sentence_embeddings: np.ndarray,
        anchor_start: np.ndarray,
        anchor_end: np.ndarray
    ) -> Dict[str, float]:
        """
        Detect semantic trajectory from start to end concept.
        
        Parameters
        ----------
        sentence_embeddings : ndarray
            Embeddings for each sentence
        anchor_start : ndarray
            Embedding for starting concept (e.g., "conflict and struggle")
        anchor_end : ndarray
            Embedding for ending concept (e.g., "resolution and peace")
            
        Returns
        -------
        arc_features : dict
            Arc trajectory features
        """
        # Compute similarities to start and end
        sim_to_start = cosine_similarity(sentence_embeddings, anchor_start.reshape(1, -1)).flatten()
        sim_to_end = cosine_similarity(sentence_embeddings, anchor_end.reshape(1, -1)).flatten()
        
        # Arc features
        arc_features = {
            'starts_with_concept': sim_to_start[0] if len(sim_to_start) > 0 else 0.0,
            'ends_with_concept': sim_to_end[-1] if len(sim_to_end) > 0 else 0.0,
            'trajectory': sim_to_end[-1] - sim_to_start[0] if len(sim_to_end) > 0 else 0.0,
            'peak_start_similarity': sim_to_start.max(),
            'peak_end_similarity': sim_to_end.max(),
            'mean_start_similarity': sim_to_start.mean(),
            'mean_end_similarity': sim_to_end.mean()
        }
        
        return arc_features

