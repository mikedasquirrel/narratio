"""
AI-Powered Semantic Analyzer

Replaces rigid keyword matching with wavelike interpretative semantic similarity.
Uses OpenAI embeddings to capture context-dependent, polysemous meanings.

Core Insight:
Concepts like "power", "modern", "optimism" have WAVELIKE properties:
- Context-dependent (collapses to specific meaning)
- Probabilistic (not binary present/absent)
- Continuous (0-1 similarity, not 0/1 match)
- Multi-dimensional (captures nuance)

This is quantum measurement applied to semantics:
- Concept exists in superposition of meanings
- Context measurement collapses to specific interpretation
- Uncertainty quantified (confidence scores)
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
import hashlib
import re


class SemanticAnalyzer:
    """
    AI-powered semantic similarity analyzer using OpenAI embeddings.
    
    Replaces rigid keyword lists with flexible semantic understanding.
    Implements wavelike interpretation with context collapse.
    
    Features:
    - Embedding-based semantic similarity
    - Context-aware interpretation
    - Caching for efficiency (avoid redundant API calls)
    - Batch processing
    - Confidence scoring
    - Hybrid mode (fast keywords + AI refinement)
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = None, hybrid_mode: bool = True):
        """
        Initialize semantic analyzer.
        
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key (will try environment variable if not provided)
        cache_dir : str, optional
            Directory for caching embeddings
        hybrid_mode : bool
            If True, use fast keyword matching + AI refinement
            If False, pure AI (slower but more accurate)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass as parameter.")
        
        # Set up OpenAI
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent.parent / 'cache' / 'embeddings'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache = {}
        self._load_cache()
        
        self.hybrid_mode = hybrid_mode
        self.batch_size = 20  # API batch limit
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        cache_file = self.cache_dir / 'embedding_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
            except:
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embeddings to disk cache."""
        cache_file = self.cache_dir / 'embedding_cache.json'
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.embedding_cache, f)
        except:
            pass  # Fail silently if can't save
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text (with caching).
        
        Parameters
        ----------
        text : str
            Text to embed
        
        Returns
        -------
        embedding : array
            Embedding vector
        """
        cache_key = self._get_cache_key(text)
        
        # Check cache
        if cache_key in self.embedding_cache:
            return np.array(self.embedding_cache[cache_key])
        
        # Get from API
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"  # Efficient model
            )
            embedding = response.data[0].embedding
            
            # Cache it
            self.embedding_cache[cache_key] = embedding
            
            # Periodically save cache (every 100 new embeddings)
            if len(self.embedding_cache) % 100 == 0:
                self._save_cache()
            
            return np.array(embedding)
        except Exception as e:
            print(f"Warning: Embedding API call failed: {e}")
            # Return zero vector as fallback
            return np.zeros(1536)  # Standard embedding size
    
    def semantic_similarity(
        self,
        text: str,
        concept: str,
        context: str = None
    ) -> float:
        """
        Compute semantic similarity between text and concept.
        
        Wavelike interpretation: Returns probability that text expresses concept.
        
        Parameters
        ----------
        text : str
            Text to analyze
        concept : str
            Concept to match (e.g., "power, strength, authority")
        context : str, optional
            Domain context for interpretation
        
        Returns
        -------
        similarity : float
            Similarity score 0-1 (wavelike probability)
        """
        # Add context to concept if provided
        if context:
            concept_with_context = f"{concept} in the context of {context}"
        else:
            concept_with_context = concept
        
        # Get embeddings
        text_emb = self.get_embedding(text[:500])  # Limit length for efficiency
        concept_emb = self.get_embedding(concept_with_context)
        
        # Cosine similarity
        similarity = np.dot(text_emb, concept_emb) / (
            np.linalg.norm(text_emb) * np.linalg.norm(concept_emb) + 1e-10
        )
        
        # Normalize to 0-1 (cosine is -1 to 1)
        similarity_normalized = (similarity + 1) / 2.0
        
        return float(similarity_normalized)
    
    def batch_semantic_similarity(
        self,
        texts: List[str],
        concepts: List[str],
        context: str = None
    ) -> np.ndarray:
        """
        Compute semantic similarity for multiple text-concept pairs (efficient).
        
        Parameters
        ----------
        texts : list of str
            Texts to analyze
        concepts : list of str
            Concepts to match
        context : str, optional
            Domain context
        
        Returns
        -------
        similarities : array, shape (len(texts), len(concepts))
            Similarity matrix
        """
        # Get all embeddings (batch API calls)
        text_embs = [self.get_embedding(t[:500]) for t in texts]
        concept_embs = [self.get_embedding(c if not context else f"{c} in {context}") for c in concepts]
        
        # Compute similarity matrix
        similarities = np.zeros((len(texts), len(concepts)))
        
        for i, text_emb in enumerate(text_embs):
            for j, concept_emb in enumerate(concept_embs):
                similarity = np.dot(text_emb, concept_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(concept_emb) + 1e-10
                )
                similarities[i, j] = (similarity + 1) / 2.0  # Normalize
        
        return similarities
    
    def hybrid_semantic_score(
        self,
        text: str,
        concept: str,
        keyword_list: List[str],
        context: str = None,
        confidence_threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Hybrid approach: Fast keyword matching + AI refinement.
        
        Strategy:
        1. Fast keyword match (O(n))
        2. If high confidence → use keyword score
        3. If ambiguous → refine with AI semantic similarity
        
        Parameters
        ----------
        text : str
            Text to analyze
        concept : str
            Concept description
        keyword_list : list of str
            Fast keywords for initial match
        context : str, optional
            Domain context
        confidence_threshold : float
            Below this, use AI refinement
        
        Returns
        -------
        result : dict
            {'score': float, 'confidence': float, 'method': str}
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Fast keyword matching
        keyword_matches = sum(1 for kw in keyword_list if kw in text_lower)
        keyword_score = keyword_matches / max(1, len(keyword_list))
        
        # Confidence assessment
        # High if many matches or zero matches (clear signal)
        # Low if few matches (ambiguous)
        if keyword_matches == 0:
            keyword_confidence = 0.9  # Confident absence
        elif keyword_matches >= 3:
            keyword_confidence = 0.9  # Confident presence
        else:
            keyword_confidence = 0.3  # Ambiguous
        
        # If confident, use keyword score
        if keyword_confidence > confidence_threshold:
            return {
                'score': keyword_score,
                'confidence': keyword_confidence,
                'method': 'keyword'
            }
        
        # Otherwise, refine with AI
        ai_score = self.semantic_similarity(text, concept, context)
        
        # Blend keyword + AI for final score
        blended_score = 0.4 * keyword_score + 0.6 * ai_score
        
        return {
            'score': blended_score,
            'confidence': 0.7,  # Moderate (used both methods)
            'method': 'hybrid'
        }
    
    def wavelike_concept_score(
        self,
        text: str,
        concept_name: str,
        concept_description: str,
        opposite_description: str = None,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Wavelike interpretation: Concept exists in superposition until measured.
        
        Returns probability distribution over concept presence, not binary.
        
        Parameters
        ----------
        text : str
            Text to analyze
        concept_name : str
            Name of concept (e.g., "power", "modernity")
        concept_description : str
            Description for embedding (e.g., "strength, authority, dominance")
        opposite_description : str, optional
            Opposite concept (e.g., "weakness, submission")
        context : str, optional
            Domain context for interpretation
        
        Returns
        -------
        result : dict
            Wavelike measurement result with uncertainty
        """
        # Get similarity to concept
        concept_similarity = self.semantic_similarity(text, concept_description, context)
        
        # Get similarity to opposite (if provided)
        if opposite_description:
            opposite_similarity = self.semantic_similarity(text, opposite_description, context)
            
            # Normalize as probability distribution
            total = concept_similarity + opposite_similarity
            if total > 0:
                concept_prob = concept_similarity / total
                opposite_prob = opposite_similarity / total
            else:
                concept_prob = 0.5
                opposite_prob = 0.5
            
            # Uncertainty (entropy of distribution)
            uncertainty = -1 * (
                concept_prob * np.log2(concept_prob + 1e-10) +
                opposite_prob * np.log2(opposite_prob + 1e-10)
            )
        else:
            concept_prob = concept_similarity
            uncertainty = 1.0 - concept_prob  # High uncertainty if low similarity
        
        return {
            'concept': concept_name,
            'probability': concept_prob,
            'uncertainty': uncertainty,
            'collapsed_value': concept_prob,  # Measurement collapses wavefunction
            'confidence': 1.0 - uncertainty
        }
    
    def multi_concept_distribution(
        self,
        text: str,
        concepts: Dict[str, str],
        context: str = None
    ) -> Dict[str, float]:
        """
        Get probability distribution over multiple concepts simultaneously.
        
        Like measuring spin in quantum mechanics - get probabilities for all states.
        
        Parameters
        ----------
        text : str
            Text to analyze
        concepts : dict
            {concept_name: concept_description} pairs
        context : str, optional
            Domain context
        
        Returns
        -------
        distribution : dict
            {concept_name: probability} normalized to sum to 1
        """
        # Get similarities for all concepts
        similarities = {}
        for name, description in concepts.items():
            sim = self.semantic_similarity(text, description, context)
            similarities[name] = sim
        
        # Normalize to probability distribution
        total = sum(similarities.values())
        if total > 0:
            distribution = {name: sim / total for name, sim in similarities.items()}
        else:
            # Uniform distribution if no signal
            uniform_prob = 1.0 / len(concepts)
            distribution = {name: uniform_prob for name in concepts}
        
        return distribution


# Global instance for easy access
_semantic_analyzer = None

def get_semantic_analyzer(api_key: str = None) -> SemanticAnalyzer:
    """Get or create global semantic analyzer instance."""
    global _semantic_analyzer
    if _semantic_analyzer is None:
        _semantic_analyzer = SemanticAnalyzer(api_key=api_key)
    return _semantic_analyzer

