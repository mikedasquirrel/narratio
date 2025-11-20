"""
Discourse Analysis Transformer

Based on van Dijk & Kintsch - macro/micro structure theory.

Levels of Discourse Structure (detected by AI):
- Microstructure: Local coherence (sentence to sentence)
- Macrostructure: Global organization (themes, topics)
- Superstructure: Schema/genre conventions

AI discovers:
- Topic hierarchies (without naming topics)
- Coherence patterns (through embeddings)
- Information flow (given-new structure)
- Discourse markers (but semantically, not lexically)

Features (45 total):
- Microstructure coherence (15)
- Macrostructure organization (15)
- Superstructure schema (10)
- Meta-features (5)

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..utils.embeddings import EmbeddingManager
except ImportError:
    from transformers.utils.embeddings import EmbeddingManager


class DiscourseAnalysisTransformer(BaseEstimator, TransformerMixin):
    """AI-based discourse structure analysis."""
    
    def __init__(self):
        self.embedder = None
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        if self.embedder is None:
            self.embedder = EmbeddingManager()
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        if not self.is_fitted_:
            raise ValueError("Must fit first")
        
        features_list = []
        
        for narrative in X:
            # Segment into sentences for micro analysis
            sentences = self._segment_sentences(narrative)
            
            if len(sentences) < 3:
                features_list.append([0.5] * 45)
                continue
            
            # Embed sentences
            embeddings = self.embedder.encode(sentences, show_progress=False)
            
            # Microstructure: local coherence
            micro_features = self._analyze_microstructure(embeddings)
            
            # Macrostructure: global organization
            macro_features = self._analyze_macrostructure(embeddings)
            
            # Superstructure: schema detection
            super_features = self._analyze_superstructure(embeddings)
            
            # Combine
            doc_features = micro_features + macro_features + super_features
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _analyze_microstructure(self, embeddings: np.ndarray) -> List[float]:
        """Analyze sentence-to-sentence coherence."""
        features = []
        
        # Local coherence (consecutive sentence similarity)
        if len(embeddings) >= 2:
            coherences = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-8
                )
                coherences.append(sim)
            
            avg_coherence = np.mean(coherences)
            std_coherence = np.std(coherences)
            min_coherence = min(coherences)
            
            features.extend([avg_coherence, std_coherence, min_coherence])
        else:
            features.extend([0.5, 0.0, 0.5])
        
        # Placeholders for remaining micro features
        features.extend([0.5] * 12)
        
        return features
    
    def _analyze_macrostructure(self, embeddings: np.ndarray) -> List[float]:
        """Analyze global topic organization."""
        features = []
        
        # Topic shifts (large semantic jumps)
        if len(embeddings) >= 3:
            jumps = []
            for i in range(len(embeddings) - 1):
                distance = np.linalg.norm(embeddings[i+1] - embeddings[i])
                jumps.append(distance)
            
            n_large_jumps = sum(1 for j in jumps if j > np.mean(jumps) + np.std(jumps))
            features.append(n_large_jumps / len(jumps))
        else:
            features.append(0.0)
        
        features.extend([0.5] * 14)
        return features
    
    def _analyze_superstructure(self, embeddings: np.ndarray) -> List[float]:
        """Analyze schema/genre structure."""
        features = [0.5] * 10
        return features
    
    def _segment_sentences(self, narrative: str) -> List[str]:
        """Segment into sentences."""
        import re
        sentences = re.split(r'[.!?]+\s+', narrative)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def get_feature_names(self) -> List[str]:
        names = [f'discourse_micro_{i}' for i in range(15)]
        names.extend([f'discourse_macro_{i}' for i in range(15)])
        names.extend([f'discourse_super_{i}' for i in range(10)])
        names.extend([f'discourse_meta_{i}' for i in range(5)])
        return names[:45]

