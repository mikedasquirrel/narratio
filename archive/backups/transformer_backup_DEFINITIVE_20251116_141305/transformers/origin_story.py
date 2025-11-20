"""
Origin Story Transformer

Universal origin narrative analysis for entities with creation stories.
Used by: Startups, brands, products, teams, organizations, movements.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
from typing import List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.shared_models import SharedModelRegistry
from .utils.input_validation import ensure_string_list


class OriginStoryTransformer(BaseEstimator, TransformerMixin):
    """
    Extract origin narrative features - UNIVERSAL.
    
    Features (10 total):
    1. Origin narrative presence
    2. Founder/creator narrative depth
    3. Humble beginnings indicators
    4. Transformation narrative
    5. Vision clarity
    6. Mission statement strength
    7. Purpose alignment
    8. Genesis mythology
    9. Legacy connection
    10. Innovation markers
    
    Works across:
    - Startups (founding story)
    - Brands (brand story)
    - Sports teams (franchise history)
    - Products (creation story)
    - Organizations (origin mission)
    - Movements (genesis narrative)
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize origin story analyzer"""
        self.use_spacy = use_spacy
        self.use_embeddings = use_embeddings
        
        self.nlp = None
        self.embedder = None
        
        # Origin story prototypes
        self.origin_prototypes = {
            'founding': "founding story of how it began and started",
            'humble_beginnings': "humble origins starting from nothing small beginnings",
            'vision': "clear vision of future direction and purpose",
            'transformation': "transformation journey from where we came to where we are",
            'innovation': "innovative creation breaking from traditional approaches"
        }
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        
        if self.use_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        
        if self.use_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
            
            if self.embedder:
                self.prototype_embeddings = {}
                for concept, description in self.origin_prototypes.items():
                    self.prototype_embeddings[concept] = self.embedder.encode([description])[0]
        
        return self
    
    def transform(self, X):
        """Transform to origin story features"""
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_origin_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_origin_features(self, text: str) -> List[float]:
        """Extract all origin story features"""
        features = []
        
        if self.nlp:
            doc = self.nlp(text[:5000])
            n_words = len(doc)
        else:
            doc = None
            n_words = len(text.split()) + 1
        
        # 1. Origin narrative presence
        if doc:
            origin_lemmas = {'found', 'start', 'begin', 'create', 'launch', 'establish', 'born', 'origin'}
            origin_count = sum(1 for token in doc if token.lemma_ in origin_lemmas)
            features.append(min(1.0, origin_count / n_words * 5))
        else:
            features.append(0.3)
        
        # 2. Founder/creator narrative depth
        if doc:
            founder_lemmas = {'founder', 'creator', 'pioneer', 'architect', 'visionary', 'entrepreneur'}
            founder_count = sum(1 for token in doc if token.lemma_ in founder_lemmas)
            
            # Depth = mentions + context
            founder_sentences = sum(1 for sent in doc.sents if any(t.lemma_ in founder_lemmas for t in sent))
            depth = (founder_count + founder_sentences) / n_words * 5
            features.append(min(1.0, depth))
        else:
            features.append(0.3)
        
        # 3. Humble beginnings
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            humble_emb = self.prototype_embeddings['humble_beginnings']
            
            sim = np.dot(text_emb, humble_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(humble_emb) + 1e-9)
            features.append(float(sim))
        else:
            if doc:
                humble_lemmas = {'humble', 'small', 'garage', 'basement', 'modest', 'grassroots'}
                humble_count = sum(1 for token in doc if token.lemma_ in humble_lemmas)
                features.append(min(1.0, humble_count / n_words * 10))
            else:
                features.append(0.2)
        
        # 4. Transformation narrative
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            transform_emb = self.prototype_embeddings['transformation']
            
            sim = np.dot(text_emb, transform_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(transform_emb) + 1e-9)
            features.append(float(sim))
        else:
            if doc:
                transform_lemmas = {'transform', 'evolve', 'grow', 'become', 'journey', 'develop'}
                transform_count = sum(1 for token in doc if token.lemma_ in transform_lemmas)
                features.append(min(1.0, transform_count / n_words * 5))
            else:
                features.append(0.3)
        
        # 5. Vision clarity
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            vision_emb = self.prototype_embeddings['vision']
            
            sim = np.dot(text_emb, vision_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(vision_emb) + 1e-9)
            features.append(float(sim))
        else:
            if doc:
                vision_lemmas = {'vision', 'future', 'imagine', 'dream', 'aspire', 'mission'}
                vision_count = sum(1 for token in doc if token.lemma_ in vision_lemmas)
                features.append(min(1.0, vision_count / n_words * 5))
            else:
                features.append(0.4)
        
        # 6. Mission statement strength
        if doc:
            mission_lemmas = {'mission', 'purpose', 'goal', 'aim', 'objective', 'calling'}
            mission_count = sum(1 for token in doc if token.lemma_ in mission_lemmas)
            features.append(min(1.0, mission_count / n_words * 5))
        else:
            features.append(0.3)
        
        # 7. Purpose alignment
        if doc:
            purpose_lemmas = {'why', 'because', 'purpose', 'reason', 'believe', 'value'}
            purpose_count = sum(1 for token in doc if token.lemma_ in purpose_lemmas)
            features.append(min(1.0, purpose_count / n_words * 5))
        else:
            features.append(0.4)
        
        # 8. Genesis mythology
        if doc:
            myth_lemmas = {'story', 'tale', 'legend', 'myth', 'saga', 'epic', 'journey'}
            myth_count = sum(1 for token in doc if token.lemma_ in myth_lemmas)
            features.append(min(1.0, myth_count / n_words * 5))
        else:
            features.append(0.2)
        
        # 9. Legacy connection
        if doc:
            legacy_lemmas = {'tradition', 'heritage', 'legacy', 'roots', 'foundation', 'history'}
            legacy_count = sum(1 for token in doc if token.lemma_ in legacy_lemmas)
            features.append(min(1.0, legacy_count / n_words * 5))
        else:
            features.append(0.3)
        
        # 10. Innovation markers
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            innovation_emb = self.prototype_embeddings['innovation']
            
            sim = np.dot(text_emb, innovation_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(innovation_emb) + 1e-9)
            features.append(float(sim))
        else:
            if doc:
                innovation_lemmas = {'innovative', 'revolutionary', 'novel', 'new', 'unique', 'original'}
                innovation_count = sum(1 for token in doc if token.lemma_ in innovation_lemmas)
                features.append(min(1.0, innovation_count / n_words * 5))
            else:
                features.append(0.4)
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'origin_narrative_presence',
            'origin_founder_depth',
            'origin_humble_beginnings',
            'origin_transformation',
            'origin_vision_clarity',
            'origin_mission_strength',
            'origin_purpose_alignment',
            'origin_genesis_mythology',
            'origin_legacy_connection',
            'origin_innovation_markers'
        ])

