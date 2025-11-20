"""
Scarcity & Exclusivity Transformer

Universal scarcity/exclusivity analysis for prestige and luxury domains.
Used by: Luxury brands, restaurants (Michelin), exclusive clubs, NFTs, prestige events.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
from typing import List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.shared_models import SharedModelRegistry
from .utils.input_validation import ensure_string_list


class ScarcityExclusivityTransformer(BaseEstimator, TransformerMixin):
    """
    Extract scarcity and exclusivity features - UNIVERSAL for prestige domains.
    
    Features (10 total):
    1. Scarcity language (limited, rare, exclusive)
    2. Access barriers (invitation-only, members-only)
    3. Selectivity markers (chosen, curated, vetted)
    4. Exclusivity positioning
    5. Prestige associations
    6. Price positioning (premium language)
    7. Luxury markers
    8. Elite social proof
    9. Scarcity urgency (limited time)
    10. Insider language (for those who know)
    
    Works across:
    - Luxury brands
    - High-end restaurants
    - Exclusive clubs
    - NFT collections
    - Prestige events
    - Premium products
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize scarcity/exclusivity analyzer"""
        self.use_spacy = use_spacy
        self.use_embeddings = use_embeddings
        
        self.nlp = None
        self.embedder = None
        
        # Scarcity/exclusivity prototypes
        self.prestige_prototypes = {
            'scarcity': "limited availability, rare opportunity, exclusive access",
            'exclusivity': "exclusive membership, select few, invitation only",
            'luxury': "luxury experience, premium quality, exceptional standards",
            'elite': "elite status, distinguished clientele, prestigious position"
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
                for concept, description in self.prestige_prototypes.items():
                    self.prototype_embeddings[concept] = self.embedder.encode([description])[0]
        
        return self
    
    def transform(self, X):
        """Transform to scarcity/exclusivity features"""
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_scarcity_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_scarcity_features(self, text: str) -> List[float]:
        """Extract all scarcity/exclusivity features"""
        features = []
        
        if self.nlp:
            doc = self.nlp(text[:5000])
            n_words = len(doc)
        else:
            doc = None
            n_words = len(text.split()) + 1
        
        # 1. Scarcity language
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            scarcity_emb = self.prototype_embeddings['scarcity']
            
            sim = np.dot(text_emb, scarcity_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(scarcity_emb) + 1e-9)
            features.append(float(sim))
        else:
            if doc:
                scarcity_lemmas = {'limited', 'rare', 'scarce', 'few', 'exclusive', 'select', 'unique'}
                scarcity_count = sum(1 for token in doc if token.lemma_ in scarcity_lemmas)
                features.append(min(1.0, scarcity_count / n_words * 10))
            else:
                features.append(0.2)
        
        # 2. Access barriers
        if doc:
            barrier_lemmas = {'invitation', 'member', 'private', 'closed', 'restricted', 'reserved'}
            barrier_phrases = len([1 for sent in doc.sents if any('only' in sent.text.lower() and t.lemma_ in barrier_lemmas for t in sent)])
            features.append(min(1.0, barrier_phrases / len(list(doc.sents)) * 5))
        else:
            features.append(0.2)
        
        # 3. Selectivity markers
        if doc:
            select_lemmas = {'select', 'chosen', 'curate', 'vet', 'handpick', 'discriminate'}
            select_count = sum(1 for token in doc if token.lemma_ in select_lemmas)
            features.append(min(1.0, select_count / n_words * 10))
        else:
            features.append(0.2)
        
        # 4. Exclusivity positioning
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            exclusivity_emb = self.prototype_embeddings['exclusivity']
            
            sim = np.dot(text_emb, exclusivity_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(exclusivity_emb) + 1e-9)
            features.append(float(sim))
        else:
            if doc:
                exclusive_lemmas = {'exclusive', 'privileged', 'elite', 'special', 'distinguished'}
                exclusive_count = sum(1 for token in doc if token.lemma_ in exclusive_lemmas)
                features.append(min(1.0, exclusive_count / n_words * 10))
            else:
                features.append(0.3)
        
        # 5. Prestige associations
        if doc:
            prestige_lemmas = {'prestige', 'prestigious', 'renowned', 'acclaimed', 'celebrated', 'distinguished'}
            prestige_count = sum(1 for token in doc if token.lemma_ in prestige_lemmas)
            features.append(min(1.0, prestige_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 6. Price positioning
        if doc:
            price_lemmas = {'premium', 'luxury', 'expensive', 'costly', 'priceless', 'value'}
            price_count = sum(1 for token in doc if token.lemma_ in price_lemmas)
            features.append(min(1.0, price_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 7. Luxury markers
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            luxury_emb = self.prototype_embeddings['luxury']
            
            sim = np.dot(text_emb, luxury_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(luxury_emb) + 1e-9)
            features.append(float(sim))
        else:
            if doc:
                luxury_lemmas = {'luxury', 'opulent', 'lavish', 'sumptuous', 'exquisite', 'refined'}
                luxury_count = sum(1 for token in doc if token.lemma_ in luxury_lemmas)
                features.append(min(1.0, luxury_count / n_words * 10))
            else:
                features.append(0.2)
        
        # 8. Elite social proof
        if doc:
            elite_lemmas = {'elite', 'celebrity', 'vip', 'notable', 'prominent', 'distinguished'}
            elite_count = sum(1 for token in doc if token.lemma_ in elite_lemmas)
            features.append(min(1.0, elite_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 9. Scarcity urgency
        if doc:
            urgency_lemmas = {'limited', 'time', 'hurry', 'soon', 'deadline', 'last', 'final'}
            urgency_count = sum(1 for token in doc if token.lemma_ in urgency_lemmas)
            # Check for "limited time" phrases
            limited_time = len([1 for sent in doc.sents if 'limited' in sent.text.lower() and 'time' in sent.text.lower()])
            features.append(min(1.0, (urgency_count + limited_time * 2) / n_words * 10))
        else:
            features.append(0.2)
        
        # 10. Insider language
        if doc:
            insider_lemmas = {'insider', 'know', 'understand', 'appreciate', 'connoisseur', 'aficionado'}
            insider_count = sum(1 for token in doc if token.lemma_ in insider_lemmas)
            features.append(min(1.0, insider_count / n_words * 10))
        else:
            features.append(0.2)
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'scarcity_language',
            'scarcity_access_barriers',
            'scarcity_selectivity',
            'scarcity_exclusivity_positioning',
            'scarcity_prestige_associations',
            'scarcity_price_positioning',
            'scarcity_luxury_markers',
            'scarcity_elite_social_proof',
            'scarcity_urgency',
            'scarcity_insider_language'
        ])

