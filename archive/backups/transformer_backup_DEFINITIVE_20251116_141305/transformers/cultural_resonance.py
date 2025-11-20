"""
Cultural Resonance Transformer

Uses NLP to analyze cultural appeal and resonance.
Semantic embeddings for cultural concepts.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.input_validation import ensure_string_list

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class CulturalResonanceTransformer(BaseEstimator, TransformerMixin):
    """
    Analyzes cultural resonance and appeal using NLP.
    
    Features (5 total):
    1. Cross-cultural appeal
    2. Cultural specificity vs universality
    3. Zeitgeist alignment
    4. Cultural bridge elements
    5. Taboo/sensitivity detection
    
    Uses:
    - Semantic embeddings for cultural concepts
    - Named entity recognition for cultural markers
    - Temporal language for zeitgeist alignment
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize cultural resonance analyzer"""
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Load models
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except:
                    self.use_spacy = False
        
        if self.use_embeddings:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.use_embeddings = False
        
        # Cultural resonance prototypes
        self.cultural_prototypes = {
            'universal': [
                "human experience transcending culture",
                "emotions everyone can relate to",
                "fundamental truths about humanity",
                "universally understood concepts",
                "cross-cultural human connections"
            ],
            'culturally_specific': [
                "unique cultural traditions",
                "specific cultural context",
                "culturally-bound practices",
                "particular cultural perspective",
                "culture-specific references"
            ],
            'contemporary': [
                "current social issues",
                "modern cultural moment",
                "present-day concerns",
                "contemporary relevance",
                "timely cultural commentary"
            ],
            'cultural_bridge': [
                "connecting different cultures",
                "bridging cultural divides",
                "shared human experience across cultures",
                "universal themes in diverse contexts",
                "cultural translation and understanding"
            ]
        }
        
        # Embed prototypes
        self.prototype_embeddings = {}
        if self.use_embeddings:
            for concept, examples in self.cultural_prototypes.items():
                self.prototype_embeddings[concept] = self.embedder.encode(examples)
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to cultural resonance features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 5)
            Cultural resonance features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_cultural_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_cultural_features(self, text: str) -> List[float]:
        """Extract all cultural resonance features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # 1. Cross-cultural appeal
            cross_cultural = self._compute_cross_cultural_appeal(text, doc)
            features.append(cross_cultural)
            
            # 2. Cultural specificity vs universality
            universality = self._compute_universality(text, doc)
            features.append(universality)
            
            # 3. Zeitgeist alignment
            zeitgeist = self._compute_zeitgeist_alignment(text, doc)
            features.append(zeitgeist)
            
            # 4. Cultural bridge elements
            bridge_elements = self._detect_cultural_bridges(text, doc)
            features.append(bridge_elements)
            
            # 5. Taboo/sensitivity detection
            sensitivity = self._detect_sensitivity(doc)
            features.append(sensitivity)
        else:
            # Fallback
            features = [0.5, 0.5, 0.3, 0.4, 0.2]
        
        return features
    
    def _compute_cross_cultural_appeal(self, text: str, doc) -> float:
        """
        Measure cross-cultural appeal using universal themes.
        """
        score = 0.0
        
        # Semantic match to universal concepts
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            universal_embs = self.prototype_embeddings['universal']
            
            sims = []
            for proto_emb in universal_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Universal human concepts
        universal_lemmas = {'family', 'love', 'fear', 'hope', 'death', 'life',
                           'struggle', 'survival', 'friendship', 'betrayal',
                           'courage', 'wisdom', 'truth', 'justice'}
        
        universal_count = sum(
            1 for sent in doc.sents
            for token in sent if token.lemma_ in universal_lemmas
        )
        
        # Normalize
        score += min(0.5, universal_count / (len(list(doc.sents)) + 1) * 2)
        
        return min(1.0, score)
    
    def _compute_universality(self, text: str, doc) -> float:
        """
        Compute universality (0) vs cultural specificity (1).
        More specific = lower universality.
        """
        specificity_score = 0.0
        
        # Semantic match to culturally specific
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            specific_embs = self.prototype_embeddings['culturally_specific']
            
            sims = []
            for proto_emb in specific_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            specificity_score += float(np.max(sims))
        
        # Cultural markers
        # Geographic/national entities
        geo_entities = sum(
            1 for ent in doc.ents
            if ent.label_ in ['GPE', 'LOC', 'NORP']
        )
        
        # Specific cultural references
        if geo_entities > 3:
            specificity_score += 0.3
        
        # Return universality (inverse of specificity)
        return 1.0 - min(1.0, specificity_score)
    
    def _compute_zeitgeist_alignment(self, text: str, doc) -> float:
        """
        Measure alignment with contemporary moment.
        """
        score = 0.0
        
        # Semantic match to contemporary concepts
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            contemporary_embs = self.prototype_embeddings['contemporary']
            
            sims = []
            for proto_emb in contemporary_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Contemporary markers
        contemporary_lemmas = {'modern', 'contemporary', 'current', 'today',
                              'now', 'recent', 'latest', 'new', 'emerging'}
        
        temporal_markers = sum(
            1 for sent in doc.sents
            for token in sent if token.lemma_ in contemporary_lemmas
        )
        
        score += min(0.3, temporal_markers / (len(list(doc.sents)) + 1) * 3)
        
        # Present tense (contemporary focus)
        present_verbs = sum(
            1 for sent in doc.sents
            for token in sent
            if token.pos_ == 'VERB' and token.tag_ in ['VB', 'VBP', 'VBZ', 'VBG']
        )
        
        total_verbs = sum(
            1 for sent in doc.sents
            for token in sent if token.pos_ == 'VERB'
        )
        
        if total_verbs > 0:
            present_ratio = present_verbs / total_verbs
            score += present_ratio * 0.3
        
        return min(1.0, score)
    
    def _detect_cultural_bridges(self, text: str, doc) -> float:
        """
        Detect elements that bridge cultures.
        """
        score = 0.0
        
        # Semantic match
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            bridge_embs = self.prototype_embeddings['cultural_bridge']
            
            sims = []
            for proto_emb in bridge_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Bridging language
        bridge_lemmas = {'connect', 'bridge', 'unite', 'share', 'common',
                        'universal', 'together', 'relate', 'understand'}
        
        # Multiple cultural references + bridging language
        geo_entities = sum(1 for ent in doc.ents if ent.label_ in ['GPE', 'NORP'])
        bridge_words = sum(
            1 for sent in doc.sents
            for token in sent if token.lemma_ in bridge_lemmas
        )
        
        if geo_entities >= 2 and bridge_words > 0:
            score += 0.3
        
        return min(1.0, score)
    
    def _detect_sensitivity(self, doc) -> float:
        """
        Detect potentially sensitive or taboo content.
        Higher score = more sensitive topics.
        """
        sensitivity_score = 0.0
        
        # Sensitive topics (detected through semantic fields)
        sensitive_lemmas = {
            # Violence
            'violence', 'kill', 'murder', 'attack', 'fight', 'war', 'weapon',
            # Discrimination
            'discrimination', 'racist', 'sexist', 'prejudice', 'bias',
            # Controversial
            'controversial', 'offensive', 'taboo', 'forbidden', 'sensitive',
            # Political
            'political', 'protest', 'revolution', 'conflict'
        }
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in sensitive_lemmas:
                    sensitivity_score += 0.1
        
        return min(1.0, sensitivity_score)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'cultural_cross_cultural_appeal',
            'cultural_universality',
            'cultural_zeitgeist_alignment',
            'cultural_bridge_elements',
            'cultural_sensitivity'
        ])

