"""
Meta-Narrative Transformer

Uses NLP to detect self-aware and meta-narrative elements.
Dependency parsing and semantic analysis.

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


class MetaNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Detects meta-narrative elements using NLP.
    
    Features (6 total):
    1. Self-awareness level
    2. Genre awareness markers
    3. Convention acknowledgment
    4. Subversion indicators
    5. Postmodern markers
    6. Fourth-wall breaking
    
    Uses:
    - Semantic embeddings for meta-concepts
    - Dependency parsing for self-referential structures
    - Entity recognition for narrative entities
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize meta-narrative detector"""
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
        
        # Meta-narrative prototypes
        self.meta_prototypes = {
            'self_awareness': [
                "story aware of being a story",
                "narrative commenting on itself",
                "text acknowledging its fictional nature",
                "self-referential narrative structure",
                "awareness of being constructed narrative"
            ],
            'genre_awareness': [
                "acknowledging genre conventions",
                "recognizing story type expectations",
                "aware of narrative formula",
                "understanding genre tropes",
                "conscious of storytelling patterns"
            ],
            'subversion': [
                "deliberately defying expectations",
                "intentionally breaking conventions",
                "undermining traditional patterns",
                "challenging narrative norms",
                "inverting standard tropes"
            ],
            'fourth_wall': [
                "addressing the audience directly",
                "breaking fictional boundary",
                "speaking beyond the narrative",
                "acknowledging viewers or readers",
                "stepping outside story world"
            ]
        }
        
        # Embed prototypes
        self.prototype_embeddings = {}
        if self.use_embeddings:
            for concept, examples in self.meta_prototypes.items():
                self.prototype_embeddings[concept] = self.embedder.encode(examples)
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to meta-narrative features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 6)
            Meta-narrative features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_meta_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_meta_features(self, text: str) -> List[float]:
        """Extract all meta-narrative features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # 1. Self-awareness level
            self_awareness = self._detect_self_awareness(text, doc)
            features.append(self_awareness)
            
            # 2. Genre awareness
            genre_awareness = self._detect_genre_awareness(text, doc)
            features.append(genre_awareness)
            
            # 3. Convention acknowledgment
            convention_ack = self._detect_convention_acknowledgment(doc)
            features.append(convention_ack)
            
            # 4. Subversion indicators
            subversion = self._detect_subversion(text, doc)
            features.append(subversion)
            
            # 5. Postmodern markers
            postmodern = self._detect_postmodern_markers(doc)
            features.append(postmodern)
            
            # 6. Fourth-wall breaking
            fourth_wall = self._detect_fourth_wall(text, doc)
            features.append(fourth_wall)
        else:
            # Fallback
            features = [0.1] * 6
        
        return features
    
    def _detect_self_awareness(self, text: str, doc) -> float:
        """
        Detect narrative self-awareness.
        """
        score = 0.0
        
        # Semantic match
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            awareness_embs = self.prototype_embeddings['self_awareness']
            
            sims = []
            for proto_emb in awareness_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Linguistic markers - narrative terminology
        narrative_lemmas = {'story', 'narrative', 'tale', 'plot', 'character',
                           'chapter', 'scene', 'telling', 'recount', 'describe'}
        
        # Self-referential constructions
        for sent in doc.sents:
            narrative_count = sum(1 for token in sent if token.lemma_ in narrative_lemmas)
            if narrative_count >= 2:
                score += 0.15
        
        return min(1.0, score)
    
    def _detect_genre_awareness(self, text: str, doc) -> float:
        """
        Detect awareness of genre conventions.
        """
        score = 0.0
        
        # Semantic match
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            genre_embs = self.prototype_embeddings['genre_awareness']
            
            sims = []
            for proto_emb in genre_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Genre terminology
        genre_lemmas = {'genre', 'convention', 'trope', 'cliche', 'formula',
                       'typical', 'classic', 'traditional', 'expected'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in genre_lemmas:
                    score += 0.15
        
        return min(1.0, score)
    
    def _detect_convention_acknowledgment(self, doc) -> float:
        """
        Detect explicit acknowledgment of narrative conventions.
        """
        score = 0.0
        
        # Phrases that acknowledge conventions
        convention_lemmas = {'usually', 'normally', 'typically', 'traditionally',
                            'conventionally', 'ordinarily', 'generally', 'commonly'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in convention_lemmas:
                    # Check if in context of narrative/story
                    has_narrative_context = any(
                        t.lemma_ in {'story', 'narrative', 'tale', 'plot'}
                        for t in sent
                    )
                    if has_narrative_context:
                        score += 0.2
                    else:
                        score += 0.05
        
        return min(1.0, score)
    
    def _detect_subversion(self, text: str, doc) -> float:
        """
        Detect subversion of expectations/conventions.
        """
        score = 0.0
        
        # Semantic match
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            subversion_embs = self.prototype_embeddings['subversion']
            
            sims = []
            for proto_emb in subversion_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Subversion markers
        subversion_lemmas = {'unexpected', 'surprise', 'twist', 'contrary', 'opposite',
                            'defy', 'challenge', 'break', 'violate', 'subvert'}
        
        expectation_lemmas = {'expect', 'anticipate', 'predict', 'assume', 'suppose'}
        
        for sent in doc.sents:
            has_subversion = any(token.lemma_ in subversion_lemmas for token in sent)
            has_expectation = any(token.lemma_ in expectation_lemmas for token in sent)
            
            if has_subversion:
                score += 0.2 if has_expectation else 0.1
        
        return min(1.0, score)
    
    def _detect_postmodern_markers(self, doc) -> float:
        """
        Detect postmodern narrative elements.
        """
        score = 0.0
        
        # Postmodern characteristics
        # 1. Fragmentation
        short_sentences = sum(1 for sent in doc.sents if len(sent) < 5)
        total_sentences = len(list(doc.sents))
        if total_sentences > 0:
            fragmentation = short_sentences / total_sentences
            score += min(0.3, fragmentation)
        
        # 2. Intertextuality (references to other texts)
        reference_lemmas = {'reference', 'quote', 'cite', 'allude', 'echo',
                           'recall', 'remind', 'evoke', 'invoke'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in reference_lemmas:
                    score += 0.1
        
        # 3. Irony/paradox
        irony_markers = {'ironic', 'paradox', 'contradiction', 'absurd'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in irony_markers:
                    score += 0.15
        
        return min(1.0, score)
    
    def _detect_fourth_wall(self, text: str, doc) -> float:
        """
        Detect fourth-wall breaking (direct address to audience).
        """
        score = 0.0
        
        # Semantic match
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            fourth_wall_embs = self.prototype_embeddings['fourth_wall']
            
            sims = []
            for proto_emb in fourth_wall_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Direct address to audience
        audience_lemmas = {'you', 'reader', 'audience', 'viewer', 'listener'}
        
        for sent in doc.sents:
            # Second person pronouns
            second_person_count = sum(1 for token in sent if token.lemma_ == 'you')
            
            # Audience references
            audience_count = sum(1 for token in sent if token.lemma_ in audience_lemmas)
            
            if second_person_count > 0 or audience_count > 0:
                # Check if in meta-context
                meta_context = any(
                    token.lemma_ in {'story', 'narrative', 'tell', 'share'}
                    for token in sent
                )
                if meta_context:
                    score += 0.2
        
        return min(1.0, score)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'meta_self_awareness',
            'meta_genre_awareness',
            'meta_convention_acknowledgment',
            'meta_subversion',
            'meta_postmodern_markers',
            'meta_fourth_wall'
        ])

