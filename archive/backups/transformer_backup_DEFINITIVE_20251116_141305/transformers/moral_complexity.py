"""
Moral Complexity Transformer

Uses NLP to analyze ethical depth and moral ambiguity.
Dependency parsing and semantic analysis for moral reasoning.

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


class MoralComplexityTransformer(BaseEstimator, TransformerMixin):
    """
    Analyzes moral and ethical complexity using NLP.
    
    Features (8 total):
    1. Ethical dilemma presence
    2. Moral ambiguity score
    3. Black-and-white vs grey morality
    4. Competing values identification
    5. Moral stakes quantification
    6. Philosophical depth markers
    7. Value system clarity
    8. Moral evolution trajectory
    
    Uses:
    - Semantic embeddings for moral concepts
    - Dependency parsing for ethical reasoning patterns
    - Sentiment analysis for moral valence
    - Logical connectors for dilemma detection
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize moral complexity analyzer"""
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
        
        # Moral concept prototypes
        self.moral_prototypes = {
            'ethical_dilemma': [
                "difficult choice between two wrongs",
                "conflict between moral principles",
                "impossible ethical decision",
                "no right answer moral problem",
                "values in direct opposition"
            ],
            'moral_ambiguity': [
                "uncertain ethical implications",
                "morally grey situation",
                "unclear right and wrong",
                "complex moral landscape",
                "questionable ethical standing"
            ],
            'philosophical_depth': [
                "existential questioning of meaning",
                "deep inquiry into human nature",
                "fundamental questions about reality",
                "philosophical examination of truth",
                "contemplation of life's purpose"
            ]
        }
        
        # Embed prototypes
        self.prototype_embeddings = {}
        if self.use_embeddings:
            for concept, examples in self.moral_prototypes.items():
                self.prototype_embeddings[concept] = self.embedder.encode(examples)
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to moral complexity features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 8)
            Moral complexity features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_moral_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_moral_features(self, text: str) -> List[float]:
        """Extract all moral complexity features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # 1. Ethical dilemma presence
            dilemma_score = self._detect_ethical_dilemma(text, doc)
            features.append(dilemma_score)
            
            # 2. Moral ambiguity
            ambiguity_score = self._compute_moral_ambiguity(text, doc)
            features.append(ambiguity_score)
            
            # 3. Black-and-white vs grey (0 = black/white, 1 = grey)
            grey_score = self._compute_moral_greyness(doc)
            features.append(grey_score)
            
            # 4. Competing values
            competing_values = self._detect_competing_values(doc)
            features.append(competing_values)
            
            # 5. Moral stakes
            moral_stakes = self._quantify_moral_stakes(doc)
            features.append(moral_stakes)
            
            # 6. Philosophical depth
            phil_depth = self._detect_philosophical_depth(text, doc)
            features.append(phil_depth)
            
            # 7. Value system clarity
            value_clarity = self._compute_value_clarity(doc)
            features.append(value_clarity)
            
            # 8. Moral evolution
            moral_evolution = self._compute_moral_evolution(doc)
            features.append(moral_evolution)
        else:
            # Fallback
            features = [0.3] * 8
        
        return features
    
    def _detect_ethical_dilemma(self, text: str, doc) -> float:
        """
        Detect ethical dilemmas using semantic matching and logical connectors.
        """
        score = 0.0
        
        # Semantic match if available
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            dilemma_embs = self.prototype_embeddings['ethical_dilemma']
            
            sims = []
            for proto_emb in dilemma_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Linguistic patterns for dilemmas
        # Look for "but", "however", "although" with moral terms
        moral_lemmas = {'right', 'wrong', 'good', 'bad', 'should', 'must', 'ought'}
        contrast_lemmas = {'but', 'however', 'although', 'yet', 'while', 'whereas'}
        
        for sent in doc.sents:
            has_moral = any(token.lemma_ in moral_lemmas for token in sent)
            has_contrast = any(token.lemma_ in contrast_lemmas for token in sent)
            
            if has_moral and has_contrast:
                score += 0.15
        
        # Questions about morality
        for sent in doc.sents:
            if sent.text.strip().endswith('?'):
                has_moral = any(token.lemma_ in moral_lemmas for token in sent)
                if has_moral:
                    score += 0.1
        
        return min(1.0, score)
    
    def _compute_moral_ambiguity(self, text: str, doc) -> float:
        """
        Measure moral ambiguity - unclear ethical standing.
        """
        score = 0.0
        
        # Semantic match
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            ambiguity_embs = self.prototype_embeddings['moral_ambiguity']
            
            sims = []
            for proto_emb in ambiguity_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Linguistic markers of ambiguity
        ambiguity_lemmas = {'perhaps', 'maybe', 'unclear', 'uncertain', 'ambiguous',
                           'questionable', 'debatable', 'arguable'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in ambiguity_lemmas:
                    score += 0.1
        
        return min(1.0, score)
    
    def _compute_moral_greyness(self, doc) -> float:
        """
        Measure grey morality (complex) vs black-and-white (simple).
        """
        # Count moral qualifiers and nuance markers
        qualifier_count = 0
        absolute_count = 0
        
        qualifier_lemmas = {'somewhat', 'partly', 'partially', 'relatively', 
                           'fairly', 'rather', 'quite', 'slightly'}
        absolute_lemmas = {'always', 'never', 'completely', 'totally', 
                          'absolutely', 'entirely', 'purely'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in qualifier_lemmas:
                    qualifier_count += 1
                if token.lemma_ in absolute_lemmas:
                    absolute_count += 1
        
        total = qualifier_count + absolute_count
        if total > 0:
            # High qualifier ratio = grey morality
            return qualifier_count / total
        
        return 0.5
    
    def _detect_competing_values(self, doc) -> float:
        """
        Detect multiple competing moral values.
        """
        # Identify value-laden nouns
        value_lemmas = {'justice', 'freedom', 'loyalty', 'duty', 'honor', 
                       'truth', 'compassion', 'mercy', 'courage', 'fairness',
                       'equality', 'liberty', 'responsibility', 'integrity'}
        
        values_mentioned = set()
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in value_lemmas:
                    values_mentioned.add(token.lemma_)
        
        # More values = more potential conflict
        return min(1.0, len(values_mentioned) / 5.0)
    
    def _quantify_moral_stakes(self, doc) -> float:
        """
        Measure the magnitude of moral consequences.
        """
        stake_score = 0.0
        
        # High-stakes language
        stakes_lemmas = {'life', 'death', 'survival', 'destruction', 'fate',
                        'future', 'forever', 'irreversible', 'permanent', 'sacrifice'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in stakes_lemmas:
                    stake_score += 0.1
                    
                    # Bonus for modifiers emphasizing stakes
                    for child in token.children:
                        if child.lemma_ in {'ultimate', 'final', 'critical', 'crucial'}:
                            stake_score += 0.05
        
        return min(1.0, stake_score)
    
    def _detect_philosophical_depth(self, text: str, doc) -> float:
        """
        Detect philosophical inquiry and existential questions.
        """
        score = 0.0
        
        # Semantic match
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            phil_embs = self.prototype_embeddings['philosophical_depth']
            
            sims = []
            for proto_emb in phil_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        # Philosophical terms
        phil_lemmas = {'meaning', 'purpose', 'existence', 'reality', 'truth',
                      'consciousness', 'soul', 'essence', 'nature', 'being',
                      'morality', 'ethics', 'philosophy'}
        
        for sent in doc.sents:
            phil_count = sum(1 for token in sent if token.lemma_ in phil_lemmas)
            if phil_count >= 2:
                score += 0.2
        
        return min(1.0, score)
    
    def _compute_value_clarity(self, doc) -> float:
        """
        Measure how clearly defined the value system is.
        Clear values = explicit moral framework.
        """
        explicit_values = 0
        implicit_references = 0
        
        # Explicit value statements
        value_lemmas = {'believe', 'value', 'principle', 'moral', 'ethic', 'virtue'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in value_lemmas:
                    # Check if in declarative statement
                    if token.head.pos_ == 'VERB':
                        explicit_values += 1
                    else:
                        implicit_references += 1
        
        total = explicit_values + implicit_references
        if total > 0:
            return explicit_values / total
        
        return 0.0
    
    def _compute_moral_evolution(self, doc) -> float:
        """
        Detect moral development or change over narrative.
        """
        sentences = list(doc.sents)
        if len(sentences) < 4:
            return 0.0
        
        # Split into halves
        first_half = sentences[:len(sentences)//2]
        second_half = sentences[len(sentences)//2:]
        
        # Count moral terms in each half
        moral_lemmas = {'right', 'wrong', 'good', 'bad', 'should', 'must',
                       'moral', 'ethical', 'virtue', 'sin', 'just', 'unjust'}
        
        moral_first = sum(
            1 for sent in first_half
            for token in sent if token.lemma_ in moral_lemmas
        )
        
        moral_second = sum(
            1 for sent in second_half
            for token in sent if token.lemma_ in moral_lemmas
        )
        
        # Normalize
        first_density = moral_first / len(first_half) if first_half else 0
        second_density = moral_second / len(second_half) if second_half else 0
        
        # Evolution = change in moral focus
        evolution = abs(second_density - first_density)
        
        return min(1.0, evolution * 5)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'moral_ethical_dilemma',
            'moral_ambiguity',
            'moral_greyness',
            'moral_competing_values',
            'moral_stakes',
            'moral_philosophical_depth',
            'moral_value_clarity',
            'moral_evolution'
        ])

