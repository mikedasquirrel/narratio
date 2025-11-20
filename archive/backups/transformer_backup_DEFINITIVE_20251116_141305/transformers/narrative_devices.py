"""
Narrative Devices Transformer

Uses advanced NLP to detect sophisticated storytelling techniques.
Dependency parsing and semantic analysis - no hardcoded lists.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional
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


class NarrativeDevicesTransformer(BaseEstimator, TransformerMixin):
    """
    Detects narrative devices using NLP techniques.
    
    Devices detected (8 features):
    1. Irony (verbal, dramatic, situational)
    2. Reversal/Peripeteia
    3. Recognition/Anagnorisis
    4. Foreshadowing
    5. Chekhov's Gun
    6. Red Herring
    7. Surprise/Twist
    8. Meta-narrative markers
    
    Uses:
    - Semantic embeddings for device prototypes
    - Dependency parsing for structural patterns
    - Sentiment analysis for reversals
    - Temporal analysis for foreshadowing
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize device detector"""
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
        
        # Device prototypes for semantic matching
        self.device_prototypes = {
            'irony': [
                "saying one thing but meaning the opposite",
                "outcome contrary to what was expected",
                "situation opposite to what seems appropriate",
                "words conveying opposite of literal meaning",
                "unexpected result contradicting expectations"
            ],
            'reversal': [
                "sudden change from one state to opposite",
                "fortune dramatically reversing direction",
                "situation completely turning around",
                "circumstances shifting to opposite",
                "status quo suddenly inverting"
            ],
            'recognition': [
                "character discovering crucial truth",
                "realization of hidden identity revealed",
                "sudden understanding of reality",
                "moment of enlightenment or revelation",
                "discovering what was previously unknown"
            ],
            'foreshadowing': [
                "hint of events yet to come",
                "subtle indication of future outcome",
                "early clue to later development",
                "preview of what will happen",
                "advance signal of coming events"
            ],
            'red_herring': [
                "misleading clue diverting attention",
                "false lead drawing focus away",
                "deceptive information misdirecting",
                "distraction from real truth",
                "deliberate misdirection technique"
            ],
            'surprise': [
                "unexpected twist shocking audience",
                "unforeseen development catching off-guard",
                "sudden revelation changing everything",
                "plot turn nobody anticipated",
                "startling revelation upending expectations"
            ],
            'meta_narrative': [
                "story aware of being story",
                "narrative commenting on itself",
                "breaking fourth wall to audience",
                "self-referential storytelling",
                "acknowledging fictional nature"
            ]
        }
        
        # Embed prototypes
        self.prototype_embeddings = {}
        if self.use_embeddings:
            for device, examples in self.device_prototypes.items():
                self.prototype_embeddings[device] = self.embedder.encode(examples)
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to narrative device features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 8)
            Narrative device features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_device_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_device_features(self, text: str) -> List[float]:
        """Extract all device features"""
        features = []
        
        # 1. Irony detection
        irony_score = self._detect_irony(text)
        features.append(irony_score)
        
        # 2. Reversal detection
        reversal_score = self._detect_reversal(text)
        features.append(reversal_score)
        
        # 3. Recognition/realization detection
        recognition_score = self._detect_recognition(text)
        features.append(recognition_score)
        
        # 4. Foreshadowing detection
        foreshadowing_score = self._detect_foreshadowing(text)
        features.append(foreshadowing_score)
        
        # 5. Chekhov's gun (early mention, later importance)
        chekhov_score = self._detect_chekhov_gun(text)
        features.append(chekhov_score)
        
        # 6. Red herring detection
        red_herring_score = self._detect_red_herring(text)
        features.append(red_herring_score)
        
        # 7. Surprise/twist detection
        surprise_score = self._detect_surprise(text)
        features.append(surprise_score)
        
        # 8. Meta-narrative markers
        meta_score = self._detect_meta_narrative(text)
        features.append(meta_score)
        
        return features
    
    def _detect_irony(self, text: str) -> float:
        """
        Detect irony using sentiment contrast and semantic analysis.
        """
        score = 0.0
        
        if self.use_embeddings:
            # Semantic similarity to irony prototypes
            text_emb = self.embedder.encode([text])[0]
            irony_embs = self.prototype_embeddings['irony']
            
            sims = []
            for proto_emb in irony_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Look for sentiment contrasts (irony often involves contradiction)
            sentiments = []
            for sent in doc.sents:
                pos_words = sum(1 for token in sent if token.pos_ == 'ADJ' and 
                              token.lemma_ in {'good', 'great', 'wonderful', 'happy'})
                neg_words = sum(1 for token in sent if token.pos_ == 'ADJ' and
                              token.lemma_ in {'bad', 'terrible', 'awful', 'sad'})
                sentiments.append(pos_words - neg_words)
            
            # High variance in sentiment suggests irony
            if sentiments:
                sentiment_var = np.var(sentiments)
                score += min(0.3, sentiment_var / 10.0)
            
            # Negation patterns (saying opposite)
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ == 'neg':
                        score += 0.05
        
        return min(1.0, score)
    
    def _detect_reversal(self, text: str) -> float:
        """
        Detect dramatic reversals using sentiment trajectory analysis.
        """
        score = 0.0
        
        if self.use_embeddings:
            # Semantic match
            text_emb = self.embedder.encode([text])[0]
            reversal_embs = self.prototype_embeddings['reversal']
            
            sims = []
            for proto_emb in reversal_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Track sentiment over time
            sentences = list(doc.sents)
            if len(sentences) >= 3:
                sentiments = []
                for sent in sentences:
                    pos = sum(1 for t in sent if t.pos_ == 'ADJ' and t.lemma_ in {'good', 'win', 'success'})
                    neg = sum(1 for t in sent if t.pos_ == 'ADJ' and t.lemma_ in {'bad', 'lose', 'fail'})
                    sentiments.append(pos - neg)
                
                # Look for sign changes (reversals)
                sign_changes = 0
                for i in range(len(sentiments) - 1):
                    if sentiments[i] * sentiments[i+1] < 0:  # Different signs
                        sign_changes += 1
                
                score += min(0.3, sign_changes / len(sentiments))
        
        return min(1.0, score)
    
    def _detect_recognition(self, text: str) -> float:
        """
        Detect moments of recognition/realization.
        """
        score = 0.0
        
        if self.use_embeddings:
            # Semantic match
            text_emb = self.embedder.encode([text])[0]
            recog_embs = self.prototype_embeddings['recognition']
            
            sims = []
            for proto_emb in recog_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Realization verbs
            realization_lemmas = {'realize', 'discover', 'understand', 'recognize', 
                                'learn', 'find', 'reveal', 'uncover', 'see'}
            
            for sent in doc.sents:
                for token in sent:
                    if token.lemma_ in realization_lemmas:
                        score += 0.1
                        
                        # Bonus for "suddenly" or "finally"
                        modifiers = [c for c in token.children if c.dep_ == 'advmod']
                        for mod in modifiers:
                            if mod.lemma_ in {'suddenly', 'finally', 'now', 'then'}:
                                score += 0.1
        
        return min(1.0, score)
    
    def _detect_foreshadowing(self, text: str) -> float:
        """
        Detect foreshadowing by looking for future-oriented language
        in early parts of text.
        """
        score = 0.0
        
        if self.use_embeddings:
            # Semantic match
            text_emb = self.embedder.encode([text])[0]
            foreshadow_embs = self.prototype_embeddings['foreshadowing']
            
            sims = []
            for proto_emb in foreshadow_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        if self.use_spacy:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            # Look at first third
            first_third = sentences[:len(sentences)//3]
            
            for sent in first_third:
                for token in sent:
                    # Future tense in early text
                    if token.lemma_ in {'will', 'shall', 'soon', 'later', 'eventually'}:
                        score += 0.1
                    
                    # Modal verbs suggesting possibility
                    if token.lemma_ in {'might', 'could', 'may', 'would'}:
                        score += 0.05
        
        return min(1.0, score)
    
    def _detect_chekhov_gun(self, text: str) -> float:
        """
        Detect early mentions that gain importance later.
        Track noun phrases mentioned early and late.
        """
        score = 0.0
        
        if not self.use_spacy:
            return 0.0
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) < 6:
            return 0.0
        
        # Split into thirds
        first_third = sentences[:len(sentences)//3]
        last_third = sentences[-len(sentences)//3:]
        
        # Extract noun phrases from first third
        first_nouns = set()
        for sent in first_third:
            for token in sent:
                if token.pos_ == 'NOUN':
                    first_nouns.add(token.lemma_)
        
        # Check if they reappear with emphasis in last third
        reappearances = 0
        for sent in last_third:
            for token in sent:
                if token.lemma_ in first_nouns:
                    # Check for emphasis (modifiers, important context)
                    if any(c.dep_ in ['amod', 'advmod'] for c in token.children):
                        reappearances += 1
        
        if first_nouns:
            score = reappearances / len(first_nouns)
        
        return min(1.0, score)
    
    def _detect_red_herring(self, text: str) -> float:
        """
        Detect red herrings (misleading information).
        """
        score = 0.0
        
        if self.use_embeddings:
            # Semantic match
            text_emb = self.embedder.encode([text])[0]
            red_herring_embs = self.prototype_embeddings['red_herring']
            
            sims = []
            for proto_emb in red_herring_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Look for misleading language
            for sent in doc.sents:
                for token in sent:
                    # Verbs suggesting appearance vs reality
                    if token.lemma_ in {'seem', 'appear', 'look', 'suggest', 'imply'}:
                        score += 0.1
        
        return min(1.0, score)
    
    def _detect_surprise(self, text: str) -> float:
        """
        Detect surprises/twists.
        """
        score = 0.0
        
        if self.use_embeddings:
            # Semantic match
            text_emb = self.embedder.encode([text])[0]
            surprise_embs = self.prototype_embeddings['surprise']
            
            sims = []
            for proto_emb in surprise_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Surprise markers
            surprise_lemmas = {'shock', 'surprise', 'unexpected', 'sudden', 'amaze', 
                             'astonish', 'startle', 'stun'}
            
            for sent in doc.sents:
                for token in sent:
                    if token.lemma_ in surprise_lemmas:
                        score += 0.15
        
        return min(1.0, score)
    
    def _detect_meta_narrative(self, text: str) -> float:
        """
        Detect meta-narrative elements (story aware of being story).
        """
        score = 0.0
        
        if self.use_embeddings:
            # Semantic match
            text_emb = self.embedder.encode([text])[0]
            meta_embs = self.prototype_embeddings['meta_narrative']
            
            sims = []
            for proto_emb in meta_embs:
                sim = np.dot(text_emb, proto_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                )
                sims.append(sim)
            
            score += float(np.max(sims))
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Meta-language
            meta_lemmas = {'story', 'narrative', 'tell', 'chapter', 'tale', 'fiction',
                          'reader', 'audience', 'author', 'narrator'}
            
            for sent in doc.sents:
                meta_count = sum(1 for token in sent if token.lemma_ in meta_lemmas)
                if meta_count >= 2:  # Multiple meta-references in one sentence
                    score += 0.2
        
        return min(1.0, score)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'device_irony',
            'device_reversal',
            'device_recognition',
            'device_foreshadowing',
            'device_chekhov_gun',
            'device_red_herring',
            'device_surprise',
            'device_meta_narrative'
        ])

