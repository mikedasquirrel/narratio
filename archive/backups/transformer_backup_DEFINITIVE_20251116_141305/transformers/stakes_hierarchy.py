"""
Stakes Hierarchy Transformer

Uses NLP to analyze and classify narrative stakes.
Semantic analysis of consequence magnitude.

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


class StakesHierarchyTransformer(BaseEstimator, TransformerMixin):
    """
    Analyzes stakes hierarchy using NLP.
    
    Features (5 total):
    1. Personal vs global stakes
    2. Short-term vs long-term stakes
    3. Reversible vs irreversible stakes
    4. Stakes escalation rate
    5. Stakes authenticity
    
    Uses:
    - Semantic embeddings for stakes concepts
    - Dependency parsing for consequence language
    - Temporal analysis for time horizon
    - Scope analysis for impact range
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize stakes analyzer"""
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
        
        # Stakes prototypes
        self.stakes_prototypes = {
            'global': [
                "world-changing consequences",
                "affects everyone globally",
                "universal impact on humanity",
                "civilization-level stakes",
                "planetary consequences"
            ],
            'personal': [
                "individual life at stake",
                "personal consequences",
                "affecting single person",
                "individual fate decided",
                "personal survival"
            ],
            'irreversible': [
                "permanent consequences",
                "no going back",
                "irreversible decision",
                "final and unchangeable",
                "lasting forever"
            ],
            'high_authenticity': [
                "genuine life-or-death",
                "real tangible consequences",
                "actual stakes not artificial",
                "meaningful real impact",
                "true consequences matter"
            ]
        }
        
        # Embed prototypes
        self.prototype_embeddings = {}
        if self.use_embeddings:
            for concept, examples in self.stakes_prototypes.items():
                self.prototype_embeddings[concept] = self.embedder.encode(examples)
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to stakes hierarchy features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 5)
            Stakes hierarchy features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_stakes_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_stakes_features(self, text: str) -> List[float]:
        """Extract all stakes features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # 1. Personal vs global (0 = personal, 1 = global)
            scope = self._compute_stakes_scope(text, doc)
            features.append(scope)
            
            # 2. Short-term vs long-term (0 = short, 1 = long)
            time_horizon = self._compute_time_horizon(doc)
            features.append(time_horizon)
            
            # 3. Reversibility (0 = reversible, 1 = irreversible)
            reversibility = self._compute_reversibility(text, doc)
            features.append(reversibility)
            
            # 4. Stakes escalation rate
            escalation = self._compute_escalation_rate(doc)
            features.append(escalation)
            
            # 5. Stakes authenticity
            authenticity = self._compute_stakes_authenticity(text, doc)
            features.append(authenticity)
        else:
            # Fallback
            features = [0.5, 0.5, 0.4, 0.3, 0.6]
        
        return features
    
    def _compute_stakes_scope(self, text: str, doc) -> float:
        """
        Determine if stakes are personal or global.
        Returns 0-1 where 1 is global, 0 is personal.
        """
        score = 0.0
        
        # Semantic match to global stakes
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            global_embs = self.prototype_embeddings['global']
            personal_embs = self.prototype_embeddings['personal']
            
            global_sims = [
                np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9)
                for emb in global_embs
            ]
            
            personal_sims = [
                np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9)
                for emb in personal_embs
            ]
            
            global_max = float(np.max(global_sims))
            personal_max = float(np.max(personal_sims))
            
            # Normalize to 0-1
            total = global_max + personal_max
            if total > 0:
                score = global_max / total
        
        # Linguistic markers of scope
        global_lemmas = {'world', 'everyone', 'humanity', 'planet', 'civilization',
                        'society', 'all', 'universal', 'global', 'entire'}
        personal_lemmas = {'i', 'me', 'my', 'mine', 'myself', 'personal', 'individual'}
        
        global_count = sum(1 for sent in doc.sents for token in sent if token.lemma_ in global_lemmas)
        personal_count = sum(1 for sent in doc.sents for token in sent if token.lemma_ in personal_lemmas)
        
        total_markers = global_count + personal_count
        if total_markers > 0:
            linguistic_score = global_count / total_markers
            score = (score + linguistic_score) / 2  # Average with semantic score
        
        return score
    
    def _compute_time_horizon(self, doc) -> float:
        """
        Determine time horizon of stakes.
        Returns 0-1 where 1 is long-term, 0 is short-term.
        """
        short_term_lemmas = {'now', 'immediately', 'today', 'tonight', 'instant',
                            'moment', 'current', 'present', 'urgent'}
        long_term_lemmas = {'future', 'forever', 'always', '永', 'legacy',
                           'generation', 'lasting', 'permanent', 'eternal', 'ultimate'}
        
        short_count = sum(1 for sent in doc.sents for token in sent if token.lemma_ in short_term_lemmas)
        long_count = sum(1 for sent in doc.sents for token in sent if token.lemma_ in long_term_lemmas)
        
        total = short_count + long_count
        if total > 0:
            return long_count / total
        
        return 0.5  # Default to middle
    
    def _compute_reversibility(self, text: str, doc) -> float:
        """
        Determine if consequences are reversible.
        Returns 0-1 where 1 is irreversible, 0 is reversible.
        """
        score = 0.0
        
        # Semantic match to irreversible concepts
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            irreversible_embs = self.prototype_embeddings['irreversible']
            
            sims = [
                np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9)
                for emb in irreversible_embs
            ]
            
            score += float(np.max(sims))
        
        # Linguistic markers
        irreversible_lemmas = {'permanent', 'forever', 'irreversible', 'final',
                              'death', 'destroy', 'end', 'never', 'always'}
        reversible_lemmas = {'temporary', 'undo', 'reverse', 'fix', 'repair',
                            'recover', 'heal', 'restore'}
        
        irreversible_count = sum(1 for sent in doc.sents for token in sent if token.lemma_ in irreversible_lemmas)
        reversible_count = sum(1 for sent in doc.sents for token in sent if token.lemma_ in reversible_lemmas)
        
        total = irreversible_count + reversible_count
        if total > 0:
            linguistic_score = irreversible_count / total
            score = (score + linguistic_score) / 2
        
        return min(1.0, score)
    
    def _compute_escalation_rate(self, doc) -> float:
        """
        Measure how quickly stakes escalate over narrative.
        """
        sentences = list(doc.sents)
        if len(sentences) < 4:
            return 0.0
        
        # Stakes intensity markers
        intensity_lemmas = {'critical', 'crucial', 'vital', 'essential', 'urgent',
                           'dire', 'desperate', 'extreme', 'ultimate', 'final'}
        
        # Divide into quarters
        quarter = len(sentences) // 4
        quarters = [sentences[i*quarter:(i+1)*quarter] for i in range(4)]
        
        intensities = []
        for quarter_sents in quarters:
            if quarter_sents:
                intensity = sum(
                    1 for sent in quarter_sents
                    for token in sent if token.lemma_ in intensity_lemmas
                )
                intensities.append(intensity / len(quarter_sents))
            else:
                intensities.append(0)
        
        # Escalation = increase over time
        if len(intensities) >= 2:
            escalation = intensities[-1] - intensities[0]
            return float(np.clip((escalation + 1) / 2, 0, 1))
        
        return 0.0
    
    def _compute_stakes_authenticity(self, text: str, doc) -> float:
        """
        Measure how authentic/genuine the stakes feel.
        """
        score = 0.0
        
        # Semantic match to authentic stakes
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            auth_embs = self.prototype_embeddings['high_authenticity']
            
            sims = [
                np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9)
                for emb in auth_embs
            ]
            
            score += float(np.max(sims))
        
        # Authenticity markers
        # Concrete, specific language = more authentic
        specific_count = sum(1 for ent in doc.ents if ent.label_ in ['PERSON', 'GPE', 'DATE', 'MONEY'])
        
        # Numbers and specifics
        import re
        numbers = len(re.findall(r'\d+', text))
        
        specificity = (specific_count + numbers) / (len(list(doc.sents)) + 1)
        score += min(0.3, specificity)
        
        # Avoid melodramatic language (reduces authenticity)
        melodrama_lemmas = {'absolutely', 'totally', 'completely', 'utterly',
                           'entirely', 'extremely', 'incredibly'}
        
        melodrama_count = sum(1 for sent in doc.sents for token in sent if token.lemma_ in melodrama_lemmas)
        
        # High melodrama reduces authenticity
        if melodrama_count > len(list(doc.sents)) / 2:
            score *= 0.7
        
        return min(1.0, score)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'stakes_scope',
            'stakes_time_horizon',
            'stakes_irreversibility',
            'stakes_escalation_rate',
            'stakes_authenticity'
        ])

