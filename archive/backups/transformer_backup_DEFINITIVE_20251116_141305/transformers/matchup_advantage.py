"""
Matchup Advantage Transformer

Uses NLP to analyze competitive matchups and style contrasts.
Domain-adaptive for sports narratives.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
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


class MatchupAdvantageTransformer(BaseEstimator, TransformerMixin):
    """
    Analyzes competitive matchups using NLP.
    
    Features (8 total):
    1. Style matchup score (contrasting approaches)
    2. Strength vs weakness matchup
    3. Historical pattern match
    4. Scheme/strategy advantage
    5. Pace/tempo compatibility
    6. Strategic approach contrast
    7. Counter-style effectiveness
    8. Matchup favorability index
    
    Uses:
    - Semantic embeddings for style analysis
    - Entity recognition for participants
    - Dependency parsing for competitive language
    - Comparative constructions analysis
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize matchup analyzer"""
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
        
        # Style contrast prototypes
        self.style_prototypes = {
            'aggressive_vs_defensive': [
                "attacking style versus defensive approach",
                "offensive mindset against conservative strategy",
                "aggressive play confronting cautious tactics"
            ],
            'fast_vs_slow': [
                "fast-paced game versus slow methodical play",
                "quick tempo against deliberate pace",
                "rapid style facing patient approach"
            ],
            'experience_vs_youth': [
                "veteran experience against youthful energy",
                "seasoned professional facing rising talent",
                "established presence versus emerging force"
            ],
            'power_vs_finesse': [
                "physical power versus technical skill",
                "raw strength against refined technique",
                "brute force facing tactical precision"
            ]
        }
        
        # Embed prototypes
        self.prototype_embeddings = {}
        if self.use_embeddings:
            for matchup, examples in self.style_prototypes.items():
                self.prototype_embeddings[matchup] = self.embedder.encode(examples)
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to matchup advantage features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 8)
            Matchup advantage features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_matchup_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_matchup_features(self, text: str) -> List[float]:
        """Extract all matchup features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # 1. Style matchup score
            style_score = self._compute_style_matchup(text, doc)
            features.append(style_score)
            
            # 2. Strength vs weakness
            strength_weakness = self._detect_strength_weakness_matchup(doc)
            features.append(strength_weakness)
            
            # 3. Historical pattern match
            historical_match = self._compute_historical_pattern(doc)
            features.append(historical_match)
            
            # 4. Scheme advantage
            scheme_advantage = self._detect_scheme_advantage(doc)
            features.append(scheme_advantage)
            
            # 5. Pace/tempo compatibility
            tempo_compatibility = self._compute_tempo_compatibility(doc)
            features.append(tempo_compatibility)
            
            # 6. Strategic approach contrast
            strategic_contrast = self._compute_strategic_contrast(doc)
            features.append(strategic_contrast)
            
            # 7. Counter-style effectiveness
            counter_effectiveness = self._compute_counter_effectiveness(doc)
            features.append(counter_effectiveness)
            
            # 8. Overall matchup favorability
            favorability = np.mean([
                style_score, strength_weakness, scheme_advantage,
                tempo_compatibility, counter_effectiveness
            ])
            features.append(favorability)
        else:
            # Fallback
            features = [0.5] * 8
        
        return features
    
    def _compute_style_matchup(self, text: str, doc) -> float:
        """
        Detect style contrasts using semantic matching.
        """
        score = 0.0
        
        if self.use_embeddings:
            text_emb = self.embedder.encode([text])[0]
            
            # Check each style matchup prototype
            max_sim = 0.0
            for matchup_type, proto_embs in self.prototype_embeddings.items():
                for proto_emb in proto_embs:
                    sim = np.dot(text_emb, proto_emb) / (
                        np.linalg.norm(text_emb) * np.linalg.norm(proto_emb) + 1e-9
                    )
                    max_sim = max(max_sim, sim)
            
            score = float(max_sim)
        
        # Linguistic markers of style contrast
        contrast_markers = {'versus', 'against', 'while', 'whereas', 'but', 'however'}
        
        for sent in doc.sents:
            has_contrast = any(token.lemma_ in contrast_markers for token in sent)
            if has_contrast:
                score += 0.1
        
        return min(1.0, score)
    
    def _detect_strength_weakness_matchup(self, doc) -> float:
        """
        Detect strength vs weakness matchups.
        """
        strength_count = 0
        weakness_count = 0
        
        strength_lemmas = {'strength', 'strong', 'advantage', 'excel', 'superior', 'dominant'}
        weakness_lemmas = {'weakness', 'weak', 'vulnerable', 'struggle', 'inferior', 'poor'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in strength_lemmas:
                    strength_count += 1
                if token.lemma_ in weakness_lemmas:
                    weakness_count += 1
        
        # Both strengths and weaknesses mentioned = explicit matchup analysis
        if strength_count > 0 and weakness_count > 0:
            # Normalize
            total = strength_count + weakness_count
            balance = 1.0 - abs(strength_count - weakness_count) / total
            return balance
        
        return 0.0
    
    def _compute_historical_pattern(self, doc) -> float:
        """
        Detect references to historical matchups or patterns.
        """
        historical_score = 0.0
        
        # Historical markers
        historical_lemmas = {'history', 'previous', 'past', 'before', 'traditionally',
                            'historically', 'record', 'last', 'prior', 'earlier'}
        
        for sent in doc.sents:
            historical_count = sum(1 for token in sent if token.lemma_ in historical_lemmas)
            if historical_count > 0:
                historical_score += 0.15
        
        return min(1.0, historical_score)
    
    def _detect_scheme_advantage(self, doc) -> float:
        """
        Detect strategic/tactical advantages.
        """
        scheme_score = 0.0
        
        # Tactical language
        tactical_lemmas = {'strategy', 'tactic', 'approach', 'scheme', 'plan',
                          'system', 'method', 'style', 'way', 'gameplan'}
        
        for sent in doc.sents:
            tactical_count = sum(1 for token in sent if token.lemma_ in tactical_lemmas)
            if tactical_count > 0:
                # Check for advantage language nearby
                advantage_lemmas = {'advantage', 'favor', 'benefit', 'suit', 'help'}
                advantage_count = sum(1 for token in sent if token.lemma_ in advantage_lemmas)
                
                if advantage_count > 0:
                    scheme_score += 0.2
        
        return min(1.0, scheme_score)
    
    def _compute_tempo_compatibility(self, doc) -> float:
        """
        Analyze pace/tempo references.
        """
        tempo_score = 0.0
        
        # Tempo markers
        tempo_lemmas = {'pace', 'tempo', 'speed', 'fast', 'slow', 'quick',
                       'rapid', 'methodical', 'deliberate', 'rushed'}
        
        tempo_mentions = sum(
            1 for sent in doc.sents
            for token in sent if token.lemma_ in tempo_lemmas
        )
        
        # Normalized by document length
        tempo_score = min(1.0, tempo_mentions / (len(list(doc.sents)) + 1) * 3)
        
        return tempo_score
    
    def _compute_strategic_contrast(self, doc) -> float:
        """
        Measure strategic differences.
        """
        contrast_score = 0.0
        
        # Look for comparative constructions
        for sent in doc.sents:
            for token in sent:
                # Comparative adjectives
                if token.tag_ in ['JJR', 'RBR']:  # Comparative forms
                    contrast_score += 0.1
                
                # "More X than Y" constructions
                if token.lemma_ == 'than':
                    contrast_score += 0.1
        
        return min(1.0, contrast_score)
    
    def _compute_counter_effectiveness(self, doc) -> float:
        """
        Detect counter-style or counter-strategy language.
        """
        counter_score = 0.0
        
        # Counter language
        counter_lemmas = {'counter', 'neutralize', 'negate', 'nullify', 'offset',
                         'match', 'answer', 'respond', 'adapt', 'adjust'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in counter_lemmas:
                    counter_score += 0.15
                    
                    # Check if describing ability/effectiveness
                    for child in token.children:
                        if child.lemma_ in {'can', 'able', 'capable', 'effective'}:
                            counter_score += 0.1
        
        return min(1.0, counter_score)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'matchup_style_contrast',
            'matchup_strength_weakness',
            'matchup_historical_pattern',
            'matchup_scheme_advantage',
            'matchup_tempo_compatibility',
            'matchup_strategic_contrast',
            'matchup_counter_effectiveness',
            'matchup_favorability_index'
        ])

