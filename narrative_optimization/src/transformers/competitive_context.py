"""
Competitive Context Transformer

Universal competitive analysis for ALL domains with competition.
Used by: sports, business, entertainment, markets, prestige domains.

Extracts competitive narratives using NLP and semantic analysis.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.shared_models import SharedModelRegistry
from .utils.input_validation import ensure_string_list

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class CompetitiveContextTransformer(BaseEstimator, TransformerMixin):
    """
    Extract competitive context features - UNIVERSAL across domains.
    
    Works for:
    - Sports (NFL, NBA, Tennis, Golf, UFC, Boxing, MLB, etc.)
    - Business (Startups, Companies competing)
    - Entertainment (Oscars, Music charts, YouTube)
    - Markets (Crypto, Stocks)
    - Any competitive domain
    
    Features (12 total):
    1. Competitive intensity
    2. Market saturation
    3. Underdog vs favorite dynamics
    4. Competitive positioning
    5. Historical dominance
    6. Rivalry intensity
    7. Competitive momentum
    8. Field strength
    9. Breakthrough potential
    10. Defensive positioning
    11. Strategic positioning
    12. Competitive volatility
    
    Uses:
    - Semantic embeddings for competitive concepts
    - NLP for competitive language
    - Statistical analysis for positioning
    """
    
    def __init__(self, use_embeddings: bool = True, use_spacy: bool = True):
        """Initialize competitive analyzer"""
        self.use_embeddings = use_embeddings
        self.use_spacy = use_spacy
        
        self.nlp = None
        self.embedder = None
        
        # Competitive concept prototypes
        self.competitive_prototypes = {
            'high_intensity': [
                "fierce competition with intense rivalry",
                "highly competitive environment with strong opponents",
                "challenging competitive landscape"
            ],
            'underdog': [
                "facing overwhelming odds as underdog",
                "competing against favored opponent",
                "challenger position against established leader"
            ],
            'dominant': [
                "dominant position with clear superiority",
                "established leader with strong advantage",
                "overwhelming competitive strength"
            ],
            'rivalry': [
                "intense rivalry with historical competition",
                "longstanding competitive relationship",
                "fierce contested matchup between rivals"
            ],
            'breakthrough': [
                "disruptive potential challenging establishment",
                "innovative approach threatening incumbents",
                "emerging force disrupting status quo"
            ]
        }
    
    def fit(self, X, y=None):
        """Fit transformer (load shared models)"""
        X = ensure_string_list(X)
        
        # Load shared models
        if self.use_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        
        if self.use_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
            
            # Embed competitive prototypes
            if self.embedder:
                self.prototype_embeddings = {}
                for concept, descriptions in self.competitive_prototypes.items():
                    self.prototype_embeddings[concept] = self.embedder.encode(descriptions)
        
        return self
    
    def transform(self, X):
        """
        Transform to competitive context features.
        
        Parameters
        ----------
        X : array-like of strings or dicts
            Competitive narratives or data
            
        Returns
        -------
        features : ndarray of shape (n_samples, 12)
            Competitive context features
        """
        X = ensure_string_list(X)
        features = []
        
        for item in X:
            feat = self._extract_competitive_features(item)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_competitive_features(self, text: str) -> List[float]:
        """Extract all competitive context features"""
        features = []
        
        if self.nlp:
            doc = self.nlp(text[:5000])
            n_words = len(doc)
        else:
            doc = None
            n_words = len(text.split()) + 1
        
        # 1. Competitive intensity
        intensity = self._compute_competitive_intensity(text, doc)
        features.append(intensity)
        
        # 2. Market saturation
        saturation = self._compute_market_saturation(text, doc)
        features.append(saturation)
        
        # 3. Underdog vs favorite dynamics
        underdog_score = self._compute_underdog_dynamic(text, doc)
        features.append(underdog_score)
        
        # 4. Competitive positioning
        positioning = self._compute_competitive_positioning(text, doc)
        features.append(positioning)
        
        # 5. Historical dominance
        dominance = self._compute_historical_dominance(text, doc)
        features.append(dominance)
        
        # 6. Rivalry intensity
        rivalry = self._compute_rivalry_intensity(text, doc)
        features.append(rivalry)
        
        # 7. Competitive momentum
        momentum = self._compute_competitive_momentum(text, doc)
        features.append(momentum)
        
        # 8. Field strength
        field_strength = self._compute_field_strength(text, doc)
        features.append(field_strength)
        
        # 9. Breakthrough potential
        breakthrough = self._compute_breakthrough_potential(text, doc)
        features.append(breakthrough)
        
        # 10. Defensive positioning
        defensive = self._compute_defensive_positioning(text, doc)
        features.append(defensive)
        
        # 11. Strategic positioning
        strategic = self._compute_strategic_positioning(text, doc)
        features.append(strategic)
        
        # 12. Competitive volatility
        volatility = self._compute_competitive_volatility(text, doc)
        features.append(volatility)
        
        return features
    
    def _compute_competitive_intensity(self, text: str, doc) -> float:
        """Measure how fierce the competition is"""
        score = 0.0
        
        # Semantic match if available
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            intensity_embs = self.prototype_embeddings['high_intensity']
            
            sims = [np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9)
                   for emb in intensity_embs]
            score += float(np.max(sims))
        
        # Linguistic markers
        if doc:
            intensity_lemmas = {'fierce', 'intense', 'competitive', 'challenging', 'difficult', 'tough'}
            intensity_count = sum(1 for token in doc if token.lemma_ in intensity_lemmas)
            score += min(0.3, intensity_count / len(doc) * 10)
        
        return min(1.0, score)
    
    def _compute_market_saturation(self, text: str, doc) -> float:
        """Measure market saturation (crowded vs open)"""
        score = 0.5  # Default to medium
        
        if doc:
            # Crowded indicators
            crowded_lemmas = {'crowded', 'saturated', 'competitive', 'many', 'numerous', 'packed'}
            crowded_count = sum(1 for token in doc if token.lemma_ in crowded_lemmas)
            
            # Open indicators  
            open_lemmas = {'open', 'opportunity', 'emerging', 'new', 'untapped', 'virgin'}
            open_count = sum(1 for token in doc if token.lemma_ in open_lemmas)
            
            total = crowded_count + open_count
            if total > 0:
                score = crowded_count / total  # High = saturated
        
        return score
    
    def _compute_underdog_dynamic(self, text: str, doc) -> float:
        """0 = heavy favorite, 1 = heavy underdog, 0.5 = even"""
        score = 0.5
        
        # Semantic match
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            underdog_embs = self.prototype_embeddings['underdog']
            
            sims = [np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9)
                   for emb in underdog_embs]
            underdog_sim = float(np.max(sims))
            
            # If high underdog similarity, return high score
            if underdog_sim > 0.3:
                score = 0.5 + underdog_sim * 0.5
        
        # Linguistic markers
        if doc:
            underdog_lemmas = {'underdog', 'challenger', 'upset', 'unlikely', 'longshot', 'outsider'}
            favorite_lemmas = {'favorite', 'expected', 'dominant', 'superior', 'leading'}
            
            underdog_count = sum(1 for token in doc if token.lemma_ in underdog_lemmas)
            favorite_count = sum(1 for token in doc if token.lemma_ in favorite_lemmas)
            
            total = underdog_count + favorite_count
            if total > 0:
                score = underdog_count / total
        
        return score
    
    def _compute_competitive_positioning(self, text: str, doc) -> float:
        """Competitive rank/position (1.0 = top, 0 = bottom)"""
        score = 0.5
        
        if doc:
            # Look for ranking language
            top_lemmas = {'first', 'top', 'leading', 'best', 'premier', 'elite', 'champion'}
            bottom_lemmas = {'last', 'bottom', 'worst', 'struggling', 'failing'}
            
            top_count = sum(1 for token in doc if token.lemma_ in top_lemmas)
            bottom_count = sum(1 for token in doc if token.lemma_ in bottom_lemmas)
            
            total = top_count + bottom_count
            if total > 0:
                score = top_count / total
        
        return score
    
    def _compute_historical_dominance(self, text: str, doc) -> float:
        """Historical competitive dominance"""
        score = 0.0
        
        if doc:
            # Dominance markers
            dominance_lemmas = {'dominant', 'dynasty', 'champion', 'winner', 'successful', 'legendary', 'historic'}
            dominance_count = sum(1 for token in doc if token.lemma_ in dominance_lemmas)
            
            # Historical markers
            historical_lemmas = {'history', 'always', 'consistently', 'traditionally', 'perennial'}
            historical_count = sum(1 for token in doc if token.lemma_ in historical_lemmas)
            
            # Combined
            score = (dominance_count + historical_count) / (len(doc) + 1) * 10
        
        return min(1.0, score)
    
    def _compute_rivalry_intensity(self, text: str, doc) -> float:
        """Measure rivalry intensity"""
        score = 0.0
        
        # Semantic match
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            rivalry_embs = self.prototype_embeddings['rivalry']
            
            sims = [np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9)
                   for emb in rivalry_embs]
            score += float(np.max(sims))
        
        if doc:
            rivalry_lemmas = {'rivalry', 'rival', 'nemesis', 'archrival', 'enemy', 'opponent'}
            rivalry_count = sum(1 for token in doc if token.lemma_ in rivalry_lemmas)
            score += min(0.3, rivalry_count / len(doc) * 10)
        
        return min(1.0, score)
    
    def _compute_competitive_momentum(self, text: str, doc) -> float:
        """Measure competitive momentum (winning streak)"""
        score = 0.5  # Neutral
        
        if doc:
            # Positive momentum
            positive_lemmas = {'winning', 'streak', 'hot', 'surging', 'rising', 'improving', 'momentum'}
            positive_count = sum(1 for token in doc if token.lemma_ in positive_lemmas)
            
            # Negative momentum
            negative_lemmas = {'losing', 'slump', 'struggling', 'declining', 'falling', 'regressing'}
            negative_count = sum(1 for token in doc if token.lemma_ in negative_lemmas)
            
            total = positive_count + negative_count
            if total > 0:
                score = positive_count / total
        
        return score
    
    def _compute_field_strength(self, text: str, doc) -> float:
        """Measure quality/strength of competition field"""
        score = 0.5
        
        if doc:
            strong_field_lemmas = {'elite', 'top', 'competitive', 'strong', 'talented', 'skilled'}
            weak_field_lemmas = {'weak', 'poor', 'inferior', 'mediocre', 'amateur'}
            
            strong_count = sum(1 for token in doc if token.lemma_ in strong_field_lemmas)
            weak_count = sum(1 for token in doc if token.lemma_ in weak_field_lemmas)
            
            total = strong_count + weak_count
            if total > 0:
                score = strong_count / total
        
        return score
    
    def _compute_breakthrough_potential(self, text: str, doc) -> float:
        """Measure potential for breakthrough/disruption"""
        score = 0.0
        
        # Semantic match
        if self.embedder and hasattr(self, 'prototype_embeddings'):
            text_emb = self.embedder.encode([text[:1000]])[0]
            breakthrough_embs = self.prototype_embeddings['breakthrough']
            
            sims = [np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9)
                   for emb in breakthrough_embs]
            score += float(np.max(sims))
        
        if doc:
            breakthrough_lemmas = {'breakthrough', 'disruptive', 'revolutionary', 'innovative', 'game-changer', 'upset'}
            breakthrough_count = sum(1 for token in doc if token.lemma_ in breakthrough_lemmas)
            score += min(0.3, breakthrough_count / len(doc) * 10)
        
        return min(1.0, score)
    
    def _compute_defensive_positioning(self, text: str, doc) -> float:
        """0 = attacking, 1 = defending, 0.5 = balanced"""
        score = 0.5
        
        if doc:
            defensive_lemmas = {'defend', 'protect', 'guard', 'preserve', 'maintain', 'hold'}
            attacking_lemmas = {'attack', 'challenge', 'pursue', 'chase', 'hunt', 'target'}
            
            defensive_count = sum(1 for token in doc if token.pos_ == 'VERB' and token.lemma_ in defensive_lemmas)
            attacking_count = sum(1 for token in doc if token.pos_ == 'VERB' and token.lemma_ in attacking_lemmas)
            
            total = defensive_count + attacking_count
            if total > 0:
                score = defensive_count / total
        
        return score
    
    def _compute_strategic_positioning(self, text: str, doc) -> float:
        """Measure strategic positioning clarity"""
        score = 0.0
        
        if doc:
            strategy_lemmas = {'strategy', 'approach', 'plan', 'tactic', 'positioning', 'advantage', 'edge'}
            strategy_count = sum(1 for token in doc if token.lemma_ in strategy_lemmas)
            score = min(1.0, strategy_count / (len(doc) + 1) * 10)
        
        return score
    
    def _compute_competitive_volatility(self, text: str, doc) -> float:
        """Measure how volatile/unpredictable the competition is"""
        score = 0.5
        
        if doc:
            volatile_lemmas = {'unpredictable', 'volatile', 'uncertain', 'variable', 'inconsistent', 'erratic'}
            stable_lemmas = {'consistent', 'predictable', 'stable', 'reliable', 'steady'}
            
            volatile_count = sum(1 for token in doc if token.lemma_ in volatile_lemmas)
            stable_count = sum(1 for token in doc if token.lemma_ in stable_lemmas)
            
            total = volatile_count + stable_count
            if total > 0:
                score = volatile_count / total
        
        return score
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'competitive_intensity',
            'competitive_saturation',
            'competitive_underdog_score',
            'competitive_positioning',
            'competitive_historical_dominance',
            'competitive_rivalry_intensity',
            'competitive_momentum',
            'competitive_field_strength',
            'competitive_breakthrough_potential',
            'competitive_defensive_positioning',
            'competitive_strategic_positioning',
            'competitive_volatility'
        ])

