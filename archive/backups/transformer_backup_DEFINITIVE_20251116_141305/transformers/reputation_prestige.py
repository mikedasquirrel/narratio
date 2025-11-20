"""
Reputation & Prestige Transformer

Universal reputation/status analysis for ALL domains with recognition dynamics.
Used by: Oscars, sports (Hall of Fame), startups, grants, restaurants, all prestige domains.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
from typing import List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.shared_models import SharedModelRegistry
from .utils.input_validation import ensure_string_list


class ReputationPrestigeTransformer(BaseEstimator, TransformerMixin):
    """
    Extract reputation and prestige features - UNIVERSAL.
    
    Features (12 total):
    1. Reputation score (overall fame/recognition)
    2. Prestige level (elite vs amateur)
    3. Legacy indicators (historical significance)
    4. Award history (previous wins/nominations)
    5. Peer recognition (expert respect)
    6. Public recognition (mass fame)
    7. Media coverage (attention level)
    8. Influencer status (reach/impact)
    9. Authority markers (credentials)
    10. Endorsement quality (who backs them)
    11. Scandal/controversy (negative reputation)
    12. Reputation trajectory (rising/falling)
    
    Works with:
    - Text (reputation language analysis)
    - Dict with reputation metrics (awards, citations, followers)
    - Mixed data
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize reputation analyzer"""
        self.use_spacy = use_spacy
        self.use_embeddings = use_embeddings
        
        self.nlp = None
        self.embedder = None
        
        # Prestige prototypes
        self.prestige_prototypes = {
            'elite': "elite status with highest recognition and prestige",
            'legendary': "legendary reputation with historic significance",
            'authority': "authoritative position with expert recognition",
            'influencer': "influential reach with broad impact",
            'controversial': "controversial reputation with mixed public perception"
        }
    
    def fit(self, X, y=None):
        """Fit transformer (load shared models)"""
        X = ensure_string_list(X)
        
        # Load shared models
        if self.use_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        
        if self.use_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
            
            # Embed prototypes
            if self.embedder:
                self.prototype_embeddings = {}
                for concept, description in self.prestige_prototypes.items():
                    self.prototype_embeddings[concept] = self.embedder.encode([description])[0]
        
        return self
    
    def transform(self, X):
        """
        Transform to reputation/prestige features.
        
        Parameters
        ----------
        X : array-like of strings or dicts
            Reputation data
            
        Returns
        -------
        features : ndarray of shape (n_samples, 12)
            Reputation features
        """
        features = []
        
        for item in X:
            feat = self._extract_reputation_features(item)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_reputation_features(self, item: Union[str, Dict]) -> List[float]:
        """Extract all reputation features"""
        # Get text
        if isinstance(item, str):
            text = item
            metrics = {}
        elif isinstance(item, dict):
            text = str(item.get('text', item.get('narrative', '')))
            metrics = item
        else:
            text = str(item)
            metrics = {}
        
        features = []
        
        if self.nlp and text:
            doc = self.nlp(text[:5000])
            n_words = len(doc)
        else:
            doc = None
            n_words = len(text.split()) + 1 if text else 1
        
        # 1. Overall reputation score
        reputation = self._compute_reputation_score(text, doc, metrics)
        features.append(reputation)
        
        # 2. Prestige level
        prestige = self._compute_prestige_level(text, doc, metrics)
        features.append(prestige)
        
        # 3. Legacy indicators
        legacy = self._compute_legacy_indicators(text, doc)
        features.append(legacy)
        
        # 4. Award history
        awards = self._compute_award_history(text, doc, metrics)
        features.append(awards)
        
        # 5. Peer recognition
        peer_recognition = self._compute_peer_recognition(text, doc, metrics)
        features.append(peer_recognition)
        
        # 6. Public recognition
        public_recognition = self._compute_public_recognition(text, doc, metrics)
        features.append(public_recognition)
        
        # 7. Media coverage
        media = self._compute_media_coverage(text, doc, metrics)
        features.append(media)
        
        # 8. Influencer status
        influence = self._compute_influencer_status(text, doc, metrics)
        features.append(influence)
        
        # 9. Authority markers
        authority = self._compute_authority_markers(text, doc)
        features.append(authority)
        
        # 10. Endorsement quality
        endorsements = self._compute_endorsement_quality(text, doc)
        features.append(endorsements)
        
        # 11. Scandal/controversy
        scandal = self._compute_scandal_score(text, doc)
        features.append(scandal)
        
        # 12. Reputation trajectory
        trajectory = self._compute_reputation_trajectory(text, doc, metrics)
        features.append(trajectory)
        
        return features
    
    def _compute_reputation_score(self, text: str, doc, metrics: Dict) -> float:
        """Overall reputation score"""
        score = 0.5
        
        # From metrics if available
        if 'reputation_score' in metrics:
            return float(metrics['reputation_score'])
        if 'fame_score' in metrics:
            return float(metrics['fame_score'])
        
        # From text
        if doc:
            reputation_lemmas = {'famous', 'renowned', 'acclaimed', 'celebrated', 'respected', 'esteemed'}
            rep_count = sum(1 for token in doc if token.lemma_ in reputation_lemmas)
            score = min(1.0, rep_count / len(doc) * 10)
        
        return score
    
    def _compute_prestige_level(self, text: str, doc, metrics: Dict) -> float:
        """Prestige level (elite vs amateur)"""
        # Check metrics first
        if 'prestige_level' in metrics:
            return float(metrics['prestige_level'])
        
        score = 0.0
        
        # Semantic match to elite
        if self.embedder and hasattr(self, 'prototype_embeddings') and text:
            text_emb = self.embedder.encode([text[:1000]])[0]
            elite_emb = self.prototype_embeddings['elite']
            
            sim = np.dot(text_emb, elite_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(elite_emb) + 1e-9)
            score += float(sim)
        
        if doc:
            prestige_lemmas = {'elite', 'premier', 'prestigious', 'exclusive', 'distinguished'}
            prestige_count = sum(1 for token in doc if token.lemma_ in prestige_lemmas)
            score += min(0.3, prestige_count / len(doc) * 10)
        
        return min(1.0, score)
    
    def _compute_legacy_indicators(self, text: str, doc) -> float:
        """Historical significance and legacy"""
        if not doc:
            return 0.3
        
        legacy_lemmas = {'legacy', 'legend', 'historic', 'legendary', 'iconic', 'immortal', 'timeless'}
        legacy_count = sum(1 for token in doc if token.lemma_ in legacy_lemmas)
        
        return min(1.0, legacy_count / len(doc) * 10)
    
    def _compute_award_history(self, text: str, doc, metrics: Dict) -> float:
        """Award and honor history"""
        # Check metrics
        if 'award_count' in metrics:
            return min(1.0, float(metrics['award_count']) / 10)
        if 'awards' in metrics:
            return min(1.0, len(metrics['awards']) / 5)
        
        # From text
        if doc:
            award_lemmas = {'award', 'prize', 'honor', 'recognition', 'winner', 'champion', 'medal'}
            award_count = sum(1 for token in doc if token.lemma_ in award_lemmas)
            return min(1.0, award_count / len(doc) * 10)
        
        return 0.3
    
    def _compute_peer_recognition(self, text: str, doc, metrics: Dict) -> float:
        """Recognition by peers/experts"""
        # Check metrics
        if 'peer_recognition' in metrics:
            return float(metrics['peer_recognition'])
        if 'citations' in metrics:
            return min(1.0, float(metrics['citations']) / 100)
        
        # From text
        if doc:
            peer_lemmas = {'peer', 'expert', 'professional', 'colleague', 'respected', 'admired'}
            peer_count = sum(1 for token in doc if token.lemma_ in peer_lemmas)
            return min(1.0, peer_count / len(doc) * 10)
        
        return 0.4
    
    def _compute_public_recognition(self, text: str, doc, metrics: Dict) -> float:
        """Mass public recognition"""
        # Check metrics
        if 'followers' in metrics:
            return min(1.0, float(metrics['followers']) / 1000000)  # Normalize to millions
        if 'fans' in metrics:
            return min(1.0, float(metrics['fans']) / 100000)
        
        # From text
        if doc:
            public_lemmas = {'popular', 'famous', 'celebrity', 'star', 'household', 'known'}
            public_count = sum(1 for token in doc if token.lemma_ in public_lemmas)
            return min(1.0, public_count / len(doc) * 10)
        
        return 0.3
    
    def _compute_media_coverage(self, text: str, doc, metrics: Dict) -> float:
        """Media attention level"""
        if 'media_mentions' in metrics:
            return min(1.0, float(metrics['media_mentions']) / 1000)
        
        if doc:
            media_lemmas = {'media', 'coverage', 'press', 'news', 'headline', 'feature', 'profile'}
            media_count = sum(1 for token in doc if token.lemma_ in media_lemmas)
            return min(1.0, media_count / len(doc) * 10)
        
        return 0.3
    
    def _compute_influencer_status(self, text: str, doc, metrics: Dict) -> float:
        """Influencer reach and impact"""
        if 'influence_score' in metrics:
            return float(metrics['influence_score'])
        
        if doc:
            influence_lemmas = {'influence', 'impact', 'shape', 'change', 'inspire', 'lead'}
            influence_count = sum(1 for token in doc if token.pos_ == 'VERB' and token.lemma_ in influence_lemmas)
            return min(1.0, influence_count / len(doc) * 10)
        
        return 0.4
    
    def _compute_authority_markers(self, text: str, doc) -> float:
        """Credential and authority markers"""
        if not doc:
            return 0.3
        
        authority_lemmas = {'expert', 'authority', 'specialist', 'professor', 'doctor', 'master'}
        titles = sum(1 for token in doc if token.text in {'Dr.', 'Prof.', 'Sir', 'Dame'})
        authority_words = sum(1 for token in doc if token.lemma_ in authority_lemmas)
        
        return min(1.0, (titles + authority_words) / len(doc) * 10)
    
    def _compute_endorsement_quality(self, text: str, doc) -> float:
        """Quality of endorsements/backing"""
        if not doc:
            return 0.3
        
        endorsement_lemmas = {'endorse', 'support', 'back', 'sponsor', 'recommend', 'approve'}
        endorsement_count = sum(1 for token in doc if token.lemma_ in endorsement_lemmas)
        
        return min(1.0, endorsement_count / len(doc) * 10)
    
    def _compute_scandal_score(self, text: str, doc) -> float:
        """Negative reputation (scandal, controversy)"""
        if not doc:
            return 0.0
        
        scandal_lemmas = {'scandal', 'controversy', 'accused', 'allegation', 'criticized', 'condemned'}
        scandal_count = sum(1 for token in doc if token.lemma_ in scandal_lemmas)
        
        return min(1.0, scandal_count / len(doc) * 10)
    
    def _compute_reputation_trajectory(self, text: str, doc, metrics: Dict) -> float:
        """Reputation trend (0 = falling, 0.5 = stable, 1 = rising)"""
        # Check metrics
        if 'reputation_trend' in metrics:
            return float(metrics['reputation_trend'])
        
        # From text
        if doc:
            rising_lemmas = {'rising', 'growing', 'emerging', 'ascending', 'climbing'}
            falling_lemmas = {'falling', 'declining', 'fading', 'descending', 'waning'}
            
            rising = sum(1 for token in doc if token.lemma_ in rising_lemmas)
            falling = sum(1 for token in doc if token.lemma_ in falling_lemmas)
            
            total = rising + falling
            if total > 0:
                return rising / total
        
        return 0.5
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'reputation_score',
            'reputation_prestige_level',
            'reputation_legacy',
            'reputation_award_history',
            'reputation_peer_recognition',
            'reputation_public_recognition',
            'reputation_media_coverage',
            'reputation_influencer_status',
            'reputation_authority_markers',
            'reputation_endorsement_quality',
            'reputation_scandal_score',
            'reputation_trajectory'
        ])

