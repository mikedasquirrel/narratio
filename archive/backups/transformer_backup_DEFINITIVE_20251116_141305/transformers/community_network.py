"""
Community & Network Transformer

Universal community/network analysis for social domains.
Used by: Crypto, social media, nonprofits, platforms, brands, movements.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
from typing import List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.shared_models import SharedModelRegistry
from .utils.input_validation import ensure_string_list


class CommunityNetworkTransformer(BaseEstimator, TransformerMixin):
    """
    Extract community and network features - UNIVERSAL.
    
    Features (15 total):
    1. Community size
    2. Community engagement (active vs passive)
    3. Network density
    4. Influencer presence
    5. Growth rate
    6. Retention rate
    7. Community health
    8. Subgroup diversity
    9. Geographic distribution
    10. Cross-community bridges
    11. Community governance
    12. Shared values clarity
    13. Collective action potential
    14. Network effects strength
    15. Viral coefficient
    
    Works with:
    - Text (community language analysis)
    - Dict with community metrics (size, engagement, growth)
    - Mixed data
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize community analyzer"""
        self.use_spacy = use_spacy
        self.use_embeddings = use_embeddings
        
        self.nlp = None
        self.embedder = None
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        
        if self.use_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        
        if self.use_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
        
        return self
    
    def transform(self, X):
        """Transform to community features"""
        features = []
        
        for item in X:
            feat = self._extract_community_features(item)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_community_features(self, item: Union[str, Dict]) -> List[float]:
        """Extract all community features"""
        # Get text and metrics
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
        
        # 1. Community size
        if 'community_size' in metrics:
            # Log scale for size
            size = float(metrics['community_size'])
            features.append(min(1.0, np.log10(size + 1) / 7))  # Normalize to 10M
        elif 'followers' in metrics:
            size = float(metrics['followers'])
            features.append(min(1.0, np.log10(size + 1) / 7))
        else:
            # From text
            if doc:
                size_lemmas = {'community', 'members', 'followers', 'users', 'people', 'participants'}
                size_count = sum(1 for token in doc if token.lemma_ in size_lemmas)
                features.append(min(1.0, size_count / n_words * 5))
            else:
                features.append(0.3)
        
        # 2. Community engagement
        if 'engagement_rate' in metrics:
            features.append(float(metrics['engagement_rate']))
        else:
            if doc:
                engagement_lemmas = {'engage', 'active', 'participate', 'contribute', 'involved', 'passionate'}
                engagement_count = sum(1 for token in doc if token.lemma_ in engagement_lemmas)
                features.append(min(1.0, engagement_count / n_words * 10))
            else:
                features.append(0.4)
        
        # 3. Network density
        if 'network_density' in metrics:
            features.append(float(metrics['network_density']))
        else:
            if doc:
                network_lemmas = {'connected', 'network', 'together', 'collaborate', 'unite'}
                network_count = sum(1 for token in doc if token.lemma_ in network_lemmas)
                features.append(min(1.0, network_count / n_words * 10))
            else:
                features.append(0.3)
        
        # 4. Influencer presence
        if 'influencer_count' in metrics:
            features.append(min(1.0, float(metrics['influencer_count']) / 10))
        else:
            if doc:
                influencer_lemmas = {'leader', 'influencer', 'ambassador', 'champion', 'advocate'}
                influencer_count = sum(1 for token in doc if token.lemma_ in influencer_lemmas)
                features.append(min(1.0, influencer_count / n_words * 10))
            else:
                features.append(0.3)
        
        # 5. Growth rate
        if 'growth_rate' in metrics:
            features.append(min(1.0, float(metrics['growth_rate']) / 2))  # 200% growth = 1.0
        else:
            if doc:
                growth_lemmas = {'grow', 'expand', 'increase', 'surge', 'explode', 'boom'}
                growth_count = sum(1 for token in doc if token.lemma_ in growth_lemmas)
                features.append(min(1.0, growth_count / n_words * 10))
            else:
                features.append(0.4)
        
        # 6. Retention rate
        if 'retention_rate' in metrics:
            features.append(float(metrics['retention_rate']))
        else:
            if doc:
                retention_lemmas = {'stay', 'remain', 'loyal', 'committed', 'dedicated', 'retain'}
                retention_count = sum(1 for token in doc if token.lemma_ in retention_lemmas)
                features.append(min(1.0, retention_count / n_words * 10))
            else:
                features.append(0.5)
        
        # 7. Community health
        if doc:
            positive_lemmas = {'supportive', 'welcoming', 'friendly', 'helpful', 'positive', 'constructive'}
            negative_lemmas = {'toxic', 'hostile', 'negative', 'divisive', 'conflict', 'drama'}
            
            positive = sum(1 for token in doc if token.lemma_ in positive_lemmas)
            negative = sum(1 for token in doc if token.lemma_ in negative_lemmas)
            
            total = positive + negative
            if total > 0:
                health = positive / total
                features.append(health)
            else:
                features.append(0.6)
        else:
            features.append(0.6)
        
        # 8. Subgroup diversity
        if doc:
            diversity_lemmas = {'diverse', 'variety', 'different', 'various', 'multicultural', 'inclusive'}
            diversity_count = sum(1 for token in doc if token.lemma_ in diversity_lemmas)
            features.append(min(1.0, diversity_count / n_words * 10))
        else:
            features.append(0.4)
        
        # 9. Geographic distribution
        if doc:
            # Count geographic entities
            geo_entities = sum(1 for ent in doc.ents if ent.label_ in ['GPE', 'LOC'])
            features.append(min(1.0, geo_entities / n_words * 20))
        else:
            features.append(0.3)
        
        # 10. Cross-community bridges
        if doc:
            bridge_lemmas = {'connect', 'bridge', 'link', 'unite', 'bring together', 'integrate'}
            bridge_count = sum(1 for token in doc if token.lemma_ in bridge_lemmas)
            features.append(min(1.0, bridge_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 11. Community governance
        if doc:
            governance_lemmas = {'govern', 'rule', 'moderate', 'manage', 'organize', 'coordinate', 'lead'}
            governance_count = sum(1 for token in doc if token.lemma_ in governance_lemmas)
            features.append(min(1.0, governance_count / n_words * 10))
        else:
            features.append(0.4)
        
        # 12. Shared values clarity
        if doc:
            values_lemmas = {'value', 'believe', 'principle', 'mission', 'vision', 'core'}
            values_count = sum(1 for token in doc if token.lemma_ in values_lemmas)
            features.append(min(1.0, values_count / n_words * 5))
        else:
            features.append(0.4)
        
        # 13. Collective action potential
        if doc:
            collective_lemmas = {'together', 'collective', 'unite', 'mobilize', 'organize', 'movement'}
            collective_count = sum(1 for token in doc if token.lemma_ in collective_lemmas)
            features.append(min(1.0, collective_count / n_words * 10))
        else:
            features.append(0.3)
        
        # 14. Network effects strength
        if 'network_effects' in metrics:
            features.append(float(metrics['network_effects']))
        else:
            if doc:
                network_lemmas = {'network', 'effect', 'viral', 'spread', 'multiply', 'exponential'}
                network_count = sum(1 for token in doc if token.lemma_ in network_lemmas)
                features.append(min(1.0, network_count / n_words * 10))
            else:
                features.append(0.3)
        
        # 15. Viral coefficient
        if 'viral_coefficient' in metrics:
            features.append(min(1.0, float(metrics['viral_coefficient'])))
        else:
            if doc:
                viral_lemmas = {'viral', 'spread', 'share', 'recommend', 'invite', 'refer'}
                viral_count = sum(1 for token in doc if token.lemma_ in viral_lemmas)
                features.append(min(1.0, viral_count / n_words * 10))
            else:
                features.append(0.3)
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'community_size',
            'community_engagement',
            'community_network_density',
            'community_influencer_presence',
            'community_growth_rate',
            'community_retention_rate',
            'community_health',
            'community_subgroup_diversity',
            'community_geographic_distribution',
            'community_cross_bridges',
            'community_governance',
            'community_shared_values',
            'community_collective_action',
            'community_network_effects',
            'community_viral_coefficient'
        ])

