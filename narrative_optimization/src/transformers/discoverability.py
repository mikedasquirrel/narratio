"""Discoverability Transformer - SEO and namespace (12 features)"""
from typing import List
import numpy as np
import re
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string

class DiscoverabilityTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(narrative_id="discoverability", description="Search and discoverability metrics")
    
    def fit(self, X, y=None):
        self.corpus = set()
        for text in X:
            self.corpus.update(re.findall(r'\b\w+\b', text.lower()))
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        self._validate_fitted()
        return np.array([self._extract(text) for text in X])
    
    def _extract(self, text: str) -> np.ndarray:
        words = re.findall(r'\b\w+\b', text.lower())
        primary = words[0] if words else ""
        features = []
        
        # Uniqueness score (proxy for search rank)
        uniqueness = 1.0 / (1.0 + sum(1 for w in self.corpus if w == primary))
        features.append(uniqueness)
        
        # Length score (5-10 chars optimal for search)
        length_opt = 1.0 if 5 <= len(primary) <= 10 else 0.5
        features.append(length_opt)
        
        # Autocomplete likelihood (short, common start)
        autocomplete = 1.0 if len(primary) <= 6 else 0.5
        features.append(autocomplete)
        
        # Namespace pollution (how many similar)
        similar = sum(1 for w in self.corpus if w[:3] == primary[:3])
        pollution = similar / max(1, len(self.corpus))
        features.append(pollution)
        
        # Domain availability (heuristic: short + unique)
        domain_avail = uniqueness * (1.0 if len(primary) <= 12 else 0.5)
        features.append(domain_avail)
        
        # Social handle availability (short + unique)
        handle_avail = uniqueness * (1.0 if len(primary) <= 15 else 0.3)
        features.append(handle_avail)
        
        # SEO friendliness (unique + length + no special chars)
        seo = uniqueness * length_opt * (1.0 if primary.isalnum() else 0.5)
        features.append(seo)
        
        # Memorability for typing
        typing_ease = 1.0 / (1.0 + len(primary) / 10.0)
        features.append(typing_ease)
        
        # Brandability (unique + short + memorable)
        brandability = (uniqueness + length_opt + typing_ease) / 3.0
        features.append(brandability)
        
        # Trademark likelihood (unique + distinctive)
        trademark = uniqueness * (1.0 - pollution)
        features.append(trademark)
        
        # Overall discoverability
        discoverability = (uniqueness + seo + brandability) / 3.0
        features.append(discoverability)
        
        # Search competition (inverse uniqueness)
        competition = 1.0 - uniqueness
        features.append(competition)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        return ['search_uniqueness', 'length_optimality', 'autocomplete_likelihood', 'namespace_pollution',
                'domain_availability', 'social_handle_availability', 'seo_friendliness', 'typing_ease',
                'brandability', 'trademark_likelihood', 'overall_discoverability', 'search_competition']

