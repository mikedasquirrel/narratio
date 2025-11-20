"""
Multi-Scale Transformer

UNIVERSAL transformer capturing narratives at nested scales.

Works across ALL domains (like DNA analysis across species):

MOVIES:
- Industry (macro) → Studio (meso) → Film (micro) → Scene (nano)

NBA:
- Season (macro) → Series (meso) → Game (micro) → Quarter (nano)

STARTUPS:
- Sector (macro) → Company (meso) → Product (micro) → Feature (nano)

POLITICS:
- Movement (macro) → Campaign (meso) → Speech (micro) → Moment (nano)

Every instance exists at multiple scales with:
- Names at each level
- Stories at each level
- Gravitational forces across scales
"""

import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Any


class MultiScaleTransformer(BaseEstimator, TransformerMixin):
    """
    Extract features at multiple nested scales.
    
    Universal across domains - extracts:
    1. Macro context (industry, season, sector)
    2. Meso context (category, series, subcategory)
    3. Micro instance (the entity itself)
    4. Nano details (moments, components, specifics)
    
    ~40 features capturing scale-dependent patterns
    """
    
    def __init__(self):
        """Initialize universal multi-scale extractor"""
        # Scale indicators (universal patterns)
        self.macro_indicators = ['industry', 'sector', 'market', 'season', 'era', 'movement', 'trend']
        self.meso_indicators = ['category', 'genre', 'series', 'type', 'class', 'division']
        self.nano_indicators = ['moment', 'scene', 'feature', 'detail', 'aspect', 'element']
    
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """
        Transform to multi-scale features.
        
        Parameters
        ----------
        X : array-like
            Can be:
            - Dicts with hierarchical data
            - Texts with scale markers
            - Any structured instance data
            
        Returns
        -------
        features : ndarray (n_samples, 40)
            Multi-scale features
        """
        features = []
        
        for item in X:
            feat = self._extract_multi_scale(item)
            features.append(feat)
        
        return np.array(features)
    
    def _extract_multi_scale(self, item):
        """Extract features at all scales"""
        if isinstance(item, dict):
            return self._extract_from_structured(item)
        else:
            return self._extract_from_text(str(item))
    
    def _extract_from_structured(self, data):
        """Extract from structured data"""
        features = []
        
        # MACRO (industry/season level) - 10 features
        macro_text = str(data.get('season', '')) + str(data.get('genre', '')) + str(data.get('sector', ''))
        features.extend([
            len(macro_text),
            float(any(ind in macro_text.lower() for ind in self.macro_indicators)),
            self._count_proper_nouns(macro_text),
            0, 0, 0, 0, 0, 0, 0
        ])
        
        # MESO (category/series level) - 10 features
        meso_text = str(data.get('category', '')) + str(data.get('matchup', '')) + str(data.get('type', ''))
        features.extend([
            len(meso_text),
            float(any(ind in meso_text.lower() for ind in self.meso_indicators)),
            self._count_proper_nouns(meso_text),
            0, 0, 0, 0, 0, 0, 0
        ])
        
        # MICRO (instance itself) - 10 features
        micro_text = str(data.get('title', '')) + str(data.get('name', '')) + str(data.get('narrative', ''))
        features.extend([
            len(micro_text),
            len(micro_text.split()),
            self._count_proper_nouns(micro_text),
            float('vs' in micro_text or 'versus' in micro_text),  # competitive framing
            0, 0, 0, 0, 0, 0
        ])
        
        # NANO (details) - 10 features
        nano_text = str(data.get('details', '')) + str(data.get('specifics', ''))
        features.extend([
            len(nano_text),
            float(any(ind in nano_text.lower() for ind in self.nano_indicators)),
            0, 0, 0, 0, 0, 0, 0, 0
        ])
        
        return features
    
    def _extract_from_text(self, text):
        """Extract from unstructured text"""
        text_lower = text.lower()
        
        features = []
        
        # Macro indicators
        macro_score = sum(1 for ind in self.macro_indicators if ind in text_lower)
        
        # Meso indicators
        meso_score = sum(1 for ind in self.meso_indicators if ind in text_lower)
        
        # Nano indicators
        nano_score = sum(1 for ind in self.nano_indicators if ind in text_lower)
        
        # Scale breadth (how many scales mentioned)
        scale_breadth = float(macro_score > 0) + float(meso_score > 0) + float(nano_score > 0)
        
        # Build features (40 total)
        features = [
            # Macro (10)
            macro_score, macro_score / max(len(text.split()), 1), 0, 0, 0, 0, 0, 0, 0, 0,
            
            # Meso (10)
            meso_score, meso_score / max(len(text.split()), 1), 0, 0, 0, 0, 0, 0, 0, 0,
            
            # Micro (10)
            len(text), len(text.split()), self._count_proper_nouns(text), 0, 0, 0, 0, 0, 0, 0,
            
            # Nano (10)
            nano_score, nano_score / max(len(text.split()), 1), scale_breadth, 0, 0, 0, 0, 0, 0, 0
        ]
        
        return features
    
    def _count_proper_nouns(self, text):
        """Count proper nouns (capitalized words)"""
        if not text:
            return 0
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        return len(proper_nouns)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = []
        for scale in ['macro', 'meso', 'micro', 'nano']:
            names.extend([f'{scale}_{i}' for i in range(10)])
        
        return np.array([f'multi_scale_{n}' for n in names])


class MultiPerspectiveTransformer(BaseEstimator, TransformerMixin):
    """
    Extract features from multiple perspectives.
    
    UNIVERSAL across domains:
    
    MOVIES:
    - Collective (studio) → Authority (director) → Stars (actors) → Supporting (crew)
    
    NBA:
    - Collective (team) → Authority (coach) → Stars (players) → Supporting (bench)
    
    STARTUPS:
    - Collective (company) → Authority (CEO) → Stars (founders) → Supporting (team)
    
    ~50 features capturing perspective-dependent narratives
    """
    
    def __init__(self):
        """Initialize perspective extractor"""
        # Universal perspective markers
        self.collective_markers = ['we', 'us', 'our', 'organization', 'team', 'company', 'together']
        self.authority_markers = ['leader', 'director', 'coach', 'CEO', 'manager', 'command', 'decision']
        self.star_markers = ['star', 'lead', 'key', 'main', 'primary', 'top', 'best']
        self.supporting_markers = ['support', 'role', 'bench', 'crew', 'staff', 'ensemble', 'depth']
    
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """Extract perspective features"""
        features = []
        
        for item in X:
            feat = self._extract_perspectives(item)
            features.append(feat)
        
        return np.array(features)
    
    def _extract_perspectives(self, item):
        """Extract perspective features"""
        if isinstance(item, dict):
            text = str(item.get('narrative', '')) + str(item.get('overview', ''))
        else:
            text = str(item)
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Collective perspective (10 features)
        collective_count = sum(1 for word in words if word in self.collective_markers)
        collective_density = collective_count / max(len(words), 1)
        
        # Authority perspective (10 features)
        authority_count = sum(1 for word in words if word in self.authority_markers)
        authority_density = authority_count / max(len(words), 1)
        
        # Star perspective (10 features)
        star_count = sum(1 for word in words if word in self.star_markers)
        star_density = star_count / max(len(words), 1)
        
        # Supporting perspective (10 features)
        supporting_count = sum(1 for word in words if word in self.supporting_markers)
        supporting_density = supporting_count / max(len(words), 1)
        
        # Perspective balance (10 features)
        total_perspective = collective_count + authority_count + star_count + supporting_count
        
        if total_perspective > 0:
            collective_dominance = collective_count / total_perspective
            authority_dominance = authority_count / total_perspective
            star_dominance = star_count / total_perspective
            supporting_dominance = supporting_count / total_perspective
        else:
            collective_dominance = authority_dominance = star_dominance = supporting_dominance = 0.25
        
        features = [
            # Collective (10)
            collective_count, collective_density, collective_dominance, 0, 0, 0, 0, 0, 0, 0,
            
            # Authority (10)
            authority_count, authority_density, authority_dominance, 0, 0, 0, 0, 0, 0, 0,
            
            # Star (10)
            star_count, star_density, star_dominance, 0, 0, 0, 0, 0, 0, 0,
            
            # Supporting (10)
            supporting_count, supporting_density, supporting_dominance, 0, 0, 0, 0, 0, 0, 0,
            
            # Balance (10)
            total_perspective, collective_dominance, authority_dominance, star_dominance, supporting_dominance,
            1.0 - max(collective_dominance, authority_dominance, star_dominance, supporting_dominance),  # diversity
            0, 0, 0, 0
        ]
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = []
        for perspective in ['collective', 'authority', 'star', 'supporting', 'balance']:
            names.extend([f'{perspective}_{i}' for i in range(10)])
        
        return np.array([f'multi_perspective_{n}' for n in names])


class ScaleInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Capture INTERACTIONS across scales.
    
    Like quantum entanglement - scales affect each other:
    - Macro × Micro: Does industry trend affect individual success?
    - Season × Game: Does team momentum affect single game?
    - Sector × Product: Does market narrative amplify product narrative?
    
    ~30 features capturing cross-scale dynamics
    """
    
    def __init__(self):
        """Initialize interaction extractor"""
        pass
    
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """Extract scale interaction features"""
        # This would need multi-scale data
        # For now, placeholder that measures scale MIXING
        
        features = []
        
        for item in X:
            if isinstance(item, dict):
                text = str(item.get('narrative', '')) + str(item.get('overview', ''))
            else:
                text = str(item)
            
            # Measure if multiple scales are present in narrative
            text_lower = text.lower()
            
            has_macro = any(word in text_lower for word in ['season', 'era', 'industry', 'market'])
            has_meso = any(word in text_lower for word in ['series', 'category', 'genre', 'type'])
            has_micro = len(text) > 50  # Has instance detail
            has_nano = any(word in text_lower for word in ['moment', 'detail', 'specific', 'particular'])
            
            # Interaction features
            feat = [
                float(has_macro),
                float(has_meso),
                float(has_micro),
                float(has_nano),
                float(has_macro and has_micro),  # Macro-micro connection
                float(has_meso and has_micro),  # Meso-micro connection
                float(has_macro and has_nano),  # Macro-nano connection
                sum([has_macro, has_meso, has_micro, has_nano]),  # Scale breadth
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Padding to 20
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Total 30
            ]
            
            features.append(feat)
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = [f'scale_interaction_{i}' for i in range(30)]
        return np.array(names)

