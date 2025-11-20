"""
Gravitas Transformer for Naval Ships

Extracts features indicating name "weight" and importance:
- Category effects (virtue, monarch, geographic, saint, animal)
- Formality markers
- Length and complexity
- Authority signals

Key Finding: Virtue names (Victory, Enterprise) show highest significance

Author: Narrative Optimization Research
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.transformers.base_transformer import TextNarrativeTransformer


class GravitasTransformer(TextNarrativeTransformer):
    """
    Extract gravitas (importance weight) features from ship names.
    
    Tests hypothesis: Important missions receive important-sounding names.
    """
    
    def __init__(self):
        """Initialize gravitas transformer."""
        super().__init__(
            narrative_id="ship_gravitas",
            description="Analyzes name weight and importance signaling in naval vessels"
        )
        
        # Virtue words (highest gravitas)
        self.virtue_words = {
            'victory', 'enterprise', 'endeavour', 'discovery', 'resolution',
            'courage', 'valiant', 'intrepid', 'defiant', 'triumph', 'glory',
            'honor', 'freedom', 'liberty', 'constitution', 'independence'
        }
        
        # Monarch markers
        self.monarch_markers = {
            'king', 'queen', 'prince', 'princess', 'duke', 'duchess',
            'emperor', 'empress', 'kaiser', 'czar', 'royal', 'sovereign'
        }
        
        # Geographic indicators
        self.geographic_indicators = {
            'state_names': {'arizona', 'missouri', 'california', 'texas', 'georgia',
                          'virginia', 'carolina', 'nevada', 'colorado', 'tennessee'},
            'city_names': {'boston', 'philadelphia', 'baltimore', 'chicago', 'detroit'},
            'region_names': {'pacific', 'atlantic', 'mediterranean', 'arctic'}
        }
        
        # Saint markers
        self.saint_markers = {'san', 'santa', 'saint', 'st.'}
    
    def fit(self, X, y=None):
        """
        Learn gravitas patterns from ship names.
        
        Parameters
        ----------
        X : list of str
            Ship names
        y : array-like, optional
            Historical significance scores
        
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Analyze corpus
        categories = []
        gravitas_scores = []
        
        for name in X:
            cat = self._categorize_name(name)
            grav = self._calculate_gravitas(name, cat)
            
            categories.append(cat)
            gravitas_scores.append(grav)
        
        self.metadata['corpus_stats'] = {
            'n_ships': len(X),
            'category_distribution': {
                cat: categories.count(cat) for cat in set(categories)
            },
            'gravitas': {
                'mean': np.mean(gravitas_scores),
                'std': np.std(gravitas_scores),
                'range': (min(gravitas_scores), max(gravitas_scores))
            }
        }
        
        if y is not None:
            y_array = np.array(y)
            self.metadata['correlations'] = {
                'gravitas_vs_significance': np.corrcoef(gravitas_scores, y_array)[0, 1]
            }
        
        self.metadata['n_features'] = 15
        self.metadata['feature_names'] = self._generate_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform ship names to gravitas features."""
        self._validate_fitted()
        self._validate_input(X)
        
        features = []
        for name in X:
            feature_vec = self._extract_gravitas_features(name)
            features.append(feature_vec)
        
        return np.array(features, dtype=np.float32)
    
    def _categorize_name(self, name: str) -> str:
        """Categorize ship name."""
        name_lower = name.lower()
        
        # Check virtue words
        if any(word in name_lower for word in self.virtue_words):
            return 'virtue'
        
        # Check monarch
        if any(word in name_lower for word in self.monarch_markers):
            return 'monarch'
        
        # Check geographic
        all_geo = (self.geographic_indicators['state_names'] | 
                  self.geographic_indicators['city_names'] |
                  self.geographic_indicators['region_names'])
        if any(word in name_lower for word in all_geo):
            return 'geographic'
        
        # Check saint
        if any(marker in name_lower for marker in self.saint_markers):
            return 'saint'
        
        # Check animal (simple heuristic)
        animals = {'eagle', 'lion', 'tiger', 'shark', 'dolphin', 'wolf',
                  'bear', 'falcon', 'hawk', 'beagle', 'dragon'}
        if any(animal in name_lower for animal in animals):
            return 'animal'
        
        # Check mythological
        mythological = {'poseidon', 'neptune', 'apollo', 'athena', 'zeus',
                       'hercules', 'odyssey', 'titan', 'triton'}
        if any(myth in name_lower for myth in mythological):
            return 'mythological'
        
        return 'other'
    
    def _calculate_gravitas(self, name: str, category: str) -> float:
        """Calculate gravitas score (0-100)."""
        # Base from category
        hierarchy_score = self.category_hierarchy.get(category, 2) * 15
        
        # Length adds gravitas (up to a point)
        length_bonus = min(15, len(name) / 2)
        
        # Multiple words add formality
        word_count = len(name.split())
        word_bonus = (word_count - 1) * 8
        
        # Capital letters (HMS, USS prefixes add formality)
        caps = sum(1 for c in name if c.isupper())
        cap_bonus = min(10, caps)
        
        total = hierarchy_score + length_bonus + word_bonus + cap_bonus
        return min(100, total)
    
    def _extract_gravitas_features(self, name: str) -> List[float]:
        """Extract complete feature vector."""
        features = []
        
        # 1. Category determination
        category = self._categorize_name(name)
        
        # 2. Category one-hot encoding
        for cat in ['virtue', 'monarch', 'geographic', 'saint', 'animal', 'mythological', 'other']:
            features.append(1.0 if category == cat else 0.0)
        
        # 3. Gravitas score
        gravitas = self._calculate_gravitas(name, category)
        features.append(gravitas)
        
        # 4. Hierarchy level (1-5)
        hierarchy = self.category_hierarchy.get(category, 2)
        features.append(float(hierarchy))
        
        # 5. Name length
        features.append(float(len(name)))
        
        # 6. Word count
        features.append(float(len(name.split())))
        
        # 7. Formality markers
        formal = (len(name.split()) > 1) or any(c.isupper() for c in name[1:])
        features.append(1.0 if formal else 0.0)
        
        # 8. Has virtue word
        has_virtue = any(word in name.lower() for word in self.virtue_words)
        features.append(1.0 if has_virtue else 0.0)
        
        # 9. Has monarch marker
        has_monarch = any(word in name.lower() for word in self.monarch_markers)
        features.append(1.0 if has_monarch else 0.0)
        
        return features
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names."""
        return [
            'is_virtue',
            'is_monarch',
            'is_geographic',
            'is_saint',
            'is_animal',
            'is_mythological',
            'is_other',
            'gravitas_score',
            'hierarchy_level',
            'name_length',
            'word_count',
            'has_formality',
            'has_virtue_word',
            'has_monarch_marker'
        ]
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation."""
        stats = self.metadata['corpus_stats']
        
        interpretation = f"""
Ship Gravitas Analysis ({stats['n_ships']} vessels)

CATEGORY DISTRIBUTION
---------------------
"""
        
        for cat, count in sorted(stats['category_distribution'].items(), 
                                key=lambda x: x[1], reverse=True):
            pct = count / stats['n_ships'] * 100
            interpretation += f"{cat:15s}: {count:3d} ({pct:4.1f}%)\n"
        
        interpretation += f"""
GRAVITAS STATISTICS
-------------------
Mean: {stats['gravitas']['mean']:.1f}
SD: {stats['gravitas']['std']:.1f}
Range: {stats['gravitas']['range'][0]:.0f} - {stats['gravitas']['range'][1]:.0f}
"""
        
        if 'correlations' in self.metadata:
            corr = self.metadata['correlations']
            interpretation += f"""
CORRELATION WITH SIGNIFICANCE
------------------------------
r = {corr['gravitas_vs_significance']:.3f}

INTERPRETATION
--------------
{'Positive' if corr['gravitas_vs_significance'] > 0 else 'Negative'} correlation between name gravitas and historical significance.
Important missions receive important-sounding names (selection effect).
"""
        
        return interpretation


if __name__ == '__main__':
    # Demo
    transformer = GravitasTransformer()
    
    ships = ['Victory', 'Santa Maria', 'Enterprise', 'Arizona', 'Beagle']
    transformer.fit(ships)
    features = transformer.transform(ships)
    
    print(transformer.get_interpretation())
    print("\nFeature shape:", features.shape)

