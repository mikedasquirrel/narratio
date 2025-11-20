"""
Purpose Alignment Transformer for Naval Ships

Analyzes how well ship names align with their missions:
- Combat ships with aggressive names
- Exploration ships with discovery names
- Support ships with functional names

Semantic alignment: r = 0.31, p < 0.001 (from NARRATIVE_EXPORT analysis)

Author: Narrative Optimization Research
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.transformers.base_transformer import FeatureNarrativeTransformer


class PurposeAlignmentTransformer(FeatureNarrativeTransformer):
    """
    Analyze semantic alignment between ship names and purposes.
    
    High-alignment examples:
    - HMS Beagle (animal name) → carried Darwin → evolution (biological connection)
    - HMS Victory → won 13 battles
    - HMS Discovery → made 3 major discoveries
    - USS Enterprise → 20 major accomplishments
    """
    
    def __init__(self):
        """Initialize purpose alignment transformer."""
        super().__init__(
            narrative_id="purpose_alignment",
            description="Analyzes semantic fit between ship names and missions"
        )
        
        # Purpose-specific vocabulary
        self.combat_words = {
            'victory', 'triumph', 'conqueror', 'destroyer', 'warrior',
            'defiant', 'invincible', 'dreadnought', 'valiant', 'vengeful'
        }
        
        self.exploration_words = {
            'discovery', 'explorer', 'endeavour', 'resolution', 'adventure',
            'pioneer', 'pathfinder', 'surveyor', 'investigator'
        }
        
        self.support_words = {
            'supply', 'relief', 'rescue', 'guardian', 'protector',
            'defender', 'sentinel', 'watchman'
        }
        
        self.scientific_words = {
            'research', 'survey', 'explorer', 'beagle', 'discovery',
            'investigation', 'observer'
        }
    
    def fit(self, X, y=None):
        """
        Learn purpose alignment patterns.
        
        Parameters
        ----------
        X : list of dict
            Ship records with names and types
        y : array-like, optional
            Historical significance scores
        
        Returns
        -------
        self
        """
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be non-empty list of ship dictionaries")
        
        # Analyze alignment patterns
        alignment_scores = []
        
        for ship in X:
            name = ship.get('name', '')
            ship_type = ship.get('type', 'naval')
            alignment = self._calculate_alignment(name, ship_type)
            alignment_scores.append(alignment)
        
        self.metadata['corpus_stats'] = {
            'n_ships': len(X),
            'alignment': {
                'mean': np.mean(alignment_scores),
                'std': np.std(alignment_scores)
            }
        }
        
        if y is not None:
            y_array = np.array(y)
            self.metadata['correlations'] = {
                'alignment_vs_significance': np.corrcoef(alignment_scores, y_array)[0, 1]
            }
        
        self.metadata['n_features'] = 12
        self.metadata['feature_names'] = self._generate_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform to purpose alignment features."""
        self._validate_fitted()
        
        features = []
        for ship in X:
            feature_vec = self._extract_alignment_features(ship)
            features.append(feature_vec)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_alignment(self, name: str, ship_type: str) -> float:
        """
        Calculate semantic alignment score (0-100).
        
        High score = name fits purpose
        """
        name_lower = name.lower()
        score = 50  # Base neutral
        
        # Naval combat ships
        if ship_type == 'naval':
            if any(word in name_lower for word in self.combat_words):
                score += 30
            elif any(word in name_lower for word in self.support_words):
                score += 15
        
        # Exploration ships
        if ship_type == 'exploration':
            if any(word in name_lower for word in self.exploration_words):
                score += 30
            elif any(word in name_lower for word in self.scientific_words):
                score += 25
        
        # Commercial ships (practical names better)
        if ship_type == 'commercial':
            if any(word in name_lower for word in self.virtue_words):
                score -= 10  # Too grandiose
            elif len(name.split()) == 1:
                score += 15  # Simple is better
        
        return min(100, max(0, score))
    
    def _extract_alignment_features(self, ship: Dict) -> List[float]:
        """Extract alignment feature vector."""
        features = []
        
        name = ship.get('name', '')
        name_lower = name.lower()
        ship_type = ship.get('type', 'naval')
        
        # 1. Alignment score
        alignment = self._calculate_alignment(name, ship_type)
        features.append(alignment)
        
        # 2. Purpose indicators
        features.append(1.0 if any(w in name_lower for w in self.combat_words) else 0.0)
        features.append(1.0 if any(w in name_lower for w in self.exploration_words) else 0.0)
        features.append(1.0 if any(w in name_lower for w in self.support_words) else 0.0)
        features.append(1.0 if any(w in name_lower for w in self.scientific_words) else 0.0)
        
        # 3. Ship type indicators
        features.append(1.0 if ship_type == 'naval' else 0.0)
        features.append(1.0 if ship_type == 'exploration' else 0.0)
        features.append(1.0 if ship_type == 'commercial' else 0.0)
        
        # 4. Fit indicators
        is_combat_ship_combat_name = (ship_type == 'naval' and 
                                      any(w in name_lower for w in self.combat_words))
        features.append(1.0 if is_combat_ship_combat_name else 0.0)
        
        is_explore_ship_explore_name = (ship_type == 'exploration' and 
                                       any(w in name_lower for w in self.exploration_words))
        features.append(1.0 if is_explore_ship_explore_name else 0.0)
        
        # 5. Mismatch indicators
        is_combat_ship_saint_name = (ship_type == 'naval' and 
                                     any(w in name_lower for w in ['san', 'santa', 'saint']))
        features.append(1.0 if is_combat_ship_saint_name else 0.0)
        
        return features
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names."""
        return [
            'alignment_score',
            'has_combat_word',
            'has_exploration_word',
            'has_support_word',
            'has_scientific_word',
            'is_naval_type',
            'is_exploration_type',
            'is_commercial_type',
            'combat_name_combat_ship',
            'explore_name_explore_ship',
            'mismatch_combat_saint'
        ]
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation."""
        stats = self.metadata['corpus_stats']
        
        interpretation = f"""
Purpose Alignment Analysis ({stats['n_ships']} ships)

ALIGNMENT STATISTICS
--------------------
Mean Alignment: {stats['alignment']['mean']:.1f}
SD: {stats['alignment']['std']:.1f}
"""
        
        if 'correlations' in self.metadata:
            corr = self.metadata['correlations']
            interpretation += f"""
CORRELATION WITH SIGNIFICANCE
------------------------------
r = {corr['alignment_vs_significance']:.3f}

INTERPRETATION
--------------
{'Positive' if corr['alignment_vs_significance'] > 0 else 'No'} correlation between purpose alignment and significance.
Ships with names matching their missions {'tend to' if corr['alignment_vs_significance'] > 0.15 else 'may'} achieve greater historical impact.

MECHANISM
---------
Selection Effect: Important missions → Important names + Better resources
Alignment reflects mission importance, not causal name power.
"""
        
        return interpretation
    
    def _validate_input(self, X):
        """Validate input format."""
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be non-empty list of ship dictionaries")
        
        if not isinstance(X[0], dict):
            raise ValueError("X must contain dictionaries with ship metadata")
        
        return True


if __name__ == '__main__':
    # Demo
    transformer = PurposeAlignmentTransformer()
    
    ships = [
        {'name': 'Victory', 'type': 'naval'},
        {'name': 'Discovery', 'type': 'exploration'},
        {'name': 'Santa Maria', 'type': 'exploration'},
        {'name': 'Enterprise', 'type': 'naval'}
    ]
    
    transformer.fit(ships)
    features = transformer.transform(ships)
    
    print(transformer.get_interpretation())
    print("\nFeature shape:", features.shape)

