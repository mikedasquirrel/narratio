"""
Crossmodal Transformer

Captures synesthetic associations (sound → color, taste, texture, etc.).
Validated: Bouba/kiki effect, sound symbolism research.

Features (12): Synesthetic color, taste, texture, temperature, weight, shape associations
"""

from typing import List
import numpy as np
import re
from .base import NarrativeTransformer


class CrossmodalTransformer(NarrativeTransformer):
    """Analyzes crossmodal/synesthetic associations."""
    
    def __init__(self):
        super().__init__(
            narrative_id="crossmodal",
            description="Crossmodal synesthesia: sound symbolism and cross-sensory associations"
        )
        
        # Bouba/kiki research: Round sounds vs angular sounds
        self.round_sounds = set('lmnrw')  # Sonorants
        self.angular_sounds = set('ptkbdg')  # Plosives
    
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        self._validate_fitted()
        return np.array([self._extract_crossmodal_features(text) for text in X])
    
    def _extract_crossmodal_features(self, text: str) -> np.ndarray:
        """Extract 12 crossmodal features."""
        words = re.findall(r'\b\w+\b', text.lower())
        primary_word = words[0] if words else ""
        
        features = []
        
        # Round vs angular (bouba/kiki)
        if primary_word:
            round_count = sum(1 for c in primary_word if c in self.round_sounds)
            angular_count = sum(1 for c in primary_word if c in self.angular_sounds)
            total = round_count + angular_count
            roundness = round_count / total if total > 0 else 0.5
        else:
            roundness = 0.5
        features.append(roundness)
        
        # Taste associations (validated: maluma=sweet, takete=sour)
        sweet_sounds = ['m', 'l', 'n']  # Sonorants = sweet
        sour_sounds = ['t', 'k', 's']  # Sharp = sour
        sweet_count = sum(1 for c in primary_word if c in sweet_sounds)
        sour_count = sum(1 for c in primary_word if c in sour_sounds)
        total_taste = sweet_count + sour_count
        sweetness = sweet_count / total_taste if total_taste > 0 else 0.5
        features.append(sweetness)
        
        # Texture (smooth vs rough)
        smooth_phonemes = set('lrsm')
        rough_phonemes = set('kgtzx')
        smooth = sum(1 for c in primary_word if c in smooth_phonemes)
        rough = sum(1 for c in primary_word if c in rough_phonemes)
        texture = smooth / max(1, smooth + rough)
        features.append(texture)
        
        # Temperature (warm vs cool)
        warm_phonemes = set('mrwl')  # Soft, flowing
        cool_phonemes = set('stzfv')  # Sharp, hissing
        warm = sum(1 for c in primary_word if c in warm_phonemes)
        cool = sum(1 for c in primary_word if c in cool_phonemes)
        temperature = warm / max(1, warm + cool)
        features.append(temperature)
        
        # Weight (heavy vs light)
        heavy_sounds = set('bpdtkg')  # Plosives = heavy
        light_sounds = set('fvs')  # Fricatives = light
        heavy = sum(1 for c in primary_word if c in heavy_sounds)
        light = sum(1 for c in primary_word if c in light_sounds)
        weight = heavy / max(1, heavy + light)
        features.append(weight)
        
        # Size (large vs small)
        large_vowels = set('ao')  # Open, back = large
        small_vowels = set('ie')  # Close, front = small
        large_v = sum(1 for c in primary_word if c in large_vowels)
        small_v = sum(1 for c in primary_word if c in small_vowels)
        size = large_v / max(1, large_v + small_v)
        features.append(size)
        
        # Speed (fast vs slow)
        fast_sounds = set('ptkfs')  # Quick release
        slow_sounds = set('mnrl')  # Continuants
        fast = sum(1 for c in primary_word if c in fast_sounds)
        slow = sum(1 for c in primary_word if c in slow_sounds)
        speed = fast / max(1, fast + slow)
        features.append(speed)
        
        # Brightness (bright vs dark)
        bright_vowels = set('ie')  # High, front = bright
        dark_vowels = set('ou')  # Low, back = dark
        bright_v = sum(1 for c in primary_word if c in bright_vowels)
        dark_v = sum(1 for c in primary_word if c in dark_vowels)
        brightness = bright_v / max(1, bright_v + dark_v)
        features.append(brightness)
        
        # Hardness (hard vs soft)
        hard_cons = set('ptkbdg')
        soft_cons = set('mnlrw')
        hard = sum(1 for c in primary_word if c in hard_cons)
        soft = sum(1 for c in primary_word if c in soft_cons)
        hardness = hard / max(1, hard + soft)
        features.append(hardness)
        
        # Overall synesthetic strength
        variances = [roundness, sweetness, texture, temperature, weight, size, speed, brightness, hardness]
        # High variance from 0.5 = strong synesthetic associations
        synesthetic_strength = np.mean([abs(v - 0.5) for v in variances]) * 2
        features.append(synesthetic_strength)
        
        # Dominant crossmodal association
        associations = [('round', roundness), ('sweet', sweetness), ('smooth', texture), 
                       ('warm', temperature), ('heavy', weight)]
        max_assoc = max(associations, key=lambda x: abs(x[1] - 0.5))
        features.append(max_assoc[1])
        
        # Crossmodal consistency (do all associations align?)
        consistency = 1.0 - np.std(variances)
        features.append(consistency)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        return ['bouba_kiki_roundness', 'taste_sweetness', 'texture_smoothness', 'temperature_warmth',
                'weight_heaviness', 'size_largeness', 'speed_fastness', 'brightness_level', 'hardness_level',
                'synesthetic_strength', 'dominant_association', 'crossmodal_consistency']

