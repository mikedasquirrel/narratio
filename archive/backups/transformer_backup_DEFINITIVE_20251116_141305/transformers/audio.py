"""
Audio Transformer

Captures audio/announcement properties: chantability, broadcast clarity,
announcer-friendliness. Critical for sports, entertainment domains.

Features (8): Chant-ability, broadcast clarity, vocal range, crowd resilience
"""

from typing import List
import numpy as np
import re
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string


class AudioTransformer(NarrativeTransformer):
    """Analyzes audio and announcement properties."""
    
    def __init__(self):
        super().__init__(
            narrative_id="audio",
            description="Audio properties: chantability, broadcast clarity, announcer-friendliness"
        )
    
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        self._validate_fitted()
        return np.array([self._extract_audio_features(text) for text in X])
    
    def _extract_audio_features(self, text: str) -> np.ndarray:
        """Extract 8 audio features."""
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        words = re.findall(r'\b\w+\b', text.lower())
        primary_word = words[0] if words else ""
        
        features = []
        
        # Chantability (short, repetitive syllables)
        syllables = self._count_syllables(primary_word)
        chantability = 1.0 / (1.0 + syllables) if syllables > 0 else 0
        features.append(chantability)
        
        # Broadcast clarity (distinct phonemes, no ambiguity)
        if primary_word:
            unique_chars = len(set(primary_word))
            clarity = unique_chars / len(primary_word)
        else:
            clarity = 0.5
        features.append(clarity)
        
        # Announcer-friendliness (easy to pronounce)
        consonant_clusters = self._count_clusters(primary_word)
        announcer_friendly = 1.0 / (1.0 + consonant_clusters)
        features.append(announcer_friendly)
        
        # Vocal range required (phonetic span)
        vowels = set('aeiou')
        vowel_variety = len(set(c for c in primary_word if c in vowels))
        vocal_range = vowel_variety / 5.0  # Normalize
        features.append(vocal_range)
        
        # Crowd noise resilience (plosives audible in noise)
        plosives = set('ptkbdg')
        plosive_ratio = sum(1 for c in primary_word if c in plosives) / max(1, len(primary_word))
        features.append(plosive_ratio)
        
        # Rhythm for cheering (syllable regularity)
        rhythm = 1.0 if syllables in [1, 2, 3] else 0.5
        features.append(rhythm)
        
        # Elongation potential (can stretch for effect: "GOOOOAL")
        elongatable = 1.0 if any(c in 'aeiou' for c in primary_word[-3:]) else 0.0
        features.append(elongatable)
        
        # Overall audio score
        audio_score = (chantability + clarity + announcer_friendly + features[4]) / 4
        features.append(audio_score)
        
        return np.array(features)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables."""
        vowels = 'aeiou'
        count = sum(1 for i, c in enumerate(word) if c in vowels and (i == 0 or word[i-1] not in vowels))
        return max(1, count)
    
    def _count_clusters(self, word: str) -> int:
        """Count consonant clusters."""
        vowels = set('aeiou')
        clusters = 0
        cluster_size = 0
        for c in word:
            if c not in vowels:
                cluster_size += 1
                if cluster_size == 2:
                    clusters += 1
            else:
                cluster_size = 0
        return clusters
    
    def get_feature_names(self) -> List[str]:
        return ['chantability', 'broadcast_clarity', 'announcer_friendliness', 'vocal_range_required',
                'crowd_noise_resilience', 'rhythm_for_cheering', 'elongation_potential', 'overall_audio_score']

