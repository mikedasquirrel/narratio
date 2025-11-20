"""Cognitive Fluency Transformer - Processing ease (15 features)"""
from typing import List
import numpy as np
import re
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string

class CognitiveFluencyTransformer(NarrativeTransformer):
    def __init__(self, domain_config=None):
        super().__init__(narrative_id="cognitive_fluency", description="Cognitive processing fluency and ease")
        self.domain_config = domain_config
    
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        self._validate_fitted()
        return np.array([self._extract(text) for text in X])
    
    def _extract(self, text: str) -> np.ndarray:
        words = re.findall(r'\b\w+\b', text.lower())
        primary = words[0] if words else ""
        features = []
        
        # Reading time proxy (length × complexity)
        syllables = max(1, sum(1 for c in primary if c in 'aeiouy'))
        reading_time = len(primary) * syllables / 10.0
        features.append(reading_time)
        
        # Working memory load (chunks needed)
        chunks = max(1, len(primary) // 3)
        memory_load = chunks / 5.0
        features.append(memory_load)
        
        # Conceptual fluency (simple structure)
        simple_structure = 1.0 if syllables <= 3 else 0.5
        features.append(simple_structure)
        
        # Processing fluency (inverse complexity)
        clusters = sum(1 for i in range(len(primary)-1) if primary[i] not in 'aeiou' and primary[i+1] not in 'aeiou')
        fluency = 1.0 / (1.0 + clusters)
        features.append(fluency)
        
        # Disfluency level
        disfluency = 1.0 - fluency
        features.append(disfluency)
        
        # Sweet spot detection (moderate disfluency)
        sweet_spot = 1.0 if 0.3 < disfluency < 0.6 else 0.0
        features.append(sweet_spot)
        
        # Recognition threshold (familiarity proxy)
        common_bigrams = ['th', 'er', 'on', 'an', 're', 'he', 'in', 'ed', 'nd', 'ha']
        bigrams = [primary[i:i+2] for i in range(len(primary)-1)]
        familiar = sum(1 for bg in bigrams if bg in common_bigrams)
        recognition = familiar / max(1, len(bigrams))
        features.append(recognition)
        
        # First fixation duration proxy (initial complexity)
        first_cluster = 0
        for c in primary[:3]:
            if c not in 'aeiou':
                first_cluster += 1
        fixation = first_cluster / 3.0
        features.append(fixation)
        
        # Cognitive ease (high fluency + high recognition)
        cognitive_ease = (fluency + recognition) / 2.0
        features.append(cognitive_ease)
        
        # Processing cost (inverse ease)
        processing_cost = 1.0 - cognitive_ease
        features.append(processing_cost)
        
        # Familiarity (recognition × simple structure)
        familiarity = recognition * simple_structure
        features.append(familiarity)
        
        # Novelty (inverse familiarity)
        novelty = 1.0 - familiarity
        features.append(novelty)
        
        # Optimal complexity (in sweet spot with good fluency)
        optimal = sweet_spot * fluency
        features.append(optimal)
        
        # Inverted-U position (complexity level)
        complexity_level = disfluency  # 0=easy, 1=hard
        features.append(complexity_level)
        
        # Overall fluency score
        overall_fluency = (cognitive_ease + fluency + familiarity) / 3.0
        features.append(overall_fluency)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        return ['reading_time_proxy', 'working_memory_load', 'conceptual_fluency', 'processing_fluency',
                'disfluency_level', 'sweet_spot_indicator', 'recognition_threshold', 'first_fixation_proxy',
                'cognitive_ease', 'processing_cost', 'familiarity', 'novelty', 'optimal_complexity',
                'inverted_u_position', 'overall_fluency_score']

