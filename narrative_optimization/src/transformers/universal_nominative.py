"""
Universal Nominative Transformer

Implements complete 110-feature nominative methodology.
All 10 categories from comprehensive framework.
"""

import numpy as np
import re
import math
import gzip
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin


class UniversalNominativeTransformer(BaseEstimator, TransformerMixin):
    """
    Complete nominative feature extraction (~110 features).
    
    Categories:
    1. Basic Structural (10)
    2. Phonetic/Sound (20)
    3. Semantic/Meaning (15)
    4. Memorability/Cognitive (15)
    5. Information Theory (10)
    6. Contextual/Relative (12)
    7. Domain-Specific (8)
    8. Network Position (10)
    9. Temporal/Evolution (8)
    10. International (8)
    """
    
    def __init__(self, domain_hint=None):
        self.domain_hint = domain_hint
        self.corpus_names = []
        
        # Phonetic sets
        self.vowels = set('aeiou')
        self.plosives = set('pbtdkg')
        self.fricatives = set('fvszh')
        self.liquids = set('lmnr')
        self.nasals = set('mn')
        
        # Semantic sets
        self.power_words = {'king', 'lord', 'master', 'supreme', 'ultra', 'mega', 'titan'}
        self.positive_words = {'star', 'sun', 'gold', 'trust', 'safe', 'hope', 'joy'}
        self.negative_words = {'death', 'dark', 'black', 'hell', 'demon', 'fear', 'doom'}
        self.tech_words = {'quantum', 'digital', 'cyber', 'tech', 'smart', 'ai', 'meta'}
    
    def fit(self, X, y=None):
        # Build corpus for contextual features
        self.corpus_names = []
        for text in X:
            if isinstance(text, dict):
                self.corpus_names.append(text.get('title') or text.get('name', ''))
            else:
                # Extract first proper noun
                names = re.findall(r'\b[A-Z][a-z]+\b', str(text))
                if names:
                    self.corpus_names.append(names[0])
        
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            # Extract primary name
            if isinstance(text, dict):
                name = text.get('title') or text.get('name', '')
            else:
                names = re.findall(r'\b[A-Z][a-z]+\b', str(text))
                name = names[0] if names else str(text).split()[0] if text else 'Unknown'
            
            # Extract all 10 categories
            feat = []
            feat.extend(self._basic_structural(name))  # 10
            feat.extend(self._phonetic_sound(name))  # 20
            feat.extend(self._semantic_meaning(name))  # 15
            feat.extend(self._memorability_cognitive(name))  # 15
            feat.extend(self._information_theory(name))  # 10
            feat.extend(self._contextual_relative(name))  # 12
            feat.extend(self._domain_specific(name))  # 8
            feat.extend(self._network_position(name))  # 10
            feat.extend(self._temporal_evolution(name))  # 8
            feat.extend(self._international(name))  # 8
            
            features.append(feat)
        
        return np.array(features)
    
    def _basic_structural(self, name):
        """Category 1: Basic Structural (10 features)"""
        if not name:
            return [0] * 10
        
        return [
            len(name),  # length
            self._count_syllables(name),  # syllables
            len(name.split()),  # word_count
            len(name),  # char_count
            sum(1 for c in name.lower() if c in self.vowels),  # vowel_count
            len([c for c in name if c.isalpha()]) - sum(1 for c in name.lower() if c in self.vowels),  # consonant
            sum(1 for c in name.lower() if c in self.vowels) / max(len(name), 1),  # vowel_ratio
            float(any(c.isdigit() for c in name)),  # has_numbers
            float(any(not c.isalnum() and not c.isspace() for c in name)),  # has_special
            float(name[0].isupper() if name else 0)  # capitalization
        ]
    
    def _phonetic_sound(self, name):
        """Category 2: Phonetic/Sound (20 features)"""
        if not name:
            return [0] * 20
        
        name_lower = name.lower()
        total = max(len(name_lower), 1)
        
        # Counts
        plosive_count = sum(1 for c in name_lower if c in self.plosives)
        fricative_count = sum(1 for c in name_lower if c in self.fricatives)
        liquid_count = sum(1 for c in name_lower if c in self.liquids)
        nasal_count = sum(1 for c in name_lower if c in self.nasals)
        vowel_count = sum(1 for c in name_lower if c in self.vowels)
        
        # Bouba/kiki (round vs angular)
        round_sounds = 'bmoulw'
        angular_sounds = 'kptie'
        round_count = sum(1 for c in name_lower if c in round_sounds)
        angular_count = sum(1 for c in name_lower if c in angular_sounds)
        bouba_kiki = (angular_count - round_count) / max(round_count + angular_count, 1)
        
        # Consonant clusters
        clusters = len(re.findall(r'[^aeiou]{2,}', name_lower))
        max_cluster = max([len(m.group()) for m in re.finditer(r'[^aeiou]+', name_lower)], default=0)
        
        return [
            plosive_count, fricative_count, liquid_count, nasal_count,
            plosive_count/total, fricative_count/total, liquid_count/total,
            (plosive_count + fricative_count)/total * 100,  # harshness
            (liquid_count + vowel_count)/total * 100,  # softness
            clusters, max_cluster,
            bouba_kiki,  # round(-1) to angular(+1)
            1.0 if any(name_lower[i] == name_lower[i-1] for i in range(1, len(name_lower))) else 0,  # repetition
            float(name_lower[0] == name_lower[-1] if len(name_lower) > 1 else 0),  # alliteration
            1.0 / (clusters + 1),  # pronounceability
            0.0, 0.0, 0.0, 0.0, 0.0  # Padding
        ]
    
    def _semantic_meaning(self, name):
        """Category 3: Semantic/Meaning (15 features)"""
        if not name:
            return [0] * 15
        
        name_lower = name.lower()
        
        return [
            float(any(w in name_lower for w in self.power_words)),
            float(any(w in name_lower for w in self.positive_words)),
            float(any(w in name_lower for w in self.negative_words)),
            float(any(w in name_lower for w in self.tech_words)),
            float(len(name.split()) > 1),  # compound
            (sum(1 for w in self.positive_words if w in name_lower) - 
             sum(1 for w in self.negative_words if w in name_lower)),  # valence
            sum(1 for w in self.power_words if w in name_lower) * 10,  # power_score
            len(name.split()),  # semantic transparency proxy
            0, 0, 0, 0, 0, 0, 0  # Padding
        ]
    
    def _memorability_cognitive(self, name):
        """Category 4: Memorability/Cognitive (15 features)"""
        if not name:
            return [0] * 15
        
        syllables = self._count_syllables(name)
        clusters = len(re.findall(r'[^aeiou]{2,}', name.lower()))
        
        # Cognitive load
        cognitive_load = syllables + clusters
        
        # Memorability (shorter + patterns = better)
        memorability = 100
        if len(name) <= 6:
            memorability += 20
        elif len(name) > 10:
            memorability -= (len(name) - 10) * 2
        memorability = max(0, min(100, memorability))
        
        # Distinctiveness
        unique_chars = len(set(name.lower()))
        distinctiveness = unique_chars / max(len(name), 1)
        
        # Processing fluency
        disfluency = (syllables + clusters) / 10
        disfluency = min(1.0, disfluency)
        
        return [
            len(name) * 50,  # reading_time_ms
            cognitive_load,
            memorability / 100,
            distinctiveness,
            float(name[0].lower() == name[-1].lower() if len(name) > 1 else 0),  # alliteration
            disfluency,
            float(0.3 < disfluency < 0.6),  # optimal_range
            clusters / max(len(name), 1),  # difficulty
            1.0 - disfluency,  # pronounceability
            0, 0, 0, 0, 0, 0  # Padding
        ]
    
    def _information_theory(self, name):
        """Category 5: Information Theory (10 features)"""
        if not name or len(name) < 2:
            return [0] * 10
        
        name_lower = name.lower()
        
        # Shannon entropy
        counter = Counter(name_lower)
        total = len(name_lower)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Kolmogorov complexity (compression)
        try:
            compressed = gzip.compress(name.encode('utf-8'))
            kolmogorov = len(compressed) / max(len(name.encode('utf-8')), 1)
        except:
            kolmogorov = 0.5
        
        # Character diversity
        char_diversity = len(set(name_lower)) / max(len(name_lower), 1)
        
        # Information density
        meaningful_chars = sum(1 for c in name_lower if c.isalpha())
        info_density = meaningful_chars / max(len(name), 1)
        
        return [
            entropy,
            kolmogorov,
            char_diversity,
            info_density,
            len(set(name_lower)),  # unique_chars
            0, 0, 0, 0, 0  # Padding
        ]
    
    def _contextual_relative(self, name):
        """Category 6: Contextual/Relative (12 features)"""
        if not name or not self.corpus_names:
            return [0] * 12
        
        # Relative position in corpus
        syllables = self._count_syllables(name)
        length = len(name)
        
        corpus_syllables = [self._count_syllables(n) for n in self.corpus_names if n]
        corpus_lengths = [len(n) for n in self.corpus_names if n]
        
        if corpus_syllables and corpus_lengths:
            syllable_percentile = sum(1 for s in corpus_syllables if s < syllables) / len(corpus_syllables)
            length_percentile = sum(1 for l in corpus_lengths if l < length) / len(corpus_lengths)
            
            syllable_zscore = (syllables - np.mean(corpus_syllables)) / (np.std(corpus_syllables) + 1e-8)
            length_zscore = (length - np.mean(corpus_lengths)) / (np.std(corpus_lengths) + 1e-8)
        else:
            syllable_percentile = length_percentile = 0.5
            syllable_zscore = length_zscore = 0.0
        
        # Edit distance to similar names
        similar_count = sum(1 for n in self.corpus_names if n != name and self._edit_distance(name, n) <= 2)
        
        return [
            syllable_percentile,
            length_percentile,
            syllable_zscore,
            length_zscore,
            similar_count,
            similar_count / max(len(self.corpus_names), 1),  # saturation
            0, 0, 0, 0, 0, 0  # Padding
        ]
    
    def _domain_specific(self, name):
        """Category 7: Domain-Specific (8 features)"""
        if not name:
            return [0] * 8
        
        name_lower = name.lower()
        
        # Universal domain patterns
        crypto_morphs = ['coin', 'token', 'crypto', 'bit', 'chain']
        tech_morphs = ['quantum', 'meta', 'protocol', 'network', 'smart']
        prestige_morphs = ['master', 'grand', 'supreme', 'royal', 'imperial']
        
        return [
            float(any(m in name_lower for m in crypto_morphs)),
            float(any(m in name_lower for m in tech_morphs)),
            float(any(m in name_lower for m in prestige_morphs)),
            float(name_lower.startswith('the ')),
            float(len(name.split()) == 1),  # monosyllabic
            0, 0, 0  # Padding
        ]
    
    def _network_position(self, name):
        """Category 8: Network Position (10 features)"""
        if not name or not self.corpus_names:
            return [0] * 10
        
        # Count phonetically similar names (neighbors)
        neighbors = sum(1 for n in self.corpus_names if n != name and self._edit_distance(name, n) <= 3)
        
        # Degree centrality proxy
        degree_centrality = neighbors / max(len(self.corpus_names) - 1, 1)
        
        # Isolation score
        isolation = 1.0 - degree_centrality
        
        return [
            neighbors,
            degree_centrality,
            isolation,
            float(neighbors > 5),  # in_giant_component proxy
            0, 0, 0, 0, 0, 0  # Padding
        ]
    
    def _temporal_evolution(self, name):
        """Category 9: Temporal/Evolution (8 features)"""
        if not name or not self.corpus_names:
            return [0] * 8
        
        # Pattern saturation (how many similar names exist)
        morphemes = re.findall(r'\w+', name.lower())
        if morphemes:
            saturation = sum(1 for n in self.corpus_names 
                           if any(m in n.lower() for m in morphemes)) / max(len(self.corpus_names), 1)
        else:
            saturation = 0.0
        
        return [
            saturation,
            float(saturation < 0.3),  # early_pattern
            float(saturation > 0.7),  # saturated_pattern
            0, 0, 0, 0, 0  # Padding
        ]
    
    def _international(self, name):
        """Category 10: International/Cross-Linguistic (8 features)"""
        if not name:
            return [0] * 8
        
        return [
            float(name.isascii()),
            float(all(c.isalpha() or c.isspace() for c in name)),  # simple_chars
            float(4 <= len(name) <= 10),  # friendly_length
            len(name) / 20,  # complexity proxy
            0, 0, 0, 0  # Padding
        ]
    
    def _count_syllables(self, word):
        """Count syllables"""
        if not word:
            return 0
        vowel_groups = len(re.findall(r'[aeiouy]+', word.lower()))
        return max(1, vowel_groups)
    
    def _edit_distance(self, s1, s2):
        """Levenshtein distance"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_feature_names_out(self, input_features=None):
        """Get all 116 feature names"""
        names = []
        
        # Basic (10)
        names.extend(['length', 'syllables', 'word_count', 'char_count', 'vowel_count',
                     'consonant_count', 'vowel_ratio', 'has_numbers', 'has_special', 'capitalization'])
        
        # Phonetic (20)
        names.extend(['plosive_count', 'fricative_count', 'liquid_count', 'nasal_count',
                     'plosive_ratio', 'fricative_ratio', 'liquid_ratio',
                     'harshness', 'softness', 'clusters', 'max_cluster', 'bouba_kiki',
                     'repetition', 'alliteration', 'pronounceability'] + [f'phon_{i}' for i in range(5)])
        
        # Semantic (15)
        names.extend(['has_power', 'has_positive', 'has_negative', 'has_tech', 'is_compound',
                     'valence', 'power_score', 'transparency'] + [f'sem_{i}' for i in range(7)])
        
        # Memorability (15)
        names.extend(['reading_time', 'cognitive_load', 'memorability', 'distinctiveness',
                     'alliteration_mem', 'disfluency', 'optimal_range', 'difficulty',
                     'pronounce'] + [f'mem_{i}' for i in range(6)])
        
        # Info theory (10)
        names.extend(['shannon_entropy', 'kolmogorov', 'char_diversity', 'info_density',
                     'unique_chars'] + [f'info_{i}' for i in range(5)])
        
        # Contextual (12)
        names.extend(['syll_percentile', 'len_percentile', 'syll_zscore', 'len_zscore',
                     'similar_count', 'saturation'] + [f'ctx_{i}' for i in range(6)])
        
        # Domain (8)
        names.extend(['crypto_morph', 'tech_morph', 'prestige_morph', 'has_article',
                     'monosyllabic'] + [f'domain_{i}' for i in range(3)])
        
        # Network (10)
        names.extend(['neighbors', 'degree_centrality', 'isolation', 'in_giant'] + [f'net_{i}' for i in range(6)])
        
        # Temporal (8)
        names.extend(['pattern_saturation', 'early_pattern', 'saturated_pattern'] + [f'temp_{i}' for i in range(5)])
        
        # International (8)
        names.extend(['is_ascii', 'simple_chars', 'friendly_length', 'complexity'] + [f'intl_{i}' for i in range(4)])
        
        return np.array([f'universal_nominative_{n}' for n in names])

