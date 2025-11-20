"""
Nominative Formula Taxonomy

Implements the six-type nominative formula classification system from theory.

Each formula type captures different aspects of how names affect outcomes:
1. Phonetic: Sound patterns (harsh/soft, memorability)
2. Semantic: Meaning fields (power, speed, warmth)
3. Structural: Length, syllables, morphemes
4. Frequency: Common vs. rare names
5. Numerology: Mathematical properties
6. Hybrid: Combinations of above

Each type gets its own formula that can be tested independently,
allowing us to discover which naming dimensions matter in which domains.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import warnings

from .base_transformer import TextNarrativeTransformer


class PhoneticFormulaTransformer(TextNarrativeTransformer):
    """
    Type 1: Phonetic Formula
    
    Analyzes sound patterns in names:
    - Harsh vs. soft consonants
    - Vowel harmony
    - Phonetic memorability
    - Prosody (stress patterns)
    - Alliteration and rhyme
    """
    
    def __init__(self):
        super().__init__(
            name="phonetic_formula",
            narrative_hypothesis="Sound patterns in names affect perception and outcomes"
        )
        
        # Phonetic classifications
        self.harsh_consonants = set('kgptdbqx')
        self.soft_consonants = set('mnlrswy')
        self.plosives = set('pbtdkg')
        self.fricatives = set('fvszh')
        self.nasals = set('mn')
        self.liquids = set('lr')
        
        self.vowels = set('aeiou')
        
    def fit(self, X, y=None):
        """Learn phonetic distributions from corpus."""
        all_names = []
        
        for text in X:
            # Extract capitalized words (likely names)
            names = re.findall(r'\b[A-Z][a-z]+\b', text)
            all_names.extend(names)
        
        if all_names:
            # Learn typical phonetic patterns
            self.corpus_harsh_ratio = np.mean([
                self._count_harsh(name) / (len(name) + 1)
                for name in all_names
            ])
            
            self.corpus_vowel_ratio = np.mean([
                self._count_vowels(name) / (len(name) + 1)
                for name in all_names
            ])
        else:
            self.corpus_harsh_ratio = 0.3
            self.corpus_vowel_ratio = 0.4
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract phonetic features."""
        features = []
        
        for text in X:
            names = re.findall(r'\b[A-Z][a-z]+\b', text)
            
            if names:
                name_features = [self._extract_phonetic_features(name) for name in names]
                # Average across all names in text
                avg_features = np.mean(name_features, axis=0)
            else:
                avg_features = np.zeros(15)
            
            features.append(avg_features)
        
        return np.array(features)
    
    def _extract_phonetic_features(self, name: str) -> np.ndarray:
        """Extract phonetic features from a single name."""
        name_lower = name.lower()
        n = len(name) + 1  # Avoid division by zero
        
        features = [
            # Sound category ratios
            self._count_harsh(name_lower) / n,  # Harshness
            self._count_soft(name_lower) / n,   # Softness
            self._count_vowels(name_lower) / n,  # Vowel density
            self._count_consonants(name_lower) / n,  # Consonant density
            
            # Specific sound types
            self._count_plosives(name_lower) / n,  # Explosive sounds
            self._count_fricatives(name_lower) / n,  # Friction sounds
            self._count_nasals(name_lower) / n,  # Nasal sounds
            self._count_liquids(name_lower) / n,  # Liquid sounds
            
            # Phonetic complexity
            self._count_consonant_clusters(name_lower),  # Cluster count
            self._syllable_count(name_lower),  # Syllables
            
            # Memorability factors
            self._has_alliteration(name),  # Alliteration (binary)
            self._phonetic_distinctiveness(name_lower),  # Unique sound combo
            
            # Prosody
            self._stress_pattern_score(name_lower),  # Stress regularity
            
            # Comparative
            (self._count_harsh(name_lower) / n) - self.corpus_harsh_ratio,  # Relative harshness
            (self._count_vowels(name_lower) / n) - self.corpus_vowel_ratio,  # Relative vowel density
        ]
        
        return np.array(features)
    
    def _count_harsh(self, text: str) -> int:
        return sum(1 for c in text if c in self.harsh_consonants)
    
    def _count_soft(self, text: str) -> int:
        return sum(1 for c in text if c in self.soft_consonants)
    
    def _count_vowels(self, text: str) -> int:
        return sum(1 for c in text if c in self.vowels)
    
    def _count_consonants(self, text: str) -> int:
        return sum(1 for c in text if c.isalpha() and c not in self.vowels)
    
    def _count_plosives(self, text: str) -> int:
        return sum(1 for c in text if c in self.plosives)
    
    def _count_fricatives(self, text: str) -> int:
        return sum(1 for c in text if c in self.fricatives)
    
    def _count_nasals(self, text: str) -> int:
        return sum(1 for c in text if c in self.nasals)
    
    def _count_liquids(self, text: str) -> int:
        return sum(1 for c in text if c in self.liquids)
    
    def _count_consonant_clusters(self, text: str) -> int:
        """Count consonant clusters (2+ consonants together)."""
        clusters = re.findall(r'[bcdfghjklmnpqrstvwxyz]{2,}', text)
        return len(clusters)
    
    def _syllable_count(self, word: str) -> int:
        """Estimate syllable count."""
        word = word.lower()
        vowel_groups = re.findall(r'[aeiou]+', word)
        count = len(vowel_groups)
        # Adjust for silent e
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
    
    def _has_alliteration(self, name: str) -> int:
        """Check if name parts start with same letter."""
        parts = name.split()
        if len(parts) >= 2:
            first_letters = [p[0].lower() for p in parts if p]
            return 1 if len(set(first_letters)) < len(first_letters) else 0
        return 0
    
    def _phonetic_distinctiveness(self, name: str) -> float:
        """How unique is the sound combination?"""
        if len(name) < 2:
            return 0.0
        
        # Count unique bigrams
        bigrams = [name[i:i+2] for i in range(len(name)-1)]
        unique_ratio = len(set(bigrams)) / len(bigrams)
        
        return unique_ratio
    
    def _stress_pattern_score(self, name: str) -> float:
        """Regularity of stressed/unstressed pattern."""
        syllables = self._syllable_count(name)
        
        # Simple heuristic: alternating vowel-consonant is regular
        pattern = []
        for c in name:
            if c in self.vowels:
                pattern.append('V')
            elif c.isalpha():
                pattern.append('C')
        
        if len(pattern) < 2:
            return 0.5
        
        # Count alternations
        alternations = sum(1 for i in range(len(pattern)-1) if pattern[i] != pattern[i+1])
        regularity = alternations / (len(pattern) - 1)
        
        return regularity
    
    def _generate_interpretation(self) -> str:
        return (
            f"Phonetic Formula Analysis:\n"
            f"- Harsh consonant baseline: {self.corpus_harsh_ratio:.2%}\n"
            f"- Vowel density baseline: {self.corpus_vowel_ratio:.2%}\n"
            f"- Analyzed sound patterns: harshness, softness, prosody, memorability\n"
            f"- Features extracted: 15 phonetic dimensions"
        )


class SemanticFormulaTransformer(TextNarrativeTransformer):
    """
    Type 2: Semantic Formula
    
    Analyzes meaning fields in names:
    - Power semantics (king, master, chief)
    - Speed semantics (swift, rapid, quick)
    - Size semantics (titan, micro, mega)
    - Temperature semantics (hot, cold, warm)
    - Innovation semantics (new, future, next)
    """
    
    def __init__(self):
        super().__init__(
            name="semantic_formula",
            narrative_hypothesis="Meaning fields in names shape expectations and outcomes"
        )
        
        # Semantic dictionaries
        self.power_words = {'king', 'queen', 'master', 'chief', 'prime', 'super', 'ultra', 'dominant', 'supreme'}
        self.speed_words = {'quick', 'fast', 'swift', 'rapid', 'instant', 'flash', 'turbo', 'speed', 'velocity'}
        self.size_words_large = {'titan', 'mega', 'giant', 'grand', 'max', 'big', 'great', 'major'}
        self.size_words_small = {'micro', 'mini', 'tiny', 'nano', 'small', 'lite', 'pocket'}
        self.temp_words_hot = {'hot', 'fire', 'blaze', 'burn', 'flame', 'heat'}
        self.temp_words_cold = {'cold', 'ice', 'frost', 'cool', 'chill', 'freeze'}
        self.innovation_words = {'new', 'next', 'future', 'modern', 'advanced', 'smart', 'ai', 'tech'}
        self.tradition_words = {'classic', 'traditional', 'heritage', 'vintage', 'original', 'old', 'ancient'}
        self.nature_words = {'natural', 'organic', 'earth', 'green', 'eco', 'bio', 'wild'}
        self.technical_words = {'system', 'tech', 'digital', 'cyber', 'quantum', 'data', 'network'}
        
    def fit(self, X, y=None):
        """Learn semantic distributions."""
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract semantic features."""
        features = []
        
        for text in X:
            text_lower = text.lower()
            
            semantic_features = [
                # Core semantic dimensions
                self._contains_any(text_lower, self.power_words),
                self._contains_any(text_lower, self.speed_words),
                self._contains_any(text_lower, self.size_words_large),
                self._contains_any(text_lower, self.size_words_small),
                self._contains_any(text_lower, self.temp_words_hot),
                self._contains_any(text_lower, self.temp_words_cold),
                self._contains_any(text_lower, self.innovation_words),
                self._contains_any(text_lower, self.tradition_words),
                self._contains_any(text_lower, self.nature_words),
                self._contains_any(text_lower, self.technical_words),
                
                # Composite dimensions
                self._contains_any(text_lower, self.power_words) and self._contains_any(text_lower, self.speed_words),  # Power + Speed
                self._contains_any(text_lower, self.innovation_words) and self._contains_any(text_lower, self.technical_words),  # Tech innovation
                self._contains_any(text_lower, self.nature_words) and self._contains_any(text_lower, self.tradition_words),  # Natural tradition
                
                # Counts
                self._count_any(text_lower, self.power_words),
                self._count_any(text_lower, self.innovation_words),
            ]
            
            features.append(semantic_features)
        
        return np.array(features, dtype=float)
    
    def _contains_any(self, text: str, word_set: Set[str]) -> float:
        """Check if text contains any words from set."""
        return 1.0 if any(word in text for word in word_set) else 0.0
    
    def _count_any(self, text: str, word_set: Set[str]) -> float:
        """Count occurrences of words from set."""
        return float(sum(text.count(word) for word in word_set))
    
    def _generate_interpretation(self) -> str:
        return (
            f"Semantic Formula Analysis:\n"
            f"- Tracked 10 primary semantic fields\n"
            f"- Analyzed power, speed, size, temperature, innovation themes\n"
            f"- Features extracted: 15 semantic dimensions"
        )


class StructuralFormulaTransformer(TextNarrativeTransformer):
    """
    Type 3: Structural Formula
    
    Analyzes structural properties of names:
    - Length (character count)
    - Syllable count
    - Morpheme count
    - Part-of-speech patterns
    - Compound structure
    """
    
    def __init__(self):
        super().__init__(
            name="structural_formula",
            narrative_hypothesis="Structural properties of names affect processing and memory"
        )
    
    def fit(self, X, y=None):
        """Learn structural distributions."""
        all_names = []
        
        for text in X:
            names = re.findall(r'\b[A-Z][a-z]+\b', text)
            all_names.extend(names)
        
        if all_names:
            self.mean_length = np.mean([len(name) for name in all_names])
            self.std_length = np.std([len(name) for name in all_names])
        else:
            self.mean_length = 6.0
            self.std_length = 2.0
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract structural features."""
        features = []
        
        for text in X:
            names = re.findall(r'\b[A-Z][a-z]+\b', text)
            
            if names:
                struct_features = [
                    # Length metrics
                    np.mean([len(n) for n in names]),
                    np.std([len(n) for n in names]) if len(names) > 1 else 0,
                    np.min([len(n) for n in names]),
                    np.max([len(n) for n in names]),
                    
                    # Syllable metrics
                    np.mean([self._syllable_count(n) for n in names]),
                    
                    # Morpheme patterns
                    np.mean([self._morpheme_count(n) for n in names]),
                    
                    # Compound detection
                    np.mean([self._is_compound(n) for n in names]),
                    
                    # Character diversity
                    np.mean([len(set(n.lower())) / (len(n) + 1) for n in names]),
                    
                    # Standardized length (z-score)
                    (np.mean([len(n) for n in names]) - self.mean_length) / (self.std_length + 1),
                    
                    # Count of names
                    len(names),
                ]
            else:
                struct_features = [0] * 10
            
            features.append(struct_features)
        
        return np.array(features)
    
    def _syllable_count(self, word: str) -> int:
        """Estimate syllable count."""
        word = word.lower()
        vowels = 'aeiou'
        vowel_groups = re.findall(r'[aeiou]+', word)
        count = len(vowel_groups)
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
    
    def _morpheme_count(self, word: str) -> int:
        """Estimate morpheme count (very rough)."""
        # Common prefixes/suffixes
        prefixes = ['un', 're', 'pre', 'post', 'anti', 'de', 'dis']
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'tion', 'able']
        
        count = 1  # Base word
        word_lower = word.lower()
        
        for prefix in prefixes:
            if word_lower.startswith(prefix):
                count += 1
                break
        
        for suffix in suffixes:
            if word_lower.endswith(suffix):
                count += 1
                break
        
        return count
    
    def _is_compound(self, word: str) -> int:
        """Check if word appears to be compound."""
        # Look for camelCase or multiple capital letters
        capitals = sum(1 for c in word if c.isupper())
        return 1 if capitals >= 2 else 0
    
    def _generate_interpretation(self) -> str:
        return (
            f"Structural Formula Analysis:\n"
            f"- Mean name length: {self.mean_length:.1f} ± {self.std_length:.1f} characters\n"
            f"- Analyzed length, syllables, morphemes, compounds\n"
            f"- Features extracted: 10 structural dimensions"
        )


class NominativeTaxonomyIntegrator:
    """
    Integrates all six nominative formula types and determines
    which matters most in each domain.
    """
    
    def __init__(self):
        self.transformers = {
            'phonetic': PhoneticFormulaTransformer(),
            'semantic': SemanticFormulaTransformer(),
            'structural': StructuralFormulaTransformer(),
            # Additional types would go here:
            # 'frequency': FrequencyFormulaTransformer(),
            # 'numerology': NumerologyFormulaTransformer(),
            # 'hybrid': HybridFormulaTransformer(),
        }
        
        self.importance_scores = {}
    
    def fit_transform(self, X, y=None):
        """Fit all transformers and extract all features."""
        all_features = []
        
        for name, transformer in self.transformers.items():
            transformer.fit(X, y)
            features = transformer.transform(X)
            all_features.append(features)
        
        # Concatenate all features
        combined = np.hstack(all_features)
        
        return combined
    
    def analyze_importance(self, X, y, model=None):
        """Determine which formula types matter most in this domain."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        
        # Fit each transformer separately and test importance
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        results = {}
        
        for name, transformer in self.transformers.items():
            transformer.fit(X, y)
            X_transformed = transformer.transform(X)
            
            # Train model
            model.fit(X_transformed, y)
            score = model.score(X_transformed, y)
            
            results[name] = {
                'accuracy': score,
                'n_features': X_transformed.shape[1]
            }
        
        # Rank by accuracy
        ranked = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        return {
            'rankings': ranked,
            'best_formula': ranked[0][0],
            'best_accuracy': ranked[0][1]['accuracy']
        }

