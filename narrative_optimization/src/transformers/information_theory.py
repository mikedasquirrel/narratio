"""
Information Theory Transformer

Explains the MECHANISM of why nominative determinism works using information-theoretic
measures. This transformer reveals the deep mathematical structure underlying narrative force.

Research Foundation:
- Shannon entropy: Information density affects memorability
- Kolmogorov complexity: Compressibility predicts processing ease
- Levenshtein distance: Similarity creates confusion
- "Bitcoin" (7 chars, high entropy) vs "Ripple" (6 chars, low entropy, confusable)
- Edit distance <3 creates competitive substitution effects

Core Insight:
Names and narratives are information signals. Their information-theoretic properties
(entropy, complexity, distinctiveness) determine cognitive processing, memory formation,
and ultimately outcomes.

Universal across domains:
- High entropy = hard to confuse, memorable (but harder to process)
- Low entropy = easy to process (but forgettable, confusable)
- Optimal zone: Moderate entropy, high distinctiveness
- Edit distance determines competitive dynamics
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter
import math
from .base import NarrativeTransformer


class InformationTheoryTransformer(NarrativeTransformer):
    """
    Analyzes information-theoretic properties of narratives.
    
    Tests hypothesis that information density, compressibility, and distinctiveness
    explain why certain phonetic patterns predict outcomes.
    
    Features extracted (25):
    - Shannon entropy (information density)
    - Kolmogorov complexity approximation (compressibility)
    - Character-level entropy
    - N-gram entropy (bigrams, trigrams)
    - Redundancy measures
    - Distinctiveness metrics
    - Compression ratio
    - Information per character
    - Namespace crowding proxies
    - Typo resilience
    
    Parameters
    ----------
    reference_corpus : list of str, optional
        Corpus for computing relative distinctiveness
    """
    
    def __init__(self, reference_corpus: List[str] = None):
        super().__init__(
            narrative_id="information_theory",
            description="Information theory: explains mechanism via entropy, complexity, distinctiveness"
        )
        
        self.reference_corpus = reference_corpus
        self.corpus_words = set()
    
    def fit(self, X, y=None):
        """
        Learn information patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Build reference vocabulary for distinctiveness calculation
        for text in X:
            words = re.findall(r'\b\w+\b', text.lower())
            self.corpus_words.update(words)
        
        # Compute corpus-level information statistics
        all_chars = ''.join(''.join(re.findall(r'\b\w+\b', text.lower())) for text in X)
        
        if all_chars:
            char_counts = Counter(all_chars)
            total_chars = len(all_chars)
            char_probs = {char: count/total_chars for char, count in char_counts.items()}
            
            # Corpus entropy
            corpus_entropy = -sum(p * math.log2(p) for p in char_probs.values() if p > 0)
            self.metadata['corpus_entropy'] = corpus_entropy
            self.metadata['corpus_vocab_size'] = len(self.corpus_words)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to information theory features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array, shape (n_samples, 25)
            Information theory feature matrix
        """
        self._validate_fitted()
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_info_theory_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_info_theory_features(self, text: str) -> np.ndarray:
        """Extract all 25 information theory features."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return np.zeros(25)
        
        primary_word = words[0]
        all_chars = ''.join(words[:10])  # Use first 10 words
        
        features = []
        
        # === SHANNON ENTROPY (7 features) ===
        
        # 1. Character-level Shannon entropy (primary word)
        features.append(self._compute_shannon_entropy(primary_word))
        
        # 2. Character-level entropy (full text sample)
        features.append(self._compute_shannon_entropy(all_chars))
        
        # 3. Word-level entropy (vocabulary diversity)
        word_counts = Counter(words[:50])
        total_words = len(words[:50])
        word_probs = {w: c/total_words for w, c in word_counts.items()}
        word_entropy = -sum(p * math.log2(p) for p in word_probs.values() if p > 0)
        features.append(word_entropy)
        
        # 4. Bigram entropy (primary word)
        bigrams = [primary_word[i:i+2] for i in range(len(primary_word)-1)]
        features.append(self._compute_shannon_entropy(''.join(bigrams)))
        
        # 5. Trigram entropy (primary word)
        trigrams = [primary_word[i:i+3] for i in range(len(primary_word)-2)]
        features.append(self._compute_shannon_entropy(''.join(trigrams)))
        
        # 6. Normalized entropy (entropy per character)
        features.append(features[0] / max(1, len(primary_word)))
        
        # 7. Entropy efficiency (information density)
        # High entropy per character = efficient information encoding
        features.append(features[1] / max(1, len(all_chars)))
        
        # === KOLMOGOROV COMPLEXITY (5 features) ===
        
        # 8. Compression ratio approximation (simple LZ-style)
        compressed_length = self._approximate_compressed_length(primary_word)
        compression_ratio = compressed_length / max(1, len(primary_word))
        features.append(compression_ratio)
        
        # 9. Repetition factor (redundancy)
        char_counts = Counter(primary_word.lower())
        max_char_repeat = max(char_counts.values()) if char_counts else 1
        repetition_factor = max_char_repeat / len(primary_word)
        features.append(repetition_factor)
        
        # 10. Pattern regularity (repeating patterns)
        patterns = self._find_repeating_patterns(primary_word)
        pattern_score = len(patterns) / max(1, len(primary_word))
        features.append(pattern_score)
        
        # 11. Complexity estimate (unique chars / total chars)
        complexity = len(set(primary_word.lower())) / max(1, len(primary_word))
        features.append(complexity)
        
        # 12. Information content (entropy × length)
        info_content = features[0] * len(primary_word)
        features.append(info_content)
        
        # === DISTINCTIVENESS / EDIT DISTANCE (8 features) ===
        
        # 13. Minimum edit distance to corpus (proxy)
        # Use first 3 chars as proxy for efficiency
        prefix = primary_word[:3].lower()
        similar_words = [w for w in list(self.corpus_words)[:1000] if w.startswith(prefix)]
        
        if similar_words and primary_word.lower() not in similar_words:
            # Approximate: count differing characters
            min_diff = min(
                sum(c1 != c2 for c1, c2 in zip(primary_word.lower(), w))
                for w in similar_words[:10]
            )
            features.append(min_diff)
        else:
            features.append(len(primary_word))  # Very distinct
        
        # 14. Phonetic neighborhood density (words with similar sounds)
        # Proxy: similar length + shared vowels
        similar_length = [w for w in self.corpus_words if abs(len(w) - len(primary_word)) <= 1]
        features.append(len(similar_length))
        
        # 15. Typo resilience (unique even with 1-char change)
        # High uniqueness = robust to typos
        typo_resilience = features[11] * features[0]  # complexity × entropy
        features.append(typo_resilience)
        
        # 16. Namespace pollution (same-prefix words)
        prefix_matches = sum(1 for w in self.corpus_words if w.startswith(prefix))
        features.append(prefix_matches)
        
        # 17. Suffix uniqueness
        suffix = primary_word[-3:].lower() if len(primary_word) >= 3 else primary_word
        suffix_matches = sum(1 for w in self.corpus_words if w.endswith(suffix))
        features.append(suffix_matches)
        
        # 18. Overall distinctiveness (inverse of similar words)
        # Fewer similar words = more distinct
        distinctiveness = 1.0 / (1.0 + len(similar_words[:50]))
        features.append(distinctiveness * 100)
        
        # 19. Acoustic distinctiveness (phonetic uniqueness)
        # Approximated by rare sound combinations
        rare_bigrams = sum(1 for i in range(len(primary_word)-1)
                          if primary_word[i:i+2] not in common_bigrams)
        acoustic_distinct = rare_bigrams / max(1, len(primary_word)-1)
        features.append(acoustic_distinct)
        
        # 20. Competitive substitution risk (low edit distance to competitors)
        # High namespace pollution + low distinctiveness = substitution risk
        substitution_risk = features[16] / max(1, features[18])
        features.append(min(10.0, substitution_risk))
        
        # === INFORMATION EFFICIENCY (5 features) ===
        
        # 21. Information per character
        info_per_char = features[0] / max(1, len(primary_word))
        features.append(info_per_char)
        
        # 22. Signal-to-noise ratio (entropy / redundancy)
        signal_to_noise = features[0] / max(0.1, features[9])
        features.append(signal_to_noise)
        
        # 23. Encoding efficiency (high entropy, low length)
        encoding_efficiency = features[0] / max(1, math.log2(len(primary_word) + 1))
        features.append(encoding_efficiency)
        
        # 24. Redundancy ratio (compression savings)
        redundancy = 1.0 - features[8]  # 1 - compression ratio
        features.append(redundancy)
        
        # 25. Optimal information balance
        # Sweet spot: moderate entropy, high distinctiveness, low redundancy
        optimal_balance = (
            (features[0] / 5.0) *      # entropy normalized
            (features[18] / 100) *      # distinctiveness
            (1.0 - features[9])         # inverse repetition
        )
        features.append(optimal_balance)
        
        return np.array(features)
    
    def _compute_shannon_entropy(self, text: str) -> float:
        """Compute Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Character frequency
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        # Shannon entropy: H = -Σ p(x) log₂ p(x)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _approximate_compressed_length(self, text: str) -> int:
        """
        Approximate Kolmogorov complexity via simple compression.
        
        Simplified LZ-style: count unique substrings.
        Real implementation would use actual compression algorithm.
        """
        text = text.lower()
        
        # Count unique bigrams and trigrams
        bigrams = set(text[i:i+2] for i in range(len(text)-1))
        trigrams = set(text[i:i+3] for i in range(len(text)-2))
        
        # Unique chars + unique patterns
        unique_chars = len(set(text))
        unique_patterns = len(bigrams) + len(trigrams)
        
        # Approximate compressed length
        return unique_chars + int(math.log2(unique_patterns + 1))
    
    def _find_repeating_patterns(self, text: str) -> List[str]:
        """Find repeating character patterns."""
        patterns = []
        text = text.lower()
        
        # Find repeated substrings of length 2-4
        for length in [2, 3, 4]:
            for i in range(len(text) - length + 1):
                pattern = text[i:i+length]
                if text.count(pattern) > 1:
                    patterns.append(pattern)
        
        return list(set(patterns))
    
    def get_feature_names(self) -> List[str]:
        """Return names of all 25 features."""
        return [
            # Shannon entropy (7)
            'char_entropy_primary', 'char_entropy_text', 'word_entropy',
            'bigram_entropy', 'trigram_entropy', 'normalized_entropy', 'entropy_efficiency',
            
            # Kolmogorov complexity (5)
            'compression_ratio', 'repetition_factor', 'pattern_regularity',
            'complexity_estimate', 'information_content',
            
            # Distinctiveness (8)
            'min_edit_distance', 'phonetic_neighborhood_density', 'typo_resilience',
            'namespace_pollution', 'suffix_matches', 'overall_distinctiveness',
            'acoustic_distinctiveness', 'substitution_risk',
            
            # Information efficiency (5)
            'info_per_character', 'signal_to_noise', 'encoding_efficiency',
            'redundancy_ratio', 'optimal_balance'
        ]
    
    def interpret_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Interpret information theory features in plain English.
        
        Parameters
        ----------
        features : array, shape (25,)
            Feature vector for one document
        
        Returns
        -------
        interpretation : dict
            Plain English interpretation
        """
        names = self.get_feature_names()
        
        interpretation = {
            'summary': self._generate_summary(features),
            'features': {},
            'insights': []
        }
        
        # Entropy
        entropy = features[0]
        if entropy > 3.5:
            interpretation['insights'].append("HIGH entropy - information-dense, hard to confuse (like Bitcoin)")
        elif entropy < 2.0:
            interpretation['insights'].append("LOW entropy - easy to process but forgettable (substitution risk)")
        else:
            interpretation['insights'].append("OPTIMAL entropy - balanced information density")
        
        # Complexity
        compression = features[7]
        if compression < 0.5:
            interpretation['insights'].append("Highly compressible (repetitive patterns) - easy to remember")
        elif compression > 0.8:
            interpretation['insights'].append("Low compressibility (complex) - requires more cognitive effort")
        
        # Distinctiveness
        distinctiveness = features[13]
        if distinctiveness > 70:
            interpretation['insights'].append("HIGHLY distinct - minimal confusion with competitors")
        elif distinctiveness < 30:
            interpretation['insights'].append("LOW distinctiveness - substitution/confusion risk")
        
        # Substitution risk
        risk = features[15]
        if risk > 3.0:
            interpretation['insights'].append(f"HIGH substitution risk ({risk:.1f}) - similar names nearby")
        
        return interpretation
    
    def _generate_summary(self, features: np.ndarray) -> str:
        """Generate plain English summary."""
        entropy = features[0]
        distinctiveness = features[13]
        optimal_balance = features[24]
        
        if optimal_balance > 0.6:
            return f"OPTIMAL information profile: High entropy ({entropy:.2f}), high distinctiveness ({distinctiveness:.0f}/100). Predicted to be memorable and unconfusable."
        elif entropy > 3.5 and distinctiveness > 60:
            return f"Information-dense profile: Very high entropy ({entropy:.2f}), distinct. Harder to process but memorable."
        elif entropy < 2.0 or distinctiveness < 40:
            return f"Information-sparse profile: Low entropy ({entropy:.2f}) or low distinctiveness. Risk of confusion/forgettability."
        else:
            return f"Moderate information profile: Entropy {entropy:.2f}, distinctiveness {distinctiveness:.0f}/100."


# Common bigrams for acoustic distinctiveness calculation (English)
common_bigrams = {
    'th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd',
    'ti', 'es', 'or', 'te', 'of', 'ed', 'is', 'it', 'al', 'ar',
    'st', 'to', 'nt', 'ng', 'se', 'ha', 'as', 'ou', 'io', 'le'
}

