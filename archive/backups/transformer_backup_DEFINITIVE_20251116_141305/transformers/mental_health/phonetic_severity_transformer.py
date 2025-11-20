"""
Phonetic Severity Transformer for Mental Health Disorders

Extracts phonetic features from disorder names that predict stigma:
- Harshness score (plosives, sibilants, hard consonants)
- Pronunciation difficulty
- Foreign/medical terminology presence
- Syllable complexity

Key Finding: Harsh phonetics → higher stigma → lower treatment seeking

Author: Narrative Optimization Research
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict
import re

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.transformers.base_transformer import TextNarrativeTransformer


class PhoneticSeverityTransformer(TextNarrativeTransformer):
    """
    Extract phonetic severity features from mental health disorder names.
    
    Tests hypothesis: Harsh-sounding diagnoses create higher stigma,
    reducing treatment-seeking behavior.
    """
    
    def __init__(self, include_interactions: bool = True):
        """
        Initialize phonetic severity transformer.
        
        Parameters
        ----------
        include_interactions : bool
            Include interaction features between phonetic elements
        """
        super().__init__(
            narrative_id="phonetic_severity",
            description="Analyzes phonetic harshness of disorder names to predict stigma"
        )
        self.include_interactions = include_interactions
        
        # Phonetic categories
        self.plosives = set('pbtdkgqc')  # Harsh sounds
        self.sibilants = set('szhSZ')     # Hissing sounds
        self.fricatives = set('fvθðszʃʒhH')  # Friction sounds
        self.sonorants = set('lrmnwyLRMNWY')  # Flowing sounds
        self.vowels = set('aeiouAEIOU')
        
        # Medical/foreign prefixes that increase perceived severity
        self.medical_prefixes = [
            'schizo', 'psycho', 'neuro', 'dys', 'hyper', 'hypo',
            'para', 'poly', 'mono', 'auto', 'hetero', 'homo'
        ]
        
        # Latin/Greek roots signal medical authority
        self.latin_greek_markers = [
            'phrenia', 'phobia', 'mania', 'path', 'somat',
            'morph', 'cephal', 'cardio', 'thym', 'neur'
        ]
    
    def fit(self, X, y=None):
        """
        Learn phonetic patterns from disorder names.
        
        Parameters
        ----------
        X : list of str
            Disorder names
        y : array-like, optional
            Stigma scores or treatment seeking rates
        
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Analyze corpus
        harshness_scores = []
        syllable_counts = []
        complexity_scores = []
        
        for name in X:
            features = self._extract_phonetic_features(name)
            harshness_scores.append(features['harshness'])
            syllable_counts.append(features['syllables'])
            complexity_scores.append(features['complexity'])
        
        self.metadata['corpus_stats'] = {
            'n_disorders': len(X),
            'harshness': {
                'mean': np.mean(harshness_scores),
                'std': np.std(harshness_scores),
                'range': (min(harshness_scores), max(harshness_scores))
            },
            'syllables': {
                'mean': np.mean(syllable_counts),
                'std': np.std(syllable_counts),
                'range': (min(syllable_counts), max(syllable_counts))
            },
            'complexity': {
                'mean': np.mean(complexity_scores),
                'std': np.std(complexity_scores)
            }
        }
        
        # If targets provided, calculate correlations
        if y is not None:
            y_array = np.array(y)
            self.metadata['correlations'] = {
                'harshness_vs_stigma': np.corrcoef(harshness_scores, y_array)[0, 1],
                'syllables_vs_stigma': np.corrcoef(syllable_counts, y_array)[0, 1],
                'complexity_vs_stigma': np.corrcoef(complexity_scores, y_array)[0, 1]
            }
        
        self.metadata['n_features'] = self._calculate_n_features()
        self.metadata['feature_names'] = self._generate_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform disorder names to phonetic severity features.
        
        Parameters
        ----------
        X : list of str
            Disorder names
        
        Returns
        -------
        numpy.ndarray
            Feature matrix (n_samples, n_features)
        """
        self._validate_fitted()
        self._validate_input(X)
        
        features = []
        for name in X:
            feature_vec = self._extract_all_features(name)
            features.append(feature_vec)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_phonetic_features(self, name: str) -> Dict:
        """Extract basic phonetic features."""
        name_lower = name.lower()
        
        # Count phoneme types
        n_plosives = sum(1 for c in name_lower if c in self.plosives)
        n_sibilants = sum(1 for c in name_lower if c in self.sibilants)
        n_fricatives = sum(1 for c in name_lower if c in self.fricatives)
        n_sonorants = sum(1 for c in name_lower if c in self.sonorants)
        n_vowels = sum(1 for c in name_lower if c in self.vowels)
        
        total_phonemes = len([c for c in name_lower if c.isalpha()])
        
        # Harshness score (weighted by harsh phonemes)
        if total_phonemes > 0:
            harshness = (n_plosives * 3 + n_sibilants * 2 + n_fricatives * 1.5) / total_phonemes * 100
        else:
            harshness = 0
        
        # Syllable count (approximate)
        syllables = self._count_syllables(name)
        
        # Complexity (ratio of consonants to vowels)
        consonants = total_phonemes - n_vowels
        complexity = consonants / (n_vowels + 1)  # Avoid division by zero
        
        return {
            'harshness': harshness,
            'syllables': syllables,
            'complexity': complexity,
            'n_plosives': n_plosives,
            'n_sibilants': n_sibilants,
            'n_fricatives': n_fricatives,
            'n_sonorants': n_sonorants
        }
    
    def _count_syllables(self, name: str) -> int:
        """Count syllables in disorder name."""
        name_lower = name.lower()
        
        # Remove silent e
        name_lower = re.sub(r'e$', '', name_lower)
        
        # Count vowel groups
        syllables = 0
        previous_was_vowel = False
        
        for char in name_lower:
            is_vowel = char in self.vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        return max(1, syllables)
    
    def _extract_all_features(self, name: str) -> List[float]:
        """Extract complete feature vector."""
        features = []
        name_lower = name.lower()
        
        # Basic phonetic features
        phonetic = self._extract_phonetic_features(name)
        
        # 1. Harshness score (primary feature)
        features.append(phonetic['harshness'])
        
        # 2. Syllable count
        features.append(float(phonetic['syllables']))
        
        # 3. Complexity ratio
        features.append(phonetic['complexity'])
        
        # 4. Individual phoneme counts (normalized)
        total_chars = len([c for c in name_lower if c.isalpha()])
        features.append(phonetic['n_plosives'] / (total_chars + 1))
        features.append(phonetic['n_sibilants'] / (total_chars + 1))
        features.append(phonetic['n_fricatives'] / (total_chars + 1))
        features.append(phonetic['n_sonorants'] / (total_chars + 1))
        
        # 5. Medical terminology markers
        has_medical_prefix = any(prefix in name_lower for prefix in self.medical_prefixes)
        features.append(1.0 if has_medical_prefix else 0.0)
        
        has_latin_greek = any(marker in name_lower for marker in self.latin_greek_markers)
        features.append(1.0 if has_latin_greek else 0.0)
        
        # 6. Length and structure
        features.append(float(len(name)))
        features.append(float(len(set(name_lower))))  # Unique characters
        
        # 7. Pronunciation difficulty (consonant clusters)
        consonant_clusters = len(re.findall(r'[^aeiou]{3,}', name_lower))
        features.append(float(consonant_clusters))
        
        # 8. Starts/ends with harsh sound
        if name_lower:
            features.append(1.0 if name_lower[0] in self.plosives else 0.0)
            features.append(1.0 if name_lower[-1] in self.plosives else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        # 9. Interaction features (if enabled)
        if self.include_interactions:
            # Harshness × Syllables
            features.append(phonetic['harshness'] * phonetic['syllables'])
            
            # Medical terminology × Harshness
            medical_flag = 1.0 if has_medical_prefix else 0.0
            features.append(medical_flag * phonetic['harshness'])
            
            # Complexity × Length
            features.append(phonetic['complexity'] * len(name))
        
        return features
    
    def _calculate_n_features(self) -> int:
        """Calculate total number of features."""
        base_features = 14  # Core phonetic features
        interaction_features = 3 if self.include_interactions else 0
        return base_features + interaction_features
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names."""
        names = [
            'harshness_score',
            'syllable_count',
            'complexity_ratio',
            'plosives_normalized',
            'sibilants_normalized',
            'fricatives_normalized',
            'sonorants_normalized',
            'has_medical_prefix',
            'has_latin_greek_root',
            'name_length',
            'unique_characters',
            'consonant_clusters',
            'starts_with_plosive',
            'ends_with_plosive'
        ]
        
        if self.include_interactions:
            names.extend([
                'harshness_x_syllables',
                'medical_x_harshness',
                'complexity_x_length'
            ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of phonetic patterns."""
        stats = self.metadata['corpus_stats']
        
        interpretation = f"""
Phonetic Severity Analysis ({stats['n_disorders']} disorders)

HARSHNESS DISTRIBUTION
----------------------
Mean: {stats['harshness']['mean']:.1f}
SD: {stats['harshness']['std']:.1f}
Range: {stats['harshness']['range'][0]:.1f} - {stats['harshness']['range'][1]:.1f}

COMPLEXITY METRICS
------------------
Mean Syllables: {stats['syllables']['mean']:.1f}
Mean Complexity: {stats['complexity']['mean']:.2f}
"""
        
        if 'correlations' in self.metadata:
            corr = self.metadata['correlations']
            interpretation += f"""
CORRELATIONS WITH STIGMA
------------------------
Harshness: r = {corr['harshness_vs_stigma']:.3f}
Syllables: r = {corr['syllables_vs_stigma']:.3f}
Complexity: r = {corr['complexity_vs_stigma']:.3f}

INTERPRETATION
--------------
{'Strong positive' if corr['harshness_vs_stigma'] > 0.25 else 'Moderate' if corr['harshness_vs_stigma'] > 0.15 else 'Weak'} correlation between phonetic harshness and stigma.
Harsh-sounding disorder names {'are' if corr['harshness_vs_stigma'] > 0.20 else 'may be'} associated with higher stigma scores.
"""
        else:
            interpretation += "\nNo stigma data provided for correlation analysis.\n"
        
        interpretation += """
MECHANISM
---------
Harsh phonetics (plosives, sibilants) → Perceived severity ↑
Medical/foreign terminology → Clinical distance ↑
Both factors → Stigma ↑ → Treatment seeking ↓
"""
        
        return interpretation
    
    def predict_stigma_from_name(self, disorder_name: str) -> Dict:
        """
        Predict stigma level from disorder name phonetics.
        
        Parameters
        ----------
        disorder_name : str
            Name of disorder
        
        Returns
        -------
        dict
            Predicted stigma components and explanation
        """
        self._validate_fitted()
        
        phonetic = self._extract_phonetic_features(disorder_name)
        name_lower = disorder_name.lower()
        
        # Base stigma from harshness (scaled 0-10)
        base_stigma = phonetic['harshness'] / 10
        
        # Medical terminology adds clinical distance
        has_medical = any(p in name_lower for p in self.medical_prefixes)
        medical_stigma = 1.5 if has_medical else 0
        
        # Complexity adds pronunciation barrier
        complexity_stigma = phonetic['complexity'] * 0.5
        
        # Total predicted stigma
        total_stigma = min(10, base_stigma + medical_stigma + complexity_stigma)
        
        return {
            'disorder_name': disorder_name,
            'predicted_stigma': total_stigma,
            'harshness_score': phonetic['harshness'],
            'harshness_contribution': base_stigma,
            'medical_contribution': medical_stigma,
            'complexity_contribution': complexity_stigma,
            'interpretation': self._explain_stigma_prediction(
                disorder_name, total_stigma, phonetic, has_medical
            )
        }
    
    def _explain_stigma_prediction(self, name: str, stigma: float,
                                   phonetic: Dict, has_medical: bool) -> str:
        """Generate explanation for stigma prediction."""
        explanation = f"Disorder: {name}\nPredicted Stigma: {stigma:.1f}/10\n\n"
        
        explanation += "Contributing Factors:\n"
        
        if phonetic['harshness'] > 70:
            explanation += f"  • HIGH phonetic harshness ({phonetic['harshness']:.0f}) → severe-sounding\n"
        elif phonetic['harshness'] > 50:
            explanation += f"  • MODERATE phonetic harshness ({phonetic['harshness']:.0f})\n"
        else:
            explanation += f"  • LOW phonetic harshness ({phonetic['harshness']:.0f}) → softer sound\n"
        
        if has_medical:
            explanation += "  • Medical/foreign terminology → clinical distance ↑\n"
        
        if phonetic['complexity'] > 2:
            explanation += f"  • High complexity ({phonetic['complexity']:.1f}) → pronunciation barrier\n"
        
        if phonetic['syllables'] > 4:
            explanation += f"  • Long name ({phonetic['syllables']} syllables) → harder to discuss\n"
        
        return explanation


if __name__ == '__main__':
    # Demo
    transformer = PhoneticSeverityTransformer()
    
    # Example disorders
    disorders = [
        'Schizophrenia',
        'Depression',
        'Anxiety',
        'Bipolar Disorder'
    ]
    
    # Fit and transform
    transformer.fit(disorders)
    features = transformer.transform(disorders)
    
    print(transformer.get_interpretation())
    print("\nStigma Predictions:")
    for disorder in disorders:
        pred = transformer.predict_stigma_from_name(disorder)
        print(f"\n{pred['disorder_name']}: {pred['predicted_stigma']:.1f}/10")
        print(pred['interpretation'])

