"""
Name Characteristics Extractor for Meta-Nominative Analysis

Extract features from researcher names: memorability, distinctiveness, authority.
Uses existing transformers (Phonetic, SocialStatus) from main framework.
"""

import json
import sys
from typing import Dict, List
from pathlib import Path
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers import (
    PhoneticTransformer,
    SocialStatusTransformer
)


class NameCharacteristicsExtractor:
    """Extract characteristics from researcher names."""
    
    def __init__(self):
        """Initialize transformers for feature extraction."""
        self.phonetic_transformer = PhoneticTransformer()
        self.social_status_transformer = SocialStatusTransformer()
        
        # Fit transformers with researcher names (will be done during extraction)
        self.fitted = False
    
    def extract_characteristics(self, researchers: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Extract name characteristics for all researchers.
        
        Args:
            researchers: Dictionary of researcher data
            
        Returns:
            Dictionary with name characteristics added
        """
        print(f"\n{'='*80}")
        print("EXTRACTING NAME CHARACTERISTICS")
        print(f"{'='*80}\n")
        
        # Collect all names for transformer fitting
        names = list(researchers.keys())
        
        print(f"Fitting transformers on {len(names)} researcher names...")
        
        # Fit transformers
        self.phonetic_transformer.fit(names)
        self.social_status_transformer.fit(names)
        self.fitted = True
        
        print("✓ Transformers fitted\n")
        
        # Extract features for each researcher
        for i, (name, data) in enumerate(researchers.items(), 1):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(researchers)} researchers...")
            
            characteristics = self._extract_single(name)
            data['name_characteristics'] = characteristics
        
        print(f"\n✓ Extracted characteristics for {len(researchers)} researchers")
        
        # Print summary
        self._print_summary(researchers)
        
        return researchers
    
    def _extract_single(self, name: str) -> Dict:
        """Extract characteristics for a single name."""
        if not self.fitted:
            raise ValueError("Transformers must be fitted before extraction")
        
        # Phonetic features (memorability, distinctiveness, etc.)
        phonetic_features = self.phonetic_transformer.transform([name])[0]
        
        # Social status features (authority, professionalism, etc.)
        status_features = self.social_status_transformer.transform([name])[0]
        
        # Extract key metrics
        characteristics = {
            # From phonetic transformer
            'memorability': self._calculate_memorability(phonetic_features),
            'distinctiveness': self._calculate_distinctiveness(phonetic_features),
            'phonetic_complexity': self._calculate_complexity(phonetic_features),
            'euphony': self._calculate_euphony(phonetic_features),
            
            # From social status transformer
            'authority_score': self._calculate_authority(status_features),
            'professional_score': self._calculate_professional(status_features),
            
            # Basic name properties
            'name_length': len(name),
            'word_count': len(name.split()),
            'has_title': any(title in name.lower() for title in ['dr', 'prof', 'phd']),
            
            # Full feature vectors (for advanced analysis)
            'phonetic_features': phonetic_features.tolist(),
            'status_features': status_features.tolist()
        }
        
        return characteristics
    
    def _calculate_memorability(self, features: np.ndarray) -> float:
        """
        Calculate memorability score from phonetic features.
        
        Memorability correlates with:
        - Moderate syllable count (not too short or long)
        - High vowel ratio
        - Rhythmic patterns
        """
        # PhoneticTransformer features include syllable count, vowel ratio, etc.
        # We'll use a heuristic based on common indices
        
        # Assuming feature vector has syllable count as first feature
        syllables = features[0] if len(features) > 0 else 2
        
        # Optimal memorability at 2-3 syllables
        syllable_score = 1.0 - abs(syllables - 2.5) / 5
        syllable_score = max(0, min(1, syllable_score))
        
        # Use mean of features as proxy for overall memorability
        feature_mean = np.mean(np.abs(features))
        
        # Combine
        memorability = (syllable_score * 0.6 + min(feature_mean / 2, 1) * 0.4) * 100
        
        return float(memorability)
    
    def _calculate_distinctiveness(self, features: np.ndarray) -> float:
        """
        Calculate how distinctive/unusual the name is.
        
        More distinctive = less common phonetic patterns.
        """
        # Distinctiveness is related to variance in features
        feature_variance = np.var(features)
        
        # Higher variance = more distinctive
        distinctiveness = min(feature_variance * 20, 100)
        
        return float(distinctiveness)
    
    def _calculate_complexity(self, features: np.ndarray) -> float:
        """
        Calculate phonetic complexity.
        
        Complex names have many consonant clusters, varied sounds, etc.
        """
        # Complexity correlates with feature magnitude
        complexity = np.linalg.norm(features) / len(features) * 10
        complexity = min(100, max(0, complexity))
        
        return float(complexity)
    
    def _calculate_euphony(self, features: np.ndarray) -> float:
        """
        Calculate euphony (how pleasant the name sounds).
        
        Euphonic names have smooth consonants, good vowel distribution.
        """
        # Euphony is inverse of harshness
        # Assuming negative features indicate harshness
        negative_features = features[features < 0]
        
        if len(negative_features) > 0:
            harshness = np.mean(np.abs(negative_features))
            euphony = (1 - min(harshness, 1)) * 100
        else:
            euphony = 70  # Neutral default
        
        return float(euphony)
    
    def _calculate_authority(self, features: np.ndarray) -> float:
        """
        Calculate authority score from social status features.
        
        High authority names sound prestigious, formal, established.
        """
        # Authority correlates with positive status features
        positive_features = features[features > 0]
        
        if len(positive_features) > 0:
            authority = np.mean(positive_features) * 20
            authority = min(100, max(0, authority))
        else:
            authority = 30  # Low default
        
        return float(authority)
    
    def _calculate_professional(self, features: np.ndarray) -> float:
        """
        Calculate professionalism score.
        
        Professional names sound serious, formal, businesslike.
        """
        # Professionalism is related to feature consistency
        feature_std = np.std(features)
        
        # Lower variance = more professional (consistent, predictable)
        professionalism = max(0, 100 - feature_std * 15)
        
        return float(professionalism)
    
    def _print_summary(self, researchers: Dict[str, Dict]):
        """Print summary statistics."""
        print(f"\nName characteristics summary:")
        
        # Collect all characteristics
        all_chars = [r['name_characteristics'] for r in researchers.values()]
        
        metrics = ['memorability', 'distinctiveness', 'authority_score', 'professional_score']
        
        for metric in metrics:
            values = [c[metric] for c in all_chars]
            print(f"\n  {metric.replace('_', ' ').title()}:")
            print(f"    Mean: {np.mean(values):.1f}")
            print(f"    Range: {np.min(values):.1f} - {np.max(values):.1f}")
            
            # Find extremes
            max_idx = np.argmax(values)
            max_name = list(researchers.keys())[max_idx]
            print(f"    Highest: {max_name} ({values[max_idx]:.1f})")


class ControlVariablesNormalizer:
    """Normalize control variables for regression analysis."""
    
    def __init__(self):
        """Initialize normalizer."""
        self.normalization_params = {}
    
    def normalize(self, researchers: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Normalize control variables.
        
        Normalizes:
        - Years since PhD
        - H-index
        - Institution prestige tier
        - Paper counts
        - Years active
        """
        print(f"\n{'='*80}")
        print("NORMALIZING CONTROL VARIABLES")
        print(f"{'='*80}\n")
        
        # Extract raw values
        variables = {
            'years_since_phd': [],
            'h_index': [],
            'institution_tier': [],
            'paper_count': [],
            'years_active_span': []
        }
        
        for data in researchers.values():
            variables['years_since_phd'].append(data.get('years_since_phd', 0) or 0)
            variables['h_index'].append(data.get('h_index', 0) or 0)
            variables['institution_tier'].append(data.get('institution_tier', 3) or 3)
            variables['paper_count'].append(data.get('paper_count', 0))
            
            years_active = data.get('years_active', {})
            span = years_active.get('span', 0) if years_active else 0
            variables['years_active_span'].append(span or 0)
        
        # Calculate normalization parameters (z-score)
        for var_name, values in variables.items():
            values_array = np.array(values)
            mean = np.mean(values_array)
            std = np.std(values_array)
            
            # Avoid division by zero
            if std < 0.001:
                std = 1.0
            
            self.normalization_params[var_name] = {
                'mean': float(mean),
                'std': float(std)
            }
            
            print(f"  {var_name}: μ={mean:.2f}, σ={std:.2f}")
        
        # Apply normalization
        for data in researchers.values():
            normalized = {}
            
            normalized['years_since_phd_norm'] = self._normalize_value(
                data.get('years_since_phd', 0) or 0,
                'years_since_phd'
            )
            
            normalized['h_index_norm'] = self._normalize_value(
                data.get('h_index', 0) or 0,
                'h_index'
            )
            
            normalized['institution_tier_norm'] = self._normalize_value(
                data.get('institution_tier', 3) or 3,
                'institution_tier'
            )
            
            normalized['paper_count_norm'] = self._normalize_value(
                data.get('paper_count', 0),
                'paper_count'
            )
            
            years_active = data.get('years_active', {})
            span = years_active.get('span', 0) if years_active else 0
            normalized['years_active_norm'] = self._normalize_value(
                span or 0,
                'years_active_span'
            )
            
            data['control_variables_normalized'] = normalized
        
        print(f"\n✓ Normalized control variables for {len(researchers)} researchers")
        
        return researchers
    
    def _normalize_value(self, value: float, var_name: str) -> float:
        """Z-score normalization."""
        params = self.normalization_params[var_name]
        return (value - params['mean']) / params['std']
    
    def save_params(self, output_path: Path):
        """Save normalization parameters."""
        with open(output_path, 'w') as f:
            json.dump(self.normalization_params, f, indent=2)
        print(f"✓ Saved normalization params to: {output_path}")


def main():
    """Extract name characteristics and normalize control variables."""
    # Load researcher metadata
    data_dir = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative'
    metadata_path = data_dir / 'researchers_metadata.json'
    
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        return
    
    with open(metadata_path) as f:
        data = json.load(f)
        researchers = data['researchers']
    
    print(f"Loaded {len(researchers)} researchers")
    
    # Extract name characteristics
    char_extractor = NameCharacteristicsExtractor()
    researchers = char_extractor.extract_characteristics(researchers)
    
    # Normalize control variables
    normalizer = ControlVariablesNormalizer()
    researchers = normalizer.normalize(researchers)
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save normalization parameters
    normalizer.save_params(data_dir / 'normalization_params.json')
    
    print(f"\n{'='*80}")
    print("✓ Feature extraction complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

