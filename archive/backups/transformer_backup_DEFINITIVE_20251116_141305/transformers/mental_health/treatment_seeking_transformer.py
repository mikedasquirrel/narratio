"""
Treatment Seeking Transformer

Combines phonetic and framing features to predict treatment-seeking behavior.

Key Pathway: Harsh name → High stigma → Low treatment seeking → Worse outcomes

Author: Narrative Optimization Research
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.transformers.base_transformer import FeatureNarrativeTransformer
from .phonetic_severity_transformer import PhoneticSeverityTransformer
from .clinical_framing_transformer import ClinicalFramingTransformer


class TreatmentSeekingTransformer(FeatureNarrativeTransformer):
    """
    Combined transformer predicting treatment seeking from disorder names.
    
    Integrates:
    - Phonetic severity (harshness)
    - Clinical framing (medical vs accessible)
    - Interaction effects
    """
    
    def __init__(self, include_interactions: bool = True):
        """
        Initialize treatment seeking transformer.
        
        Parameters
        ----------
        include_interactions : bool
            Include interaction between phonetic and framing features
        """
        super().__init__(
            narrative_id="treatment_seeking",
            description="Predicts treatment-seeking behavior from disorder name features"
        )
        self.include_interactions = include_interactions
        
        # Sub-transformers
        self.phonetic_transformer = PhoneticSeverityTransformer(include_interactions=False)
        self.framing_transformer = ClinicalFramingTransformer()
    
    def fit(self, X, y=None):
        """
        Fit all sub-transformers.
        
        Parameters
        ----------
        X : list of str
            Disorder names
        y : array-like, optional
            Treatment seeking rates
        
        Returns
        -------
        self
        """
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be non-empty list of disorder names")
        
        # Fit sub-transformers
        self.phonetic_transformer.fit(X, y)
        self.framing_transformer.fit(X, y)
        
        # Store metadata
        self.metadata['n_disorders'] = len(X)
        self.metadata['phonetic_features'] = self.phonetic_transformer.metadata['n_features']
        self.metadata['framing_features'] = self.framing_transformer.metadata['n_features']
        
        self.metadata['n_features'] = self._calculate_n_features()
        self.metadata['feature_names'] = self._generate_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform to combined feature representation.
        
        Parameters
        ----------
        X : list of str
            Disorder names
        
        Returns
        -------
        numpy.ndarray
            Combined feature matrix
        """
        self._validate_fitted()
        
        # Get features from sub-transformers
        phonetic_features = self.phonetic_transformer.transform(X)
        framing_features = self.framing_transformer.transform(X)
        
        # Combine
        combined = []
        
        for i, name in enumerate(X):
            feature_vec = []
            
            # 1. Phonetic features
            feature_vec.extend(phonetic_features[i])
            
            # 2. Framing features
            feature_vec.extend(framing_features[i])
            
            # 3. Interaction features (if enabled)
            if self.include_interactions:
                interaction_feats = self._calculate_interactions(
                    phonetic_features[i],
                    framing_features[i]
                )
                feature_vec.extend(interaction_feats)
            
            combined.append(feature_vec)
        
        return np.array(combined, dtype=np.float32)
    
    def _calculate_interactions(self, phonetic_feats: np.ndarray,
                               framing_feats: np.ndarray) -> List[float]:
        """
        Calculate interaction features.
        
        Tests hypotheses about combined effects of phonetics and framing.
        """
        interactions = []
        
        # Extract key features
        harshness = phonetic_feats[0]  # First feature is harshness
        clinical_score = framing_feats[0]  # First feature is clinical score
        accessibility = framing_feats[1]  # Second is accessibility
        
        # Interaction 1: Harsh × Medical framing
        # Double barrier: harsh AND overly medical
        harsh_medical = (harshness / 100) * (clinical_score / 100)
        interactions.append(harsh_medical)
        
        # Interaction 2: Harshness × (Low accessibility)
        # Harsh name + inaccessible terminology
        harsh_inaccessible = (harshness / 100) * (1 - accessibility / 100)
        interactions.append(harsh_inaccessible)
        
        # Interaction 3: Barrier multiplier
        # Combined barriers multiply rather than add
        barrier_multiplier = harsh_medical * harsh_inaccessible
        interactions.append(barrier_multiplier)
        
        return interactions
    
    def _calculate_n_features(self) -> int:
        """Calculate total feature count."""
        n = (self.metadata['phonetic_features'] + 
             self.metadata['framing_features'])
        
        if self.include_interactions:
            n += 3  # Interaction features
        
        return n
    
    def _generate_feature_names(self) -> List[str]:
        """Generate comprehensive feature names."""
        names = []
        
        # Phonetic features (prefixed)
        phon_names = self.phonetic_transformer.get_feature_names()
        names.extend([f"phon_{name}" for name in phon_names])
        
        # Framing features (prefixed)
        frame_names = self.framing_transformer.get_feature_names()
        names.extend([f"frame_{name}" for name in frame_names])
        
        # Interaction features
        if self.include_interactions:
            names.extend([
                'interact_harsh_x_medical',
                'interact_harsh_x_inaccessible',
                'interact_barrier_multiplier'
            ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of combined model."""
        interpretation = f"""
Treatment Seeking Prediction Model ({self.metadata['n_disorders']} disorders)

FEATURE COMPOSITION
-------------------
Phonetic Features: {self.metadata['phonetic_features']}
Framing Features: {self.metadata['framing_features']}
Interaction Features: {3 if self.include_interactions else 0}
Total Features: {self.metadata['n_features']}

SUB-TRANSFORMER INSIGHTS
------------------------

PHONETIC ANALYSIS:
{self.phonetic_transformer.get_interpretation()}

FRAMING ANALYSIS:
{self.framing_transformer.get_interpretation()}

COMBINED HYPOTHESIS
-------------------
Disorder names create barriers to treatment seeking through:
1. Phonetic harshness → Perceived severity ↑ → Fear/avoidance
2. Medical framing → Clinical distance → "Not for me" feeling
3. Combined effect → Multiple barriers → Reduced help-seeking

Expected pathway:
Harsh + Medical name → High stigma → Low treatment seeking → Delayed care → Worse outcomes
"""
        
        return interpretation
    
    def predict_treatment_barrier(self, disorder_name: str) -> Dict:
        """
        Predict treatment barrier score for a disorder name.
        
        Parameters
        ----------
        disorder_name : str
            Name of disorder
        
        Returns
        -------
        dict
            Barrier scores and recommendations
        """
        self._validate_fitted()
        
        # Get sub-predictions
        stigma_pred = self.phonetic_transformer.predict_stigma_from_name(disorder_name)
        
        phonetic_features = self.phonetic_transformer.transform([disorder_name])[0]
        framing_features = self.framing_transformer.transform([disorder_name])[0]
        
        harshness = phonetic_features[0]
        clinical_score = framing_features[0]
        accessibility = framing_features[1]
        
        # Calculate barrier components
        phonetic_barrier = harshness / 100
        framing_barrier = clinical_score / 100
        accessibility_bonus = accessibility / 100
        
        # Combined barrier (multiplicative)
        total_barrier = phonetic_barrier * framing_barrier * (1 - accessibility_bonus)
        total_barrier = min(1.0, total_barrier * 1.5)  # Scale up
        
        # Estimate treatment seeking probability
        base_seeking = 0.60  # 60% baseline treatment seeking
        barrier_effect = -0.30 * total_barrier  # Up to 30% reduction
        predicted_seeking = max(0.2, base_seeking + barrier_effect)
        
        return {
            'disorder_name': disorder_name,
            'total_barrier_score': total_barrier,
            'predicted_treatment_seeking_rate': predicted_seeking,
            'phonetic_barrier': phonetic_barrier,
            'framing_barrier': framing_barrier,
            'accessibility_score': accessibility,
            'harshness_score': harshness,
            'clinical_score': clinical_score,
            'recommendations': self._generate_recommendations(
                disorder_name, harshness, clinical_score, accessibility
            )
        }
    
    def _generate_recommendations(self, name: str, harshness: float,
                                 clinical: float, accessibility: float) -> List[str]:
        """Generate recommendations to reduce barriers."""
        recommendations = []
        
        if harshness > 60:
            recommendations.append(
                "Consider less harsh terminology alternatives"
            )
        
        if clinical > 70:
            recommendations.append(
                "Use more accessible colloquial terms in patient communication"
            )
        
        if accessibility < 40:
            recommendations.append(
                "Simplify terminology for public-facing materials"
            )
        
        if clinical > 60 and harshness > 60:
            recommendations.append(
                "⚠️ DOUBLE BARRIER: Both harsh phonetics and medical framing create barriers"
            )
        
        if not recommendations:
            recommendations.append(
                "✅ Name is relatively accessible with minimal phonetic barriers"
            )
        
        return recommendations
    
    def _validate_input(self, X):
        """Validate input is list of strings."""
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be non-empty list of disorder names (strings)")
        return True


if __name__ == '__main__':
    # Demo
    transformer = TreatmentSeekingTransformer()
    
    disorders = [
        'Major Depressive Disorder',
        'Depression',
        'Schizophrenia',
        'Anxiety',
        'PTSD'
    ]
    
    transformer.fit(disorders)
    
    print(transformer.get_interpretation())
    
    print("\n" + "="*70)
    print("TREATMENT BARRIER PREDICTIONS")
    print("="*70 + "\n")
    
    for disorder in disorders:
        pred = transformer.predict_treatment_barrier(disorder)
        print(f"{disorder}:")
        print(f"  Barrier Score: {pred['total_barrier_score']:.2f}")
        print(f"  Predicted Seeking: {pred['predicted_treatment_seeking_rate']:.1%}")
        print(f"  Recommendations:")
        for rec in pred['recommendations']:
            print(f"    • {rec}")
        print()

