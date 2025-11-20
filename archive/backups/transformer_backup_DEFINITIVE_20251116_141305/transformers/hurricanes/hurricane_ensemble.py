"""
Hurricane Ensemble Transformer

Combines nominative, weather narrative, and severity features
to test interaction effects and create comprehensive prediction model.

Tests hypothesis: Combined narrative features (name + context + severity)
outperform any single feature set alone.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional

# Add narrative_optimization to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..base_transformer import FeatureNarrativeTransformer
from .nominative_hurricane import HurricaneNominativeTransformer
from .weather_narrative import WeatherNarrativeTransformer


class HurricaneEnsembleTransformer(FeatureNarrativeTransformer):
    """
    Ensemble transformer combining multiple narrative perspectives.
    
    Integrates:
    1. Nominative features (name characteristics)
    2. Weather narrative (temporal/geographic context)
    3. Actual severity (control variable)
    4. Interaction effects
    """
    
    def __init__(self, include_interactions: bool = True,
                 include_severity: bool = True,
                 severity_weight: float = 0.5):
        """
        Initialize hurricane ensemble transformer.
        
        Parameters
        ----------
        include_interactions : bool
            Include interaction terms between feature sets
        include_severity : bool
            Include actual severity as control variable
        severity_weight : float
            Weight for severity vs. perception features (0-1)
        """
        super().__init__(
            narrative_id="hurricane_ensemble",
            description="Combines nominative, contextual, and severity features "
                       "to predict hurricane perception and response"
        )
        self.include_interactions = include_interactions
        self.include_severity = include_severity
        self.severity_weight = severity_weight
        
        # Sub-transformers
        self.nominative_transformer = HurricaneNominativeTransformer(
            include_interactions=True,
            normalize_features=True
        )
        self.weather_transformer = WeatherNarrativeTransformer(
            include_temporal=True,
            include_geographic=True
        )
    
    def fit(self, X, y=None):
        """
        Fit all sub-transformers and learn interaction patterns.
        
        Parameters
        ----------
        X : list of dict
            Hurricane records with names and metadata
        y : array-like, optional
            Target outcomes
        
        Returns
        -------
        self
        """
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be non-empty list of hurricane dictionaries")
        
        # Extract names for nominative transformer
        names = [h.get('name', '') for h in X]
        
        # Fit sub-transformers
        self.nominative_transformer.fit(names, y)
        self.weather_transformer.fit(X, y)
        
        # Store ensemble metadata
        self.metadata['n_hurricanes'] = len(X)
        self.metadata['nominative_features'] = self.nominative_transformer.metadata['n_features']
        self.metadata['weather_features'] = self.weather_transformer.metadata['n_features']
        
        # Calculate feature statistics
        if self.include_severity:
            severities = [h.get('actual_severity', {}).get('category', 3) for h in X]
            self.metadata['severity_stats'] = {
                'mean': np.mean(severities),
                'std': np.std(severities),
                'range': (min(severities), max(severities))
            }
        
        # Store feature metadata
        self.metadata['n_features'] = self._calculate_n_features()
        self.metadata['feature_names'] = self._generate_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform to ensemble feature representation.
        
        Parameters
        ----------
        X : list of dict
            Hurricane records
        
        Returns
        -------
        numpy.ndarray
            Combined feature matrix
        """
        self._validate_fitted()
        
        # Extract names
        names = [h.get('name', '') for h in X]
        
        # Get features from sub-transformers
        nominative_features = self.nominative_transformer.transform(names)
        weather_features = self.weather_transformer.transform(X)
        
        # Combine features
        combined = []
        
        for i, hurricane in enumerate(X):
            feature_vec = []
            
            # 1. Nominative features
            feature_vec.extend(nominative_features[i])
            
            # 2. Weather narrative features
            feature_vec.extend(weather_features[i])
            
            # 3. Severity features (if enabled)
            if self.include_severity:
                severity_feats = self._extract_severity_features(hurricane)
                feature_vec.extend(severity_feats)
            
            # 4. Interaction features (if enabled)
            if self.include_interactions:
                interaction_feats = self._calculate_interactions(
                    nominative_features[i],
                    weather_features[i],
                    hurricane
                )
                feature_vec.extend(interaction_feats)
            
            combined.append(feature_vec)
        
        return np.array(combined, dtype=np.float32)
    
    def _extract_severity_features(self, hurricane: Dict) -> List[float]:
        """Extract actual severity features."""
        severity = hurricane.get('actual_severity', {})
        
        features = []
        
        # Category
        category = severity.get('category', 3)
        features.append(float(category) / 5.0)
        
        # Wind speed (normalized)
        wind = severity.get('max_wind_speed_mph', 100)
        features.append((wind - 74) / 126)  # 74-200 mph range
        
        # Pressure (normalized, inverted)
        pressure = severity.get('min_pressure_mb', 970)
        features.append((1013 - pressure) / 131)  # Lower pressure = higher intensity
        
        # Duration (log-normalized)
        duration = severity.get('duration_hours', 24)
        features.append(np.log(duration + 1) / np.log(200))
        
        # Accumulated energy
        energy = severity.get('accumulated_energy', wind**2 * duration / 100)
        features.append(np.log(energy + 1) / 15)  # Log scale, rough normalization
        
        return features
    
    def _calculate_interactions(self, nominative_feats: np.ndarray,
                               weather_feats: np.ndarray,
                               hurricane: Dict) -> List[float]:
        """
        Calculate interaction features.
        
        Tests hypotheses like:
        - Gender effect stronger for moderate storms?
        - Memorability matters more in recent era?
        - Regional experience moderates name effects?
        """
        interactions = []
        
        # Get key features
        # Nominative: gender_rating is feature 0
        gender_rating = nominative_feats[0]
        # Nominative: memorability is feature 8 (after normalize)
        memorability = nominative_feats[8] if len(nominative_feats) > 8 else 0.5
        
        # Weather: year_normalized is feature 0 (if temporal enabled)
        year_norm = weather_feats[0] if len(weather_feats) > 0 else 0.5
        
        # Severity
        severity = hurricane.get('actual_severity', {})
        category = severity.get('category', 3)
        category_norm = category / 5.0
        
        # Interaction 1: Gender × Severity
        # Hypothesis: Name gender matters more for moderate storms
        # (Very severe storms override perception, very weak storms don't matter)
        gender_severity = gender_rating * category_norm
        interactions.append(gender_severity)
        
        # Interaction 2: Gender × Era
        # Hypothesis: Gender bias may be stronger in modern era with more media
        gender_era = gender_rating * year_norm
        interactions.append(gender_era)
        
        # Interaction 3: Memorability × Severity
        # Hypothesis: Memorable names help more for severe storms
        memo_severity = memorability * category_norm
        interactions.append(memo_severity)
        
        # Interaction 4: Gender × Memorability
        # Hypothesis: Memorable feminine names may reduce bias
        gender_memo = gender_rating * memorability
        interactions.append(gender_memo)
        
        # Interaction 5: Quadratic severity
        # Test if perception effects are nonlinear with severity
        severity_squared = category_norm ** 2
        interactions.append(severity_squared)
        
        # Interaction 6: Gender × Severity² (peak effect at moderate severity)
        gender_severity_quad = gender_rating * severity_squared
        interactions.append(gender_severity_quad)
        
        return interactions
    
    def _calculate_n_features(self) -> int:
        """Calculate total number of ensemble features."""
        n = 0
        n += self.metadata['nominative_features']
        n += self.metadata['weather_features']
        
        if self.include_severity:
            n += 5  # Severity features
        
        if self.include_interactions:
            n += 6  # Interaction features
        
        return n
    
    def _generate_feature_names(self) -> List[str]:
        """Generate comprehensive feature names."""
        names = []
        
        # Nominative features (prefixed)
        nom_names = self.nominative_transformer.get_feature_names()
        names.extend([f"nom_{name}" for name in nom_names])
        
        # Weather features (prefixed)
        weather_names = self.weather_transformer.get_feature_names()
        names.extend([f"weather_{name}" for name in weather_names])
        
        # Severity features
        if self.include_severity:
            names.extend([
                'severity_category_norm',
                'severity_wind_norm',
                'severity_pressure_norm',
                'severity_duration_norm',
                'severity_energy_norm'
            ])
        
        # Interaction features
        if self.include_interactions:
            names.extend([
                'interact_gender_x_severity',
                'interact_gender_x_era',
                'interact_memo_x_severity',
                'interact_gender_x_memo',
                'interact_severity_squared',
                'interact_gender_x_severity_squared'
            ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of ensemble model."""
        interpretation = f"Hurricane Ensemble Model ({self.metadata['n_hurricanes']} storms)\n\n"
        
        interpretation += "Feature Groups:\n"
        interpretation += f"  Nominative features: {self.metadata['nominative_features']}\n"
        interpretation += f"  Weather narrative features: {self.metadata['weather_features']}\n"
        
        if self.include_severity:
            interpretation += f"  Severity features: 5\n"
        
        if self.include_interactions:
            interpretation += f"  Interaction features: 6\n"
        
        interpretation += f"\nTotal features: {self.metadata['n_features']}\n\n"
        
        interpretation += "Model Philosophy:\n"
        interpretation += "  This ensemble combines multiple narrative perspectives to\n"
        interpretation += "  predict hurricane perception and response. By including both\n"
        interpretation += "  name-based features (nominative) and contextual features\n"
        interpretation += "  (temporal, geographic, historical), we can test whether\n"
        interpretation += "  perception effects persist after controlling for actual severity.\n\n"
        
        interpretation += "Key Hypotheses:\n"
        interpretation += "  H1: Name gender predicts evacuation (controlling for severity)\n"
        interpretation += "  H2: Effect is stronger for moderate-intensity storms\n"
        interpretation += "  H3: Memorability moderates gender bias\n"
        interpretation += "  H4: Combined model outperforms name-only or severity-only\n\n"
        
        interpretation += "Sub-transformer Interpretations:\n"
        interpretation += "=" * 60 + "\n\n"
        interpretation += "NOMINATIVE TRANSFORMER:\n"
        interpretation += self.nominative_transformer.get_interpretation()
        interpretation += "\n\n" + "=" * 60 + "\n\n"
        interpretation += "WEATHER NARRATIVE TRANSFORMER:\n"
        interpretation += self.weather_transformer.get_interpretation()
        
        return interpretation
    
    def predict_evacuation_rate(self, hurricane: Dict) -> Dict[str, Any]:
        """
        Predict evacuation rate for a hurricane.
        
        Parameters
        ----------
        hurricane : dict
            Hurricane data including name, severity, metadata
        
        Returns
        -------
        dict
            Prediction with breakdown by feature group
        """
        self._validate_fitted()
        
        # Transform to features
        features = self.transform([hurricane])[0]
        
        # Get feature group contributions
        n_nom = self.metadata['nominative_features']
        n_weather = self.metadata['weather_features']
        
        nominative_feats = features[:n_nom]
        weather_feats = features[n_nom:n_nom+n_weather]
        
        # Simple linear prediction (in practice, would use trained model)
        # This is illustrative - actual prediction would come from fitted model
        
        # Extract key features for interpretation
        gender_rating = nominative_feats[0]  # Normalized gender
        category = hurricane.get('actual_severity', {}).get('category', 3)
        
        # Base evacuation from severity
        base_evac = 0.3 + (category - 1) * 0.15
        
        # Gender adjustment (from research: -8.2% for feminine)
        gender_adjustment = -0.082 * gender_rating
        
        # Final prediction
        predicted_evac = base_evac + gender_adjustment
        predicted_evac = max(0.1, min(0.95, predicted_evac))
        
        return {
            'predicted_evacuation_rate': predicted_evac,
            'base_from_severity': base_evac,
            'gender_adjustment': gender_adjustment,
            'gender_effect_pct': gender_adjustment * 100,
            'interpretation': f"Severity suggests {base_evac:.1%} evacuation, "
                            f"but name gender adjusts this by {gender_adjustment:+.1%}, "
                            f"yielding final prediction of {predicted_evac:.1%}"
        }
    
    def _validate_input(self, X):
        """Validate input format."""
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be non-empty list of hurricane dictionaries")
        
        if not isinstance(X[0], dict):
            raise ValueError("X must contain dictionaries with hurricane data")
        
        return True

