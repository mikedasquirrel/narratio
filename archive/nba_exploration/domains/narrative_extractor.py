"""
NBA Narrative Feature Extraction

Applies the 6 narrative transformers to NBA game narratives to extract
predictive features for game outcome prediction.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.relational import RelationalValueTransformer
from src.transformers.ensemble import EnsembleNarrativeTransformer


class NBANarrativeExtractor:
    """
    Extracts narrative features from NBA team/game descriptions.
    
    Applies all 6 narrative transformers to extract 100+ features
    that capture psychological signals, confidence markers, momentum
    indicators, and identity patterns.
    """
    
    def __init__(self):
        """Initialize all narrative transformers."""
        self.transformers = {
            'nominative': NominativeAnalysisTransformer(
                track_proper_nouns=True,
                track_categories=True
            ),
            'self_perception': SelfPerceptionTransformer(
                track_attribution=True,
                track_growth=True,
                track_coherence=True
            ),
            'narrative_potential': NarrativePotentialTransformer(
                track_modality=True,
                track_flexibility=True,
                track_arc_position=True
            ),
            'linguistic': LinguisticPatternsTransformer(
                track_evolution=True,
                n_segments=3
            ),
            'relational': RelationalValueTransformer(
                n_features=50,
                complementarity_threshold=0.3
            ),
            'ensemble': EnsembleNarrativeTransformer(
                n_top_terms=30,
                network_metrics=True,
                diversity_metrics=True
            )
        }
        
        self.fitted = False
        self.feature_names = []
    
    def fit(self, narratives: List[str]):
        """
        Fit all transformers to a corpus of NBA narratives.
        
        Parameters
        ----------
        narratives : list of str
            Team/game narratives for fitting
        """
        print(f"Fitting narrative transformers on {len(narratives)} samples...")
        
        for name, transformer in self.transformers.items():
            print(f"  Fitting {name}...")
            transformer.fit(narratives)
        
        # Build feature name list
        self.feature_names = self._build_feature_names()
        
        self.fitted = True
        print("✅ All transformers fitted successfully")
    
    def extract_features(self, narrative: str) -> np.ndarray:
        """
        Extract all narrative features from a single narrative.
        
        Parameters
        ----------
        narrative : str
            Team or game narrative text
        
        Returns
        -------
        features : np.ndarray
            Combined feature vector (100+ features)
        """
        if not self.fitted:
            raise ValueError("Transformers must be fitted before extraction. Call fit() first.")
        
        all_features = []
        
        for name, transformer in self.transformers.items():
            features = transformer.transform([narrative])[0]
            all_features.append(features)
        
        # Combine all features
        combined = np.concatenate(all_features)
        return combined
    
    def extract_game_features(self, home_narrative: str, away_narrative: str) -> Dict[str, Any]:
        """
        Extract features for a complete game (both teams).
        
        Parameters
        ----------
        home_narrative : str
            Home team narrative
        away_narrative : str
            Away team narrative
        
        Returns
        -------
        game_features : dict
            Contains home_features, away_features, and differential features
        """
        home_features = self.extract_features(home_narrative)
        away_features = self.extract_features(away_narrative)
        
        # Calculate differentials (home - away)
        differential = home_features - away_features
        
        return {
            'home_features': home_features,
            'away_features': away_features,
            'differential': differential,
            'combined': np.concatenate([home_features, away_features, differential])
        }
    
    def _build_feature_names(self) -> List[str]:
        """Build comprehensive list of feature names."""
        names = []
        
        # Nominative (24 features)
        semantic_fields = ['motion', 'cognition', 'emotion', 'perception', 'communication', 
                          'creation', 'change', 'possession', 'existence', 'social']
        names.extend([f'nom_{field}' for field in semantic_fields])
        names.extend(['nom_dominant', 'nom_entropy', 'nom_proper_nouns', 'nom_diversity',
                     'nom_repetition', 'nom_categories', 'nom_cat_diversity', 'nom_identity',
                     'nom_comparison', 'nom_consistency', 'nom_specificity', 'nom_categorical',
                     'nom_balance', 'nom_construction'])
        
        # Self-Perception (21 features)
        names.extend(['sp_fp_sing', 'sp_fp_plur', 'sp_focus_ratio', 'sp_pos_attr', 'sp_neg_attr',
                     'sp_attr_balance', 'sp_confidence', 'sp_growth', 'sp_stasis', 'sp_growth_mind',
                     'sp_aspirational', 'sp_descriptive', 'sp_asp_ratio', 'sp_high_agency', 'sp_low_agency',
                     'sp_agency_score', 'sp_coherence', 'sp_complexity', 'sp_awareness',
                     'sp_transformation', 'sp_positioning'])
        
        # Narrative Potential (25 features)
        names.extend(['np_future_tense', 'np_future_intent', 'np_future_orient', 'np_poss_modals',
                     'np_potential_words', 'np_poss_score', 'np_growth_verbs', 'np_change_words',
                     'np_growth_mind', 'np_flexibility', 'np_rigidity', 'np_flex_ratio',
                     'np_poss_words', 'np_constraints', 'np_net_poss', 'np_begin_phase',
                     'np_middle_phase', 'np_resolution', 'np_dominant_arc', 'np_conditional',
                     'np_alternatives', 'np_openness', 'np_temporal_breadth', 'np_actualization',
                     'np_momentum'])
        
        # Linguistic (26 features)
        names.extend(['ling_fp', 'ling_sp', 'ling_tp', 'ling_voice_entropy', 'ling_past',
                     'ling_present', 'ling_future', 'ling_temp_entropy', 'ling_active',
                     'ling_passive', 'ling_agency', 'ling_sentiment', 'ling_emotion_int',
                     'ling_subordinate', 'ling_relative', 'ling_modality', 'ling_complexity',
                     'ling_voice_trend', 'ling_temp_trend', 'ling_comp_trend', 'ling_voice_var',
                     'ling_temp_var', 'ling_comp_var', 'ling_voice_const', 'ling_temp_const',
                     'ling_comp_const'])
        
        # Relational (9 features)
        names.extend(['rel_internal_comp', 'rel_density', 'rel_synergy', 'rel_comp_potential',
                     'rel_value_ratio', 'rel_entropy', 'rel_balance', 'rel_peaks', 'rel_coherence'])
        
        # Ensemble (11 features)
        names.extend(['ens_size', 'ens_cooccur', 'ens_diversity', 'ens_avg_central',
                     'ens_max_central', 'ens_central_std', 'ens_betweenness', 'ens_coherence',
                     'ens_reach'])
        
        return names
    
    def get_feature_importance_names(self, feature_indices: np.ndarray) -> List[str]:
        """Get human-readable names for important features."""
        if not self.feature_names:
            self.feature_names = self._build_feature_names()
        
        return [self.feature_names[i] for i in feature_indices if i < len(self.feature_names)]
    
    def interpret_features(self, features: np.ndarray, team_name: str = "Team") -> Dict[str, Any]:
        """
        Provide human-readable interpretation of extracted features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature vector
        team_name : str
            Team name for personalized interpretation
        
        Returns
        -------
        interpretation : dict
            Human-readable feature interpretations
        """
        if len(features) < 10:
            return {'error': 'Insufficient features for interpretation'}
        
        interpretations = {
            'confidence_signal': self._interpret_confidence(features),
            'momentum_indicator': self._interpret_momentum(features),
            'identity_strength': self._interpret_identity(features),
            'narrative_coherence': self._interpret_coherence(features),
            'competitive_framing': self._interpret_competitive(features)
        }
        
        return interpretations
    
    def _interpret_confidence(self, features: np.ndarray) -> str:
        """Interpret confidence-related features."""
        # Self-perception confidence (index ~16)
        # Narrative potential future orientation (index ~46)
        if len(features) > 50:
            conf_score = features[40] if len(features) > 40 else 0  # sp_confidence approx
            if conf_score > 0.6:
                return "HIGH - Strong confidence markers in narrative"
            elif conf_score > 0.3:
                return "MODERATE - Balanced confidence indicators"
            else:
                return "LOW - Cautious or uncertain narrative framing"
        return "UNKNOWN - Insufficient data"
    
    def _interpret_momentum(self, features: np.ndarray) -> str:
        """Interpret momentum-related features."""
        # Narrative potential momentum (last feature)
        # Motion semantic field (first nominative feature)
        if len(features) > 68:
            momentum = features[68] if len(features) > 68 else 0  # np_momentum approx
            if momentum > 0.1:
                return "POSITIVE - Forward momentum language detected"
            elif momentum < -0.1:
                return "NEGATIVE - Backward-looking or regressive framing"
            else:
                return "NEUTRAL - Balanced temporal orientation"
        return "UNKNOWN"
    
    def _interpret_identity(self, features: np.ndarray) -> str:
        """Interpret identity construction features."""
        # Nominative identity construction (index ~23)
        if len(features) > 23:
            identity = features[23]
            if identity > 0.15:
                return "STRONG - Clear identity and self-definition"
            elif identity > 0.08:
                return "MODERATE - Developing identity"
            else:
                return "WEAK - Unclear or unstable identity"
        return "UNKNOWN"
    
    def _interpret_coherence(self, features: np.ndarray) -> str:
        """Interpret narrative coherence."""
        # Self-perception coherence (index ~40)
        if len(features) > 40:
            coherence = features[40] if len(features) > 40 else 0
            if coherence > 0.7:
                return "HIGH - Consistent narrative throughout"
            elif coherence > 0.4:
                return "MODERATE - Some narrative shifts"
            else:
                return "LOW - Inconsistent or fragmented narrative"
        return "UNKNOWN"
    
    def _interpret_competitive(self, features: np.ndarray) -> str:
        """Interpret competitive framing."""
        # Motion semantic field (index 0)
        # Agency score (index ~39)
        if len(features) > 10:
            motion = features[0] if len(features) > 0 else 0
            if motion > 0.12:
                return "AGGRESSIVE - High action/motion language, competitive framing"
            elif motion > 0.06:
                return "MODERATE - Balanced competitive language"
            else:
                return "PASSIVE - Limited competitive framing"
        return "UNKNOWN"

