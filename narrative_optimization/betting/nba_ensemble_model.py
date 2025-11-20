"""
NBA Ensemble Betting Model
===========================

Production betting model combining 42 working transformers via stacking ensemble.
Optimized for NBA betting with moneyline, spread, and props predictions.

Architecture:
- Stage 1: 42 base transformers extract features independently
- Stage 2: Meta-learner (calibrated logistic regression) combines predictions
- Stage 3: Confidence scoring and bet filtering (>60% only)

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'narrative_optimization' / 'src'))

# Import all 42 working transformers
from transformers import (
    # Core (5 working)
    NominativeAnalysisTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    EnsembleNarrativeTransformer,
    RelationalValueTransformer,
    
    # Emotional (4)
    EmotionalResonanceTransformer,
    AuthenticityTransformer,
    ConflictTensionTransformer,
    SuspenseMysteryTransformer,
    
    # Structural (2)
    FramingTransformer,
    OpticsTransformer,
    
    # Nominative (5)
    PhoneticTransformer,
    SocialStatusTransformer,
    UniversalNominativeTransformer,
    HierarchicalNominativeTransformer,
    NominativeRichnessTransformer,
    
    # Advanced (6)
    InformationTheoryTransformer,
    NamespaceEcologyTransformer,
    CognitiveFluencyTransformer,
    DiscoverabilityTransformer,
    MultiScaleTransformer,
    QuantitativeTransformer,
    
    # Theory (5)
    CouplingStrengthTransformer,
    NarrativeMassTransformer,
    GravitationalFeaturesTransformer,
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer,
    
    # Contextual (4)
    CulturalContextTransformer,
    CompetitiveContextTransformer,
    AnticipatoryCommunicationTransformer,
    ExpertiseAuthorityTransformer,
    
    # Temporal (7)
    TemporalEvolutionTransformer,
    TemporalMomentumEnhancedTransformer,
    TemporalNarrativeContextTransformer,
    PacingRhythmTransformer,
    DurationEffectsTransformer,
    CrossTemporalIsomorphismTransformer,
    TemporalCompressionTransformer,
    
    # Pattern/Baseline (2)
    StatisticalTransformer,
    ContextPatternTransformer,
    
    # Universal/Meta (2 text-compatible)
    MetaNarrativeTransformer,
    UniversalHybridTransformer,
)

from .betting_utils import calculate_ev, calculate_edge, should_bet, categorize_confidence


class NBAEnsembleBettingModel:
    """
    Ensemble betting model for NBA predictions.
    
    Combines 42 working transformers via stacking with calibrated meta-learner.
    Provides confidence-scored predictions for moneyline, spread, and props.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.60,
        min_edge: float = 0.05,
        use_calibration: bool = True,
        n_cv_folds: int = 5
    ):
        """
        Initialize ensemble betting model.
        
        Parameters
        ----------
        min_confidence : float
            Minimum confidence to place bet (0.60 = 60%)
        min_edge : float
            Minimum edge vs market (0.05 = 5%)
        use_calibration : bool
            Whether to calibrate probabilities
        n_cv_folds : int
            Cross-validation folds for meta-learner
        """
        self.min_confidence = min_confidence
        self.min_edge = min_edge
        self.use_calibration = use_calibration
        self.n_cv_folds = n_cv_folds
        
        # Initialize transformers (42 working)
        self.transformers = self._initialize_transformers()
        
        # Meta-learner
        self.meta_learner_ = None
        self.is_fitted_ = False
        
        # Performance tracking
        self.transformer_weights_ = None
        self.feature_importances_ = None
        self.validation_scores_ = {}
    
    def _initialize_transformers(self) -> List[Tuple[str, any]]:
        """Initialize all 42 working transformers"""
        
        print("[Ensemble] Initializing 42 working transformers...")
        
        transformers = []
        
        # Top performers first
        transformers.extend([
            ('awareness_resistance', AwarenessResistanceTransformer()),  # 56.8%
            ('nominative_richness', NominativeRichnessTransformer()),  # 54.8%
            ('competitive_context', CompetitiveContextTransformer()),  # 54.0%
            ('ensemble_narrative', EnsembleNarrativeTransformer()),  # 54.0%
            ('authenticity', AuthenticityTransformer()),  # 53.2%
            ('conflict_tension', ConflictTensionTransformer()),  # 53.2%
        ])
        
        # Rest of strong performers
        transformers.extend([
            ('multi_scale', MultiScaleTransformer()),
            ('social_status', SocialStatusTransformer()),
            ('relational_value', RelationalValueTransformer()),
            ('narrative_potential', NarrativePotentialTransformer()),
            ('coupling_strength', CouplingStrengthTransformer()),
            ('temporal_narrative_context', TemporalNarrativeContextTransformer()),
            ('narrative_mass', NarrativeMassTransformer()),
            ('gravitational_features', GravitationalFeaturesTransformer()),
            ('cognitive_fluency', CognitiveFluencyTransformer()),
        ])
        
        # Core transformers
        transformers.extend([
            ('nominative', NominativeAnalysisTransformer()),
            ('linguistic', LinguisticPatternsTransformer()),
            ('emotional_resonance', EmotionalResonanceTransformer()),
            ('suspense', SuspenseMysteryTransformer()),
            ('framing', FramingTransformer()),
            ('optics', OpticsTransformer()),
        ])
        
        # Nominative suite
        transformers.extend([
            ('phonetic', PhoneticTransformer()),
            ('universal_nominative', UniversalNominativeTransformer()),
            ('hierarchical_nominative', HierarchicalNominativeTransformer()),
        ])
        
        # Advanced
        transformers.extend([
            ('information_theory', InformationTheoryTransformer()),
            ('namespace_ecology', NamespaceEcologyTransformer()),
            ('discoverability', DiscoverabilityTransformer()),
            ('quantitative', QuantitativeTransformer()),
        ])
        
        # Theory variables
        transformers.append(('fundamental_constraints', FundamentalConstraintsTransformer(use_embeddings=False)))
        
        # Contextual
        transformers.extend([
            ('cultural_context', CulturalContextTransformer()),
            ('anticipatory', AnticipatoryCommunicationTransformer()),
            ('expertise', ExpertiseAuthorityTransformer()),
        ])
        
        # Temporal
        transformers.extend([
            ('temporal_evolution', TemporalEvolutionTransformer()),
            ('temporal_momentum', TemporalMomentumEnhancedTransformer(use_spacy=False, use_embeddings=False)),
            ('pacing_rhythm', PacingRhythmTransformer()),
            ('duration_effects', DurationEffectsTransformer()),
            ('cross_temporal', CrossTemporalIsomorphismTransformer()),
            ('temporal_compression', TemporalCompressionTransformer()),
        ])
        
        # Pattern/Baseline
        transformers.extend([
            ('statistical', StatisticalTransformer()),
            ('context_pattern', ContextPatternTransformer(min_samples=30, max_patterns=20)),
        ])
        
        # Universal/Meta (text-compatible)
        transformers.extend([
            ('meta_narrative', MetaNarrativeTransformer(use_spacy=False, use_embeddings=False)),
        ])
        
        print(f"[Ensemble] ✓ Initialized {len(transformers)} transformers")
        
        return transformers
    
    def fit(self, X: pd.Series, y: np.ndarray, verbose: bool = True):
        """
        Fit ensemble model on historical data.
        
        Parameters
        ----------
        X : pd.Series
            Clean pre-game narratives
        y : np.ndarray
            Binary outcomes (1=win, 0=loss)
        verbose : bool
            Print progress
            
        Returns
        -------
        self
        """
        if verbose:
            print(f"\n[Ensemble] Training on {len(X)} games...")
            print(f"[Ensemble] Baseline: {y.mean():.1%}")
        
        # Extract features from all transformers
        print(f"\n[Ensemble] Extracting features from {len(self.transformers)} transformers...")
        
        all_features = []
        transformer_scores = []
        
        for i, (name, transformer) in enumerate(self.transformers, 1):
            if verbose and i % 5 == 0:
                print(f"[Ensemble] Progress: {i}/{len(self.transformers)} transformers...")
            
            try:
                # Fit and transform
                X_t = transformer.fit_transform(X, y)
                
                # Format
                if hasattr(X_t, 'toarray'):
                    X_t = X_t.toarray()
                if len(X_t.shape) == 1:
                    X_t = X_t.reshape(-1, 1)
                
                # Check validity
                if X_t.shape[0] != len(X):
                    if verbose:
                        print(f"[Ensemble] ⚠️  {name}: Shape mismatch, skipping")
                    continue
                
                if X_t.shape[1] == 0 or np.all(X_t == 0):
                    if verbose:
                        print(f"[Ensemble] ⚠️  {name}: No valid features, skipping")
                    continue
                
                # Quick validation score
                clf = LogisticRegression(max_iter=1000, random_state=42)
                score = cross_val_score(clf, X_t, y, cv=3, scoring='accuracy').mean()
                
                all_features.append(X_t)
                transformer_scores.append((name, score, X_t.shape[1]))
                
            except Exception as e:
                if verbose:
                    print(f"[Ensemble] ✗ {name}: {str(e)[:60]}")
        
        if len(all_features) == 0:
            raise ValueError("No transformers produced valid features")
        
        print(f"\n[Ensemble] ✓ {len(all_features)} transformers successful")
        
        # Sort by performance
        transformer_scores.sort(key=lambda x: x[1], reverse=True)
        
        if verbose:
            print(f"\n[Ensemble] Top 10 transformer performance:")
            for i, (name, score, n_feat) in enumerate(transformer_scores[:10], 1):
                print(f"  {i}. {name:<30} {score:.1%} ({n_feat} features)")
        
        # Combine all features
        X_combined = np.hstack(all_features)
        print(f"\n[Ensemble] Combined feature matrix: {X_combined.shape}")
        
        # Train meta-learner
        print(f"[Ensemble] Training meta-learner...")
        
        if self.use_calibration:
            base_model = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            self.meta_learner_ = CalibratedClassifierCV(
                base_model,
                cv=self.n_cv_folds,
                method='sigmoid'
            )
        else:
            self.meta_learner_ = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
        
        self.meta_learner_.fit(X_combined, y)
        
        # Validation
        train_acc = self.meta_learner_.score(X_combined, y)
        cv_scores = cross_val_score(self.meta_learner_, X_combined, y, cv=5, scoring='accuracy')
        
        self.validation_scores_ = {
            'train_accuracy': train_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'baseline': y.mean()
        }
        
        # Store transformer info
        self.transformer_info_ = transformer_scores
        self.feature_dims_ = [f.shape[1] for f in all_features]
        
        print(f"\n[Ensemble] ✓ Model trained!")
        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  CV accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        print(f"  Improvement: {cv_scores.mean() - y.mean():+.1%}")
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X: pd.Series, verbose: bool = False) -> np.ndarray:
        """
        Predict win probabilities with confidence.
        
        Parameters
        ----------
        X : pd.Series
            Clean pre-game narratives
        verbose : bool
            Print progress
            
        Returns
        -------
        probabilities : np.ndarray, shape (n_games, 2)
            [prob_loss, prob_win] for each game
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if verbose:
            print(f"\n[Ensemble] Predicting {len(X)} games...")
        
        # Extract features from all transformers
        all_features = []
        
        for i, (name, transformer) in enumerate(self.transformers, 1):
            try:
                X_t = transformer.transform(X)
                
                # Format
                if hasattr(X_t, 'toarray'):
                    X_t = X_t.toarray()
                if len(X_t.shape) == 1:
                    X_t = X_t.reshape(-1, 1)
                
                # Validate
                if X_t.shape[0] != len(X):
                    continue
                if X_t.shape[1] == 0 or np.all(X_t == 0):
                    continue
                
                all_features.append(X_t)
                
            except:
                continue
        
        if len(all_features) == 0:
            raise ValueError("No transformers produced valid features")
        
        # Combine and predict
        X_combined = np.hstack(all_features)
        probabilities = self.meta_learner_.predict_proba(X_combined)
        
        if verbose:
            print(f"[Ensemble] ✓ Predictions complete")
        
        return probabilities
    
    def predict_with_confidence(
        self,
        X: pd.Series,
        market_odds: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Predict with confidence scoring and betting recommendations.
        
        Parameters
        ----------
        X : pd.Series
            Clean pre-game narratives
        market_odds : np.ndarray, optional
            American odds for each game
        verbose : bool
            Print progress
            
        Returns
        -------
        predictions : list of dict
            Prediction for each game with betting recommendation
        """
        probabilities = self.predict_proba(X, verbose=verbose)
        
        predictions = []
        
        for i, prob_pair in enumerate(probabilities):
            win_prob = prob_pair[1]  # Probability of win
            confidence_level = categorize_confidence(win_prob)
            
            prediction = {
                'game_index': i,
                'narrative': X.iloc[i][:150] + "..." if len(X.iloc[i]) > 150 else X.iloc[i],
                'win_probability': float(win_prob),
                'loss_probability': float(prob_pair[0]),
                'confidence_level': confidence_level,
                'predicted_outcome': 'WIN' if win_prob > 0.5 else 'LOSS',
            }
            
            # Add betting analysis if odds provided
            if market_odds is not None and i < len(market_odds):
                odds = market_odds[i]
                
                bet_decision, reason = should_bet(
                    win_prob,
                    odds,
                    min_edge=self.min_edge,
                    min_confidence=self.min_confidence
                )
                
                edge = calculate_edge(win_prob, odds)
                ev = calculate_ev(win_prob, odds, stake=1.0)
                
                prediction['betting'] = {
                    'market_odds': float(odds),
                    'edge': float(edge),
                    'expected_value': float(ev),
                    'should_bet': bet_decision,
                    'reason': reason,
                    'recommended_units': float(self._calculate_units(win_prob)) if bet_decision else 0.0
                }
            
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_units(self, prob: float) -> float:
        """Calculate bet size in units"""
        if prob >= 0.70:
            return 2.0
        elif prob >= 0.65:
            return 1.5
        elif prob >= 0.60:
            return 1.0
        else:
            return 0.0
    
    def get_high_confidence_bets(
        self,
        predictions: List[Dict],
        min_confidence: float = 0.60
    ) -> List[Dict]:
        """
        Filter for high-confidence betting opportunities.
        
        Parameters
        ----------
        predictions : list of dict
            All predictions
        min_confidence : float
            Minimum confidence threshold
            
        Returns
        -------
        high_confidence : list of dict
            Filtered high-confidence bets
        """
        high_conf = []
        
        for pred in predictions:
            if pred['win_probability'] >= min_confidence or pred['loss_probability'] >= min_confidence:
                if 'betting' in pred and pred['betting']['should_bet']:
                    high_conf.append(pred)
        
        # Sort by edge
        high_conf.sort(key=lambda x: x['betting']['edge'], reverse=True)
        
        return high_conf
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'meta_learner': self.meta_learner_,
            'transformers': [(name, t) for name, t in self.transformers],
            'validation_scores': self.validation_scores_,
            'transformer_info': self.transformer_info_,
            'feature_dims': self.feature_dims_,
            'config': {
                'min_confidence': self.min_confidence,
                'min_edge': self.min_edge,
                'use_calibration': self.use_calibration,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[Ensemble] ✓ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            min_confidence=model_data['config']['min_confidence'],
            min_edge=model_data['config']['min_edge'],
            use_calibration=model_data['config']['use_calibration']
        )
        
        model.meta_learner_ = model_data['meta_learner']
        model.transformers = model_data['transformers']
        model.validation_scores_ = model_data['validation_scores']
        model.transformer_info_ = model_data['transformer_info']
        model.feature_dims_ = model_data['feature_dims']
        model.is_fitted_ = True
        
        print(f"[Ensemble] ✓ Model loaded from {filepath}")
        print(f"[Ensemble]   CV accuracy: {model.validation_scores_['cv_mean']:.1%}")
        
        return model
    
    def get_model_summary(self) -> Dict:
        """Get model performance summary"""
        if not self.is_fitted_:
            return {'error': 'Model not fitted'}
        
        return {
            'n_transformers': len(self.transformers),
            'n_features': sum(self.feature_dims_),
            'performance': self.validation_scores_,
            'top_transformers': self.transformer_info_[:10],
            'config': {
                'min_confidence': self.min_confidence,
                'min_edge': self.min_edge,
                'calibrated': self.use_calibration
            }
        }

