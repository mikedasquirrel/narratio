"""
NBA Pattern-Optimized Betting Model
====================================

OPTIMIZED VERSION combining:
1. 225 discovered context patterns (64.8% accuracy, +52.8% ROI)
2. 42 transformer ensemble (56.8% accuracy)
3. Pattern-transformer hybrid for maximum edge

This leverages your EXISTING pattern discovery work!

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
from sklearn.ensemble import VotingClassifier
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from narrative_optimization.betting.nba_ensemble_model import NBAEnsembleBettingModel
from narrative_optimization.betting.betting_utils import calculate_ev, calculate_edge, should_bet


class NBAPatternOptimizedModel:
    """
    NBA betting model optimized for discovered patterns.
    
    Combines:
    1. Pattern matching from 225 discovered patterns
    2. Transformer ensemble for narrative features
    3. Hybrid voting for maximum accuracy
    
    Strategy:
    - If game matches high-confidence pattern (64%+): Use pattern prediction
    - If game matches medium pattern (60-64%): Blend pattern + transformers
    - If no strong pattern match: Use transformer ensemble
    """
    
    def __init__(
        self,
        patterns_path: str = 'discovered_player_patterns.json',
        min_pattern_accuracy: float = 0.60,
        min_pattern_samples: int = 100,
        hybrid_weight: float = 0.5
    ):
        """
        Initialize pattern-optimized model.
        
        Parameters
        ----------
        patterns_path : str
            Path to discovered patterns JSON
        min_pattern_accuracy : float
            Minimum pattern accuracy to use
        min_pattern_samples : int
            Minimum samples for pattern validity
        hybrid_weight : float
            Weight for pattern vs transformer (0.5 = equal)
        """
        self.patterns_path = patterns_path
        self.min_pattern_accuracy = min_pattern_accuracy
        self.min_pattern_samples = min_pattern_samples
        self.hybrid_weight = hybrid_weight
        
        # Load patterns
        self.patterns_ = self._load_patterns()
        
        # Initialize transformer ensemble
        self.transformer_ensemble_ = None
        
        self.is_fitted_ = False
    
    def _load_patterns(self) -> List[Dict]:
        """Load discovered patterns"""
        path = Path(self.patterns_path)
        
        if not path.exists():
            print(f"[Pattern] ⚠️  Patterns file not found: {path}")
            print(f"[Pattern] Run: python discover_player_patterns.py")
            return []
        
        with open(path) as f:
            data = json.load(f)
        
        patterns = data.get('patterns', [])
        
        # Filter for high-quality patterns
        quality_patterns = [
            p for p in patterns
            if p['accuracy'] >= self.min_pattern_accuracy
            and p['sample_size'] >= self.min_pattern_samples
        ]
        
        print(f"[Pattern] ✓ Loaded {len(patterns)} total patterns")
        print(f"[Pattern] ✓ {len(quality_patterns)} high-quality patterns (acc≥{self.min_pattern_accuracy:.0%}, n≥{self.min_pattern_samples})")
        
        if len(quality_patterns) > 0:
            best = quality_patterns[0]
            print(f"[Pattern] 🏆 Best pattern: {best['accuracy']:.1%} accuracy ({best['sample_size']} games)")
        
        return quality_patterns
    
    def _match_pattern(self, game_features: Dict) -> Optional[Dict]:
        """
        Check if game matches any discovered pattern.
        
        Parameters
        ----------
        game_features : dict
            Game features (home, season_win_pct, l10_win_pct, etc.)
            
        Returns
        -------
        pattern : dict or None
            Matched pattern with accuracy, or None
        """
        for pattern in self.patterns_:
            conditions = pattern['conditions']
            matches = True
            
            for feature, constraint in conditions.items():
                if feature not in game_features:
                    matches = False
                    break
                
                value = game_features[feature]
                
                # Check constraints
                if 'eq' in constraint and value != constraint['eq']:
                    matches = False
                    break
                if 'min' in constraint and value < constraint['min']:
                    matches = False
                    break
                if 'max' in constraint and value > constraint['max']:
                    matches = False
                    break
            
            if matches:
                return pattern
        
        return None
    
    def fit(self, X_narratives: pd.Series, X_features: pd.DataFrame, y: np.ndarray, verbose: bool = True):
        """
        Fit pattern-optimized model.
        
        Parameters
        ----------
        X_narratives : pd.Series
            Clean pre-game narratives for transformers
        X_features : pd.DataFrame
            Numerical features for pattern matching
        y : np.ndarray
            Outcomes
        verbose : bool
            Print progress
            
        Returns
        -------
        self
        """
        if verbose:
            print(f"\n{'='*80}")
            print("TRAINING PATTERN-OPTIMIZED MODEL")
            print('='*80)
        
        # Train transformer ensemble
        print(f"\n[Training] Stage 1: Transformer Ensemble")
        self.transformer_ensemble_ = NBAEnsembleBettingModel(min_confidence=0.60, min_edge=0.05)
        self.transformer_ensemble_.fit(X_narratives, y, verbose=verbose)
        
        # Patterns are already fitted (from discovery phase)
        print(f"\n[Training] Stage 2: Pattern Integration")
        print(f"[Training] ✓ Using {len(self.patterns_)} pre-discovered patterns")
        
        # Validate hybrid approach
        print(f"\n[Training] Stage 3: Hybrid Validation")
        
        correct_pattern = 0
        correct_transformer = 0
        correct_hybrid = 0
        pattern_matches = 0
        
        for i in range(len(X_narratives)):
            game_feats = X_features.iloc[i].to_dict()
            actual = y[i]
            
            # Pattern prediction
            pattern = self._match_pattern(game_feats)
            if pattern:
                pattern_matches += 1
                pattern_pred = 1 if pattern['accuracy'] > 0.5 else 0
                if pattern_pred == actual:
                    correct_pattern += 1
            
            # Transformer prediction  
            trans_prob = self.transformer_ensemble_.predict_proba(X_narratives.iloc[i:i+1])[0][1]
            trans_pred = 1 if trans_prob > 0.5 else 0
            if trans_pred == actual:
                correct_transformer += 1
            
            # Hybrid (when pattern available, blend predictions)
            if pattern:
                hybrid_prob = (pattern['accuracy'] * self.hybrid_weight + 
                             trans_prob * (1 - self.hybrid_weight))
                hybrid_pred = 1 if hybrid_prob > 0.5 else 0
                if hybrid_pred == actual:
                    correct_hybrid += 1
            else:
                # No pattern, use transformer
                if trans_pred == actual:
                    correct_hybrid += 1
        
        pattern_acc = correct_pattern / pattern_matches if pattern_matches > 0 else 0
        trans_acc = correct_transformer / len(X_narratives)
        hybrid_acc = correct_hybrid / len(X_narratives)
        
        print(f"\n[Validation] Pattern accuracy: {pattern_acc:.1%} ({pattern_matches} matches)")
        print(f"[Validation] Transformer accuracy: {trans_acc:.1%}")
        print(f"[Validation] Hybrid accuracy: {hybrid_acc:.1%}")
        print(f"[Validation] Improvement: {hybrid_acc - max(pattern_acc, trans_acc):+.1%}")
        
        self.is_fitted_ = True
        return self
    
    def predict_with_patterns(
        self,
        X_narratives: pd.Series,
        X_features: pd.DataFrame,
        market_odds: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Predict using pattern-optimized approach.
        
        Parameters
        ----------
        X_narratives : pd.Series
            Narratives for transformers
        X_features : pd.DataFrame
            Features for pattern matching
        market_odds : np.ndarray, optional
            Market odds
        verbose : bool
            Print progress
            
        Returns
        -------
        predictions : list of dict
            Enhanced predictions with pattern info
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        if verbose:
            print(f"\n[Predict] Analyzing {len(X_narratives)} games with pattern optimization...")
        
        # Get transformer predictions
        transformer_preds = self.transformer_ensemble_.predict_proba(X_narratives, verbose=False)
        
        predictions = []
        
        for i in range(len(X_narratives)):
            game_feats = X_features.iloc[i].to_dict()
            trans_prob = transformer_preds[i][1]
            
            # Check pattern match
            pattern = self._match_pattern(game_feats)
            
            if pattern:
                # Use hybrid prediction
                pattern_prob = pattern['accuracy']
                final_prob = (pattern_prob * self.hybrid_weight + 
                            trans_prob * (1 - self.hybrid_weight))
                
                method = "PATTERN+TRANSFORMER"
                confidence_boost = pattern_prob - 0.5  # How much above baseline
            else:
                # Use transformer only
                final_prob = trans_prob
                method = "TRANSFORMER"
                confidence_boost = 0
            
            prediction = {
                'game_index': i,
                'narrative': X_narratives.iloc[i][:120] + "...",
                'win_probability': float(final_prob),
                'loss_probability': float(1 - final_prob),
                'method': method,
                'pattern_matched': pattern is not None,
                'pattern_accuracy': float(pattern['accuracy']) if pattern else None,
                'transformer_probability': float(trans_prob),
                'confidence_boost': float(confidence_boost),
                'predicted_outcome': 'WIN' if final_prob > 0.5 else 'LOSS'
            }
            
            # Add betting analysis
            if market_odds is not None and i < len(market_odds):
                odds = market_odds[i]
                
                bet_decision, reason = should_bet(
                    final_prob,
                    odds,
                    min_edge=0.05,
                    min_confidence=0.60
                )
                
                edge = calculate_edge(final_prob, odds)
                ev = calculate_ev(final_prob, odds, stake=1.0)
                
                # Enhanced units for pattern matches
                if pattern and pattern['accuracy'] >= 0.64:
                    units = 2.5  # Maximum for proven patterns
                elif pattern and pattern['accuracy'] >= 0.62:
                    units = 2.0
                elif final_prob >= 0.70:
                    units = 2.0
                elif final_prob >= 0.65:
                    units = 1.5
                elif final_prob >= 0.60:
                    units = 1.0
                else:
                    units = 0.0
                
                prediction['betting'] = {
                    'market_odds': float(odds),
                    'edge': float(edge),
                    'expected_value': float(ev),
                    'should_bet': bet_decision,
                    'reason': reason,
                    'recommended_units': float(units) if bet_decision else 0.0,
                    'pattern_enhanced': pattern is not None
                }
            
            predictions.append(prediction)
        
        if verbose:
            pattern_matches = sum(1 for p in predictions if p['pattern_matched'])
            print(f"[Predict] ✓ {pattern_matches}/{len(predictions)} games matched patterns")
        
        return predictions
    
    def get_high_confidence_bets(
        self,
        predictions: List[Dict],
        min_confidence: float = 0.60,
        prioritize_patterns: bool = True
    ) -> List[Dict]:
        """
        Get high-confidence betting opportunities, prioritizing pattern matches.
        
        Parameters
        ----------
        predictions : list of dict
            All predictions
        min_confidence : float
            Minimum confidence
        prioritize_patterns : bool
            Sort pattern matches first
            
        Returns
        -------
        high_confidence : list of dict
            Sorted by confidence/edge
        """
        high_conf = []
        
        for pred in predictions:
            if pred['win_probability'] >= min_confidence or pred['loss_probability'] >= min_confidence:
                if 'betting' in pred and pred['betting']['should_bet']:
                    high_conf.append(pred)
        
        # Sort: pattern matches first, then by edge
        if prioritize_patterns:
            high_conf.sort(key=lambda x: (
                -1 if x['pattern_matched'] else 0,  # Patterns first
                -x['betting']['edge']  # Then by edge
            ))
        else:
            high_conf.sort(key=lambda x: -x['betting']['edge'])
        
        return high_conf
    
    def save_model(self, filepath: str):
        """Save model"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        model_data = {
            'transformer_ensemble': self.transformer_ensemble_,
            'patterns': self.patterns_,
            'config': {
                'min_pattern_accuracy': self.min_pattern_accuracy,
                'min_pattern_samples': self.min_pattern_samples,
                'hybrid_weight': self.hybrid_weight
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[Model] ✓ Saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data['config']
        model = cls(
            min_pattern_accuracy=config['min_pattern_accuracy'],
            min_pattern_samples=config['min_pattern_samples'],
            hybrid_weight=config['hybrid_weight']
        )
        
        model.transformer_ensemble_ = model_data['transformer_ensemble']
        model.patterns_ = model_data['patterns']
        model.is_fitted_ = True
        
        print(f"[Model] ✓ Loaded from {filepath}")
        print(f"[Model]   Patterns: {len(model.patterns_)}")
        print(f"[Model]   Hybrid weight: {model.hybrid_weight}")
        
        return model

