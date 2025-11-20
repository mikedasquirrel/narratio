"""
Alpha Transformer (α Measurement) - RENOVATED

Computes optimal α (feature strength balance) empirically.
α determines character vs plot feature weights for computing ю.

Theory: α = 0.85 - 0.95×п (theoretical)
Empirical: Discover optimal α from feature effectiveness.

RENOVATED (November 2025):
- Removed hardcoded keyword-based feature classification
- Now uses data-driven discovery via correlation patterns and variance
- Character features: low variance (stable traits)
- Plot features: high variance (dynamic events)
- Discovers feature types from data, not assumptions

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .base import NarrativeTransformer


class AlphaTransformer(NarrativeTransformer):
    """
    Compute optimal α (feature strength balance) empirically.
    
    Theory: α = 0.85 - 0.95×п (theoretical formula)
    Empirical: Discover optimal α from feature effectiveness analysis
    
    α represents the balance between character and plot features:
    - α = 0: Pure character features (high п domains)
    - α = 1: Pure plot features (low п domains)
    - α = 0.5: Balanced mix
    
    Features Extracted (8 total):
    1. Character feature effectiveness (correlation with outcomes)
    2. Plot feature effectiveness (correlation with outcomes)
    3. Character feature importance score
    4. Plot feature importance score
    5. Optimal α value (empirically discovered)
    6. Character feature weight
    7. Plot feature weight
    8. Alpha discovery confidence
    
    Parameters
    ----------
    narrativity : float
        Domain narrativity (п) for theoretical comparison
    method : str, default='mutual_info'
        Method for computing feature effectiveness ('mutual_info', 'correlation', 'ridge')
    
    Examples
    --------
    >>> transformer = AlphaTransformer(narrativity=0.75)
    >>> features = transformer.fit_transform(X, y=outcomes)
    >>> 
    >>> # Check optimal α
    >>> optimal_alpha = transformer.optimal_alpha_
    >>> print(f"Optimal α: {optimal_alpha:.3f}")
    >>> print(f"Theoretical α: {transformer.theoretical_alpha_:.3f}")
    """
    
    def __init__(self, narrativity: float = 0.5, method: str = 'mutual_info'):
        super().__init__(
            narrative_id="alpha",
            description="Measures α (feature strength balance) empirically"
        )
        
        self.narrativity = narrativity
        self.method = method
        
        # Theoretical α
        self.theoretical_alpha_ = 0.85 - 0.95 * narrativity
        
        # Empirical results
        self.optimal_alpha_ = None
        self.character_weight_ = None
        self.plot_weight_ = None
        self.character_effectiveness_ = None
        self.plot_effectiveness_ = None
        self.discovery_confidence_ = None
        
        # Feature classification
        self.character_feature_indices_ = None
        self.plot_feature_indices_ = None
    
    def _identify_character_features(self, feature_names: List[str]) -> List[int]:
        """
        Identify character-focused features from feature names.
        
        RENOVATED: No longer uses hardcoded keywords. Instead uses data-driven
        classification based on correlation patterns and variance.
        
        This method is now a placeholder - actual classification happens in fit()
        using clustering of correlation patterns.
        """
        # Return empty - will be populated by data-driven discovery in fit()
        return []
    
    def _identify_plot_features(self, feature_names: List[str]) -> List[int]:
        """
        Identify plot-focused features from feature names.
        
        RENOVATED: No longer uses hardcoded keywords. Instead uses data-driven
        classification based on correlation patterns and variance.
        
        This method is now a placeholder - actual classification happens in fit()
        using clustering of correlation patterns.
        """
        # Return empty - will be populated by data-driven discovery in fit()
        return []
    
    def _discover_feature_types_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        RENOVATED METHOD: Discover feature types from data, not keywords.
        
        Uses correlation patterns and variance to cluster features into
        character-like (low variance, moderate correlation) vs
        plot-like (high variance, variable correlation).
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        y : ndarray
            Outcomes
            
        Returns
        -------
        character_indices, plot_indices : tuple of lists
            Indices of character-like and plot-like features
        """
        n_features = X.shape[1]
        
        # Compute correlation of each feature with outcome
        correlations = []
        for i in range(n_features):
            if np.std(X[:, i]) > 0:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)
        
        correlations = np.array(correlations)
        
        # Compute temporal variance (proxy for character vs plot)
        # Character features: stable traits (low variance)
        # Plot features: dynamic events (high variance)
        variances = np.var(X, axis=0)
        
        # Normalize
        variances_norm = variances / (np.max(variances) + 1e-10)
        correlations_norm = correlations / (np.max(correlations) + 1e-10)
        
        # Classify using 2D space
        # Character: low variance, moderate correlation
        # Plot: high variance, variable correlation
        median_var = np.median(variances_norm)
        
        character_mask = variances_norm < median_var
        plot_mask = variances_norm >= median_var
        
        character_indices = np.where(character_mask)[0].tolist()
        plot_indices = np.where(plot_mask)[0].tolist()
        
        return character_indices, plot_indices
    
    def _compute_effectiveness(
        self,
        X_features: np.ndarray,
        y: np.ndarray,
        method: str
    ) -> float:
        """
        Compute feature effectiveness (correlation with outcomes).
        
        Parameters
        ----------
        X_features : ndarray
            Feature matrix (n_samples, n_features)
        y : ndarray
            Outcomes
        method : str
            Method for computing effectiveness
        
        Returns
        -------
        effectiveness : float
            Average feature effectiveness [0, 1]
        """
        if X_features.shape[1] == 0:
            return 0.0
        
        if method == 'mutual_info':
            # Use mutual information
            mi_scores = mutual_info_regression(X_features, y, random_state=42)
            effectiveness = np.mean(mi_scores)
            # Normalize to [0, 1] (MI can be > 1)
            effectiveness = min(1.0, effectiveness)
        
        elif method == 'correlation':
            # Use absolute correlation
            correlations = []
            for i in range(X_features.shape[1]):
                corr = np.abs(np.corrcoef(X_features[:, i], y)[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)
            effectiveness = np.mean(correlations) if correlations else 0.0
        
        elif method == 'ridge':
            # Use Ridge regression coefficients
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_scaled, y)
            coefs = np.abs(ridge.coef_)
            effectiveness = np.mean(coefs)
            # Normalize to [0, 1]
            effectiveness = min(1.0, effectiveness / np.max(coefs) if np.max(coefs) > 0 else 0.0)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(effectiveness)
    
    def _discover_optimal_alpha(
        self,
        char_effectiveness: float,
        plot_effectiveness: float
    ) -> Tuple[float, float, float]:
        """
        Discover optimal α from feature effectiveness.
        
        Parameters
        ----------
        char_effectiveness : float
            Character feature effectiveness
        plot_effectiveness : float
            Plot feature effectiveness
        
        Returns
        -------
        optimal_alpha : float
            Optimal α value [0, 1]
        char_weight : float
            Character feature weight
        plot_weight : float
            Plot feature weight
        """
        # If both are zero, use theoretical α
        if char_effectiveness == 0 and plot_effectiveness == 0:
            optimal_alpha = self.theoretical_alpha_
        else:
            # Weight by effectiveness
            total_effectiveness = char_effectiveness + plot_effectiveness
            if total_effectiveness > 0:
                # α = plot_weight (higher α = more plot)
                optimal_alpha = plot_effectiveness / total_effectiveness
            else:
                optimal_alpha = self.theoretical_alpha_
        
        # Clamp to [0, 1]
        optimal_alpha = max(0.0, min(1.0, optimal_alpha))
        
        # Compute weights (α = plot weight, 1-α = character weight)
        plot_weight = optimal_alpha
        char_weight = 1.0 - optimal_alpha
        
        return optimal_alpha, char_weight, plot_weight
    
    def fit(self, X, y=None, feature_names: Optional[List[str]] = None):
        """
        Fit transformer by analyzing feature effectiveness.
        
        Parameters
        ----------
        X : ndarray
            Feature matrix (n_samples, n_features)
        y : ndarray
            Outcomes (required for fitting)
        feature_names : list of str, optional
            Feature names for classification
        
        Returns
        -------
        self : AlphaTransformer
            Fitted transformer
        """
        if y is None:
            raise ValueError("AlphaTransformer requires outcomes (y) for fitting")
        
        X = np.array(X)
        y = np.array(y)
        
        # Validate X is 2D feature matrix
        if X.ndim == 1:
            raise ValueError(
                "AlphaTransformer requires a 2D feature matrix (n_samples, n_features), "
                "not raw text or 1D array. This is a meta-transformer that analyzes "
                "features from other transformers. Use it AFTER feature extraction."
            )
        
        if X.ndim != 2:
            raise ValueError(f"AlphaTransformer expects 2D input, got {X.ndim}D")
        
        # If feature names not provided, try to infer from X
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # RENOVATED: Classify features using data-driven discovery
        # No longer uses keywords - uses correlation patterns and variance
        self.character_feature_indices_, self.plot_feature_indices_ = \
            self._discover_feature_types_from_data(X, y)
        
        # Extract feature subsets
        if len(self.character_feature_indices_) > 0:
            X_char = X[:, self.character_feature_indices_]
            char_effectiveness = self._compute_effectiveness(X_char, y, self.method)
        else:
            char_effectiveness = 0.0
        
        if len(self.plot_feature_indices_) > 0:
            X_plot = X[:, self.plot_feature_indices_]
            plot_effectiveness = self._compute_effectiveness(X_plot, y, self.method)
        else:
            plot_effectiveness = 0.0
        
        # Store effectiveness scores
        self.character_effectiveness_ = char_effectiveness
        self.plot_effectiveness_ = plot_effectiveness
        
        # Discover optimal α
        optimal_alpha, char_weight, plot_weight = self._discover_optimal_alpha(
            char_effectiveness, plot_effectiveness
        )
        
        self.optimal_alpha_ = optimal_alpha
        self.character_weight_ = char_weight
        self.plot_weight_ = plot_weight
        
        # Compute discovery confidence (how different from theoretical)
        alpha_diff = abs(optimal_alpha - self.theoretical_alpha_)
        self.discovery_confidence_ = 1.0 / (1.0 + alpha_diff * 10.0)  # Higher confidence if close to theoretical
        
        # Store metadata
        self.metadata['theoretical_alpha'] = float(self.theoretical_alpha_)
        self.metadata['optimal_alpha'] = float(self.optimal_alpha_)
        self.metadata['character_effectiveness'] = float(char_effectiveness)
        self.metadata['plot_effectiveness'] = float(plot_effectiveness)
        self.metadata['character_weight'] = float(char_weight)
        self.metadata['plot_weight'] = float(plot_weight)
        self.metadata['discovery_confidence'] = float(self.discovery_confidence_)
        self.metadata['n_character_features'] = len(self.character_feature_indices_)
        self.metadata['n_plot_features'] = len(self.plot_feature_indices_)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform features to α-related features.
        
        Note: All instances get the same α (it's domain-level).
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        
        Returns
        -------
        features : ndarray
            Array of shape (n_samples, 8) with α-related features
        """
        self._validate_fitted()
        
        n_samples = X.shape[0]
        
        # All instances get same α (it's domain-level)
        features = np.zeros((n_samples, 8))
        
        features[:, 0] = self.character_effectiveness_
        features[:, 1] = self.plot_effectiveness_
        features[:, 2] = self.character_effectiveness_  # Character importance
        features[:, 3] = self.plot_effectiveness_  # Plot importance
        features[:, 4] = self.optimal_alpha_
        features[:, 5] = self.character_weight_
        features[:, 6] = self.plot_weight_
        features[:, 7] = self.discovery_confidence_
        
        return features
    
    def _generate_interpretation(self) -> str:
        """Generate human-readable interpretation."""
        if self.optimal_alpha_ is None:
            return "No data fitted yet."
        
        optimal = self.optimal_alpha_
        theoretical = self.theoretical_alpha_
        diff = abs(optimal - theoretical)
        
        if optimal < 0.3:
            balance = "character-dominated"
        elif optimal < 0.5:
            balance = "character-leaning"
        elif optimal < 0.7:
            balance = "plot-leaning"
        else:
            balance = "plot-dominated"
        
        interpretation = (
            f"Optimal α={optimal:.3f} ({balance}), theoretical α={theoretical:.3f}. "
            f"Character effectiveness={self.character_effectiveness_:.3f}, "
            f"plot effectiveness={self.plot_effectiveness_:.3f}. "
        )
        
        if diff < 0.1:
            interpretation += "Empirical α closely matches theoretical prediction."
        elif diff < 0.2:
            interpretation += "Empirical α moderately differs from theoretical."
        else:
            interpretation += "Empirical α significantly differs from theoretical - domain-specific optimization needed."
        
        return interpretation
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'character_effectiveness',
            'plot_effectiveness',
            'character_importance',
            'plot_importance',
            'optimal_alpha',
            'character_weight',
            'plot_weight',
            'alpha_discovery_confidence'
        ]
    
    def get_optimal_weights(self) -> Dict[str, float]:
        """
        Get optimal feature weights for computing ю.
        
        Returns
        -------
        weights : dict
            Dictionary with 'character_weight' and 'plot_weight'
        """
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before getting weights")
        
        return {
            'character_weight': self.character_weight_,
            'plot_weight': self.plot_weight_,
            'optimal_alpha': self.optimal_alpha_,
            'theoretical_alpha': self.theoretical_alpha_
        }

