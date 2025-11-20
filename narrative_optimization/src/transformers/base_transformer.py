"""
Base Narrative Transformer

Abstract base class for all narrative feature transformers.
Follows sklearn transformer API (fit/transform) with added interpretation methods.

Author: Narrative Integration System
Date: November 2025
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NarrativeTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for narrative feature transformers.
    
    All narrative transformers must:
    1. Inherit from this class
    2. Implement fit() and transform() methods
    3. Implement _generate_interpretation() for interpretability
    4. Follow sklearn transformer API
    
    Attributes:
        narrative_id (str): Unique identifier for this narrative approach
        description (str): Human-readable description of narrative hypothesis
        metadata (Dict): Dictionary storing learned parameters and statistics
        is_fitted_ (bool): Flag indicating if transformer has been fitted
    """
    
    def __init__(self, narrative_id: str, description: str):
        """
        Initialize narrative transformer.
        
        Args:
            narrative_id: Unique identifier (e.g., 'nominative_analysis')
            description: Description of narrative hypothesis
        """
        self.narrative_id = narrative_id
        self.description = description
        self.metadata = {}
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Learn narrative patterns from training data.
        
        Args:
            X: Training data (list of texts or array of features)
            y: Target labels (optional)
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """
        Transform data to narrative feature representation.
        
        Args:
            X: Data to transform (same format as fit)
        
        Returns:
            numpy.ndarray: Feature matrix (n_samples, n_features)
        """
        pass
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Args:
            X: Training data
            y: Target labels (optional)
        
        Returns:
            numpy.ndarray: Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def _validate_fitted(self):
        """Check if transformer has been fitted."""
        if not self.is_fitted_:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before transform. "
                "Call fit() or fit_transform() first."
            )
    
    def _validate_input(self, X):
        """
        Validate input data format.
        
        Args:
            X: Input data
        
        Returns:
            bool: True if valid
        
        Raises:
            ValueError: If input format is invalid
        """
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty")
        return True
    
    @abstractmethod
    def _generate_interpretation(self) -> str:
        """
        Generate human-readable interpretation of learned patterns.
        
        This method should explain:
        - What narrative patterns were discovered
        - Which features are most important
        - How the narrative hypothesis manifests in data
        
        Returns:
            str: Interpretation text
        """
        pass
    
    def get_interpretation(self) -> str:
        """
        Get interpretation of learned narrative patterns.
        
        Returns:
            str: Human-readable interpretation
        """
        self._validate_fitted()
        return self._generate_interpretation()
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of output features.
        
        Returns:
            List[str]: Feature names
        """
        self._validate_fitted()
        if 'feature_names' in self.metadata:
            return self.metadata['feature_names']
        else:
            # Default to generic names
            n_features = self.metadata.get('n_features', 0)
            return [f"{self.narrative_id}_feature_{i}" for i in range(n_features)]
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get transformer metadata and learned parameters.
        
        Returns:
            Dict: Metadata dictionary
        """
        return {
            'narrative_id': self.narrative_id,
            'description': self.description,
            'is_fitted': self.is_fitted_,
            'learned_metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(narrative_id='{self.narrative_id}')"
    
    def __str__(self) -> str:
        """Friendly string representation."""
        fitted_status = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.__class__.__name__} [{fitted_status}]: {self.description}"


class TextNarrativeTransformer(NarrativeTransformer):
    """
    Base class for transformers that operate on text data.
    
    Provides common text preprocessing utilities.
    """
    
    def __init__(self, narrative_id: str, description: str):
        super().__init__(narrative_id, description)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing.
        
        Args:
            text: Raw text
        
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = text.strip()
        
        return text
    
    def _validate_input(self, X):
        """Validate that input is list of texts."""
        super()._validate_input(X)
        
        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("Input X must be list or array of texts")
        
        # Check first element is string-like
        if len(X) > 0 and not isinstance(X[0], str):
            raise ValueError("Input X must contain text strings")
        
        return True


class FeatureNarrativeTransformer(NarrativeTransformer):
    """
    Base class for transformers that operate on pre-extracted features.
    
    Provides utilities for feature-based transformations.
    """
    
    def __init__(self, narrative_id: str, description: str):
        super().__init__(narrative_id, description)
    
    def _validate_input(self, X):
        """Validate that input is feature matrix."""
        super()._validate_input(X)
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim != 2:
            raise ValueError(f"Input X must be 2D array, got shape {X.shape}")
        
        return True


class MixedNarrativeTransformer(NarrativeTransformer):
    """
    Base class for transformers that combine text and feature inputs.
    
    Handles mixed input types for Format C data.
    """
    
    def __init__(self, narrative_id: str, description: str):
        super().__init__(narrative_id, description)
    
    def _validate_input(self, X):
        """Validate mixed input format."""
        super()._validate_input(X)
        
        # X can be dict with 'texts' and 'features' keys
        # or just texts/features alone
        if isinstance(X, dict):
            if 'texts' not in X and 'features' not in X:
                raise ValueError("Mixed input dict must have 'texts' or 'features' key")
        
        return True

