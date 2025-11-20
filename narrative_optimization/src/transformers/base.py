"""
Base transformer class for narrative-driven feature engineering.

All custom transformers inherit from NarrativeTransformer to maintain
consistent interface and narrative metadata.
"""

from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class NarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for all narrative-driven transformers.
    
    Each transformer encodes a specific narrative hypothesis about the domain
    and maintains metadata about its story, rationale, and theoretical foundation.
    
    Parameters
    ----------
    narrative_id : str
        Unique identifier for this narrative component
    description : str
        Human-readable description of the narrative hypothesis
    
    Attributes
    ----------
    metadata : dict
        Additional metadata about the narrative, populated during fit
    is_fitted_ : bool
        Whether the transformer has been fitted
    """
    
    def __init__(self, narrative_id: str, description: str):
        self.narrative_id = narrative_id
        self.description = description
        self.metadata: Dict[str, Any] = {}
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the training data.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Training data
        y : array-like, optional
            Target values
        
        Returns
        -------
        self : NarrativeTransformer
            Fitted transformer
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement fit()"
        )
    
    def transform(self, X):
        """
        Transform the input data according to the narrative.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Data to transform
        
        Returns
        -------
        X_transformed : array-like or DataFrame
            Transformed data
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement transform()"
        )
    
    def get_narrative_report(self) -> Dict[str, Any]:
        """
        Return human-readable explanation of this narrative component.
        
        Returns
        -------
        report : dict
            Dictionary containing:
            - narrative_id: identifier
            - description: hypothesis description
            - is_fitted: whether transformer is fitted
            - metadata: additional context about the narrative
            - interpretation: human-readable explanation of what this narrative captures
        """
        if not self.is_fitted_:
            interpretation = "Transformer not yet fitted. No insights available."
        else:
            interpretation = self._generate_interpretation()
        
        return {
            'narrative_id': self.narrative_id,
            'description': self.description,
            'is_fitted': self.is_fitted_,
            'metadata': self.metadata,
            'interpretation': interpretation
        }
    
    def _generate_interpretation(self) -> str:
        """
        Generate human-readable interpretation of the fitted transformer.
        
        Subclasses should override this to provide narrative-specific insights.
        
        Returns
        -------
        interpretation : str
            Human-readable explanation of what the narrative learned
        """
        return "No interpretation method implemented for this transformer."
    
    def _validate_fitted(self):
        """Check if transformer has been fitted."""
        if not self.is_fitted_:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before transform. "
                f"Call fit() or fit_transform() first."
            )

