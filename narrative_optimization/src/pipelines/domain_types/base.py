"""
Base Domain Type

Abstract base class for all domain types. Defines the common interface
that all domain types must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pipelines.domain_config import DomainConfig, DomainType


class BaseDomainType(ABC):
    """
    Base class for domain type templates.
    
    Each domain type (sports, entertainment, etc.) extends this class
    to provide domain-specific customization:
    - Default transformer suite
    - Data preprocessing requirements
    - Validation metrics
    - Reporting templates
    - Baseline comparison strategy
    """
    
    def __init__(self, config: DomainConfig):
        """
        Initialize domain type with configuration.
        
        Parameters
        ----------
        config : DomainConfig
            Domain configuration
        """
        self.config = config
        self.domain_name = config.domain
        self.pi = config.pi
        self.domain_type = config.type
    
    @abstractmethod
    def get_default_transformers(self, п: float) -> List[str]:
        """
        Get ADDITIONAL transformers for this domain type (beyond core).
        
        NOTE: Core transformers (nominative, self_perception, narrative_potential,
        linguistic, ensemble, relational) are available to ALL domains as they
        represent universal narrative features. This method returns domain-type
        specific ADDITIONAL transformers.
        
        Parameters
        ----------
        п : float
            Domain narrativity
            
        Returns
        -------
        transformers : list of str
            Additional transformer keys for this domain type
        """
        pass
    
    @abstractmethod
    def get_validation_metrics(self) -> List[str]:
        """
        Get validation metrics appropriate for this domain type.
        
        Returns
        -------
        metrics : list of str
            Metric names (e.g., 'r2', 'betting_roi', 'auc')
        """
        pass
    
    def get_data_preprocessing(self) -> Dict[str, Any]:
        """
        Get data preprocessing requirements.
        
        Returns
        -------
        preprocessing : dict
            Preprocessing configuration
        """
        return {
            'normalize_text': True,
            'handle_missing': 'skip',
            'min_text_length': 10
        }
    
    def get_baseline_comparison(self) -> Dict[str, Any]:
        """
        Get baseline comparison strategy.
        
        Returns
        -------
        baseline : dict
            Baseline configuration
        """
        return {
            'method': 'statistical',
            'features': ['context_features'] if self.config.data.context_fields else []
        }
    
    def get_reporting_template(self) -> str:
        """
        Get reporting template name.
        
        Returns
        -------
        template : str
            Template name for this domain type
        """
        return 'default'
    
    def customize_transformer_selection(
        self,
        selected_transformers: List[str],
        п: float
    ) -> List[str]:
        """
        Customize transformer selection after initial п-based selection.
        
        Parameters
        ----------
        selected_transformers : list of str
            Initially selected transformers
        п : float
            Domain narrativity
            
        Returns
        -------
        customized : list of str
            Customized transformer list
        """
        # Default: no customization
        return selected_transformers
    
    def get_domain_specific_insights(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate domain-specific insights from results.
        
        Parameters
        ----------
        results : dict
            Analysis results
            
        Returns
        -------
        insights : list of str
            Domain-specific insight strings
        """
        return []
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform domain-specific validation checks.
        
        Parameters
        ----------
        results : dict
            Analysis results
            
        Returns
        -------
        validation : dict
            Validation results with domain-specific checks
        """
        return {
            'domain_specific_checks': [],
            'all_passed': True
        }
    
    def get_perspective_preferences(self) -> List[str]:
        """
        Get preferred perspectives for this domain type.
        
        Returns
        -------
        perspectives : list of str
            Perspective names (e.g., ['director', 'audience', 'critic'])
        """
        # Default: all perspectives
        return ['director', 'audience', 'critic', 'character', 'cultural', 'meta']
    
    def get_quality_method_preferences(self) -> List[str]:
        """
        Get preferred quality calculation methods for this domain type.
        
        Returns
        -------
        methods : list of str
            Method names (e.g., ['weighted_mean', 'ensemble'])
        """
        # Default: weighted mean and ensemble
        return ['weighted_mean', 'ensemble']
    
    def get_scale_preferences(self) -> List[str]:
        """
        Get preferred scales for this domain type.
        
        Returns
        -------
        scales : list of str
            Scale names (e.g., ['micro', 'meso', 'macro'])
        """
        # Default: micro, meso, macro
        return ['micro', 'meso', 'macro']
    
    def get_summary(self) -> str:
        """Get human-readable summary of domain type"""
        return f"{self.__class__.__name__} for {self.domain_name} (п={self.pi:.3f})"

