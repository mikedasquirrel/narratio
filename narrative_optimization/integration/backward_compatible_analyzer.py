"""
Backward Compatible Analyzer

Wrapper that provides backward compatibility for existing code
while using new domain-specific architecture internally.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer


class BackwardCompatibleAnalyzer:
    """
    Provides backward-compatible interface to old UniversalDomainAnalyzer
    while using new DomainSpecificAnalyzer internally.
    
    This allows existing code to work without changes while benefiting
    from new architecture.
    """
    
    def __init__(self, domain_name: str, narrativity: float):
        """
        Initialize with same signature as old UniversalDomainAnalyzer.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        narrativity : float
            Domain π (narrativity)
        """
        self.domain_name = domain_name
        self.п = narrativity
        
        # Use new analyzer internally
        self._analyzer = DomainSpecificAnalyzer(domain_name, narrativity)
        
    def analyze_complete(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        names: Optional[List[str]] = None,
        genome: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        masses: Optional[np.ndarray] = None,
        baseline_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze with backward-compatible interface.
        
        Parameters match old UniversalDomainAnalyzer signature.
        If genome provided, uses it; otherwise extracts using new system.
        """
        # If genome provided, we can still use it but extract additional features
        if genome is not None:
            # Use provided genome but add historial/uniquity
            # This is a hybrid approach
            print("Note: Using provided genome, adding historial/uniquity features...")
            
            # Extract historial and uniquity separately
            from src.config import HistorialCalculator, UniquityCalculator
            
            historial_calc = HistorialCalculator()
            uniquity_calc = UniquityCalculator()
            
            # Fit on provided data
            historial_calc.fit(texts, outcomes)
            uniquity_calc.fit(texts)
            
            # Extract additional features
            historial_features = np.array([historial_calc.transform(text) for text in texts])
            uniquity_features = np.array([uniquity_calc.transform(text) for text in texts])
            
            # Combine with provided genome
            enhanced_genome = np.hstack([genome, historial_features, uniquity_features])
            
            # Use story quality calculator
            from src.analysis.story_quality import StoryQualityCalculator
            story_quality_calc = StoryQualityCalculator(self.п)
            story_quality = story_quality_calc.compute_ю(enhanced_genome, feature_names or [])
            
            # Calculate Д
            from src.analysis.bridge_calculator import BridgeCalculator
            bridge_calc = BridgeCalculator()
            D_results = bridge_calc.calculate_D(
                story_quality, outcomes,
                baseline_features=baseline_features,
                domain_hint=self.domain_name
            )
            
            # Return backward-compatible format
            return {
                'domain': self.domain_name,
                'п': self.п,
                'genome': enhanced_genome,
                'story_quality': story_quality,
                'outcomes': outcomes,
                'delta': D_results['Д'],
                'r_narrative': D_results['r_narrative'],
                'r_baseline': D_results['r_baseline'],
                'passes_threshold': D_results['passes_threshold'],
                'interpretation': D_results['interpretation'],
                # New features
                'historial_features': historial_features,
                'uniquity_features': uniquity_features,
                'architecture': 'hybrid_backward_compatible'
            }
        
        else:
            # Use new system fully
            results = self._analyzer.analyze_complete(
                texts=texts,
                outcomes=outcomes,
                names=names
            )
            
            # Convert to backward-compatible format
            return {
                'domain': results['domain'],
                'п': results['narrativity'],
                'genome': results['genomes'],
                'story_quality': results['story_quality'],
                'outcomes': results['outcomes'],
                'delta': results['delta'],
                'r_narrative': results['r'],
                'r_baseline': None,  # Not computed in new system
                'passes_threshold': results['passes_threshold'],
                'interpretation': f"Domain-specific Ξ architecture: R²={results['r_squared']:.1%}",
                # New features
                'historial_features': results['historial_features'],
                'uniquity_features': results['uniquity_features'],
                'architecture': 'domain_specific_xi'
            }


# Alias for backward compatibility
UniversalDomainAnalyzer = BackwardCompatibleAnalyzer

