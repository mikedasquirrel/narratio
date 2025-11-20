"""
Domain-Aware Narrative Transformer (Transformer #35)

The CRITICAL transformer that understands:
- Domain structure (1v1, team vs team, vs market)
- Hierarchical context (match in season in career)
- Relational features (Home - Away, Player A vs Player B)
- Narrative intensity (comeback, upset, stakes)

This fixes the fundamental flaw: narratives are RELATIONAL, not absolute.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys

# Add core utilities to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from core.relational_features import (
    compute_relational_features,
    compute_narrative_intensity,
    compute_hierarchical_context
)


class DomainAwareNarrativeTransformer:
    """
    Transformer #35: Domain-Aware Narrative Analysis
    
    Extracts relational features based on domain structure.
    Understands that narratives are RELATIVE between competitors.
    """
    
    def __init__(
        self,
        domain_name: str = 'nfl',
        schema_dir: Optional[str] = None
    ):
        """
        Initialize with domain structure schema
        
        Args:
            domain_name: Name of the domain ('nfl', 'nba', 'tennis', etc.)
            schema_dir: Optional path to schema directory
        """
        self.domain_name = domain_name
        
        # Load domain structure schema
        if schema_dir is None:
            # Default to project root / domain_schemas / structural_definitions
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            schema_dir = project_root / 'domain_schemas' / 'structural_definitions'
        else:
            schema_dir = Path(schema_dir)
        
        schema_path = schema_dir / f'{domain_name}_structure.json'
        
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Domain structure schema not found: {schema_path}"
            )
        
        with open(schema_path, 'r') as f:
            self.domain_structure = json.load(f)
        
        self.relational_structure = self.domain_structure.get('relational_structure', '1v1')
        self.feature_names_ = []
        
    def transform(self, narratives: List[str]) -> np.ndarray:
        """
        Transform narratives into domain-aware relational features
        
        Args:
            narratives: List of narrative texts (can be enriched JSON strings)
            
        Returns:
            features: Array of shape (n_samples, n_features)
        """
        features_list = []
        
        for narrative in narratives:
            # Parse narrative (could be JSON or plain text)
            narrative_data = self._parse_narrative(narrative)
            
            # Extract features for this narrative
            features = self._extract_features(narrative_data)
            features_list.append(features)
        
        # Stack into array
        features_array = np.array(features_list)
        
        return features_array
    
    def _parse_narrative(self, narrative: str) -> Dict[str, Any]:
        """Parse narrative text or JSON into structured data"""
        try:
            # Try to parse as JSON
            data = json.loads(narrative)
            return data
        except (json.JSONDecodeError, TypeError):
            # Plain text narrative
            return {
                'text': str(narrative),
                'type': 'plain_text'
            }
    
    def _extract_features(self, narrative_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract all domain-aware features from narrative data
        
        Features include:
        - Hierarchical context (position in season/career)
        - Narrative intensity (comeback, upset, drama)
        - Domain-specific metrics
        """
        features = []
        feature_names = []
        
        # 1. Hierarchical Context Features
        context_features = compute_hierarchical_context(
            narrative_data,
            self.domain_structure
        )
        for name, value in context_features.items():
            # Handle NaN/None values
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0.0
            features.append(float(value))
            feature_names.append(f'domain_{name}')
        
        # 2. Narrative Intensity Features
        intensity_features = compute_narrative_intensity(
            narrative_data,
            self.domain_structure
        )
        for name, value in intensity_features.items():
            # Handle NaN/None values
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0.0
            features.append(float(value))
            feature_names.append(f'intensity_{name}')
        
        # 3. Domain-Specific Available Metrics
        available_metrics = self.domain_structure.get('available_metrics', [])
        for metric in available_metrics:
            value = narrative_data.get(metric, 0.0)
            # Handle NaN/None values
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0.0
            features.append(float(value))
            feature_names.append(f'metric_{metric}')
        
        # Store feature names for first call
        if not self.feature_names_:
            self.feature_names_ = feature_names
        
        # Ensure no NaN values in output
        features_array = np.array(features, dtype=float)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array
    
    def fit(self, narratives: List[str], y=None):
        """Fit transformer (mainly to establish feature names)"""
        # Extract features from first narrative to get feature names
        if narratives:
            sample_data = self._parse_narrative(narratives[0])
            self._extract_features(sample_data)
        return self
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names_


class RelationalNarrativeTransformer(DomainAwareNarrativeTransformer):
    """
    Extended transformer that computes RELATIONAL features between competitors
    
    This is the key to fixing NFL/NBA/Tennis models:
    - Takes PAIRED data (Home + Away)
    - Computes differentials, ratios, interactions
    - Outputs relational feature space
    """
    
    def __init__(
        self,
        domain_name: str = 'nfl',
        schema_dir: Optional[str] = None,
        compute_diff: bool = True,
        compute_ratio: bool = True,
        compute_interaction: bool = False
    ):
        """
        Initialize relational transformer
        
        Args:
            domain_name: Domain name
            schema_dir: Schema directory path
            compute_diff: Compute differential features (A - B)
            compute_ratio: Compute ratio features (A / B)
            compute_interaction: Compute interaction features (A * B)
        """
        super().__init__(domain_name, schema_dir)
        
        self.compute_diff = compute_diff
        self.compute_ratio = compute_ratio
        self.compute_interaction = compute_interaction
        
    def transform_paired(
        self,
        entity_a_narratives: List[str],
        entity_b_narratives: List[str]
    ) -> np.ndarray:
        """
        Transform paired narratives (Home vs Away, Player A vs Player B)
        
        Args:
            entity_a_narratives: Narratives for entity A (e.g., home team)
            entity_b_narratives: Narratives for entity B (e.g., away team)
            
        Returns:
            relational_features: Combined features (absolute + relational)
        """
        # Extract features for both entities
        features_a = self.transform(entity_a_narratives)
        features_b = self.transform(entity_b_narratives)
        
        # Get base feature names
        base_feature_names = self.get_feature_names()
        
        # Compute relational features for each pair
        n_samples = features_a.shape[0]
        relational_features_list = []
        
        for i in range(n_samples):
            # Compute relational features between A and B
            rel_features, rel_names = compute_relational_features(
                features_a[i],
                features_b[i],
                base_feature_names,
                compute_diff=self.compute_diff,
                compute_ratio=self.compute_ratio,
                compute_interaction=self.compute_interaction
            )
            relational_features_list.append(rel_features)
        
        # Stack relational features
        relational_array = np.array(relational_features_list)
        
        # Combine: [features_a, features_b, relational_features]
        combined = np.hstack([features_a, features_b, relational_array])
        
        # Update feature names
        if not hasattr(self, 'combined_feature_names_'):
            a_names = [f'entity_a_{name}' for name in base_feature_names]
            b_names = [f'entity_b_{name}' for name in base_feature_names]
            
            # Get relational names from one sample
            _, rel_names = compute_relational_features(
                features_a[0],
                features_b[0],
                base_feature_names,
                compute_diff=self.compute_diff,
                compute_ratio=self.compute_ratio,
                compute_interaction=self.compute_interaction
            )
            
            self.combined_feature_names_ = a_names + b_names + rel_names
        
        return combined
    
    def get_feature_names(self) -> List[str]:
        """Return combined feature names (absolute + relational)"""
        if hasattr(self, 'combined_feature_names_'):
            return self.combined_feature_names_
        else:
            return super().get_feature_names()


# Export classes
__all__ = [
    'DomainAwareNarrativeTransformer',
    'RelationalNarrativeTransformer'
]

