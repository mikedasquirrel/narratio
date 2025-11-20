"""
Domain Configuration System

YAML/JSON schema for domain definitions with automatic narrativity calculation
and validation. This is the foundation of the unified pipeline system.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml
import json
from enum import Enum


class DomainType(str, Enum):
    """Domain type classification"""
    SPORTS = "sports"
    SPORTS_INDIVIDUAL = "sports_individual"
    SPORTS_TEAM = "sports_team"
    ENTERTAINMENT = "entertainment"
    NOMINATIVE = "nominative"
    BUSINESS = "business"
    MEDICAL = "medical"
    HYBRID = "hybrid"


class OutcomeType(str, Enum):
    """Type of outcome variable"""
    BINARY = "binary"
    CONTINUOUS = "continuous"
    RANKED = "ranked"


@dataclass
class NarrativityComponents:
    """Narrativity component scores"""
    structural: float  # How many narrative paths possible? [0, 1]
    temporal: float  # Does it unfold over time? [0, 1]
    agency: float  # Do actors have choice? [0, 1]
    interpretive: float  # Is judgment subjective? [0, 1]
    format: float  # How flexible is the medium? [0, 1]
    
    def validate(self):
        """Validate all components are in [0, 1]"""
        for name, value in asdict(self).items():
            if not 0 <= value <= 1:
                raise ValueError(f"п_{name} must be in [0, 1], got {value}")
    
    def calculate_pi(self) -> float:
        """
        Calculate overall narrativity (п) from components.
        
        Formula:
        п = 0.30×structural + 0.20×temporal + 0.25×agency + 
            0.15×interpretive + 0.10×format
        """
        self.validate()
        return (
            0.30 * self.structural +
            0.20 * self.temporal +
            0.25 * self.agency +
            0.15 * self.interpretive +
            0.10 * self.format
        )


@dataclass
class DataSchema:
    """Data schema definition"""
    text_fields: List[str]  # Fields containing narrative text
    outcome_field: str  # Field containing outcomes
    context_fields: Optional[List[str]] = None  # Optional context features
    name_field: Optional[str] = None  # Field containing organism names
    
    def validate(self):
        """Validate schema is complete"""
        if not self.text_fields:
            raise ValueError("At least one text_field is required")
        if not self.outcome_field:
            raise ValueError("outcome_field is required")


@dataclass
class DomainConfig:
    """
    Complete domain configuration.
    
    This is the single source of truth for domain definitions.
    """
    domain: str  # Domain name (e.g., "tennis", "movies")
    type: DomainType  # Domain type classification
    narrativity: NarrativityComponents  # Narrativity components
    data: DataSchema  # Data schema
    outcome_type: OutcomeType  # Type of outcome variable
    
    # Optional fields
    transformer_augmentation: List[str] = field(default_factory=list)
    custom_validators: List[str] = field(default_factory=list)
    description: Optional[str] = None
    sample_size: Optional[int] = None
    
    # Multi-perspective configuration
    perspectives: List[str] = field(default_factory=lambda: ['director', 'audience', 'critic'])
    quality_methods: List[str] = field(default_factory=lambda: ['weighted_mean', 'ensemble'])
    scales: List[str] = field(default_factory=lambda: ['micro', 'meso', 'macro'])
    aggregation_method: str = 'importance_weighted'  # How to aggregate dimensions
    
    # Computed properties
    _pi: Optional[float] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Validate and compute derived values"""
        self.narrativity.validate()
        self.data.validate()
        self._pi = self.narrativity.calculate_pi()
    
    @property
    def pi(self) -> float:
        """Get calculated narrativity (п)"""
        if self._pi is None:
            self._pi = self.narrativity.calculate_pi()
        return self._pi
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainConfig':
        """
        Create DomainConfig from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with domain configuration
            
        Returns
        -------
        config : DomainConfig
            Validated domain configuration
        """
        # Convert narrativity dict to NarrativityComponents
        narrativity_data = data.get('narrativity', {})
        narrativity = NarrativityComponents(
            structural=narrativity_data.get('structural', 0.5),
            temporal=narrativity_data.get('temporal', 0.5),
            agency=narrativity_data.get('agency', 0.5),
            interpretive=narrativity_data.get('interpretive', 0.5),
            format=narrativity_data.get('format', 0.5)
        )
        
        # Convert data dict to DataSchema
        data_schema = data.get('data', {})
        schema = DataSchema(
            text_fields=data_schema.get('text_fields', []),
            outcome_field=data_schema.get('outcome_field', 'outcome'),
            context_fields=data_schema.get('context_features'),
            name_field=data_schema.get('name_field')
        )
        
        # Convert type string to DomainType enum
        domain_type_str = data.get('type', 'hybrid')
        try:
            domain_type = DomainType(domain_type_str)
        except ValueError:
            # Try to map common variations
            type_mapping = {
                'sports': DomainType.SPORTS,
                'entertainment': DomainType.ENTERTAINMENT,
                'nominative': DomainType.NOMINATIVE,
                'business': DomainType.BUSINESS,
                'medical': DomainType.MEDICAL,
            }
            domain_type = type_mapping.get(domain_type_str, DomainType.HYBRID)
        
        # Convert outcome_type string to OutcomeType enum
        outcome_type_str = data.get('outcome_type', 'continuous')
        try:
            outcome_type = OutcomeType(outcome_type_str)
        except ValueError:
            outcome_type = OutcomeType.CONTINUOUS
        
        return cls(
            domain=data.get('domain', 'unknown'),
            type=domain_type,
            narrativity=narrativity,
            data=schema,
            outcome_type=outcome_type,
            transformer_augmentation=data.get('transformer_augmentation', []),
            custom_validators=data.get('custom_validators', []),
            description=data.get('description'),
            sample_size=data.get('sample_size'),
            perspectives=data.get('perspectives', ['director', 'audience', 'critic']),
            quality_methods=data.get('quality_methods', ['weighted_mean', 'ensemble']),
            scales=data.get('scales', ['micro', 'meso', 'macro']),
            aggregation_method=data.get('aggregation_method', 'importance_weighted')
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'DomainConfig':
        """
        Load domain configuration from YAML file.
        
        Parameters
        ----------
        yaml_path : str or Path
            Path to YAML configuration file
            
        Returns
        -------
        config : DomainConfig
            Validated domain configuration
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DomainConfig':
        """
        Load domain configuration from JSON file.
        
        Parameters
        ----------
        json_path : str or Path
            Path to JSON configuration file
            
        Returns
        -------
        config : DomainConfig
            Validated domain configuration
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'domain': self.domain,
            'type': self.type.value,
            'narrativity': {
                'structural': self.narrativity.structural,
                'temporal': self.narrativity.temporal,
                'agency': self.narrativity.agency,
                'interpretive': self.narrativity.interpretive,
                'format': self.narrativity.format
            },
            'data': {
                'text_fields': self.data.text_fields,
                'outcome_field': self.data.outcome_field,
                'context_features': self.data.context_fields,
                'name_field': self.data.name_field
            },
            'outcome_type': self.outcome_type.value,
            'transformer_augmentation': self.transformer_augmentation,
            'custom_validators': self.custom_validators,
            'description': self.description,
            'sample_size': self.sample_size,
            'pi': self.pi,  # Include computed п
            'perspectives': self.perspectives,
            'quality_methods': self.quality_methods,
            'scales': self.scales,
            'aggregation_method': self.aggregation_method
        }
    
    def to_yaml(self, yaml_path: Union[str, Path]):
        """Save configuration to YAML file"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, json_path: Union[str, Path]):
        """Save configuration to JSON file"""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False)
    
    def validate_complete(self) -> List[str]:
        """
        Perform complete validation and return list of issues.
        
        Returns
        -------
        issues : list of str
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Validate narrativity
        try:
            self.narrativity.validate()
        except ValueError as e:
            issues.append(f"Narrativity validation: {e}")
        
        # Validate data schema
        try:
            self.data.validate()
        except ValueError as e:
            issues.append(f"Data schema validation: {e}")
        
        # Validate п is reasonable
        if not 0 <= self.pi <= 1:
            issues.append(f"Calculated п ({self.pi:.3f}) is outside [0, 1]")
        
        # Validate domain name
        if not self.domain or not isinstance(self.domain, str):
            issues.append("Domain name must be a non-empty string")
        
        return issues
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Domain: {self.domain}",
            f"Type: {self.type.value}",
            f"Narrativity (п): {self.pi:.3f}",
            f"  Structural: {self.narrativity.structural:.2f}",
            f"  Temporal: {self.narrativity.temporal:.2f}",
            f"  Agency: {self.narrativity.agency:.2f}",
            f"  Interpretive: {self.narrativity.interpretive:.2f}",
            f"  Format: {self.narrativity.format:.2f}",
            f"Outcome Type: {self.outcome_type.value}",
            f"Text Fields: {', '.join(self.data.text_fields)}",
            f"Outcome Field: {self.data.outcome_field}",
        ]
        
        if self.data.context_fields:
            lines.append(f"Context Fields: {', '.join(self.data.context_fields)}")
        
        if self.transformer_augmentation:
            lines.append(f"Custom Transformers: {', '.join(self.transformer_augmentation)}")
        
        return "\n".join(lines)

