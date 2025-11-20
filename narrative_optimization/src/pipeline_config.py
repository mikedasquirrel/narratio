"""
Pipeline Configuration

Central configuration for the complete narrative analysis pipeline.

Author: Narrative Integration System
Date: November 2025
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    """
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = None
    output_dir: Path = None
    cache_dir: Path = None
    
    # Learning parameters
    learning_enabled: bool = True
    incremental_learning: bool = True
    auto_prune_patterns: bool = True
    min_improvement_threshold: float = 0.01
    
    # Discovery parameters
    min_pattern_frequency: float = 0.03
    n_patterns_per_domain: int = 8
    n_universal_patterns: int = 15
    
    # Validation parameters
    validation_alpha: float = 0.05
    min_effect_size: float = 0.2
    min_predictive_power: float = 0.55
    
    # Performance parameters
    enable_caching: bool = True
    enable_profiling: bool = False
    max_cache_size_mb: int = 1000
    
    # Integration parameters
    transfer_learning: bool = True
    min_transfer_similarity: float = 0.6
    few_shot_n: int = 5
    
    def __post_init__(self):
        """Initialize paths."""
        if self.data_dir is None:
            self.data_dir = self.project_root / 'data' / 'domains'
        
        if self.output_dir is None:
            self.output_dir = self.project_root / 'narrative_optimization' / 'domains'
        
        if self.cache_dir is None:
            self.cache_dir = Path.home() / '.narrative_optimization' / 'cache'
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_domain_data_path(self, domain_name: str) -> Optional[Path]:
        """Get data file path for domain."""
        # Try multiple patterns
        patterns = [
            self.data_dir / f'{domain_name}.json',
            self.data_dir / f'{domain_name}_complete_dataset.json',
            self.data_dir / f'{domain_name}_data.json',
            self.project_root / 'data' / 'domains' / f'{domain_name}.json'
        ]
        
        for path in patterns:
            if path.exists():
                return path
        
        return None
    
    def get_domain_output_path(self, domain_name: str) -> Path:
        """Get output directory for domain."""
        path = self.output_dir / domain_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'PipelineConfig':
        """Load config from JSON file."""
        import json
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def save_to_file(self, config_path: Path):
        """Save config to JSON file."""
        import json
        
        config_dict = {
            'project_root': str(self.project_root),
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'cache_dir': str(self.cache_dir),
            'learning_enabled': self.learning_enabled,
            'incremental_learning': self.incremental_learning,
            'auto_prune_patterns': self.auto_prune_patterns,
            'min_improvement_threshold': self.min_improvement_threshold,
            'min_pattern_frequency': self.min_pattern_frequency,
            'n_patterns_per_domain': self.n_patterns_per_domain,
            'n_universal_patterns': self.n_universal_patterns,
            'validation_alpha': self.validation_alpha,
            'min_effect_size': self.min_effect_size,
            'min_predictive_power': self.min_predictive_power,
            'enable_caching': self.enable_caching,
            'enable_profiling': self.enable_profiling,
            'max_cache_size_mb': self.max_cache_size_mb,
            'transfer_learning': self.transfer_learning,
            'min_transfer_similarity': self.min_transfer_similarity,
            'few_shot_n': self.few_shot_n
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


# Global config instance
_global_config = None


def get_config() -> PipelineConfig:
    """Get global pipeline config."""
    global _global_config
    if _global_config is None:
        _global_config = PipelineConfig()
    return _global_config


def set_config(config: PipelineConfig):
    """Set global pipeline config."""
    global _global_config
    _global_config = config

