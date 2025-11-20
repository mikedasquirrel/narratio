"""
Transformer Factory

Creates and configures transformers with domain awareness.

Author: Narrative Integration System
Date: November 2025
"""

# FIX TENSORFLOW MUTEX DEADLOCK ON MACOS
import os
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from typing import List, Optional, Dict
from pathlib import Path

from ..config import DomainConfig
from .registry import get_transformer_registry


class TransformerFactory:
    """
    Factory for creating domain-aware transformers.
    
    Handles:
    - Creating transformers with domain config
    - Injecting learned patterns
    - Managing transformer lifecycle
    """
    
    def __init__(self, domain_config: Optional[DomainConfig] = None):
        self.domain_config = domain_config
        self.learned_patterns = {}
        
    def set_learned_patterns(self, patterns: Dict):
        """Set learned patterns to inject into transformers."""
        self.learned_patterns = patterns
    
    def create_transformer(self, transformer_name: str, **kwargs):
        """
        Create a transformer with domain configuration.
        
        Parameters
        ----------
        transformer_name : str
            Transformer class name or ID
        **kwargs
            Additional parameters
        
        Returns
        -------
        object
            Configured transformer instance
        """
        registry = get_transformer_registry()
        metadata = registry.resolve(transformer_name)
        if metadata is None:
            suggestions = registry.suggest(transformer_name)
            hint = ""
            if suggestions:
                hint = f" Did you mean: {', '.join(suggestions)}?"
            raise ValueError(
                f"Transformer not found: '{transformer_name}'."
                f"{hint} Run 'python -m narrative_optimization.tools.list_transformers' "
                "to inspect the available catalog."
            )
        
        # Import transformers lazily (still uses __getattr__ caching)
        from .. import transformers
        transformer_class = getattr(transformers, metadata.class_name)
        
        # Create with domain config
        try:
            transformer = transformer_class(domain_config=self.domain_config, **kwargs)
        except TypeError:
            # Doesn't accept domain_config yet
            transformer = transformer_class(**kwargs)
        
        # Inject learned patterns if available
        if hasattr(transformer, 'update_learned_patterns') and self.learned_patterns:
            transformer.update_learned_patterns(self.learned_patterns)
        
        return transformer
    
    def create_all_transformers(self) -> List:
        """Create all available transformers."""
        transformer_names = [
            'Statistical',
            'NarrativePotential',
            'SelfPerception',
            'Framing',
            'CognitiveFluency',
            'Ensemble',
            'Phonetic',
            'TemporalEvolution',
            'InformationTheory',
            'SocialStatus',
            'Optics',
            'Relational',
            'AwarenessAmplification'
        ]
        
        transformers = []
        
        for name in transformer_names:
            try:
                transformer = self.create_transformer(name)
                transformers.append(transformer)
            except Exception as e:
                print(f"  ⚠ Could not create {name}: {e}")
        
        return transformers
    
    def create_domain_specific_set(self, domain_name: str) -> List:
        """
        Create optimal transformer set for a domain.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        
        Returns
        -------
        list
            Optimized transformer set
        """
        # Load domain config
        config = DomainConfig(domain_name)
        
        # Create factory with config
        factory = TransformerFactory(config)
        
        # Get domain archetype transformer
        from .archetypes import (
            GolfArchetypeTransformer,
            TennisArchetypeTransformer,
            ChessArchetypeTransformer,
            BoxingArchetypeTransformer,
            NBAArchetypeTransformer,
            WWEArchetypeTransformer,
            OscarsArchetypeTransformer,
            CryptoArchetypeTransformer,
            MentalHealthArchetypeTransformer,
            StartupsArchetypeTransformer,
            HurricanesArchetypeTransformer,
            HousingArchetypeTransformer
        )
        
        archetype_map = {
            'golf': GolfArchetypeTransformer,
            'tennis': TennisArchetypeTransformer,
            'chess': ChessArchetypeTransformer,
            'boxing': BoxingArchetypeTransformer,
            'nba': NBAArchetypeTransformer,
            'wwe': WWEArchetypeTransformer,
            'oscars': OscarsArchetypeTransformer,
            'crypto': CryptoArchetypeTransformer,
            'mental_health': MentalHealthArchetypeTransformer,
            'startups': StartupsArchetypeTransformer,
            'hurricanes': HurricanesArchetypeTransformer,
            'housing': HousingArchetypeTransformer
        }
        
        transformers = []
        
        # Add domain archetype transformer
        if domain_name in archetype_map:
            transformers.append(archetype_map[domain_name]())
        
        # Add core transformers
        core_transformers = factory.create_all_transformers()
        transformers.extend(core_transformers)
        
        return transformers

