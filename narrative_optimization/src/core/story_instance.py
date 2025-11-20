"""
Story Instance - The Fundamental Unit of Narrative Analysis

A story instance represents a single narrative within a domain.
This is the complete data structure containing all narrative physics.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, field, asdict


@dataclass
class StoryInstance:
    """
    A single story instance within a domain.
    The fundamental unit of narrative analysis.
    
    Attributes
    ----------
    instance_id : str
        Unique identifier for this instance
    domain : str
        Domain name (e.g., 'golf', 'supreme_court', 'nba')
    narrative_text : str
        The narrative description text
    timestamp : datetime, optional
        When this instance occurred
    context : dict
        Contextual information about the instance
    
    Genome Components (ж)
    --------------------
    genome_full : np.ndarray
        Complete genome vector (ж)
    genome_nominative : np.ndarray
        Nominative component
    genome_archetypal : np.ndarray
        Archetypal component (distance from Ξ)
    genome_historial : np.ndarray
        Historial component (historical positioning)
    genome_uniquity : np.ndarray
        Uniquity component (rarity score)
    genome_concurrent : np.ndarray, optional
        Concurrent narratives component (multi-stream)
    
    Story Quality (ю)
    ----------------
    story_quality : float
        Aggregated story quality score (0-1)
    story_quality_method : str
        Method used to calculate ю
    
    Outcome (❊)
    ----------
    outcome : float
        Success/failure or performance result
    outcome_type : str
        Type of outcome (binary, continuous, etc.)
    
    Mass (μ)
    -------
    mass : float
        Gravitational mass = importance × stakes
    importance_score : float
        Intrinsic importance
    stakes_multiplier : float
        Context-dependent stakes (1.0 - 3.0)
    
    Forces
    ------
    narrative_gravity : dict
        ф forces to other instances in domain
    nominative_gravity : dict
        ة forces (name-based attraction)
    imperative_gravity : dict
        ф_imperative forces to instances in other domains
    
    Dynamic Properties
    -----------------
    pi_effective : float
        Instance-specific narrativity (varies by complexity)
    pi_domain_base : float
        Domain base narrativity
    complexity_factors : dict
        Factors determining instance complexity
    blind_narratio : float, optional
        Instance-specific Β (equilibrium ratio)
    
    Awareness Components
    -------------------
    theta_resistance : float, optional
        Awareness suppressing narrative effects
    theta_amplification : float, optional
        Awareness amplifying potential energy
    awareness_features : dict, optional
        Detailed awareness feature breakdown
    
    Concurrent Narratives
    --------------------
    concurrent_narratives : list
        Multi-stream analysis results
    stream_count : int, optional
        Number of concurrent narrative threads
    stream_features : dict, optional
        Stream-specific features
    
    Feature Storage
    --------------
    features_all : dict
        All extracted features by transformer
    feature_names : list
        Names of all features in genome
    """
    
    # Identity
    instance_id: str
    domain: str
    narrative_text: str = ""
    timestamp: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Genome Components (ж)
    genome_full: Optional[np.ndarray] = None
    genome_nominative: Optional[np.ndarray] = None
    genome_archetypal: Optional[np.ndarray] = None
    genome_historial: Optional[np.ndarray] = None
    genome_uniquity: Optional[np.ndarray] = None
    genome_concurrent: Optional[np.ndarray] = None
    
    # Story Quality (ю)
    story_quality: Optional[float] = None
    story_quality_method: str = "weighted_mean"
    
    # Outcome (❊)
    outcome: Optional[float] = None
    outcome_type: str = "binary"
    
    # Mass (μ)
    mass: Optional[float] = None
    importance_score: float = 1.0
    stakes_multiplier: float = 1.0
    
    # Forces
    narrative_gravity: Dict[str, float] = field(default_factory=dict)
    nominative_gravity: Dict[str, float] = field(default_factory=dict)
    imperative_gravity: Dict[str, float] = field(default_factory=dict)
    
    # Dynamic Properties
    pi_effective: Optional[float] = None
    pi_domain_base: Optional[float] = None
    complexity_factors: Dict[str, float] = field(default_factory=dict)
    blind_narratio: Optional[float] = None
    
    # Awareness Components
    theta_resistance: Optional[float] = None
    theta_amplification: Optional[float] = None
    awareness_features: Dict[str, float] = field(default_factory=dict)
    
    # Concurrent Narratives
    concurrent_narratives: List[Dict] = field(default_factory=list)
    stream_count: Optional[int] = None
    stream_features: Dict[str, float] = field(default_factory=dict)
    
    # Feature Storage
    features_all: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    def calculate_mass(self) -> float:
        """
        Calculate gravitational mass (μ) from importance and stakes.
        
        μ = importance_score × stakes_multiplier
        
        Returns
        -------
        float
            Mass value (typically 0.3 - 3.0)
        """
        self.mass = self.importance_score * self.stakes_multiplier
        return self.mass
    
    def set_genome_components(
        self,
        nominative: np.ndarray,
        archetypal: np.ndarray,
        historial: np.ndarray,
        uniquity: np.ndarray,
        concurrent: Optional[np.ndarray] = None
    ):
        """
        Set genome components and assemble full genome.
        
        Parameters
        ----------
        nominative : ndarray
            Nominative features
        archetypal : ndarray
            Archetypal features
        historial : ndarray
            Historial features
        uniquity : ndarray
            Uniquity features
        concurrent : ndarray, optional
            Concurrent narrative features
        """
        self.genome_nominative = nominative
        self.genome_archetypal = archetypal
        self.genome_historial = historial
        self.genome_uniquity = uniquity
        self.genome_concurrent = concurrent
        
        # Assemble full genome
        components = [nominative, archetypal, historial, uniquity]
        if concurrent is not None:
            components.append(concurrent)
        
        self.genome_full = np.concatenate(components)
        self.updated_at = datetime.now()
    
    def calculate_story_quality(
        self,
        weights: Dict[str, float],
        feature_groups: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate ю (story quality) from genome using weights.
        
        Parameters
        ----------
        weights : dict
            Feature weights by transformer
        feature_groups : dict
            Features grouped by transformer
        
        Returns
        -------
        float
            Story quality score (0-1)
        """
        if self.genome_full is None:
            raise ValueError("Genome must be set before calculating story quality")
        
        weighted_scores = []
        for transformer_name, weight in weights.items():
            if transformer_name in feature_groups:
                features = feature_groups[transformer_name]
                score = np.mean(features) * weight
                weighted_scores.append(score)
        
        self.story_quality = np.sum(weighted_scores) if weighted_scores else 0.5
        
        # Normalize to [0, 1]
        self.story_quality = np.clip(self.story_quality, 0.0, 1.0)
        self.updated_at = datetime.now()
        
        return self.story_quality
    
    def calculate_blind_narratio(
        self,
        deterministic_forces: float,
        free_will_forces: float
    ) -> float:
        """
        Calculate Β (Blind Narratio) - equilibrium ratio.
        
        Β = deterministic_forces / free_will_forces
        
        Where:
        - deterministic_forces = ة (nominative) + λ (constraints)
        - free_will_forces = θ (awareness) + agency
        
        Parameters
        ----------
        deterministic_forces : float
            Sum of nominative gravity and fundamental constraints
        free_will_forces : float
            Sum of awareness resistance and agency
        
        Returns
        -------
        float
            Blind Narratio ratio
        """
        if free_will_forces == 0:
            self.blind_narratio = np.inf
        else:
            self.blind_narratio = deterministic_forces / free_will_forces
        
        self.updated_at = datetime.now()
        return self.blind_narratio
    
    def add_imperative_gravity(
        self,
        target_domain: str,
        target_instance_id: str,
        force_magnitude: float
    ):
        """
        Add cross-domain imperative gravity force.
        
        Parameters
        ----------
        target_domain : str
            Domain of target instance
        target_instance_id : str
            ID of target instance
        force_magnitude : float
            Magnitude of gravitational attraction
        """
        key = f"{target_domain}::{target_instance_id}"
        self.imperative_gravity[key] = force_magnitude
        self.updated_at = datetime.now()
    
    def get_top_imperative_neighbors(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N imperative gravity neighbors (strongest cross-domain pulls).
        
        Parameters
        ----------
        n : int
            Number of neighbors to return
        
        Returns
        -------
        list of tuple
            [(domain::instance_id, force), ...]
        """
        sorted_forces = sorted(
            self.imperative_gravity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_forces[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns
        -------
        dict
            Serializable dictionary representation
        """
        data = asdict(self)
        
        # Convert numpy arrays to lists
        for key in ['genome_full', 'genome_nominative', 'genome_archetypal', 
                    'genome_historial', 'genome_uniquity', 'genome_concurrent']:
            if data[key] is not None:
                data[key] = data[key].tolist()
        
        # Convert datetime objects
        if data['timestamp']:
            data['timestamp'] = data['timestamp'].isoformat()
        data['created_at'] = data['created_at'].isoformat()
        data['updated_at'] = data['updated_at'].isoformat()
        
        # Convert nested numpy arrays in features_all
        features_all_serializable = {}
        for transformer, features in data['features_all'].items():
            if isinstance(features, np.ndarray):
                features_all_serializable[transformer] = features.tolist()
            else:
                features_all_serializable[transformer] = features
        data['features_all'] = features_all_serializable
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryInstance':
        """
        Create StoryInstance from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary representation
        
        Returns
        -------
        StoryInstance
            Reconstructed instance
        """
        # Convert lists back to numpy arrays
        for key in ['genome_full', 'genome_nominative', 'genome_archetypal',
                    'genome_historial', 'genome_uniquity', 'genome_concurrent']:
            if data.get(key) is not None:
                data[key] = np.array(data[key])
        
        # Convert datetime strings
        if data.get('timestamp'):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert features_all lists back to arrays
        if 'features_all' in data:
            features_all_arrays = {}
            for transformer, features in data['features_all'].items():
                if isinstance(features, list):
                    features_all_arrays[transformer] = np.array(features)
                else:
                    features_all_arrays[transformer] = features
            data['features_all'] = features_all_arrays
        
        return cls(**data)
    
    def save(self, filepath: str):
        """
        Save instance to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'StoryInstance':
        """
        Load instance from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to load file
        
        Returns
        -------
        StoryInstance
            Loaded instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"StoryInstance(id='{self.instance_id}', domain='{self.domain}', "
                f"ю={self.story_quality:.3f if self.story_quality else 'None'}, "
                f"❊={self.outcome if self.outcome is not None else 'None'}, "
                f"π_eff={self.pi_effective:.3f if self.pi_effective else 'None'})")

