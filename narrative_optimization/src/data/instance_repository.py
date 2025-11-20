"""
Instance Repository - Centralized Storage for Story Instances

Manages all story instances across all domains with efficient indexing
and querying for cross-domain analysis and imperative gravity calculations.

Author: Narrative Optimization Framework
Date: November 2025
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.story_instance import StoryInstance


class InstanceRepository:
    """
    Centralized storage and query system for all story instances.
    
    Features:
    - Multi-index system for fast queries
    - Cross-domain lookups
    - Imperative neighbor finding
    - Persistence to disk
    - Efficient similarity calculations
    
    Parameters
    ----------
    storage_path : str or Path, optional
        Path to repository storage directory
        Defaults to ~/.narrative_cache/instance_repository/
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize repository with storage location."""
        if storage_path is None:
            storage_path = Path.home() / '.narrative_cache' / 'instance_repository'
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Main storage
        self.instances: Dict[str, StoryInstance] = {}
        
        # Indices for fast lookup
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)
        self.pi_index: Dict[Tuple[float, float], Set[str]] = defaultdict(set)
        self.beta_index: Dict[Tuple[float, float], Set[str]] = defaultdict(set)
        self.outcome_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Structural similarity cache
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Load existing data if available
        self._load_from_disk()
    
    def add_instance(self, instance: StoryInstance, reindex: bool = True):
        """
        Add a story instance to the repository.
        
        Parameters
        ----------
        instance : StoryInstance
            Instance to add
        reindex : bool
            Whether to rebuild indices (default True)
        """
        instance_id = instance.instance_id
        
        # Store instance
        self.instances[instance_id] = instance
        
        if reindex:
            # Update domain index
            self.domain_index[instance.domain].add(instance_id)
            
            # Update π index
            if instance.pi_effective is not None:
                pi_bin = self._get_pi_bin(instance.pi_effective)
                self.pi_index[pi_bin].add(instance_id)
            
            # Update Β index
            if instance.blind_narratio is not None:
                beta_bin = self._get_beta_bin(instance.blind_narratio)
                self.beta_index[beta_bin].add(instance_id)
            
            # Update outcome index
            if instance.outcome is not None:
                outcome_category = 'success' if instance.outcome > 0.5 else 'failure'
                self.outcome_index[outcome_category].add(instance_id)
    
    def add_instances_bulk(self, instances: List[StoryInstance]):
        """
        Add multiple instances efficiently (single reindex at end).
        
        Parameters
        ----------
        instances : list of StoryInstance
            Instances to add
        """
        # Add without reindexing each time
        for instance in instances:
            self.add_instance(instance, reindex=False)
        
        # Rebuild all indices once
        self._rebuild_indices()
    
    def get_instance(self, instance_id: str) -> Optional[StoryInstance]:
        """
        Retrieve instance by ID.
        
        Parameters
        ----------
        instance_id : str
            Instance identifier
        
        Returns
        -------
        StoryInstance or None
        """
        return self.instances.get(instance_id)
    
    def get_instances_by_domain(self, domain: str) -> List[StoryInstance]:
        """
        Get all instances from a specific domain.
        
        Parameters
        ----------
        domain : str
            Domain name
        
        Returns
        -------
        list of StoryInstance
        """
        instance_ids = self.domain_index.get(domain, set())
        return [self.instances[iid] for iid in instance_ids]
    
    def query_by_structure(
        self,
        pi_range: Optional[Tuple[float, float]] = None,
        theta_range: Optional[Tuple[float, float]] = None,
        exclude_domain: Optional[str] = None
    ) -> List[StoryInstance]:
        """
        Find instances with similar structural properties.
        
        Parameters
        ----------
        pi_range : tuple of float, optional
            (min_pi, max_pi)
        theta_range : tuple of float, optional
            (min_theta, max_theta)
        exclude_domain : str, optional
            Domain to exclude from results
        
        Returns
        -------
        list of StoryInstance
            Matching instances
        """
        candidates = set(self.instances.keys())
        
        # Filter by π
        if pi_range is not None:
            pi_candidates = set()
            for pi_bin, instance_ids in self.pi_index.items():
                bin_min, bin_max = pi_bin
                if (bin_min >= pi_range[0] and bin_max <= pi_range[1]) or \
                   (bin_min <= pi_range[0] and bin_max >= pi_range[0]) or \
                   (bin_min <= pi_range[1] and bin_max >= pi_range[1]):
                    pi_candidates.update(instance_ids)
            candidates &= pi_candidates
        
        # Filter by θ
        if theta_range is not None:
            theta_candidates = set()
            for iid in candidates:
                instance = self.instances[iid]
                if instance.theta_resistance is not None:
                    if theta_range[0] <= instance.theta_resistance <= theta_range[1]:
                        theta_candidates.add(iid)
            candidates &= theta_candidates
        
        # Filter by domain
        if exclude_domain is not None:
            candidates = {iid for iid in candidates 
                         if self.instances[iid].domain != exclude_domain}
        
        return [self.instances[iid] for iid in candidates]
    
    def calculate_imperative_neighbors(
        self,
        instance: StoryInstance,
        n_neighbors: int = 5,
        exclude_same_domain: bool = True
    ) -> List[Tuple[StoryInstance, float]]:
        """
        Find instances from other domains with highest imperative gravity.
        
        Parameters
        ----------
        instance : StoryInstance
            Query instance
        domain : str
            Current instance's domain
        n_neighbors : int
            Number of neighbors to return
        exclude_same_domain : bool
            Whether to exclude instances from same domain
        
        Returns
        -------
        list of tuple
            [(neighbor_instance, similarity_score), ...]
        """
        similarities = []
        
        for other_id, other_instance in self.instances.items():
            # Skip self
            if other_id == instance.instance_id:
                continue
            
            # Skip same domain if requested
            if exclude_same_domain and other_instance.domain == instance.domain:
                continue
            
            # Calculate structural similarity
            similarity = self._calculate_structural_similarity(instance, other_instance)
            similarities.append((other_instance, similarity))
        
        # Sort by similarity (descending) and take top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_neighbors]
    
    def get_domain_statistics(self, domain: str) -> Dict:
        """
        Get statistical summary for a domain.
        
        Parameters
        ----------
        domain : str
            Domain name
        
        Returns
        -------
        dict
            Statistics including count, avg π, avg Β, etc.
        """
        instances = self.get_instances_by_domain(domain)
        
        if not instances:
            return {'count': 0}
        
        pi_values = [i.pi_effective for i in instances if i.pi_effective is not None]
        beta_values = [i.blind_narratio for i in instances if i.blind_narratio is not None]
        quality_values = [i.story_quality for i in instances if i.story_quality is not None]
        outcome_values = [i.outcome for i in instances if i.outcome is not None]
        
        return {
            'count': len(instances),
            'pi_mean': np.mean(pi_values) if pi_values else None,
            'pi_std': np.std(pi_values) if pi_values else None,
            'pi_min': np.min(pi_values) if pi_values else None,
            'pi_max': np.max(pi_values) if pi_values else None,
            'beta_mean': np.mean(beta_values) if beta_values else None,
            'beta_std': np.std(beta_values) if beta_values else None,
            'quality_mean': np.mean(quality_values) if quality_values else None,
            'outcome_mean': np.mean(outcome_values) if outcome_values else None,
            'success_rate': np.mean([o > 0.5 for o in outcome_values]) if outcome_values else None
        }
    
    def get_all_domains(self) -> List[str]:
        """Get list of all domains in repository."""
        return list(self.domain_index.keys())
    
    def get_repository_statistics(self) -> Dict:
        """Get overall repository statistics."""
        return {
            'total_instances': len(self.instances),
            'total_domains': len(self.domain_index),
            'domains': {domain: len(ids) for domain, ids in self.domain_index.items()}
        }
    
    def _calculate_structural_similarity(
        self,
        instance1: StoryInstance,
        instance2: StoryInstance
    ) -> float:
        """
        Calculate structural similarity between two instances.
        
        Considers:
        - π similarity
        - θ similarity
        - Genome similarity (if available)
        
        Returns
        -------
        float
            Similarity score (0-1)
        """
        # Check cache
        cache_key = tuple(sorted([instance1.instance_id, instance2.instance_id]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        similarities = []
        
        # π similarity
        if instance1.pi_effective is not None and instance2.pi_effective is not None:
            pi_sim = 1.0 - abs(instance1.pi_effective - instance2.pi_effective)
            similarities.append(pi_sim)
        
        # θ similarity
        if instance1.theta_resistance is not None and instance2.theta_resistance is not None:
            theta_sim = 1.0 - abs(instance1.theta_resistance - instance2.theta_resistance)
            similarities.append(theta_sim)
        
        # Genome similarity (cosine)
        if instance1.genome_full is not None and instance2.genome_full is not None:
            # Ensure same dimensionality
            if len(instance1.genome_full) == len(instance2.genome_full):
                dot_product = np.dot(instance1.genome_full, instance2.genome_full)
                norm1 = np.linalg.norm(instance1.genome_full)
                norm2 = np.linalg.norm(instance2.genome_full)
                if norm1 > 0 and norm2 > 0:
                    cosine_sim = dot_product / (norm1 * norm2)
                    similarities.append((cosine_sim + 1) / 2)  # Scale to 0-1
        
        # Average similarities
        if similarities:
            similarity = np.mean(similarities)
        else:
            similarity = 0.0
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def _get_pi_bin(self, pi: float, bin_size: float = 0.1) -> Tuple[float, float]:
        """Get π bin for indexing."""
        bin_min = (pi // bin_size) * bin_size
        bin_max = bin_min + bin_size
        return (round(bin_min, 2), round(bin_max, 2))
    
    def _get_beta_bin(self, beta: float, bin_size: float = 0.2) -> Tuple[float, float]:
        """Get Β bin for indexing."""
        # Handle infinity
        if np.isinf(beta):
            return (10.0, np.inf)
        
        bin_min = (beta // bin_size) * bin_size
        bin_max = bin_min + bin_size
        return (round(bin_min, 2), round(bin_max, 2))
    
    def _rebuild_indices(self):
        """Rebuild all indices from scratch."""
        # Clear indices
        self.domain_index.clear()
        self.pi_index.clear()
        self.beta_index.clear()
        self.outcome_index.clear()
        
        # Rebuild
        for instance_id, instance in self.instances.items():
            # Domain index
            self.domain_index[instance.domain].add(instance_id)
            
            # π index
            if instance.pi_effective is not None:
                pi_bin = self._get_pi_bin(instance.pi_effective)
                self.pi_index[pi_bin].add(instance_id)
            
            # Β index
            if instance.blind_narratio is not None:
                beta_bin = self._get_beta_bin(instance.blind_narratio)
                self.beta_index[beta_bin].add(instance_id)
            
            # Outcome index
            if instance.outcome is not None:
                outcome_category = 'success' if instance.outcome > 0.5 else 'failure'
                self.outcome_index[outcome_category].add(instance_id)
    
    def save_to_disk(self):
        """Persist repository to disk."""
        # Save instances
        instances_dir = self.storage_path / 'instances'
        instances_dir.mkdir(exist_ok=True)
        
        for instance_id, instance in self.instances.items():
            # Create domain subdirectory
            domain_dir = instances_dir / instance.domain
            domain_dir.mkdir(exist_ok=True)
            
            # Save instance
            filepath = domain_dir / f"{instance_id}.json"
            instance.save(str(filepath))
        
        # Save indices
        indices_path = self.storage_path / 'indices.pkl'
        with open(indices_path, 'wb') as f:
            pickle.dump({
                'domain_index': dict(self.domain_index),
                'pi_index': dict(self.pi_index),
                'beta_index': dict(self.beta_index),
                'outcome_index': dict(self.outcome_index)
            }, f)
        
        print(f"✓ Saved {len(self.instances)} instances to {self.storage_path}")
    
    def _load_from_disk(self):
        """Load repository from disk if it exists."""
        instances_dir = self.storage_path / 'instances'
        if not instances_dir.exists():
            return
        
        # Load all instances
        loaded_count = 0
        for domain_dir in instances_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            
            for instance_file in domain_dir.glob('*.json'):
                try:
                    instance = StoryInstance.load(str(instance_file))
                    self.instances[instance.instance_id] = instance
                    loaded_count += 1
                except Exception as e:
                    print(f"Warning: Could not load {instance_file}: {e}")
        
        # Load indices if available
        indices_path = self.storage_path / 'indices.pkl'
        if indices_path.exists():
            try:
                with open(indices_path, 'rb') as f:
                    indices = pickle.load(f)
                self.domain_index = defaultdict(set, indices['domain_index'])
                self.pi_index = defaultdict(set, indices['pi_index'])
                self.beta_index = defaultdict(set, indices['beta_index'])
                self.outcome_index = defaultdict(set, indices['outcome_index'])
            except Exception as e:
                print(f"Warning: Could not load indices, rebuilding: {e}")
                self._rebuild_indices()
        else:
            # Rebuild indices
            self._rebuild_indices()
        
        if loaded_count > 0:
            print(f"✓ Loaded {loaded_count} instances from {self.storage_path}")
    
    def clear(self):
        """Clear all instances and indices."""
        self.instances.clear()
        self.domain_index.clear()
        self.pi_index.clear()
        self.beta_index.clear()
        self.outcome_index.clear()
        self.similarity_cache.clear()
    
    def __len__(self) -> int:
        """Return number of instances in repository."""
        return len(self.instances)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"InstanceRepository(instances={len(self.instances)}, domains={len(self.domain_index)})"

