"""
Galaxy Formation Detector

Finds gravitational clusters (galaxies) in narrative space.

High-mass, similar comparisons gravitate together forming galaxy-like structures.
This module detects and characterizes these clusters.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import warnings


@dataclass
class Galaxy:
    """A gravitational cluster of comparisons."""
    galaxy_id: str
    center_body_id: str  # Supermassive center
    member_ids: List[str]
    total_mass: float
    mean_similarity: float
    cohesion: float  # How tightly bound
    domain_composition: Dict[str, int]
    
    def __repr__(self):
        return (
            f"Galaxy {self.galaxy_id}: {len(self.member_ids)} bodies, "
            f"mass={self.total_mass:.1f}, cohesion={self.cohesion:.2f}"
        )


class GalaxyDetector:
    """
    Detects galaxy formation in narrative space.
    
    Uses gravitational forces to identify tight clusters of
    high-mass, similar comparisons.
    """
    
    def __init__(self, gravity_calculator: Any):
        """
        Parameters
        ----------
        gravity_calculator : GravityCalculator
            Pre-computed gravitational system
        """
        self.gravity_calc = gravity_calculator
        self.galaxies: Dict[str, Galaxy] = {}
    
    def detect_galaxies(
        self,
        force_threshold: float = 1.0,
        min_members: int = 3
    ) -> Dict[str, Galaxy]:
        """
        Detect galaxy clusters based on gravitational forces.
        
        Parameters
        ----------
        force_threshold : float
            Minimum force to be in same galaxy
        min_members : int
            Minimum bodies to form a galaxy
            
        Returns
        -------
        Dict mapping galaxy_id to Galaxy objects
        """
        if not self.gravity_calc.forces:
            self.gravity_calc.calculate_all_forces()
        
        # Build adjacency based on strong forces
        body_ids = list(self.gravity_calc.bodies.keys())
        n = len(body_ids)
        
        # Create force matrix
        force_matrix = np.zeros((n, n))
        
        for (id1, id2), force in self.gravity_calc.forces.items():
            i = body_ids.index(id1)
            j = body_ids.index(id2)
            force_matrix[i, j] = force.force_magnitude
            force_matrix[j, i] = force.force_magnitude
        
        # Cluster based on forces
        # Convert force to distance (inverse relationship)
        distance_matrix = 1.0 / (force_matrix + 0.1)
        
        # Hierarchical clustering
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method='average')
        
        # Cut tree to form clusters
        cluster_labels = fcluster(
            linkage_matrix,
            t=1.0/force_threshold,  # Threshold in distance space
            criterion='distance'
        )
        
        # Build galaxies from clusters
        galaxy_id_counter = 0
        
        for cluster_id in np.unique(cluster_labels):
            member_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(member_indices) < min_members:
                continue
            
            member_ids = [body_ids[i] for i in member_indices]
            
            galaxy = self._create_galaxy(
                f"galaxy_{galaxy_id_counter}",
                member_ids
            )
            
            self.galaxies[galaxy.galaxy_id] = galaxy
            galaxy_id_counter += 1
        
        return self.galaxies
    
    def _create_galaxy(
        self,
        galaxy_id: str,
        member_ids: List[str]
    ) -> Galaxy:
        """Create Galaxy object from member bodies."""
        # Find center (highest mass)
        members = [self.gravity_calc.bodies[id] for id in member_ids]
        center = max(members, key=lambda b: b.mass)
        
        # Total mass
        total_mass = sum(b.mass for b in members)
        
        # Mean pairwise similarity
        similarities = []
        for i, id1 in enumerate(member_ids):
            for id2 in member_ids[i+1:]:
                key = tuple(sorted([id1, id2]))
                if key in self.gravity_calc.forces:
                    similarities.append(self.gravity_calc.forces[key].similarity)
        
        mean_similarity = np.mean(similarities) if similarities else 0.0
        
        # Cohesion: ratio of internal to external forces
        internal_forces = []
        external_forces = []
        
        for id1 in member_ids:
            for id2, force in self.gravity_calc.forces.items():
                if id1 in id2:
                    other_id = id2[0] if id2[1] == id1 else id2[1]
                    if other_id in member_ids:
                        internal_forces.append(force.force_magnitude)
                    else:
                        external_forces.append(force.force_magnitude)
        
        cohesion = (
            np.mean(internal_forces) / (np.mean(external_forces) + 1e-10)
            if internal_forces else 0.0
        )
        
        # Domain composition
        domain_counts = {}
        for body in members:
            domain_counts[body.domain] = domain_counts.get(body.domain, 0) + 1
        
        return Galaxy(
            galaxy_id=galaxy_id,
            center_body_id=center.body_id,
            member_ids=member_ids,
            total_mass=total_mass,
            mean_similarity=mean_similarity,
            cohesion=cohesion,
            domain_composition=domain_counts
        )
    
    def find_galaxy_collisions(self) -> List[Tuple[str, str, float]]:
        """
        Find pairs of galaxies that are gravitationally attracted.
        
        Returns
        -------
        List of (galaxy1_id, galaxy2_id, force) tuples
        """
        collisions = []
        
        galaxy_ids = list(self.galaxies.keys())
        
        for i, gal1_id in enumerate(galaxy_ids):
            for gal2_id in galaxy_ids[i+1:]:
                gal1 = self.galaxies[gal1_id]
                gal2 = self.galaxies[gal2_id]
                
                # Calculate force between galaxy centers
                center1_id = gal1.center_body_id
                center2_id = gal2.center_body_id
                
                key = tuple(sorted([center1_id, center2_id]))
                if key in self.gravity_calc.forces:
                    force = self.gravity_calc.forces[key].force_magnitude
                    
                    if force > 0.5:  # Significant attraction
                        collisions.append((gal1_id, gal2_id, force))
        
        return sorted(collisions, key=lambda x: x[2], reverse=True)
    
    def detect_voids(
        self,
        grid_resolution: int = 10
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Detect voids (empty regions) in narrative space.
        
        Parameters
        ----------
        grid_resolution : int
            Resolution of spatial grid
            
        Returns
        -------
        List of (position, void_size) tuples
        """
        # Get all body positions
        positions = np.array([
            body.position for body in self.gravity_calc.bodies.values()
        ])
        
        if len(positions) < 2:
            return []
        
        # Create grid
        mins = np.min(positions, axis=0)
        maxs = np.max(positions, axis=0)
        
        # Sample grid points
        voids = []
        
        for _ in range(100):  # Sample 100 random points
            point = np.random.uniform(mins, maxs)
            
            # Find distance to nearest body
            distances = [
                np.linalg.norm(point - pos)
                for pos in positions
            ]
            
            min_distance = min(distances)
            
            # If far from all bodies, it's a void
            if min_distance > 2.0:  # Threshold
                voids.append((point, min_distance))
        
        return sorted(voids, key=lambda x: x[1], reverse=True)[:10]
    
    def generate_report(self) -> str:
        """Generate galaxy detection report."""
        report = []
        report.append("=" * 80)
        report.append("GALAXY FORMATION ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        if not self.galaxies:
            report.append("No galaxies detected.")
            return "\n".join(report)
        
        report.append(f"GALAXIES DETECTED: {len(self.galaxies)}")
        report.append("")
        
        for galaxy_id, galaxy in sorted(
            self.galaxies.items(),
            key=lambda x: x[1].total_mass,
            reverse=True
        ):
            report.append(str(galaxy))
            report.append(f"  Center: {galaxy.center_body_id}")
            report.append(f"  Members: {len(galaxy.member_ids)}")
            report.append(f"  Domains: {galaxy.domain_composition}")
            report.append(f"  Cohesion: {galaxy.cohesion:.2f}")
            report.append("")
        
        # Galaxy collisions
        collisions = self.find_galaxy_collisions()
        if collisions:
            report.append("GALAXY COLLISIONS:")
            for gal1, gal2, force in collisions[:5]:
                report.append(f"  {gal1} ← → {gal2} (force: {force:.2f})")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

