"""
Gravitational Force Calculator

Implements the gravitational metaphor from theory:
- Each comparison has mass (context_weight × importance)
- Comparisons attract each other with force proportional to mass and similarity
- High-mass, similar comparisons form tight clusters (galaxies)

Formula: Force = (mass1 × mass2 × similarity) / distance²
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
import warnings


@dataclass
class GravitationalBody:
    """A comparison treated as gravitational body."""
    body_id: str
    mass: float  # context_weight × importance
    position: np.ndarray  # Feature vector (DNA) as position in narrative space
    domain: str
    velocity: Optional[np.ndarray] = None  # For dynamic simulations
    
    def __repr__(self):
        mass_class = self._classify_mass()
        return f"{mass_class} | {self.body_id} (mass={self.mass:.2f}, domain={self.domain})"
    
    def _classify_mass(self) -> str:
        """Classify by mass using theory's weight classes."""
        if self.mass >= 4.0:  # weight>=2.0 with high importance
            return "SUPERMASSIVE"
        elif self.mass >= 2.5:
            return "MASSIVE"
        elif self.mass >= 1.0:
            return "STANDARD"
        elif self.mass >= 0.5:
            return "DWARF"
        else:
            return "ASTEROID"


@dataclass
class GravitationalForce:
    """Force between two bodies."""
    body1_id: str
    body2_id: str
    force_magnitude: float
    force_vector: np.ndarray
    similarity: float  # Cosine similarity (0-1)
    distance: float  # Euclidean distance in feature space


class GravityCalculator:
    """
    Calculate gravitational forces between narrative comparisons.
    
    Implements: Force = (m1 × m2 × similarity) / distance²
    
    Where:
    - mass = context_weight × importance
    - similarity = cosine similarity of feature vectors
    - distance = Euclidean distance in feature space
    """
    
    def __init__(self, gravitational_constant: float = 1.0):
        """
        Parameters
        ----------
        gravitational_constant : float
            Scaling factor for force calculation
        """
        self.G = gravitational_constant
        self.bodies: Dict[str, GravitationalBody] = {}
        self.forces: Dict[Tuple[str, str], GravitationalForce] = {}
    
    def add_body(
        self,
        body_id: str,
        context_weight: float,
        importance: float,
        position: np.ndarray,
        domain: str
    ) -> GravitationalBody:
        """
        Add a gravitational body (comparison).
        
        Parameters
        ----------
        body_id : str
            Unique identifier
        context_weight : float
            Context weight (0.3-3.0 from theory)
        importance : float
            Importance score (0-1)
        position : np.ndarray
            Feature vector (position in narrative space)
        domain : str
            Domain name
            
        Returns
        -------
        GravitationalBody
        """
        # Mass = weight × importance
        mass = context_weight * (importance + 0.1)  # +0.1 to avoid zero mass
        
        body = GravitationalBody(
            body_id=body_id,
            mass=mass,
            position=position,
            domain=domain
        )
        
        self.bodies[body_id] = body
        return body
    
    def calculate_force(
        self,
        body1_id: str,
        body2_id: str
    ) -> GravitationalForce:
        """
        Calculate gravitational force between two bodies.
        
        Parameters
        ----------
        body1_id, body2_id : str
            IDs of bodies
            
        Returns
        -------
        GravitationalForce
        """
        body1 = self.bodies[body1_id]
        body2 = self.bodies[body2_id]
        
        # Distance in feature space
        distance = np.linalg.norm(body1.position - body2.position)
        
        if distance < 0.001:  # Prevent division by zero
            distance = 0.001
        
        # Similarity (cosine similarity, scaled 0-1)
        similarity = self._cosine_similarity(body1.position, body2.position)
        similarity = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Gravitational force
        # F = G × (m1 × m2 × similarity) / distance²
        force_magnitude = (
            self.G * body1.mass * body2.mass * similarity
        ) / (distance ** 2)
        
        # Force vector (direction from body1 to body2)
        direction = (body2.position - body1.position) / distance
        force_vector = force_magnitude * direction
        
        force = GravitationalForce(
            body1_id=body1_id,
            body2_id=body2_id,
            force_magnitude=force_magnitude,
            force_vector=force_vector,
            similarity=similarity,
            distance=distance
        )
        
        self.forces[(body1_id, body2_id)] = force
        return force
    
    def calculate_all_forces(self) -> Dict[Tuple[str, str], GravitationalForce]:
        """Calculate forces between all pairs of bodies."""
        body_ids = list(self.bodies.keys())
        
        for i, id1 in enumerate(body_ids):
            for id2 in body_ids[i+1:]:
                self.calculate_force(id1, id2)
        
        return self.forces
    
    def get_net_force(self, body_id: str) -> np.ndarray:
        """
        Calculate net force on a body from all others.
        
        Parameters
        ----------
        body_id : str
            ID of body
            
        Returns
        -------
        np.ndarray : Net force vector
        """
        if body_id not in self.bodies:
            raise ValueError(f"Body {body_id} not found")
        
        body = self.bodies[body_id]
        net_force = np.zeros_like(body.position)
        
        # Sum forces from all other bodies
        for other_id in self.bodies:
            if other_id == body_id:
                continue
            
            # Get or calculate force
            force_key = tuple(sorted([body_id, other_id]))
            if force_key not in self.forces:
                self.calculate_force(body_id, other_id)
            
            # Get force (need to check direction)
            if force_key == (body_id, other_id):
                force = self.forces[force_key]
                net_force += force.force_vector
            else:
                force = self.forces[force_key]
                net_force -= force.force_vector  # Reverse direction
        
        return net_force
    
    def find_strongest_attractions(self, top_n: int = 10) -> List[GravitationalForce]:
        """
        Find pairs with strongest gravitational attraction.
        
        Parameters
        ----------
        top_n : int
            Number of top pairs to return
            
        Returns
        -------
        List of GravitationalForce sorted by magnitude
        """
        if not self.forces:
            self.calculate_all_forces()
        
        sorted_forces = sorted(
            self.forces.values(),
            key=lambda f: f.force_magnitude,
            reverse=True
        )
        
        return sorted_forces[:top_n]
    
    def find_gravitational_center(self) -> Tuple[str, GravitationalBody]:
        """
        Find the gravitational center (body with highest total force).
        
        Returns
        -------
        Tuple of (body_id, body)
        """
        max_total_force = 0.0
        center_id = None
        
        for body_id in self.bodies:
            net_force = self.get_net_force(body_id)
            total_force = np.linalg.norm(net_force)
            
            if total_force > max_total_force:
                max_total_force = total_force
                center_id = body_id
        
        if center_id is None:
            # Return highest mass body as fallback
            center_id = max(self.bodies.items(), key=lambda x: x[1].mass)[0]
        
        return center_id, self.bodies[center_id]
    
    def get_gravitational_potential_energy(self) -> float:
        """
        Calculate total gravitational potential energy of system.
        
        U = -G × Σ(m1 × m2 × similarity / distance)
        
        Returns
        -------
        float : Total potential energy (negative = bound system)
        """
        if not self.forces:
            self.calculate_all_forces()
        
        total_energy = 0.0
        
        for force in self.forces.values():
            body1 = self.bodies[force.body1_id]
            body2 = self.bodies[force.body2_id]
            
            # U = -G × m1 × m2 × similarity / distance
            energy = -(
                self.G * body1.mass * body2.mass * force.similarity / force.distance
            )
            
            total_energy += energy
        
        return total_energy
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def simulate_motion(
        self,
        n_steps: int = 100,
        dt: float = 0.01,
        damping: float = 0.9
    ) -> Dict[str, List[np.ndarray]]:
        """
        Simulate gravitational motion (positions evolve under forces).
        
        Parameters
        ----------
        n_steps : int
            Number of simulation steps
        dt : float
            Time step
        damping : float
            Velocity damping factor (0-1)
            
        Returns
        -------
        Dict mapping body_id to list of positions over time
        """
        # Initialize velocities if not set
        for body in self.bodies.values():
            if body.velocity is None:
                body.velocity = np.zeros_like(body.position)
        
        # Track trajectories
        trajectories = {body_id: [body.position.copy()] for body_id in self.bodies}
        
        for step in range(n_steps):
            # Calculate all forces
            self.calculate_all_forces()
            
            # Update velocities and positions
            for body_id, body in self.bodies.items():
                # Net force
                F_net = self.get_net_force(body_id)
                
                # Acceleration (F = ma, so a = F/m)
                acceleration = F_net / body.mass
                
                # Update velocity
                body.velocity += acceleration * dt
                body.velocity *= damping  # Damping
                
                # Update position
                body.position += body.velocity * dt
                
                # Store trajectory
                trajectories[body_id].append(body.position.copy())
        
        return trajectories
    
    def generate_report(self) -> str:
        """Generate gravitational analysis report."""
        report = []
        report.append("=" * 80)
        report.append("GRAVITATIONAL ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        if not self.bodies:
            report.append("No bodies in system.")
            return "\n".join(report)
        
        # System statistics
        total_mass = sum(b.mass for b in self.bodies.values())
        mean_mass = np.mean([b.mass for b in self.bodies.values()])
        
        report.append("SYSTEM STATISTICS:")
        report.append(f"  Total bodies: {len(self.bodies)}")
        report.append(f"  Total mass: {total_mass:.2f}")
        report.append(f"  Mean mass: {mean_mass:.2f}")
        report.append(f"  Gravitational constant: {self.G}")
        report.append("")
        
        # Mass distribution
        mass_classes = {'SUPERMASSIVE': 0, 'MASSIVE': 0, 'STANDARD': 0, 'DWARF': 0, 'ASTEROID': 0}
        for body in self.bodies.values():
            mass_classes[body._classify_mass()] += 1
        
        report.append("MASS DISTRIBUTION:")
        for class_name, count in mass_classes.items():
            if count > 0:
                report.append(f"  {class_name}: {count}")
        report.append("")
        
        # Strongest attractions
        top_forces = self.find_strongest_attractions(5)
        
        report.append("STRONGEST GRAVITATIONAL ATTRACTIONS:")
        for i, force in enumerate(top_forces, 1):
            report.append(f"  {i}. {force.body1_id} ← → {force.body2_id}")
            report.append(f"     Force: {force.force_magnitude:.4f}")
            report.append(f"     Similarity: {force.similarity:.3f}")
            report.append(f"     Distance: {force.distance:.3f}")
            report.append("")
        
        # Gravitational center
        center_id, center_body = self.find_gravitational_center()
        report.append("GRAVITATIONAL CENTER:")
        report.append(f"  {center_id} ({center_body.domain})")
        report.append(f"  Mass: {center_body.mass:.2f}")
        report.append("")
        
        # System energy
        if self.forces:
            energy = self.get_gravitational_potential_energy()
            report.append("SYSTEM ENERGY:")
            report.append(f"  Potential energy: {energy:.2f}")
            if energy < 0:
                report.append("  Status: BOUND SYSTEM (comparisons gravitate together)")
            else:
                report.append("  Status: UNBOUND SYSTEM (comparisons dispersing)")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

