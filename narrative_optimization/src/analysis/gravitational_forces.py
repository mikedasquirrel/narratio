"""
Gravitational Forces Calculator

Calculates ф (narrative gravity) and ة (nominative gravity).

Implements:
- ф = (μ₁ × μ₂ × story_similarity) / distance(ж)²
- ة = (μ₁ × μ₂ × name_similarity) / distance(names)²
- ф_net = ф + ة (can be in tension)

These forces cluster organisms in narrative/nominative space.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity


class GravitationalCalculator:
    """
    Calculate gravitational forces between organisms.
    
    Implements biological metaphor: organisms attract each other
    based on narrative similarity (ф) and name similarity (ة).
    """
    
    def __init__(self):
        """Initialize calculator"""
        pass
    
    def calculate_all_forces(
        self,
        genomes: np.ndarray,
        names: List[str],
        masses: np.ndarray,
        story_quality: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all gravitational forces.
        
        Parameters
        ----------
        genomes : ndarray
            Genomes (ж) (n_organisms, n_features)
        names : list of str
            Organism names
        masses : ndarray
            Masses (μ) (n_organisms,)
        story_quality : ndarray, optional
            Story quality scores (ю) (if not provided, compute from ж)
            
        Returns
        -------
        forces : dict
            Contains ф, ة, ф_net matrices and distances
        """
        n = len(genomes)
        
        # Compute ю if not provided
        if story_quality is None:
            story_quality = np.mean(genomes, axis=1)  # Simple aggregate
        
        # === NARRATIVE GRAVITY (ф) ===
        
        # Story similarity (using ю scores)
        story_similarity = 1 - squareform(pdist(story_quality.reshape(-1, 1), metric='euclidean'))
        story_similarity = np.clip(story_similarity, 0, 1)
        
        # Narrative distance (in ж space)
        narrative_distances = squareform(pdist(genomes, metric='euclidean'))
        
        # Calculate ф
        phi_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                force = (masses[i] * masses[j] * story_similarity[i, j]) / (narrative_distances[i, j]**2 + 1e-6)
                phi_matrix[i, j] = force
                phi_matrix[j, i] = force
        
        # === NOMINATIVE GRAVITY (ة) ===
        
        # Name similarity (phonetic + semantic)
        name_similarities = self._compute_name_similarities(names)
        
        # Nominative distance (inverse of similarity)
        nominative_distances = 1 - name_similarities
        nominative_distances = np.clip(nominative_distances, 0.01, 2.0)  # Prevent div by zero
        
        # Calculate ة
        tah_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                force = (masses[i] * masses[j] * name_similarities[i, j]) / (nominative_distances[i, j]**2 + 1e-6)
                tah_matrix[i, j] = force
                tah_matrix[j, i] = force
        
        # === NET GRAVITY (ф_net) ===
        
        phi_net = phi_matrix + tah_matrix
        
        # Identify gravitational tensions (ф and ة pull different directions)
        tensions = self._identify_tensions(phi_matrix, tah_matrix, names)
        
        return {
            'ф': phi_matrix,
            'ة': tah_matrix,
            'ф_net': phi_net,
            'narrative_distances': narrative_distances,
            'nominative_distances': nominative_distances,
            'story_similarities': story_similarity,
            'name_similarities': name_similarities,
            'tensions': tensions
        }
    
    def _compute_name_similarities(self, names: List[str]) -> np.ndarray:
        """
        Compute name similarity matrix.
        
        Uses phonetic + semantic similarity.
        """
        n = len(names)
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarities[i, j] = 1.0
                else:
                    # Phonetic similarity (simple: shared characters)
                    name1 = names[i].lower()
                    name2 = names[j].lower()
                    
                    # Jaccard similarity on character bigrams
                    bigrams1 = set(name1[i:i+2] for i in range(len(name1)-1))
                    bigrams2 = set(name2[i:i+2] for i in range(len(name2)-1))
                    
                    if len(bigrams1 | bigrams2) > 0:
                        phonetic_sim = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
                    else:
                        phonetic_sim = 0.0
                    
                    # Length similarity
                    len_sim = 1 - abs(len(name1) - len(name2)) / max(len(name1), len(name2))
                    
                    # Combined
                    similarity = 0.7 * phonetic_sim + 0.3 * len_sim
                    
                    similarities[i, j] = similarity
                    similarities[j, i] = similarity
        
        return similarities
    
    def _identify_tensions(
        self,
        ф: np.ndarray,
        ة: np.ndarray,
        names: List[str]
    ) -> List[Dict]:
        """
        Identify pairs where ф and ة pull in different directions.
        
        Tension occurs when:
        - High narrative similarity (strong ф) but low name similarity (weak ة)
        - OR vice versa
        """
        tensions = []
        n = len(names)
        
        # Normalize forces for comparison
        ф_norm = (ф - ф.min()) / (ф.max() - ф.min() + 1e-8)
        ة_norm = (ة - ة.min()) / (ة.max() - ة.min() + 1e-8)
        
        for i in range(n):
            for j in range(i+1, n):
                # Check if forces differ significantly
                force_diff = abs(ф_norm[i, j] - ة_norm[i, j])
                
                if force_diff > 0.5:  # Significant tension
                    tension_type = 'narrative_dominant' if ф_norm[i, j] > ة_norm[i, j] else 'nominative_dominant'
                    
                    tensions.append({
                        'organism_i': i,
                        'organism_j': j,
                        'name_i': names[i],
                        'name_j': names[j],
                        'ф_force': float(ф[i, j]),
                        'ة_force': float(ة[i, j]),
                        'tension_magnitude': float(force_diff),
                        'tension_type': tension_type
                    })
        
        # Sort by tension magnitude
        tensions.sort(key=lambda x: x['tension_magnitude'], reverse=True)
        
        return tensions[:20]  # Top 20 tensions
    
    def find_clusters(
        self,
        ф_net: np.ndarray,
        names: List[str],
        method='hierarchical',
        n_clusters: int = 5
    ) -> Dict:
        """
        Find gravitational clusters (galaxies).
        
        Parameters
        ----------
        ф_net : ndarray
            Net gravitational forces
        names : list
            Organism names
        method : str
            Clustering method
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        clusters : dict
            Cluster assignments and statistics
        """
        from sklearn.cluster import AgglomerativeClustering
        
        # Use net gravity as affinity
        affinity = ф_net
        
        # Cluster
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(names)),
            metric='precomputed',
            linkage='average'
        )
        
        # Convert to distance matrix (inverse of affinity)
        max_force = ф_net.max()
        distance_matrix = max_force - ф_net
        distance_matrix = np.clip(distance_matrix, 0, None)
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Analyze clusters
        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_members = [names[i] for i in range(len(names)) if cluster_mask[i]]
            
            # Cluster statistics
            clusters[cluster_id] = {
                'members': cluster_members,
                'size': int(cluster_mask.sum()),
                'avg_internal_force': float(ф_net[np.ix_(cluster_mask, cluster_mask)].mean()),
                'cohesion': float(ф_net[np.ix_(cluster_mask, cluster_mask)].std())
            }
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'clusters': clusters
        }
    
    def compute_gravitational_potential(
        self,
        organism_idx: int,
        ф_net: np.ndarray
    ) -> float:
        """
        Compute total gravitational potential for an organism.
        
        Sum of all forces acting on it.
        """
        return ф_net[organism_idx, :].sum()

