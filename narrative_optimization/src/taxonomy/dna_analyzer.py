"""
DNA Overlap Matrix Calculator

Calculates pairwise similarity between comparison "DNA" (feature vectors)
and generates visualizations of genetic relationships.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings


class DNAOverlapAnalyzer:
    """
    Analyzes DNA overlap (feature vector similarity) between comparisons.
    
    Generates:
    - Pairwise similarity matrices
    - Genetic distance matrices
    - Heatmap visualizations
    - Cluster identification
    """
    
    def __init__(self):
        self.comparison_ids: List[str] = []
        self.dna_vectors: List[np.ndarray] = []
        self.similarity_matrix: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
    
    def add_comparison(self, comparison_id: str, dna: np.ndarray):
        """Add a comparison's DNA to the analysis."""
        self.comparison_ids.append(comparison_id)
        self.dna_vectors.append(dna)
    
    def compute_similarity_matrix(self, metric: str = 'cosine') -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        
        Parameters
        ----------
        metric : str
            Similarity metric: 'cosine', 'correlation', 'euclidean'
            
        Returns
        -------
        np.ndarray : n x n similarity matrix
        """
        if not self.dna_vectors:
            return np.array([[]])
        
        dna_matrix = np.array(self.dna_vectors)
        
        # Compute distances
        if metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            distances = pdist(dna_matrix, metric='cosine')
            similarities = 1 - squareform(distances)
        elif metric == 'correlation':
            distances = pdist(dna_matrix, metric='correlation')
            similarities = 1 - squareform(distances)
        elif metric == 'euclidean':
            # Convert Euclidean distance to similarity
            distances = pdist(dna_matrix, metric='euclidean')
            distance_matrix = squareform(distances)
            # Normalize to 0-1, where 1 is identical
            max_dist = np.max(distance_matrix)
            similarities = 1 - (distance_matrix / (max_dist + 1e-10))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Set diagonal to 1 (perfect self-similarity)
        np.fill_diagonal(similarities, 1.0)
        
        self.similarity_matrix = similarities
        return similarities
    
    def compute_distance_matrix(self, metric: str = 'euclidean') -> np.ndarray:
        """
        Compute genetic distance matrix.
        
        Returns
        -------
        np.ndarray : n x n distance matrix
        """
        if not self.dna_vectors:
            return np.array([[]])
        
        dna_matrix = np.array(self.dna_vectors)
        distances = pdist(dna_matrix, metric=metric)
        self.distance_matrix = squareform(distances)
        
        return self.distance_matrix
    
    def find_genetic_clusters(
        self,
        similarity_threshold: float = 0.7
    ) -> Dict[int, List[str]]:
        """
        Find clusters of genetically similar comparisons.
        
        Parameters
        ----------
        similarity_threshold : float
            Minimum similarity to be in same cluster
            
        Returns
        -------
        Dict mapping cluster_id to list of comparison_ids
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        n = len(self.comparison_ids)
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Start new cluster
            cluster = [i]
            assigned.add(i)
            
            # Find similar comparisons
            for j in range(i+1, n):
                if j in assigned:
                    continue
                
                if self.similarity_matrix[i, j] >= similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            # Store cluster
            cluster_names = [self.comparison_ids[idx] for idx in cluster]
            clusters[cluster_id] = cluster_names
            cluster_id += 1
        
        return clusters
    
    def identify_hybrid_vigor(
        self,
        domain_labels: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Identify hybrid comparisons (cross-domain DNA).
        
        Hybrid vigor: comparisons with DNA from multiple domains
        may have advantages.
        
        Parameters
        ----------
        domain_labels : List[str]
            Domain label for each comparison
            
        Returns
        -------
        List of (comparison_id, domains, diversity_score) tuples
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        hybrids = []
        
        for i, comp_id in enumerate(self.comparison_ids):
            # Find most similar comparisons
            similarities = self.similarity_matrix[i]
            top_similar_indices = np.argsort(similarities)[-6:-1]  # Top 5 (excluding self)
            
            # Get their domains
            similar_domains = [domain_labels[idx] for idx in top_similar_indices]
            unique_domains = set(similar_domains)
            
            # If similar to multiple domains, it's a hybrid
            if len(unique_domains) >= 3:
                diversity = len(unique_domains) / len(similar_domains)
                hybrids.append((comp_id, list(unique_domains), diversity))
        
        return sorted(hybrids, key=lambda x: x[2], reverse=True)
    
    def generate_heatmap_data(self) -> Dict[str, Any]:
        """
        Generate data for heatmap visualization.
        
        Returns
        -------
        Dict with heatmap data and metadata
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        return {
            'matrix': self.similarity_matrix.tolist(),
            'labels': self.comparison_ids,
            'n_comparisons': len(self.comparison_ids),
            'mean_similarity': float(np.mean(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])),
            'max_similarity': float(np.max(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])),
            'min_similarity': float(np.min(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]))
        }

