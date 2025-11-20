"""
Cross-Domain Learning

Learns patterns that transfer across multiple domains and identifies
domain-agnostic narrative structures.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


class CrossDomainLearner:
    """
    Learns patterns across multiple domains simultaneously.
    
    Discovers:
    - Shared narrative structures
    - Domain-invariant features
    - Transfer coefficients
    - Meta-patterns
    """
    
    def __init__(self):
        self.domain_data = {}
        self.shared_patterns = {}
        self.transfer_matrix = None
        
    def ingest_domains(
        self,
        domain_data: Dict[str, Dict]
    ):
        """
        Ingest data from multiple domains.
        
        Parameters
        ----------
        domain_data : dict
            domain_name -> {'texts': [...], 'outcomes': [...]}
        """
        self.domain_data = domain_data
        
    def discover_shared_patterns(
        self,
        min_domains: int = 2,
        min_frequency: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Discover patterns that appear in multiple domains.
        
        Parameters
        ----------
        min_domains : int
            Minimum number of domains pattern must appear in
        min_frequency : float
            Minimum frequency within each domain
        
        Returns
        -------
        dict
            Shared patterns
        """
        # Extract patterns from each domain
        domain_patterns = {}
        
        for domain_name, data in self.domain_data.items():
            texts = data['texts']
            
            # Simple n-gram extraction
            all_words = []
            for text in texts:
                words = str(text).lower().split()
                all_words.extend(words)
            
            # Count n-grams
            word_counts = Counter(all_words)
            total_words = len(all_words)
            
            # Filter by frequency
            domain_patterns[domain_name] = {
                word: count / total_words
                for word, count in word_counts.items()
                if count / total_words >= min_frequency and len(word) > 3
            }
        
        # Find shared patterns
        shared = {}
        
        # Get all unique words
        all_words = set()
        for patterns in domain_patterns.values():
            all_words.update(patterns.keys())
        
        # Check which appear in multiple domains
        for word in all_words:
            appearing_in = [
                domain for domain, patterns in domain_patterns.items()
                if word in patterns
            ]
            
            if len(appearing_in) >= min_domains:
                shared[word] = {
                    'domains': appearing_in,
                    'n_domains': len(appearing_in),
                    'avg_frequency': np.mean([
                        domain_patterns[domain][word]
                        for domain in appearing_in
                    ])
                }
        
        self.shared_patterns = shared
        return shared
    
    def compute_transfer_matrix(self) -> np.ndarray:
        """
        Compute transfer matrix showing pattern flow between domains.
        
        Returns
        -------
        ndarray
            Transfer matrix (domain x domain)
        """
        domain_names = list(self.domain_data.keys())
        n_domains = len(domain_names)
        
        transfer = np.zeros((n_domains, n_domains))
        
        # For each pair of domains
        for i, source in enumerate(domain_names):
            for j, target in enumerate(domain_names):
                if i == j:
                    continue
                
                # Calculate pattern overlap
                source_texts = self.domain_data[source]['texts']
                target_texts = self.domain_data[target]['texts']
                
                # Simple: word overlap
                source_words = set()
                for text in source_texts:
                    source_words.update(str(text).lower().split())
                
                target_words = set()
                for text in target_texts:
                    target_words.update(str(text).lower().split())
                
                if len(source_words) > 0 and len(target_words) > 0:
                    overlap = len(source_words & target_words)
                    union = len(source_words | target_words)
                    transfer[i, j] = overlap / union if union > 0 else 0
        
        self.transfer_matrix = transfer
        return transfer
    
    def identify_domain_clusters(
        self,
        n_clusters: int = 3
    ) -> Dict[int, List[str]]:
        """
        Cluster domains by pattern similarity.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        
        Returns
        -------
        dict
            Cluster assignments
        """
        from sklearn.cluster import AgglomerativeClustering
        
        if self.transfer_matrix is None:
            self.compute_transfer_matrix()
        
        # Convert to distance matrix
        distance = 1 - self.transfer_matrix
        
        # Cluster
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(self.domain_data)),
            affinity='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(distance)
        
        # Group by cluster
        domain_names = list(self.domain_data.keys())
        clusters = defaultdict(list)
        
        for domain, label in zip(domain_names, labels):
            clusters[int(label)].append(domain)
        
        return dict(clusters)
    
    def extract_domain_invariant_features(
        self,
        n_features: int = 50
    ) -> np.ndarray:
        """
        Extract features that are consistent across domains.
        
        Uses non-negative matrix factorization to find common bases.
        
        Parameters
        ----------
        n_features : int
            Number of invariant features
        
        Returns
        -------
        ndarray
            Invariant feature matrix
        """
        # Combine all texts
        all_texts = []
        domain_indices = []
        
        for domain_name, data in self.domain_data.items():
            texts = data['texts']
            all_texts.extend(texts)
            domain_indices.extend([domain_name] * len(texts))
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=n_features * 3)
        X = vectorizer.fit_transform(all_texts)
        
        # NMF to find shared components
        nmf = NMF(n_components=n_features, random_state=42)
        W = nmf.fit_transform(X)
        H = nmf.components_
        
        # H contains the invariant features
        return H
    
    def predict_domain_from_text(self, text: str) -> Tuple[str, float]:
        """
        Predict which domain a text belongs to.
        
        Parameters
        ----------
        text : str
            Text to classify
        
        Returns
        -------
        tuple
            (predicted_domain, confidence)
        """
        text_words = set(str(text).lower().split())
        
        scores = {}
        
        for domain_name, data in self.domain_data.items():
            # Calculate word overlap with domain
            domain_words = set()
            for domain_text in data['texts']:
                domain_words.update(str(domain_text).lower().split())
            
            if len(domain_words) > 0:
                overlap = len(text_words & domain_words)
                scores[domain_name] = overlap / len(text_words) if len(text_words) > 0 else 0
        
        if not scores:
            return 'unknown', 0.0
        
        best_domain = max(scores.items(), key=lambda x: x[1])
        return best_domain[0], best_domain[1]
    
    def analyze_cross_domain_effects(self) -> Dict[str, Dict]:
        """
        Analyze how patterns transfer between domains.
        
        Returns
        -------
        dict
            Cross-domain effect analysis
        """
        effects = {}
        
        # For each shared pattern
        for pattern, pattern_data in self.shared_patterns.items():
            domains_with_pattern = pattern_data['domains']
            
            # Check correlation in each domain
            domain_correlations = {}
            
            for domain in domains_with_pattern:
                data = self.domain_data[domain]
                texts = data['texts']
                outcomes = data['outcomes']
                
                # Find texts with pattern
                has_pattern = np.array([
                    pattern in str(text).lower()
                    for text in texts
                ])
                
                if np.sum(has_pattern) > 5 and len(np.unique(outcomes)) > 1:
                    # Calculate correlation
                    corr = np.corrcoef(has_pattern.astype(float), outcomes)[0, 1]
                    domain_correlations[domain] = corr
            
            if len(domain_correlations) > 0:
                effects[pattern] = {
                    'domain_correlations': domain_correlations,
                    'avg_effect': np.mean(list(domain_correlations.values())),
                    'consistency': np.std(list(domain_correlations.values())),
                    'n_domains': len(domain_correlations)
                }
        
        return effects
    
    def recommend_next_domain(
        self,
        candidate_domains: List[str]
    ) -> Tuple[str, float]:
        """
        Recommend next domain to analyze for maximum learning.
        
        Parameters
        ----------
        candidate_domains : list
            Candidate domains
        
        Returns
        -------
        tuple
            (recommended_domain, expected_value)
        """
        if not candidate_domains:
            return None, 0.0
        
        # Score each candidate by diversity (dissimilar to existing)
        scores = {}
        
        existing_domains = list(self.domain_data.keys())
        
        for candidate in candidate_domains:
            # Higher score if different from existing
            # (Simplified - would need actual similarity computation)
            scores[candidate] = 1.0
        
        best = max(scores.items(), key=lambda x: x[1])
        return best

