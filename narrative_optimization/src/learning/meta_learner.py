"""
Meta-Learning System

Learns across domains to enable transfer learning and few-shot adaptation.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class MetaLearner:
    """
    Meta-learning for cross-domain pattern transfer.
    
    Capabilities:
    - Detect domain similarity
    - Transfer patterns between similar domains
    - Few-shot learning for new domains
    - Zero-shot prediction using universal patterns
    """
    
    def __init__(self):
        self.domain_embeddings = {}
        self.domain_patterns = {}
        self.transfer_history = []
        
    def compute_domain_similarity(
        self,
        domain1: str,
        domain2: str,
        pattern_overlap: bool = True,
        performance_similarity: bool = True
    ) -> float:
        """
        Compute similarity between two domains.
        
        Parameters
        ----------
        domain1, domain2 : str
            Domain names
        pattern_overlap : bool
            Consider pattern overlap
        performance_similarity : bool
            Consider performance similarity
        
        Returns
        -------
        float
            Similarity score (0-1)
        """
        similarities = []
        
        # Pattern overlap similarity
        if pattern_overlap and domain1 in self.domain_patterns and domain2 in self.domain_patterns:
            patterns1 = set(self.domain_patterns[domain1].keys())
            patterns2 = set(self.domain_patterns[domain2].keys())
            
            if len(patterns1) > 0 and len(patterns2) > 0:
                overlap = len(patterns1 & patterns2)
                union = len(patterns1 | patterns2)
                jaccard = overlap / union if union > 0 else 0.0
                similarities.append(jaccard)
        
        # Embedding similarity
        if domain1 in self.domain_embeddings and domain2 in self.domain_embeddings:
            emb1 = self.domain_embeddings[domain1]
            emb2 = self.domain_embeddings[domain2]
            
            cos_sim = cosine_similarity([emb1], [emb2])[0, 0]
            similarities.append((cos_sim + 1) / 2)  # Scale to [0, 1]
        
        return np.mean(similarities) if similarities else 0.0
    
    def find_similar_domains(
        self,
        target_domain: str,
        n_similar: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find domains most similar to target.
        
        Parameters
        ----------
        target_domain : str
            Target domain
        n_similar : int
            Number of similar domains to return
        
        Returns
        -------
        list of (domain, similarity)
            Similar domains
        """
        similarities = []
        
        for domain in self.domain_patterns.keys():
            if domain != target_domain:
                sim = self.compute_domain_similarity(target_domain, domain)
                similarities.append((domain, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def transfer_patterns(
        self,
        source_domain: str,
        target_domain: str,
        min_transferability: float = 0.6,
        max_patterns: int = 10
    ) -> Dict[str, Dict]:
        """
        Transfer patterns from source to target domain.
        
        Parameters
        ----------
        source_domain : str
            Source domain
        target_domain : str
            Target domain
        min_transferability : float
            Minimum similarity for transfer
        max_patterns : int
            Maximum patterns to transfer
        
        Returns
        -------
        dict
            Transferred patterns
        """
        if source_domain not in self.domain_patterns:
            return {}
        
        # Check domain similarity
        similarity = self.compute_domain_similarity(source_domain, target_domain)
        
        if similarity < min_transferability:
            return {}
        
        # Transfer patterns
        source_patterns = self.domain_patterns[source_domain]
        transferred = {}
        
        for pattern_name, pattern_data in list(source_patterns.items())[:max_patterns]:
            # Adapt pattern name for target
            target_pattern_name = pattern_name.replace(source_domain, target_domain)
            
            # Scale quality by similarity
            adapted_data = pattern_data.copy()
            if 'quality' in adapted_data:
                adapted_data['quality'] *= similarity
            adapted_data['transferred_from'] = source_domain
            adapted_data['transfer_confidence'] = similarity
            
            transferred[target_pattern_name] = adapted_data
        
        # Record transfer
        self.transfer_history.append({
            'source': source_domain,
            'target': target_domain,
            'similarity': similarity,
            'n_patterns': len(transferred)
        })
        
        return transferred
    
    def few_shot_adaptation(
        self,
        new_domain: str,
        few_shot_texts: List[str],
        few_shot_outcomes: np.ndarray,
        n_shots: int = 5
    ) -> Dict[str, Dict]:
        """
        Adapt to new domain with few examples.
        
        Parameters
        ----------
        new_domain : str
            New domain name
        few_shot_texts : list
            Few example texts
        few_shot_outcomes : ndarray
            Few example outcomes
        n_shots : int
            Number of shots to use
        
        Returns
        -------
        dict
            Adapted patterns
        """
        # Use only first n_shots
        texts = few_shot_texts[:n_shots]
        outcomes = few_shot_outcomes[:n_shots]
        
        # Find most similar existing domain
        similar_domains = self.find_similar_domains(new_domain, n_similar=1)
        
        if len(similar_domains) == 0:
            # No similar domains - use universal patterns only
            return {}
        
        source_domain, similarity = similar_domains[0]
        
        # Transfer patterns from similar domain
        transferred = self.transfer_patterns(
            source_domain,
            new_domain,
            min_transferability=0.5
        )
        
        # Fine-tune on few shots
        # (Simplified - would need actual model updating)
        for pattern_name, pattern_data in transferred.items():
            # Validate on few shots
            keywords = pattern_data.get('patterns', [])
            matches = [i for i, text in enumerate(texts) if any(k in text.lower() for k in keywords)]
            
            if len(matches) > 0:
                match_outcomes = outcomes[matches]
                pattern_data['few_shot_performance'] = np.mean(match_outcomes)
        
        return transferred
    
    def zero_shot_prediction(
        self,
        new_domain: str,
        text: str,
        universal_patterns: Dict[str, Dict]
    ) -> float:
        """
        Make prediction for new domain without training data.
        
        Uses only universal patterns.
        
        Parameters
        ----------
        new_domain : str
            New domain
        text : str
            Text to predict
        universal_patterns : dict
            Universal patterns
        
        Returns
        -------
        float
            Prediction
        """
        # Score text on universal patterns
        scores = []
        
        for pattern_name, pattern_data in universal_patterns.items():
            keywords = pattern_data.get('keywords', [])
            
            # Check presence
            present = any(keyword.lower() in text.lower() for keyword in keywords)
            
            if present:
                # Use pattern's win rate as signal
                win_rate = pattern_data.get('win_rate', 0.5)
                scores.append(win_rate)
        
        # Average if any matches, else 0.5
        return np.mean(scores) if scores else 0.5
    
    def create_domain_embedding(
        self,
        domain: str,
        patterns: Dict[str, Dict],
        performance_metrics: Dict[str, float]
    ) -> np.ndarray:
        """
        Create embedding vector for a domain.
        
        Parameters
        ----------
        domain : str
            Domain name
        patterns : dict
            Domain patterns
        performance_metrics : dict
            Performance metrics
        
        Returns
        -------
        ndarray
            Domain embedding
        """
        features = []
        
        # Pattern statistics
        features.append(len(patterns))  # Number of patterns
        
        if len(patterns) > 0:
            frequencies = [p.get('frequency', 0.0) for p in patterns.values()]
            features.append(np.mean(frequencies))
            features.append(np.std(frequencies))
            
            coherences = [p.get('coherence', 0.0) for p in patterns.values()]
            features.append(np.mean(coherences))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Performance metrics
        features.append(performance_metrics.get('r_squared', 0.0))
        features.append(performance_metrics.get('delta', 0.0))
        
        embedding = np.array(features)
        self.domain_embeddings[domain] = embedding
        
        return embedding
    
    def cluster_domains(self, n_clusters: int = 3) -> Dict[int, List[str]]:
        """
        Cluster domains by similarity.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        
        Returns
        -------
        dict
            Cluster ID -> list of domains
        """
        if len(self.domain_embeddings) < n_clusters:
            return {0: list(self.domain_embeddings.keys())}
        
        from sklearn.cluster import KMeans
        
        domain_names = list(self.domain_embeddings.keys())
        embeddings = np.array([self.domain_embeddings[d] for d in domain_names])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        for domain, label in zip(domain_names, labels):
            clusters[int(label)].append(domain)
        
        return dict(clusters)
    
    def recommend_next_domain(
        self,
        existing_domains: List[str],
        candidate_domains: List[str]
    ) -> str:
        """
        Recommend next domain to add for maximum learning value.
        
        Parameters
        ----------
        existing_domains : list
            Already learned domains
        candidate_domains : list
            Candidates to consider
        
        Returns
        -------
        str
            Recommended domain
        """
        scores = {}
        
        for candidate in candidate_domains:
            if candidate in existing_domains:
                continue
            
            # Score = average dissimilarity to existing (want diversity)
            dissimilarities = []
            for existing in existing_domains:
                sim = self.compute_domain_similarity(candidate, existing)
                dissimilarities.append(1.0 - sim)
            
            scores[candidate] = np.mean(dissimilarities) if dissimilarities else 1.0
        
        if not scores:
            return candidate_domains[0] if candidate_domains else None
        
        return max(scores.items(), key=lambda x: x[1])[0]

