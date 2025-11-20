"""
Pattern Refinement System

Refines discovered patterns through validation cycles:
- Prune weak patterns
- Merge similar patterns
- Split conflated patterns
- Optimize pattern definitions

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class PatternRefiner:
    """
    Refines patterns through iterative validation cycles.
    
    Operations:
    - Prune: Remove weak/redundant patterns
    - Merge: Consolidate similar patterns
    - Split: Separate conflated patterns
    - Optimize: Improve pattern definitions
    """
    
    def __init__(
        self,
        min_correlation: float = 0.05,
        min_frequency: float = 0.02,
        merge_threshold: float = 0.8,
        split_threshold: float = 0.3
    ):
        self.min_correlation = min_correlation
        self.min_frequency = min_frequency
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        
        self.refinement_history = []
        
    def refine_patterns(
        self,
        patterns: Dict[str, Dict],
        texts: List[str],
        outcomes: np.ndarray,
        iterations: int = 3
    ) -> Dict[str, Dict]:
        """
        Run complete refinement cycle.
        
        Parameters
        ----------
        patterns : dict
            Initial patterns
        texts : list
            Texts
        outcomes : ndarray
            Outcomes
        iterations : int
            Number of refinement iterations
        
        Returns
        -------
        dict
            Refined patterns
        """
        refined = patterns.copy()
        
        for iteration in range(iterations):
            print(f"  Refinement iteration {iteration + 1}/{iterations}")
            
            # 1. Prune weak patterns
            refined, n_pruned = self.prune_weak_patterns(refined, texts, outcomes)
            print(f"    Pruned: {n_pruned}")
            
            # 2. Merge similar patterns
            refined, n_merged = self.merge_similar_patterns(refined)
            print(f"    Merged: {n_merged}")
            
            # 3. Split conflated patterns
            refined, n_split = self.split_conflated_patterns(refined, texts, outcomes)
            print(f"    Split: {n_split}")
            
            # 4. Optimize definitions
            refined = self.optimize_pattern_definitions(refined, texts, outcomes)
            
            # Record
            self.refinement_history.append({
                'iteration': iteration + 1,
                'n_patterns': len(refined),
                'n_pruned': n_pruned,
                'n_merged': n_merged,
                'n_split': n_split
            })
        
        return refined
    
    def prune_weak_patterns(
        self,
        patterns: Dict[str, Dict],
        texts: List[str],
        outcomes: np.ndarray
    ) -> Tuple[Dict[str, Dict], int]:
        """
        Remove weak patterns based on correlation and frequency.
        
        Returns
        -------
        tuple
            (refined_patterns, n_pruned)
        """
        to_remove = []
        
        for pattern_name, pattern_data in patterns.items():
            # Check correlation
            correlation = pattern_data.get('correlation', 0.0)
            if abs(correlation) < self.min_correlation:
                to_remove.append(pattern_name)
                continue
            
            # Check frequency
            frequency = pattern_data.get('frequency', 0.0)
            if frequency < self.min_frequency:
                to_remove.append(pattern_name)
                continue
            
            # Check validation
            if 'validation' in pattern_data:
                if not pattern_data['validation'].get('validated', False):
                    to_remove.append(pattern_name)
        
        # Remove
        refined = {k: v for k, v in patterns.items() if k not in to_remove}
        
        return refined, len(to_remove)
    
    def merge_similar_patterns(
        self,
        patterns: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict], int]:
        """
        Merge patterns with high similarity.
        
        Returns
        -------
        tuple
            (merged_patterns, n_merged)
        """
        pattern_list = list(patterns.items())
        merged = patterns.copy()
        merged_count = 0
        
        for i in range(len(pattern_list)):
            for j in range(i + 1, len(pattern_list)):
                name_i, data_i = pattern_list[i]
                name_j, data_j = pattern_list[j]
                
                # Skip if already removed
                if name_i not in merged or name_j not in merged:
                    continue
                
                # Calculate similarity
                similarity = self._pattern_similarity(data_i, data_j)
                
                if similarity >= self.merge_threshold:
                    # Merge j into i
                    merged[name_i] = self._merge_pattern_data(data_i, data_j)
                    del merged[name_j]
                    merged_count += 1
        
        return merged, merged_count
    
    def _pattern_similarity(
        self,
        pattern1: Dict,
        pattern2: Dict
    ) -> float:
        """Calculate similarity between two patterns."""
        # Get keywords
        keywords1 = set(pattern1.get('keywords', pattern1.get('patterns', [])))
        keywords2 = set(pattern2.get('keywords', pattern2.get('patterns', [])))
        
        if len(keywords1) == 0 or len(keywords2) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_pattern_data(
        self,
        data1: Dict,
        data2: Dict
    ) -> Dict:
        """Merge two pattern data dicts."""
        merged = data1.copy()
        
        # Combine keywords
        keywords1 = set(data1.get('keywords', data1.get('patterns', [])))
        keywords2 = set(data2.get('keywords', data2.get('patterns', [])))
        merged['keywords'] = list(keywords1 | keywords2)
        merged['patterns'] = list(keywords1 | keywords2)
        
        # Average numeric fields
        for field in ['frequency', 'correlation', 'coherence']:
            if field in data1 and field in data2:
                merged[field] = (data1[field] + data2[field]) / 2
        
        return merged
    
    def split_conflated_patterns(
        self,
        patterns: Dict[str, Dict],
        texts: List[str],
        outcomes: np.ndarray
    ) -> Tuple[Dict[str, Dict], int]:
        """
        Split patterns that seem to conflate multiple concepts.
        
        Returns
        -------
        tuple
            (split_patterns, n_split)
        """
        split_patterns = patterns.copy()
        n_split = 0
        
        for pattern_name, pattern_data in list(patterns.items()):
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            
            if len(keywords) < 4:
                continue  # Too few to split
            
            # Find texts matching this pattern
            matching_texts = [
                text for text in texts
                if any(kw.lower() in text.lower() for kw in keywords)
            ]
            
            if len(matching_texts) < 20:
                continue  # Not enough data
            
            # Check if matching texts form coherent clusters
            coherence = self._check_coherence(matching_texts)
            
            if coherence < self.split_threshold:
                # Split pattern
                sub_patterns = self._split_pattern(pattern_name, keywords, matching_texts)
                
                if len(sub_patterns) > 1:
                    # Remove original
                    del split_patterns[pattern_name]
                    
                    # Add sub-patterns
                    for sub_name, sub_data in sub_patterns.items():
                        split_patterns[sub_name] = sub_data
                    
                    n_split += 1
        
        return split_patterns, n_split
    
    def _check_coherence(self, texts: List[str]) -> float:
        """Check if texts form coherent cluster."""
        if len(texts) < 2:
            return 1.0
        
        try:
            # Create embeddings
            vectorizer = TfidfVectorizer(max_features=50)
            embeddings = vectorizer.fit_transform(texts).toarray()
            
            # Average pairwise similarity
            similarities = cosine_similarity(embeddings)
            
            # Exclude diagonal
            mask = ~np.eye(len(similarities), dtype=bool)
            avg_similarity = similarities[mask].mean()
            
            return avg_similarity
        except:
            return 0.5
    
    def _split_pattern(
        self,
        pattern_name: str,
        keywords: List[str],
        texts: List[str]
    ) -> Dict[str, Dict]:
        """Split pattern into sub-patterns."""
        from sklearn.cluster import KMeans
        
        try:
            # Cluster texts
            vectorizer = TfidfVectorizer(max_features=50)
            embeddings = vectorizer.fit_transform(texts).toarray()
            
            # Try 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Create sub-patterns
            sub_patterns = {}
            
            for cluster_id in range(2):
                cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
                
                # Extract keywords for this cluster
                cluster_keywords = self._extract_cluster_keywords(cluster_texts)
                
                sub_name = f"{pattern_name}_sub{cluster_id + 1}"
                sub_patterns[sub_name] = {
                    'patterns': cluster_keywords,
                    'keywords': cluster_keywords,
                    'frequency': len(cluster_texts) / len(texts),
                    'split_from': pattern_name
                }
            
            return sub_patterns
        except:
            return {}
    
    def _extract_cluster_keywords(self, texts: List[str], n_keywords: int = 5) -> List[str]:
        """Extract representative keywords from cluster."""
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Filter short words and stopwords
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        meaningful = [w for w in all_words if len(w) >= 4 and w not in stop_words]
        
        word_counts = Counter(meaningful)
        return [word for word, count in word_counts.most_common(n_keywords)]
    
    def optimize_pattern_definitions(
        self,
        patterns: Dict[str, Dict],
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Optimize pattern keyword definitions.
        
        For each pattern, find best keywords that maximize correlation.
        """
        optimized = {}
        
        for pattern_name, pattern_data in patterns.items():
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            
            if len(keywords) == 0:
                optimized[pattern_name] = pattern_data
                continue
            
            # Test removing each keyword
            best_keywords = keywords.copy()
            best_correlation = pattern_data.get('correlation', 0.0)
            
            for kw in keywords:
                # Try without this keyword
                test_keywords = [k for k in keywords if k != kw]
                
                if len(test_keywords) == 0:
                    continue
                
                # Evaluate
                matches = np.array([
                    any(k.lower() in text.lower() for k in test_keywords)
                    for text in texts
                ])
                
                if matches.sum() > 0:
                    correlation = abs(np.corrcoef(matches.astype(float), outcomes)[0, 1])
                    
                    if correlation > best_correlation:
                        best_keywords = test_keywords
                        best_correlation = correlation
            
            # Update
            optimized_data = pattern_data.copy()
            optimized_data['keywords'] = best_keywords
            optimized_data['patterns'] = best_keywords
            optimized_data['correlation'] = best_correlation
            
            optimized[pattern_name] = optimized_data
        
        return optimized
    
    def get_refinement_stats(self) -> Dict:
        """Get refinement statistics."""
        if len(self.refinement_history) == 0:
            return {}
        
        total_pruned = sum(h['n_pruned'] for h in self.refinement_history)
        total_merged = sum(h['n_merged'] for h in self.refinement_history)
        total_split = sum(h['n_split'] for h in self.refinement_history)
        
        return {
            'iterations': len(self.refinement_history),
            'total_pruned': total_pruned,
            'total_merged': total_merged,
            'total_split': total_split,
            'final_patterns': self.refinement_history[-1]['n_patterns']
        }

