"""
Advanced Archetype Discovery System

Enhanced discovery capabilities:
- Semantic pattern discovery (beyond n-grams)
- Sub-archetype hierarchies
- Temporal evolution tracking
- Cross-domain pattern transfer

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import networkx as nx


class SemanticArchetypeDiscovery:
    """
    Advanced semantic pattern discovery using embeddings.
    
    Goes beyond n-grams to discover abstract patterns:
    - Semantic similarity clustering
    - Abstract concept extraction
    - Pattern relationships
    """
    
    def __init__(self, use_embeddings: bool = True):
        """
        Parameters
        ----------
        use_embeddings : bool
            Use word embeddings for semantic similarity (if False, uses TF-IDF)
        """
        self.use_embeddings = use_embeddings
        self.vectorizer = None
        
    def discover_semantic_patterns(
        self,
        texts: List[str],
        n_patterns: int = 10,
        min_similarity: float = 0.6
    ) -> Dict[str, List[str]]:
        """
        Discover semantic patterns using clustering.
        
        Parameters
        ----------
        texts : list of str
            Texts to analyze
        n_patterns : int
            Number of patterns to discover
        min_similarity : float
            Minimum similarity for pattern membership
        
        Returns
        -------
        dict
            Pattern clusters with representative texts
        """
        # Create embeddings
        if self.use_embeddings:
            try:
                # Try to use sentence transformers if available
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode(texts)
            except ImportError:
                # Fallback to TF-IDF
                self.vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 3))
                embeddings = self.vectorizer.fit_transform(texts).toarray()
        else:
            self.vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 3))
            embeddings = self.vectorizer.fit_transform(texts).toarray()
        
        # Cluster semantically similar texts
        if len(texts) < n_patterns:
            n_patterns = max(2, len(texts) // 2)
        
        kmeans = KMeans(n_clusters=n_patterns, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Extract patterns from each cluster
        patterns = {}
        for cluster_id in range(n_patterns):
            cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
            
            if len(cluster_texts) == 0:
                continue
            
            # Find common semantic elements
            pattern_keywords = self._extract_semantic_keywords(cluster_texts)
            
            patterns[f'semantic_pattern_{cluster_id + 1}'] = {
                'keywords': pattern_keywords,
                'sample_texts': cluster_texts[:3],  # Representative samples
                'cluster_size': len(cluster_texts),
                'centroid': kmeans.cluster_centers_[cluster_id].tolist()
            }
        
        return patterns
    
    def _extract_semantic_keywords(self, texts: List[str]) -> List[str]:
        """Extract keywords that represent semantic pattern."""
        # Simple approach: most frequent meaningful words
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            all_words.extend(words)
        
        # Filter common stop words
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'their', 'there'}
        meaningful = [w for w in all_words if w not in stop_words]
        
        word_counts = Counter(meaningful)
        return [word for word, count in word_counts.most_common(10)]


class HierarchicalArchetypeDiscovery:
    """
    Discovers multi-level archetype hierarchies.
    
    Builds parent → child → grandchild relationships:
    - mental_game → pressure_performance → championship_pressure
    - elite_skill → technical_mastery → course_specific_skill
    """
    
    def __init__(self):
        self.hierarchy = nx.DiGraph()
        
    def build_hierarchy(
        self,
        parent_patterns: List[str],
        texts: List[str],
        max_depth: int = 3,
        min_samples_per_level: int = 10
    ) -> Dict[str, Dict]:
        """
        Build hierarchical archetype structure.
        
        Parameters
        ----------
        parent_patterns : list of str
            Patterns defining parent archetype
        texts : list of str
            Texts containing parent patterns
        max_depth : int
            Maximum hierarchy depth
        min_samples_per_level : int
            Minimum samples needed to create sub-level
        
        Returns
        -------
        dict
            Hierarchical structure
        """
        hierarchy = {}
        
        def build_level(parent_name: str, patterns: List[str], level: int, parent_texts: List[str]):
            if level > max_depth or len(parent_texts) < min_samples_per_level:
                return
            
            # Filter texts containing parent patterns
            matching_texts = [
                text for text in parent_texts
                if any(pattern.lower() in text.lower() for pattern in patterns)
            ]
            
            if len(matching_texts) < min_samples_per_level:
                return
            
            # Discover sub-patterns within parent
            discovery = SemanticArchetypeDiscovery(use_embeddings=False)
            sub_patterns = discovery.discover_semantic_patterns(
                matching_texts, n_patterns=3
            )
            
            # Create sub-archetypes
            children = {}
            for sub_name, sub_data in sub_patterns.items():
                child_name = f"{parent_name}::{sub_name}"
                children[child_name] = {
                    'patterns': sub_data['keywords'],
                    'level': level,
                    'parent': parent_name,
                    'sample_count': sub_data['cluster_size']
                }
                
                # Recursively build deeper levels
                build_level(
                    child_name,
                    sub_data['keywords'],
                    level + 1,
                    sub_data['sample_texts']
                )
            
            hierarchy[parent_name] = {
                'patterns': patterns,
                'level': level - 1,
                'children': children,
                'sample_count': len(matching_texts)
            }
        
        # Start with root
        root_name = 'root_archetype'
        build_level(root_name, parent_patterns, 1, texts)
        
        return hierarchy


class TemporalArchetypeEvolution:
    """
    Tracks how archetypes evolve over time.
    
    Discovers:
    - Archetype emergence (when did pattern first appear?)
    - Archetype drift (how has pattern changed?)
    - Archetype disappearance (when did pattern fade?)
    """
    
    def __init__(self):
        self.temporal_patterns = {}
        
    def track_evolution(
        self,
        texts: List[str],
        timestamps: np.ndarray,
        outcomes: np.ndarray,
        window_size: int = 100
    ) -> Dict[str, Dict]:
        """
        Track archetype evolution over time.
        
        Parameters
        ----------
        texts : list of str
            Historical texts
        timestamps : ndarray
            Timestamps for each text
        outcomes : ndarray
            Outcomes
        window_size : int
            Number of texts per time window
        
        Returns
        -------
        dict
            Evolution data for each archetype
        """
        # Sort by time
        sorted_indices = np.argsort(timestamps)
        texts_sorted = [texts[i] for i in sorted_indices]
        outcomes_sorted = outcomes[sorted_indices]
        timestamps_sorted = timestamps[sorted_indices]
        
        # Create time windows
        n_windows = len(texts) // window_size
        evolution_data = {}
        
        discovery = SemanticArchetypeDiscovery(use_embeddings=False)
        
        for window_id in range(n_windows):
            start_idx = window_id * window_size
            end_idx = min((window_id + 1) * window_size, len(texts))
            
            window_texts = texts_sorted[start_idx:end_idx]
            window_times = timestamps_sorted[start_idx:end_idx]
            window_outcomes = outcomes_sorted[start_idx:end_idx]
            
            # Discover patterns in this window
            patterns = discovery.discover_semantic_patterns(window_texts, n_patterns=5)
            
            # Track pattern presence over time
            for pattern_name, pattern_data in patterns.items():
                if pattern_name not in evolution_data:
                    evolution_data[pattern_name] = {
                        'timestamps': [],
                        'prevalence': [],
                        'effectiveness': []
                    }
                
                # Calculate prevalence (how common is this pattern?)
                prevalence = pattern_data['cluster_size'] / len(window_texts)
                
                # Calculate effectiveness (does it predict outcomes?)
                pattern_texts = pattern_data['sample_texts']
                pattern_indices = [i for i, t in enumerate(window_texts) if t in pattern_texts]
                if len(pattern_indices) > 0:
                    pattern_outcomes = window_outcomes[pattern_indices]
                    effectiveness = np.mean(pattern_outcomes) if len(pattern_outcomes) > 0 else 0.0
                else:
                    effectiveness = 0.0
                
                evolution_data[pattern_name]['timestamps'].append(np.mean(window_times))
                evolution_data[pattern_name]['prevalence'].append(prevalence)
                evolution_data[pattern_name]['effectiveness'].append(effectiveness)
        
        return evolution_data
    
    def detect_emergence(self, evolution_data: Dict[str, Dict], threshold: float = 0.1) -> Dict[str, float]:
        """
        Detect when archetypes first emerged.
        
        Returns
        -------
        dict
            Archetype name → emergence timestamp
        """
        emergences = {}
        
        for pattern_name, data in evolution_data.items():
            prevalences = data['prevalence']
            
            # Find first time prevalence exceeded threshold
            for i, prev in enumerate(prevalences):
                if prev >= threshold:
                    emergences[pattern_name] = data['timestamps'][i]
                    break
        
        return emergences
    
    def detect_drift(self, evolution_data: Dict[str, Dict]) -> Dict[str, float]:
        """
        Measure how much archetypes have drifted over time.
        
        Returns
        -------
        dict
            Archetype name → drift score (0-1, higher = more drift)
        """
        drifts = {}
        
        for pattern_name, data in evolution_data.items():
            prevalences = data['prevalence']
            
            if len(prevalences) < 2:
                drifts[pattern_name] = 0.0
                continue
            
            # Measure variance in prevalence
            drift = np.std(prevalences) / (np.mean(prevalences) + 1e-8)
            drifts[pattern_name] = min(1.0, drift)  # Cap at 1.0
        
        return drifts


class CrossDomainPatternTransfer:
    """
    Identifies patterns that transfer across domains.
    
    Discovers:
    - Domain-agnostic archetypes
    - Meta-patterns (patterns of patterns)
    - Transfer learning opportunities
    """
    
    def __init__(self):
        self.domain_patterns = {}
        
    def register_domain_patterns(self, domain_name: str, patterns: Dict[str, List[str]]):
        """Register patterns discovered for a domain."""
        self.domain_patterns[domain_name] = patterns
    
    def find_transferable_patterns(
        self,
        min_domains: int = 2,
        min_pattern_overlap: float = 0.3
    ) -> Dict[str, List[str]]:
        """
        Find patterns that appear in multiple domains.
        
        Parameters
        ----------
        min_domains : int
            Minimum number of domains pattern must appear in
        min_pattern_overlap : float
            Minimum overlap between pattern sets
        
        Returns
        -------
        dict
            Transferable patterns with domain list
        """
        # Collect all patterns across domains
        all_patterns = defaultdict(list)
        
        for domain, patterns in self.domain_patterns.items():
            for pattern_name, pattern_list in patterns.items():
                # Normalize pattern names (remove domain prefix)
                normalized_name = pattern_name.replace(f'{domain}_', '').replace('_', ' ')
                
                for pattern in pattern_list:
                    all_patterns[pattern].append(domain)
        
        # Find patterns appearing in multiple domains
        transferable = {}
        
        for pattern, domains_list in all_patterns.items():
            unique_domains = list(set(domains_list))
            
            if len(unique_domains) >= min_domains:
                transferable[pattern] = {
                    'domains': unique_domains,
                    'frequency': len(domains_list),
                    'domain_count': len(unique_domains)
                }
        
        return transferable
    
    def discover_meta_archetypes(
        self,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Dict]:
        """
        Discover meta-archetypes (patterns of patterns).
        
        Groups similar archetypes across domains into meta-categories.
        
        Returns
        -------
        dict
            Meta-archetypes with constituent patterns
        """
        # Collect all archetype patterns
        all_archetypes = []
        archetype_to_domain = {}
        
        for domain, patterns in self.domain_patterns.items():
            for arch_name, arch_patterns in patterns.items():
                all_archetypes.append({
                    'name': f"{domain}::{arch_name}",
                    'patterns': arch_patterns,
                    'domain': domain
                })
                archetype_to_domain[f"{domain}::{arch_name}"] = domain
        
        # Cluster similar archetypes
        if len(all_archetypes) < 2:
            return {}
        
        # Create pattern vectors
        all_patterns_set = set()
        for arch in all_archetypes:
            all_patterns_set.update(arch['patterns'])
        
        pattern_list = list(all_patterns_set)
        pattern_to_idx = {p: i for i, p in enumerate(pattern_list)}
        
        # Create binary vectors (pattern present/absent)
        vectors = []
        for arch in all_archetypes:
            vec = np.zeros(len(pattern_list))
            for pattern in arch['patterns']:
                if pattern in pattern_to_idx:
                    vec[pattern_to_idx[pattern]] = 1.0
            vectors.append(vec)
        
        vectors = np.array(vectors)
        
        # Cluster by similarity
        similarities = cosine_similarity(vectors)
        
        # Find groups of similar archetypes
        meta_archetypes = {}
        used = set()
        
        for i, arch in enumerate(all_archetypes):
            if i in used:
                continue
            
            # Find similar archetypes
            similar_indices = np.where(similarities[i] >= similarity_threshold)[0]
            similar_archetypes = [all_archetypes[j] for j in similar_indices if j != i]
            
            if len(similar_archetypes) > 0:
                meta_name = f"meta_{arch['name']}"
                meta_archetypes[meta_name] = {
                    'constituent_archetypes': [arch['name']] + [a['name'] for a in similar_archetypes],
                    'domains': list(set([arch['domain']] + [a['domain'] for a in similar_archetypes])),
                    'common_patterns': self._find_common_patterns([arch] + similar_archetypes),
                    'similarity_score': np.mean([similarities[i, j] for j in similar_indices])
                }
                
                used.update(similar_indices)
        
        return meta_archetypes
    
    def _find_common_patterns(self, archetypes: List[Dict]) -> List[str]:
        """Find patterns common to multiple archetypes."""
        if len(archetypes) == 0:
            return []
        
        # Intersection of all pattern sets
        common = set(archetypes[0]['patterns'])
        for arch in archetypes[1:]:
            common &= set(arch['patterns'])
        
        return list(common)

