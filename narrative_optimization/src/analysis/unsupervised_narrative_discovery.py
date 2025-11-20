"""
Unsupervised Narrative Pattern Discovery

CRITICAL PHILOSOPHY:
- DO NOT impose archetypes (Campbell, Jung, etc.)
- DO NOT hardcode patterns to look for
- DO NOT explain mechanisms
- LET DATA REVEAL structure
- LET AI find patterns without guidance
- KEEP MECHANISMS ELUSIVE for better analysis

Approach:
1. Collect large narrative corpus (10K-100K narratives)
2. Embed using AI (capture semantic structure)
3. Discover natural clusters (unsupervised)
4. Let patterns emerge without preconception
5. Correlate emergent patterns with outcomes
6. DO NOT name patterns prematurely (let them remain mysterious)

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import sys
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.manifold import TSNE, Isomap
import warnings

# Optional imports
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    HDBSCAN = None

try:
    from umap import UMAP
except ImportError:
    UMAP = None

try:
    from ..transformers.utils.embeddings import EmbeddingManager
    from ..transformers.utils.shared_models import SharedModelRegistry
except ImportError:
    # When imported from project root, ensure narrative_optimization/src is on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers.utils.embeddings import EmbeddingManager
    from transformers.utils.shared_models import SharedModelRegistry


class UnsupervisedNarrativeDiscovery:
    """
    Discover narrative patterns without presupposing what they are.
    
    Philosophy:
    - NO predefined archetypes
    - NO mechanistic explanations
    - ONLY measure what emerges
    - Let patterns remain mysterious
    - Better analysis through NOT knowing
    
    Process:
    1. Embed narratives (capture ALL semantic structure)
    2. Reduce dimensions (find natural structure)
    3. Cluster (discover natural groupings)
    4. Correlate with outcomes (which clusters win?)
    5. Name patterns ONLY after validation (if ever)
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        cache_dir: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize discovery system.
        
        Parameters
        ----------
        embedding_model : str
            Sentence transformer model (captures semantic structure)
        cache_dir : str, optional
            Cache directory for embeddings
        random_state : int
            For reproducibility in clustering
        """
        self.embedder = EmbeddingManager(
            model_name=embedding_model,
            cache_dir=cache_dir,
            use_cache=True
        )
        
        self.random_state = random_state
        
        # Will be populated during analysis
        self.embeddings_ = None
        self.clusters_ = None
        self.patterns_ = None
        self.outcome_correlations_ = None
        
    def discover_patterns(
        self,
        narratives: List[str],
        outcomes: Optional[np.ndarray] = None,
        min_cluster_size: int = 50,
        n_latent_dimensions: int = 50,
        method: str = 'auto',
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Discover natural patterns in narrative corpus.
        
        Parameters
        ----------
        narratives : list of str
            Raw narrative texts (NO preprocessing)
        outcomes : ndarray, optional
            Outcome measures (if available) to correlate patterns with success
        min_cluster_size : int
            Minimum cluster size for HDBSCAN
        n_latent_dimensions : int
            Number of latent dimensions to extract
        method : str
            Discovery method: 'auto', 'clustering', 'decomposition', 'manifold'
            
        Returns
        -------
        discovery : dict
            {
                'embeddings': Full semantic embeddings,
                'latent_structure': Reduced dimensions,
                'clusters': Natural groupings,
                'patterns': Emergent patterns (NOT named archetypes),
                'outcome_correlations': Which patterns predict success,
                'mysterious_dimensions': Dimensions we can measure but not interpret
            }
        """
        print(f"\n{'='*80}")
        print("UNSUPERVISED NARRATIVE PATTERN DISCOVERY")
        print(f"{'='*80}\n")
        print(f"Corpus: {len(narratives):,} narratives")
        print(f"Philosophy: Let data reveal structure. Do not impose theory.")
        
        if features is not None:
            print(f"\n[1/5] Using supplied transformer feature matrix...")
            self.embeddings_ = np.asarray(features)
            print(f"      ✓ Shape: {self.embeddings_.shape} (narratives × transformer features)")
        else:
            print(f"\n[1/5] Embedding narratives (capturing full semantic structure)...")
            self.embeddings_ = self.embedder.encode(
                narratives,
                show_progress=True
            )
            print(f"      ✓ Shape: {self.embeddings_.shape} (narratives × semantic dimensions)")
        
        # Step 2: Discover latent structure (multiple methods)
        print(f"\n[2/5] Discovering latent structure...")
        latent_structure = self._discover_latent_structure(
            self.embeddings_,
            n_dimensions=n_latent_dimensions
        )
        print(f"      ✓ Extracted {latent_structure.shape[1]} latent dimensions")
        print(f"      ℹ These dimensions capture narrative structure WITHOUT naming them")
        
        # Step 3: Cluster naturally (find groupings without presupposition)
        print(f"\n[3/5] Discovering natural clusters (unsupervised)...")
        clusters = self._discover_clusters(
            latent_structure,
            min_size=min_cluster_size
        )
        print(f"      ✓ Found {len(set(clusters)) - (1 if -1 in clusters else 0)} natural clusters")
        print(f"      ℹ Clusters emerged from data, not from theory")
        
        # Step 4: Characterize patterns (measure, don't explain)
        print(f"\n[4/5] Characterizing emergent patterns...")
        patterns = self._characterize_patterns(
            narratives,
            self.embeddings_,
            latent_structure,
            clusters
        )
        print(f"      ✓ {len(patterns)} patterns characterized")
        print(f"      ℹ Patterns described by measurements, not interpretations")
        
        # Step 5: Correlate with outcomes (if provided)
        if outcomes is not None:
            print(f"\n[5/5] Correlating patterns with outcomes...")
            outcome_correlations = self._correlate_with_outcomes(
                latent_structure,
                clusters,
                patterns,
                outcomes
            )
            print(f"      ✓ Identified {len([c for c in outcome_correlations if c['significant']])} significant pattern-outcome relationships")
            print(f"      ℹ Patterns that predict success (mechanism unknown)")
        else:
            outcome_correlations = None
            print(f"\n[5/5] Skipping outcome correlation (no outcomes provided)")
        
        print(f"\n{'='*80}")
        print("DISCOVERY COMPLETE")
        print(f"{'='*80}\n")
        print("REMINDER: Patterns are EMERGENT. Do not force-fit to existing theories.")
        print("Let the mystery remain. Better analysis through NOT explaining.\n")
        
        return {
            'embeddings': self.embeddings_,
            'latent_structure': latent_structure,
            'clusters': clusters,
            'patterns': patterns,
            'outcome_correlations': outcome_correlations,
            'metadata': {
                'n_narratives': len(narratives),
                'embedding_dim': self.embeddings_.shape[1],
                'latent_dim': n_latent_dimensions,
                'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                'method': method
            }
        }
    
    def _discover_latent_structure(
        self,
        embeddings: np.ndarray,
        n_dimensions: int
    ) -> np.ndarray:
        """
        Extract latent dimensions without presupposing what they mean.
        
        Uses multiple decomposition methods:
        - PCA: Linear variance maximization
        - ICA: Statistical independence
        - NMF: Non-negative parts
        - UMAP: Topological structure
        
        DO NOT interpret dimensions. Measure, don't explain.
        """
        latent_representations = {}
        
        max_allowed = min(
            n_dimensions,
            embeddings.shape[0] - 1 if embeddings.shape[0] > 1 else embeddings.shape[0],
            embeddings.shape[1],
        )
        n_components = max(1, max_allowed)
        if n_components < n_dimensions:
            print(f"      ℹ Reducing latent dimensions from {n_dimensions} to {n_components} due to sample limits")
        # Method 1: PCA (linear variance structure)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        latent_representations['pca'] = pca.fit_transform(embeddings)
        
        # Method 2: ICA (statistically independent components)
        ica = FastICA(n_components=n_components, random_state=self.random_state, max_iter=500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            latent_representations['ica'] = ica.fit_transform(embeddings)
        
        # Method 3: NMF (non-negative parts-based)
        # Shift embeddings to be non-negative
        embeddings_shifted = embeddings - embeddings.min() + 0.1
        nmf = NMF(n_components=n_components, random_state=self.random_state, max_iter=500)
        latent_representations['nmf'] = nmf.fit_transform(embeddings_shifted)
        
        # Method 4: UMAP (topological/manifold structure) - if available
        if UMAP is not None:
            umap = UMAP(
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=min(50, len(embeddings) // 10),
                min_dist=0.1
            )
            latent_representations['umap'] = umap.fit_transform(embeddings)
        else:
            # Fallback: use TSNE with reduced dimensions (barnes_hut limitation)
            tsne_dims = min(3, n_components)  # TSNE barnes_hut requires < 4
            perplexity = min(30, max(1, len(embeddings) // 2))
            if perplexity >= len(embeddings):
                perplexity = max(1, len(embeddings) - 1)
            tsne = TSNE(
                n_components=tsne_dims,
                random_state=self.random_state,
                perplexity=perplexity or 1,
            )
            tsne_result = tsne.fit_transform(embeddings)
            # Pad to n_dimensions if needed
            if tsne_dims < n_components:
                padding = np.zeros((len(embeddings), n_components - tsne_dims))
                latent_representations['umap'] = np.hstack([tsne_result, padding])
            else:
                latent_representations['umap'] = tsne_result
        
        # Combine all methods (ensemble)
        # Each captures different aspect of structure
        combined = np.hstack([
            latent_representations['pca'],
            latent_representations['ica'][:, :max(1, n_components // 2)],  # Use half of ICA
            latent_representations['umap'][:, :max(1, n_components // 2)]  # Use half of UMAP
        ])
        
        # Reduce back to n_dimensions (meta-structure across methods)
        final_pca = PCA(n_components=n_components, random_state=self.random_state)
        final_latent = final_pca.fit_transform(combined)
        
        # Store variance explained
        self.variance_explained_ = final_pca.explained_variance_ratio_
        
        return final_latent
    
    def _discover_clusters(
        self,
        latent_structure: np.ndarray,
        min_size: int
    ) -> np.ndarray:
        """
        Find natural clusters WITHOUT presupposing how many or what they are.
        
        Uses HDBSCAN: density-based clustering that discovers number of clusters.
        NO need to specify K - algorithm finds natural groupings.
        """
        # Use HDBSCAN if available, otherwise KMeans
        if HDBSCAN is not None:
            try:
                clusterer = HDBSCAN(
                    min_cluster_size=min_size,
                    min_samples=min_size // 2,
                    metric='euclidean',
                    cluster_selection_method='eom'  # Excess of mass
                )
                
                clusters = clusterer.fit_predict(latent_structure)
                
                # Store cluster info
                self.cluster_probabilities_ = clusterer.probabilities_
                self.cluster_outlier_scores_ = clusterer.outlier_scores_
            except (TypeError, AttributeError):
                # HDBSCAN version incompatibility, fallback to KMeans
                HDBSCAN_working = None
                n_clusters = max(3, len(latent_structure) // (min_size * 2))
                n_clusters = min(n_clusters, 20)
                clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                clusters = clusterer.fit_predict(latent_structure)
                self.cluster_probabilities_ = None
                self.cluster_outlier_scores_ = None
        else:
            # Fallback: KMeans with estimated cluster count
            n_clusters = max(3, len(latent_structure) // (min_size * 2))
            n_clusters = min(n_clusters, 20)  # Cap at 20
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            clusters = clusterer.fit_predict(latent_structure)
            
            self.cluster_probabilities_ = None
            self.cluster_outlier_scores_ = None
        
        return clusters
    
    def _characterize_patterns(
        self,
        narratives: List[str],
        embeddings: np.ndarray,
        latent_structure: np.ndarray,
        clusters: np.ndarray
    ) -> List[Dict]:
        """
        Characterize each cluster/pattern WITHOUT naming or explaining it.
        
        Measurements only:
        - Centroid position
        - Variance structure
        - Size
        - Density
        - Separation from other clusters
        
        DO NOT interpret. DO NOT name. DO NOT explain.
        Let patterns remain mysterious.
        """
        unique_clusters = [c for c in set(clusters) if c != -1]  # Exclude noise
        patterns = []
        
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_narratives = [n for n, m in zip(narratives, mask) if m]
            cluster_embeddings = embeddings[mask]
            cluster_latent = latent_structure[mask]
            
            # Measure pattern characteristics (NO interpretation)
            pattern = {
                'id': f'pattern_{cluster_id}',
                'size': int(np.sum(mask)),
                'size_pct': float(np.mean(mask)),
                
                # Centroid (where pattern lives in space)
                'centroid_latent': cluster_latent.mean(axis=0).tolist(),
                'centroid_embedding': cluster_embeddings.mean(axis=0),  # Don't serialize
                
                # Spread (how tight or loose)
                'variance': float(cluster_latent.var()),
                'std': float(cluster_latent.std()),
                
                # Density (how concentrated)
                'density': float(self._estimate_density(cluster_latent)),
                
                # Separation (how distinct from others)
                'separation': float(self._measure_separation(cluster_id, clusters, latent_structure)),
                
                # Coherence (internal consistency)
                'coherence': float(self._measure_coherence(cluster_latent)),
                
                # Example narratives (for human inspection, NOT for training)
                'example_narratives': cluster_narratives[:5] if len(cluster_narratives) >= 5 else cluster_narratives,
                
                # Mysterious dimensions (latent factors, no interpretation)
                'latent_signature': self._extract_signature(cluster_latent)
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def _correlate_with_outcomes(
        self,
        latent_structure: np.ndarray,
        clusters: np.ndarray,
        patterns: List[Dict],
        outcomes: np.ndarray
    ) -> List[Dict]:
        """
        Find which patterns predict success WITHOUT explaining why.
        
        Correlation only. No causation claims. No mechanism identification.
        Just: Pattern X correlates with success at r=0.XX.
        """
        from scipy import stats
        
        correlations = []
        
        # Method 1: Cluster membership predicts outcome
        unique_clusters = [c for c in set(clusters) if c != -1]
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            
            if np.sum(mask) < 30:  # Skip tiny clusters
                continue
            
            cluster_outcomes = outcomes[mask]
            other_outcomes = outcomes[~mask]
            
            # Test if outcomes differ
            if outcomes.dtype in [np.float32, np.float64]:
                # Continuous outcomes - t-test
                t_stat, p_val = stats.ttest_ind(cluster_outcomes, other_outcomes)
                effect_size = (cluster_outcomes.mean() - other_outcomes.mean()) / outcomes.std()
                test_type = 't-test'
            else:
                # Binary outcomes - proportion test
                cluster_mean = cluster_outcomes.mean()
                other_mean = other_outcomes.mean()
                
                # Use t-test on binary as approximation (valid for large samples)
                t_stat, p_val = stats.ttest_ind(cluster_outcomes, other_outcomes)
                effect_size = cluster_mean - other_mean
                test_type = 'proportion_test'
            
            correlations.append({
                'pattern_id': f'pattern_{cluster_id}',
                'correlation_type': 'cluster_membership',
                'effect_size': float(effect_size),
                'p_value': float(p_val),
                'significant': p_val < 0.05,
                'cluster_mean': float(cluster_outcomes.mean()),
                'other_mean': float(other_outcomes.mean()),
                'test': test_type,
                'n_in_cluster': int(np.sum(mask))
            })
        
        # Method 2: Latent dimensions predict outcome
        for dim_idx in range(min(latent_structure.shape[1], 20)):  # Check first 20 dimensions
            dim_values = latent_structure[:, dim_idx]
            
            # Correlation with outcome
            r, p_val = stats.pearsonr(dim_values, outcomes)
            
            if abs(r) > 0.10 or p_val < 0.05:  # Noteworthy correlation
                correlations.append({
                    'pattern_id': f'latent_dim_{dim_idx}',
                    'correlation_type': 'latent_dimension',
                    'effect_size': float(r),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05,
                    'interpretation': 'UNKNOWN - dimension eludes interpretation',
                    'note': 'Correlation exists. Mechanism remains mysterious.'
                })
        
        # Multiple-comparison corrections (Bonferroni + FDR/BH)
        if correlations:
            pvals = np.array([c.get('p_value', 1.0) for c in correlations], dtype=float)
            bonf = np.minimum(pvals * len(pvals), 1.0)
            order = np.argsort(pvals)
            ranks = np.arange(1, len(pvals) + 1)
            fdr = np.empty_like(pvals)
            fdr[order] = pvals[order] * len(pvals) / ranks
            fdr = np.minimum.accumulate(fdr[::-1])[::-1]
            fdr = np.minimum(fdr, 1.0)
            
            for idx, corr in enumerate(correlations):
                corr['p_value_bonferroni'] = float(bonf[idx])
                corr['p_value_fdr'] = float(fdr[idx])
                corr['significant_bonferroni'] = bool(bonf[idx] < 0.05)
                corr['significant_fdr'] = bool(fdr[idx] < 0.05)
        
        # Sort by effect size
        correlations.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        
        return correlations
    
    def _estimate_density(self, points: np.ndarray) -> float:
        """Estimate cluster density (tightness)."""
        if len(points) < 2:
            return 0.0
        
        # Average distance to centroid
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        avg_distance = distances.mean()
        
        # Density = 1 / avg_distance
        density = 1.0 / (1.0 + avg_distance)
        return density
    
    def _measure_separation(
        self,
        cluster_id: int,
        all_clusters: np.ndarray,
        latent_structure: np.ndarray
    ) -> float:
        """Measure how separated this cluster is from others."""
        mask = all_clusters == cluster_id
        this_cluster = latent_structure[mask]
        other_clusters = latent_structure[~mask]
        
        if len(this_cluster) == 0 or len(other_clusters) == 0:
            return 0.0
        
        # Distance between centroids
        this_centroid = this_cluster.mean(axis=0)
        other_centroid = other_clusters.mean(axis=0)
        
        separation = np.linalg.norm(this_centroid - other_centroid)
        
        # Normalize by within-cluster spread
        this_spread = this_cluster.std()
        other_spread = other_clusters.std()
        avg_spread = (this_spread + other_spread) / 2
        
        normalized_separation = separation / (avg_spread + 0.1)
        
        return min(normalized_separation, 5.0) / 5.0  # Cap and normalize
    
    def _measure_coherence(self, points: np.ndarray) -> float:
        """Measure internal coherence (how similar members are)."""
        if len(points) < 2:
            return 1.0
        
        # Pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(points)
        
        # Coherence = inverse of average distance
        avg_distance = distances.mean()
        coherence = 1.0 / (1.0 + avg_distance)
        
        return coherence
    
    def _extract_signature(self, cluster_latent: np.ndarray) -> Dict:
        """
        Extract pattern signature WITHOUT interpreting what it means.
        
        Returns dimensional profile: which latent dimensions are prominent.
        DO NOT name dimensions. Just measure them.
        """
        centroid = cluster_latent.mean(axis=0)
        
        # Find prominent dimensions (high absolute values)
        abs_centroid = np.abs(centroid)
        top_dims = np.argsort(abs_centroid)[-10:][::-1]  # Top 10 dimensions
        
        signature = {
            'prominent_dimensions': top_dims.tolist(),
            'dimension_values': centroid[top_dims].tolist(),
            'dimension_prominence': (abs_centroid[top_dims] / abs_centroid.sum()).tolist(),
            'note': 'These dimensions predict patterns. What they MEAN remains elusive.'
        }
        
        return signature
    
    def extract_features_for_downstream(
        self,
        narratives: List[str],
        use_clusters: bool = True,
        use_latent: bool = True
    ) -> np.ndarray:
        """
        Extract features for downstream analysis (prediction, betting, etc.).
        
        Features are:
        - Latent dimensions (mysterious but predictive)
        - Cluster memberships (patterns without names)
        - Similarity to cluster centroids
        
        NO interpretation required. Features work whether we understand them or not.
        
        Returns
        -------
        features : ndarray
            Shape (n_narratives, n_features)
            Features for prediction WITHOUT knowing what they mean
        """
        if self.embeddings_ is None:
            raise ValueError("Must run discover_patterns() first")
        
        # Embed new narratives
        new_embeddings = self.embedder.encode(narratives)
        
        # Project to latent space (using learned structure)
        # For now, just return embeddings (TODO: proper projection)
        
        feature_list = []
        
        if use_latent:
            # Latent dimensions (first 50 are most predictive)
            feature_list.append(new_embeddings[:, :50])
        
        if use_clusters and self.clusters_ is not None:
            # Distance to each cluster centroid
            unique_clusters = [c for c in set(self.clusters_) if c != -1]
            
            cluster_features = []
            for cluster_id in unique_clusters:
                mask = self.clusters_ == cluster_id
                centroid = self.embeddings_[mask].mean(axis=0)
                
                # Distance to centroid
                distances = np.linalg.norm(new_embeddings - centroid, axis=1)
                similarities = 1.0 / (1.0 + distances)
                
                cluster_features.append(similarities)
            
            if cluster_features:
                feature_list.append(np.column_stack(cluster_features))
        
        # Combine all features
        if feature_list:
            features = np.hstack(feature_list)
        else:
            features = new_embeddings
        
        return features


class BackgroundNarrativeCorpusBuilder:
    """
    Build large-scale narrative corpus for unsupervised discovery.
    
    Purpose: Collect 10K-100K narratives across media forms to discover
    universal patterns through AI analysis WITHOUT presupposing what they are.
    
    Sources:
    - Literature (novels, short stories)
    - Film (screenplays, plot summaries)
    - Television (episode summaries)
    - Historical narratives
    - News stories
    - Sports narratives
    - Business narratives
    - Personal narratives
    
    Process:
    - Collect systematically
    - Standardize format (but NOT content)
    - Label with outcomes (when available)
    - DO NOT filter by quality
    - DO NOT select for specific types
    - LET VARIETY EXIST
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize corpus builder.
        
        Parameters
        ----------
        output_dir : str, optional
            Where to store collected narratives
        """
        self.output_dir = Path(output_dir) if output_dir else Path('data/narrative_corpus')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.corpus = {
            'narratives': [],
            'metadata': [],
            'outcomes': [],
            'sources': []
        }
    
    def add_source(
        self,
        source_name: str,
        narratives: List[str],
        outcomes: Optional[List[Any]] = None,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add narratives from a source.
        
        Parameters
        ----------
        source_name : str
            Source identifier (e.g., 'imdb_plots', 'gutenberg_novels')
        narratives : list of str
            Raw narrative texts
        outcomes : list, optional
            Outcome measures (ratings, success, etc.)
        metadata : list of dict, optional
            Metadata for each narrative
        """
        n = len(narratives)
        
        self.corpus['narratives'].extend(narratives)
        self.corpus['sources'].extend([source_name] * n)
        
        if outcomes:
            self.corpus['outcomes'].extend(outcomes)
        else:
            self.corpus['outcomes'].extend([None] * n)
        
        if metadata:
            self.corpus['metadata'].extend(metadata)
        else:
            self.corpus['metadata'].extend([{}] * n)
        
        print(f"Added {n:,} narratives from {source_name}")
        print(f"Total corpus: {len(self.corpus['narratives']):,} narratives")
    
    def save_corpus(self, filename: str = 'narrative_corpus_v1.json'):
        """Save collected corpus."""
        output_path = self.output_dir / filename
        
        corpus_data = {
            'narratives': self.corpus['narratives'],
            'sources': self.corpus['sources'],
            'outcomes': self.corpus['outcomes'],
            'metadata': self.corpus['metadata'],
            'stats': {
                'total_narratives': len(self.corpus['narratives']),
                'sources': list(set(self.corpus['sources'])),
                'source_counts': {
                    source: self.corpus['sources'].count(source)
                    for source in set(self.corpus['sources'])
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(corpus_data, f, indent=2)
        
        print(f"\n✓ Corpus saved to: {output_path}")
        print(f"  Total: {len(self.corpus['narratives']):,} narratives")
        return output_path
    
    def load_corpus(self, filename: str):
        """Load existing corpus."""
        input_path = self.output_dir / filename
        
        with open(input_path, 'r') as f:
            corpus_data = json.load(f)
        
        self.corpus = corpus_data
        
        print(f"✓ Loaded corpus: {len(self.corpus['narratives']):,} narratives")
        return self.corpus


class NarrativePatternLibrary:
    """
    Library of discovered patterns (NOT predefined archetypes).
    
    Stores:
    - Emergent patterns from unsupervised discovery
    - Correlations with outcomes
    - Feature extractors based on patterns
    
    DOES NOT store:
    - Named archetypes (Campbell, Jung)
    - Mechanistic explanations
    - Causal theories
    
    Philosophy: Let patterns work without understanding them.
    Sometimes not knowing WHY enables better HOW.
    """
    
    def __init__(self, library_path: Optional[str] = None):
        """Initialize pattern library."""
        self.library_path = Path(library_path) if library_path else Path('narrative_optimization/data/pattern_library.json')
        self.patterns = {}
        
        if self.library_path.exists():
            self.load()
    
    def add_discovered_patterns(
        self,
        domain: str,
        patterns: List[Dict],
        discovery_info: Dict
    ):
        """
        Add patterns discovered from domain analysis.
        
        Parameters
        ----------
        domain : str
            Domain name
        patterns : list of dict
            Patterns from unsupervised discovery
        discovery_info : dict
            Metadata about discovery process
        """
        self.patterns[domain] = {
            'patterns': patterns,
            'discovery_info': discovery_info,
            'discovered_date': str(Path(__file__).stat().st_mtime)
        }
        
        print(f"Added {len(patterns)} patterns from {domain}")
    
    def get_patterns_for_domain(self, domain: str) -> Optional[List[Dict]]:
        """Retrieve discovered patterns for domain."""
        return self.patterns.get(domain, {}).get('patterns')
    
    def save(self):
        """Save pattern library."""
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.library_path, 'w') as f:
            # Don't serialize numpy arrays
            serializable = {}
            for domain, data in self.patterns.items():
                serializable[domain] = {
                    'patterns': [
                        {k: v for k, v in p.items() 
                         if k not in ['centroid_embedding']}  # Skip numpy
                        for p in data['patterns']
                    ],
                    'discovery_info': data['discovery_info']
                }
            
            json.dump(serializable, f, indent=2)
        
        print(f"✓ Pattern library saved: {len(self.patterns)} domains")
    
    def load(self):
        """Load pattern library."""
        with open(self.library_path, 'r') as f:
            self.patterns = json.load(f)
        
        print(f"✓ Loaded patterns from {len(self.patterns)} domains")


def analyze_corpus_unsupervised(
    narratives: List[str],
    outcomes: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
) -> Dict:
    """
    Main function: Discover patterns in narrative corpus WITHOUT preconceptions.
    
    This is the core analysis that should run on large corpora to discover
    what narrative structures exist naturally, not what theories say should exist.
    
    Parameters
    ----------
    narratives : list of str
        Large corpus (10K+ recommended)
    outcomes : ndarray, optional
        Success measures
    output_path : str, optional
        Where to save results
        
    Returns
    -------
    discovery : dict
        Emergent patterns, correlations, mysterious dimensions
    """
    print("\n" + "="*80)
    print("UNSUPERVISED NARRATIVE PATTERN DISCOVERY")
    print("="*80)
    print("\nPhilosophy:")
    print("- Do not impose Campbell, Jung, Aristotle, or any theory")
    print("- Let AI find natural structure in embedding space")
    print("- Discover patterns without naming them")
    print("- Correlate with outcomes without explaining why")
    print("- Keep mechanisms elusive for better analysis")
    print("="*80 + "\n")
    
    # Discover patterns
    discoverer = UnsupervisedNarrativeDiscovery()
    results = discoverer.discover_patterns(
        narratives=narratives,
        outcomes=outcomes,
        min_cluster_size=max(50, len(narratives) // 100),  # Adaptive
        n_latent_dimensions=50
    )
    
    # Save if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable results
        serializable_results = {
            'latent_structure': results['latent_structure'].tolist(),
            'clusters': results['clusters'].tolist(),
            'patterns': results['patterns'],
            'outcome_correlations': results['outcome_correlations'],
            'metadata': results['metadata']
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    return results

