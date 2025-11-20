# Transformer Speed Optimization Guide

**Date:** November 16, 2025  
**Purpose:** Comprehensive guide for optimizing slow transformers and preventing performance issues

---

## Quick Reference: Transformer Speed Tiers

### ⚡ **Tier S: Ultra-Fast** (<0.05s per 1,000 samples)
- **Multi-Scale**: 0.016s
- **Cognitive Fluency**: 0.019s
- **Context Pattern**: 0.062s

**Why they're fast:** Simple statistical operations, minimal string processing, efficient numpy operations.

---

### ✅ **Tier A: Fast** (0.05s - 0.3s per 1,000 samples)
- Quantitative: 0.102s
- Temporal Evolution: 0.115s
- Suspense Mystery: 0.119s
- Discoverability: 0.127s
- Hierarchical Nominative: 0.165s
- Optics: 0.169s
- Authenticity: 0.221s
- Coupling Strength: 0.238s

**Why they're fast:** Efficient feature extraction, minimal external dependencies, good vectorization.

---

### ⚠️ **Tier B: Moderate** (0.3s - 1.0s per 1,000 samples)
- Phonetic: 0.336s
- Emotional Resonance: 0.386s
- Narrative Mass: 0.526s

**Why they're moderate:** More complex linguistic analysis, some pairwise computations.

---

### 🔶 **Tier C: Slow** (1.0s - 5.0s per 1,000 samples)
- Awareness Resistance: 1.03s
- Information Theory: 1.61s
- Universal Nominative: 1.70s
- **Gravitational Features: 4.95s** ← PRIMARY OPTIMIZATION TARGET

**Why they're slow:**
- Multiple clustering passes
- Pairwise similarity computations
- Complex entropy calculations
- 100+ features

---

## Optimization Strategies by Problem Type

### Problem 1: Clustering Operations (Gravitational Features)

**Current bottleneck:**
```python
# Performs K-means twice, then computes all pairwise distances
story_clusters = KMeans(n_clusters=3).fit(story_vectors)
name_clusters = KMeans(n_clusters=3).fit(name_vectors)

for i in range(len(X)):
    for j in range(len(X)):  # O(n²) - SLOW!
        distance = compute_distance(i, j)
```

**Optimization strategies:**

#### Strategy 1: MiniBatch K-Means
```python
from sklearn.cluster import MiniBatchKMeans

# 5-10x faster for large datasets
story_clusters = MiniBatchKMeans(
    n_clusters=3,
    batch_size=100,
    max_iter=10
).fit(story_vectors)
```

**Expected speedup:** 5-10x  
**When to use:** Always (minimal accuracy loss)

#### Strategy 2: Approximate Nearest Neighbors
```python
from sklearn.neighbors import NearestNeighbors

# Instead of all pairwise distances, query only nearest
nn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
nn.fit(vectors)
distances, indices = nn.kneighbors(vectors)
```

**Expected speedup:** 10-100x  
**When to use:** When full pairwise matrix not needed

#### Strategy 3: Centroid-Only Distances
```python
# Instead of computing distance to every point,
# only compute distance to cluster centroids
centroids = story_clusters.cluster_centers_

# O(n * k) instead of O(n²)
centroid_distances = euclidean_distances(X, centroids)
```

**Expected speedup:** 100-1000x (for large n)  
**When to use:** When cluster-level patterns sufficient

---

### Problem 2: Entropy Calculations (Information Theory)

**Current bottleneck:**
```python
for text in X:
    # Character-level entropy
    char_counts = Counter(text)
    entropy = -sum(p * log(p) for p in probs)
    
    # N-gram entropy
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    trigrams = [text[i:i+3] for i in range(len(text)-2)]
```

**Optimization strategies:**

#### Strategy 1: Vectorized Entropy
```python
from scipy.stats import entropy

# Pre-compute character counts for all texts
all_chars = ''.join(X)
global_char_counts = Counter(all_chars)

# Vectorized entropy calculation
def fast_entropy(text):
    counts = np.array([global_char_counts.get(c, 0) for c in text])
    return entropy(counts)
```

**Expected speedup:** 2-3x

#### Strategy 2: Cache Common Patterns
```python
# Cache entropy for common words
@lru_cache(maxsize=10000)
def cached_entropy(word):
    return compute_entropy(word)

# Reuse for repeated words
entropies = [cached_entropy(word) for word in words]
```

**Expected speedup:** 5-10x for repetitive texts

#### Strategy 3: Approximate Kolmogorov Complexity
```python
# Instead of full compression, use faster approximation
def fast_complexity(text):
    # Simple compression ratio
    return len(text) / len(set(text))
```

**Expected speedup:** 10-50x  
**Trade-off:** Less theoretically pure but often correlates well

---

### Problem 3: Large Feature Spaces (Universal Nominative)

**Current bottleneck:**
```python
# Generates 116 features per sample
# Many features may be correlated or low-value
```

**Optimization strategies:**

#### Strategy 1: Feature Selection
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# During training, identify top-K features
selector = SelectKBest(mutual_info_classif, k=50)
selector.fit(X_all_features, y)

# Use only selected features in production
selected_indices = selector.get_support(indices=True)
```

**Expected speedup:** 2-3x  
**Trade-off:** May lose some signal from low-importance features

#### Strategy 2: Lazy Feature Computation
```python
class LazyUniversalNominative:
    def fit(self, X, y=None):
        # Fit but don't compute features yet
        self.is_fitted_ = True
        return self
    
    def transform(self, X, feature_groups=['fast']):
        # Compute only requested feature groups
        if 'fast' in feature_groups:
            features = self._compute_fast_features(X)
        if 'slow' in feature_groups:
            features = np.hstack([features, self._compute_slow_features(X)])
        return features
```

**Expected speedup:** Variable (user-controlled)

#### Strategy 3: Parallel Feature Extraction
```python
from joblib import Parallel, delayed

def extract_features_parallel(texts, n_jobs=-1):
    # Extract features in parallel across CPU cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_features_single)(text)
        for text in texts
    )
    return np.array(results)
```

**Expected speedup:** 2-8x (depending on cores)

---

## General Optimization Techniques

### Technique 1: Caching with joblib

```python
from joblib import Memory

memory = Memory(location='./cache', verbose=0)

@memory.cache
def expensive_computation(text):
    # This will only run once per unique input
    return compute_expensive_features(text)

# Subsequent calls with same input return cached result
features = expensive_computation(text)  # Fast on repeat!
```

**When to use:** 
- Repeated analyses of same data
- Development/testing with same dataset
- APIs serving repeated requests

**Speedup:** 100-1000x for cached results

---

### Technique 2: Batched Processing

```python
class BatchedTransformer:
    def transform(self, X, batch_size=100):
        """Process in batches for better memory efficiency"""
        results = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_features = self._transform_batch(batch)
            results.append(batch_features)
        
        return np.vstack(results)
    
    def _transform_batch(self, batch):
        # Vectorized operations on batch
        # Better cache usage, can use GPU
        return vectorized_operation(batch)
```

**When to use:** Large datasets (>10K samples)  
**Speedup:** 2-5x plus better memory usage

---

### Technique 3: Numba JIT Compilation

```python
from numba import jit

@jit(nopython=True)
def fast_distance_computation(a, b):
    """Compiled to machine code for speed"""
    total = 0.0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return np.sqrt(total)
```

**When to use:** 
- Tight loops over numeric arrays
- Mathematical computations
- No string operations (numba limitation)

**Speedup:** 10-100x for numeric loops

---

### Technique 4: Profiling Before Optimizing

```python
import cProfile
import pstats

def profile_transformer(transformer, X, y):
    """Identify actual bottlenecks"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    transformer.fit_transform(X, y)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Show top 20 slowest functions
```

**ALWAYS profile before optimizing!** You might be surprised what's actually slow.

---

## Specific Transformer Optimization Plans

### 🎯 Gravitational Features (4.95s → 0.5s target)

**Bottlenecks identified:**
1. Double clustering (story + names)
2. Pairwise distance computations
3. TF-IDF vectorization (2x)

**Optimization plan:**

```python
class OptimizedGravitationalFeatures(GravitationalFeaturesTransformer):
    def __init__(self, *args, use_approximate=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_approximate = use_approximate
    
    def fit(self, X, y=None):
        # Use MiniBatch K-Means
        from sklearn.cluster import MiniBatchKMeans
        
        # Vectorize once, reuse
        self.story_vectorizer_ = TfidfVectorizer(max_features=50)  # Reduced from 100
        self.name_vectorizer_ = TfidfVectorizer(max_features=30)
        
        story_vectors = self.story_vectorizer_.fit_transform(X)
        name_vectors = self.name_vectorizer_.fit_transform(X)
        
        # Faster clustering
        self.story_clusters_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=100
        ).fit(story_vectors)
        
        self.name_clusters_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=100
        ).fit(name_vectors)
        
        # Store centroids only, not all pairwise distances
        self.story_centroids_ = self.story_clusters_.cluster_centers_
        self.name_centroids_ = self.name_clusters_.cluster_centers_
        
        return self
    
    def transform(self, X):
        story_vectors = self.story_vectorizer_.transform(X)
        name_vectors = self.name_vectorizer_.transform(X)
        
        # Compute distances to centroids only (not all pairs)
        story_distances = euclidean_distances(
            story_vectors.toarray(),
            self.story_centroids_
        )
        
        name_distances = euclidean_distances(
            name_vectors.toarray(),
            self.name_centroids_
        )
        
        # Build features from centroid distances
        features = np.column_stack([
            story_distances.min(axis=1),  # Distance to nearest story cluster
            story_distances.mean(axis=1),  # Average distance to all clusters
            name_distances.min(axis=1),
            name_distances.mean(axis=1),
            # ... 16 more features derived from centroids
        ])
        
        return features
```

**Expected result:** 4.95s → 0.5-1.0s (5-10x speedup)

---

### 🎯 Universal Nominative (1.70s → 0.5s target)

**Bottlenecks identified:**
1. 116 features (many correlated)
2. Multiple string operations per sample
3. No feature selection

**Optimization plan:**

```python
class OptimizedUniversalNominative(UniversalNominativeTransformer):
    def __init__(self, *args, n_features=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features = n_features
        self.feature_selector_ = None
    
    def fit(self, X, y=None):
        # Extract all 116 features during training
        X_full = super().fit_transform(X, y)
        
        if y is not None and len(np.unique(y)) > 1:
            # Select top-K features based on mutual information
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            
            self.feature_selector_ = SelectKBest(
                mutual_info_classif,
                k=self.n_features
            )
            self.feature_selector_.fit(X_full, y)
        
        return self
    
    def transform(self, X):
        X_full = super().transform(X)
        
        if self.feature_selector_:
            # Use only selected features
            return self.feature_selector_.transform(X_full)
        else:
            # No selection, return top 50 by default
            return X_full[:, :self.n_features]
```

**Expected result:** 1.70s → 0.5-0.8s (2-3x speedup)

---

### 🎯 Information Theory (1.61s → 0.5s target)

**Bottlenecks identified:**
1. Entropy calculations per sample
2. N-gram generation
3. Compression simulations

**Optimization plan:**

```python
from functools import lru_cache
from scipy.stats import entropy as scipy_entropy

class OptimizedInformationTheory(InformationTheoryTransformer):
    def __init__(self, *args, use_cache=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = use_cache
    
    @lru_cache(maxsize=10000)
    def _cached_entropy(self, text_tuple):
        """Cached entropy for repeated strings"""
        text = ''.join(text_tuple)  # Tuples are hashable
        counts = Counter(text)
        probs = np.array(list(counts.values())) / len(text)
        return scipy_entropy(probs, base=2)
    
    def _extract_info_theory_features(self, text):
        # Use cached version for repeated texts
        if self.use_cache:
            entropy = self._cached_entropy(tuple(text))
        else:
            entropy = super()._compute_entropy(text)
        
        # Approximate Kolmogorov complexity (fast)
        approx_complexity = len(text) / (len(set(text)) + 1)
        
        # Skip expensive bigram/trigram analysis, use approximations
        # ...
        
        return features
```

**Expected result:** 1.61s → 0.5-0.8s (2-3x speedup)

---

## Testing Optimizations

### Performance Test Template

```python
import time
import numpy as np

def benchmark_transformer(transformer_class, X, y, n_runs=3):
    """Benchmark transformer performance"""
    times = []
    
    for _ in range(n_runs):
        transformer = transformer_class()
        
        start = time.time()
        transformer.fit_transform(X, y)
        elapsed = time.time() - start
        
        times.append(elapsed)
    
    print(f"{transformer_class.__name__}:")
    print(f"  Mean: {np.mean(times):.3f}s")
    print(f"  Std: {np.std(times):.3f}s")
    print(f"  Min: {np.min(times):.3f}s")
    print(f"  Max: {np.max(times):.3f}s")
    
    return np.mean(times)

# Compare original vs optimized
original_time = benchmark_transformer(GravitationalFeaturesTransformer, X, y)
optimized_time = benchmark_transformer(OptimizedGravitationalFeatures, X, y)

speedup = original_time / optimized_time
print(f"\nSpeedup: {speedup:.1f}x")
```

---

## Optimization Priority Matrix

| Transformer | Current Time | Priority | Expected Speedup | Effort | Value |
|------------|--------------|----------|------------------|--------|-------|
| **Gravitational Features** | 4.95s | 🔴 HIGH | 5-10x | Medium | High |
| **Universal Nominative** | 1.70s | 🟡 MEDIUM | 2-3x | Low | Medium |
| **Information Theory** | 1.61s | 🟡 MEDIUM | 2-3x | Low | Medium |
| Awareness Resistance | 1.03s | 🟢 LOW | 2x | Low | Low |
| Narrative Mass | 0.53s | 🟢 LOW | 1.5x | Low | Low |

**Recommendation:** Focus on Gravitational Features first, then revisit others only if needed for production.

---

## Production Deployment Checklist

### Before deploying optimized transformers:

- [ ] Profile to identify actual bottlenecks (not assumptions!)
- [ ] Benchmark original vs optimized with realistic data sizes
- [ ] Verify feature output matches (or acceptable differences)
- [ ] Test with edge cases (1 sample, 10 samples, 100K samples)
- [ ] Check memory usage (not just speed)
- [ ] Add unit tests for optimization code paths
- [ ] Document any trade-offs (accuracy vs speed)
- [ ] Monitor production performance metrics

---

## When NOT to Optimize

**Don't optimize if:**
1. Transformer is rarely used (<1% of workload)
2. Current speed is acceptable for use case
3. Optimization would sacrifice significant accuracy
4. Time would be better spent on other improvements
5. Dataset size will remain small (<10K samples)

**Remember:** Premature optimization is the root of all evil. Only optimize what actually matters for your production use case.

---

## Conclusion

The good news: Most transformers are already fast enough for production use. Only Gravitational Features warrants immediate optimization if used at scale.

Focus effort on:
1. **Fixing the 17 broken transformers** (higher priority than optimization)
2. **Optimizing Gravitational Features** if used in production
3. **Adding caching layer** for repeated analyses
4. **Monitoring production metrics** to identify real bottlenecks

The 80/20 rule applies: Fix the critical bugs first, optimize only the bottlenecks that matter.

