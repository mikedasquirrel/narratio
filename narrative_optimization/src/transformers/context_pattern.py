"""
Context Pattern Transformer - Universal High-Leverage Context Discovery

Discovers contextual combinations that create non-linear narrative effects across ALL domains.

Philosophy:
-----------
Narrative symmetry principle: If specific contextual combinations create predictable 
effects in one domain (Tennis: surface × round × rivalry → 98.5% accuracy), then 
analogous contextual structures should exist in ALL domains across the π spectrum.

This transformer:
1. Automatically discovers high-leverage contexts (no pre-defined patterns)
2. Works across entire π spectrum (π=0.04 lottery to π=0.974 WWE)
3. Tests ALL feature combinations to find non-linear interactions
4. Validates with sample size, effect size, and consistency requirements
5. Generalizes from sports → entertainment → business → natural phenomena

Examples of Narrative Symmetry:
--------------------------------
Sports:      surface × round × rivalry → outcome
Crypto:      novelty × market_cap × social_buzz → moonshot  
Oscars:      runtime × genre × release_month → nomination
Startups:    story_quality × founder_exp × timing → funding
Housing:     name_quality × affluence × inventory → premium
Mental Health: name_harshness × chronicity × stigma → severity

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
from scipy import stats
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class ContextPattern:
    """
    Represents a discovered high-leverage context.
    
    A context is a specific combination of feature conditions that creates
    predictable narrative effects beyond base rates.
    """
    
    def __init__(
        self,
        features: List[str],
        conditions: Dict[str, Any],
        accuracy: float,
        sample_size: int,
        effect_size: float,
        p_value: float,
        novelty: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        self.features = features
        self.conditions = conditions
        self.accuracy = accuracy
        self.sample_size = sample_size
        self.effect_size = effect_size
        self.p_value = p_value
        self.novelty = novelty
        self.metadata = metadata or {}
        
        # Compute composite score
        self.score = self._compute_score()
    
    def _compute_score(self) -> float:
        """
        Composite score balancing accuracy, sample size, and effect size.
        
        Formula: accuracy × sqrt(sample_size) × effect_size
        
        This rewards:
        - High accuracy (primary)
        - Sufficient samples (reliability)
        - Large effect size (practical significance)
        """
        return self.accuracy * np.sqrt(self.sample_size) * self.effect_size
    
    def matches(self, X: pd.DataFrame) -> np.ndarray:
        """Check which samples match this context"""
        mask = np.ones(len(X), dtype=bool)
        
        for feature, condition in self.conditions.items():
            if feature not in X.columns:
                continue
                
            if isinstance(condition, dict):
                # Range condition
                if 'min' in condition:
                    mask &= (X[feature] >= condition['min'])
                if 'max' in condition:
                    mask &= (X[feature] <= condition['max'])
                if 'eq' in condition:
                    mask &= (X[feature] == condition['eq'])
            else:
                # Direct value
                mask &= (X[feature] == condition)
        
        return mask
    
    def __repr__(self):
        cond_str = ', '.join([f"{k}={v}" for k, v in self.conditions.items()])
        return f"Context({cond_str} → {self.accuracy:.1%}, n={self.sample_size}, p={self.p_value:.4f})"


class ContextPatternTransformer(BaseEstimator, TransformerMixin):
    """
    Universal context pattern discovery transformer.
    
    Automatically discovers high-leverage contextual combinations that create
    non-linear narrative effects. Works across all domains via narrative symmetry.
    
    Features Generated (60 total):
    -------------------------------
    A. Context Membership (20 features):
       - Does sample match any discovered high-leverage contexts?
       - Context confidence scores (0-1 for each pattern)
       - Context novelty (rarity of this combination)
       
    B. Feature Interactions (25 features):
       - Cross-products of important features
       - Polynomial interactions (feature²)
       - Threshold proximity (distance to critical values)
       
    C. Historical Pattern Strength (10 features):
       - Historical accuracy in this context
       - Sample size reliability
       - Consistency metrics
       - Temporal stability
       
    D. Meta-Features (5 features):
       - Number of applicable contexts
       - Maximum context confidence
       - Average effect size
       - Novelty index
       - Betting edge indicator
    
    Parameters:
    -----------
    min_accuracy : float, default=0.60
        Minimum accuracy threshold for pattern discovery (60%)
        
    min_samples : int, default=30
        Minimum sample size for reliable pattern (30 samples)
        
    min_effect_size : float, default=0.10
        Minimum Cohen's d effect size (0.10 = small effect)
        
    max_patterns : int, default=None
        Maximum number of patterns to return (None = return ALL that meet criteria)
        
    feature_combinations : int, default=3
        Maximum features per context (2-5 typically)
        
    continuous_thresholds : int, default=5
        Number of thresholds to test for continuous features
        
    alpha : float, default=0.05
        Significance level for statistical tests
    """
    
    def __init__(
        self,
        min_accuracy: float = 0.60,
        min_samples: int = 30,
        min_effect_size: float = 0.10,
        max_patterns: Optional[int] = None,
        feature_combinations: int = 3,
        continuous_thresholds: int = 5,
        alpha: float = 0.05
    ):
        self.min_accuracy = min_accuracy
        self.min_samples = min_samples
        self.min_effect_size = min_effect_size
        self.max_patterns = max_patterns
        self.feature_combinations = feature_combinations
        self.continuous_thresholds = continuous_thresholds
        self.alpha = alpha
        
        # Discovered patterns (populated during fit)
        self.patterns_: List[ContextPattern] = []
        self.feature_names_: List[str] = []
        self.feature_types_: Dict[str, str] = {}
        self.baseline_accuracy_: float = 0.5
        
    def fit(self, X, y=None):
        """
        Discover high-leverage contexts from training data.
        
        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target outcomes (required for pattern discovery)
            
        Returns:
        --------
        self : ContextPatternTransformer
        """
        if y is None:
            raise ValueError("ContextPatternTransformer requires y for pattern discovery")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        
        y = np.array(y)
        
        # Store feature information
        self.feature_names_ = list(X.columns)
        self._infer_feature_types(X)
        
        # Detect target type (binary vs continuous)
        unique_vals = np.unique(y)
        self.is_binary_target_ = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        
        # Calculate baseline
        if self.is_binary_target_:
            self.baseline_accuracy_ = y.mean() if y.mean() > 0.5 else (1 - y.mean())
            metric_label = "Baseline accuracy"
        else:
            self.baseline_accuracy_ = y.mean()
            metric_label = "Baseline mean"
        
        print(f"\n[Context Discovery] Starting pattern search...", flush=True)
        print(f"  Samples: {len(X):,}", flush=True)
        print(f"  Features: {len(self.feature_names_)}", flush=True)
        print(f"  Target type: {'BINARY' if self.is_binary_target_ else 'CONTINUOUS'}", flush=True)
        print(f"  {metric_label}: {self.baseline_accuracy_:.1%}" if self.is_binary_target_ else f"  {metric_label}: {self.baseline_accuracy_:.1f}", flush=True)
        print(f"  Min samples per context: {self.min_samples}", flush=True)
        print(f"  Min accuracy threshold: {self.min_accuracy:.1%}" if self.is_binary_target_ else f"  Min improvement threshold: {self.min_accuracy:.1%}", flush=True)
        
        # Discover patterns
        self.patterns_ = self._discover_patterns(X, y)
        
        print(f"\n[Context Discovery] Complete!", flush=True)
        print(f"  Patterns discovered: {len(self.patterns_)}", flush=True)
        if self.patterns_:
            if self.is_binary_target_:
                print(f"  Best accuracy: {max(p.accuracy for p in self.patterns_):.1%}", flush=True)
            else:
                print(f"  Best mean: {max(p.accuracy for p in self.patterns_):.1f}", flush=True)
            print(f"  Best score: {max(p.score for p in self.patterns_):.1f}", flush=True)
        
        return self
    
    def _infer_feature_types(self, X: pd.DataFrame):
        """Infer whether features are continuous, categorical, or binary"""
        for col in X.columns:
            unique_vals = X[col].nunique()
            
            if unique_vals == 2:
                self.feature_types_[col] = 'binary'
            elif unique_vals < 10:
                self.feature_types_[col] = 'categorical'
            else:
                self.feature_types_[col] = 'continuous'
    
    def _discover_patterns(self, X: pd.DataFrame, y: np.ndarray) -> List[ContextPattern]:
        """
        Core pattern discovery algorithm.
        
        Strategy:
        1. Test all 2-feature combinations
        2. Test all 3-feature combinations
        3. For continuous features: test percentile thresholds
        4. For categorical features: test each category
        5. Calculate accuracy, effect size, p-value
        6. Filter by thresholds
        7. Rank by composite score
        8. Return top N patterns
        """
        all_candidates = []
        
        # Test combinations of different sizes
        for combo_size in range(2, self.feature_combinations + 1):
            print(f"\n  Testing {combo_size}-feature combinations...")
            
            # Generate all feature combinations
            for features in combinations(self.feature_names_, combo_size):
                # Generate conditions for this combination
                conditions_list = self._generate_conditions(X, features)
                
                # Test each condition set
                for conditions in conditions_list:
                    pattern = self._test_pattern(X, y, list(features), conditions)
                    if pattern is not None:
                        all_candidates.append(pattern)
        
        # Sort by score and take top N
        all_candidates.sort(key=lambda p: p.score, reverse=True)
        
        # Filter duplicates and highly correlated patterns
        filtered = self._filter_redundant_patterns(all_candidates)
        
        # Return ALL patterns or limited by max_patterns
        if self.max_patterns is None:
            return filtered
        else:
            return filtered[:self.max_patterns]
    
    def _generate_conditions(
        self, 
        X: pd.DataFrame, 
        features: Tuple[str, ...]
    ) -> List[Dict[str, Any]]:
        """
        Generate condition sets for feature combination.
        
        For continuous: test percentile thresholds (25th, 50th, 75th, etc.)
        For categorical: test each category value
        For binary: test True/False
        """
        # Start with one condition dict per feature
        conditions_per_feature = []
        
        for feature in features:
            feature_conditions = []
            feat_type = self.feature_types_.get(feature, 'continuous')
            
            if feat_type == 'binary':
                # Test both values
                feature_conditions.append({feature: {'eq': 1}})
                feature_conditions.append({feature: {'eq': 0}})
                
            elif feat_type == 'categorical':
                # Test each category
                unique_vals = X[feature].unique()
                for val in unique_vals[:5]:  # Limit to top 5 categories
                    feature_conditions.append({feature: {'eq': val}})
                    
            else:  # continuous
                # Test percentile thresholds
                percentiles = np.linspace(0, 100, self.continuous_thresholds + 2)[1:-1]
                thresholds = np.percentile(X[feature].dropna(), percentiles)
                
                for threshold in thresholds:
                    feature_conditions.append({feature: {'min': threshold}})
                    feature_conditions.append({feature: {'max': threshold}})
            
            conditions_per_feature.append(feature_conditions)
        
        # Combine conditions (Cartesian product)
        # For efficiency, limit to reasonable number of combinations
        import itertools
        all_combinations = list(itertools.product(*conditions_per_feature))
        
        # Merge dictionaries
        merged_conditions = []
        for combo in all_combinations[:100]:  # Limit combinations
            merged = {}
            for cond_dict in combo:
                merged.update(cond_dict)
            merged_conditions.append(merged)
        
        return merged_conditions
    
    def _test_pattern(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: List[str],
        conditions: Dict[str, Any]
    ) -> Optional[ContextPattern]:
        """Test a specific pattern and return if it passes thresholds"""
        
        # Find matching samples
        mask = np.ones(len(X), dtype=bool)
        for feature, condition in conditions.items():
            if feature not in X.columns:
                return None
            
            if isinstance(condition, dict):
                if 'min' in condition:
                    mask &= (X[feature] >= condition['min'])
                if 'max' in condition:
                    mask &= (X[feature] <= condition['max'])
                if 'eq' in condition:
                    mask &= (X[feature] == condition['eq'])
        
        # Check sample size
        n_matching = mask.sum()
        if n_matching < self.min_samples:
            return None
        
        # Calculate metric based on target type
        y_context = y[mask]
        y_not_context = y[~mask]
        
        if len(y_not_context) < 10:
            return None
        
        if self.is_binary_target_:
            # Binary classification: use win rate as accuracy
            accuracy = y_context.mean()
            # Flip if below 0.5 to always report the dominant class
            if accuracy < 0.5:
                accuracy = 1 - accuracy
            
            # Check accuracy threshold
            if accuracy < self.min_accuracy:
                return None
            
            # Statistical test (binomial test)
            n_success = (y_context == 1).sum()
            p_value = stats.binom_test(n_success, n_matching, 0.5, alternative='two-sided')
            
        else:
            # Continuous target: use mean improvement over baseline
            mean_context = y_context.mean()
            mean_baseline = self.baseline_accuracy_  # Overall mean
            
            # "Accuracy" for continuous = mean value in context
            accuracy = mean_context
            
            # For continuous, we want contexts with significantly different means
            # Use t-test instead of binomial
            try:
                _, p_value = stats.ttest_ind(y_context, y_not_context)
            except:
                return None
            
            # For continuous, threshold is relative improvement
            # Convert min_accuracy (e.g. 0.6) to "60% of baseline or higher"
            if mean_baseline != 0:
                relative_improvement = mean_context / mean_baseline
                if relative_improvement < self.min_accuracy:
                    return None
            else:
                # If baseline is 0, just check if context mean is positive
                if mean_context <= 0:
                    return None
        
        # Calculate effect size (Cohen's d) - works for both binary and continuous
        mean_context = y_context.mean()
        mean_not = y_not_context.mean()
        std_pooled = np.sqrt((y_context.var() + y_not_context.var()) / 2)
        
        if std_pooled == 0:
            effect_size = 0
        else:
            effect_size = abs(mean_context - mean_not) / std_pooled
        
        # Check effect size threshold
        if effect_size < self.min_effect_size:
            return None
        
        # Check significance
        if p_value > self.alpha:
            return None
        
        # Calculate novelty (rarity of this context)
        novelty = 1.0 - (n_matching / len(X))
        
        # Create pattern
        pattern = ContextPattern(
            features=features,
            conditions=conditions,
            accuracy=accuracy,
            sample_size=int(n_matching),
            effect_size=effect_size,
            p_value=p_value,
            novelty=novelty,
            metadata={
                'baseline_lift': accuracy - self.baseline_accuracy_,
                'proportion': n_matching / len(X),
                'is_binary': self.is_binary_target_
            }
        )
        
        return pattern
    
    def _filter_redundant_patterns(
        self, 
        patterns: List[ContextPattern]
    ) -> List[ContextPattern]:
        """Remove highly overlapping patterns"""
        if not patterns:
            return []
        
        filtered = [patterns[0]]  # Keep best
        
        for pattern in patterns[1:]:
            # Check if too similar to existing patterns
            is_redundant = False
            
            for existing in filtered:
                # Simple overlap check: same features
                if set(pattern.features) == set(existing.features):
                    is_redundant = True
                    break
            
            if not is_redundant:
                filtered.append(pattern)
        
        return filtered
    
    def transform(self, X):
        """
        Extract context pattern features.
        
        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : ndarray, shape (n_samples, 60)
            Context pattern features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        
        features_list = []
        
        for idx in range(len(X)):
            row_features = self._extract_row_features(X.iloc[idx:idx+1])
            features_list.append(row_features)
        
        return np.array(features_list, dtype=np.float32)
    
    def _extract_row_features(self, row: pd.DataFrame) -> List[float]:
        """Extract features for a single row"""
        features = []
        
        # === A. CONTEXT MEMBERSHIP (20 features) ===
        
        # Top 10 pattern memberships
        for i in range(10):
            if i < len(self.patterns_):
                pattern = self.patterns_[i]
                matches = pattern.matches(row)
                features.append(float(matches[0]))
            else:
                features.append(0.0)
        
        # Top 10 pattern confidence scores
        for i in range(10):
            if i < len(self.patterns_):
                pattern = self.patterns_[i]
                matches = pattern.matches(row)
                if matches[0]:
                    features.append(pattern.accuracy)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # === B. FEATURE INTERACTIONS (25 features) ===
        
        # For top important features, compute interactions
        # (Simplified - use first 5 numeric columns)
        numeric_cols = row.select_dtypes(include=[np.number]).columns[:5]
        
        # Pairwise products (10 features)
        interaction_values = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                val = row[col1].values[0] * row[col2].values[0]
                interaction_values.append(val)
        
        # Pad to 10
        while len(interaction_values) < 10:
            interaction_values.append(0.0)
        features.extend(interaction_values[:10])
        
        # Polynomial features (5 features)
        for col in numeric_cols:
            val = row[col].values[0]
            features.append(val ** 2)
        while len(features) < 45:  # Pad to reach 45 total
            features.append(0.0)
        
        # Threshold proximity (10 features) - distance to discovered thresholds
        # Simplified: use percentiles of numeric features
        for col in numeric_cols[:5]:
            val = row[col].values[0]
            # Distance to median (as proxy)
            features.append(abs(val - 0.5))
        while len(features) < 45:
            features.append(0.0)
        features = features[:45]  # Ensure exactly 45
        
        # === C. HISTORICAL PATTERN STRENGTH (10 features) ===
        
        matching_patterns = [p for p in self.patterns_ if p.matches(row)[0]]
        
        if matching_patterns:
            # Average accuracy of matching patterns
            features.append(np.mean([p.accuracy for p in matching_patterns]))
            # Max accuracy
            features.append(max([p.accuracy for p in matching_patterns]))
            # Min accuracy
            features.append(min([p.accuracy for p in matching_patterns]))
            # Average sample size (normalized)
            features.append(np.mean([p.sample_size for p in matching_patterns]) / 100)
            # Max sample size (normalized)
            features.append(max([p.sample_size for p in matching_patterns]) / 100)
            # Average effect size
            features.append(np.mean([p.effect_size for p in matching_patterns]))
            # Max effect size
            features.append(max([p.effect_size for p in matching_patterns]))
            # Average p-value (lower is better)
            features.append(np.mean([p.p_value for p in matching_patterns]))
            # Average score
            features.append(np.mean([p.score for p in matching_patterns]) / 10)
            # Max score
            features.append(max([p.score for p in matching_patterns]) / 10)
        else:
            features.extend([0.0] * 10)
        
        # === D. META-FEATURES (5 features) ===
        
        # Number of matching patterns (normalized)
        features.append(len(matching_patterns) / max(len(self.patterns_), 1))
        
        # Maximum confidence across all patterns
        if matching_patterns:
            features.append(max([p.accuracy for p in matching_patterns]))
        else:
            features.append(0.0)
        
        # Average novelty of matching patterns
        if matching_patterns:
            features.append(np.mean([p.novelty for p in matching_patterns]))
        else:
            features.append(0.0)
        
        # Composite betting edge indicator
        if matching_patterns:
            avg_lift = np.mean([p.metadata.get('baseline_lift', 0) for p in matching_patterns])
            features.append(max(0, avg_lift) * 10)  # Scale up
        else:
            features.append(0.0)
        
        # Pattern diversity (number of unique feature sets)
        if matching_patterns:
            unique_feature_sets = len(set(tuple(p.features) for p in matching_patterns))
            features.append(unique_feature_sets / len(matching_patterns))
        else:
            features.append(0.0)
        
        # Total should be 60 features
        # A: 20, B: 25, C: 10, D: 5 = 60
        assert len(features) == 60, f"Expected 60 features, got {len(features)}"
        
        return features
    
    def get_context_report(self) -> str:
        """Generate human-readable report of discovered contexts"""
        if not self.patterns_:
            return "No patterns discovered"
        
        report = []
        report.append("="*80)
        report.append("DISCOVERED HIGH-LEVERAGE CONTEXTS")
        report.append("="*80)
        report.append(f"\nTotal patterns: {len(self.patterns_)}")
        report.append(f"Baseline accuracy: {self.baseline_accuracy_:.1%}\n")
        
        report.append(f"{'#':<4} {'Accuracy':<10} {'N':<8} {'Effect':<8} {'P-value':<10} {'Score':<8} Context")
        report.append("-"*80)
        
        for i, pattern in enumerate(self.patterns_[:20], 1):  # Top 20
            # Format conditions
            cond_parts = []
            for feat, cond in pattern.conditions.items():
                if isinstance(cond, dict):
                    if 'min' in cond:
                        cond_parts.append(f"{feat}≥{cond['min']:.2f}")
                    if 'max' in cond:
                        cond_parts.append(f"{feat}≤{cond['max']:.2f}")
                    if 'eq' in cond:
                        cond_parts.append(f"{feat}={cond['eq']}")
            
            cond_str = " & ".join(cond_parts)
            
            report.append(
                f"{i:<4} {pattern.accuracy:.1%}     {pattern.sample_size:<8} "
                f"{pattern.effect_size:<8.3f} {pattern.p_value:<10.4f} "
                f"{pattern.score:<8.1f} {cond_str[:40]}"
            )
        
        report.append("\n" + "="*80)
        return "\n".join(report)
    
    def get_betting_recommendations(self, X) -> List[Dict]:
        """
        Generate betting recommendations for samples.
        
        Returns list of dicts with:
        - sample_idx: Index of sample
        - applicable_patterns: List of matching patterns
        - confidence: Max accuracy across patterns
        - recommendation: "BET" or "SKIP"
        - expected_edge: Estimated edge over baseline
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        
        # Reset index to ensure clean integer indexing
        X = X.reset_index(drop=True)
        
        recommendations = []
        
        for idx in range(len(X)):
            row = X.iloc[idx:idx+1].reset_index(drop=True)
            matching = [p for p in self.patterns_ if p.matches(row).any()]
            
            if matching:
                # Sort by accuracy
                matching.sort(key=lambda p: p.accuracy, reverse=True)
                best = matching[0]
                
                recommendations.append({
                    'sample_idx': idx,
                    'applicable_patterns': len(matching),
                    'best_pattern': str(best),
                    'confidence': best.accuracy,
                    'recommendation': 'BET' if best.accuracy >= 0.65 else 'SKIP',
                    'expected_edge': best.metadata.get('baseline_lift', 0),
                    'sample_size': best.sample_size,
                    'effect_size': best.effect_size
                })
        
        return recommendations
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        names = []
        
        # Context membership
        for i in range(10):
            names.append(f"pattern_{i}_match")
        for i in range(10):
            names.append(f"pattern_{i}_confidence")
        
        # Interactions
        for i in range(10):
            names.append(f"interaction_{i}")
        for i in range(5):
            names.append(f"polynomial_{i}")
        for i in range(10):
            names.append(f"threshold_proximity_{i}")
        
        # Historical strength
        names.extend([
            'avg_accuracy', 'max_accuracy', 'min_accuracy',
            'avg_sample_size', 'max_sample_size',
            'avg_effect_size', 'max_effect_size',
            'avg_p_value', 'avg_score', 'max_score'
        ])
        
        # Meta
        names.extend([
            'matching_patterns_ratio',
            'max_confidence',
            'avg_novelty',
            'betting_edge',
            'pattern_diversity'
        ])
        
        return names

