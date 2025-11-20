"""
Bridge Calculator

Calculates Д (the bridge) properly following theoretical framework.

Supports multiple formulas:
1. Standard: Д = r_narrative - r_baseline
2. Framework: Д = п × |r| × κ
3. Three-force: Д = ة - θ - λ (regular) or Д = ة + θ - λ (prestige)

Auto-selects formula based on available features.

NOT determinism - measures advantage, preserves free will.
Universal hypothesis: Д > 0.10 (narrative provides meaningful edge)
"""

import numpy as np
from typing import Dict, Optional, Union
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler


class BridgeCalculator:
    """
    Calculate Д (the bridge) - narrative advantage over baseline.
    
    Implements:
    Step 1: r_baseline = correlation(objective_features, ❊)
    Step 2: r_narrative = correlation(ю, ❊)
    Step 3: Д = r_narrative - r_baseline
    
    Universal Hypothesis: Д > 0.10 (preserves free will, shows meaningful edge)
    """
    
    def __init__(self):
        """Initialize calculator"""
        pass
    
    def calculate_D(
        self,
        story_quality: np.ndarray,
        outcomes: np.ndarray,
        baseline_features: Optional[np.ndarray] = None,
        domain_hint: Optional[str] = None,
        # Framework formula parameters
        narrativity: Optional[float] = None,
        coupling: Optional[Union[float, np.ndarray]] = None,
        # Three-force parameters
        nominative_gravity: Optional[np.ndarray] = None,
        awareness_resistance: Optional[np.ndarray] = None,
        fundamental_constraints: Optional[np.ndarray] = None,
        is_prestige: bool = False
    ) -> Dict[str, float]:
        """
        Calculate Д (the bridge) using best available formula.
        
        Auto-selects formula based on available features:
        1. Three-force (if all forces available): Д = ة - θ - λ
        2. Framework (if п, κ available): Д = п × |r| × κ
        3. Standard (fallback): Д = r_narrative - r_baseline
        
        Parameters
        ----------
        story_quality : ndarray
            Story quality scores (ю) (n_samples,)
        outcomes : ndarray
            Outcomes (❊) (n_samples,)
        baseline_features : ndarray, optional
            Objective features for baseline (genre, stats, etc.)
        domain_hint : str, optional
            Domain type for baseline estimation
        narrativity : float, optional
            Domain narrativity (п) for framework formula
        coupling : float or ndarray, optional
            Coupling strength (κ) - scalar or per-instance
        nominative_gravity : ndarray, optional
            Nominative gravity (ة) per instance
        awareness_resistance : ndarray, optional
            Awareness resistance (θ) per instance
        fundamental_constraints : ndarray, optional
            Fundamental constraints (λ) per instance
        is_prestige : bool, default=False
            Whether domain is prestige type (awareness amplifies)
            
        Returns
        -------
        results : dict
            Complete bridge analysis with all metrics
        """
        # Auto-select formula based on available features
        if self._can_use_three_force(nominative_gravity, awareness_resistance, fundamental_constraints):
            return self._calculate_three_force(
                nominative_gravity, awareness_resistance, fundamental_constraints, is_prestige
            )
        elif self._can_use_framework(narrativity, coupling, story_quality, outcomes):
            return self._calculate_framework(narrativity, coupling, story_quality, outcomes)
        else:
            return self._calculate_standard(story_quality, outcomes, baseline_features, domain_hint)
    
    def _can_use_three_force(
        self,
        nominative_gravity: Optional[np.ndarray],
        awareness_resistance: Optional[np.ndarray],
        fundamental_constraints: Optional[np.ndarray]
    ) -> bool:
        """Check if three-force model can be used."""
        return (
            nominative_gravity is not None and
            awareness_resistance is not None and
            fundamental_constraints is not None and
            len(nominative_gravity) > 0 and
            len(awareness_resistance) > 0 and
            len(fundamental_constraints) > 0
        )
    
    def _can_use_framework(
        self,
        narrativity: Optional[float],
        coupling: Optional[Union[float, np.ndarray]],
        story_quality: np.ndarray,
        outcomes: np.ndarray
    ) -> bool:
        """Check if framework formula can be used."""
        return (
            narrativity is not None and
            coupling is not None and
            len(story_quality) > 0 and
            len(outcomes) > 0
        )
    
    def _calculate_three_force(
        self,
        nominative_gravity: np.ndarray,
        awareness_resistance: np.ndarray,
        fundamental_constraints: np.ndarray,
        is_prestige: bool = False
    ) -> Dict[str, float]:
        """
        Calculate Д using three-force model.
        
        Formula:
        - Regular: Д = ة - θ - λ
        - Prestige: Д = ة + θ - λ
        """
        # Ensure arrays are same length
        min_len = min(len(nominative_gravity), len(awareness_resistance), len(fundamental_constraints))
        ة = nominative_gravity[:min_len]
        θ = awareness_resistance[:min_len]
        λ = fundamental_constraints[:min_len]
        
        # Calculate Д per instance
        if is_prestige:
            Д_per_instance = ة + θ - λ  # Awareness amplifies
            equation_used = 'ة + θ - λ'
        else:
            Д_per_instance = ة - θ - λ  # Awareness suppresses
            equation_used = 'ة - θ - λ'
        
        # Clamp to [0, 1]
        Д_per_instance = np.clip(Д_per_instance, 0, 1)
        
        # Aggregate to domain-level (mean)
        Д = float(np.mean(Д_per_instance))
        
        # Determine dominant force
        mean_ة = float(np.mean(ة))
        mean_θ = float(np.mean(θ))
        mean_λ = float(np.mean(λ))
        
        if mean_ة > mean_θ + mean_λ:
            dominant_force = 'ة_dominates'
        elif mean_θ > mean_ة:
            dominant_force = 'θ_dominates'
        elif mean_λ > mean_ة:
            dominant_force = 'λ_dominates'
        else:
            dominant_force = 'equilibrium'
        
        # Test threshold
        passes_threshold = Д > 0.10
        
        return {
            'formula_used': 'three_force',
            'equation': equation_used,
            'Д': Д,
            'nominative_gravity_mean': mean_ة,
            'awareness_resistance_mean': mean_θ,
            'fundamental_constraints_mean': mean_λ,
            'dominant_force': dominant_force,
            'is_prestige': is_prestige,
            'passes_threshold': bool(passes_threshold),
            'interpretation': self._interpret_D(Д),
            'effect_size': self._categorize_effect_size(Д),
            'Д_per_instance': Д_per_instance.tolist() if len(Д_per_instance) <= 100 else None  # Limit size
        }
    
    def _calculate_framework(
        self,
        narrativity: float,
        coupling: Union[float, np.ndarray],
        story_quality: np.ndarray,
        outcomes: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Д using framework formula: Д = п × |r| × κ
        """
        # Calculate correlation
        is_binary = (set(outcomes) == {0, 1} or outcomes.dtype == bool)
        
        if is_binary:
            r = self._calculate_auc(story_quality, outcomes)
            metric_name = 'AUC'
        else:
            r, _ = stats.pearsonr(story_quality, outcomes)
            metric_name = 'r (Pearson)'
        
        abs_r = abs(r)
        
        # Get coupling (scalar or mean of array)
        if isinstance(coupling, np.ndarray):
            κ = float(np.mean(coupling))
        else:
            κ = float(coupling)
        
        # Calculate Д
        Д = narrativity * abs_r * κ
        
        # Calculate efficiency
        efficiency = Д / narrativity if narrativity > 0 else 0.0
        
        # Test threshold
        passes_threshold = Д > 0.10
        passes_efficiency = efficiency > 0.5
        
        return {
            'formula_used': 'framework',
            'equation': 'п × |r| × κ',
            'Д': float(Д),
            'narrativity': float(narrativity),
            'r': float(r),
            'abs_r': float(abs_r),
            'coupling': float(κ),
            'efficiency': float(efficiency),
            'metric_type': metric_name,
            'passes_threshold': bool(passes_threshold),
            'passes_efficiency': bool(passes_efficiency),
            'interpretation': self._interpret_D(Д),
            'effect_size': self._categorize_effect_size(Д)
        }
    
    def _calculate_standard(
        self,
        story_quality: np.ndarray,
        outcomes: np.ndarray,
        baseline_features: Optional[np.ndarray] = None,
        domain_hint: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate Д using standard formula: Д = r_narrative - r_baseline
        """
        # Determine if binary or continuous outcome
        is_binary = (set(outcomes) == {0, 1} or outcomes.dtype == bool)
        
        # Step 1: Calculate r_narrative
        if is_binary:
            r_narrative = self._calculate_auc(story_quality, outcomes)
            metric_name = 'AUC'
        else:
            r_narrative, p_narrative = stats.pearsonr(story_quality, outcomes)
            metric_name = 'r (Pearson)'
        
        # Step 2: Calculate r_baseline
        if baseline_features is not None:
            r_baseline = self._calculate_baseline_actual(baseline_features, outcomes, is_binary)
            baseline_method = 'measured'
        else:
            r_baseline = self._estimate_baseline(outcomes, domain_hint, is_binary)
            baseline_method = 'estimated'
        
        # Step 3: Calculate Д
        Д = r_narrative - r_baseline
        
        # Step 4: Test universal hypothesis
        passes_threshold = Д > 0.10
        
        # Compile results
        results = {
            'formula_used': 'standard',
            'equation': 'r_narrative - r_baseline',
            'metric_type': metric_name,
            'r_narrative': float(r_narrative),
            'r_baseline': float(r_baseline),
            'baseline_method': baseline_method,
            'Д': float(Д),
            'passes_threshold': bool(passes_threshold),
            'interpretation': self._interpret_D(Д),
            'effect_size': self._categorize_effect_size(Д)
        }
        
        # Add p-values if available
        if not is_binary:
            _, p_narrative = stats.pearsonr(story_quality, outcomes)
            results['p_narrative'] = float(p_narrative)
            results['R2_narrative'] = float(r_narrative ** 2)
        
        return results
    
    def _calculate_auc(self, predictions, outcomes):
        """Calculate AUC for binary outcomes"""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(outcomes, predictions)
        except:
            # If error, use correlation as proxy
            r, _ = stats.pearsonr(predictions, outcomes)
            return (r + 1) / 2  # Map [-1, 1] to [0, 1]
    
    def _calculate_baseline_actual(
        self,
        baseline_features: np.ndarray,
        outcomes: np.ndarray,
        is_binary: bool
    ) -> float:
        """
        Calculate actual baseline from objective features.
        
        Parameters
        ----------
        baseline_features : ndarray
            Objective features (genre, year, stats, etc.)
        outcomes : ndarray
            Outcomes
        is_binary : bool
            Whether outcomes are binary
            
        Returns
        -------
        r_baseline : float
            Baseline performance
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(baseline_features)
        
        if is_binary:
            # Logistic regression baseline
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled, outcomes)
            predictions = model.predict_proba(X_scaled)[:, 1]
            r_baseline = self._calculate_auc(predictions, outcomes)
        else:
            # Ridge regression baseline
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, outcomes)
            predictions = model.predict(X_scaled)
            r_baseline, _ = stats.pearsonr(predictions, outcomes)
        
        return r_baseline
    
    def _estimate_baseline(
        self,
        outcomes: np.ndarray,
        domain_hint: Optional[str],
        is_binary: bool
    ) -> float:
        """
        Estimate baseline when objective features not provided.
        
        Based on domain type and outcome distribution.
        """
        if is_binary:
            # For binary: baseline depends on class balance and domain
            class_balance = outcomes.mean()
            
            # More imbalanced = harder baseline
            imbalance = abs(class_balance - 0.5)
            
            if domain_hint in ['movies', 'oscars', 'entertainment']:
                # Genre/year provide weak baseline
                return 0.55 + imbalance * 0.1
            elif domain_hint in ['sports', 'games']:
                # Stats provide strong baseline
                return 0.65 + imbalance * 0.15
            elif domain_hint in ['startups', 'profiles', 'personal']:
                # Demographics provide weak baseline
                return 0.52 + imbalance * 0.08
            else:
                # Generic estimate
                return 0.58
        
        else:
            # For continuous: baseline depends on domain
            if domain_hint in ['movies', 'entertainment']:
                return 0.20  # Genre + year
            elif domain_hint in ['sports', 'games']:
                return 0.40  # Historical stats
            elif domain_hint in ['startups', 'business']:
                return 0.15  # Funding amount, sector
            else:
                return 0.20  # Generic
    
    def _interpret_D(self, Д: float) -> str:
        """Interpret Д value in plain English"""
        if Д < 0.05:
            return f"Negligible narrative effect (Д={Д:.3f}) - stories barely matter, physics/stats dominate"
        elif Д < 0.10:
            return f"Weak narrative effect (Д={Д:.3f}) - stories provide small edge, below universal threshold"
        elif Д < 0.30:
            return f"Moderate narrative advantage (Д={Д:.3f}) - stories provide meaningful edge"
        elif Д < 0.50:
            return f"Strong narrative advantage (Д={Д:.3f}) - stories significantly determine outcomes"
        else:
            return f"Dominant narrative advantage (Д={Д:.3f}) - stories are primary determinant"
    
    def _categorize_effect_size(self, Д: float) -> str:
        """Categorize effect size"""
        if Д < 0.10:
            return "negligible"
        elif Д < 0.30:
            return "small"
        elif Д < 0.50:
            return "medium"
        else:
            return "large"
    
    def validate_hypothesis(self, Д: float) -> Dict[str, any]:
        """
        Test universal hypothesis: Д > 0.10
        
        Parameters
        ----------
        Д : float
            Calculated bridge value
            
        Returns
        -------
        validation : dict
            Hypothesis test results
        """
        threshold = 0.10
        passes = Д > threshold
        
        return {
            'hypothesis': f"Д > {threshold}",
            'measured_D': float(Д),
            'passes': bool(passes),
            'margin': float(Д - threshold),
            'conclusion': 'Narrative agency validated' if passes else 'Narrative agency weak/absent',
            'implication': self._explain_implication(passes, Д)
        }
    
    def _explain_implication(self, passes: bool, Д: float) -> str:
        """Explain what result means"""
        if passes:
            return (f"Better stories win in this domain. Narrative provides {Д:.1%} advantage "
                   f"beyond objective baseline. This preserves free will (not deterministic) "
                   f"while showing meaningful narrative agency.")
        else:
            return (f"Stories provide minimal advantage ({Д:.1%}) in this domain. "
                   f"Outcomes are primarily determined by objective factors (stats, physics, etc.). "
                   f"Narrative agency is weak or absent - domain is highly constrained.")

