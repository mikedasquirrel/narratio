"""
Blind Narratio Calculator

Discovers the emergent equilibrium ratio (Β) between deterministic
and free will forces.

Β = (deterministic_forces) / (free_will_forces)

Where:
- Deterministic forces: ة (nominative gravity) + λ (fundamental constraints)
- Free will forces: θ (awareness resistance) + agency

Key Properties:
- Domain-specific (cannot be predicted, only discovered)
- Stable in long run (short-term variance)
- May vary by instance complexity within domain
- Dual existence proof: BOTH determinism AND free will operate

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.story_instance import StoryInstance


class BlindNarratioCalculator:
    """
    Calculate and analyze Blind Narratio (Β) ratios.
    
    Discovers the equilibrium between deterministic and free will forces
    for domains and individual instances.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.domain_betas: Dict[str, Dict] = {}
        self.instance_betas: Dict[str, float] = {}
        
    def calculate_domain_blind_narratio(
        self,
        instances: List[StoryInstance],
        domain_name: str,
        nominative_weight: float = 0.5,
        constraint_weight: float = 0.5,
        awareness_weight: float = 0.6,
        agency_weight: float = 0.4
    ) -> Dict[str, Any]:
        """
        Calculate Β for entire domain.
        
        Parameters
        ----------
        instances : list of StoryInstance
            All instances in the domain
        domain_name : str
            Domain name
        nominative_weight : float
            Weight of nominative gravity in deterministic forces
        constraint_weight : float
            Weight of fundamental constraints in deterministic forces
        awareness_weight : float
            Weight of awareness resistance in free will forces
        agency_weight : float
            Weight of agency in free will forces
        
        Returns
        -------
        dict
            {
                'Β': float,  # The ratio
                'stability': float,  # Variance across instances
                'deterministic_component': float,  # ة + λ
                'free_will_component': float,  # θ + agency
                'equilibrium_quality': float,  # How well it predicts
                'variance_by_context': dict,  # Does Β vary by complexity?
                'n_instances': int
            }
        """
        if not instances:
            return {'error': 'No instances provided'}
        
        # Extract force components for each instance
        deterministic_forces = []
        free_will_forces = []
        instance_betas = []
        complexities = []
        
        for instance in instances:
            # Deterministic forces: ة (nominative) + λ (constraints)
            nominative = self._estimate_nominative_force(instance)
            constraints = self._estimate_constraint_force(instance)
            deterministic = (nominative_weight * nominative + 
                           constraint_weight * constraints)
            deterministic_forces.append(deterministic)
            
            # Free will forces: θ (awareness) + agency
            awareness = instance.theta_resistance if instance.theta_resistance else 0.5
            agency = self._estimate_agency(instance)
            free_will = (awareness_weight * awareness + 
                        agency_weight * agency)
            free_will_forces.append(free_will)
            
            # Calculate instance Β
            if free_will > 0:
                beta = deterministic / free_will
                instance_betas.append(beta)
                instance.blind_narratio = beta
            else:
                instance_betas.append(np.inf)
                instance.blind_narratio = np.inf
            
            # Track complexity if available
            if instance.complexity_factors:
                complexity = np.mean(list(instance.complexity_factors.values()))
                complexities.append(complexity)
        
        # Domain-level Β
        deterministic_forces = np.array(deterministic_forces)
        free_will_forces = np.array(free_will_forces)
        instance_betas = np.array(instance_betas)
        
        # Filter out infinities for statistics
        finite_betas = instance_betas[np.isfinite(instance_betas)]
        
        if len(finite_betas) == 0:
            domain_beta = np.inf
            stability = 0.0
        else:
            domain_beta = np.mean(finite_betas)
            stability = 1.0 - (np.std(finite_betas) / (domain_beta + 1e-8))
        
        # Analyze variance by complexity
        variance_by_context = {}
        if complexities and len(complexities) == len(finite_betas):
            complexities = np.array(complexities)
            
            # Split by complexity tertiles
            low_complexity_mask = complexities < np.percentile(complexities, 33)
            mid_complexity_mask = (complexities >= np.percentile(complexities, 33)) & \
                                 (complexities < np.percentile(complexities, 67))
            high_complexity_mask = complexities >= np.percentile(complexities, 67)
            
            if low_complexity_mask.sum() > 0:
                variance_by_context['low_complexity'] = {
                    'beta_mean': float(np.mean(finite_betas[low_complexity_mask])),
                    'n': int(low_complexity_mask.sum())
                }
            
            if mid_complexity_mask.sum() > 0:
                variance_by_context['mid_complexity'] = {
                    'beta_mean': float(np.mean(finite_betas[mid_complexity_mask])),
                    'n': int(mid_complexity_mask.sum())
                }
            
            if high_complexity_mask.sum() > 0:
                variance_by_context['high_complexity'] = {
                    'beta_mean': float(np.mean(finite_betas[high_complexity_mask])),
                    'n': int(high_complexity_mask.sum())
                }
        
        # Test predictive power
        equilibrium_quality = self._test_predictive_power(
            finite_betas,
            [i.outcome for i in instances if i.outcome is not None]
        )
        
        result = {
            'domain': domain_name,
            'Β': float(domain_beta),
            'stability': float(stability),
            'deterministic_component_mean': float(np.mean(deterministic_forces)),
            'free_will_component_mean': float(np.mean(free_will_forces)),
            'equilibrium_quality': float(equilibrium_quality),
            'variance_by_context': variance_by_context,
            'n_instances': len(instances),
            'n_finite_betas': len(finite_betas),
            'beta_min': float(np.min(finite_betas)) if len(finite_betas) > 0 else None,
            'beta_max': float(np.max(finite_betas)) if len(finite_betas) > 0 else None,
            'beta_std': float(np.std(finite_betas)) if len(finite_betas) > 0 else None,
            'calculated_at': datetime.now().isoformat()
        }
        
        # Store domain result
        self.domain_betas[domain_name] = result
        
        return result
    
    def calculate_instance_blind_narratio(
        self,
        instance: StoryInstance,
        context: Optional[Dict] = None
    ) -> float:
        """
        Calculate Β for specific instance.
        
        Tests: Does Β vary within domain by instance complexity?
        
        Parameters
        ----------
        instance : StoryInstance
            Instance to analyze
        context : dict, optional
            Additional context for calculation
        
        Returns
        -------
        float
            Instance-specific Β
        """
        # Deterministic forces
        nominative = self._estimate_nominative_force(instance)
        constraints = self._estimate_constraint_force(instance)
        deterministic = 0.5 * nominative + 0.5 * constraints
        
        # Free will forces
        awareness = instance.theta_resistance if instance.theta_resistance else 0.5
        agency = self._estimate_agency(instance)
        free_will = 0.6 * awareness + 0.4 * agency
        
        # Calculate Β
        if free_will > 0:
            beta = deterministic / free_will
        else:
            beta = np.inf
        
        # Store on instance
        instance.blind_narratio = beta
        self.instance_betas[instance.instance_id] = beta
        
        return beta
    
    def test_universal_blind_narratio(
        self,
        all_domains: Dict[str, List[StoryInstance]]
    ) -> Dict[str, Any]:
        """
        Test: Is there a UNIVERSAL Β across all domains?
        
        Parameters
        ----------
        all_domains : dict
            {domain_name: [instances]}
        
        Returns
        -------
        dict
            {
                'universal_beta_exists': bool,
                'universal_beta': float or None,
                'cross_domain_variance': float,
                'domain_betas': dict,
                'conclusion': str
            }
        """
        domain_betas = []
        domain_results = {}
        
        for domain_name, instances in all_domains.items():
            result = self.calculate_domain_blind_narratio(instances, domain_name)
            if not np.isinf(result['Β']):
                domain_betas.append(result['Β'])
                domain_results[domain_name] = result['Β']
        
        if len(domain_betas) < 2:
            return {
                'universal_beta_exists': False,
                'error': 'Insufficient domains for universal test'
            }
        
        domain_betas = np.array(domain_betas)
        
        # Test for universal Β
        mean_beta = np.mean(domain_betas)
        std_beta = np.std(domain_betas)
        cv = std_beta / mean_beta if mean_beta > 0 else np.inf
        
        # Universal Β exists if coefficient of variation < 0.3
        universal_exists = cv < 0.3
        
        return {
            'universal_beta_exists': bool(universal_exists),
            'universal_beta': float(mean_beta) if universal_exists else None,
            'cross_domain_variance': float(std_beta),
            'coefficient_of_variation': float(cv),
            'n_domains': len(domain_betas),
            'domain_betas': domain_results,
            'conclusion': (
                f"Universal Β exists: {mean_beta:.3f} (CV={cv:.3f})" if universal_exists
                else f"Β is domain-specific (CV={cv:.3f} > 0.30)"
            )
        }
    
    def _estimate_nominative_force(self, instance: StoryInstance) -> float:
        """
        Estimate nominative gravity (ة) for instance.
        
        Uses:
        - Number of proper nouns
        - Nominative richness features
        - Name-based gravity calculations
        """
        if instance.genome_nominative is not None:
            # Use nominative component of genome
            return np.mean(instance.genome_nominative)
        elif 'nominative' in instance.features_all:
            return np.mean(instance.features_all['nominative'])
        else:
            # Default estimate based on domain
            return 0.5
    
    def _estimate_constraint_force(self, instance: StoryInstance) -> float:
        """
        Estimate fundamental constraints (λ) for instance.
        
        Uses:
        - Training requirements
        - Physical barriers
        - Economic barriers
        """
        # Check if we have constraint features
        if 'fundamental_constraints' in instance.features_all:
            return np.mean(instance.features_all['fundamental_constraints'])
        
        # Domain-based estimate
        domain = instance.domain
        
        # High constraint domains
        high_constraint_domains = ['nba', 'nfl', 'nhl', 'tennis', 'golf', 'boxing', 'aviation']
        if domain in high_constraint_domains:
            return 0.7
        
        # Low constraint domains
        low_constraint_domains = ['novels', 'movies', 'startups', 'music']
        if domain in low_constraint_domains:
            return 0.3
        
        # Default
        return 0.5
    
    def _estimate_agency(self, instance: StoryInstance) -> float:
        """
        Estimate agency/free will for instance.
        
        Factors:
        - Individual vs collective decision
        - Direct control over outcome
        - Number of actors
        """
        # Individual agency domains
        individual_domains = ['golf', 'tennis', 'chess', 'boxing']
        if instance.domain in individual_domains:
            return 0.8
        
        # No agency domains
        no_agency_domains = ['hurricanes', 'oscars', 'wwe']
        if instance.domain in no_agency_domains:
            return 0.2
        
        # Collective agency
        collective_domains = ['nba', 'nfl', 'startups']
        if instance.domain in collective_domains:
            return 0.5
        
        # Default moderate agency
        return 0.6
    
    def _test_predictive_power(
        self,
        betas: np.ndarray,
        outcomes: List[float]
    ) -> float:
        """
        Test how well Β predicts outcomes.
        
        Returns correlation coefficient.
        """
        if len(betas) == 0 or len(outcomes) == 0:
            return 0.0
        
        # Align lengths
        min_len = min(len(betas), len(outcomes))
        betas = betas[:min_len]
        outcomes = np.array(outcomes[:min_len])
        
        # Remove NaNs
        valid_mask = ~np.isnan(betas) & ~np.isnan(outcomes)
        betas = betas[valid_mask]
        outcomes = outcomes[valid_mask]
        
        if len(betas) < 2:
            return 0.0
        
        # Correlation
        try:
            corr = np.corrcoef(betas, outcomes)[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def export_results(self, output_path: str):
        """
        Export all calculated Β values to JSON.
        
        Parameters
        ----------
        output_path : str
            Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'domain_betas': self.domain_betas,
            'n_instances_analyzed': len(self.instance_betas),
            'calculated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Exported Blind Narratio results to {output_path}")
    
    def summarize_results(self) -> str:
        """
        Generate summary report of all calculated Β values.
        
        Returns
        -------
        str
            Formatted summary report
        """
        if not self.domain_betas:
            return "No Blind Narratio values calculated yet."
        
        report = ["=" * 60]
        report.append("BLIND NARRATIO (Β) ANALYSIS SUMMARY")
        report.append("=" * 60)
        report.append("")
        report.append(f"Total domains analyzed: {len(self.domain_betas)}")
        report.append(f"Total instances analyzed: {len(self.instance_betas)}")
        report.append("")
        report.append("DOMAIN-LEVEL Β VALUES:")
        report.append("-" * 60)
        
        # Sort by Β value
        sorted_domains = sorted(
            self.domain_betas.items(),
            key=lambda x: x[1]['Β'] if not np.isinf(x[1]['Β']) else 999
        )
        
        for domain, result in sorted_domains:
            beta = result['Β']
            stability = result['stability']
            n = result['n_instances']
            
            beta_str = f"{beta:.3f}" if not np.isinf(beta) else "∞"
            report.append(f"{domain:20s} Β={beta_str:8s} (stability={stability:.3f}, n={n})")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

