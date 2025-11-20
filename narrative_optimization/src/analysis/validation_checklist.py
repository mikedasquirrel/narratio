"""
Narrative Law Validation Checklist

Implements "presume and prove" methodology:
1. State hypothesis (presumption)
2. Test empirically  
3. Validate against threshold
4. Report pass/fail honestly

Ensures scientific rigor - no domain assumptions.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ValidationResult:
    """Results of narrative law validation for a domain"""
    domain_name: str
    narrativity: float
    correlation: float
    coupling: float
    narrative_agency: float
    efficiency: float
    passes: bool
    interpretation: str
    transformer_rationale: Dict[str, str]
    
    def __str__(self):
        status = "✓ PASS" if self.passes else "❌ FAIL"
        return f"""
{'='*80}
NARRATIVE LAW VALIDATION: {self.domain_name}
{'='*80}

HYPOTHESIS (Presumption):
  Narrative laws should apply (Д/п > 0.5)

DOMAIN CHARACTERISTICS:
  Narrativity (п): {self.narrativity:.3f}
  Expected Coupling (κ): {self.coupling:.3f}
  
EMPIRICAL RESULTS:
  Measured Correlation (r): {self.correlation:.3f}
  Narrative Agency (Д): {self.narrative_agency:.3f}
  Efficiency: Д/п = {self.efficiency:.3f}
  
VALIDATION RESULT:
  {status} - Efficiency threshold: {self.efficiency:.3f} {'>' if self.passes else '<'} 0.5
  
INTERPRETATION:
  {self.interpretation}

{'='*80}
"""


class NarrativeLawValidator:
    """
    Validates whether narrative laws apply in a specific domain.
    
    Implements "presume and prove" methodology:
    - Start with hypothesis that Д/п > 0.5
    - Calculate domain characteristics
    - Select appropriate transformers with rationale
    - Test empirically
    - Report honest results
    """
    
    def __init__(self):
        self.efficiency_threshold = 0.5
        self.validation_history: List[ValidationResult] = []
    
    def validate_domain(
        self,
        domain_name: str,
        narrativity: float,
        correlation: float,
        coupling: float = None,
        transformer_info: Dict[str, str] = None
    ) -> ValidationResult:
        """
        Test narrative laws for a specific domain.
        
        Parameters
        ----------
        domain_name : str
            Name of the domain
        narrativity : float
            Calculated п (0-1)
        correlation : float
            Measured r between ю and ❊
        coupling : float, optional
            κ - narrator-narrated coupling (defaults based on domain type)
        transformer_info : dict, optional
            Rationale for transformer selection
        
        Returns
        -------
        result : ValidationResult
            Complete validation report with pass/fail
        """
        # Estimate coupling if not provided
        if coupling is None:
            coupling = self._estimate_coupling(narrativity)
        
        # Calculate Д (narrative agency)
        narrative_agency = narrativity * correlation * coupling
        
        # Calculate efficiency
        efficiency = narrative_agency / narrativity if narrativity > 0 else 0
        
        # Test threshold
        passes = efficiency > self.efficiency_threshold
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            domain_name, narrativity, efficiency, passes
        )
        
        # Create result
        result = ValidationResult(
            domain_name=domain_name,
            narrativity=narrativity,
            correlation=correlation,
            coupling=coupling,
            narrative_agency=narrative_agency,
            efficiency=efficiency,
            passes=passes,
            interpretation=interpretation,
            transformer_rationale=transformer_info or {}
        )
        
        # Store in history
        self.validation_history.append(result)
        
        return result
    
    def _estimate_coupling(self, narrativity: float) -> float:
        """
        Estimate κ (coupling) based on domain characteristics.
        
        Higher narrativity often correlates with higher coupling
        (more subjective = narrator has more influence on outcome)
        """
        # Base coupling estimation
        if narrativity > 0.85:
            return 0.9  # Self-rated, highly subjective
        elif narrativity > 0.7:
            return 0.7  # Character-driven
        elif narrativity > 0.5:
            return 0.4  # Mixed domains
        elif narrativity > 0.3:
            return 0.2  # Constrained
        else:
            return 0.1  # Objective/physics
    
    def _generate_interpretation(
        self,
        domain_name: str,
        narrativity: float,
        efficiency: float,
        passes: bool
    ) -> str:
        """Generate human-readable interpretation of results"""
        
        if passes:
            if narrativity > 0.8:
                return (
                    f"{domain_name} is a highly narrative domain (п={narrativity:.2f}). "
                    f"With efficiency {efficiency:.2f}, narrative quality strongly influences outcomes. "
                    f"Better stories WIN in this domain."
                )
            else:
                return (
                    f"{domain_name} (п={narrativity:.2f}) shows narrative agency with efficiency {efficiency:.2f}. "
                    f"While constrained by reality, narrative still plays a meaningful role."
                )
        else:
            if efficiency > 0.3:
                return (
                    f"{domain_name} shows narrative correlation (efficiency={efficiency:.2f}) but fails "
                    f"the 0.5 threshold. Reality constraints limit narrative agency. "
                    f"This is the 'startup paradox' - high correlation, low agency."
                )
            elif efficiency > 0.1:
                return (
                    f"{domain_name} has low narrative efficiency ({efficiency:.2f}). "
                    f"External factors (genre, budget, skill) dominate outcomes. "
                    f"Narrative matters but doesn't determine success."
                )
            else:
                return (
                    f"{domain_name} is constrained by objective reality (efficiency={efficiency:.2f}). "
                    f"Physics, logic, or performance dominate. Narrative has minimal causal impact."
                )
    
    def generate_transformer_rationale(
        self,
        narrativity: float,
        selected_transformers: List[str]
    ) -> Dict[str, str]:
        """
        Generate rationale for why these transformers were selected.
        
        Parameters
        ----------
        narrativity : float
            Domain narrativity (п)
        selected_transformers : list
            Names of selected transformers
        
        Returns
        -------
        rationale : dict
            Mapping of transformer name to rationale
        """
        rationale = {}
        
        # Generic rationale based on п
        if narrativity < 0.3:
            base_reason = "Constrained domain - focuses on plot/objective features"
        elif narrativity > 0.7:
            base_reason = "Open domain - captures character/subjective patterns"
        else:
            base_reason = "Mixed domain - provides balanced feature coverage"
        
        # Transformer-specific rationale
        transformer_reasons = {
            'statistical': "Baseline features - objective measurements",
            'nominative': "Name-based features - phonetics, semantics, cultural associations",
            'linguistic': "Language patterns - syntax, complexity, style",
            'ensemble': "Multi-character dynamics - relationships, interactions",
            'self_perception': "Identity and self-concept markers",
            'narrative_potential': "Growth, transformation, future orientation",
            'conflict': "Tension, opposition, stakes",
            'suspense': "Information control, mystery, anticipation",
            'emotional_semantic': "Emotion detection via embeddings (intelligent)",
            'authenticity': "Specificity, consistency, truth markers",
            'expertise': "Authority, credibility, competence signals"
        }
        
        for transformer in selected_transformers:
            # Get specific reason or use generic
            specific = transformer_reasons.get(transformer.lower(), base_reason)
            rationale[transformer] = f"{base_reason}. {specific}"
        
        return rationale
    
    def compare_to_theory(
        self,
        predicted_efficiency: float,
        actual_efficiency: float
    ) -> Dict[str, any]:
        """
        Compare theoretical prediction to empirical results.
        
        Shows "presumption vs proof"
        """
        difference = actual_efficiency - predicted_efficiency
        percent_diff = (difference / predicted_efficiency * 100) if predicted_efficiency > 0 else 0
        
        if abs(percent_diff) < 10:
            match = "Excellent - theory predicts reality"
        elif abs(percent_diff) < 25:
            match = "Good - theory approximately correct"
        elif abs(percent_diff) < 50:
            match = "Moderate - theory partially correct"
        else:
            match = "Poor - theory fails for this domain"
        
        return {
            'predicted': predicted_efficiency,
            'actual': actual_efficiency,
            'difference': difference,
            'percent_difference': percent_diff,
            'match_quality': match,
            'interpretation': self._interpret_theory_match(difference)
        }
    
    def _interpret_theory_match(self, difference: float) -> str:
        """Interpret difference between theory and reality"""
        if abs(difference) < 0.05:
            return "Theory precisely predicts empirical results."
        elif difference > 0:
            return "Narrative agency EXCEEDS theoretical prediction - domain more open than estimated."
        else:
            return "Narrative agency BELOW theoretical prediction - reality constrains more than expected."
    
    def get_validation_summary(self) -> Dict[str, any]:
        """
        Get summary of all validated domains.
        
        Returns pass rate, distributions, etc.
        """
        if not self.validation_history:
            return {
                'total_domains': 0,
                'passes': 0,
                'fails': 0,
                'pass_rate': 0.0
            }
        
        total = len(self.validation_history)
        passes = sum(1 for v in self.validation_history if v.passes)
        fails = total - passes
        
        return {
            'total_domains': total,
            'passes': passes,
            'fails': fails,
            'pass_rate': passes / total,
            'avg_efficiency': np.mean([v.efficiency for v in self.validation_history]),
            'avg_narrativity': np.mean([v.narrativity for v in self.validation_history]),
            'domains': [
                {
                    'name': v.domain_name,
                    'efficiency': v.efficiency,
                    'passes': v.passes
                }
                for v in self.validation_history
            ]
        }
    
    def print_validation_summary(self):
        """Print formatted validation summary"""
        summary = self.get_validation_summary()
        
        print("\n" + "="*80)
        print("CROSS-DOMAIN VALIDATION SUMMARY")
        print("="*80)
        print(f"\nTotal Domains Validated: {summary['total_domains']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%} ({summary['passes']}/{summary['total_domains']})")
        print(f"Average Efficiency: {summary['avg_efficiency']:.3f}")
        print(f"Average Narrativity: {summary['avg_narrativity']:.3f}")
        
        print("\n" + "-"*80)
        print("DOMAIN RESULTS:")
        print("-"*80)
        
        for domain in summary['domains']:
            status = "✓" if domain['passes'] else "❌"
            print(f"{status} {domain['name']:30s} - Efficiency: {domain['efficiency']:.3f}")
        
        print("="*80)
        
        # Interpretation
        if summary['pass_rate'] < 0.3:
            print("\nNarrative laws apply to a MINORITY of domains.")
            print("This validates honest testing - framework has limits.")
        elif summary['pass_rate'] < 0.6:
            print("\nNarrative laws apply to SOME domains.")
            print("Domain characteristics (п, κ) determine applicability.")
        else:
            print("\nNarrative laws apply to MOST domains.")
            print("Framework broadly applicable across domain types.")

