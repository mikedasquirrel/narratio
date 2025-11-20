"""
Seven-Force Unified Model

Complete integration of all narrative forces:
1. ф (Phi): Narrative Gravity - story similarity attraction
2. ة (Ta Marbuta): Nominative Gravity - name similarity attraction  
3. θ (Theta): Awareness Resistance - conscious rejection
4. λ (Lambda): Fundamental Constraints - training, aptitude, resources
5. τ (Tau): Temporal Dynamics - duration effects and modulation
6. Φ (Phi_capital): Embodiment Constraints - physical/cognitive limits
7. Λ_physical: Physical Laws - inviolable natural constraints

Unified equation: Д = (ф + ة) × (1 - Φ) × τ_mod - (θ + λ + Λ_physical)

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class SevenForceState:
    """
    Complete force state for a domain or narrative instance.
    
    Attributes represent all forces that determine whether better stories win.
    """
    # Narrative forces (enabling)
    narrative_gravity_phi: float  # ф: Story similarity attraction
    nominative_gravity_ta: float  # ة: Name similarity attraction
    
    # Resistance forces (opposing)
    awareness_theta: float  # θ: Conscious resistance
    fundamental_lambda: float  # λ: Skill/resource requirements
    physical_law_Lambda: float  # Λ: Inviolable physics
    
    # Modulation forces
    temporal_tau: float  # τ: Duration effect (multiplier)
    embodiment_Phi: float  # Φ: Accessibility reduction (multiplier)
    
    # Computed
    narrative_force_total: Optional[float] = None
    resistance_total: Optional[float] = None
    net_effect: Optional[float] = None
    
    def __post_init__(self):
        """Compute derived quantities."""
        self.narrative_force_total = (self.narrative_gravity_phi + self.nominative_gravity_ta) * (1 - self.embodiment_Phi)
        self.resistance_total = self.awareness_theta + self.fundamental_lambda + self.physical_law_Lambda
        self.net_effect = self.narrative_force_total * self.temporal_tau - self.resistance_total


class SevenForceModel:
    """
    Unified model for calculating narrative effects across all domains.
    
    Usage:
    ------
    model = SevenForceModel()
    
    # Define force state for domain
    forces = SevenForceState(
        narrative_gravity_phi=0.50,
        nominative_gravity_ta=0.60,
        awareness_theta=0.515,
        fundamental_lambda=0.531,
        physical_law_Lambda=0.55,
        temporal_tau=1.0,
        embodiment_Phi=0.35
    )
    
    # Compute outcome
    prediction = model.predict_narrative_effect(forces)
    """
    
    def __init__(self):
        """Initialize unified model."""
        pass
    
    def predict_narrative_effect(
        self,
        forces: SevenForceState
    ) -> Dict[str, float]:
        """
        Predict narrative effect from seven forces.
        
        Parameters
        ----------
        forces : SevenForceState
            Complete force configuration
            
        Returns
        -------
        prediction : dict
            {
                'net_effect': Д (the Bridge),
                'narrative_total': Enabling forces,
                'resistance_total': Opposing forces,
                'efficiency': Д / π (if π provided),
                'interpretation': String describing state
            }
        """
        # Already computed in SevenForceState.__post_init__
        net = forces.net_effect
        
        # Interpret (but note this is just description, not mechanism)
        if net > 0.20:
            interpretation = "Narrative forces dominate. Better stories win."
        elif net > 0:
            interpretation = "Narrative forces slight edge. Stories matter modestly."
        elif net > -0.20:
            interpretation = "Forces in equilibrium. Appears null but forces balance."
        else:
            interpretation = "Resistance forces dominate. Reality constrains narrative."
        
        return {
            'net_effect': net,
            'narrative_total': forces.narrative_force_total,
            'resistance_total': forces.resistance_total,
            'interpretation': interpretation,
            'note': 'Prediction from forces. Actual mechanisms remain elusive.'
        }
    
    def explain_domain(
        self,
        domain_name: str,
        forces: SevenForceState,
        observed_effect: Optional[float] = None
    ) -> str:
        """
        Explain domain behavior through force analysis.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        forces : SevenForceState
            Force configuration
        observed_effect : float, optional
            Actually measured Д (for comparison)
            
        Returns
        -------
        explanation : str
            Multi-line explanation of force dynamics
        """
        prediction = self.predict_narrative_effect(forces)
        predicted_effect = prediction['net_effect']
        
        explanation = []
        explanation.append(f"\n{'='*80}")
        explanation.append(f"SEVEN-FORCE ANALYSIS: {domain_name}")
        explanation.append(f"{'='*80}\n")
        
        explanation.append("NARRATIVE FORCES (Enabling):")
        explanation.append(f"  ф (Narrative Gravity):  {forces.narrative_gravity_phi:.3f}")
        explanation.append(f"  ة (Nominative Gravity): {forces.nominative_gravity_ta:.3f}")
        explanation.append(f"  Sum: {forces.narrative_gravity_phi + forces.nominative_gravity_ta:.3f}")
        
        explanation.append(f"\nMODULATION FORCES:")
        explanation.append(f"  τ (Temporal Effect):   {forces.temporal_tau:.3f} (×)")
        explanation.append(f"  Φ (Embodiment Loss):   {forces.embodiment_Phi:.3f} (reduces by {forces.embodiment_Phi*100:.1f}%)")
        explanation.append(f"  Effective Narrative: {forces.narrative_force_total:.3f}")
        
        explanation.append(f"\nRESISTANCE FORCES (Opposing):")
        explanation.append(f"  θ (Awareness):         {forces.awareness_theta:.3f}")
        explanation.append(f"  λ (Fundamentals):      {forces.fundamental_lambda:.3f}")
        explanation.append(f"  Λ (Physical Laws):     {forces.physical_law_Lambda:.3f}")
        explanation.append(f"  Sum: {forces.resistance_total:.3f}")
        
        explanation.append(f"\nNET EFFECT:")
        explanation.append(f"  Predicted Д: {predicted_effect:+.3f}")
        
        if observed_effect is not None:
            error = abs(predicted_effect - observed_effect)
            explanation.append(f"  Observed Д:  {observed_effect:+.3f}")
            explanation.append(f"  Error: {error:.3f}")
        
        explanation.append(f"\n{prediction['interpretation']}")
        explanation.append(f"\n{'='*80}\n")
        
        return '\n'.join(explanation)
    
    def compare_domains(
        self,
        domain_configs: Dict[str, SevenForceState]
    ) -> str:
        """
        Compare multiple domains through force lens.
        
        Shows WHY domains differ in narrative power.
        """
        comparison = []
        comparison.append(f"\n{'='*80}")
        comparison.append("CROSS-DOMAIN FORCE COMPARISON")
        comparison.append(f"{'='*80}\n")
        
        # Sort by net effect
        sorted_domains = sorted(
            domain_configs.items(),
            key=lambda x: x[1].net_effect,
            reverse=True
        )
        
        comparison.append(f"{'Domain':<20} | {'Narrative':<10} | {'Resistance':<10} | {'Net Effect':<10} | Result")
        comparison.append("-" * 80)
        
        for domain, forces in sorted_domains:
            narrative = forces.narrative_force_total
            resistance = forces.resistance_total
            net = forces.net_effect
            
            result = "✓ Wins" if net > 0.20 else ("≈ Equilibrium" if net > -0.20 else "✗ Constrained")
            
            comparison.append(
                f"{domain:<20} | {narrative:>9.3f} | {resistance:>9.3f} | {net:>+9.3f} | {result}"
            )
        
        comparison.append("\n" + "="*80 + "\n")
        
        return '\n'.join(comparison)


# Example domain configurations
DOMAIN_FORCE_CONFIGS = {
    'tennis': SevenForceState(
        narrative_gravity_phi=0.50,
        nominative_gravity_ta=0.60,
        awareness_theta=0.515,
        fundamental_lambda=0.531,
        physical_law_Lambda=0.55,
        temporal_tau=1.0,
        embodiment_Phi=0.35
    ),
    
    'ufc': SevenForceState(
        narrative_gravity_phi=0.40,
        nominative_gravity_ta=0.55,
        awareness_theta=0.535,
        fundamental_lambda=0.544,
        physical_law_Lambda=0.70,
        temporal_tau=1.0,
        embodiment_Phi=0.50
    ),
    
    'literature': SevenForceState(
        narrative_gravity_phi=0.70,
        nominative_gravity_ta=0.75,
        awareness_theta=0.60,
        fundamental_lambda=0.20,
        physical_law_Lambda=0.05,
        temporal_tau=1.2,  # Extended duration helps
        embodiment_Phi=0.60  # Reading effort
    ),
    
    'lottery': SevenForceState(
        narrative_gravity_phi=0.10,
        nominative_gravity_ta=0.20,
        awareness_theta=0.70,
        fundamental_lambda=0.05,
        physical_law_Lambda=0.95,
        temporal_tau=1.0,
        embodiment_Phi=0.10
    ),
    
    'wwe': SevenForceState(
        narrative_gravity_phi=0.80,
        nominative_gravity_ta=0.85,
        awareness_theta=0.90,  # POSITIVE in prestige domain!
        fundamental_lambda=0.15,
        physical_law_Lambda=0.05,
        temporal_tau=1.0,
        embodiment_Phi=0.45
    )
}


def demonstrate_seven_force_model():
    """
    Demonstrate seven-force model across domains.
    
    Shows how same model explains vastly different outcomes.
    """
    model = SevenForceModel()
    
    print("\n" + "="*80)
    print("SEVEN-FORCE MODEL DEMONSTRATION")
    print("="*80 + "\n")
    
    for domain_name, forces in DOMAIN_FORCE_CONFIGS.items():
        explanation = model.explain_domain(domain_name, forces)
        print(explanation)
    
    # Cross-domain comparison
    comparison = model.compare_domains(DOMAIN_FORCE_CONFIGS)
    print(comparison)
    
    print("NOTE: Forces measured. Mechanisms deliberately kept elusive.")
    print("Better analysis through accepting mystery.\n")


if __name__ == '__main__':
    demonstrate_seven_force_model()

