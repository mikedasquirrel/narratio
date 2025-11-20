"""
Universal Domain Analyzer

Complete framework implementation for ANY domain.

Calculates ALL variables following theoretical framework:
ж → ю → ❊ → Д → п → μ → ф → ة → Ξ

This is the definitive implementation of the formal variable system.
"""

import numpy as np
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .story_quality import StoryQualityCalculator
from .bridge_calculator import BridgeCalculator
from .gravitational_forces import GravitationalCalculator
# from .golden_narratio import GoldenNarratio  # TODO: Implement or remove


class UniversalDomainAnalyzer:
    """
    Analyzes ANY domain following complete theoretical framework.
    
    Ensures all variables (ж, ю, ❊, Д, п, μ, ф, ة, Ξ) are properly calculated
    according to formal system.
    
    This is the authoritative implementation.
    """
    
    def __init__(self, domain_name: str, narrativity: float):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        narrativity : float
            Domain's п value [0, 1]
        """
        self.domain_name = domain_name
        self.п = narrativity
        
        # Initialize calculators
        self.story_quality_calc = StoryQualityCalculator(self.п)
        self.bridge_calc = BridgeCalculator()
        self.gravitational_calc = GravitationalCalculator()
        self.golden_narratio = GoldenNarratio()
    
    def analyze_complete(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        names: List[str],
        genome: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        masses: Optional[np.ndarray] = None,
        baseline_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Complete domain analysis with ALL variables.
        
        Parameters
        ----------
        texts : list of str
            Narrative texts
        outcomes : ndarray
            Outcomes (❊)
        names : list of str
            Organism names
        genome : ndarray, optional
            Pre-extracted genomes (ж) (if None, must extract)
        feature_names : list, optional
            Names of features in ж
        masses : ndarray, optional
            Gravitational masses (μ) (if None, estimates)
        baseline_features : ndarray, optional
            Objective features for baseline calculation
            
        Returns
        -------
        results : dict
            Complete analysis with all variables
        """
        print("="*80)
        print(f"UNIVERSAL DOMAIN ANALYSIS: {self.domain_name}")
        print("="*80)
        print(f"\nNarrativity (п): {self.п:.2f}")
        print(f"Organisms: {len(texts)}")
        
        # === EXTRACT ж (GENOME) ===
        
        if genome is None:
            print("\n[ERROR] Genome (ж) must be pre-extracted with transformers")
            print("Use: transformer_library.get_for_narrativity(п) to select transformers")
            print("Then: apply transformers to extract ж")
            raise ValueError("genome must be provided")
        
        print(f"\n✓ Genome (ж) provided: {genome.shape}")
        
        # === COMPUTE ю (STORY QUALITY) ===
        
        print("\nComputing story quality (ю)...")
        story_quality = self.story_quality_calc.compute_ю(genome, feature_names or [])
        
        print(f"✓ Story quality (ю) computed")
        print(f"  Mean: {story_quality.mean():.3f}")
        print(f"  Std: {story_quality.std():.3f}")
        print(f"  Range: [{story_quality.min():.3f}, {story_quality.max():.3f}]")
        
        # === ❊ (OUTCOMES) PROVIDED ===
        
        print(f"\n✓ Outcomes (❊) provided")
        if set(outcomes) == {0, 1}:
            print(f"  Type: Binary")
            print(f"  Success rate: {outcomes.mean():.1%}")
        else:
            print(f"  Type: Continuous")
            print(f"  Mean: {outcomes.mean():.3f}")
        
        # === CALCULATE Д (THE BRIDGE) ===
        
        print("\nCalculating the bridge (Д)...")
        D_results = self.bridge_calc.calculate_D(
            story_quality, outcomes,
            baseline_features=baseline_features,
            domain_hint=self.domain_name
        )
        
        print(f"✓ The bridge calculated")
        print(f"  r_narrative: {D_results['r_narrative']:.4f}")
        print(f"  r_baseline: {D_results['r_baseline']:.4f} ({D_results['baseline_method']})")
        print(f"  Д: {D_results['Д']:.4f}")
        print(f"  Passes threshold (Д > 0.10): {D_results['passes_threshold']}")
        print(f"  {D_results['interpretation']}")
        
        # === ESTIMATE μ (MASSES) ===
        
        if masses is None:
            # Default: uniform masses
            mu = np.ones(len(genome))
            print(f"\n✓ Masses (μ) set to uniform (1.0)")
        else:
            mu = masses
            print(f"\n✓ Masses (μ) provided: mean={mu.mean():.2f}")
        
        # === CALCULATE GRAVITATIONAL FORCES ===
        
        print("\nCalculating gravitational forces...")
        forces = self.gravitational_calc.calculate_all_forces(genome, names, mu, story_quality)
        
        print(f"✓ Gravitational forces calculated")
        print(f"  ф (narrative gravity) mean: {forces['ф'].mean():.6f}")
        print(f"  ة (nominative gravity) mean: {forces['ة'].mean():.6f}")
        print(f"  ф_net mean: {forces['ф_net'].mean():.6f}")
        print(f"  Tensions identified: {len(forces['tensions'])}")
        
        # === ESTIMATE Ξ (GOLDEN NARRATIO) ===
        
        print("\nEstimating Golden Narratio (Ξ)...")
        Xi = self.golden_narratio.estimate_Ξ(genome, outcomes, method='winners')
        
        print(f"✓ Ξ estimated via {self.golden_narratio.estimation_method}")
        
        # Test if proximity to Ξ correlates with success
        Xi_validation = self.golden_narratio.test_Ξ_correlation(genome, outcomes, Xi)
        
        print(f"  Ξ correlation with success: r={Xi_validation['correlation']:.3f} (p={Xi_validation['p_value']:.4f})")
        print(f"  Validated: {Xi_validation['validated']}")
        
        # === COMPILE COMPLETE RESULTS ===
        
        results = {
            # Core variables
            'domain': self.domain_name,
            'п': float(self.п),
            'ж': genome,
            'ю': story_quality,
            '❊': outcomes,
            'μ': mu,
            
            # Feature info
            'n_features': genome.shape[1],
            'n_organisms': len(genome),
            'feature_names': feature_names or [],
            
            # Bridge results
            'Д_results': D_results,
            'Д': D_results['Д'],
            'r_narrative': D_results['r_narrative'],
            'r_baseline': D_results['r_baseline'],
            
            # Gravitational results
            'ф': forces['ф'],
            'ة': forces['ة'],
            'ф_net': forces['ф_net'],
            'gravitational_tensions': forces['tensions'],
            
            # Golden Narratio
            'Ξ': Xi,
            'Ξ_validation': Xi_validation,
            'distances_from_Ξ': self.golden_narratio.distance_from_Ξ(genome, Xi),
            
            # Meta
            'story_quality_weights': self.story_quality_calc.weights,
            'story_quality_interpretation': self.story_quality_calc.interpretation
        }
        
        # === PRINT SUMMARY ===
        
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("COMPLETE VARIABLE ANALYSIS")
        print("="*80)
        
        print(f"\nDomain: {results['domain']}")
        print(f"Narrativity (п): {results['п']:.2f}")
        print(f"Organisms: {results['n_organisms']}")
        print(f"Features: {results['n_features']}")
        
        print(f"\n--- STORY QUALITY (ю) ---")
        print(f"Weighting: {results['story_quality_interpretation']}")
        print(f"Mean: {results['ю'].mean():.3f} ± {results['ю'].std():.3f}")
        print(f"Range: [{results['ю'].min():.3f}, {results['ю'].max():.3f}]")
        
        print(f"\n--- THE BRIDGE (Д) ---")
        print(f"r_narrative: {results['r_narrative']:.4f}")
        print(f"r_baseline: {results['r_baseline']:.4f} ({results['Д_results']['baseline_method']})")
        print(f"Д (advantage): {results['Д']:.4f}")
        print(f"Threshold (>0.10): {'✓ PASS' if results['Д_results']['passes_threshold'] else '✗ FAIL'}")
        print(f"Interpretation: {results['Д_results']['interpretation']}")
        
        print(f"\n--- GRAVITATIONAL FORCES ---")
        print(f"ф (narrative) mean: {results['ф'].mean():.6f}")
        print(f"ة (nominative) mean: {results['ة'].mean():.6f}")
        print(f"Tensions (ф ≠ ة): {len(results['gravitational_tensions'])} identified")
        
        if len(results['gravitational_tensions']) > 0:
            top_tension = results['gravitational_tensions'][0]
            print(f"  Strongest: {top_tension['name_i']} ←→ {top_tension['name_j']}")
            print(f"    Tension: {top_tension['tension_magnitude']:.3f} ({top_tension['tension_type']})")
        
        print(f"\n--- GOLDEN NARRATIO (Ξ) ---")
        print(f"Estimation: {self.golden_narratio.estimation_method}")
        print(f"Ξ correlation: r={results['Ξ_validation']['correlation']:.3f} (p={results['Ξ_validation']['p_value']:.4f})")
        print(f"Validated: {results['Ξ_validation']['validated']}")
        print(f"Mean distance from Ξ: {results['distances_from_Ξ'].mean():.3f}")
        
        print(f"\n{'='*80}")
        print("FORMAL SYSTEM COMPLETE")
        print(f"{'='*80}")
        print("\nAll variables calculated according to theoretical framework:")
        print("  ✓ ж (genome) - feature vector")
        print("  ✓ ю (story quality) - weighted by п")
        print("  ✓ ❊ (outcomes) - actual results")
        print("  ✓ Д (bridge) - narrative advantage")
        print("  ✓ п (narrativity) - domain openness")
        print("  ✓ μ (mass) - gravitational weight")
        print("  ✓ ф (narrative gravity) - story-based clustering")
        print("  ✓ ة (nominative gravity) - name-based clustering")
        print("  ✓ Ξ (golden narratio) - universal ideal")
    
    def calculate_narrativity(
        self,
        domain_characteristics: Dict[str, float]
    ) -> float:
        """
        Calculate п (narrativity) from domain characteristics.
        
        Implements:
        п = 0.30×п_structural + 0.20×п_temporal + 0.25×п_agency +
            0.15×п_interpretation + 0.10×п_format
        
        Parameters
        ----------
        domain_characteristics : dict
            Contains п_structural, п_temporal, п_agency, п_interpretation, п_format
            
        Returns
        -------
        п : float
            Overall narrativity score
        """
        components = {
            'structural': 0.30,
            'temporal': 0.20,
            'agency': 0.25,
            'interpretation': 0.15,
            'format': 0.10
        }
        
        п = 0.0
        for component, weight in components.items():
            component_key = f'п_{component}'
            if component_key in domain_characteristics:
                п += weight * domain_characteristics[component_key]
        
        return п

