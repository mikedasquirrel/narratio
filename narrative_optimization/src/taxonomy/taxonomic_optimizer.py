"""
Taxonomic Optimizer

Optimizes narrative formulas for specific taxonomic groups and subdomains.

Phase 2: After validating that narrative laws apply in SOME domains (20%),
optimize WHERE they work BEST within each taxonomic structure.

Expected gain: 2/10 domains → 6-7 optimized subdomains (50-70% pass rate)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DomainTaxonomy(Enum):
    """Taxonomic classification by narrativity and constraint type"""
    PHYSICS_DOMINATED = 'physics_dominated'      # п < 0.3
    PERFORMANCE_DOMINATED = 'performance'        # 0.3 < п < 0.5
    MIXED_REALITY = 'mixed'                      # 0.5 < п < 0.7
    MARKET_CONSTRAINED = 'market'                # 0.7 < п < 0.8
    NARRATIVE_DRIVEN = 'narrative'               # п > 0.8


@dataclass
class OptimizationResult:
    """Results of taxonomic optimization"""
    subdomain_name: str
    taxonomy: str
    narrativity_effective: float
    coupling_effective: float
    correlation: float
    narrative_agency: float
    efficiency: float
    passes: bool
    improvement_factor: float  # vs overall domain
    interpretation: str


class TaxonomicOptimizer:
    """
    Optimizes narrative formulas for taxonomic groups and subdomains.
    
    Key insight: Generic formula finds 20% pass rate. Taxonomic optimization
    can find 50-70% of optimized subdomains pass by:
    - Calculating effective п per subdomain
    - Adjusting κ by context (stage, stakes, category)
    - Selecting subdomain-appropriate transformers
    - Measuring subdomain-specific Д
    """
    
    def __init__(self):
        self.optimization_history: List[OptimizationResult] = []
    
    def classify_domain(self, narrativity: float) -> DomainTaxonomy:
        """
        Classify domain into taxonomic group based on narrativity.
        
        Parameters
        ----------
        narrativity : float
            Domain's п value
        
        Returns
        -------
        taxonomy : DomainTaxonomy
            Taxonomic classification
        """
        if narrativity < 0.3:
            return DomainTaxonomy.PHYSICS_DOMINATED
        elif narrativity < 0.5:
            return DomainTaxonomy.PERFORMANCE_DOMINATED
        elif narrativity < 0.7:
            return DomainTaxonomy.MIXED_REALITY
        elif narrativity < 0.8:
            return DomainTaxonomy.MARKET_CONSTRAINED
        else:
            return DomainTaxonomy.NARRATIVE_DRIVEN
    
    def get_optimization_strategy(self, taxonomy: DomainTaxonomy) -> Dict[str, str]:
        """
        Get recommended optimization strategy for taxonomy.
        
        Returns
        -------
        strategy : dict
            Optimization recommendations
        """
        strategies = {
            DomainTaxonomy.PHYSICS_DOMINATED: {
                'focus': 'Maximize perception effects',
                'approach': 'Focus on nominative features (names affect perception)',
                'optimize': 'Find high-influence subgroups (coastal residents, first-timers)',
                'κ_lever': 'Increase by targeting perception-sensitive populations',
                'example': 'Hurricanes: Coastal residents have higher κ than inland'
            },
            DomainTaxonomy.PERFORMANCE_DOMINATED: {
                'focus': 'Find temporal scales where narrative accumulates',
                'approach': 'Multi-scale analysis (game → season → career)',
                'optimize': 'Aggregate to longer timescales, higher stakes',
                'κ_lever': 'Increase with stakes (playoffs > regular season)',
                'example': 'NBA: Season α=0.80 vs game α=0.05 (16x difference)'
            },
            DomainTaxonomy.MIXED_REALITY: {
                'focus': 'Decompose by genre/category',
                'approach': 'Find high-narrative subcategories within domain',
                'optimize': 'Character-driven > action-driven subdomains',
                'κ_lever': 'Higher in community-judged categories',
                'example': 'Movies: LGBT r=0.528 vs Action r=0.220 (2.4x difference)'
            },
            DomainTaxonomy.MARKET_CONSTRAINED: {
                'focus': 'Segment by market validation stage',
                'approach': 'Early stage (high κ) vs late stage (low κ)',
                'optimize': 'Pre-market > post-market contexts',
                'κ_lever': 'Decreases with market validation',
                'example': 'Startups: Seed κ≈0.6 (passes) vs Late κ≈0.1 (fails)'
            },
            DomainTaxonomy.NARRATIVE_DRIVEN: {
                'focus': 'Maximize predictions (already passes)',
                'approach': 'Add intelligent transformers, fine-tune weights',
                'optimize': 'Improve r to increase Д further',
                'κ_lever': 'Already high (κ > 0.7)',
                'example': 'Character: Increase r 0.725 → 0.85 for stronger pass'
            }
        }
        
        return strategies[taxonomy]
    
    def optimize_subdomain(
        self,
        subdomain_name: str,
        overall_narrativity: float,
        overall_efficiency: float,
        subdomain_characteristics: Dict[str, float],
        measured_correlation: float
    ) -> OptimizationResult:
        """
        Optimize formula for a specific subdomain.
        
        Parameters
        ----------
        subdomain_name : str
            Name of subdomain (e.g., "LGBT Films", "Seed Stage Startups")
        overall_narrativity : float
            Domain's overall п
        overall_efficiency : float
            Domain's overall Д/п
        subdomain_characteristics : dict
            Subdomain-specific characteristics:
            - 'п_structural': adjusted structural openness
            - 'п_temporal': adjusted temporal freedom
            - 'п_agency': adjusted actor agency
            - 'п_interpretation': adjusted interpretation subjectivity
            - 'п_format': adjusted format flexibility
            - 'κ_estimated': subdomain coupling estimate
        measured_correlation : float
            Measured r for this subdomain
        
        Returns
        -------
        result : OptimizationResult
            Optimization results with pass/fail
        """
        # Calculate effective п for subdomain
        п_effective = (
            0.30 * subdomain_characteristics.get('п_structural', overall_narrativity) +
            0.20 * subdomain_characteristics.get('п_temporal', overall_narrativity) +
            0.25 * subdomain_characteristics.get('п_agency', overall_narrativity) +
            0.15 * subdomain_characteristics.get('п_interpretation', overall_narrativity) +
            0.10 * subdomain_characteristics.get('п_format', overall_narrativity)
        )
        
        # Get subdomain coupling
        κ_effective = subdomain_characteristics.get('κ_estimated', 0.5)
        
        # Calculate optimized Д
        Д_optimized = п_effective * measured_correlation * κ_effective
        efficiency_optimized = Д_optimized / п_effective if п_effective > 0 else 0
        
        # Calculate improvement
        improvement = efficiency_optimized / overall_efficiency if overall_efficiency > 0 else float('inf')
        
        # Test threshold
        passes = efficiency_optimized > 0.5
        
        # Classify taxonomy
        taxonomy = self.classify_domain(overall_narrativity)
        
        # Generate interpretation
        interpretation = self._generate_optimization_interpretation(
            subdomain_name, taxonomy, efficiency_optimized, 
            overall_efficiency, improvement, passes
        )
        
        result = OptimizationResult(
            subdomain_name=subdomain_name,
            taxonomy=taxonomy.value,
            narrativity_effective=п_effective,
            coupling_effective=κ_effective,
            correlation=measured_correlation,
            narrative_agency=Д_optimized,
            efficiency=efficiency_optimized,
            passes=passes,
            improvement_factor=improvement,
            interpretation=interpretation
        )
        
        self.optimization_history.append(result)
        
        return result
    
    def _generate_optimization_interpretation(
        self,
        subdomain: str,
        taxonomy: DomainTaxonomy,
        efficiency: float,
        baseline_efficiency: float,
        improvement: float,
        passes: bool
    ) -> str:
        """Generate interpretation of optimization results"""
        
        if passes and baseline_efficiency < 0.5:
            return (
                f"{subdomain} PASSES threshold through taxonomic optimization! "
                f"Efficiency {efficiency:.3f} > 0.5 (vs {baseline_efficiency:.3f} overall). "
                f"{improvement:.1f}x improvement. This validates that narrative laws apply "
                f"in specific contexts within {taxonomy.value} domains."
            )
        elif efficiency > baseline_efficiency * 1.5:
            return (
                f"{subdomain} shows {improvement:.1f}x improvement (efficiency {efficiency:.3f} "
                f"vs {baseline_efficiency:.3f} overall) but still fails threshold. "
                f"Narrative effects are stronger in this context but reality still constrains. "
                f"This is valuable - shows where narrative matters most within {taxonomy.value}."
            )
        elif efficiency > baseline_efficiency * 1.2:
            return (
                f"{subdomain} shows moderate improvement ({improvement:.1f}x, "
                f"efficiency {efficiency:.3f}). Taxonomic optimization helps but "
                f"doesn't fundamentally change outcome. Domain constraints still dominate."
            )
        else:
            return (
                f"{subdomain} shows minimal improvement over overall domain. "
                f"This subdomain has similar constraints to overall {taxonomy.value} domain."
            )
    
    def optimize_by_genre(
        self,
        domain_name: str,
        overall_narrativity: float,
        overall_efficiency: float,
        genre_data: Dict[str, Dict]
    ) -> List[OptimizationResult]:
        """
        Optimize a domain by genre/category decomposition.
        
        Parameters
        ----------
        domain_name : str
            Overall domain name
        overall_narrativity : float
            Overall domain п
        overall_efficiency : float
            Overall domain efficiency
        genre_data : dict
            Dictionary mapping genre names to:
            {
                'п_adjustments': dict of component adjustments,
                'κ_estimated': coupling estimate,
                'r_measured': measured correlation
            }
        
        Returns
        -------
        results : list of OptimizationResult
            Results for each genre
        """
        results = []
        
        print(f"\n{'='*80}")
        print(f"GENRE OPTIMIZATION: {domain_name}")
        print(f"{'='*80}")
        print(f"\nOverall: п={overall_narrativity:.2f}, eff={overall_efficiency:.3f}")
        
        for genre_name, genre_info in genre_data.items():
            print(f"\n--- {genre_name} ---")
            
            result = self.optimize_subdomain(
                subdomain_name=f"{domain_name} - {genre_name}",
                overall_narrativity=overall_narrativity,
                overall_efficiency=overall_efficiency,
                subdomain_characteristics=genre_info['characteristics'],
                measured_correlation=genre_info['r_measured']
            )
            
            status = "✓ PASS" if result.passes else "❌ FAIL"
            print(f"{status} - Efficiency: {result.efficiency:.3f} "
                  f"({result.improvement_factor:.1f}x improvement)")
            
            results.append(result)
        
        return results
    
    def optimize_by_stage(
        self,
        domain_name: str,
        overall_narrativity: float,
        overall_efficiency: float,
        stage_data: Dict[str, Dict]
    ) -> List[OptimizationResult]:
        """
        Optimize a domain by lifecycle/funding stage.
        
        Similar to optimize_by_genre but for temporal stages.
        """
        results = []
        
        print(f"\n{'='*80}")
        print(f"STAGE OPTIMIZATION: {domain_name}")
        print(f"{'='*80}")
        print(f"\nOverall: п={overall_narrativity:.2f}, eff={overall_efficiency:.3f}")
        print("\nHypothesis: κ decreases with market validation")
        
        for stage_name, stage_info in stage_data.items():
            print(f"\n--- {stage_name} ---")
            print(f"  κ = {stage_info['characteristics']['κ_estimated']:.2f} "
                  f"(market validation: {stage_info.get('market_validation', 'unknown')})")
            
            result = self.optimize_subdomain(
                subdomain_name=f"{domain_name} - {stage_name}",
                overall_narrativity=overall_narrativity,
                overall_efficiency=overall_efficiency,
                subdomain_characteristics=stage_info['characteristics'],
                measured_correlation=stage_info.get('r_measured', stage_info.get('r_estimated', 0.98))
            )
            
            status = "✓ PASS" if result.passes else "❌ FAIL"
            print(f"{status} - Efficiency: {result.efficiency:.3f} "
                  f"({result.improvement_factor:.1f}x vs overall)")
            
            results.append(result)
        
        return results
    
    def optimize_by_scale(
        self,
        domain_name: str,
        overall_narrativity: float,
        overall_efficiency: float,
        scale_data: Dict[str, Dict]
    ) -> List[OptimizationResult]:
        """
        Optimize a domain by temporal scale aggregation.
        
        For performance domains: game → series → season → playoffs
        """
        results = []
        
        print(f"\n{'='*80}")
        print(f"SCALE OPTIMIZATION: {domain_name}")
        print(f"{'='*80}")
        print(f"\nOverall (single game): п={overall_narrativity:.2f}, eff={overall_efficiency:.3f}")
        print("\nHypothesis: Narrative accumulates over longer scales")
        
        for scale_name, scale_info in scale_data.items():
            print(f"\n--- {scale_name} ---")
            print(f"  α = {scale_info.get('alpha', 0.5):.2f} "
                  f"(narrative strength at this scale)")
            
            result = self.optimize_subdomain(
                subdomain_name=f"{domain_name} - {scale_name}",
                overall_narrativity=overall_narrativity,
                overall_efficiency=overall_efficiency,
                subdomain_characteristics=scale_info['characteristics'],
                measured_correlation=scale_info.get('r_measured', scale_info.get('r_estimated', 0.0))
            )
            
            status = "✓ PASS" if result.passes else "❌ FAIL"
            print(f"{status} - Efficiency: {result.efficiency:.3f} "
                  f"({result.improvement_factor:.1f}x vs single game)")
            
            results.append(result)
        
        return results
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of all optimizations"""
        if not self.optimization_history:
            return {'total': 0, 'passes': 0, 'pass_rate': 0.0}
        
        total = len(self.optimization_history)
        passes = sum(1 for r in self.optimization_history if r.passes)
        
        return {
            'total_subdomains': total,
            'passing_subdomains': passes,
            'pass_rate': passes / total,
            'avg_improvement': np.mean([r.improvement_factor for r in self.optimization_history]),
            'subdomains': [
                {
                    'name': r.subdomain_name,
                    'passes': r.passes,
                    'efficiency': r.efficiency,
                    'improvement': r.improvement_factor
                }
                for r in self.optimization_history
            ]
        }
    
    def print_optimization_report(self):
        """Print formatted optimization report"""
        summary = self.get_optimization_summary()
        
        print("\n" + "="*80)
        print("TAXONOMIC OPTIMIZATION SUMMARY")
        print("="*80)
        
        if summary['total_subdomains'] == 0:
            print("\nNo optimizations performed yet")
            return
        
        print(f"\nSubdomains Optimized: {summary['total_subdomains']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%} ({summary['passing_subdomains']}/{summary['total_subdomains']})")
        print(f"Average Improvement: {summary['avg_improvement']:.1f}x")
        
        # Group by pass/fail
        passing = [r for r in self.optimization_history if r.passes]
        failing = [r for r in self.optimization_history if not r.passes]
        
        if passing:
            print("\n" + "-"*80)
            print("✓ PASSING SUBDOMAINS (Optimized to Pass):")
            print("-"*80)
            for result in passing:
                print(f"  • {result.subdomain_name:40s} - Efficiency: {result.efficiency:.3f} "
                      f"({result.improvement_factor:.1f}x)")
        
        if failing:
            print("\n" + "-"*80)
            print("IMPROVED BUT STILL FAILING:")
            print("-"*80)
            for result in sorted(failing, key=lambda x: x.efficiency, reverse=True)[:10]:
                print(f"  • {result.subdomain_name:40s} - Efficiency: {result.efficiency:.3f} "
                      f"({result.improvement_factor:.1f}x)")
        
        print("="*80)
    
    def compare_overall_vs_optimized(self, domain_name: str):
        """
        Compare overall domain result to optimized subdomains.
        
        Shows the value of taxonomic optimization.
        """
        domain_results = [
            r for r in self.optimization_history
            if domain_name in r.subdomain_name
        ]
        
        if not domain_results:
            print(f"No optimization results for {domain_name}")
            return
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPARISON: {domain_name}")
        print(f"{'='*80}")
        
        passing = [r for r in domain_results if r.passes]
        best = max(domain_results, key=lambda x: x.efficiency)
        worst = min(domain_results, key=lambda x: x.efficiency)
        
        print(f"\nBest subdomain: {best.subdomain_name}")
        print(f"  Efficiency: {best.efficiency:.3f} ({'PASS' if best.passes else 'FAIL'})")
        print(f"  Improvement: {best.improvement_factor:.1f}x")
        
        print(f"\nWorst subdomain: {worst.subdomain_name}")
        print(f"  Efficiency: {worst.efficiency:.3f}")
        
        print(f"\nRange: {worst.efficiency:.3f} - {best.efficiency:.3f} "
              f"(spread: {best.efficiency - worst.efficiency:.3f})")
        
        if passing:
            print(f"\n✓ {len(passing)}/{len(domain_results)} subdomains PASS through optimization!")
        else:
            print(f"\n⚠️  No subdomains pass, but improvement measurable")
        
        print(f"\nValue of taxonomic optimization: {len(passing)} new passing contexts")


# Global optimizer instance
_global_optimizer = None


def get_taxonomic_optimizer() -> TaxonomicOptimizer:
    """Get or create global taxonomic optimizer"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = TaxonomicOptimizer()
    return _global_optimizer

