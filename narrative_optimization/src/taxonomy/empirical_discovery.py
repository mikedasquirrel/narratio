"""
Empirical Discovery System

PURE DATA-DRIVEN: Let data reveal where narrative is most prominent.

Core Principle:
- DON'T: Theory predicts → measure to confirm
- DO: Measure everything → discover patterns → explain with theory

This is how science should work.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from scipy import stats
from collections import Counter
import itertools


@dataclass
class EmpiricalContext:
    """A discovered context where narrative shows measurable strength"""
    name: str
    filters: Dict[str, Any]
    n_samples: int
    r_measured: float
    r_squared: float
    p_value: float
    significance_level: str  # '***', '**', '*', 'ns'
    rank_overall: int
    rank_within_dimension: int
    dimension: str
    
    def __str__(self):
        return (f"{self.name:50s} | r={self.r_measured:+.3f} | "
                f"R²={self.r_squared:.3f} | n={self.n_samples:5d} | "
                f"p={self.p_value:.4f} {self.significance_level}")


class EmpiricalDiscoveryEngine:
    """
    Discovers optimal narrative contexts through exhaustive empirical search.
    
    Philosophy:
    1. Measure r in ALL possible subdivisions of data
    2. Rank by empirical strength (measured r)
    3. Top contexts = where narrative is ACTUALLY strongest
    4. Optimize formula for those (data-identified, not theory-predicted)
    5. Theory explains afterwards why those contexts are strong
    
    NOT: "LGBT should work because identity..." (theory → data)
    BUT: "Horror has r=0.342 in data, here's why..." (data → theory)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.discoveries: List[EmpiricalContext] = []
    
    def discover_all_contexts(
        self,
        story_quality: np.ndarray,
        outcomes: np.ndarray,
        metadata: pd.DataFrame,
        dimensions_to_explore: List[str],
        min_samples: int = 30,
        max_subdivisions: int = 200
    ) -> List[EmpiricalContext]:
        """
        Exhaustive empirical discovery across all dimensions.
        
        Parameters
        ----------
        story_quality : ndarray
            Computed ю (narrative quality) for all samples
        outcomes : ndarray
            Observed outcomes for all samples
        metadata : DataFrame
            All available metadata (genres, years, categories, etc.)
        dimensions_to_explore : list
            Column names to segment by
        min_samples : int
            Minimum samples required per subdivision
        max_subdivisions : int
            Maximum subdivisions to analyze (for performance)
        
        Returns
        -------
        contexts : list of EmpiricalContext
            All discovered contexts ranked by measured r
        """
        if self.verbose:
            print("\n" + "="*80)
            print("EMPIRICAL DISCOVERY ENGINE")
            print("="*80)
            print(f"\nData: {len(story_quality)} samples")
            print(f"Dimensions to explore: {dimensions_to_explore}")
            print(f"Approach: Measure r in ALL subdivisions, rank by strength")
        
        # Measure overall baseline
        r_overall, p_overall = stats.pearsonr(story_quality, outcomes)
        
        if self.verbose:
            print(f"\nBaseline (overall): r={r_overall:.3f} (p={p_overall:.6f})")
            print("\nSearching for contexts where r is HIGHER than baseline...")
        
        all_contexts = []
        
        # === SINGLE-DIMENSION SEARCH ===
        
        for dimension in dimensions_to_explore:
            if dimension not in metadata.columns:
                continue
            
            contexts = self._explore_dimension(
                dimension=dimension,
                story_quality=story_quality,
                outcomes=outcomes,
                metadata=metadata,
                min_samples=min_samples
            )
            
            all_contexts.extend(contexts)
        
        # === TWO-DIMENSION INTERACTIONS (if not too many) ===
        
        if len(dimensions_to_explore) <= 5 and len(all_contexts) < max_subdivisions:
            if self.verbose:
                print("\n--- INTERACTION EFFECTS (2-Way) ---")
            
            for dim1, dim2 in itertools.combinations(dimensions_to_explore[:3], 2):
                if dim1 in metadata.columns and dim2 in metadata.columns:
                    interaction_contexts = self._explore_interaction(
                        dim1, dim2, story_quality, outcomes, metadata, min_samples
                    )
                    all_contexts.extend(interaction_contexts)
        
        # === RANK BY EMPIRICAL STRENGTH ===
        
        all_contexts.sort(key=lambda x: abs(x.r_measured), reverse=True)
        
        # Assign overall ranks
        for i, ctx in enumerate(all_contexts, 1):
            ctx.rank_overall = i
        
        self.discoveries = all_contexts
        
        if self.verbose:
            print(f"\n✓ Discovered {len(all_contexts)} contexts")
            print(f"✓ Ranked by measured r (empirical strength)")
        
        return all_contexts
    
    def _explore_dimension(
        self,
        dimension: str,
        story_quality: np.ndarray,
        outcomes: np.ndarray,
        metadata: pd.DataFrame,
        min_samples: int
    ) -> List[EmpiricalContext]:
        """Explore a single dimension"""
        
        if self.verbose:
            print(f"\n--- {dimension.upper()} ---")
        
        contexts = []
        unique_values = metadata[dimension].unique()
        
        for rank, value in enumerate(unique_values, 1):
            mask = metadata[dimension] == value
            n = mask.sum()
            
            if n < min_samples:
                continue
            
            yu_subset = story_quality[mask]
            out_subset = outcomes[mask]
            
            # Need variation
            if len(np.unique(yu_subset)) < 2 or len(np.unique(out_subset)) < 2:
                continue
            
            # MEASURE r (pure empirical)
            r, p = stats.pearsonr(yu_subset, out_subset)
            
            # Significance
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            context = EmpiricalContext(
                name=f"{dimension}={value}",
                filters={dimension: value},
                n_samples=n,
                r_measured=r,
                r_squared=r**2,
                p_value=p,
                significance_level=sig,
                rank_overall=0,  # Will be assigned after global sort
                rank_within_dimension=rank,
                dimension=dimension
            )
            
            contexts.append(context)
            
            if self.verbose and n >= min_samples:
                print(f"  {str(value)[:40]:40s}: r={r:+.3f} (n={n:4d}, p={p:.4f}) {sig}")
        
        return contexts
    
    def _explore_interaction(
        self,
        dim1: str,
        dim2: str,
        story_quality: np.ndarray,
        outcomes: np.ndarray,
        metadata: pd.DataFrame,
        min_samples: int
    ) -> List[EmpiricalContext]:
        """Explore two-way interaction effects"""
        
        contexts = []
        
        # Get unique combinations
        for val1 in metadata[dim1].unique():
            for val2 in metadata[dim2].unique():
                mask = (metadata[dim1] == val1) & (metadata[dim2] == val2)
                n = mask.sum()
                
                if n < min_samples:
                    continue
                
                yu_subset = story_quality[mask]
                out_subset = outcomes[mask]
                
                if len(np.unique(yu_subset)) < 2 or len(np.unique(out_subset)) < 2:
                    continue
                
                r, p = stats.pearsonr(yu_subset, out_subset)
                
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                
                context = EmpiricalContext(
                    name=f"{dim1}={val1} × {dim2}={val2}",
                    filters={dim1: val1, dim2: val2},
                    n_samples=n,
                    r_measured=r,
                    r_squared=r**2,
                    p_value=p,
                    significance_level=sig,
                    rank_overall=0,
                    rank_within_dimension=0,
                    dimension=f"{dim1}×{dim2}"
                )
                
                contexts.append(context)
        
        return contexts
    
    def get_top_contexts(self, n: int = 10, min_r: float = 0.0) -> List[EmpiricalContext]:
        """Get top N contexts by measured r"""
        filtered = [c for c in self.discoveries if abs(c.r_measured) >= min_r]
        return filtered[:n]
    
    def get_passing_contexts(self, narrativity_estimator: Optional[Callable] = None) -> List[Dict]:
        """
        Estimate which discovered contexts would pass threshold.
        
        Uses measured r + estimated п and κ from context characteristics.
        """
        passing = []
        
        for ctx in self.discoveries[:20]:  # Top 20
            # Estimate п and κ based on measured r strength
            # (Data shows strength, we infer characteristics)
            
            if ctx.r_measured > 0.4:
                п_eff = 0.90  # Very strong r → high narrativity context
                κ_eff = 0.6   # Strong community/subjective judgment
            elif ctx.r_measured > 0.3:
                п_eff = 0.80
                κ_eff = 0.5
            elif ctx.r_measured > 0.2:
                п_eff = 0.70
                κ_eff = 0.4
            else:
                continue  # Too weak
            
            # Calculate Д
            Д = п_eff * ctx.r_measured * κ_eff
            efficiency = Д / п_eff
            
            if efficiency > 0.5:
                passing.append({
                    'context': str(ctx.name),
                    'r_measured': float(ctx.r_measured),
                    'п_inferred': float(п_eff),
                    'κ_inferred': float(κ_eff),
                    'Д': float(Д),
                    'efficiency': float(efficiency),
                    'n': int(ctx.n_samples)
                })
        
        return passing
    
    def print_discovery_report(self, top_n: int = 20):
        """Print formatted discovery report"""
        
        print("\n" + "="*80)
        print("EMPIRICAL DISCOVERY REPORT")
        print("="*80)
        print("\nData-driven approach: Measured, ranked, discovered")
        print(f"Total contexts found: {len(self.discoveries)}")
        
        # Group by dimension
        by_dimension = {}
        for ctx in self.discoveries:
            if ctx.dimension not in by_dimension:
                by_dimension[ctx.dimension] = []
            by_dimension[ctx.dimension].append(ctx)
        
        print(f"\nDimensions explored: {list(by_dimension.keys())}")
        
        # Top overall
        print(f"\n{'='*80}")
        print(f"TOP {top_n} CONTEXTS (Ranked by Measured r)")
        print("="*80)
        print(f"{'Rank':<6} {'Context':<45} {'r':<10} {'R²':<8} {'n':<8} {'Sig'}")
        print("-"*85)
        
        for ctx in self.discoveries[:top_n]:
            print(f"{ctx.rank_overall:<6} {ctx.name[:44]:<45} {ctx.r_measured:>+.3f}   "
                  f"{ctx.r_squared:>6.3f}  {ctx.n_samples:>6}  {ctx.significance_level}")
        
        # Best per dimension
        print(f"\n{'='*80}")
        print("BEST CONTEXT PER DIMENSION")
        print("="*80)
        
        for dimension, contexts in sorted(by_dimension.items()):
            best = max(contexts, key=lambda x: abs(x.r_measured))
            print(f"  {dimension:20s}: {best.name:40s} r={best.r_measured:+.3f}")
        
        # Passing contexts
        passing = self.get_passing_contexts()
        
        if passing:
            print(f"\n{'='*80}")
            print("✓ CONTEXTS THAT PASS THRESHOLD (Data-Discovered)")
            print("="*80)
            for p in passing:
                print(f"\n  • {p['context']}")
                print(f"    Measured r: {p['r_measured']:.3f} (DATA FACT)")
                print(f"    Inferred п_eff: {p['п_inferred']:.2f}")
                print(f"    Inferred κ: {p['κ_inferred']:.2f}")
                print(f"    Calculated Д: {p['Д']:.3f}")
                print(f"    Efficiency: {p['efficiency']:.3f} ✓ PASSES")
        else:
            print(f"\n⚠️  No contexts pass threshold with current data")
            print(f"    (Best r={self.discoveries[0].r_measured:.3f} needs stronger subdivision)")
    
    def export_discoveries(self, output_path: str):
        """Export discoveries to JSON"""
        import json
        
        data = {
            'total_contexts': len(self.discoveries),
            'top_contexts': [
                {
                    'rank': int(ctx.rank_overall),
                    'name': str(ctx.name),
                    'dimension': str(ctx.dimension),
                    'r': float(ctx.r_measured),
                    'r_squared': float(ctx.r_squared),
                    'n': int(ctx.n_samples),
                    'p': float(ctx.p_value),
                    'significant': bool(ctx.significance_level != 'ns')
                }
                for ctx in self.discoveries[:50]
            ],
            'passing_contexts': self.get_passing_contexts(),
            'approach': 'pure_empirical_discovery',
            'principle': 'Data leads, theory explains'
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def discover_optimal_contexts_for_domain(
    domain_name: str,
    story_quality: np.ndarray,
    outcomes: np.ndarray,
    metadata_dict: List[Dict],
    dimensions: List[str],
    min_samples: int = 30
) -> List[EmpiricalContext]:
    """
    Main entry point: Discover where narrative is strongest in a domain.
    
    Parameters
    ----------
    domain_name : str
        Domain being analyzed
    story_quality : ndarray
        Computed ю values
    outcomes : ndarray
        Outcomes
    metadata_dict : list of dict
        Metadata for each sample
    dimensions : list
        Dimensions to explore (e.g., ['genre', 'year', 'category'])
    min_samples : int
        Minimum per subdivision
    
    Returns
    -------
    contexts : list
        Discovered contexts ranked by measured r
    """
    print("="*80)
    print(f"DATA-DRIVEN DISCOVERY: {domain_name}")
    print("="*80)
    print("\nPrinciple: MEASURE EVERYTHING, let data reveal patterns")
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata_dict)
    df['story_quality'] = story_quality
    df['outcome'] = outcomes
    
    # Discover
    engine = EmpiricalDiscoveryEngine(verbose=True)
    contexts = engine.discover_all_contexts(
        story_quality=story_quality,
        outcomes=outcomes,
        metadata=df,
        dimensions_to_explore=dimensions,
        min_samples=min_samples
    )
    
    # Report
    engine.print_discovery_report(top_n=15)
    
    # Export
    output_path = f"{domain_name.lower().replace(' ', '_')}_empirical_discoveries.json"
    engine.export_discoveries(output_path)
    
    print(f"\n✓ Discoveries exported: {output_path}")
    
    return contexts

