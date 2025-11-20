"""
Data-Driven Taxonomic Optimizer

LET THE DATA TELL YOU where narrative is most prominent.

Approach:
1. Decompose domain by ALL available dimensions (genre, stage, scale, etc.)
2. MEASURE actual r (correlation) in each subdivision
3. FIND where r is highest (data tells us where narrative matters)
4. OPTIMIZE formula for those empirically-identified contexts
5. THEN explain why with theory

NOT: Theory predicts → measure
BUT: Measure → discover → explain with theory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from collections import defaultdict


@dataclass
class EmpiricalSubdomain:
    """Discovered subdomain with measured narrative strength"""
    name: str
    filter_criteria: Dict[str, Any]
    n_samples: int
    r_measured: float
    p_value: float
    narrativity_calculated: float
    coupling_estimated: float
    efficiency: float
    passes: bool
    rank: int  # Rank by r (1 = highest)


class DataDrivenOptimizer:
    """
    Discover where narrative is most prominent through data exploration.
    
    Philosophy: Don't predict where narrative should matter.
    MEASURE where it actually matters, then optimize those contexts.
    """
    
    def __init__(self):
        self.discoveries: List[EmpiricalSubdomain] = []
    
    def discover_optimal_subdivisions(
        self,
        data: pd.DataFrame,
        text_column: str,
        outcome_column: str,
        narrative_quality_column: str,
        segmentation_columns: List[str],
        min_samples: int = 30
    ) -> List[EmpiricalSubdomain]:
        """
        Discover where narrative is most prominent by exhaustive segmentation.
        
        Parameters
        ----------
        data : DataFrame
            Complete dataset with all available dimensions
        text_column : str
            Column with narrative text
        outcome_column : str
            Column with outcomes
        narrative_quality_column : str
            Column with computed story quality (ю)
        segmentation_columns : list
            Columns to segment by (genre, stage, category, etc.)
        min_samples : int
            Minimum samples required per subdivision
        
        Returns
        -------
        discoveries : list of EmpiricalSubdomain
            Subdomains ranked by measured r (highest first)
        
        This is PURE DATA EXPLORATION - no theoretical predictions!
        """
        print("\n" + "="*80)
        print("DATA-DRIVEN DISCOVERY: Where Is Narrative Most Prominent?")
        print("="*80)
        print("\nApproach: Measure r in ALL subdivisions, rank by strength")
        print("(Let data tell us, don't predict from theory)")
        
        subdivisions = []
        
        # For each segmentation dimension
        for seg_col in segmentation_columns:
            if seg_col not in data.columns:
                continue
            
            print(f"\n--- Segmenting by: {seg_col} ---")
            
            # Get unique values
            unique_values = data[seg_col].unique()
            
            for value in unique_values:
                # Filter to this subdivision
                mask = data[seg_col] == value
                subset = data[mask]
                
                if len(subset) < min_samples:
                    continue  # Skip small samples
                
                # MEASURE r (let data speak!)
                ю = subset[narrative_quality_column].values
                ❊ = subset[outcome_column].values
                
                if len(np.unique(ю)) < 2 or len(np.unique(❊)) < 2:
                    continue  # Skip if no variation
                
                r, p = stats.pearsonr(ю, ❊)
                
                subdivisions.append({
                    'name': f"{seg_col}={value}",
                    'filter': {seg_col: value},
                    'n': len(subset),
                    'r': r,
                    'p': p
                })
                
                print(f"  {value:30s}: r={r:+.3f} (p={p:.4f}, n={len(subset)})")
        
        # RANK by measured r (highest narrative strength first)
        subdivisions.sort(key=lambda x: x['r'], reverse=True)
        
        print("\n" + "="*80)
        print("TOP 10 HIGHEST NARRATIVE STRENGTH (Measured from Data)")
        print("="*80)
        
        for i, sub in enumerate(subdivisions[:10], 1):
            print(f"{i:2d}. {sub['name']:40s}: r={sub['r']:+.3f} (n={sub['n']})")
        
        # Convert to EmpiricalSubdomain objects
        # (Now we can add theory to explain, but DATA discovered these)
        
        return subdivisions
    
    def optimize_discovered_contexts(
        self,
        discoveries: List[Dict],
        overall_narrativity: float,
        overall_efficiency: float
    ) -> List[EmpiricalSubdomain]:
        """
        Optimize formula for empirically-discovered high-r contexts.
        
        Now that DATA told us where narrative is strong,
        optimize the formula for those specific contexts.
        """
        results = []
        
        print("\n" + "="*80)
        print("OPTIMIZING EMPIRICALLY-DISCOVERED CONTEXTS")
        print("="*80)
        print("\nData told us where narrative is strongest.")
        print("Now optimize formula for those contexts.")
        
        for rank, discovery in enumerate(discoveries[:10], 1):  # Top 10
            # Estimate π and κ for this context (theory explains data)
            п_estimated = self._estimate_narrativity_from_r(discovery['r'], overall_narrativity)
            κ_estimated = self._estimate_coupling_from_context(discovery['name'])
            
            # Calculate Д
            Д = п_estimated * discovery['r'] * κ_estimated
            efficiency = Д / п_estimated if п_estimated > 0 else 0
            passes = efficiency > 0.5
            
            result = EmpiricalSubdomain(
                name=discovery['name'],
                filter_criteria=discovery['filter'],
                n_samples=discovery['n'],
                r_measured=discovery['r'],
                p_value=discovery['p'],
                narrativity_calculated=п_estimated,
                coupling_estimated=κ_estimated,
                efficiency=efficiency,
                passes=passes,
                rank=rank
            )
            
            results.append(result)
            
            status = "✓ PASS" if passes else "❌ FAIL"
            print(f"{rank:2d}. {status} {discovery['name']:40s}: "
                  f"r={discovery['r']:+.3f}, eff={efficiency:.3f}")
        
        self.discoveries = results
        return results
    
    def _estimate_narrativity_from_r(self, r: float, baseline_п: float) -> float:
        """
        Estimate effective π for a context based on measured r.
        
        Higher r suggests higher effective narrativity in this context.
        """
        # If r is much higher than baseline, context is more narrative
        if r > 0.5:
            return min(0.95, baseline_п * 1.3)  # High narrative context
        elif r > 0.3:
            return min(0.85, baseline_п * 1.1)  # Moderate increase
        elif r > 0:
            return baseline_п  # Similar to baseline
        else:
            return max(0.1, baseline_п * 0.8)  # Lower narrative
    
    def _estimate_coupling_from_context(self, context_name: str) -> float:
        """
        Estimate coupling based on context characteristics.
        
        This is informed by theory but validated by data strength.
        """
        name_lower = context_name.lower()
        
        # High coupling contexts (narrator influences outcome)
        if any(word in name_lower for word in ['lgbt', 'identity', 'personal', 'self', 'diary']):
            return 0.6
        elif any(word in name_lower for word in ['character', 'drama', 'biography']):
            return 0.5
        elif any(word in name_lower for word in ['sports', 'underdog', 'redemption']):
            return 0.5
        # Market-judged
        elif any(word in name_lower for word in ['action', 'commercial', 'blockbuster']):
            return 0.3
        # Performance-judged
        elif any(word in name_lower for word in ['game', 'match', 'competition']):
            return 0.3
        else:
            return 0.4  # Default moderate
    
    def print_discovery_report(self):
        """Print what data discovered"""
        if not self.discoveries:
            print("No discoveries yet")
            return
        
        print("\n" + "="*80)
        print("DATA-DRIVEN DISCOVERIES")
        print("="*80)
        print("\nData told us narrative is MOST prominent in:")
        
        passing = [d for d in self.discoveries if d.passes]
        high_r = [d for d in self.discoveries if d.r_measured > 0.4]
        
        if passing:
            print(f"\n✓ {len(passing)} contexts PASS threshold:")
            for d in passing:
                print(f"  • {d.name:40s}: r={d.r_measured:+.3f}, eff={d.efficiency:.3f}")
        
        if high_r and not passing:
            print(f"\n⚠️  {len(high_r)} contexts show strong effects (r>0.4) but don't pass:")
            for d in high_r:
                print(f"  • {d.name:40s}: r={d.r_measured:+.3f}, eff={d.efficiency:.3f}")
        
        print("\n" + "="*80)
        print("PATTERN DISCOVERY")
        print("="*80)
        print("\nEmpirical patterns (from data, not theory):")
        
        # Group by characteristics
        character_driven = [d for d in self.discoveries if d.narrativity_calculated > 0.8]
        plot_driven = [d for d in self.discoveries if d.narrativity_calculated < 0.5]
        
        if character_driven:
            print(f"\nCharacter-driven contexts (high measured r):")
            for d in character_driven[:5]:
                print(f"  • {d.name}: r={d.r_measured:+.3f}")
        
        if plot_driven:
            print(f"\nPlot/performance-driven contexts (low measured r):")
            for d in plot_driven[:5]:
                print(f"  • {d.name}: r={d.r_measured:+.3f}")


def apply_to_real_data(
    data_path: str,
    text_col: str,
    outcome_col: str,
    story_quality_col: str,
    segment_cols: List[str]
):
    """
    Apply data-driven discovery to real dataset.
    
    This is the main entry point - give it your data,
    it discovers where narrative is strongest.
    """
    print("="*80)
    print("DATA-DRIVEN OPTIMIZATION")
    print("="*80)
    print("\nLoading data and discovering optimal contexts...")
    
    # Load data
    data = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_json(data_path)
    
    print(f"✓ Loaded {len(data)} samples")
    print(f"  Segmentation dimensions: {segment_cols}")
    
    # Discover
    optimizer = DataDrivenOptimizer()
    discoveries = optimizer.discover_optimal_subdivisions(
        data=data,
        text_column=text_col,
        outcome_column=outcome_col,
        narrative_quality_column=story_quality_col,
        segmentation_columns=segment_cols,
        min_samples=30
    )
    
    # Optimize discovered contexts
    # Use overall domain efficiency as baseline
    overall_r = data[story_quality_col].corr(data[outcome_col])
    overall_eff = 0.04  # Example
    
    results = optimizer.optimize_discovered_contexts(
        discoveries,
        overall_narrativity=0.65,  # Will be calculated from data
        overall_efficiency=overall_eff
    )
    
    optimizer.print_discovery_report()
    
    return results


# Global optimizer
_global_data_optimizer = None


def get_data_driven_optimizer() -> DataDrivenOptimizer:
    """Get or create global data-driven optimizer"""
    global _global_data_optimizer
    if _global_data_optimizer is None:
        _global_data_optimizer = DataDrivenOptimizer()
    return _global_data_optimizer

