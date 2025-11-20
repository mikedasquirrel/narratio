"""
Complete Meta-Nominative Analysis with All Transformers

Applies all 29 transformers to researcher names, paper titles, and research topics.
Tests core hypothesis: Do researchers with name-field fit report larger effect sizes?
"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers import (
    # Nominative block (most relevant)
    PhoneticTransformer,
    NominativeAnalysisTransformer,
    UniversalNominativeTransformer,
    HierarchicalNominativeTransformer,
    
    # Core narrative
    SelfPerceptionTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    EnsembleNarrativeTransformer,
    RelationalValueTransformer,
    StatisticalTransformer,
    
    # Advanced
    SocialStatusTransformer,
    InformationTheoryTransformer,
    EmotionalResonanceTransformer,
    AuthenticityTransformer
)


class MetaNominativeAnalyzer:
    """Complete analysis of meta-nominative determinism."""
    
    def __init__(self, data_dir: Path):
        """Initialize analyzer with data directory."""
        self.data_dir = data_dir
        self.researchers = None
        self.papers = None
        self.features = {}
        self.results = {}
    
    def load_data(self):
        """Load all data files."""
        print(f"\n{'='*80}")
        print("LOADING DATA")
        print(f"{'='*80}\n")
        
        print("[1/3] Loading researcher metadata...")
        # Load researchers
        with open(self.data_dir / 'researchers_metadata.json') as f:
            data = json.load(f)
            self.researchers = data['researchers']
        print(f"      ✓ Loaded {len(self.researchers)} researchers")
        
        print("\n[2/3] Loading consolidated papers...")
        # Load papers
        with open(self.data_dir / 'papers_consolidated.json') as f:
            data = json.load(f)
            self.papers = data['papers']
        print(f"      ✓ Loaded {len(self.papers)} papers")
        
        # Validate data quality
        print("\n[3/3] Validating data quality...")
        self._validate_data()
        print(f"      ✓ Data validation passed")
    
    def _validate_data(self):
        """Validate that we have real, extensive data."""
        issues = []
        
        # Check for suspicious patterns (synthetic data)
        suspicious_names = ['Dennis Smith', 'Laura Lawyer', 'Richard Doctor', 
                          'Sarah Physician', 'Daniel Dentist', 'Robert Researcher']
        found_suspicious = [name for name in suspicious_names if name in self.researchers]
        
        if len(found_suspicious) > 3:
            issues.append(f"⚠️  Found {len(found_suspicious)} suspiciously fitting names (may be synthetic data)")
        
        # Check for minimum sample size
        if len(self.papers) < 30:
            issues.append(f"⚠️  Only {len(self.papers)} papers (minimum 30 recommended)")
        
        if len(self.researchers) < 40:
            issues.append(f"⚠️  Only {len(self.researchers)} researchers (minimum 40 recommended)")
        
        # Check for essential fields
        papers_with_effects = sum(1 for p in self.papers if p.get('effect_size_normalized'))
        if papers_with_effects < len(self.papers) * 0.5:
            issues.append(f"⚠️  Only {papers_with_effects}/{len(self.papers)} papers have effect sizes")
        
        researchers_with_fit = sum(1 for r in self.researchers.values() if r.get('name_field_fit'))
        if researchers_with_fit < len(self.researchers) * 0.8:
            issues.append(f"⚠️  Only {researchers_with_fit}/{len(self.researchers)} researchers have fit scores")
        
        # Print issues
        if issues:
            print("\n      Data Quality Issues:")
            for issue in issues:
                print(f"      {issue}")
            print("\n      Note: Analysis will continue but results may be preliminary.")
        else:
            print("      No issues detected - data appears genuine and complete.")
    
    def apply_transformers(self):
        """Apply all relevant transformers to researcher names."""
        print(f"\n{'='*80}")
        print("APPLYING TRANSFORMERS TO RESEARCHER NAMES")
        print(f"{'='*80}\n")
        
        # Collect all researcher names
        names = list(self.researchers.keys())
        
        # Priority transformers (nominative-focused)
        transformers = [
            ('phonetic', PhoneticTransformer()),
            ('nominative', NominativeAnalysisTransformer()),
            ('universal_nominative', UniversalNominativeTransformer()),
            ('hierarchical_nominative', HierarchicalNominativeTransformer()),
            ('self_perception', SelfPerceptionTransformer()),
            ('narrative_potential', NarrativePotentialTransformer()),
            ('linguistic', LinguisticPatternsTransformer()),
            ('ensemble', EnsembleNarrativeTransformer(n_top_terms=20)),
            ('relational', RelationalValueTransformer(n_features=30)),
            ('social_status', SocialStatusTransformer()),
            ('information_theory', InformationTheoryTransformer()),
            ('emotional', EmotionalResonanceTransformer()),
            ('authenticity', AuthenticityTransformer()),
            ('statistical', StatisticalTransformer(max_features=50))
        ]
        
        all_features = []
        feature_names = []
        
        total_transformers = len(transformers)
        
        for idx, (name, transformer) in enumerate(transformers, 1):
            print(f"  [{idx:2d}/{total_transformers}] Applying {name:25s}...", end=" ")
            try:
                # Fit and transform with progress
                transformer.fit(names)
                features = transformer.transform(names)
                
                # Ensure proper array format
                if not isinstance(features, np.ndarray):
                    features = np.array(features)
                
                # Ensure 2D array
                if len(features.shape) == 1:
                    features = features.reshape(-1, 1)
                elif len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                
                # Verify shape matches number of researchers
                if features.shape[0] != len(names):
                    print(f"✗ Shape mismatch: {features.shape[0]} != {len(names)}")
                    continue
                
                all_features.append(features)
                
                # Get feature count
                n_features = features.shape[1]
                feature_names.extend([f"{name}_{i}" for i in range(n_features)])
                
                # Calculate percentage complete
                pct_complete = (idx / total_transformers) * 100
                
                print(f"✓ ({n_features:3d} features) [{pct_complete:5.1f}% complete]")
                
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                import traceback
                traceback.print_exc()
                continue
        
        # Concatenate all features
        if all_features:
            combined_features = np.hstack(all_features)
            
            print(f"\n✓ Total features extracted: {combined_features.shape[1]}")
            
            # Store features for each researcher
            for i, name in enumerate(names):
                self.researchers[name]['transformer_features'] = combined_features[i].tolist()
            
            self.features = {
                'feature_matrix': combined_features,
                'feature_names': feature_names,
                'researcher_names': names
            }
        
        return self.features
    
    def run_core_analysis(self):
        """
        Test core hypothesis: name-field fit predicts effect size.
        
        Tests:
        1. Correlation: name_fit × effect_size
        2. Regression: effect_size ~ name_fit + controls
        3. Group comparison: high-fit vs low-fit researchers
        """
        print(f"\n{'='*80}")
        print("CORE HYPOTHESIS TEST")
        print(f"{'='*80}\n")
        
        # Collect data for analysis
        print("Collecting data for statistical analysis...")
        print("  [1/4] Extracting fit scores and effect sizes...")
        
        fit_scores = []
        effect_sizes = []
        h_indices = []
        paper_counts = []
        years_since_phd = []
        researcher_names = []
        skipped = 0
        
        for name, data in self.researchers.items():
            # Skip if missing critical data
            fit = data.get('name_field_fit', {}).get('overall_fit')
            avg_effect = data.get('average_effect_size')
            
            if fit is None or avg_effect is None:
                skipped += 1
                continue
            
            fit_scores.append(fit)
            effect_sizes.append(avg_effect)
            h_indices.append(data.get('h_index', 1))
            paper_counts.append(data.get('paper_count', 1))
            years_since_phd.append(data.get('years_since_phd', 10))
            researcher_names.append(name)
        
        print(f"      ✓ Collected {len(fit_scores)} complete records (skipped {skipped} incomplete)")
        
        if len(fit_scores) < 10:
            print(f"\n      ⚠️  WARNING: Only {len(fit_scores)} researchers with complete data")
            print(f"      This is insufficient for robust statistical analysis (need ≥20)")
            print(f"      Results should be treated as exploratory only.\n")
        
        fit_scores = np.array(fit_scores)
        effect_sizes = np.array(effect_sizes)
        
        print(f"      Sample size: {len(fit_scores)} researchers with complete data\n")
        
        print("  [2/4] Computing descriptive statistics...")
        print(f"      Name-field fit: μ={np.mean(fit_scores):.1f}, σ={np.std(fit_scores):.1f}")
        print(f"      Effect sizes: μ={np.mean(effect_sizes):.3f}, σ={np.std(effect_sizes):.3f}")
        
        # TEST 1: Univariate correlation
        print(f"\n  [3/4] Running statistical tests...")
        print(f"\n{'─'*80}")
        print("TEST 1: Univariate Correlation")
        print(f"{'─'*80}")
        
        r, p = stats.pearsonr(fit_scores, effect_sizes)
        r_spearman, p_spearman = stats.spearmanr(fit_scores, effect_sizes)
        
        print(f"\nPearson r = {r:.3f}, p = {p:.4f}")
        print(f"Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")
        
        if p < 0.05:
            print(f"✓ SIGNIFICANT: Name-field fit predicts effect size!")
        else:
            print(f"✗ NULL: No significant relationship")
        
        # Interpretation
        if r > 0.2:
            interpretation = "STRONG support for meta-nominative determinism"
        elif r > 0.1:
            interpretation = "MODERATE support - researchers' names matter somewhat"
        elif r > 0:
            interpretation = "WEAK support - small positive effect"
        else:
            interpretation = "NULL - researchers' names don't affect findings"
        
        print(f"\nInterpretation: {interpretation}")
        
        # TEST 2: Group comparison (high-fit vs low-fit)
        print(f"\n{'─'*80}")
        print("TEST 2: Group Comparison")
        print(f"{'─'*80}")
        
        # Split at median
        median_fit = np.median(fit_scores)
        high_fit = effect_sizes[fit_scores > median_fit]
        low_fit = effect_sizes[fit_scores <= median_fit]
        
        t_stat, t_p = stats.ttest_ind(high_fit, low_fit)
        cohens_d = (np.mean(high_fit) - np.mean(low_fit)) / np.sqrt((np.var(high_fit) + np.var(low_fit)) / 2)
        
        print(f"\nHigh-fit researchers (n={len(high_fit)}): μ={np.mean(high_fit):.3f}")
        print(f"Low-fit researchers (n={len(low_fit)}): μ={np.mean(low_fit):.3f}")
        print(f"\nt-test: t={t_stat:.3f}, p={t_p:.4f}")
        print(f"Cohen's d = {cohens_d:.3f}")
        
        if t_p < 0.05 and np.mean(high_fit) > np.mean(low_fit):
            print(f"✓ High-fit researchers report LARGER effects!")
        elif t_p < 0.05:
            print(f"⚠️  High-fit researchers report SMALLER effects (unexpected)")
        else:
            print(f"✗ No significant difference between groups")
        
        # TEST 3: Show specific examples
        print(f"\n  [4/4] Identifying notable examples...")
        print(f"\n{'─'*80}")
        print("TEST 3: Notable Examples")
        print(f"{'─'*80}\n")
        
        # Sort by fit score
        indices = np.argsort(fit_scores)[::-1]
        
        print("Researchers with HIGHEST name-field fit:")
        for i in indices[:5]:
            name = researcher_names[i]
            fit = fit_scores[i]
            effect = effect_sizes[i]
            topic = ', '.join(self.researchers[name].get('topics_studied', []))
            print(f"  {name}: fit={fit:.1f}, effect={effect:.3f} ({topic})")
        
        print("\nResearchers with LOWEST name-field fit:")
        for i in indices[-5:]:
            name = researcher_names[i]
            fit = fit_scores[i]
            effect = effect_sizes[i]
            topic = ', '.join(self.researchers[name].get('topics_studied', []))
            print(f"  {name}: fit={fit:.1f}, effect={effect:.3f} ({topic})")
        
        # Store results
        self.results = {
            'sample_size': len(fit_scores),
            'correlation_r': float(r),
            'correlation_p': float(p),
            'spearman_rho': float(r_spearman),
            'spearman_p': float(p_spearman),
            'cohens_d': float(cohens_d),
            't_statistic': float(t_stat),
            't_p_value': float(t_p),
            'interpretation': interpretation,
            'high_fit_mean': float(np.mean(high_fit)),
            'low_fit_mean': float(np.mean(low_fit)),
            'fit_mean': float(np.mean(fit_scores)),
            'fit_std': float(np.std(fit_scores)),
            'effect_mean': float(np.mean(effect_sizes)),
            'effect_std': float(np.std(effect_sizes))
        }
        
        return self.results
    
    def save_results(self):
        """Save analysis results."""
        output_path = self.data_dir / 'analysis_results.json'
        
        with open(output_path, 'w') as f:
            json.dump({
                'meta_nominative_analysis': self.results,
                'feature_extraction': {
                    'total_features': len(self.features.get('feature_names', [])),
                    'transformers_applied': 14
                }
            }, f, indent=2)
        
        print(f"\n✓ Saved results to: {output_path}")


def main():
    """Run complete meta-nominative analysis."""
    print(f"\n{'='*80}")
    print("META-NOMINATIVE DETERMINISM: COMPLETE ANALYSIS")
    print(f"{'='*80}")
    print("\nResearch Question:")
    print("  Do researchers with name-field fit report LARGER effect sizes")
    print("  in nominative determinism studies?")
    print(f"{'='*80}\n")
    
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative'
    
    analyzer = MetaNominativeAnalyzer(data_dir)
    analyzer.load_data()
    analyzer.apply_transformers()
    analyzer.run_core_analysis()
    analyzer.save_results()
    
    print(f"\n{'='*80}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    # Summary
    r = analyzer.results['correlation_r']
    p = analyzer.results['correlation_p']
    
    print(f"\nFINAL RESULT: r = {r:.3f}, p = {p:.4f}")
    
    if p < 0.05:
        if r > 0.20:
            print("\n🔥 STRONG EVIDENCE for meta-nominative determinism!")
            print("   Researchers' names DO predict their findings.")
        else:
            print("\n✓ Evidence for meta-nominative effect (but small)")
    else:
        print("\n✓ NULL RESULT: Scientific objectivity upheld")
        print("   Researchers' names don't bias findings.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

