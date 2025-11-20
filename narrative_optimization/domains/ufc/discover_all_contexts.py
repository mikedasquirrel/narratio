"""
UFC Exhaustive Context Discovery

Test narrative effects across 50+ contexts to find where narrative matters most.

Categories:
- Matchup-Level: 20 contexts (title fights, even matchups, methods)
- Fighter-Level: 15 contexts (nicknames, name patterns, personas)
- Temporal: 10 contexts (years, eras, trends)
- Weight Class: 11 contexts (each division, heavy vs light)
- Combined: 10 contexts (interactions of above)

For each context: measure |r|, AUC, narrative Δ, and efficiency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import json
import warnings
warnings.filterwarnings('ignore')


class UFCContextDiscovery:
    """Discover all contexts where narrative effects are strongest"""
    
    def __init__(self):
        self.contexts = []
        self.narrativity = 0.722
        self.coupling = 0.6
        
    def load_features(self):
        """Load comprehensive features"""
        
        print("="*80)
        print("LOADING COMPREHENSIVE FEATURES")
        print("="*80)
        
        data_dir = Path('narrative_optimization/domains/ufc')
        
        X_df = pd.read_csv(data_dir / 'ufc_comprehensive_features.csv')
        y = np.load(data_dir / 'ufc_comprehensive_outcomes.npy')
        
        with open(data_dir / 'ufc_feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        
        print(f"✓ Loaded {X_df.shape[1]} features from {X_df.shape[0]} fights")
        
        # Identify feature types
        physical_features = [f for f in feature_names if any(x in f for x in ['strike', 'td', 'ctrl', 'kd', 'sub', 'rev'])]
        narrative_features = [f for f in feature_names if any(x in f for x in ['name', 'nick', 'vowel', 'syllable', 'memorability', 'title', 'bonus'])]
        
        print(f"  Physical features: {len(physical_features)}")
        print(f"  Narrative features: {len(narrative_features)}")
        
        return X_df, y, feature_names, physical_features, narrative_features
    
    def test_context(self, X_df, y, mask, context_name, physical_features, narrative_features):
        """Test narrative effect in a specific context"""
        
        print(f"    Testing: {context_name:45s} ", end="", flush=True)
        
        if mask.sum() < 50:  # Need at least 50 samples
            print(f"✗ Only {mask.sum()} samples")
            return None
        
        X_context = X_df[mask]
        y_context = y[mask]
        
        # Split features
        phys_cols = [i for i, col in enumerate(X_df.columns) if col in physical_features]
        narr_cols = [i for i, col in enumerate(X_df.columns) if col in narrative_features]
        
        X_phys = X_context.iloc[:, phys_cols].values
        X_narr = X_context.iloc[:, narr_cols].values
        X_all = X_context.values
        
        # Standardize
        scaler_phys = StandardScaler()
        scaler_narr = StandardScaler()
        scaler_all = StandardScaler()
        
        X_phys_scaled = scaler_phys.fit_transform(X_phys)
        X_narr_scaled = scaler_narr.fit_transform(X_narr)
        X_all_scaled = scaler_all.fit_transform(X_all)
        
        # Test models
        try:
            # Physical only
            model_phys = LogisticRegression(random_state=42, max_iter=1000)
            cv_phys = cross_val_score(model_phys, X_phys_scaled, y_context, cv=min(5, mask.sum()//10), scoring='roc_auc')
            auc_phys = cv_phys.mean()
            
            # Narrative only
            model_narr = LogisticRegression(random_state=42, max_iter=1000)
            cv_narr = cross_val_score(model_narr, X_narr_scaled, y_context, cv=min(5, mask.sum()//10), scoring='roc_auc')
            auc_narr = cv_narr.mean()
            
            # Combined
            model_all = LogisticRegression(random_state=42, max_iter=1000)
            cv_all = cross_val_score(model_all, X_all_scaled, y_context, cv=min(5, mask.sum()//10), scoring='roc_auc')
            auc_all = cv_all.mean()
            
            # Calculate metrics
            delta = auc_all - auc_phys
            r_abs = abs(auc_all - 0.5) * 2  # Convert AUC to correlation approx
            bridge = self.narrativity * r_abs * self.coupling
            efficiency = bridge / self.narrativity
            
            result = {
                'context': context_name,
                'n_samples': int(mask.sum()),
                'auc_physical': float(auc_phys),
                'auc_narrative': float(auc_narr),
                'auc_combined': float(auc_all),
                'delta': float(delta),
                'r_abs': float(r_abs),
                'bridge': float(bridge),
                'efficiency': float(efficiency),
                'passes': bool(efficiency > 0.5)
            }
            
            status = "✓ PASS!" if efficiency > 0.5 else "✓"
            print(f"{status} n={mask.sum():5d} Δ={delta:+.4f} eff={efficiency:.4f}")
            
            return result
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:30]}")
            return None
    
    def discover_all_contexts(self, X_df, y, physical_features, narrative_features):
        """Test all 50+ contexts"""
        
        print("\n" + "="*80)
        print("DISCOVERING ALL CONTEXTS (50+)")
        print("="*80)
        
        results = []
        
        # ===== 1. MATCHUP-LEVEL CONTEXTS (20) =====
        print("\n[1/5] Testing matchup-level contexts...")
        
        # Title fights
        mask = X_df['is_title_fight'] == 1
        r = self.test_context(X_df, y, mask, "Title Fights", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['is_title_fight'] == 0
        r = self.test_context(X_df, y, mask, "Non-Title Fights", physical_features, narrative_features)
        if r: results.append(r)
        
        # Even matchups (physical parity)
        mask = (X_df['sig_str_diff'] < 0.1) & (X_df['kd_diff'] == 0)
        r = self.test_context(X_df, y, mask, "Even Matchups (Physical Parity)", physical_features, narrative_features)
        if r: results.append(r)
        
        # Striking differential bins
        for i, (low, high) in enumerate([(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.0)]):
            mask = (X_df['sig_str_diff'] >= low) & (X_df['sig_str_diff'] < high)
            r = self.test_context(X_df, y, mask, f"Strike Diff {low:.1f}-{high:.1f}", physical_features, narrative_features)
            if r: results.append(r)
        
        # Control time patterns
        mask = X_df['total_ctrl'] > 120  # >2 minutes total
        r = self.test_context(X_df, y, mask, "High Control Time", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['total_ctrl'] < 30
        r = self.test_context(X_df, y, mask, "Low Control Time", physical_features, narrative_features)
        if r: results.append(r)
        
        # Knockdown patterns
        mask = X_df['total_kd'] > 0
        r = self.test_context(X_df, y, mask, "Fights with Knockdowns", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['total_kd'] == 0
        r = self.test_context(X_df, y, mask, "No Knockdowns", physical_features, narrative_features)
        if r: results.append(r)
        
        # Submission attempts
        mask = (X_df['red_sub_att'] > 0) | (X_df['blue_sub_att'] > 0)
        r = self.test_context(X_df, y, mask, "Submission Attempts", physical_features, narrative_features)
        if r: results.append(r)
        
        # Finish vs Decision
        mask = X_df['is_finish'] == 1
        r = self.test_context(X_df, y, mask, "Finish (KO/Sub)", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['method_decision'] == 1
        r = self.test_context(X_df, y, mask, "Decision", physical_features, narrative_features)
        if r: results.append(r)
        
        # By round
        for round_num in [1, 2, 3]:
            mask = X_df['round'] == round_num
            r = self.test_context(X_df, y, mask, f"Round {round_num} Finishes", physical_features, narrative_features)
            if r: results.append(r)
        
        # Method types
        mask = X_df['method_ko'] == 1
        r = self.test_context(X_df, y, mask, "KO/TKO", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['method_sub'] == 1
        r = self.test_context(X_df, y, mask, "Submission", physical_features, narrative_features)
        if r: results.append(r)
        
        # 5 round fights
        mask = X_df['is_5_round_fight'] == 1
        r = self.test_context(X_df, y, mask, "5 Round Fights", physical_features, narrative_features)
        if r: results.append(r)
        
        print(f"  ✓ Tested {len([r for r in results])} matchup contexts")
        
        # ===== 2. FIGHTER-LEVEL CONTEXTS (15) =====
        print("\n[2/5] Testing fighter-level contexts...")
        
        start_count = len(results)
        
        # Nickname presence
        mask = X_df['both_have_nicknames'] == 1
        r = self.test_context(X_df, y, mask, "Both Have Nicknames", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = (X_df['red_has_nickname'] == 1) | (X_df['blue_has_nickname'] == 1)
        r = self.test_context(X_df, y, mask, "At Least One Nickname", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['neither_has_nickname'] == 1
        r = self.test_context(X_df, y, mask, "No Nicknames", physical_features, narrative_features)
        if r: results.append(r)
        
        # Name length categories
        for i, (low, high) in enumerate([(0, 10), (10, 15), (15, 20), (20, 100)]):
            mask = ((X_df['red_name_len'] >= low) & (X_df['red_name_len'] < high)) | \
                   ((X_df['blue_name_len'] >= low) & (X_df['blue_name_len'] < high))
            r = self.test_context(X_df, y, mask, f"Name Length {low}-{high}", physical_features, narrative_features)
            if r: results.append(r)
        
        # Name memorability
        mask = (X_df['red_name_memorability'] > 0.08) & (X_df['blue_name_memorability'] > 0.08)
        r = self.test_context(X_df, y, mask, "High Memorability Names", physical_features, narrative_features)
        if r: results.append(r)
        
        # Cultural patterns
        mask = (X_df['red_slavic'] == 1) | (X_df['blue_slavic'] == 1)
        r = self.test_context(X_df, y, mask, "Slavic Fighter", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = (X_df['red_brazilian'] == 1) | (X_df['blue_brazilian'] == 1)
        r = self.test_context(X_df, y, mask, "Brazilian Fighter", physical_features, narrative_features)
        if r: results.append(r)
        
        # Style types
        mask = X_df['style_clash'] == 1
        r = self.test_context(X_df, y, mask, "Style Clash (Striker vs Grappler)", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['striker_vs_striker'] == 1
        r = self.test_context(X_df, y, mask, "Striker vs Striker", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['grappler_vs_grappler'] == 1
        r = self.test_context(X_df, y, mask, "Grappler vs Grappler", physical_features, narrative_features)
        if r: results.append(r)
        
        print(f"  ✓ Tested {len(results) - start_count} fighter-level contexts")
        
        # ===== 3. TEMPORAL CONTEXTS (10) =====
        print("\n[3/5] Testing temporal contexts...")
        
        start_count = len(results)
        
        # By year groups
        for year_start in [2010, 2015, 2020]:
            year_end = year_start + 4
            mask = (X_df['year'] >= year_start) & (X_df['year'] <= year_end)
            r = self.test_context(X_df, y, mask, f"Years {year_start}-{year_end}", physical_features, narrative_features)
            if r: results.append(r)
        
        # Era
        mask = X_df['era_early'] == 1
        r = self.test_context(X_df, y, mask, "Early Era (<2015)", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['era_middle'] == 1
        r = self.test_context(X_df, y, mask, "Middle Era (2015-2019)", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['era_recent'] == 1
        r = self.test_context(X_df, y, mask, "Recent Era (2020+)", physical_features, narrative_features)
        if r: results.append(r)
        
        # Weekend fights
        mask = X_df['is_weekend'] == 1
        r = self.test_context(X_df, y, mask, "Weekend Fights", physical_features, narrative_features)
        if r: results.append(r)
        
        # Performance bonus era (more emphasis)
        mask = (X_df['year'] >= 2020) & (X_df['has_bonus'] == 1)
        r = self.test_context(X_df, y, mask, "Recent Era with Bonuses", physical_features, narrative_features)
        if r: results.append(r)
        
        print(f"  ✓ Tested {len(results) - start_count} temporal contexts")
        
        # ===== 4. WEIGHT CLASS CONTEXTS (11) =====
        print("\n[4/5] Testing weight class contexts...")
        
        start_count = len(results)
        
        # Each weight class
        weight_class_map = {
            1: "Flyweight", 2: "Bantamweight", 3: "Featherweight",
            4: "Lightweight", 5: "Welterweight", 6: "Middleweight",
            7: "Light Heavyweight", 8: "Heavyweight"
        }
        
        for wc_num, wc_name in weight_class_map.items():
            mask = X_df['weight_class_num'] == wc_num
            r = self.test_context(X_df, y, mask, wc_name, physical_features, narrative_features)
            if r: results.append(r)
        
        # Heavy vs Light
        mask = X_df['is_heavyweight_division'] == 1
        r = self.test_context(X_df, y, mask, "Heavyweight Divisions (LHW+HW)", physical_features, narrative_features)
        if r: results.append(r)
        
        mask = X_df['is_lightweight_division'] == 1
        r = self.test_context(X_df, y, mask, "Lightweight Divisions (FLW-LW)", physical_features, narrative_features)
        if r: results.append(r)
        
        # Women's fights
        mask = X_df['is_womens_fight'] == 1
        r = self.test_context(X_df, y, mask, "Women's Fights", physical_features, narrative_features)
        if r: results.append(r)
        
        print(f"  ✓ Tested {len(results) - start_count} weight class contexts")
        
        # ===== 5. COMBINED CONTEXTS (10) =====
        print("\n[5/5] Testing combined contexts...")
        
        start_count = len(results)
        
        # Title fight × Even matchup
        mask = (X_df['is_title_fight'] == 1) & (X_df['sig_str_diff'] < 0.1)
        r = self.test_context(X_df, y, mask, "Title Fight × Even Matchup", physical_features, narrative_features)
        if r: results.append(r)
        
        # High stakes × Physical parity
        mask = (X_df['is_title_fight'] == 1) & (X_df['kd_diff'] == 0) & (X_df['sig_str_diff'] < 0.15)
        r = self.test_context(X_df, y, mask, "High Stakes × Physical Parity", physical_features, narrative_features)
        if r: results.append(r)
        
        # Finish fights × Title bouts
        mask = (X_df['is_finish'] == 1) & (X_df['is_title_fight'] == 1)
        r = self.test_context(X_df, y, mask, "Title Fight Finishes", physical_features, narrative_features)
        if r: results.append(r)
        
        # Both nicknames × High profile
        mask = (X_df['both_have_nicknames'] == 1) & (X_df['is_title_fight'] == 1)
        r = self.test_context(X_df, y, mask, "Both Nicknames × Title Fight", physical_features, narrative_features)
        if r: results.append(r)
        
        # Style clash × Even matchup
        mask = (X_df['style_clash'] == 1) & (X_df['sig_str_diff'] < 0.1)
        r = self.test_context(X_df, y, mask, "Style Clash × Even Matchup", physical_features, narrative_features)
        if r: results.append(r)
        
        # Recent era × Title fights
        mask = (X_df['year'] >= 2020) & (X_df['is_title_fight'] == 1)
        r = self.test_context(X_df, y, mask, "Recent Era Title Fights", physical_features, narrative_features)
        if r: results.append(r)
        
        # Heavyweight × Nicknames
        mask = (X_df['is_heavyweight_division'] == 1) & (X_df['both_have_nicknames'] == 1)
        r = self.test_context(X_df, y, mask, "Heavyweight × Both Nicknames", physical_features, narrative_features)
        if r: results.append(r)
        
        # Early finish × Title fight
        mask = (X_df['early_finish'] == 1) & (X_df['is_title_fight'] == 1)
        r = self.test_context(X_df, y, mask, "Early Finish × Title Fight", physical_features, narrative_features)
        if r: results.append(r)
        
        # 5 round × Even
        mask = (X_df['is_5_round_fight'] == 1) & (X_df['sig_str_diff'] < 0.1)
        r = self.test_context(X_df, y, mask, "5 Round × Even Matchup", physical_features, narrative_features)
        if r: results.append(r)
        
        # Bonus fights × Nicknames
        mask = (X_df['has_bonus'] == 1) & (X_df['both_have_nicknames'] == 1)
        r = self.test_context(X_df, y, mask, "Bonus Fight × Both Nicknames", physical_features, narrative_features)
        if r: results.append(r)
        
        print(f"  ✓ Tested {len(results) - start_count} combined contexts")
        
        return results
    
    def rank_contexts(self, results):
        """Rank contexts by different metrics"""
        
        print("\n" + "="*80)
        print("RANKING CONTEXTS")
        print("="*80)
        
        # Sort by efficiency (primary metric)
        by_efficiency = sorted(results, key=lambda x: x['efficiency'], reverse=True)
        
        print(f"\n✓ TOP 10 CONTEXTS BY EFFICIENCY:")
        for i, ctx in enumerate(by_efficiency[:10], 1):
            status = "✓ PASSES" if ctx['passes'] else "✗"
            print(f"  {i:2d}. {ctx['context']:40s} | eff={ctx['efficiency']:.4f} {status} | n={ctx['n_samples']:5d}")
        
        # Sort by narrative delta
        by_delta = sorted(results, key=lambda x: x['delta'], reverse=True)
        
        print(f"\n✓ TOP 10 CONTEXTS BY NARRATIVE Δ:")
        for i, ctx in enumerate(by_delta[:10], 1):
            print(f"  {i:2d}. {ctx['context']:40s} | Δ={ctx['delta']:+.4f} | n={ctx['n_samples']:5d}")
        
        # Sort by |r|
        by_r = sorted(results, key=lambda x: x['r_abs'], reverse=True)
        
        print(f"\n✓ TOP 10 CONTEXTS BY CORRELATION |r|:")
        for i, ctx in enumerate(by_r[:10], 1):
            print(f"  {i:2d}. {ctx['context']:40s} | |r|={ctx['r_abs']:.4f} | n={ctx['n_samples']:5d}")
        
        # Check for passing contexts
        passing = [ctx for ctx in results if ctx['passes']]
        
        if passing:
            print(f"\n✓ {len(passing)} CONTEXTS PASS THRESHOLD (efficiency > 0.5)!")
            for ctx in passing:
                print(f"  - {ctx['context']:40s} | eff={ctx['efficiency']:.4f}")
        else:
            print(f"\n✗ No contexts pass efficiency > 0.5 threshold")
            print(f"  Highest efficiency: {by_efficiency[0]['efficiency']:.4f} ({by_efficiency[0]['context']})")
        
        return {
            'by_efficiency': by_efficiency,
            'by_delta': by_delta,
            'by_r': by_r,
            'passing': passing
        }


def main():
    """Run exhaustive context discovery"""
    
    discovery = UFCContextDiscovery()
    
    # Load features
    X_df, y, feature_names, physical_features, narrative_features = discovery.load_features()
    
    # Discover all contexts
    results = discovery.discover_all_contexts(X_df, y, physical_features, narrative_features)
    
    print(f"\n✓ Tested {len(results)} total contexts")
    
    # Rank contexts
    rankings = discovery.rank_contexts(results)
    
    # Save results
    output_path = Path('narrative_optimization/domains/ufc/ufc_context_discovery.json')
    with open(output_path, 'w') as f:
        json.dump({
            'all_contexts': results,
            'top_by_efficiency': rankings['by_efficiency'][:20],
            'top_by_delta': rankings['by_delta'][:20],
            'top_by_r': rankings['by_r'][:20],
            'passing_contexts': rankings['passing'],
            'total_tested': len(results)
        }, f, indent=2)
    
    print(f"\n✓ Saved results: {output_path}")


if __name__ == "__main__":
    main()

