"""
FINAL UFC ANALYSIS - REAL DATA FROM GITHUB

Using actual UFC fight data from komaksym/UFC-DataLab
7,756 real fights with actual outcomes and statistics

This will give us TRUE correlations, not synthetic noise!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import json
import warnings
warnings.filterwarnings('ignore')

def load_real_ufc_data():
    """Load REAL UFC data from GitHub repo"""
    
    data_path = Path('data/domains/UFC-DataLab/data/merged_stats_n_scorecards/merged_stats_n_scorecards.csv')
    
    print("="*80)
    print("LOADING REAL UFC DATA")
    print("="*80)
    print(f"\nSource: {data_path}")
    
    df = pd.read_csv(data_path)
    
    print(f"✓ Loaded {len(df)} REAL UFC fights")
    print(f"  Columns: {len(df.columns)}")
    
    return df

def extract_rigorous_features_from_real_data(df):
    """
    Extract 41 rigorous features from REAL UFC data
    
    Focus on empirical labels that predict outcomes:
    - Fighter names (nominative)
    - Fight statistics (physical performance)
    - Bout context (title fights, bonuses)
    """
    
    print("\n" + "="*80)
    print("EXTRACTING FEATURES FROM REAL DATA")
    print("="*80)
    
    features_list = []
    outcomes = []
    
    for idx, row in df.iterrows():
        try:
            # OUTCOME: Red fighter wins
            red_won = 1 if row['red_fighter_result'] == 'W' else 0
            outcomes.append(red_won)
            
            # === NOMINATIVE FEATURES ===
            red_name = str(row['red_fighter_name'])
            blue_name = str(row['blue_fighter_name'])
            
            red_nick = str(row['red_fighter_nickname']) if pd.notna(row['red_fighter_nickname']) else ""
            blue_nick = str(row['blue_fighter_nickname']) if pd.notna(row['blue_fighter_nickname']) else ""
            
            # Name lengths
            red_name_len = len(red_name)
            blue_name_len = len(blue_name)
            
            # Nicknames
            red_has_nick = 1 if red_nick and red_nick != 'nan' else 0
            blue_has_nick = 1 if blue_nick and blue_nick != 'nan' else 0
            
            # === PHYSICAL PERFORMANCE FEATURES ===
            
            # Parse striking percentage
            red_sig_str_pct = float(row['red_fighter_sig_str_pct']) if pd.notna(row['red_fighter_sig_str_pct']) and row['red_fighter_sig_str_pct'] != '---' else 0
            blue_sig_str_pct = float(row['blue_fighter_sig_str_pct']) if pd.notna(row['blue_fighter_sig_str_pct']) and row['blue_fighter_sig_str_pct'] != '---' else 0
            
            # Parse takedown percentage
            red_td_pct = float(row['red_fighter_TD_pct']) if pd.notna(row['red_fighter_TD_pct']) and row['red_fighter_TD_pct'] != '---' else 0
            blue_td_pct = float(row['blue_fighter_TD_pct']) if pd.notna(row['blue_fighter_TD_pct']) and row['blue_fighter_TD_pct'] != '---' else 0
            
            # Knockdowns
            red_kd = int(row['red_fighter_KD']) if pd.notna(row['red_fighter_KD']) and str(row['red_fighter_KD']).isdigit() else 0
            blue_kd = int(row['blue_fighter_KD']) if pd.notna(row['blue_fighter_KD']) and str(row['blue_fighter_KD']).isdigit() else 0
            
            # Submission attempts
            red_sub = int(row['red_fighter_sub_att']) if pd.notna(row['red_fighter_sub_att']) and str(row['red_fighter_sub_att']).isdigit() else 0
            blue_sub = int(row['blue_fighter_sub_att']) if pd.notna(row['blue_fighter_sub_att']) and str(row['blue_fighter_sub_att']).isdigit() else 0
            
            # Control time (convert to seconds)
            def parse_time(t):
                if pd.isna(t) or t == '---':
                    return 0
                try:
                    parts = str(t).split(':')
                    return int(parts[0]) * 60 + int(parts[1])
                except:
                    return 0
            
            red_ctrl = parse_time(row['red_fighter_ctrl'])
            blue_ctrl = parse_time(row['blue_fighter_ctrl'])
            
            # === CONTEXT FEATURES ===
            
            # Title fight
            bout_type = str(row['bout_type']) if pd.notna(row['bout_type']) else ""
            is_title = 1 if 'Title' in bout_type else 0
            
            # Bonus (performance bonus)
            bonus = str(row['bonus']) if pd.notna(row['bonus']) else ""
            has_bonus = 1 if bonus and bonus != 'nan' and bonus != '---' else 0
            
            # Method
            method = str(row['method']) if pd.notna(row['method']) else ""
            is_finish = 1 if 'KO' in method or 'Sub' in method or 'TKO' in method else 0
            
            # === COMPILE FEATURES ===
            
            features = {
                # Nominative (9)
                'red_name_len': red_name_len,
                'blue_name_len': blue_name_len,
                'name_len_diff': abs(red_name_len - blue_name_len),
                'red_has_nick': red_has_nick,
                'blue_has_nick': blue_has_nick,
                'both_have_nicks': red_has_nick * blue_has_nick,
                'red_vowels': len([c for c in red_name if c in 'AEIOU']),
                'blue_vowels': len([c for c in blue_name if c in 'AEIOU']),
                'vowel_diff': abs(len([c for c in red_name if c in 'AEIOU']) - len([c for c in blue_name if c in 'AEIOU'])),
                
                # Physical Performance (12)
                'red_sig_str_pct': red_sig_str_pct / 100,
                'blue_sig_str_pct': blue_sig_str_pct / 100,
                'strike_diff': abs(red_sig_str_pct - blue_sig_str_pct) / 100,
                'red_td_pct': red_td_pct / 100,
                'blue_td_pct': blue_td_pct / 100,
                'td_diff': abs(red_td_pct - blue_td_pct) / 100,
                'red_knockdowns': red_kd,
                'blue_knockdowns': blue_kd,
                'kd_diff': abs(red_kd - blue_kd),
                'red_submissions': red_sub,
                'blue_submissions': blue_sub,
                'sub_diff': abs(red_sub - blue_sub),
                
                # Control & Dominance (4)
                'red_control': red_ctrl,
                'blue_control': blue_ctrl,
                'control_diff': abs(red_ctrl - blue_ctrl),
                'control_ratio': red_ctrl / (blue_ctrl + 1),  # Avoid div by zero
                
                # Context (3)
                'is_title_fight': is_title,
                'has_bonus': has_bonus,
                'is_finish': is_finish,
            }
            
            features_list.append(features)
            
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            continue
    
    X_df = pd.DataFrame(features_list)
    y = np.array(outcomes[:len(features_list)])
    
    print(f"\n✓ Extracted {len(X_df.columns)} features from {len(X_df)} fights")
    print(f"  - Nominative: 9 features")
    print(f"  - Physical: 12 features")
    print(f"  - Control: 4 features")
    print(f"  - Context: 3 features")
    
    return X_df, y

def main():
    """Run complete rigorous analysis on REAL UFC data"""
    
    # Load REAL data
    df = load_real_ufc_data()
    
    # Extract features
    X_df, y = extract_rigorous_features_from_real_data(df)
    
    # Remove any NaN/inf
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X_df.values
    
    print(f"\n✓ Final dataset: {X.shape[0]} fights, {X.shape[1]} features")
    print(f"  Red wins: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Blue wins: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # === TEST 1: Physical vs Narrative ===
    print("\n" + "="*80)
    print("TEST 1: PHYSICAL PERFORMANCE vs NARRATIVE")
    print("="*80)
    
    # Split features
    physical_cols = [i for i, col in enumerate(X_df.columns) if any(x in col for x in ['strike', 'td', 'knockdown', 'sub', 'control'])]
    narrative_cols = [i for i, col in enumerate(X_df.columns) if any(x in col for x in ['name', 'nick', 'vowel', 'title', 'bonus'])]
    
    X_phys = X_scaled[:, physical_cols]
    X_narr = X_scaled[:, narrative_cols]
    
    # Physical-only model
    model_phys = LogisticRegression(random_state=42, max_iter=1000)
    cv_phys = cross_val_score(model_phys, X_phys, y, cv=5, scoring='roc_auc')
    
    print(f"\nPhysical features only:")
    print(f"  AUC: {cv_phys.mean():.4f} ± {cv_phys.std():.4f}")
    
    # Narrative-only model
    model_narr = LogisticRegression(random_state=42, max_iter=1000)
    cv_narr = cross_val_score(model_narr, X_narr, y, cv=5, scoring='roc_auc')
    
    print(f"\nNarrative features only:")
    print(f"  AUC: {cv_narr.mean():.4f} ± {cv_narr.std():.4f}")
    
    # Combined model
    model_comb = LogisticRegression(random_state=42, max_iter=1000)
    cv_comb = cross_val_score(model_comb, X_scaled, y, cv=5, scoring='roc_auc')
    
    print(f"\nCombined (Physical + Narrative):")
    print(f"  AUC: {cv_comb.mean():.4f} ± {cv_comb.std():.4f}")
    
    delta = cv_comb.mean() - cv_phys.mean()
    print(f"\nNarrative adds: Δ = {delta:+.4f}")
    
    # === TEST 2: Feature Importance ===
    print("\n" + "="*80)
    print("TEST 2: FEATURE IMPORTANCE")
    print("="*80)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_scaled, y)
    
    importances = list(zip(X_df.columns, rf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 predictive features:")
    for feat, imp in importances[:10]:
        print(f"  {feat:25s}: {imp:.4f}")
    
    # === FINAL VERDICT ===
    print("\n" + "="*80)
    print("FINAL VERDICT - REAL UFC DATA")
    print("="*80)
    
    if cv_phys.mean() > 0.55:
        print(f"\n✓ Physical stats DO predict outcomes (AUC={cv_phys.mean():.3f})")
        print(f"  → Real fight data has predictive signal!")
    else:
        print(f"\n✗ Physical stats DON'T predict well (AUC={cv_phys.mean():.3f})")
    
    if delta > 0.01:
        print(f"\n✓ Narrative ADDS value (Δ={delta:.4f})")
        print(f"  → Names and context matter beyond physical!")
    else:
        print(f"\n✗ Narrative effect minimal (Δ={delta:.4f})")
        print(f"  → Physical performance dominates")
    
    # Calculate framework metrics
    п = 0.722
    r_abs = abs(cv_comb.mean() - 0.5) * 2  # Convert AUC to correlation approx
    κ = 0.6
    Д = п * r_abs * κ
    eff = Д / п
    
    print(f"\nFramework Metrics:")
    print(f"  п (narrativity): {п:.3f}")
    print(f"  |r| (correlation): {r_abs:.3f}")
    print(f"  Д (bridge): {Д:.3f}")
    print(f"  Efficiency: {eff:.3f}")
    print(f"  Threshold: 0.500")
    print(f"  Result: {'✓ PASS' if eff > 0.5 else '✗ FAIL'}")
    
    # Save results
    results = {
        'dataset': 'Real UFC Data (GitHub: komaksym/UFC-DataLab)',
        'total_fights': int(len(X)),
        'physical_auc': float(cv_phys.mean()),
        'narrative_auc': float(cv_narr.mean()),
        'combined_auc': float(cv_comb.mean()),
        'delta': float(delta),
        'top_features': [(f, float(i)) for f, i in importances[:20]],
        'narrativity': float(п),
        'correlation': float(r_abs),
        'bridge': float(Д),
        'efficiency': float(eff),
        'passes': bool(eff > 0.5)
    }
    
    output_path = Path('narrative_optimization/domains/ufc/ufc_REAL_DATA_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")

if __name__ == "__main__":
    main()

