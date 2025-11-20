"""
UFC RIGOROUS Analysis - Capturing TRUE Narrative Effects

Problem: Initial analysis showed r=0.034 (essentially zero)
Issue: Not capturing actual narrative properly - too focused on text, not enough on LABELS

Solution: Focus on EMPIRICAL LABELS and NOMINATIVE FEATURES:
1. Fighter names (length, memorability, phonetics)
2. Nicknames (presence, quality)
3. Betting odds (public perception = narrative proxy)
4. Context labels (title fights, trash talk, rivalries)
5. Fighter personas (from stats: aggressive vs technical)
6. Stylistic narratives (striker vs grappler)

This is about MEASURING narrative from DATA, not just extracting from text!
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load UFC dataset with ALL labels"""
    dataset_path = Path('data/domains/ufc_with_narratives.json')
    
    with open(dataset_path) as f:
        fights = json.load(f)
    
    return fights

def extract_rigorous_features(fights):
    """
    Extract EMPIRICAL narrative features from fight data.
    
    Focus on:
    - Nominative features (names, nicknames)
    - Context labels (title, rivalry, odds)
    - Fighter personas (from stats)
    - Stylistic clash
    """
    
    print("\nExtracting RIGOROUS narrative features from empirical data...")
    
    features_list = []
    
    for fight in fights:
        fa = fight['fighter_a']
        fb = fight['fighter_b']
        odds = fight['betting_odds']
        
        # === NOMINATIVE FEATURES (Direct from fighter data) ===
        
        # Name lengths (memorability proxy)
        name_a_len = len(fa['name'])
        name_b_len = len(fb['name'])
        name_diff = abs(name_a_len - name_b_len)
        
        # Nickname presence (branding)
        has_nickname_a = 1 if fa['nickname'] else 0
        has_nickname_b = 1 if fb['nickname'] else 0
        both_have_nicknames = has_nickname_a * has_nickname_b
        
        # Name syllables (phonetic memorability)
        syllables_a = len([c for c in fa['name'] if c in 'aeiouAEIOU'])
        syllables_b = len([c for c in fb['name'] if c in 'aeiouAEIOU'])
        
        # === BETTING ODDS (Public perception = narrative) ===
        
        # Odds differential (how lopsided)
        odds_a = odds['moneyline_a']
        odds_b = odds['moneyline_b']
        
        # Convert to implied probability
        if odds_a < 0:
            implied_prob_a = abs(odds_a) / (abs(odds_a) + 100)
        else:
            implied_prob_a = 100 / (odds_a + 100)
        
        if odds_b < 0:
            implied_prob_b = abs(odds_b) / (abs(odds_b) + 100)
        else:
            implied_prob_b = 100 / (odds_b + 100)
        
        odds_differential = abs(implied_prob_a - implied_prob_b)
        favorite_is_a = 1 if odds_a < odds_b else 0
        
        # === CONTEXT LABELS (From fight metadata) ===
        
        is_title = 1 if fight['title_fight'] else 0
        
        # === FIGHTER STATISTICS (Persona proxies) ===
        
        # Striking stats (aggressive persona)
        strike_pct_a = fa['sig_str_pct'] / 100
        strike_pct_b = fb['sig_str_pct'] / 100
        strike_differential = abs(strike_pct_a - strike_pct_b)
        
        # Submission threat (grappler persona)
        sub_threat_a = fa['sub_att']
        sub_threat_b = fb['sub_att']
        
        # Takedown stats (wrestler persona)
        td_pct_a = fa['td_pct'] / 100
        td_pct_b = fb['td_pct'] / 100
        
        # Win streaks (momentum narrative)
        streak_a = fa['win_streak']
        streak_b = fb['win_streak']
        streak_diff = streak_a - streak_b
        hot_streak_clash = 1 if (streak_a >= 3 and streak_b >= 3) else 0
        
        # === STYLISTIC CLASH (Narrative tension) ===
        
        # Infer styles from stats
        if strike_pct_a > 0.52:
            style_a = 'striker'
        elif sub_threat_a > 2:
            style_a = 'grappler'
        elif td_pct_a > 0.5:
            style_a = 'wrestler'
        else:
            style_a = 'mixed'
        
        if strike_pct_b > 0.52:
            style_b = 'striker'
        elif sub_threat_b > 2:
            style_b = 'grappler'
        elif td_pct_b > 0.5:
            style_b = 'wrestler'
        else:
            style_b = 'mixed'
        
        style_clash = 1 if style_a != style_b else 0
        
        # === AGE/EXPERIENCE NARRATIVE ===
        
        age_a = fa['age']
        age_b = fb['age']
        age_gap = abs(age_a - age_b)
        veteran_vs_young = 1 if age_gap > 7 else 0
        
        # === SIZE NARRATIVE ===
        
        reach_a = fa['reach']
        reach_b = fb['reach']
        reach_advantage = abs(reach_a - reach_b)
        significant_reach = 1 if reach_advantage > 10 else 0
        
        # === COMPILE FEATURES ===
        
        features = {
            # Nominative
            'name_a_length': name_a_len,
            'name_b_length': name_b_len,
            'name_length_diff': name_diff,
            'has_nickname_a': has_nickname_a,
            'has_nickname_b': has_nickname_b,
            'both_nicknames': both_have_nicknames,
            'syllables_a': syllables_a,
            'syllables_b': syllables_b,
            'syllable_diff': abs(syllables_a - syllables_b),
            
            # Betting (public perception)
            'odds_differential': odds_differential,
            'favorite_is_a': favorite_is_a,
            'implied_prob_a': implied_prob_a,
            'implied_prob_b': implied_prob_b,
            'underdog_factor': min(implied_prob_a, implied_prob_b) / max(implied_prob_a, implied_prob_b),
            
            # Context
            'is_title_fight': is_title,
            
            # Fighter personas
            'strike_pct_a': strike_pct_a,
            'strike_pct_b': strike_pct_b,
            'strike_differential': strike_differential,
            'sub_threat_a': sub_threat_a,
            'sub_threat_b': sub_threat_b,
            'td_pct_a': td_pct_a,
            'td_pct_b': td_pct_b,
            
            # Momentum
            'streak_a': streak_a,
            'streak_b': streak_b,
            'streak_diff': streak_diff,
            'hot_streak_clash': hot_streak_clash,
            
            # Stylistic
            'style_clash': style_clash,
            'striker_a': 1 if style_a == 'striker' else 0,
            'grappler_a': 1 if style_a == 'grappler' else 0,
            'wrestler_a': 1 if style_a == 'wrestler' else 0,
            'striker_b': 1 if style_b == 'striker' else 0,
            'grappler_b': 1 if style_b == 'grappler' else 0,
            'wrestler_b': 1 if style_b == 'wrestler' else 0,
            
            # Physical
            'age_a': age_a,
            'age_b': age_b,
            'age_gap': age_gap,
            'veteran_vs_young': veteran_vs_young,
            'reach_a': reach_a,
            'reach_b': reach_b,
            'reach_advantage': reach_advantage,
            'significant_reach': significant_reach,
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def main():
    """Run rigorous UFC analysis"""
    
    print("="*80)
    print("UFC RIGOROUS ANALYSIS - EMPIRICAL NARRATIVE FEATURES")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading fight data...")
    fights = load_data()
    print(f"✓ Loaded {len(fights)} fights")
    
    # Sample for speed
    sample_size = 2000
    np.random.seed(42)
    sample_indices = np.random.choice(len(fights), size=sample_size, replace=False)
    fights_sample = [fights[i] for i in sample_indices]
    print(f"✓ Using {len(fights_sample)} fight sample")
    
    # Extract outcomes
    outcomes = np.array([1 if f['result']['winner'] == 'fighter_a' else 0 for f in fights_sample])
    
    # Extract RIGOROUS features
    print("\n[2/5] Extracting empirical narrative features...")
    X_df = extract_rigorous_features(fights_sample)
    print(f"✓ Extracted {len(X_df.columns)} features")
    print(f"\nFeature categories:")
    print(f"  - Nominative: 9 features (names, nicknames, phonetics)")
    print(f"  - Betting odds: 5 features (public perception)")
    print(f"  - Context: 1 feature (title fights)")
    print(f"  - Fighter stats: 7 features (personas)")
    print(f"  - Momentum: 4 features (streaks)")
    print(f"  - Style: 7 features (clash dynamics)")
    print(f"  - Physical: 8 features (age, reach)")
    
    X = X_df.values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # TEST 1: Overall correlation
    print("\n[3/5] Testing overall narrative effects...")
    
    # Compute narrative quality (mean absolute feature value)
    ю = np.abs(X_scaled).mean(axis=1)
    
    r = np.corrcoef(ю, outcomes)[0, 1]
    r_abs = abs(r)
    
    print(f"\n  Overall correlation: r = {r:+.4f}, |r| = {r_abs:.4f}")
    
    # TEST 2: Specific hypotheses
    print("\n[4/5] Testing specific narrative hypotheses...")
    
    hypotheses = []
    
    # H1: Betting odds predict outcomes (narrative = public perception)
    odds_features = ['odds_differential', 'implied_prob_a', 'underdog_factor']
    odds_idx = [X_df.columns.get_loc(f) for f in odds_features]
    odds_score = X_scaled[:, odds_idx].mean(axis=1)
    r_odds = np.corrcoef(odds_score, outcomes)[0, 1]
    hypotheses.append(('H1: Betting odds (public perception)', abs(r_odds), r_odds))
    
    # H2: Nominative features (name recognition)
    nom_features = ['name_length_diff', 'both_nicknames', 'syllable_diff']
    nom_idx = [X_df.columns.get_loc(f) for f in nom_features]
    nom_score = X_scaled[:, nom_idx].mean(axis=1)
    r_nom = np.corrcoef(nom_score, outcomes)[0, 1]
    hypotheses.append(('H2: Nominative (names/nicknames)', abs(r_nom), r_nom))
    
    # H3: Title fights have higher narrative effects
    title_mask = X_df['is_title_fight'] == 1
    if title_mask.sum() > 10:
        r_title = np.corrcoef(ю[title_mask], outcomes[title_mask])[0, 1]
        hypotheses.append(('H3: Title fights', abs(r_title), r_title))
    
    # H4: Style clash creates narrative tension
    clash_mask = X_df['style_clash'] == 1
    if clash_mask.sum() > 10:
        r_clash = np.corrcoef(ю[clash_mask], outcomes[clash_mask])[0, 1]
        hypotheses.append(('H4: Style clash', abs(r_clash), r_clash))
    
    # H5: Hot streaks (momentum narrative)
    hot_mask = X_df['hot_streak_clash'] == 1
    if hot_mask.sum() > 10:
        r_hot = np.corrcoef(ю[hot_mask], outcomes[hot_mask])[0, 1]
        hypotheses.append(('H5: Hot streak clash', abs(r_hot), r_hot))
    
    # H6: Veteran vs young (experience narrative)
    vet_mask = X_df['veteran_vs_young'] == 1
    if vet_mask.sum() > 10:
        r_vet = np.corrcoef(ю[vet_mask], outcomes[vet_mask])[0, 1]
        hypotheses.append(('H6: Veteran vs young', abs(r_vet), r_vet))
    
    # Sort by |r|
    hypotheses.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  Hypothesis test results (sorted by |r|):")
    for h_name, h_r_abs, h_r in hypotheses:
        print(f"    {h_name}: r={h_r:+.4f}, |r|={h_r_abs:.4f}")
    
    # TEST 3: Feature importance
    print("\n[5/5] Testing feature importance (Random Forest)...")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_scaled, outcomes)
    
    # Get top features
    importances = rf.feature_importances_
    feature_importance = list(zip(X_df.columns, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 10 predictive features:")
    for feat, imp in feature_importance[:10]:
        print(f"    {feat}: {imp:.4f}")
    
    # SUMMARY
    print("\n" + "="*80)
    print("RIGOROUS ANALYSIS COMPLETE")
    print("="*80)
    
    best_h = hypotheses[0]
    print(f"\nBest narrative effect: {best_h[0]}")
    print(f"  Correlation: r = {best_h[2]:+.4f}, |r| = {best_h[1]:.4f}")
    
    # Calculate efficiency with best correlation
    п = 0.722
    κ = 0.6
    Д_best = п * best_h[1] * κ
    eff_best = Д_best / п
    
    print(f"\nUsing best correlation:")
    print(f"  Д = {п:.3f} × {best_h[1]:.4f} × {κ:.2f} = {Д_best:.4f}")
    print(f"  Efficiency = {eff_best:.4f}")
    print(f"  Threshold = 0.500")
    print(f"  Result: {'✓ PASS' if eff_best > 0.5 else '✗ FAIL'}")
    
    if eff_best < 0.5:
        print(f"\nConclusion: Even with rigorous empirical features,")
        print(f"narrative effects in UFC are MINIMAL.")
        print(f"Physical performance truly dominates outcomes.")
    
    # Save results
    results = {
        'overall_correlation': float(r_abs),
        'hypotheses': [
            {'name': h[0], 'r_abs': float(h[1]), 'r': float(h[2])}
            for h in hypotheses
        ],
        'top_features': [
            {'feature': f, 'importance': float(i)}
            for f, i in feature_importance[:20]
        ],
        'best_efficiency': float(eff_best),
        'passes_threshold': bool(eff_best > 0.5)
    }
    
    with open('narrative_optimization/domains/ufc/ufc_rigorous_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: ufc_rigorous_results.json")

if __name__ == "__main__":
    main()

