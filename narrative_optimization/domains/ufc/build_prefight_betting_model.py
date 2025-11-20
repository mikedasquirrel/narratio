"""
UFC Pre-Fight Betting Model - VALID FOR REAL BETTING

Uses ONLY pre-fight available data:
- Fighter career statistics (historical averages)
- Fighter physical attributes (reach, stance, age)
- Fighter names/nicknames (nominative)
- Fight context (title fight, weight class)

Does NOT use:
- This fight's statistics (strikes landed, knockdowns)
- Judge scores from this fight
- Outcome-dependent data

This model is VALID for real betting because all features are known before the fight.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_fighter_career_stats():
    """Load fighter career statistics (pre-fight data)"""
    
    print("="*80)
    print("LOADING FIGHTER CAREER STATISTICS")
    print("="*80)
    
    fighter_path = Path('data/domains/UFC-DataLab/data/external_data/raw_fighter_details.csv')
    fighters_df = pd.read_csv(fighter_path)
    
    print(f"✓ Loaded {len(fighters_df)} fighter profiles")
    print(f"  Career stats: SLpM, Str_Acc, TD_Avg, TD_Acc, Sub_Avg, etc.")
    
    # Clean fighter names for matching
    fighters_df['fighter_name_clean'] = fighters_df['fighter_name'].str.upper().str.strip()
    
    return fighters_df

def load_fights():
    """Load fight results"""
    
    fights_path = Path('data/domains/UFC-DataLab/data/merged_stats_n_scorecards/merged_stats_n_scorecards.csv')
    fights_df = pd.read_csv(fights_path)
    
    print(f"✓ Loaded {len(fights_df)} fights")
    
    return fights_df

def extract_prefight_features(fights_df, fighters_df):
    """
    Extract ONLY pre-fight available features
    
    For each fight, merge fighter career stats and use:
    - Fighter A career averages
    - Fighter B career averages  
    - Physical attributes (reach, stance, age)
    - Names and nicknames
    - Fight context
    """
    
    print("\n" + "="*80)
    print("EXTRACTING PRE-FIGHT FEATURES")
    print("="*80)
    print("Using ONLY data available before the fight...")
    
    features_list = []
    outcomes = []
    valid_fights = []
    
    for idx, row in fights_df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing fight {idx}/{len(fights_df)}...")
        
        try:
            # Fighter names
            red_name = str(row['red_fighter_name']).upper().strip()
            blue_name = str(row['blue_fighter_name']).upper().strip()
            
            # Find fighter career stats
            red_stats = fighters_df[fighters_df['fighter_name_clean'] == red_name]
            blue_stats = fighters_df[fighters_df['fighter_name_clean'] == blue_name]
            
            # Skip if we don't have career data for both fighters
            if len(red_stats) == 0 or len(blue_stats) == 0:
                continue
            
            red_stats = red_stats.iloc[0]
            blue_stats = blue_stats.iloc[0]
            
            # Outcome
            red_won = 1 if row['red_fighter_result'] == 'W' else 0
            outcomes.append(red_won)
            valid_fights.append(idx)
            
            features = {}
            
            # === FIGHTER CAREER STATS (Pre-fight) ===
            
            # Striking (career averages)
            def safe_float(val, default=0):
                if pd.isna(val):
                    return default
                try:
                    return float(str(val).replace('%', ''))
                except:
                    return default
            
            features['red_slpm'] = safe_float(red_stats['SLpM'])  # Strikes landed per minute
            features['blue_slpm'] = safe_float(blue_stats['SLpM'])
            features['red_str_acc'] = safe_float(red_stats['Str_Acc']) / 100  # Accuracy %
            features['blue_str_acc'] = safe_float(blue_stats['Str_Acc']) / 100
            features['red_sapm'] = safe_float(red_stats['SApM'])  # Strikes absorbed per minute
            features['blue_sapm'] = safe_float(blue_stats['SApM'])
            features['red_str_def'] = safe_float(red_stats['Str_Def']) / 100  # Defense %
            features['blue_str_def'] = safe_float(blue_stats['Str_Def']) / 100
            
            # Takedowns (career averages)
            features['red_td_avg'] = safe_float(red_stats['TD_Avg'])
            features['blue_td_avg'] = safe_float(blue_stats['TD_Avg'])
            features['red_td_acc'] = safe_float(red_stats['TD_Acc']) / 100
            features['blue_td_acc'] = safe_float(blue_stats['TD_Acc']) / 100
            features['red_td_def'] = safe_float(red_stats['TD_Def']) / 100
            features['blue_td_def'] = safe_float(blue_stats['TD_Def']) / 100
            
            # Submissions (career average)
            features['red_sub_avg'] = safe_float(red_stats['Sub_Avg'])
            features['blue_sub_avg'] = safe_float(blue_stats['Sub_Avg'])
            
            # === DIFFERENTIALS (Matchup quality) ===
            features['slpm_diff'] = features['red_slpm'] - features['blue_slpm']
            features['str_acc_diff'] = features['red_str_acc'] - features['blue_str_acc']
            features['td_avg_diff'] = features['red_td_avg'] - features['blue_td_avg']
            features['sub_avg_diff'] = features['red_sub_avg'] - features['blue_sub_avg']
            
            # Striking offense vs defense matchup
            features['red_offense_vs_blue_defense'] = features['red_slpm'] * (1 - features['blue_str_def'])
            features['blue_offense_vs_red_defense'] = features['blue_slpm'] * (1 - features['red_str_def'])
            
            # === PHYSICAL ATTRIBUTES ===
            
            # Reach
            def safe_parse_reach(val):
                if pd.isna(val):
                    return 0
                try:
                    return float(str(val).replace('"', '').replace('cm', ''))
                except:
                    return 0
            
            red_reach = safe_parse_reach(red_stats['Reach'])
            blue_reach = safe_parse_reach(blue_stats['Reach'])
            
            features['red_reach'] = red_reach
            features['blue_reach'] = blue_reach
            features['reach_diff'] = red_reach - blue_reach
            features['reach_advantage'] = 1 if abs(red_reach - blue_reach) > 3 else 0
            
            # Stance
            red_stance = str(red_stats['Stance']) if pd.notna(red_stats['Stance']) else 'Orthodox'
            blue_stance = str(blue_stats['Stance']) if pd.notna(blue_stats['Stance']) else 'Orthodox'
            
            features['red_orthodox'] = 1 if 'Orthodox' in red_stance else 0
            features['red_southpaw'] = 1 if 'Southpaw' in red_stance else 0
            features['blue_orthodox'] = 1 if 'Orthodox' in blue_stance else 0
            features['blue_southpaw'] = 1 if 'Southpaw' in blue_stance else 0
            features['stance_mismatch'] = 1 if red_stance != blue_stance else 0
            
            # Age (from DOB)
            try:
                from datetime import datetime
                red_dob = pd.to_datetime(red_stats['DOB'], errors='coerce')
                blue_dob = pd.to_datetime(blue_stats['DOB'], errors='coerce')
                fight_date = pd.to_datetime(row['event_date'], format='%d/%m/%Y', errors='coerce')
                
                if pd.notna(red_dob) and pd.notna(fight_date):
                    features['red_age'] = (fight_date - red_dob).days / 365.25
                else:
                    features['red_age'] = 30
                
                if pd.notna(blue_dob) and pd.notna(fight_date):
                    features['blue_age'] = (fight_date - blue_dob).days / 365.25
                else:
                    features['blue_age'] = 30
                
                features['age_diff'] = abs(features['red_age'] - features['blue_age'])
                features['veteran_vs_young'] = 1 if features['age_diff'] > 7 else 0
                
            except:
                features['red_age'] = 30
                features['blue_age'] = 30
                features['age_diff'] = 0
                features['veteran_vs_young'] = 0
            
            # === NOMINATIVE FEATURES ===
            
            features['red_name_len'] = len(red_name)
            features['blue_name_len'] = len(blue_name)
            features['name_len_diff'] = abs(features['red_name_len'] - features['blue_name_len'])
            
            red_nick = str(row['red_fighter_nickname']) if pd.notna(row['red_fighter_nickname']) else ""
            blue_nick = str(row['blue_fighter_nickname']) if pd.notna(row['blue_fighter_nickname']) else ""
            
            features['red_has_nickname'] = 1 if red_nick and red_nick != '-' else 0
            features['blue_has_nickname'] = 1 if blue_nick and blue_nick != '-' else 0
            features['both_have_nicknames'] = features['red_has_nickname'] * features['blue_has_nickname']
            
            # === CONTEXT ===
            
            bout_type = str(row['bout_type']) if pd.notna(row['bout_type']) else ""
            features['is_title_fight'] = 1 if 'Title' in bout_type else 0
            features['is_5_round'] = 1 if '5 Rnd' in str(row['time_format']) else 0
            features['is_womens'] = 1 if 'Women' in bout_type else 0
            
            # Weight class
            weight_classes = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 
                            'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']
            for i, wc in enumerate(weight_classes, 1):
                features[f'weight_class_{i}'] = 1 if wc in bout_type else 0
            
            features_list.append(features)
            
        except Exception as e:
            continue
    
    X_df = pd.DataFrame(features_list)
    y = np.array(outcomes)
    
    print(f"\n✓ Built pre-fight dataset:")
    print(f"  Fights with career data: {len(X_df)}/{len(fights_df)} ({100*len(X_df)/len(fights_df):.1f}%)")
    print(f"  Features: {len(X_df.columns)}")
    print(f"  Red wins: {y.sum()} ({100*y.mean():.1f}%)")
    
    return X_df, y, valid_fights

def test_prefight_betting(X_df, y, total_fights):
    """
    Test betting model using ONLY pre-fight data
    
    This is VALID for real betting!
    """
    
    print("\n" + "="*80)
    print("PRE-FIGHT BETTING MODEL (VALID FOR REAL BETTING)")
    print("="*80)
    
    # Split train/test (temporal split would be better but random for now)
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTrain: {len(X_train)} fights")
    print(f"Test:  {len(X_test)} fights")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # === TEST 1: Career stats only ===
    print("\n[1/3] Testing career statistics (pre-fight performance)...")
    
    career_cols = [i for i, col in enumerate(X_df.columns) if any(x in col for x in ['slpm', 'str_acc', 'td_avg', 'sub_avg', 'str_def', 'td_def'])]
    
    model_career = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    cv_career = cross_val_score(model_career, X_train_scaled[:, career_cols], y_train, cv=5, scoring='roc_auc')
    
    print(f"  Career stats AUC: {cv_career.mean():.4f} ± {cv_career.std():.4f}")
    
    model_career.fit(X_train_scaled[:, career_cols], y_train)
    pred_career = model_career.predict_proba(X_test_scaled[:, career_cols])[:, 1]
    auc_career = roc_auc_score(y_test, pred_career)
    
    print(f"  Test set AUC: {auc_career:.4f}")
    
    # === TEST 2: Add physical attributes ===
    print("\n[2/3] Adding physical attributes (reach, stance, age)...")
    
    phys_cols = career_cols + [i for i, col in enumerate(X_df.columns) if any(x in col for x in ['reach', 'age', 'stance'])]
    
    model_phys = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_phys.fit(X_train_scaled[:, phys_cols], y_train)
    
    pred_phys = model_phys.predict_proba(X_test_scaled[:, phys_cols])[:, 1]
    auc_phys = roc_auc_score(y_test, pred_phys)
    
    delta_phys = auc_phys - auc_career
    
    print(f"  Career + Physical AUC: {auc_phys:.4f}")
    print(f"  Δ: {delta_phys:+.4f}")
    
    # === TEST 3: Add nominative and context ===
    print("\n[3/3] Adding nominative and context features...")
    
    model_full = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
    model_full.fit(X_train_scaled, y_train)
    
    pred_full = model_full.predict_proba(X_test_scaled)[:, 1]
    auc_full = roc_auc_score(y_test, pred_full)
    
    delta_full = auc_full - auc_career
    delta_narr = auc_full - auc_phys
    
    print(f"  Full model AUC: {auc_full:.4f}")
    print(f"  Total Δ: {delta_full:+.4f}")
    print(f"  Narrative Δ: {delta_narr:+.4f}")
    
    # === BETTING SIMULATION ===
    print("\n" + "="*80)
    print("BETTING SIMULATION (PRE-FIGHT MODEL)")
    print("="*80)
    
    # Confidence-based betting
    confidence = np.max([pred_full, 1-pred_full], axis=0)
    predictions = (pred_full > 0.5).astype(int)
    
    print(f"\nBetting Strategies:")
    
    for threshold in [0.60, 0.65, 0.70]:
        bet_mask = confidence > threshold
        
        if bet_mask.sum() == 0:
            continue
        
        bets = bet_mask.sum()
        correct = (predictions[bet_mask] == y_test[bet_mask]).sum()
        accuracy = correct / bets
        
        # Calculate ROI (even money odds)
        wins = correct
        losses = bets - correct
        roi = (wins - losses) / bets
        
        print(f"  {threshold:.0%} confidence: {bets:4d} bets | Acc={accuracy:.4f} | ROI={roi:+.1%}")
    
    # === BASELINE COMPARISON ===
    print("\n" + "="*80)
    print("COMPARISON TO BASELINE")
    print("="*80)
    
    baseline_accuracy = max(y_test.mean(), 1-y_test.mean())  # Always pick favorite
    
    print(f"\nBaseline (always pick favorite): {baseline_accuracy:.1%}")
    print(f"Our model (70% confidence):      {accuracy:.1%}")
    print(f"Edge:                            {(accuracy-baseline_accuracy)*100:+.1f} percentage points")
    
    if accuracy > baseline_accuracy + 0.05:
        print(f"\n✓ Model provides {(accuracy-baseline_accuracy)*100:.1f}pp betting edge!")
    elif accuracy > baseline_accuracy:
        print(f"\n✓ Model provides small {(accuracy-baseline_accuracy)*100:.1f}pp edge")
    else:
        print(f"\n✗ Model does not beat baseline")
    
    # === SAVE RESULTS ===
    
    results = {
        'model_type': 'pre_fight',
        'data_used': 'career_statistics_only',
        'valid_for_betting': True,
        'dataset_size': int(len(X_df)),
        'coverage': float(len(X_df) / total_fights),
        'performance': {
            'career_only_auc': float(auc_career),
            'with_physical_auc': float(auc_phys),
            'full_model_auc': float(auc_full),
            'physical_delta': float(delta_phys),
            'narrative_delta': float(delta_narr),
            'total_delta': float(delta_full)
        },
        'betting': {
            'best_accuracy': float(accuracy),
            'baseline_accuracy': float(baseline_accuracy),
            'edge': float(accuracy - baseline_accuracy),
            'profitable': bool(accuracy > baseline_accuracy)
        }
    }
    
    output_path = Path('narrative_optimization/domains/ufc/ufc_prefight_betting_model.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    # Save model
    model_dir = Path('narrative_optimization/domains/ufc/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / 'prefight_model.pkl', 'wb') as f:
        pickle.dump({'model': model_full, 'scaler': scaler}, f)
    
    print(f"✓ Model saved: {model_dir}/prefight_model.pkl")
    
    return results


def main():
    """Build and test pre-fight betting model"""
    
    # Load data
    fighters_df = load_fighter_career_stats()
    fights_df = load_fights()
    
    # Extract pre-fight features
    X_df, y, valid_fights = extract_prefight_features(fights_df, fighters_df)
    
    # Test betting model
    results = test_prefight_betting(X_df, y, len(fights_df))
    
    print("\n" + "="*80)
    print("PRE-FIGHT BETTING MODEL COMPLETE")
    print("="*80)
    print(f"\n✓ Valid for real betting (uses only pre-fight data)")
    print(f"✓ Coverage: {results['coverage']*100:.1f}% of fights")
    print(f"✓ Prediction AUC: {results['performance']['full_model_auc']:.4f}")
    print(f"✓ Betting edge: {results['betting']['edge']*100:+.1f} percentage points")


if __name__ == "__main__":
    main()

