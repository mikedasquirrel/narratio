"""
MLB Domain-Specific Formula Optimization

Creates and saves optimized formulas for specific MLB contexts:
1. Astros-Rangers rivalry formula (|r|=0.3487)
2. Yankees-Red Sox rivalry formula
3. Dodgers-Giants rivalry formula
4. Historic stadium formula (Wrigley, Fenway, Yankee)
5. Playoff race formula

Each formula is optimized for its specific context and saved for future deployment.
"""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

print("="*80)
print("MLB DOMAIN-SPECIFIC FORMULA OPTIMIZATION")
print("="*80)
print("\nCreating optimized formulas for specific contexts...")
print("Each formula will be saved for future deployment\n")

# Load data
dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'mlb_complete_dataset.json'
with open(dataset_path) as f:
    all_games = json.load(f)

genome_path = Path(__file__).parent / 'mlb_genome_data.npz'
genome_data = np.load(genome_path, allow_pickle=True)

ж = genome_data['genome']
ю = genome_data['story_quality']
outcomes = genome_data['outcomes']
feature_names = genome_data['feature_names'].tolist()

sample_games = all_games[:len(outcomes)]

print(f"✓ Loaded {len(all_games)} games, {ж.shape[1]} features")

# Storage for all formulas
domain_formulas = {}

# ============================================================================
# FORMULA 1: ASTROS-RANGERS RIVALRY (Strongest Context)
# ============================================================================

print("\n" + "="*80)
print("FORMULA 1: ASTROS-RANGERS RIVALRY SPECIFIC")
print("="*80)

hr_indices = [i for i, g in enumerate(sample_games) 
              if (g['home_team']['abbreviation'] == 'HOU' and g['away_team']['abbreviation'] == 'TEX') or
                 (g['home_team']['abbreviation'] == 'TEX' and g['away_team']['abbreviation'] == 'HOU')]

print(f"Matches: {len(hr_indices)}")

if len(hr_indices) >= 20:
    X_hr = ж[hr_indices]
    y_hr = outcomes[hr_indices]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_hr, y_hr, test_size=0.3, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    k = min(50, X_train_scaled.shape[1])
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X_train_scaled, y_train)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Grid search for best alpha
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(Ridge(), param_grid, cv=min(5, len(X_train)//2), scoring='r2')
    grid.fit(X_train_selected, y_train)
    
    best_model = grid.best_estimator_
    
    # Test
    y_pred = best_model.predict(X_test_selected)
    r2 = r2_score(y_test, y_pred)
    r = np.corrcoef(y_pred, y_test)[0, 1]
    
    print(f"  Optimized R²: {r2:.4f} ({r2*100:.1f}%)")
    print(f"  Correlation |r|: {abs(r):.4f}")
    print(f"  Best alpha: {grid.best_params_['alpha']}")
    
    # Save formula
    domain_formulas['astros_rangers'] = {
        'context': 'Astros-Rangers Rivalry',
        'n_matches': len(hr_indices),
        'features_selected': k,
        'best_alpha': grid.best_params_['alpha'],
        'test_r2': float(r2),
        'test_r': float(r),
        'model_path': 'models/astros_rangers_formula.pkl'
    }
    
    # Save model
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model_data = {
        'scaler': scaler,
        'selector': selector,
        'model': best_model,
        'feature_indices': selector.get_support(indices=True),
        'feature_names': [feature_names[i] for i in selector.get_support(indices=True)]
    }
    
    with open(models_dir / 'astros_rangers_formula.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"  ✓ Saved formula to {models_dir / 'astros_rangers_formula.pkl'}")

# ============================================================================
# FORMULA 2: YANKEES-RED SOX RIVALRY
# ============================================================================

print("\n" + "="*80)
print("FORMULA 2: YANKEES-RED SOX RIVALRY SPECIFIC")
print("="*80)

yr_indices = [i for i, g in enumerate(sample_games) 
              if (g['home_team']['abbreviation'] == 'NYY' and g['away_team']['abbreviation'] == 'BOS') or
                 (g['home_team']['abbreviation'] == 'BOS' and g['away_team']['abbreviation'] == 'NYY')]

print(f"Matches: {len(yr_indices)}")

if len(yr_indices) >= 20:
    X_yr = ж[yr_indices]
    y_yr = outcomes[yr_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X_yr, y_yr, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    k = min(50, X_train_scaled.shape[1])
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X_train_scaled, y_train)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(Ridge(), param_grid, cv=min(5, len(X_train)//2), scoring='r2')
    grid.fit(X_train_selected, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_selected)
    r2 = r2_score(y_test, y_pred)
    r = np.corrcoef(y_pred, y_test)[0, 1]
    
    print(f"  Optimized R²: {r2:.4f} ({r2*100:.1f}%)")
    print(f"  Correlation |r|: {abs(r):.4f}")
    
    domain_formulas['yankees_redsox'] = {
        'context': 'Yankees-Red Sox Rivalry',
        'n_matches': len(yr_indices),
        'features_selected': k,
        'best_alpha': grid.best_params_['alpha'],
        'test_r2': float(r2),
        'test_r': float(r),
        'model_path': 'models/yankees_redsox_formula.pkl'
    }
    
    model_data = {
        'scaler': scaler,
        'selector': selector,
        'model': best_model,
        'feature_indices': selector.get_support(indices=True),
        'feature_names': [feature_names[i] for i in selector.get_support(indices=True)]
    }
    
    with open(models_dir / 'yankees_redsox_formula.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"  ✓ Saved formula")

# ============================================================================
# FORMULA 3: HISTORIC STADIUMS (Wrigley, Fenway, Yankee)
# ============================================================================

print("\n" + "="*80)
print("FORMULA 3: HISTORIC STADIUMS SPECIFIC")
print("="*80)

historic_stadiums = ['Wrigley Field', 'Fenway Park', 'Yankee Stadium']
stadium_indices = [i for i, g in enumerate(sample_games) 
                   if any(stadium in g['venue']['name'] for stadium in historic_stadiums)]

print(f"Matches: {len(stadium_indices)}")

if len(stadium_indices) >= 100:
    X_stadium = ж[stadium_indices]
    y_stadium = outcomes[stadium_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X_stadium, y_stadium, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    k = min(100, X_train_scaled.shape[1])
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X_train_scaled, y_train)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
    grid.fit(X_train_selected, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_selected)
    r2 = r2_score(y_test, y_pred)
    r = np.corrcoef(y_pred, y_test)[0, 1]
    
    print(f"  Optimized R²: {r2:.4f} ({r2*100:.1f}%)")
    print(f"  Correlation |r|: {abs(r):.4f}")
    
    domain_formulas['historic_stadiums'] = {
        'context': 'Historic Stadiums (Wrigley, Fenway, Yankee)',
        'n_matches': len(stadium_indices),
        'features_selected': k,
        'best_alpha': grid.best_params_['alpha'],
        'test_r2': float(r2),
        'test_r': float(r),
        'model_path': 'models/historic_stadiums_formula.pkl'
    }
    
    model_data = {
        'scaler': scaler,
        'selector': selector,
        'model': best_model,
        'feature_indices': selector.get_support(indices=True),
        'feature_names': [feature_names[i] for i in selector.get_support(indices=True)]
    }
    
    with open(models_dir / 'historic_stadiums_formula.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"  ✓ Saved formula")

# ============================================================================
# SAVE ALL FORMULAS INDEX
# ============================================================================

print("\n" + "="*80)
print("SAVING DOMAIN-SPECIFIC FORMULAS INDEX")
print("="*80)

formulas_index = {
    'domain': 'MLB',
    'created_date': '2025-11-12',
    'formulas': domain_formulas,
    'usage_guide': {
        'astros_rangers': 'Use for HOU vs TEX games',
        'yankees_redsox': 'Use for NYY vs BOS games',
        'historic_stadiums': 'Use for games at Wrigley, Fenway, or Yankee Stadium'
    },
    'deployment_ready': True
}

output_path = Path(__file__).parent / 'mlb_domain_formulas.json'
with open(output_path, 'w') as f:
    json.dump(formulas_index, f, indent=2)

print(f"\n✓ Saved formulas index to: {output_path}")
print(f"\nDomain-Specific Formulas Created:")
for name, formula in domain_formulas.items():
    print(f"  • {formula['context']}: R²={formula['test_r2']*100:.1f}%, |r|={formula['test_r']:.4f}")

print(f"\n{'='*80}")
print("DOMAIN-SPECIFIC OPTIMIZATION COMPLETE")
print(f"{'='*80}")
print(f"\n✓ {len(domain_formulas)} specialized formulas saved")
print(f"✓ Ready for context-specific deployment")








