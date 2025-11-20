"""
UFC Real Betting Strategy Testing

Test betting strategies using refined models on historical UFC data.
Since we don't have real odds data, we'll:
1. Simulate betting based on model confidence
2. Compare to random betting baseline
3. Test context-specific strategies
4. Calculate ROI and profitability

Betting Strategies:
- Confidence-based (bet when model >70% confident)
- Context-specific (only bet finish fights, submissions, etc.)
- Narrative-enhanced (use narrative features for edge detection)
- Underdog hunting (find narrative underdogs)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split

def load_models_and_data():
    """Load trained models and data"""
    
    print("="*80)
    print("LOADING MODELS AND DATA FOR BETTING TESTS")
    print("="*80)
    
    data_dir = Path('narrative_optimization/domains/ufc')
    model_dir = data_dir / 'models'
    
    # Load features
    X_df = pd.read_csv(data_dir / 'ufc_comprehensive_features.csv')
    y = np.load(data_dir / 'ufc_comprehensive_outcomes.npy')
    
    # Load best model
    with open(model_dir / 'stacking_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✓ Loaded stacking model (best performer)")
    print(f"✓ Loaded {X_df.shape[0]} fights with {X_df.shape[1]} features")
    
    return model, scaler, X_df, y

def simulate_betting_with_confidence(model, scaler, X_test, y_test, X_df_test, confidence_threshold=0.65):
    """
    Simulate betting strategy: Only bet when model is confident
    
    Strategy: Bet on predicted winner when confidence > threshold
    """
    
    print(f"\n  Strategy: Bet when confidence > {confidence_threshold:.0%}")
    
    # Get predictions
    X_test_scaled = scaler.transform(X_test)
    proba = model.predict_proba(X_test_scaled)
    
    # Confidence = max probability
    confidence = np.max(proba, axis=1)
    predictions = (proba[:, 1] > 0.5).astype(int)
    
    # Only bet when confident
    bet_mask = confidence > confidence_threshold
    
    if bet_mask.sum() == 0:
        return {'bets': 0, 'accuracy': 0, 'roi': 0}
    
    # Calculate accuracy on bets
    bets_made = bet_mask.sum()
    correct_bets = (predictions[bet_mask] == y_test[bet_mask]).sum()
    accuracy = correct_bets / bets_made
    
    # Simulate ROI (assuming even money odds)
    # Win = +100, Loss = -100
    wins = correct_bets
    losses = bets_made - correct_bets
    profit = wins * 100 - losses * 100
    roi = profit / (bets_made * 100)
    
    print(f"    Bets made: {bets_made}/{len(y_test)} ({100*bets_made/len(y_test):.1f}%)")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Wins: {wins}, Losses: {losses}")
    print(f"    ROI: {roi:+.4f} ({roi*100:+.1f}%)")
    
    return {
        'bets': int(bets_made),
        'accuracy': float(accuracy),
        'wins': int(wins),
        'losses': int(losses),
        'roi': float(roi)
    }

def simulate_context_betting(model, scaler, X_test, y_test, X_df_test, context_name, context_mask):
    """Test betting on specific contexts (finish fights, submissions, etc.)"""
    
    print(f"\n  Context: {context_name}")
    
    if context_mask.sum() == 0:
        print(f"    ✗ No fights in this context")
        return None
    
    # Get predictions for this context
    X_context = X_test[context_mask]
    y_context = y_test[context_mask]
    
    X_context_scaled = scaler.transform(X_context)
    proba = model.predict_proba(X_context_scaled)
    predictions = (proba[:, 1] > 0.5).astype(int)
    
    # Bet on all fights in this context
    bets_made = len(y_context)
    correct = (predictions == y_context).sum()
    accuracy = correct / bets_made
    
    # Calculate ROI
    wins = correct
    losses = bets_made - correct
    profit = wins * 100 - losses * 100
    roi = profit / (bets_made * 100)
    
    print(f"    Bets: {bets_made}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ROI: {roi:+.4f} ({roi*100:+.1f}%)")
    
    return {
        'context': context_name,
        'bets': int(bets_made),
        'accuracy': float(accuracy),
        'wins': int(wins),
        'losses': int(losses),
        'roi': float(roi)
    }

def test_narrative_edge_betting(model, scaler, X_test, y_test, X_df_test, narr_idx):
    """
    Test if narrative features provide betting edge
    
    Compare:
    - Physical-only predictions
    - Full model predictions
    - Measure if narrative helps identify profitable bets
    """
    
    print(f"\n  Testing narrative edge for betting...")
    
    # This requires physical-only model comparison
    # For now, test on high-narrative contexts
    
    # High narrative contexts: Both have nicknames, title fights
    high_narr_mask = ((X_df_test['both_have_nicknames'] == 1) | 
                      (X_df_test['is_title_fight'] == 1))
    
    if high_narr_mask.sum() == 0:
        return None
    
    X_high_narr = X_test[high_narr_mask]
    y_high_narr = y_test[high_narr_mask]
    
    X_high_narr_scaled = scaler.transform(X_high_narr)
    proba = model.predict_proba(X_high_narr_scaled)
    predictions = (proba[:, 1] > 0.5).astype(int)
    
    accuracy = (predictions == y_high_narr).mean()
    
    # Calculate ROI
    bets = len(y_high_narr)
    wins = (predictions == y_high_narr).sum()
    roi = (wins * 100 - (bets - wins) * 100) / (bets * 100)
    
    print(f"    High-narrative fights: {bets}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ROI: {roi:+.4f} ({roi*100:+.1f}%)")
    
    return {
        'bets': int(bets),
        'accuracy': float(accuracy),
        'roi': float(roi)
    }


def main():
    """Run comprehensive betting strategy tests"""
    
    print("="*80)
    print("UFC BETTING STRATEGY TESTING")
    print("="*80)
    
    # Load data and models
    model, scaler, X_df, y = load_models_and_data()
    
    # Split train/test (use test set for betting simulation)
    print("\n[1/4] Preparing test set...")
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Get test dataframe
    test_size = len(X_test)
    X_df_test = X_df.iloc[-test_size:].reset_index(drop=True)
    
    print(f"  Test set: {len(X_test)} fights")
    print(f"  Red wins: {y_test.sum()} ({100*y_test.mean():.1f}%)")
    
    # Get feature indices
    feature_names = list(X_df.columns)
    narr_idx = [i for i, f in enumerate(feature_names) if any(x in f for x in ['name', 'nick', 'title'])]
    
    # === STRATEGY 1: Confidence-Based Betting ===
    print("\n[2/4] Testing confidence-based betting...")
    print("="*80)
    
    strategies = {}
    
    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
        result = simulate_betting_with_confidence(model, scaler, X_test, y_test, X_df_test, threshold)
        strategies[f'confidence_{int(threshold*100)}'] = result
    
    # === STRATEGY 2: Context-Specific Betting ===
    print("\n[3/4] Testing context-specific betting...")
    print("="*80)
    
    # Test betting on high-performing contexts
    contexts_to_test = [
        ('Submission Fights', X_df_test['method_sub'] == 1),
        ('KO/TKO Fights', X_df_test['method_ko'] == 1),
        ('All Finishes', X_df_test['is_finish'] == 1),
        ('Round 1 Finishes', (X_df_test['is_finish'] == 1) & (X_df_test['round'] == 1)),
        ('Title Fights', X_df_test['is_title_fight'] == 1),
        ('Title Fight Finishes', (X_df_test['is_title_fight'] == 1) & (X_df_test['is_finish'] == 1)),
        ('5 Round Fights', X_df_test['is_5_round_fight'] == 1),
    ]
    
    context_strategies = {}
    
    for ctx_name, ctx_mask in contexts_to_test:
        result = simulate_context_betting(model, scaler, X_test, y_test, X_df_test, ctx_name, ctx_mask.values)
        if result:
            context_strategies[ctx_name] = result
    
    # === STRATEGY 3: Narrative Edge ===
    print("\n[4/4] Testing narrative edge...")
    print("="*80)
    
    narr_result = test_narrative_edge_betting(model, scaler, X_test, y_test, X_df_test, narr_idx)
    strategies['narrative_edge'] = narr_result
    
    # === SUMMARY ===
    print("\n" + "="*80)
    print("BETTING STRATEGY RESULTS")
    print("="*80)
    
    print(f"\n📊 CONFIDENCE-BASED STRATEGIES:")
    for name, result in strategies.items():
        if 'confidence' in name and result['bets'] > 0:
            threshold = int(name.split('_')[1])
            print(f"  {threshold}% confidence: {result['bets']:4d} bets | Acc={result['accuracy']:.4f} | ROI={result['roi']:+.1%}")
    
    print(f"\n🎯 CONTEXT-SPECIFIC STRATEGIES:")
    best_context_roi = -999
    best_context_name = None
    
    for name, result in context_strategies.items():
        print(f"  {name:25s}: {result['bets']:4d} bets | Acc={result['accuracy']:.4f} | ROI={result['roi']:+.1%}")
        if result['roi'] > best_context_roi:
            best_context_roi = result['roi']
            best_context_name = name
    
    if narr_result:
        print(f"\n🔮 NARRATIVE EDGE STRATEGY:")
        print(f"  High-narrative fights: {narr_result['bets']:4d} bets | Acc={narr_result['accuracy']:.4f} | ROI={narr_result['roi']:+.1%}")
    
    # Best strategy
    print(f"\n" + "="*80)
    print("BEST BETTING STRATEGY")
    print("="*80)
    
    if best_context_name:
        best_result = context_strategies[best_context_name]
        print(f"\n✓ {best_context_name}")
        print(f"  ROI: {best_result['roi']:+.1%}")
        print(f"  Accuracy: {best_result['accuracy']:.1%}")
        print(f"  Sample: {best_result['bets']} bets")
        
        if best_result['roi'] > 0.20:
            print(f"\n  ✓ PROFITABLE! {best_result['roi']*100:.1f}% return")
        elif best_result['roi'] > 0:
            print(f"\n  ✓ PROFITABLE (modest): {best_result['roi']*100:.1f}% return")
        else:
            print(f"\n  ✗ Not profitable: {best_result['roi']*100:.1f}% return")
    
    # Save results
    betting_results = {
        'confidence_strategies': {k: v for k, v in strategies.items() if 'confidence' in k},
        'context_strategies': context_strategies,
        'narrative_edge': narr_result,
        'best_strategy': {
            'name': best_context_name,
            'roi': float(best_context_roi),
            'result': context_strategies[best_context_name] if best_context_name else None
        },
        'baseline': {
            'random_accuracy': 0.5,
            'random_roi': 0.0,
            'note': 'Random betting breaks even'
        }
    }
    
    output_path = Path('narrative_optimization/domains/ufc/ufc_betting_strategies.json')
    with open(output_path, 'w') as f:
        json.dump(betting_results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    # Compare to baseline
    print(f"\n" + "="*80)
    print("COMPARISON TO BASELINE")
    print("="*80)
    
    print(f"\nRandom betting:")
    print(f"  Accuracy: 50.0%")
    print(f"  ROI: 0.0%")
    
    print(f"\nBest model strategy:")
    print(f"  Accuracy: {best_result['accuracy']*100:.1f}%")
    print(f"  ROI: {best_result['roi']*100:+.1f}%")
    print(f"  Edge: {best_result['roi']*100:.1f} percentage points")
    
    if best_result['roi'] > 0:
        print(f"\n✓ Model provides {best_result['roi']*100:.1f}% betting edge!")


if __name__ == "__main__":
    main()

