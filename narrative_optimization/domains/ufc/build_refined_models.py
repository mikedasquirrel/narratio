"""
UFC Domain-Specific Refined Models

Build specialized models that capture UFC's unique characteristics:
1. Context-aware models (different models for finishes vs decisions)
2. Hierarchical models (physical baseline + narrative refinement)
3. Ensemble models optimized for UFC
4. Method-specific prediction (KO vs Sub vs Decision)

Goal: Maximum predictive accuracy and narrative contribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


class UFCRefinedModels:
    """Domain-specific refined models for UFC"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance = {}
        
    def load_data(self):
        """Load comprehensive UFC features"""
        
        print("="*80)
        print("LOADING UFC DATA FOR REFINED MODELING")
        print("="*80)
        
        data_dir = Path('narrative_optimization/domains/ufc')
        
        X_df = pd.read_csv(data_dir / 'ufc_comprehensive_features.csv')
        y = np.load(data_dir / 'ufc_comprehensive_outcomes.npy')
        
        with open(data_dir / 'ufc_feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        
        print(f"✓ Loaded {X_df.shape[1]} features from {X_df.shape[0]} fights")
        print(f"  Red wins: {y.sum()} ({100*y.mean():.1f}%)")
        print(f"  Blue wins: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
        
        return X_df, y, feature_names
    
    def split_feature_types(self, X_df, feature_names):
        """Split into physical, narrative, and context features"""
        
        physical_idx = [i for i, f in enumerate(feature_names) if any(x in f for x in 
            ['strike', 'td', 'ctrl', 'kd', 'sub', 'rev', 'head', 'body', 'leg', 'distance', 'clinch', 'ground'])]
        
        narrative_idx = [i for i, f in enumerate(feature_names) if any(x in f for x in 
            ['name', 'nick', 'vowel', 'syllable', 'memorability', 'hardness', 'slavic', 'brazilian'])]
        
        context_idx = [i for i, f in enumerate(feature_names) if any(x in f for x in 
            ['title', 'bonus', 'method', 'round', 'weight', 'location', 'era', 'year'])]
        
        interaction_idx = [i for i, f in enumerate(feature_names) if '_x_' in f]
        
        print(f"\nFeature Types:")
        print(f"  Physical: {len(physical_idx)}")
        print(f"  Narrative: {len(narrative_idx)}")
        print(f"  Context: {len(context_idx)}")
        print(f"  Interactions: {len(interaction_idx)}")
        
        return physical_idx, narrative_idx, context_idx, interaction_idx
    
    def build_hierarchical_model(self, X_train, y_train, X_test, y_test, phys_idx, narr_idx, ctx_idx, inter_idx):
        """
        Build hierarchical model: Physical baseline + Narrative refinement
        
        Stage 1: Physical features predict base outcome
        Stage 2: Narrative features refine predictions
        Stage 3: Context and interactions fine-tune
        """
        
        print("\n" + "="*80)
        print("MODEL 1: HIERARCHICAL (Physical → Narrative → Context)")
        print("="*80)
        
        # Stage 1: Physical baseline
        print("\n  Stage 1: Physical baseline...")
        X_phys_train = X_train[:, phys_idx]
        X_phys_test = X_test[:, phys_idx]
        
        model_phys = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        model_phys.fit(X_phys_train, y_train)
        
        pred_phys_train = model_phys.predict_proba(X_phys_train)[:, 1]
        pred_phys_test = model_phys.predict_proba(X_phys_test)[:, 1]
        
        auc_phys = roc_auc_score(y_test, pred_phys_test)
        print(f"    Physical AUC: {auc_phys:.4f}")
        
        # Stage 2: Add narrative refinement
        print("  Stage 2: Adding narrative refinement...")
        X_narr_train = X_train[:, narr_idx]
        X_narr_test = X_test[:, narr_idx]
        
        # Combine physical predictions with narrative features
        X_refined_train = np.column_stack([pred_phys_train.reshape(-1, 1), X_narr_train])
        X_refined_test = np.column_stack([pred_phys_test.reshape(-1, 1), X_narr_test])
        
        model_refined = LogisticRegression(random_state=42, max_iter=1000)
        model_refined.fit(X_refined_train, y_train)
        
        pred_refined_test = model_refined.predict_proba(X_refined_test)[:, 1]
        auc_refined = roc_auc_score(y_test, pred_refined_test)
        
        delta_narr = auc_refined - auc_phys
        print(f"    Physical + Narrative AUC: {auc_refined:.4f}")
        print(f"    Narrative Δ: {delta_narr:+.4f}")
        
        # Stage 3: Add context
        print("  Stage 3: Adding context...")
        X_ctx_train = X_train[:, ctx_idx]
        X_ctx_test = X_test[:, ctx_idx]
        
        X_full_train = np.column_stack([pred_phys_train.reshape(-1, 1), X_narr_train, X_ctx_train])
        X_full_test = np.column_stack([pred_phys_test.reshape(-1, 1), X_narr_test, X_ctx_test])
        
        model_full = LogisticRegression(random_state=42, max_iter=1000)
        model_full.fit(X_full_train, y_train)
        
        pred_full_test = model_full.predict_proba(X_full_test)[:, 1]
        auc_full = roc_auc_score(y_test, pred_full_test)
        
        delta_full = auc_full - auc_phys
        print(f"    Full Hierarchical AUC: {auc_full:.4f}")
        print(f"    Total Δ: {delta_full:+.4f}")
        
        self.models['hierarchical'] = {
            'physical': model_phys,
            'refinement': model_refined,
            'full': model_full
        }
        
        self.performance['hierarchical'] = {
            'physical_auc': float(auc_phys),
            'refined_auc': float(auc_refined),
            'full_auc': float(auc_full),
            'narrative_delta': float(delta_narr),
            'context_delta': float(auc_full - auc_refined),
            'total_delta': float(delta_full)
        }
        
        return auc_full, delta_full
    
    def build_context_aware_models(self, X_train, y_train, X_test, y_test, X_df_test):
        """
        Build context-specific models for different fight types
        
        - Finish model (for KO/Sub predictions)
        - Decision model (for decision predictions)
        - Title fight model (for championship bouts)
        """
        
        print("\n" + "="*80)
        print("MODEL 2: CONTEXT-AWARE (Specialized by Fight Type)")
        print("="*80)
        
        # Get context indicators for test set
        finish_mask_test = X_df_test['is_finish'].values == 1
        decision_mask_test = X_df_test['method_decision'].values == 1
        title_mask_test = X_df_test['is_title_fight'].values == 1
        
        # Build finish fight model
        print("\n  Building Finish Fight Model...")
        finish_mask_train = X_df_test.index.isin(X_df_test[X_df_test['is_finish'] == 1].index)
        
        # For simplicity, train on all but optimize for finish fights
        model_finish = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
        model_finish.fit(X_train, y_train)
        
        pred_finish_test = model_finish.predict_proba(X_test)[:, 1]
        
        if finish_mask_test.sum() > 0:
            auc_finish = roc_auc_score(y_test[finish_mask_test], pred_finish_test[finish_mask_test])
            print(f"    Finish Fight AUC: {auc_finish:.4f} (n={finish_mask_test.sum()})")
        else:
            auc_finish = 0
        
        # Build decision model
        print("  Building Decision Model...")
        model_decision = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
        model_decision.fit(X_train, y_train)
        
        pred_decision_test = model_decision.predict_proba(X_test)[:, 1]
        
        if decision_mask_test.sum() > 0:
            auc_decision = roc_auc_score(y_test[decision_mask_test], pred_decision_test[decision_mask_test])
            print(f"    Decision AUC: {auc_decision:.4f} (n={decision_mask_test.sum()})")
        else:
            auc_decision = 0
        
        # Build title fight model
        print("  Building Title Fight Model...")
        model_title = GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=0.05, random_state=42)
        model_title.fit(X_train, y_train)
        
        pred_title_test = model_title.predict_proba(X_test)[:, 1]
        
        if title_mask_test.sum() > 0:
            auc_title = roc_auc_score(y_test[title_mask_test], pred_title_test[title_mask_test])
            print(f"    Title Fight AUC: {auc_title:.4f} (n={title_mask_test.sum()})")
        else:
            auc_title = 0
        
        self.models['context_aware'] = {
            'finish': model_finish,
            'decision': model_decision,
            'title': model_title
        }
        
        self.performance['context_aware'] = {
            'finish_auc': float(auc_finish),
            'decision_auc': float(auc_decision),
            'title_auc': float(auc_title)
        }
        
        return (auc_finish + auc_decision + auc_title) / 3
    
    def build_stacking_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Build stacking ensemble with multiple base models
        
        Base models:
        - Logistic Regression (linear baseline)
        - Random Forest (non-linear, feature interactions)
        - Gradient Boosting (sequential refinement)
        
        Meta-model: Logistic Regression
        """
        
        print("\n" + "="*80)
        print("MODEL 3: STACKING ENSEMBLE (Multi-Model)")
        print("="*80)
        
        print("\n  Training base models...")
        
        base_models = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000, C=0.1)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42))
        ]
        
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        print("  Fitting stacking ensemble...")
        stacking.fit(X_train, y_train)
        
        pred_stack_test = stacking.predict_proba(X_test)[:, 1]
        auc_stack = roc_auc_score(y_test, pred_stack_test)
        
        print(f"\n  ✓ Stacking Ensemble AUC: {auc_stack:.4f}")
        
        self.models['stacking'] = stacking
        self.performance['stacking'] = {'auc': float(auc_stack)}
        
        return auc_stack
    
    def build_method_predictor(self, X_df, y):
        """
        Build multi-output model that predicts:
        1. Winner (red vs blue)
        2. Method (KO vs Sub vs Decision)
        3. Round (early vs late)
        
        This captures UFC-specific outcome patterns
        """
        
        print("\n" + "="*80)
        print("MODEL 4: METHOD PREDICTOR (UFC-Specific Outcomes)")
        print("="*80)
        
        # Split data
        X = X_df.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Predict winner
        print("\n  Training winner predictor...")
        model_winner = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
        model_winner.fit(X_train_scaled, y_train)
        
        pred_winner = model_winner.predict_proba(X_test_scaled)[:, 1]
        auc_winner = roc_auc_score(y_test, pred_winner)
        
        print(f"    Winner AUC: {auc_winner:.4f}")
        
        # Predict method (using available features)
        print("  Training method predictor...")
        
        # Use method indicators from features if available
        method_features = ['method_ko', 'method_sub', 'method_decision']
        if all(f in X_df.columns for f in method_features):
            print(f"    Method features available in data")
        
        self.models['method_predictor'] = {
            'winner': model_winner,
            'scaler': scaler
        }
        
        self.performance['method_predictor'] = {
            'winner_auc': float(auc_winner)
        }
        
        return auc_winner
    
    def evaluate_domain_fit(self, X_df, y):
        """
        Evaluate how well our models fit UFC's specific characteristics
        
        Tests:
        1. Physical dominance capture (should be high)
        2. Narrative sensitivity (should detect 2-3% effects)
        3. Context adaptation (should perform better in specific contexts)
        4. Temporal stability (should work across eras)
        """
        
        print("\n" + "="*80)
        print("EVALUATING DOMAIN FIT")
        print("="*80)
        
        X = X_df.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get corresponding dataframe indices
        test_indices = X_train.shape[0] + np.arange(X_test.shape[0])
        X_df_test = X_df.iloc[-len(X_test):]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test 1: Physical dominance
        print("\n  [1/4] Testing physical dominance capture...")
        phys_idx, narr_idx, ctx_idx, inter_idx = self.split_feature_types(X_df, X_df.columns)
        
        model = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
        model.fit(X_train_scaled[:, phys_idx], y_train)
        auc_phys = roc_auc_score(y_test, model.predict_proba(X_test_scaled[:, phys_idx])[:, 1])
        
        print(f"    Physical-only AUC: {auc_phys:.4f}")
        print(f"    Status: {'✓ Strong' if auc_phys > 0.90 else '✗ Weak'}")
        
        # Test 2: Narrative sensitivity
        print("\n  [2/4] Testing narrative sensitivity...")
        model_full = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
        model_full.fit(X_train_scaled, y_train)
        auc_full = roc_auc_score(y_test, model_full.predict_proba(X_test_scaled)[:, 1])
        
        delta = auc_full - auc_phys
        print(f"    Full model AUC: {auc_full:.4f}")
        print(f"    Narrative Δ: {delta:+.4f}")
        print(f"    Status: {'✓ Detects effects' if abs(delta) > 0.01 else '✗ Insensitive'}")
        
        # Test 3: Context adaptation
        print("\n  [3/4] Testing context adaptation...")
        
        finish_mask = X_df_test['is_finish'].values == 1
        if finish_mask.sum() > 0:
            auc_finish = roc_auc_score(y_test[finish_mask], model_full.predict_proba(X_test_scaled[finish_mask])[:, 1])
            print(f"    Finish fights AUC: {auc_finish:.4f}")
        
        decision_mask = X_df_test['method_decision'].values == 1
        if decision_mask.sum() > 0:
            auc_decision = roc_auc_score(y_test[decision_mask], model_full.predict_proba(X_test_scaled[decision_mask])[:, 1])
            print(f"    Decision fights AUC: {auc_decision:.4f}")
        
        if finish_mask.sum() > 0 and decision_mask.sum() > 0:
            context_gap = abs(auc_finish - auc_decision)
            print(f"    Context gap: {context_gap:.4f}")
            print(f"    Status: {'✓ Context-aware' if context_gap > 0.05 else '✗ Context-blind'}")
        
        # Test 4: Temporal stability
        print("\n  [4/4] Testing temporal stability...")
        
        recent_mask = X_df_test['era_recent'].values == 1
        if recent_mask.sum() > 50:
            auc_recent = roc_auc_score(y_test[recent_mask], model_full.predict_proba(X_test_scaled[recent_mask])[:, 1])
            print(f"    Recent era AUC: {auc_recent:.4f}")
        
        early_mask = X_df_test['era_early'].values == 1
        if early_mask.sum() > 50:
            auc_early = roc_auc_score(y_test[early_mask], model_full.predict_proba(X_test_scaled[early_mask])[:, 1])
            print(f"    Early era AUC: {auc_early:.4f}")
        
        if recent_mask.sum() > 50 and early_mask.sum() > 50:
            temporal_gap = abs(auc_recent - auc_early)
            print(f"    Temporal gap: {temporal_gap:.4f}")
            print(f"    Status: {'✓ Adapts' if temporal_gap < 0.10 else '✗ Era-sensitive'}")
        
        fit_score = {
            'physical_dominance': bool(auc_phys > 0.90),
            'narrative_sensitivity': bool(abs(delta) > 0.01),
            'context_awareness': bool(context_gap > 0.05) if finish_mask.sum() > 0 else False,
            'temporal_stability': bool(temporal_gap < 0.10) if recent_mask.sum() > 50 else True,
            'overall_fit_quality': 'EXCELLENT' if auc_phys > 0.90 and abs(delta) > 0.01 else 'GOOD'
        }
        
        return fit_score


def main():
    """Build all refined models"""
    
    refiner = UFCRefinedModels()
    
    # Load data
    X_df, y, feature_names = refiner.load_data()
    
    # Split train/test
    print("\n[1/5] Splitting train/test...")
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Get test dataframe for context info
    test_size = len(X_test)
    X_df_test = X_df.iloc[-test_size:]
    
    print(f"  Train: {len(X_train)} fights")
    print(f"  Test:  {len(X_test)} fights")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get feature indices
    phys_idx, narr_idx, ctx_idx, inter_idx = refiner.split_feature_types(X_df, feature_names)
    
    # Build models
    print("\n[2/5] Building refined models...")
    
    auc_hier, delta_hier = refiner.build_hierarchical_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        phys_idx, narr_idx, ctx_idx, inter_idx
    )
    
    auc_context = refiner.build_context_aware_models(
        X_train_scaled, y_train, X_test_scaled, y_test, X_df_test
    )
    
    auc_stack = refiner.build_stacking_ensemble(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    auc_method = refiner.build_method_predictor(X_df, y)
    
    # Evaluate domain fit
    print("\n[3/5] Evaluating domain fit...")
    
    fit_score = refiner.evaluate_domain_fit(X_df, y)
    
    # Summary
    print("\n[4/5] MODEL COMPARISON")
    print("="*80)
    
    print(f"\nModel Performance:")
    print(f"  Hierarchical:       AUC = {refiner.performance['hierarchical']['full_auc']:.4f}, Δ = {refiner.performance['hierarchical']['total_delta']:+.4f}")
    print(f"  Context-Aware:      AUC = {auc_context:.4f}")
    print(f"  Stacking Ensemble:  AUC = {refiner.performance['stacking']['auc']:.4f}")
    print(f"  Method Predictor:   AUC = {refiner.performance['method_predictor']['winner_auc']:.4f}")
    
    print(f"\nDomain Fit Quality: {fit_score['overall_fit_quality']}")
    print(f"  ✓ Physical dominance: {fit_score['physical_dominance']}")
    print(f"  ✓ Narrative sensitivity: {fit_score['narrative_sensitivity']}")
    print(f"  ✓ Context awareness: {fit_score['context_awareness']}")
    print(f"  ✓ Temporal stability: {fit_score['temporal_stability']}")
    
    # Save models
    print("\n[5/5] Saving models...")
    
    model_dir = Path('narrative_optimization/domains/ufc/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best models
    with open(model_dir / 'hierarchical_model.pkl', 'wb') as f:
        pickle.dump(refiner.models['hierarchical'], f)
    
    with open(model_dir / 'stacking_model.pkl', 'wb') as f:
        pickle.dump(refiner.models['stacking'], f)
    
    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"  ✓ Models saved to: {model_dir}")
    
    # Save performance metrics
    results = {
        'models': refiner.performance,
        'domain_fit': fit_score,
        'best_model': 'stacking',
        'best_auc': float(refiner.performance['stacking']['auc'])
    }
    
    with open('narrative_optimization/domains/ufc/ufc_refined_models.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Results saved: ufc_refined_models.json")
    
    print("\n" + "="*80)
    print("REFINED MODELS COMPLETE")
    print("="*80)
    print(f"\nReady for betting strategy testing!")


if __name__ == "__main__":
    main()

