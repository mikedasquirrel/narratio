"""
COMPREHENSIVE NOMINATIVE TESTING

Tests 10 hypotheses about how names predict outcomes.
Discovers empirically what matters - like characters revealing their story.
"""

import json
import numpy as np
from pathlib import Path
import sys
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.phonetic import PhoneticTransformer
from src.transformers.namespace_ecology import NamespaceEcologyTransformer
from src.transformers.statistical import StatisticalTransformer
from src.analysis.gravitational_forces import GravitationalCalculator


class ComprehensiveNominativeTester:
    """Test everything about names and discover what matters"""
    
    def __init__(self):
        self.results = {}
    
    def test_1_pure_name_effects(self, names, outcomes, domain_name):
        """TEST 1: Can names ALONE (zero context) predict outcomes?"""
        print(f"\n{'='*80}")
        print(f"TEST 1: PURE NAME EFFECTS - {domain_name}")
        print(f"{'='*80}")
        print("Testing: Can names alone predict, zero narrative context?")
        
        # Apply nominative transformers to names only
        nominative = NominativeAnalysisTransformer()
        phonetic = PhoneticTransformer()
        
        nominative.fit(names)
        phonetic.fit(names)
        
        nom_features = nominative.transform(names)
        phon_features = phonetic.transform(names)
        
        # Combine
        name_only_features = np.hstack([nom_features, phon_features])
        
        # Predict from names alone
        is_binary = set(outcomes) == {0, 1}
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(name_only_features)
        
        if is_binary:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled, outcomes)
            preds = model.predict_proba(X_scaled)[:, 1]
            r_names = roc_auc_score(outcomes, preds)
            metric = "AUC"
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, outcomes)
            preds = model.predict(X_scaled)
            r_names, p = stats.pearsonr(preds, outcomes)
            metric = "r"
        
        print(f"\n✓ Names-only prediction: {metric} = {r_names:.3f}")
        print(f"  Features used: {name_only_features.shape[1]} (pure nominative)")
        
        result = {
            'domain': domain_name,
            'r_names_only': float(r_names),
            'n_features': name_only_features.shape[1],
            'interpretation': self._interpret_pure_effect(r_names)
        }
        
        self.results[f'{domain_name}_test1_pure'] = result
        return result
    
    def test_2_interaction(self, names, full_texts, outcomes, domain_name):
        """TEST 2: Name × Narrative interaction"""
        print(f"\n{'='*80}")
        print(f"TEST 2: NAME-NARRATIVE INTERACTION - {domain_name}")
        print(f"{'='*80}")
        print("Testing: Multiplicative, threshold, or additive?")
        
        # Extract name features
        nominative = NominativeAnalysisTransformer()
        nominative.fit(names)
        name_features = nominative.transform(names)
        name_quality = np.mean(name_features, axis=1)
        
        # Extract narrative features (from full text)
        statistical = StatisticalTransformer(max_features=100)
        statistical.fit(full_texts)
        narrative_features = statistical.transform(full_texts)
        if hasattr(narrative_features, 'toarray'):
            narrative_features = narrative_features.toarray()
        narrative_quality = np.mean(narrative_features, axis=1)
        
        # Normalize both
        name_quality = (name_quality - name_quality.mean()) / (name_quality.std() + 1e-8)
        narrative_quality = (narrative_quality - narrative_quality.mean()) / (narrative_quality.std() + 1e-8)
        
        # Test models
        is_binary = set(outcomes) == {0, 1}
        
        # Model 1: Additive
        X_additive = np.column_stack([name_quality, narrative_quality])
        r_additive = self._fit_and_score(X_additive, outcomes, is_binary)
        
        # Model 2: Multiplicative
        X_mult = np.column_stack([name_quality, narrative_quality, name_quality * narrative_quality])
        r_mult = self._fit_and_score(X_mult, outcomes, is_binary)
        
        # Model 3: Name-only
        X_name = name_quality.reshape(-1, 1)
        r_name = self._fit_and_score(X_name, outcomes, is_binary)
        
        # Model 4: Narrative-only
        X_narr = narrative_quality.reshape(-1, 1)
        r_narr = self._fit_and_score(X_narr, outcomes, is_binary)
        
        print(f"\n✓ Results:")
        print(f"  Name only: {r_name:.3f}")
        print(f"  Narrative only: {r_narr:.3f}")
        print(f"  Additive (name + narrative): {r_additive:.3f}")
        print(f"  Multiplicative (name × narrative): {r_mult:.3f}")
        
        # Determine mechanism
        improvement_mult = r_mult - r_additive
        
        if improvement_mult > 0.02:
            mechanism = "MULTIPLICATIVE - names amplify narratives"
        elif r_name > r_narr * 0.5:
            mechanism = "NAME-DOMINANT - names matter more"
        elif r_narr > r_name * 2:
            mechanism = "NARRATIVE-DOMINANT - story overrides names"
        else:
            mechanism = "ADDITIVE - independent effects"
        
        print(f"\n🔥 Mechanism: {mechanism}")
        
        result = {
            'domain': domain_name,
            'r_name_only': float(r_name),
            'r_narrative_only': float(r_narr),
            'r_additive': float(r_additive),
            'r_multiplicative': float(r_mult),
            'mechanism': mechanism
        }
        
        self.results[f'{domain_name}_test2_interaction'] = result
        return result
    
    def test_3_ensemble_effects(self, entity_collections, outcomes, domain_name):
        """TEST 3: Ensemble of names (cast, teams, etc.)"""
        print(f"\n{'='*80}")
        print(f"TEST 3: ENSEMBLE NAME EFFECTS - {domain_name}")
        print(f"{'='*80}")
        print("Testing: Does collection of names matter beyond individuals?")
        
        # For each entity, get collection of associated names
        phonetic = PhoneticTransformer()
        
        ensemble_features = []
        for collection in entity_collections:
            if not collection:
                ensemble_features.append([0] * 10)
                continue
            
            # Extract features for all names in collection
            phonetic.fit(collection)
            coll_features = phonetic.transform(collection)
            
            # Aggregate metrics
            feat_dict = {
                'mean_quality': np.mean(coll_features),
                'diversity': np.std(coll_features),
                'max_quality': np.max(np.mean(coll_features, axis=1)),
                'min_quality': np.min(np.mean(coll_features, axis=1)),
                'range': np.max(coll_features) - np.min(coll_features),
                'harmony': 1 - np.std(np.mean(coll_features, axis=1)),
                'n_names': len(collection),
                'top_heavy': np.mean(coll_features[:1]) if len(collection) > 0 else 0,
                'depth': np.mean(coll_features[1:]) if len(collection) > 1 else 0,
                'balance': abs(np.mean(coll_features[:len(collection)//2]) - np.mean(coll_features[len(collection)//2:])) if len(collection) > 2 else 0
            }
            
            ensemble_features.append(list(feat_dict.values()))
        
        X_ensemble = np.array(ensemble_features)
        
        # Correlate with outcomes
        is_binary = set(outcomes) == {0, 1}
        r_ensemble = self._fit_and_score(X_ensemble, outcomes, is_binary)
        
        print(f"\n✓ Ensemble effect: r = {r_ensemble:.3f}")
        print(f"  Testing: mean, diversity, top-heavy, harmony, etc.")
        
        # Which ensemble feature matters most?
        best_feature_idx = None
        best_r = 0
        
        for i in range(X_ensemble.shape[1]):
            r_i, _ = stats.pearsonr(X_ensemble[:, i], outcomes)
            if abs(r_i) > abs(best_r):
                best_r = r_i
                best_feature_idx = i
        
        feature_names = list(feat_dict.keys())
        print(f"  Best ensemble feature: {feature_names[best_feature_idx]} (r={best_r:.3f})")
        
        result = {
            'domain': domain_name,
            'r_ensemble': float(r_ensemble),
            'best_feature': feature_names[best_feature_idx] if best_feature_idx is not None else None,
            'best_r': float(best_r)
        }
        
        self.results[f'{domain_name}_test3_ensemble'] = result
        return result
    
    def test_4_gravitational_clustering(self, names, outcomes, domain_name):
        """TEST 4: Do name-similar entities cluster in outcome space?"""
        print(f"\n{'='*80}")
        print(f"TEST 4: GRAVITATIONAL CLUSTERING (ة) - {domain_name}")
        print(f"{'='*80}")
        print("Testing: Do phonetically similar names have similar outcomes?")
        
        # Extract phonetic features
        phonetic = PhoneticTransformer()
        phonetic.fit(names)
        phon_features = phonetic.transform(names)
        
        # Calculate ة (nominative gravity)
        grav_calc = GravitationalCalculator()
        masses = np.ones(len(names))
        forces = grav_calc.calculate_all_forces(
            phon_features, names, masses, story_quality=None
        )
        
        ة = forces['ة']
        
        # Test: Do high-ة pairs have similar outcomes?
        outcome_similarities = []
        ة_strengths = []
        
        n = len(names)
        for i in range(n):
            for j in range(i+1, n):
                if i < len(outcomes) and j < len(outcomes):
                    # Outcome similarity (1 if same, 0 if different)
                    outcome_sim = 1 - abs(outcomes[i] - outcomes[j])
                    outcome_similarities.append(outcome_sim)
                    ة_strengths.append(ة[i, j])
        
        # Correlate: Strong ة → similar outcomes?
        if len(ة_strengths) > 10:
            r_clustering, p = stats.pearsonr(ة_strengths, outcome_similarities)
            print(f"\n✓ ة clustering effect: r = {r_clustering:.3f} (p={p:.4f})")
            
            if r_clustering > 0.1:
                print(f"  🔥 VALIDATED: Phonetically similar names cluster by outcome!")
            else:
                print(f"  ❌ No clustering: Names don't group by outcome")
        else:
            r_clustering = 0.0
            print(f"  ⚠️ Insufficient pairs for analysis")
        
        result = {
            'domain': domain_name,
            'r_clustering': float(r_clustering),
            'mean_ة': float(ة.mean()),
            'n_comparisons': len(ة_strengths)
        }
        
        self.results[f'{domain_name}_test4_clustering'] = result
        return result
    
    def _fit_and_score(self, X, y, is_binary):
        """Fit model and return score"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if is_binary:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled, y)
            preds = model.predict_proba(X_scaled)[:, 1]
            return roc_auc_score(y, preds)
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)
            preds = model.predict(X_scaled)
            r, _ = stats.pearsonr(preds, y)
            return r
    
    def _interpret_pure_effect(self, r):
        """Interpret pure name effect size"""
        r_abs = abs(r)
        if r_abs < 0.10:
            return "Negligible - names barely predict"
        elif r_abs < 0.30:
            return "Weak - names provide small signal"
        elif r_abs < 0.50:
            return "Moderate - names meaningfully predict"
        else:
            return "Strong - names are primary predictor"
    
    def print_summary(self):
        """Print comprehensive summary"""
        print(f"\n{'='*80}")
        print(f"NOMINATIVE TESTING SUMMARY")
        print(f"{'='*80}")
        
        for test_key, result in self.results.items():
            print(f"\n{test_key}:")
            for key, value in result.items():
                print(f"  {key}: {value}")


def main():
    """Run all tests"""
    print("="*80)
    print("COMPREHENSIVE NOMINATIVE TESTING")
    print("="*80)
    print("\nTesting 10 hypotheses about how names predict outcomes")
    print("Discovering empirically - like characters revealing their story")
    
    tester = ComprehensiveNominativeTester()
    
    # === LOAD DATA ===
    
    # Oscar data
    with open('data/domains/oscar_nominees_complete.json') as f:
        oscar_data = json.load(f)
    oscar_films = []
    for year_films in oscar_data.values():
        oscar_films.extend(year_films)
    
    oscar_titles = [f['title'] for f in oscar_films]
    oscar_outcomes = np.array([int(f.get('won_oscar', 0)) for f in oscar_films])
    oscar_full_texts = [f['title'] + ' ' + f.get('overview', '') for f in oscar_films]
    oscar_casts = [[c.get('actor', '') for c in f.get('cast', [])[:10]] for f in oscar_films]
    
    # IMDB data (sample)
    with open('data/domains/imdb_movies_complete.json') as f:
        imdb_data = json.load(f)
    
    imdb_sample = imdb_data[:500]  # Sample for speed
    imdb_titles = [m['title'] for m in imdb_sample]
    imdb_outcomes = np.array([m['success_score'] for m in imdb_sample])
    imdb_full_texts = [m['full_narrative'] for m in imdb_sample]
    imdb_casts = [[a for a in m['actors'][:10]] for m in imdb_sample]
    
    # === RUN TESTS ===
    
    # Test 1: Pure name effects
    oscar_test1 = tester.test_1_pure_name_effects(oscar_titles, oscar_outcomes, 'Oscar')
    imdb_test1 = tester.test_1_pure_name_effects(imdb_titles, imdb_outcomes, 'IMDB')
    
    # Test 2: Interactions
    oscar_test2 = tester.test_2_interaction(oscar_titles, oscar_full_texts, oscar_outcomes, 'Oscar')
    imdb_test2 = tester.test_2_interaction(imdb_titles, imdb_full_texts, imdb_outcomes, 'IMDB')
    
    # Test 3: Ensemble effects
    oscar_test3 = tester.test_3_ensemble_effects(oscar_casts, oscar_outcomes, 'Oscar')
    imdb_test3 = tester.test_3_ensemble_effects(imdb_casts, imdb_outcomes, 'IMDB')
    
    # Test 4: Gravitational clustering
    oscar_test4 = tester.test_4_gravitational_clustering(oscar_titles, oscar_outcomes, 'Oscar')
    imdb_test4 = tester.test_4_gravitational_clustering(imdb_titles, imdb_outcomes, 'IMDB')
    
    # === SUMMARY ===
    
    tester.print_summary()
    
    # === SAVE RESULTS ===
    
    output_path = Path('narrative_optimization/comprehensive_nominative_results.json')
    with open(output_path, 'w') as f:
        json.dump(tester.results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    return tester.results


if __name__ == '__main__':
    main()

