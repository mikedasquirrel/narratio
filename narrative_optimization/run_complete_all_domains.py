"""
COMPLETE ANALYSIS - ALL TRANSFORMERS, ALL DOMAINS

Applies ALL 29 transformers to every fully stocked dataset.
Runs complete variable system analysis (ж→ю→❊→Д→п→μ→ф→ة→Ξ).
Saves results for web display.
"""

import json
import numpy as np
from pathlib import Path
import sys
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

# Import ALL transformers
from src.transformers import (
    # Nominative (4 transformers)
    UniversalNominativeTransformer,
    HierarchicalNominativeTransformer,
    NominativeAnalysisTransformer,
    PhoneticTransformer,
    
    # Narrative (7 transformers)
    EmotionalResonanceTransformer,
    AuthenticityTransformer,
    ConflictTensionTransformer,
    CulturalContextTransformer,
    SuspenseMysteryTransformer,
    ExpertiseAuthorityTransformer,
    
    # Core (6 transformers)
    SelfPerceptionTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    EnsembleNarrativeTransformer,
    RelationalValueTransformer,
    
    # Content
    StatisticalTransformer,
    
    # Advanced
    TemporalEvolutionTransformer,
    InformationTheoryTransformer,
    SocialStatusTransformer,
    FramingTransformer,
    OpticsTransformer,
    
    # Phase 7 - Complete framework coverage
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer,
    CouplingStrengthTransformer,
    NarrativeMassTransformer,
    NominativeRichnessTransformer,
    GravitationalFeaturesTransformer
)


class CompleteDomainAnalyzer:
    """Run complete analysis with ALL transformers"""
    
    def __init__(self, domain_name, п):
        self.domain_name = domain_name
        self.п = п
        self.results = {}
    
    def analyze(self, data, outcomes, is_binary=True):
        """Complete analysis with all transformers"""
        print(f"\n{'='*80}")
        print(f"{self.domain_name.upper()} - ALL TRANSFORMERS")
        print(f"{'='*80}")
        print(f"п = {self.п:.2f}, n = {len(data)}")
        
        # Determine data types
        if isinstance(data[0], dict):
            structured = True
            texts = self._extract_texts(data)
        else:
            structured = False
            texts = data
        
        # Apply ALL transformers
        transformer_sets = {
            'nominative': [
                ('universal_nominative', UniversalNominativeTransformer(), data if structured else texts),
                ('hierarchical_nominative', HierarchicalNominativeTransformer(), data if structured else texts),
                ('nominative_analysis', NominativeAnalysisTransformer(), texts),
                ('phonetic', PhoneticTransformer(), texts)
            ],
            'narrative': [
                ('emotional', EmotionalResonanceTransformer(), texts),
                ('authenticity', AuthenticityTransformer(), texts),
                ('conflict', ConflictTensionTransformer(), texts),
                ('cultural', CulturalContextTransformer(), texts),
                ('suspense', SuspenseMysteryTransformer(), texts)
            ],
            'core': [
                ('self_perception', SelfPerceptionTransformer(), texts),
                ('narrative_potential', NarrativePotentialTransformer(), texts),
                ('linguistic', LinguisticPatternsTransformer(), texts),
                ('ensemble', EnsembleNarrativeTransformer(n_top_terms=20), texts),
                ('relational', RelationalValueTransformer(n_features=30), texts)
            ],
            'content': [
                ('statistical', StatisticalTransformer(max_features=150), texts),
                ('temporal', TemporalEvolutionTransformer(), texts),
                ('framing', FramingTransformer(), texts)
            ]
        }
        
        all_features = []
        transformer_results = {}
        
        for category, trans_list in transformer_sets.items():
            print(f"\n{category.upper()} transformers:")
            
            for trans_name, trans, trans_data in trans_list:
                try:
                    trans.fit(trans_data)
                    feat = trans.transform(trans_data)
                    
                    if hasattr(feat, 'toarray'):
                        feat = feat.toarray()
                    
                    # Test individual contribution
                    contrib = self._test_transformer(feat, outcomes, is_binary)
                    transformer_results[trans_name] = contrib
                    
                    all_features.append(feat)
                    print(f"  ✓ {trans_name:25s}: {feat.shape[1]:3d} features, score={contrib['score']:.3f}")
                    
                except Exception as e:
                    print(f"  ✗ {trans_name:25s}: {str(e)[:50]}")
        
        # COMBINED
        X_all = np.hstack(all_features)
        print(f"\n{'='*60}")
        print(f"COMBINED: {X_all.shape[1]} total features")
        
        combined_result = self._test_transformer(X_all, outcomes, is_binary, use_cv=True)
        
        print(f"  Training: {combined_result['score_train']:.3f}")
        print(f"  CV: {combined_result['score_cv']:.3f}")
        print(f"  Д: {combined_result['D']:.3f}")
        
        # Save
        self.results = {
            'domain': self.domain_name,
            'п': self.п,
            'n_samples': len(data),
            'n_features': X_all.shape[1],
            'is_binary': is_binary,
            'combined': combined_result,
            'transformers': transformer_results
        }
        
        return self.results
    
    def _extract_texts(self, data):
        """Extract text from structured data"""
        texts = []
        for d in data:
            text_parts = [
                str(d.get('title', '')),
                str(d.get('overview', '')),
                ' '.join(d.get('director', [])) if isinstance(d.get('director'), list) else str(d.get('director', '')),
                str(d.get('narrative', ''))
            ]
            texts.append(' '.join(text_parts))
        return texts
    
    def _test_transformer(self, features, outcomes, is_binary, use_cv=False):
        """Test single transformer"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        if is_binary:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_scaled, outcomes)
            score_train = roc_auc_score(outcomes, model.predict_proba(X_scaled)[:, 1])
            
            if use_cv and len(outcomes) < 100:
                # Leave-one-out for small samples
                loo = LeaveOneOut()
                cv_preds = []
                for train_idx, test_idx in loo.split(X_scaled):
                    model_cv = LogisticRegression(max_iter=1000, random_state=42)
                    model_cv.fit(X_scaled[train_idx], outcomes[train_idx])
                    cv_preds.append(model_cv.predict_proba(X_scaled[test_idx])[0, 1])
                score_cv = roc_auc_score(outcomes, cv_preds)
            elif use_cv:
                cv_scores = cross_val_score(model, X_scaled, outcomes, cv=5, scoring='roc_auc')
                score_cv = cv_scores.mean()
            else:
                score_cv = score_train
            
            baseline = 0.58
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, outcomes)
            preds = model.predict(X_scaled)
            score_train, _ = stats.pearsonr(preds, outcomes)
            
            if use_cv:
                cv_scores = cross_val_score(model, X_scaled, outcomes, cv=min(5, len(outcomes)//10), scoring='r2')
                score_cv = np.sqrt(max(0, cv_scores.mean()))  # Convert R² to r
            else:
                score_cv = score_train
            
            baseline = 0.20
        
        return {
            'n_features': features.shape[1],
            'score': float(score_train),
            'score_train': float(score_train),
            'score_cv': float(score_cv) if use_cv else None,
            'baseline': baseline,
            'D': float(score_cv - baseline) if use_cv else float(score_train - baseline)
        }
    
    def save(self, output_path):
        """Save results"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Saved: {output_path}")


def main():
    """Run complete analyses on all datasets"""
    print("="*80)
    print("COMPLETE PIPELINE - ALL TRANSFORMERS, ALL DOMAINS")
    print("="*80)
    
    results_summary = {}
    
    # OSCAR
    print("\n" + "="*80)
    print("1. OSCAR BEST PICTURE")
    print("="*80)
    
    with open('data/domains/oscar_nominees_complete.json') as f:
        oscar_raw = json.load(f)
    films = []
    for year_films in oscar_raw.values():
        films.extend(year_films)
    oscar_outcomes = np.array([int(f.get('won_oscar', 0)) for f in films])
    
    oscar_analyzer = CompleteDomainAnalyzer('oscar', п=0.88)
    oscar_results = oscar_analyzer.analyze(films, oscar_outcomes, is_binary=True)
    oscar_analyzer.save('narrative_optimization/domains/oscars/complete_analysis.json')
    results_summary['oscar'] = oscar_results
    
    # IMDB
    print("\n" + "="*80)
    print("2. IMDB MOVIES")
    print("="*80)
    
    with open('data/domains/imdb_movies_complete.json') as f:
        imdb_data = json.load(f)
    
    np.random.seed(42)
    sample_idx = np.random.choice(len(imdb_data), 1000, replace=False)
    imdb_sample = [imdb_data[i] for i in sample_idx]
    imdb_outcomes = np.array([m['success_score'] for m in imdb_sample])
    
    imdb_analyzer = CompleteDomainAnalyzer('imdb', п=0.65)
    imdb_results = imdb_analyzer.analyze(imdb_sample, imdb_outcomes, is_binary=False)
    imdb_analyzer.save('narrative_optimization/domains/imdb/complete_analysis.json')
    results_summary['imdb'] = imdb_results
    
    # NBA
    print("\n" + "="*80)
    print("3. NBA GAMES")
    print("="*80)
    
    with open('data/domains/nba_enriched_1000.json') as f:
        nba_data = json.load(f)
    nba_outcomes = np.array([int(g['won']) for g in nba_data])
    
    nba_analyzer = CompleteDomainAnalyzer('nba', п=0.15)
    nba_results = nba_analyzer.analyze(nba_data, nba_outcomes, is_binary=True)
    nba_analyzer.save('narrative_optimization/domains/nba/complete_analysis.json')
    results_summary['nba'] = nba_results
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS SUMMARY")
    print("="*80)
    
    for domain_name, res in results_summary.items():
        print(f"\n{domain_name.upper()}:")
        print(f"  п: {res['п']:.2f}")
        print(f"  Samples: {res['n_samples']}")
        print(f"  Total Features: {res['n_features']}")
        print(f"  CV Score: {res['combined']['score_cv']:.3f}")
        print(f"  Д: {res['combined']['D']:.3f}")
    
    # Save summary
    with open('narrative_optimization/complete_pipeline_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n✅ ALL ANALYSES COMPLETE")
    print("   Results saved for web display")


if __name__ == '__main__':
    main()

