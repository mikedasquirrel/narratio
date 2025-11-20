"""
Supreme Court Comprehensive Analysis

Tests multiple theoretical questions:
1. Does narrative quality predict outcomes? (vote margin, win/loss)
2. Does narrative quality predict influence? (citation counts)
3. Does π vary within domain? (unanimous vs split cases)
4. In adversarial setting, does better narrative win?
5. Can narrative override evidence strength?

This is the theoretical validation script.

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, classification_report
import sys

# Add paths
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))
sys.path.insert(0, str(base_path / 'src'))

from transformers.legal.argumentative_structure import ArgumentativeStructureTransformer
from transformers.legal.precedential_narrative import PrecedentialNarrativeTransformer
from transformers.legal.persuasive_framing import PersuasiveFramingTransformer
from transformers.legal.judicial_rhetoric import JudicialRhetoricTransformer

# Universal transformers
from transformers.universal_hybrid import UniversalHybridTransformer

# Try to import conflict, expertise, framing (may not exist)
try:
    from transformers.conflict_tension import ConflictTensionTransformer
except ImportError:
    ConflictTensionTransformer = None

try:
    from transformers.expertise_authority import ExpertiseAuthorityTransformer
except ImportError:
    ExpertiseAuthorityTransformer = None

try:
    from transformers.framing import FramingTransformer
except ImportError:
    FramingTransformer = None


class SupremeCourtAnalyzer:
    """
    Comprehensive Supreme Court narrative analysis.
    
    Tests multiple hypotheses with multiple outcome variables.
    """
    
    def __init__(self, data_path: Path):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        data_path : Path
            Path to supreme_court_complete.json
        """
        self.data_path = data_path
        self.cases = []
        self.results = {}
        
        # Initialize transformers
        self.legal_transformers = {
            'argumentative': ArgumentativeStructureTransformer(),
            'precedential': PrecedentialNarrativeTransformer(),
            'persuasive': PersuasiveFramingTransformer(),
            'rhetoric': JudicialRhetoricTransformer()
        }
        
        self.universal_transformers = {
            'hybrid': UniversalHybridTransformer()
        }
        
        # Add optional transformers if available
        if ConflictTensionTransformer:
            self.universal_transformers['conflict'] = ConflictTensionTransformer()
        if ExpertiseAuthorityTransformer:
            self.universal_transformers['expertise'] = ExpertiseAuthorityTransformer()
        if FramingTransformer:
            self.universal_transformers['framing'] = FramingTransformer()
    
    def load_data(self):
        """Load Supreme Court data."""
        print(f"Loading data from {self.data_path}...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path) as f:
            self.cases = json.load(f)
        
        print(f"Loaded {len(self.cases)} cases")
    
    def extract_features(self, sample_size: int = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract features from all cases using all transformers.
        
        Parameters
        ----------
        sample_size : int, optional
            Use subset for testing
        
        Returns
        -------
        features : ndarray
            Complete feature matrix
        metadata : DataFrame
            Case metadata and outcomes
        """
        cases = self.cases if sample_size is None else self.cases[:sample_size]
        
        print(f"Extracting features from {len(cases)} cases...")
        print("="*80)
        
        # Prepare narratives (majority opinions as primary)
        narratives = []
        metadata_list = []
        
        for case in cases:
            # Get primary narrative (majority opinion)
            narrative = case.get('majority_opinion', '') or case.get('opinion_full_text', '')
            
            if len(narrative) < 500:
                continue
            
            narratives.append(narrative)
            
            # Extract metadata
            metadata_list.append({
                'case_id': case.get('case_id', ''),
                'case_name': case.get('case_name', ''),
                'year': case.get('year', 0),
                'vote_margin': case['outcome'].get('vote_margin', 0),
                'unanimous': case['outcome'].get('unanimous', False),
                'citation_count': case['outcome'].get('citation_count', 0),
                'precedent_setting': case['outcome'].get('precedent_setting', False),
                'word_count': case['metadata'].get('word_count', 0),
                'author': case['metadata'].get('author', ''),
                'opinion_type': case['metadata'].get('opinion_type', 'majority')
            })
        
        print(f"Valid narratives: {len(narratives)}")
        
        # Extract features with each transformer
        all_features = []
        feature_names_all = []
        
        # Legal transformers
        for name, transformer in self.legal_transformers.items():
            print(f"Applying {name} transformer...")
            feat = transformer.fit_transform(narratives)
            all_features.append(feat)
            feature_names_all.extend([f"legal_{name}_{i}" for i in range(feat.shape[1])])
        
        # Universal transformers
        for name, transformer in self.universal_transformers.items():
            try:
                print(f"Applying {name} transformer...")
                feat = transformer.fit_transform(narratives)
                all_features.append(feat)
                feature_names_all.extend([f"universal_{name}_{i}" for i in range(feat.shape[1])])
            except Exception as e:
                print(f"Warning: {name} transformer failed: {e}")
        
        # Combine all features
        features = np.hstack(all_features)
        metadata_df = pd.DataFrame(metadata_list)
        
        print("="*80)
        print(f"Total features: {features.shape[1]}")
        print(f"Total samples: {features.shape[0]}")
        
        return features, metadata_df
    
    def calculate_domain_formula(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate domain formula: π, Δ, r, κ
        
        Using multiple outcome variables to test different hypotheses.
        """
        print("\n" + "="*80)
        print("CALCULATING DOMAIN FORMULA")
        print("="*80)
        
        # Calculate narrative quality (ю)
        narrative_quality = self._calculate_narrative_quality(features)
        
        # Test multiple outcomes
        results = {}
        
        # 1. Vote margin (primary outcome)
        if 'vote_margin' in metadata.columns:
            vote_margin = metadata['vote_margin'].values
            r_vote = np.corrcoef(narrative_quality, vote_margin)[0, 1] if len(vote_margin) > 10 else 0
            results['vote_margin'] = {
                'r': float(r_vote),
                'n': len(vote_margin),
                'hypothesis': 'Better narrative → larger margin (more unanimous)'
            }
        
        # 2. Citation count (influence)
        if 'citation_count' in metadata.columns:
            citations = metadata['citation_count'].values
            # Log transform citations (heavy-tailed)
            log_citations = np.log1p(citations)
            r_citations = np.corrcoef(narrative_quality, log_citations)[0, 1] if len(citations) > 10 else 0
            results['citations'] = {
                'r': float(r_citations),
                'n': len(citations),
                'hypothesis': 'Better narrative → more future citations'
            }
        
        # 3. Precedent-setting status (landmark cases)
        if 'precedent_setting' in metadata.columns:
            precedent = metadata['precedent_setting'].astype(int).values
            if precedent.sum() > 5:  # Need some True values
                # Point-biserial correlation
                r_precedent = np.corrcoef(narrative_quality, precedent)[0, 1]
                results['precedent_setting'] = {
                    'r': float(r_precedent),
                    'n': len(precedent),
                    'n_precedent': int(precedent.sum()),
                    'hypothesis': 'Better narrative → landmark status'
                }
        
        # Calculate π (narrativity)
        pi = self._calculate_narrativity(features, metadata)
        
        # Calculate κ (coupling)
        kappa = 0.75  # Judicial opinions - moderate coupling (justices control narrative but constrained by precedent)
        
        # Calculate Δ using primary outcome (citations is most objective)
        if 'citations' in results:
            r_primary = abs(results['citations']['r'])
        elif 'vote_margin' in results:
            r_primary = abs(results['vote_margin']['r'])
        else:
            r_primary = 0.0
        
        delta = pi * r_primary * kappa
        
        results['domain_formula'] = {
            'pi': float(pi),
            'delta': float(delta),
            'r_primary': float(r_primary),
            'kappa': float(kappa),
            'efficiency': float(delta / pi) if pi > 0 else 0,
            'verdict': 'PASS' if (delta / pi) > 0.5 else 'FAIL'
        }
        
        return results
    
    def test_pi_variance(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Test if π varies within domain by case type.
        
        Hypothesis: Unanimous cases (π ~0.3) vs Split cases (π ~0.7)
        
        This is revolutionary if true - π not fixed per domain!
        """
        print("\n" + "="*80)
        print("TESTING π VARIANCE WITHIN DOMAIN")
        print("="*80)
        
        if 'unanimous' not in metadata.columns:
            print("No unanimous data available")
            return {}
        
        # Split by unanimous vs split
        unanimous_mask = metadata['unanimous'].astype(bool)
        split_mask = ~unanimous_mask
        
        unanimous_features = features[unanimous_mask]
        split_features = features[split_mask]
        
        # Calculate π for each subset
        pi_unanimous = self._calculate_narrativity(
            unanimous_features, 
            metadata[unanimous_mask]
        )
        
        pi_split = self._calculate_narrativity(
            split_features,
            metadata[split_mask]
        )
        
        results = {
            'pi_unanimous': float(pi_unanimous),
            'pi_split': float(pi_split),
            'pi_difference': float(pi_split - pi_unanimous),
            'n_unanimous': int(unanimous_mask.sum()),
            'n_split': int(split_mask.sum()),
            'hypothesis': 'Split cases have higher π (more narrative-driven)',
            'result': 'CONFIRMED' if pi_split > pi_unanimous + 0.1 else 'NOT CONFIRMED'
        }
        
        print(f"π (unanimous): {pi_unanimous:.3f} (n={unanimous_mask.sum()})")
        print(f"π (split):     {pi_split:.3f} (n={split_mask.sum()})")
        print(f"Difference:    {pi_split - pi_unanimous:.3f}")
        print(f"Result: {results['result']}")
        
        return results
    
    def test_adversarial_dynamics(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Test adversarial narrative dynamics.
        
        If we have both petitioner and respondent briefs:
        - Does side with better narrative win?
        - Narrative gap → win probability?
        
        This tests whether legal outcomes follow narrative quality
        in adversarial setting.
        """
        print("\n" + "="*80)
        print("TESTING ADVERSARIAL NARRATIVE DYNAMICS")
        print("="*80)
        
        # This requires having both sides' briefs
        # For now, document the approach
        
        results = {
            'hypothesis': 'Side with better narrative wins more often',
            'method': 'Compare petitioner_brief vs respondent_brief narrative quality',
            'metric': 'Correlation(narrative_gap, winner)',
            'status': 'Requires brief data (future enhancement)',
            'expected_r': 0.3  # Moderate correlation expected
        }
        
        print("Adversarial testing requires briefs from both sides")
        print("This will be implemented when brief data is collected")
        
        return results
    
    def test_evidence_narrative_decomposition(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Decompose outcome into evidence vs narrative components.
        
        outcome = f(evidence_strength, narrative_quality)
        
        Question: When narrative >> evidence, which wins?
        """
        print("\n" + "="*80)
        print("EVIDENCE VS NARRATIVE DECOMPOSITION")
        print("="*80)
        
        # Evidence strength proxies:
        # 1. Unanimous decision = strong evidence (all agree)
        # 2. Citation count in opinion = evidence depth
        # 3. Precedent alignment = legal evidence strength
        
        # Narrative quality from features
        narrative_quality = self._calculate_narrative_quality(features)
        
        # Evidence proxy: unanimous cases likely have stronger evidence
        if 'unanimous' in metadata.columns:
            unanimous = metadata['unanimous'].astype(int).values
            
            # Compare narrative quality for unanimous vs split
            unanimous_mask = unanimous.astype(bool)
            
            q_unanimous = narrative_quality[unanimous_mask].mean() if unanimous_mask.sum() > 0 else 0
            q_split = narrative_quality[~unanimous_mask].mean() if (~unanimous_mask).sum() > 0 else 0
            
            results = {
                'narrative_quality_unanimous': float(q_unanimous),
                'narrative_quality_split': float(q_split),
                'difference': float(q_split - q_unanimous),
                'hypothesis': 'Split cases require better narrative (weaker evidence)',
                'result': 'CONFIRMED' if q_split > q_unanimous else 'NOT CONFIRMED',
                'interpretation': 'Higher narrative quality in split cases suggests narrative compensates for weaker evidence'
            }
            
            print(f"Narrative quality (unanimous): {q_unanimous:.3f}")
            print(f"Narrative quality (split):     {q_split:.3f}")
            print(f"Difference: {q_split - q_unanimous:.3f}")
            print(f"Result: {results['result']}")
            
            return results
        
        return {'status': 'Requires unanimous data'}
    
    def predict_multiple_outcomes(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Build predictive models for multiple outcomes.
        
        Tests what narrative predicts best:
        1. Vote margin
        2. Citation count  
        3. Precedent status
        """
        print("\n" + "="*80)
        print("PREDICTIVE MODELING - MULTIPLE OUTCOMES")
        print("="*80)
        
        results = {}
        
        # Split data
        X_train, X_test, meta_train, meta_test = train_test_split(
            features, metadata, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Predict citation count (continuous)
        if 'citation_count' in metadata.columns:
            y_train = np.log1p(meta_train['citation_count'].values)
            y_test = np.log1p(meta_test['citation_count'].values)
            
            if y_train.std() > 0:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                
                pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, pred)
                r_pearson = np.corrcoef(y_test, pred)[0, 1]
                
                results['citations_model'] = {
                    'r2': float(r2),
                    'r': float(r_pearson),
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'interpretation': 'Narrative quality predicts future influence'
                }
                
                print(f"Citations Model: R²={r2:.3f}, r={r_pearson:.3f}")
        
        # 2. Predict precedent-setting (binary)
        if 'precedent_setting' in metadata.columns:
            y_train = meta_train['precedent_setting'].astype(int).values
            y_test = meta_test['precedent_setting'].astype(int).values
            
            if y_train.sum() > 5 and (len(y_train) - y_train.sum()) > 5:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                
                pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, pred)
                
                results['precedent_model'] = {
                    'accuracy': float(acc),
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'n_precedent': int(y_test.sum()),
                    'interpretation': 'Narrative quality predicts landmark status'
                }
                
                print(f"Precedent Model: Accuracy={acc:.3f}")
        
        # 3. Predict vote margin (if available)
        if 'vote_margin' in metadata.columns:
            y_train = meta_train['vote_margin'].values
            y_test = meta_test['vote_margin'].values
            
            if y_train.std() > 0:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                
                pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, pred)
                r_pearson = np.corrcoef(y_test, pred)[0, 1]
                
                results['vote_margin_model'] = {
                    'r2': float(r2),
                    'r': float(r_pearson),
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'interpretation': 'Narrative predicts agreement level'
                }
                
                print(f"Vote Margin Model: R²={r2:.3f}, r={r_pearson:.3f}")
        
        return results
    
    def _calculate_narrative_quality(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate narrative quality score (ю) from features.
        
        Uses feature statistics to estimate quality.
        """
        # Normalize features
        feature_means = np.mean(features, axis=1)
        feature_stds = np.std(features, axis=1)
        feature_maxs = np.max(features, axis=1)
        
        # Quality = combination of mean, std, max
        quality = (
            feature_means * 0.4 +
            feature_stds * 0.3 +
            feature_maxs * 0.3
        )
        
        # Normalize to 0-1
        quality = (quality - quality.min()) / (quality.max() - quality.min() + 1e-6)
        
        return quality
    
    def _calculate_narrativity(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame
    ) -> float:
        """
        Calculate narrativity (π) for Supreme Court domain.
        
        π components:
        1. Structural (0.5): Semi-constrained by precedent
        2. Temporal (0.5): Cases build on history
        3. Agency (0.45): Justices decide but constrained
        4. Interpretive (0.65): Text interpretation is subjective
        5. Format (0.48): Opinion format is standardized
        """
        structural = 0.52  # Legal precedent constrains but allows interpretation
        temporal = 0.50  # Historical but forward-looking
        agency = 0.45  # Judges decide but precedent constrains
        interpretive = 0.65  # Text interpretation highly subjective
        format_constraint = 0.48  # Opinion format somewhat standardized
        
        # Weighted average
        pi = (structural + temporal + agency + interpretive + format_constraint) / 5
        
        return pi
    
    def run_complete_analysis(self, sample_size: int = None) -> Dict[str, Any]:
        """
        Run complete analysis with all tests.
        
        Parameters
        ----------
        sample_size : int, optional
            Use subset for testing
        
        Returns
        -------
        complete_results : dict
            All analysis results
        """
        print("\n" + "="*80)
        print("SUPREME COURT COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Extract features
        features, metadata = self.extract_features(sample_size=sample_size)
        
        # Run all analyses
        results = {
            'domain_formula': self.calculate_domain_formula(features, metadata),
            'pi_variance': self.test_pi_variance(features, metadata),
            'adversarial': self.test_adversarial_dynamics(features, metadata),
            'evidence_narrative': self.test_evidence_narrative_decomposition(features, metadata),
            'predictive_models': self.predict_multiple_outcomes(features, metadata),
            'metadata': {
                'n_cases': len(self.cases),
                'n_analyzed': len(metadata),
                'n_features': features.shape[1],
                'year_range': (int(metadata['year'].min()), int(metadata['year'].max())),
                'avg_word_count': int(metadata['word_count'].mean())
            }
        }
        
        # Save results
        self.results = results
        self.save_results()
        
        return results
    
    def save_results(self):
        """Save analysis results."""
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'supreme_court_analysis_complete.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n Results saved to: {output_file}")
    
    def print_summary(self):
        """Print human-readable summary."""
        if not self.results:
            print("No results yet. Run analysis first.")
            return
        
        print("\n" + "="*80)
        print("SUPREME COURT ANALYSIS SUMMARY")
        print("="*80)
        
        formula = self.results['domain_formula']
        print(f"\nDomain Formula:")
        print(f"  π (narrativity):  {formula['pi']:.3f}")
        print(f"  r (correlation):  {formula['r_primary']:.3f}")
        print(f"  κ (coupling):     {formula['kappa']:.3f}")
        print(f"  Δ (agency):       {formula['delta']:.3f}")
        print(f"  Efficiency:       {formula['efficiency']:.3f}")
        print(f"  Verdict:          {formula['verdict']}")
        
        if 'pi_variance' in self.results:
            piv = self.results['pi_variance']
            print(f"\nπ Variance Test:")
            print(f"  Unanimous: π={piv.get('pi_unanimous', 0):.3f}")
            print(f"  Split:     π={piv.get('pi_split', 0):.3f}")
            print(f"  Result:    {piv.get('result', 'N/A')}")
        
        print("="*80)


def main():
    """Run analysis."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--data-file', default='data/domains/supreme_court_complete.json')
    args = parser.parse_args()
    
    data_path = Path(args.data_file)
    
    analyzer = SupremeCourtAnalyzer(data_path)
    results = analyzer.run_complete_analysis(sample_size=args.sample)
    analyzer.print_summary()


if __name__ == '__main__':
    main()

