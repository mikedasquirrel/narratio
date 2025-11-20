"""
Real Data Validation System

Loads actual domain data files and validates the new architecture
on real datasets with comprehensive error handling.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import traceback
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer


class RealDataValidator:
    """
    Validates domain-specific architecture on real data files.
    
    Handles:
    - Multiple data formats
    - Missing fields
    - Encoding issues
    - Edge cases (empty sets, single samples)
    - Data quality checks
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {}
        
    def find_domain_data(self, domain_name: str) -> List[Path]:
        """
        Find all possible data files for a domain.
        
        Returns
        -------
        list of Path
            Potential data file paths
        """
        possible_paths = []
        
        # Check data/domains/
        data_dir = self.project_root / 'data' / 'domains'
        possible_paths.extend(data_dir.glob(f'{domain_name}*.json'))
        possible_paths.extend(data_dir.glob(f'{domain_name}*.csv'))
        
        # Check narrative_optimization/domains/
        domain_dir = self.project_root / 'narrative_optimization' / 'domains' / domain_name
        if domain_dir.exists():
            possible_paths.extend(domain_dir.glob('*.json'))
            possible_paths.extend(domain_dir.glob('*.csv'))
            possible_paths.extend(domain_dir.glob('*results*.json'))
            possible_paths.extend(domain_dir.glob('*data*.json'))
        
        return [p for p in possible_paths if p.exists()]
    
    def load_domain_data(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """
        Load domain data, trying multiple paths and formats.
        
        Returns
        -------
        dict or None
            Data with texts, outcomes, and metadata
        """
        possible_files = self.find_domain_data(domain_name)
        
        if not possible_files:
            print(f"  ✗ No data files found for {domain_name}")
            return None
        
        print(f"  Found {len(possible_files)} potential data files")
        
        for data_file in possible_files:
            try:
                print(f"    Trying: {data_file.name}")
                data = self._load_file(data_file)
                
                if data and len(data.get('texts', [])) > 0:
                    print(f"    ✓ Successfully loaded {len(data['texts'])} samples")
                    return data
                    
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue
        
        return None
    
    def _load_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single file, handling various formats."""
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            # Try different structures
            if isinstance(data, dict):
                # Check for narratives/texts
                texts = data.get('narratives', data.get('texts', data.get('stories', [])))
                
                if not texts and 'results' in data:
                    # Might be list of results
                    if isinstance(data['results'], list) and len(data['results']) > 0:
                        first = data['results'][0]
                        if 'narrative' in first or 'text' in first:
                            texts = [r.get('narrative', r.get('text', '')) for r in data['results']]
                
                if not texts:
                    return None
                
                outcomes = data.get('outcomes', data.get('results', data.get('y', [])))
                
                # Convert outcomes to array
                if isinstance(outcomes, list):
                    if len(outcomes) > 0 and isinstance(outcomes[0], dict):
                        # List of dicts - extract outcome field
                        outcomes = [o.get('outcome', o.get('result', o.get('y', 0))) for o in outcomes]
                    outcomes = np.array(outcomes)
                else:
                    outcomes = np.array([outcomes] * len(texts))
                
                return {
                    'texts': texts,
                    'outcomes': outcomes,
                    'names': data.get('names', None),
                    'timestamps': data.get('timestamps', data.get('dates', None))
                }
            
            elif isinstance(data, list):
                # List of records
                texts = []
                outcomes = []
                
                for item in data:
                    if isinstance(item, dict):
                        text = item.get('narrative', item.get('text', item.get('description', str(item))))
                        outcome = item.get('outcome', item.get('result', item.get('y', 0)))
                    else:
                        text = str(item)
                        outcome = 0
                    
                    texts.append(text)
                    outcomes.append(outcome)
                
                return {
                    'texts': texts,
                    'outcomes': np.array(outcomes),
                    'names': None,
                    'timestamps': None
                }
        
        elif file_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            
            # Find text column
            text_col = None
            for col in ['narrative', 'text', 'description', 'story', 'content']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                return None
            
            texts = df[text_col].fillna('').astype(str).tolist()
            
            # Find outcome column
            outcome_col = None
            for col in ['outcome', 'result', 'y', 'target', 'label']:
                if col in df.columns:
                    outcome_col = col
                    break
            
            if outcome_col:
                outcomes = df[outcome_col].values
            else:
                outcomes = np.zeros(len(df))
            
            return {
                'texts': texts,
                'outcomes': outcomes,
                'names': df.get('name', None),
                'timestamps': df.get('timestamp', df.get('date', None))
            }
        
        return None
    
    def validate_domain(
        self,
        domain_name: str,
        expected_r2: Optional[float] = None,
        min_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Validate a single domain on real data.
        
        Parameters
        ----------
        domain_name : str
            Domain to validate
        expected_r2 : float, optional
            Expected R² for comparison
        min_samples : int
            Minimum samples required
        
        Returns
        -------
        dict
            Validation results
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING: {domain_name.upper()}")
        print(f"{'='*80}")
        
        result = {
            'domain': domain_name,
            'status': 'unknown',
            'error': None,
            'r_squared': None,
            'delta': None,
            'samples': 0,
            'data_file': None
        }
        
        try:
            # Load data
            print(f"\n[1/5] Loading data...")
            data = self.load_domain_data(domain_name)
            
            if data is None:
                result['status'] = 'no_data'
                result['error'] = 'No data files found'
                return result
            
            texts = data['texts']
            outcomes = data['outcomes']
            
            # Filter empty texts
            valid_indices = [i for i, t in enumerate(texts) if t and len(str(t).strip()) > 0]
            texts = [texts[i] for i in valid_indices]
            outcomes = outcomes[valid_indices]
            
            if len(texts) < min_samples:
                result['status'] = 'insufficient_data'
                result['error'] = f'Only {len(texts)} samples (need {min_samples})'
                result['samples'] = len(texts)
                return result
            
            result['samples'] = len(texts)
            print(f"  ✓ {len(texts)} valid samples")
            
            # Check outcomes
            print(f"\n[2/5] Validating outcomes...")
            if len(np.unique(outcomes)) < 2:
                result['status'] = 'no_variance'
                result['error'] = 'Outcomes have no variance'
                return result
            
            print(f"  ✓ Outcomes: {len(np.unique(outcomes))} unique values")
            print(f"    Range: [{outcomes.min():.3f}, {outcomes.max():.3f}]")
            
            # Create analyzer
            print(f"\n[3/5] Creating analyzer...")
            analyzer = DomainSpecificAnalyzer(domain_name)
            print(f"  ✓ Analyzer created (π={analyzer.narrativity:.3f})")
            
            # Run analysis
            print(f"\n[4/5] Running analysis...")
            analysis_results = analyzer.analyze_complete(
                texts=texts,
                outcomes=outcomes,
                names=data.get('names'),
                timestamps=data.get('timestamps')
            )
            
            result['r_squared'] = analysis_results['r_squared']
            result['delta'] = analysis_results['delta']
            result['efficiency'] = analysis_results['efficiency']
            result['passes_threshold'] = analysis_results['passes_threshold']
            
            print(f"  ✓ Analysis complete")
            print(f"    R²: {result['r_squared']:.1%}")
            print(f"    Д: {result['delta']:.4f}")
            print(f"    Efficiency: {result['efficiency']:.4f}")
            
            # Compare to expected
            print(f"\n[5/5] Validation...")
            if expected_r2 is not None:
                r2_diff = result['r_squared'] - expected_r2
                r2_pct_diff = (r2_diff / expected_r2) * 100 if expected_r2 > 0 else 0
                
                print(f"  Expected R²: {expected_r2:.1%}")
                print(f"  Achieved R²: {result['r_squared']:.1%}")
                print(f"  Difference: {r2_pct_diff:+.1f}%")
                
                # Within 10% is acceptable
                if abs(r2_pct_diff) < 10:
                    result['status'] = 'validated'
                    print(f"  ✓ VALIDATED (within 10% of expected)")
                else:
                    result['status'] = 'deviated'
                    result['error'] = f'R² deviated by {abs(r2_pct_diff):.1f}%'
                    print(f"  ⚠ Deviated from expected")
            else:
                result['status'] = 'completed'
                print(f"  ✓ Completed (no expected value)")
            
            # Check Д sign for high performers
            if domain_name in ['golf', 'tennis'] and result['delta'] < 0:
                result['status'] = 'negative_delta'
                result['error'] = 'Д is negative (should be positive for high performers)'
                print(f"  ⚠ Д is negative (investigate)")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"\n✗ ERROR: {e}")
            traceback.print_exc()
        
        return result
    
    def validate_all_priority_domains(self) -> Dict[str, Dict]:
        """
        Validate all priority domains.
        
        Returns
        -------
        dict
            Results for each domain
        """
        priority_domains = {
            'golf': 0.977,
            'tennis': 0.931,
            'boxing': 0.004,
            'nba': 0.15,
            'wwe': 0.743
        }
        
        print("="*80)
        print("REAL DATA VALIDATION - ALL PRIORITY DOMAINS")
        print("="*80)
        
        results = {}
        
        for domain, expected_r2 in priority_domains.items():
            result = self.validate_domain(domain, expected_r2)
            results[domain] = result
        
        # Summary
        print(f"\n\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}\n")
        
        for domain, result in results.items():
            status_icon = {
                'validated': '✓',
                'completed': '✓',
                'deviated': '⚠',
                'negative_delta': '⚠',
                'no_data': '✗',
                'insufficient_data': '✗',
                'no_variance': '✗',
                'error': '✗'
            }.get(result['status'], '?')
            
            status_text = result['status'].replace('_', ' ').title()
            
            if result['r_squared'] is not None:
                print(f"  {status_icon} {domain.upper():15s}: {status_text:20s} "
                      f"R²={result['r_squared']:.1%}, Д={result['delta']:.4f}, "
                      f"n={result['samples']}")
            else:
                print(f"  {status_icon} {domain.upper():15s}: {status_text:20s} "
                      f"({result.get('error', 'Unknown')})")
        
        # Save results
        results_file = self.project_root / 'narrative_optimization' / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'validation_date': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")
        
        return results


if __name__ == '__main__':
    validator = RealDataValidator()
    validator.validate_all_priority_domains()

