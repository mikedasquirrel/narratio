"""
Quick Fix Script for Common Transformer Input Shape Errors

Identifies and suggests fixes for the 11 transformers with input shape errors.

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

# Transformers with input shape errors
PROBLEMATIC_TRANSFORMERS = [
    "nominative.py",
    "self_perception.py",
    "narrative_potential.py",
    "linguistic_advanced.py",
    "conflict_tension.py",
    "framing.py",
    "expertise_authority.py",
    "cultural_context.py",
    "anticipatory_commitment.py",
    "social_status.py",
]

def analyze_transformer(filepath: Path) -> dict:
    """Analyze a transformer file for common issues"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for .reshape(-1, 1) on single samples
    if re.search(r'\.reshape\(-1,\s*1\)', content):
        issues.append({
            'type': 'reshape',
            'severity': 'high',
            'description': 'Uses reshape(-1, 1) which may cause issues with single samples',
            'fix': 'Check if reshaping a scalar or 1-element array'
        })
    
    # Check for cross_val_score or similar sklearn validation
    if 'cross_val_score' in content or 'cross_validate' in content:
        issues.append({
            'type': 'cross_validation',
            'severity': 'critical',
            'description': 'Uses cross-validation which requires multiple samples',
            'fix': 'Ensure X has shape (n_samples, n_features) before cross-validation'
        })
    
    # Check for sklearn estimator.fit() in transform()
    if 'def transform(' in content:
        transform_section = content.split('def transform(')[1].split('def ')[0]
        if '.fit(' in transform_section and 'is_fitted_' not in transform_section:
            issues.append({
                'type': 'fit_in_transform',
                'severity': 'high',
                'description': 'Calls .fit() inside transform() method',
                'fix': 'Move fitting to fit() method only'
            })
    
    # Check for hardcoded array indexing
    if re.search(r'\[\d+\](?!\s*=)', content):
        issues.append({
            'type': 'indexing',
            'severity': 'medium',
            'description': 'Uses hardcoded array indexing which may fail on unexpected shapes',
            'fix': 'Add shape validation and bounds checking'
        })
    
    return {
        'file': filepath.name,
        'path': str(filepath),
        'issues': issues,
        'lines': content.count('\n') + 1
    }


def main():
    print("\n" + "="*80)
    print("TRANSFORMER INPUT SHAPE ERROR ANALYSIS")
    print("="*80)
    
    transformers_dir = Path('narrative_optimization/src/transformers')
    
    all_results = []
    
    for filename in PROBLEMATIC_TRANSFORMERS:
        filepath = transformers_dir / filename
        
        if not filepath.exists():
            print(f"\n✗ {filename} not found at {filepath}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Analyzing: {filename}")
        print(f"{'='*80}")
        
        result = analyze_transformer(filepath)
        all_results.append(result)
        
        if result['issues']:
            print(f"Found {len(result['issues'])} potential issues:\n")
            for i, issue in enumerate(result['issues'], 1):
                severity_color = {
                    'critical': '🔴',
                    'high': '🟠', 
                    'medium': '🟡',
                    'low': '🟢'
                }
                print(f"  {severity_color.get(issue['severity'], '⚪')} Issue {i}: {issue['type'].upper()}")
                print(f"     Severity: {issue['severity']}")
                print(f"     Description: {issue['description']}")
                print(f"     Fix: {issue['fix']}")
                print()
        else:
            print("✓ No obvious issues detected")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_issues = sum(len(r['issues']) for r in all_results)
    critical_count = sum(1 for r in all_results for i in r['issues'] if i['severity'] == 'critical')
    high_count = sum(1 for r in all_results for i in r['issues'] if i['severity'] == 'high')
    
    print(f"\nFiles analyzed: {len(all_results)}")
    print(f"Total issues found: {total_issues}")
    print(f"  - Critical: {critical_count}")
    print(f"  - High: {high_count}")
    
    print("\n" + "="*80)
    print("RECOMMENDED FIXES")
    print("="*80)
    
    print("""
The most common issue is shape handling in fit_transform/transform methods.

Common Pattern That Fails:
```python
def fit_transform(self, X, y=None):
    # This fails if X is a single string or has shape (1,)
    X = np.array([X]) if isinstance(X, str) else X
    
    # This causes "inconsistent samples" error
    clf = LogisticRegression()
    scores = cross_val_score(clf, X.reshape(-1, 1), y)
    # When X has 1 element, reshape makes (1, 1)
    # But cross_val_score tries to split, causing error
```

Recommended Fix:
```python
def fit_transform(self, X, y=None):
    # Validate input shape
    if isinstance(X, str):
        X = pd.Series([X])
    elif not isinstance(X, (list, pd.Series, np.ndarray)):
        X = pd.Series([X])
    
    # Ensure Series for consistent handling
    if isinstance(X, list):
        X = pd.Series(X)
    
    # Check minimum samples for cross-validation
    min_samples = 5  # or cv folds + 1
    if len(X) < min_samples:
        # Use simple validation or skip CV entirely
        features = self._extract_features_without_cv(X)
    else:
        # Normal CV-based feature extraction
        features = self._extract_features_with_cv(X, y)
    
    return features

def _extract_features_without_cv(self, X):
    # Extract features directly without cross-validation
    pass
    
def _extract_features_with_cv(self, X, y):
    # Extract features using cross-validation
    pass
```

Key Principles:
1. **Validate input format** - ensure X is consistently a Series or array
2. **Check sample size** - don't use CV with <5 samples
3. **Add fallback logic** - have non-CV feature extraction path
4. **Test edge cases** - test with 1, 2, 5, and 1000 samples
""")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. For each problematic transformer:
   a. Add input validation at start of fit() and fit_transform()
   b. Add minimum sample checks before cross-validation
   c. Implement fallback feature extraction for small samples
   
2. Test each fix with:
   python -c "from transformers.TRANSFORMER import TRANSFORMER; t = TRANSFORMER(); t.fit_transform(['test'], [1])"
   
3. Run full performance analysis again:
   python analyze_transformer_performance_simple.py
   
4. Update documentation with minimum sample requirements
""")


if __name__ == "__main__":
    main()

