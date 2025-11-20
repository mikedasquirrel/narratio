"""
Mass Transformer Fix - Add Input Validation to All Transformers

Adds ensure_string_list() calls to all transformers that don't already have them.
"""

import re
from pathlib import Path

# Transformers to fix
transformer_files = [
    'narrative_potential.py',
    'relational.py',
    'emotional_resonance.py',
    'authenticity.py',
    'suspense_mystery.py',
    'visual_multimodal.py',
    'phonetic.py',
    'temporal_evolution.py',
    'information_theory.py',
    'social_status.py',
    'namespace_ecology.py',
    'anticipatory_commitment.py',
    'quantitative.py',
    'crossmodal.py',
    'audio.py',
    'crosslingual.py',
    'discoverability.py',
    'cognitive_fluency.py',
    'framing.py',
    'optics.py',
]

transformers_dir = Path(__file__).parent.parent / 'src' / 'transformers'

import_line = "from .utils.input_validation import ensure_string_list, ensure_string\n"

fit_check = """        # Ensure X is list of strings
        X = ensure_string_list(X)
        """

transform_check = """        # Ensure X is list of strings
        X = ensure_string_list(X)
        """

print("=" * 80)
print("MASS TRANSFORMER FIX - ADD INPUT VALIDATION")
print("=" * 80)

for filename in transformer_files:
    filepath = transformers_dir / filename
    
    if not filepath.exists():
        print(f"⚠ {filename} - Not found, skipping")
        continue
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        modified = False
        
        # Add import if not present
        if 'ensure_string_list' not in content:
            # Find the imports section (after docstring, before class)
            # Insert after the last import
            import_section_end = 0
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_section_end = i
            
            if import_section_end > 0:
                lines.insert(import_section_end + 1, "from .utils.input_validation import ensure_string_list, ensure_string")
                content = '\n'.join(lines)
                modified = True
        
        # Add validation to fit() if not present
        if 'def fit(self, X' in content and 'ensure_string_list(X)' not in content:
            # Find fit method and add validation
            content = re.sub(
                r'(def fit\(self, X[^)]*\):.*?\n\s+"""[^"]*""")\n',
                r'\1\n        # Ensure X is list of strings\n        X = ensure_string_list(X)\n        \n',
                content,
                flags=re.DOTALL
            )
            modified = True
        
        # Add validation to transform() if not present
        if 'def transform(self, X' in content and 'transform' in content:
            # Check if validation already exists
            if content.count('ensure_string_list(X)') < 2:  # Not in both methods
                content = re.sub(
                    r'(def transform\(self, X[^)]*\):.*?\n\s+"""[^"]*""")\n(\s+)(self\._validate_fitted\(\))',
                    r'\1\n\2\3\n\2\n\2# Ensure X is list of strings\n\2X = ensure_string_list(X)\n\2',
                    content,
                    flags=re.DOTALL
                )
                modified = True
        
        if modified:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✓ {filename} - Fixed")
        else:
            print(f"  {filename} - Already OK or no changes needed")
            
    except Exception as e:
        print(f"✗ {filename} - Error: {e}")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)

