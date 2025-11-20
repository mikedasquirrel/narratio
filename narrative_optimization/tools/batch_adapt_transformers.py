"""
Batch Transformer Adaptation

Adapts multiple transformers at once to be domain-aware.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def adapt_transformer_imports():
    """Update imports in key files to include domain_config."""
    
    transformers_to_adapt = [
        'relational.py',
        'phonetic.py',
        'optics.py',
        'temporal_evolution.py',
        'information_theory.py',
        'social_status.py',
        'anticipatory_commitment.py',
        'namespace_ecology.py',
        'crossmodal.py',
        'quantitative.py',
        'audio.py',
        'crosslingual.py',
        'discoverability.py',
        'self_perception.py',
        'nominative.py',
        'linguistic_advanced.py'
    ]
    
    transformers_dir = Path(__file__).parent.parent / 'src' / 'transformers'
    
    print("="*80)
    print("BATCH ADAPTING TRANSFORMERS")
    print("="*80)
    
    adapted = 0
    skipped = 0
    
    for transformer_file in transformers_to_adapt:
        file_path = transformers_dir / transformer_file
        
        if not file_path.exists():
            print(f"  ⊙ {transformer_file}: Not found")
            skipped += 1
            continue
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if already adapted
            if 'domain_config=None' in content:
                print(f"  ⊙ {transformer_file}: Already adapted")
                skipped += 1
                continue
            
            # Simple adaptation: add parameter to first __init__ found
            import re
            
            # Find __init__ method
            init_pattern = r'def __init__\(self([^)]*)\):'
            match = re.search(init_pattern, content)
            
            if match:
                old_params = match.group(1)
                new_params = old_params + ', domain_config=None' if old_params else ', domain_config=None'
                
                new_init = f'def __init__(self{new_params}):'
                content = re.sub(init_pattern, new_init, content, count=1)
                
                # Add domain_config storage after super().__init__
                super_pattern = r'(super\(\).__init__\([^)]+\))'
                if re.search(super_pattern, content):
                    content = re.sub(
                        super_pattern,
                        r'\1\n        self.domain_config = domain_config',
                        content,
                        count=1
                    )
                
                # Write back
                with open(file_path, 'w') as f:
                    f.write(content)
                
                print(f"  ✓ {transformer_file}: Adapted")
                adapted += 1
            else:
                print(f"  ⚠ {transformer_file}: No __init__ found")
                skipped += 1
                
        except Exception as e:
            print(f"  ✗ {transformer_file}: Error - {e}")
            skipped += 1
    
    print(f"\n{'='*80}")
    print(f"BATCH ADAPTATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Adapted: {adapted}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(transformers_to_adapt)}")


if __name__ == '__main__':
    adapt_transformer_imports()

