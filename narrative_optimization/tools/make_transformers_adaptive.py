"""
Transformer Adaptation Tool

Batch converts transformers to domain-adaptive versions.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
from typing import List, Dict
import re

sys.path.insert(0, str(Path(__file__).parent.parent))


class TransformerAdapter:
    """
    Adapts existing transformers to accept domain_config.
    
    Changes:
    1. Add domain_config parameter to __init__
    2. Load domain-specific patterns from config
    3. Merge with base patterns
    """
    
    def __init__(self):
        self.transformers_dir = Path(__file__).parent.parent / 'src' / 'transformers'
        self.adapted_count = 0
        
    def adapt_all_transformers(self):
        """Adapt all transformers in directory."""
        print("="*80)
        print("MAKING ALL TRANSFORMERS DOMAIN-ADAPTIVE")
        print("="*80)
        
        # Find all transformer files
        transformer_files = list(self.transformers_dir.glob('*.py'))
        
        # Exclude special files
        exclude = ['__init__.py', 'domain_archetype.py', 'domain_adaptive_base.py']
        transformer_files = [f for f in transformer_files if f.name not in exclude]
        
        print(f"\nFound {len(transformer_files)} transformer files")
        
        for transformer_file in transformer_files:
            print(f"\nProcessing {transformer_file.name}...")
            
            try:
                self._adapt_transformer_file(transformer_file)
                self.adapted_count += 1
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        print(f"\n{'='*80}")
        print(f"ADAPTATION COMPLETE: {self.adapted_count}/{len(transformer_files)} transformers")
        print(f"{'='*80}")
    
    def _adapt_transformer_file(self, file_path: Path):
        """Adapt a single transformer file."""
        # Read file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if already adapted
        if 'domain_config' in content and 'DomainConfig' in content:
            print(f"  ⊙ Already adaptive")
            return
        
        # Find class definition
        class_match = re.search(r'class (\w+)\(NarrativeTransformer\):', content)
        
        if not class_match:
            print(f"  ⊙ Not a NarrativeTransformer subclass")
            return
        
        class_name = class_match.group(1)
        
        # Find __init__ method
        init_match = re.search(
            r'def __init__\(self[^)]*\):',
            content
        )
        
        if not init_match:
            print(f"  ⊙ No __init__ method found")
            return
        
        # Add domain_config parameter
        init_signature = init_match.group(0)
        
        # Check if already has domain_config
        if 'domain_config' in init_signature:
            print(f"  ⊙ Already has domain_config parameter")
            return
        
        # Add parameter
        new_init = init_signature.replace(
            '):'',
            ', domain_config=None):'
        )
        
        content = content.replace(init_signature, new_init)
        
        # Add import if not present
        if 'from ..config import DomainConfig' not in content and 'from ...config import DomainConfig' not in content:
            # Add after existing imports
            import_section_end = content.find('\n\n')
            if import_section_end > 0:
                content = (content[:import_section_end] +
                          '\nfrom ..config import DomainConfig' +
                          content[import_section_end:])
        
        # Add domain config storage in __init__
        # Find super().__init__ call
        super_match = re.search(r'super\(\).__init__\([^)]+\)', content)
        
        if super_match:
            insert_pos = super_match.end()
            # Add domain config storage
            addition = '\n        self.domain_config = domain_config\n        self._load_domain_patterns()'
            content = content[:insert_pos] + addition + content[insert_pos:]
            
            # Add _load_domain_patterns method
            # Find end of class
            # Add before last line of class (simplified - add at end of file)
            load_method = '''
    
    def _load_domain_patterns(self):
        """Load domain-specific patterns from config."""
        if not self.domain_config:
            return
        
        # Try to get domain-specific patterns for this transformer
        domain_patterns = self.domain_config.get_domain_specific_patterns(
            self.narrative_id
        )
        
        if domain_patterns:
            # Merge with existing patterns
            for pattern_type, patterns in domain_patterns.items():
                if hasattr(self, pattern_type):
                    # Extend existing list
                    existing = getattr(self, pattern_type)
                    if isinstance(existing, list):
                        existing.extend(patterns)
'''
            
            content += load_method
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"  ✓ Adapted {class_name}")


def main():
    adapter = TransformerAdapter()
    adapter.adapt_all_transformers()


if __name__ == '__main__':
    main()

