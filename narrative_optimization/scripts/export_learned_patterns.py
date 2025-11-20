"""
Export Learned Patterns

Exports learned patterns in various formats for use in other systems.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.learning import LearningPipeline
from src.registry import get_domain_registry


def export_to_json(pipeline: LearningPipeline, output_path: Path):
    """Export patterns to JSON."""
    export_data = {
        'universal_patterns': pipeline.universal_learner.get_patterns(),
        'domain_patterns': {},
        'metadata': {
            'iteration': pipeline.iteration,
            'n_domains': len(pipeline.domain_learners),
            'export_date': str(Path(__file__).stat().st_mtime)
        }
    }
    
    for domain, learner in pipeline.domain_learners.items():
        export_data['domain_patterns'][domain] = learner.get_patterns()
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ JSON export: {output_path}")


def export_to_csv(pipeline: LearningPipeline, output_path: Path):
    """Export patterns to CSV."""
    rows = []
    
    # Universal patterns
    for pattern_name, pattern_data in pipeline.universal_learner.get_patterns().items():
        rows.append({
            'pattern_name': pattern_name,
            'type': 'universal',
            'domain': 'all',
            'keywords': ', '.join(pattern_data.get('keywords', pattern_data.get('patterns', []))[:5]),
            'frequency': pattern_data.get('frequency', 0),
            'correlation': pattern_data.get('correlation', 0),
            'win_rate': pattern_data.get('win_rate', 0)
        })
    
    # Domain patterns
    for domain, learner in pipeline.domain_learners.items():
        for pattern_name, pattern_data in learner.get_patterns().items():
            rows.append({
                'pattern_name': pattern_name,
                'type': 'domain_specific',
                'domain': domain,
                'keywords': ', '.join(pattern_data.get('patterns', [])[:5]),
                'frequency': pattern_data.get('frequency', 0),
                'correlation': pattern_data.get('correlation', 0),
                'win_rate': pattern_data.get('win_rate', 0.5)
            })
    
    # Write CSV
    if len(rows) > 0:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✓ CSV export: {output_path}")
    else:
        print(f"⊙ No patterns to export")


def export_pattern_library():
    """Export complete pattern library."""
    print("="*80)
    print("EXPORTING LEARNED PATTERNS")
    print("="*80)
    
    # Load pipeline state
    state_path = Path(__file__).parent.parent / 'pipeline_state.json'
    
    if not state_path.exists():
        print("\n✗ No pipeline state found")
        print("  Run learning cycle first: python RUN_COMPLETE_SYSTEM.py")
        return
    
    # Create pipeline and load
    pipeline = LearningPipeline()
    pipeline.load_state(state_path)
    
    print(f"\n✓ Pipeline loaded")
    print(f"  Iteration: {pipeline.iteration}")
    print(f"  Domains: {len(pipeline.domain_learners)}")
    
    # Export to multiple formats
    output_dir = Path(__file__).parent.parent / 'exports'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting patterns...")
    
    # JSON
    json_path = output_dir / 'learned_patterns.json'
    export_to_json(pipeline, json_path)
    
    # CSV
    csv_path = output_dir / 'learned_patterns.csv'
    export_to_csv(pipeline, csv_path)
    
    # Pattern count
    universal_count = len(pipeline.universal_learner.get_patterns())
    domain_count = sum(len(learner.get_patterns()) for learner in pipeline.domain_learners.values())
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"  Universal patterns: {universal_count}")
    print(f"  Domain patterns: {domain_count}")
    print(f"  Total: {universal_count + domain_count}")
    print(f"  Location: {output_dir}")


if __name__ == '__main__':
    export_pattern_library()

