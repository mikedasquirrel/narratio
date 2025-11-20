"""
Quick test run with smaller sample to verify pipeline works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.pipeline_composer import PipelineComposer
from src.pipelines.domain_config import DomainConfig
import json

domain = 'golf'
project_root = Path(__file__).parent.parent
config_path = project_root / 'domains' / domain / 'config.yaml'
data_path = project_root.parent / 'data' / 'domains' / 'golf_enhanced_narratives.json'

print(f"Loading config: {config_path}")
config = DomainConfig.from_yaml(config_path)

print(f"Loading data: {data_path}")
with open(data_path, 'r') as f:
    data = json.load(f)

# Sample just 100 records for quick test
import random
random.seed(42)
test_data = random.sample(data, min(100, len(data)))

print(f"Testing with {len(test_data)} records...")

# Create temp file
import tempfile
import os
temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
json.dump(test_data, temp_file)
temp_file.close()

try:
    composer = PipelineComposer(project_root)
    print("\n" + "="*80)
    print("RUNNING PIPELINE")
    print("="*80 + "\n")
    
    results = composer.run_pipeline(
        config,
        data_path=Path(temp_file.name),
        target_feature_count=300,
        use_cache=False  # Disable cache for quick test
    )
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"Features shape: {results.get('features', {}).get('shape', 'unknown')}")
    print(f"Analysis keys: {list(results.get('analysis', {}).keys())}")
    
finally:
    os.unlink(temp_file.name)

