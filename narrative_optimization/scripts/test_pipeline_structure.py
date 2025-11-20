#!/usr/bin/env python3
"""Test that pipeline structure is correct (FeatureUnion, not sequential)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing pipeline structure...")
print()

from src.pipelines.pipeline_composer import PipelineComposer
from src.pipelines.domain_config import DomainConfig
from sklearn.pipeline import FeatureUnion

project_root = Path(__file__).parent.parent
config_path = project_root / 'domains' / 'golf' / 'config.yaml'

# Load config
config = DomainConfig.from_yaml(config_path)
print(f"✓ Config loaded: п={config.pi:.3f}")

# Create composer
composer = PipelineComposer(project_root)
print(f"✓ Composer ready")

# Compose pipeline
print("\nComposing pipeline...")
pipeline_info = composer.compose_pipeline(
    config,
    data_path=None,
    target_feature_count=100,
    use_cache=False
)

pipeline = pipeline_info['pipeline']

print("\n" + "="*80)
print("PIPELINE STRUCTURE ANALYSIS")
print("="*80)

# Check pipeline steps
print(f"\nPipeline steps: {len(pipeline.steps)}")
for i, (name, step) in enumerate(pipeline.steps, 1):
    print(f"  {i}. {name}: {type(step).__name__}")
    
# Check if using FeatureUnion
if 'features' in pipeline.named_steps:
    features_step = pipeline.named_steps['features']
    if isinstance(features_step, FeatureUnion):
        print(f"\n✓ CORRECT: Using FeatureUnion with {len(features_step.transformer_list)} transformers")
        print(f"\nTransformers in parallel:")
        for name, transformer in features_step.transformer_list:
            print(f"  - {name}: {type(transformer).__name__}")
    else:
        print(f"\n✗ ERROR: 'features' step is not a FeatureUnion!")
        print(f"  Type: {type(features_step)}")
else:
    print(f"\n✗ ERROR: No 'features' step in pipeline!")
    print(f"  Steps: {list(pipeline.named_steps.keys())}")

# Test with dummy data
print("\n" + "="*80)
print("TESTING WITH DUMMY DATA")
print("="*80)

dummy_texts = [
    "Player won the tournament with excellent play.",
    "Great performance led to victory.",
    "Champion emerged after final round."
]

print(f"\nFitting with {len(dummy_texts)} texts...")
try:
    pipeline.fit(dummy_texts)
    print(f"✓ Fit successful")
    
    print(f"\nTransforming...")
    features = pipeline.transform(dummy_texts)
    print(f"✓ Transform successful")
    print(f"  Feature shape: {features.shape}")
    print(f"  Expected: ({len(dummy_texts)}, ~300-400)")
    
    if features.shape[0] == len(dummy_texts):
        print(f"\n✓ PASS: Correct number of samples")
    else:
        print(f"\n✗ FAIL: Wrong number of samples")
        
    if features.shape[1] > 100 and features.shape[1] < 1000:
        print(f"✓ PASS: Reasonable number of features")
    else:
        print(f"✗ FAIL: Suspicious number of features")
        
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

