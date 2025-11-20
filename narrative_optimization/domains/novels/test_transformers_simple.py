"""
Simple transformer test with immediate output.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Force immediate output
def log(msg):
    print(msg, flush=True)

log("="*80)
log("STARTING TRANSFORMER ANALYSIS")
log("="*80)

# Add to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
log(f"Project root: {project_root}")

# Load dataset
log("\n[1/5] Loading novels dataset...")
dataset_path = Path(__file__).parent / 'data' / 'novels_dataset.json'
log(f"Dataset path: {dataset_path}")

with open(dataset_path, 'r') as f:
    novels = json.load(f)

log(f"✓ Loaded {len(novels)} novels")

# Check data quality
log("\n[2/5] Checking data quality...")
with_plot = sum(1 for n in novels if n.get('plot_summary'))
with_chars = sum(1 for n in novels if n.get('character_names'))
log(f"  Novels with plots: {with_plot}/{len(novels)}")
log(f"  Novels with characters: {with_chars}/{len(novels)}")

# Extract texts for transformers
log("\n[3/5] Preparing texts...")
texts = []
for i, novel in enumerate(novels):
    plot = novel.get('plot_summary', '') or novel.get('full_narrative', '') or novel.get('description', '')
    if plot:
        texts.append(plot)
    
    if (i + 1) % 100 == 0:
        log(f"  Processed {i+1}/{len(novels)} novels...")

log(f"✓ Prepared {len(texts)} text samples")

# Run statistical transformer (simplest)
log("\n[4/5] Running Statistical Transformer...")
try:
    from narrative_optimization.src.transformers.statistical import StatisticalTransformer
    log("  Initializing...")
    stat_transformer = StatisticalTransformer(max_features=50)
    log("  Fitting and transforming...")
    stat_features = stat_transformer.fit_transform(texts[:100])  # Test on first 100
    log(f"  ✓ Statistical features: {stat_features.shape}")
except Exception as e:
    log(f"  ✗ Error: {e}")
    import traceback
    log(traceback.format_exc())

# Run nominative transformer
log("\n[5/5] Running Nominative Transformer...")
try:
    from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
    log("  Initializing...")
    nom_transformer = NominativeAnalysisTransformer()
    
    # Prepare nominative context
    log("  Extracting character names...")
    author_names = [n.get('author', '') for n in novels[:100]]
    character_names = [n.get('character_names', []) for n in novels[:100]]
    log(f"    Authors: {len(author_names)}")
    log(f"    Characters: {sum(len(c) for c in character_names)} total")
    
    log("  Fitting and transforming...")
    nom_features = nom_transformer.fit_transform(
        texts[:100],
        author_names=author_names,
        character_names=character_names
    )
    log(f"  ✓ Nominative features: {nom_features.shape}")
except Exception as e:
    log(f"  ✗ Error: {e}")
    import traceback
    log(traceback.format_exc())

log("\n" + "="*80)
log("TEST COMPLETE")
log("="*80)






