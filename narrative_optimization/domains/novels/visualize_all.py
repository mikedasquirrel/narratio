"""
Generate All Visualizations

Runs all visualization scripts in sequence.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("GENERATING ALL VISUALIZATIONS")
print("="*80)

# Import and run all visualization modules
print("\n[1/4] Transformer interactions...")
try:
    from narrative_optimization.domains.novels.visualize_transformer_interactions import main as vis_interactions
    vis_interactions()
except Exception as e:
    print(f"  ⚠️  Skipped (data not available or error): {e}")

print("\n[2/4] Multi-scale analysis...")
try:
    from narrative_optimization.domains.novels.visualize_multiscale import main as vis_multiscale
    vis_multiscale()
except Exception as e:
    print(f"  ⚠️  Skipped (data not available or error): {e}")

print("\n[3/4] Feature attribution...")
try:
    from narrative_optimization.domains.novels.visualize_attribution import main as vis_attribution
    vis_attribution()
except Exception as e:
    print(f"  ⚠️  Skipped (data not available or error): {e}")

print("\n[4/4] Selection logic...")
try:
    from narrative_optimization.domains.novels.visualize_selection_logic import main as vis_selection
    vis_selection()
except Exception as e:
    print(f"  ⚠️  Skipped (data not available or error): {e}")

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print("="*80)
print("\nOutput directory: narrative_optimization/domains/novels/visualizations/")

