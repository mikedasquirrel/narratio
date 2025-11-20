"""
Visualize Feature Attribution

Creates charts showing which transformers contribute most.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


def create_attribution_charts(attribution_file: Path, output_dir: Path):
    """Create feature attribution visualizations."""
    print("Creating feature attribution charts...")
    
    # Load attribution data
    with open(attribution_file, 'r') as f:
        data = json.load(f)
    
    ablation = data['ablation_study']
    permutation = data['permutation_importance']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0a0a0a')
    
    # 1. Ablation Impact
    ax = axes[0]
    sorted_ablation = sorted(ablation.items(), key=lambda x: x[1]['impact'], reverse=True)[:15]
    trans_names = [t[0] for t in sorted_ablation]
    impacts = [t[1]['impact'] for t in sorted_ablation]
    
    bars = ax.barh(range(len(trans_names)), impacts, color='#ef4444', alpha=0.7)
    ax.set_yticks(range(len(trans_names)))
    ax.set_yticklabels(trans_names, fontsize=9, color='white')
    ax.set_xlabel('Impact (R² Drop)', fontsize=12, color='white')
    ax.set_title('Top 15 Transformers by Ablation Impact', fontsize=12, color='white', pad=15)
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a0a2e')
    ax.invert_yaxis()
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars, impacts)):
        ax.text(val, i, f' {val:.4f}', va='center', color='white', fontsize=8)
    
    # 2. Permutation Importance
    ax = axes[1]
    sorted_perm = sorted(permutation.items(), key=lambda x: x[1]['mean'], reverse=True)[:15]
    trans_names = [t[0] for t in sorted_perm]
    means = [t[1]['mean'] for t in sorted_perm]
    stds = [t[1]['std'] for t in sorted_perm]
    
    bars = ax.barh(range(len(trans_names)), means, xerr=stds, color='#10b981', alpha=0.7, capsize=3)
    ax.set_yticks(range(len(trans_names)))
    ax.set_yticklabels(trans_names, fontsize=9, color='white')
    ax.set_xlabel('Permutation Importance', fontsize=12, color='white')
    ax.set_title('Top 15 Transformers by Permutation Importance', fontsize=12, color='white', pad=15)
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a0a2e')
    ax.invert_yaxis()
    
    # 3. Comparison: Ablation vs Permutation
    ax = axes[2]
    
    # Get common transformers
    common = set(ablation.keys()) & set(permutation.keys())
    common_sorted = sorted(common, key=lambda x: ablation[x]['impact'], reverse=True)[:15]
    
    ablation_vals = [ablation[t]['impact'] for t in common_sorted]
    perm_vals = [permutation[t]['mean'] for t in common_sorted]
    
    x = np.arange(len(common_sorted))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, ablation_vals, width, label='Ablation', color='#ef4444', alpha=0.7)
    bars2 = ax.barh(x + width/2, perm_vals, width, label='Permutation', color='#10b981', alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels(common_sorted, fontsize=9, color='white')
    ax.set_xlabel('Importance Score', fontsize=12, color='white')
    ax.set_title('Ablation vs Permutation (Top 15)', fontsize=12, color='white', pad=15)
    ax.legend(loc='lower right', fontsize=10, facecolor='#1a0a2e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a0a2e')
    ax.invert_yaxis()
    
    plt.tight_layout()
    output_file = output_dir / 'feature_attribution.png'
    plt.savefig(output_file, dpi=300, facecolor='#0a0a0a', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved visualization to {output_file}")


def main():
    """Generate feature attribution visualizations."""
    print("="*80)
    print("FEATURE ATTRIBUTION VISUALIZATION")
    print("="*80)
    
    attribution_file = Path(__file__).parent / 'feature_attribution.json'
    output_dir = Path(__file__).parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    if not attribution_file.exists():
        print(f"❌ Attribution file not found: {attribution_file}")
        print("Run feature_attribution.py first")
        return
    
    create_attribution_charts(attribution_file, output_dir)
    
    print("\n" + "="*80)
    print("Visualization Complete")
    print("="*80)


if __name__ == '__main__':
    main()

