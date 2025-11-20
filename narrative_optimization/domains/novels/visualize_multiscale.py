"""
Visualize Multi-Scale Analysis

Creates heatmap showing transformer performance across scales.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


def create_multiscale_heatmap(multiscale_file: Path, output_file: Path):
    """Create multi-scale performance heatmap."""
    print("Creating multi-scale heatmap...")
    
    # Load multi-scale data
    with open(multiscale_file, 'r') as f:
        data = json.load(f)
    
    by_transformer = data.get('by_transformer', {})
    
    if not by_transformer:
        print("❌ No transformer data found")
        return
    
    # Create matrix
    scales = ['nano', 'micro', 'meso', 'macro']
    transformer_names = list(by_transformer.keys())
    
    # Build R² matrix
    r2_matrix = np.zeros((len(transformer_names), len(scales)))
    
    for i, trans_name in enumerate(transformer_names):
        for j, scale in enumerate(scales):
            if scale in by_transformer[trans_name]:
                r2_matrix[i, j] = by_transformer[trans_name][scale]['r2']
            else:
                r2_matrix[i, j] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Plot heatmap
    im = ax.imshow(r2_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(scales)))
    ax.set_yticks(range(len(transformer_names)))
    ax.set_xticklabels(scales, fontsize=12, color='white')
    ax.set_yticklabels(transformer_names, fontsize=10, color='white')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R² Score', rotation=270, labelpad=20, color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Add values in cells
    for i in range(len(transformer_names)):
        for j in range(len(scales)):
            if not np.isnan(r2_matrix[i, j]):
                text = ax.text(j, i, f'{r2_matrix[i, j]:.3f}',
                             ha='center', va='center',
                             color='white' if r2_matrix[i, j] < 0.5 else 'black',
                             fontsize=8, fontweight='bold')
    
    ax.set_title('Transformer Performance Across Scales', fontsize=16, color='white', pad=20)
    ax.set_xlabel('Scale', fontsize=12, color='white')
    ax.set_ylabel('Transformer', fontsize=12, color='white')
    ax.set_facecolor('#1a0a2e')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, facecolor='#0a0a0a', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved visualization to {output_file}")


def main():
    """Generate multi-scale visualizations."""
    print("="*80)
    print("MULTI-SCALE VISUALIZATION")
    print("="*80)
    
    multiscale_file = Path(__file__).parent / 'multi_scale_analysis.json'
    output_file = Path(__file__).parent / 'visualizations' / 'multiscale_heatmap.png'
    output_file.parent.mkdir(exist_ok=True)
    
    if not multiscale_file.exists():
        print(f"❌ Multi-scale file not found: {multiscale_file}")
        print("Run multi_scale_analysis.py first")
        return
    
    create_multiscale_heatmap(multiscale_file, output_file)
    
    print("\n" + "="*80)
    print("Visualization Complete")
    print("="*80)


if __name__ == '__main__':
    main()

