"""
Visualize Transformer Interactions

Creates network graph showing how transformers interact:
- Nodes: Transformers
- Edges: Correlation strength
- Colors: Interaction types
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import warnings
warnings.filterwarnings('ignore')


def create_interaction_network(interactions_file: Path, output_file: Path):
    """Create transformer interaction network visualization."""
    print("Creating transformer interaction network...")
    
    # Load interactions data
    with open(interactions_file, 'r') as f:
        data = json.load(f)
    
    transformer_names = data['transformer_names']
    correlation_matrix = np.array(data['correlation_matrix'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0a0a0a')
    
    # 1. Correlation Heatmap
    ax = axes[0, 0]
    im = ax.imshow(correlation_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(transformer_names)))
    ax.set_yticks(range(len(transformer_names)))
    ax.set_xticklabels(transformer_names, rotation=90, ha='right', fontsize=8, color='white')
    ax.set_yticklabels(transformer_names, fontsize=8, color='white')
    ax.set_title('Transformer Correlation Matrix', fontsize=14, color='white', pad=20)
    ax.set_facecolor('#1a0a2e')
    plt.colorbar(im, ax=ax)
    
    # 2. Interaction Types Bar Chart
    ax = axes[0, 1]
    interaction_counts = [
        data['summary']['n_complementary'],
        data['summary']['n_redundant'],
        data['summary']['n_synergistic'],
        data['summary']['n_antagonistic']
    ]
    interaction_labels = ['Complementary', 'Redundant', 'Synergistic', 'Antagonistic']
    colors = ['#06b6d4', '#fbbf24', '#10b981', '#ef4444']
    
    bars = ax.bar(interaction_labels, interaction_counts, color=colors, alpha=0.7)
    ax.set_title('Interaction Types', fontsize=14, color='white', pad=20)
    ax.set_ylabel('Count', fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a0a2e')
    
    # Add counts on bars
    for bar, count in zip(bars, interaction_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', color='white', fontweight='bold')
    
    # 3. Top Synergistic Pairs
    ax = axes[1, 0]
    synergistic_pairs = sorted(
        data['interactions']['synergistic'],
        key=lambda x: x['synergy'],
        reverse=True
    )[:10]
    
    if synergistic_pairs:
        pair_labels = [f"{p['transformer1'][:15]}\n+\n{p['transformer2'][:15]}" for p in synergistic_pairs]
        synergies = [p['synergy'] for p in synergistic_pairs]
        
        bars = ax.barh(range(len(pair_labels)), synergies, color='#10b981', alpha=0.7)
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels, fontsize=8, color='white')
        ax.set_xlabel('Synergy Score', fontsize=12, color='white')
        ax.set_title('Top 10 Synergistic Pairs', fontsize=14, color='white', pad=20)
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a0a2e')
        ax.invert_yaxis()
    
    # 4. Individual Transformer Performance
    ax = axes[1, 1]
    individual_imps = data['individual_importances']
    sorted_transformers = sorted(individual_imps.items(), key=lambda x: x[1], reverse=True)[:15]
    
    trans_labels = [t[0] for t in sorted_transformers]
    trans_values = [t[1] for t in sorted_transformers]
    
    bars = ax.barh(range(len(trans_labels)), trans_values, color='#a855f7', alpha=0.7)
    ax.set_yticks(range(len(trans_labels)))
    ax.set_yticklabels(trans_labels, fontsize=9, color='white')
    ax.set_xlabel('Individual R²', fontsize=12, color='white')
    ax.set_title('Top 15 Transformers (Individual Performance)', fontsize=14, color='white', pad=20)
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a0a2e')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, facecolor='#0a0a0a', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved visualization to {output_file}")


def main():
    """Generate transformer interaction visualizations."""
    print("="*80)
    print("TRANSFORMER INTERACTION VISUALIZATION")
    print("="*80)
    
    interactions_file = Path(__file__).parent / 'transformer_interactions.json'
    output_file = Path(__file__).parent / 'visualizations' / 'transformer_interactions.png'
    output_file.parent.mkdir(exist_ok=True)
    
    if not interactions_file.exists():
        print(f"❌ Interactions file not found: {interactions_file}")
        print("Run transformer_interactions.py first")
        return
    
    create_interaction_network(interactions_file, output_file)
    
    print("\n" + "="*80)
    print("Visualization Complete")
    print("="*80)


if __name__ == '__main__':
    main()

