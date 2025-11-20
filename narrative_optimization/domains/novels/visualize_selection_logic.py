"""
Visualize Transformer Selection Logic

Creates decision tree visualization showing how transformers are selected
based on domain characteristics.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


def create_selection_decision_chart(output_file: Path):
    """Create transformer selection logic visualization."""
    print("Creating transformer selection logic chart...")
    
    # Define selection logic
    selection_logic = {
        'Low π (< 0.3)': {
            'description': 'Constrained domains',
            'recommended': ['Statistical', 'Structural', 'Temporal'],
            'optional': ['Nominative', 'Phonetic'],
            'avoid': ['Semantic', 'Ensemble']
        },
        'Medium π (0.3-0.7)': {
            'description': 'Balanced domains',
            'recommended': ['Statistical', 'Nominative', 'Phonetic', 'Structural', 'Semantic'],
            'optional': ['Ensemble', 'Relational', 'Multi-Scale'],
            'avoid': []
        },
        'High π (> 0.7)': {
            'description': 'Open domains',
            'recommended': ['All Nominative', 'Phonetic', 'Ensemble', 'Semantic', 'Multi-Scale'],
            'optional': ['Statistical'],
            'avoid': []
        }
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#1a0a2e')
    
    # Hide axes
    ax.axis('off')
    
    # Draw decision tree
    y_pos = 0.9
    x_center = 0.5
    
    # Title
    ax.text(x_center, 0.95, 'Transformer Selection Decision Logic',
            ha='center', va='top', fontsize=18, color='white', fontweight='bold')
    
    # Root node
    ax.text(x_center, y_pos, 'Domain Narrativity (π)',
            ha='center', va='center', fontsize=14, color='#a855f7',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#2a1a4e', edgecolor='#a855f7', linewidth=2))
    
    # Branch positions
    branches = [
        (0.2, 'Low π\n(< 0.3)'),
        (0.5, 'Medium π\n(0.3-0.7)'),
        (0.8, 'High π\n(> 0.7)')
    ]
    
    y_branch = 0.75
    
    # Draw branches
    for x, label in branches:
        # Draw line from root to branch
        ax.plot([x_center, x], [y_pos - 0.05, y_branch + 0.05],
                color='#a855f7', linewidth=2, alpha=0.5)
        
        # Branch node
        ax.text(x, y_branch, label,
                ha='center', va='center', fontsize=11, color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a0a2e', edgecolor='#06b6d4', linewidth=1.5))
    
    # Details for each branch
    y_details = 0.55
    
    for i, (x, label_text) in enumerate(branches):
        key = list(selection_logic.keys())[i]
        logic = selection_logic[key]
        
        # Description
        ax.text(x, y_details, logic['description'],
                ha='center', va='top', fontsize=9, color=(1, 1, 1, 0.8),
                style='italic')
        
        # Recommended transformers
        y_rec = y_details - 0.08
        ax.text(x, y_rec, 'Recommended:',
                ha='center', va='top', fontsize=9, color='#10b981', fontweight='bold')
        
        rec_text = '\n'.join(logic['recommended'][:3])
        if len(logic['recommended']) > 3:
            rec_text += f'\n+ {len(logic["recommended"]) - 3} more'
        
        ax.text(x, y_rec - 0.05, rec_text,
                ha='center', va='top', fontsize=8, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a3a2e', alpha=0.7))
    
    # Add examples at bottom
    y_examples = 0.15
    ax.text(x_center, y_examples, 'Domain Examples',
            ha='center', va='top', fontsize=12, color='#a855f7', fontweight='bold')
    
    examples = {
        0.2: 'Lottery (π=0.04)\nAviation (π=0.12)',
        0.5: 'Movies (π=0.65)\nNonfiction (π=0.61)',
        0.8: 'Novels (π=0.72)\nWWE (π=0.974)'
    }
    
    for x, ex_text in examples.items():
        ax.text(x, y_examples - 0.08, ex_text,
                ha='center', va='top', fontsize=8, color=(1, 1, 1, 0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a0a', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, facecolor='#0a0a0a', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved visualization to {output_file}")


def main():
    """Generate transformer selection visualizations."""
    print("="*80)
    print("TRANSFORMER SELECTION VISUALIZATION")
    print("="*80)
    
    output_file = Path(__file__).parent / 'visualizations' / 'selection_logic.png'
    output_file.parent.mkdir(exist_ok=True)
    
    create_selection_decision_chart(output_file)
    
    print("\n" + "="*80)
    print("Visualization Complete")
    print("="*80)


if __name__ == '__main__':
    main()

