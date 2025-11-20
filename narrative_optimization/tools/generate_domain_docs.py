"""
Automatic Documentation Generator

Generates comprehensive documentation for domains.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_domain_registry


def generate_domain_documentation(domain_name: str, results: dict) -> str:
    """
    Generate comprehensive documentation for a domain.
    
    Parameters
    ----------
    domain_name : str
        Domain name
    results : dict
        Analysis results
    
    Returns
    -------
    str
        Markdown documentation
    """
    doc = f"# {domain_name.title()} Domain Analysis\n\n"
    
    # Metadata
    doc += "## Overview\n\n"
    doc += f"**Domain**: {domain_name}\n"
    doc += f"**π (Narrativity)**: {results.get('narrativity', results.get('pi', 'N/A'))}\n"
    doc += f"**Status**: Active\n\n"
    
    # Performance metrics
    doc += "## Performance\n\n"
    doc += f"- **R²**: {results.get('r_squared', 0):.1%}\n"
    doc += f"- **Д (Bridge)**: {results.get('delta', 0):.3f}\n"
    doc += f"- **Efficiency**: {results.get('efficiency', 0):.3f}\n"
    doc += f"- **Sample Size**: {results.get('n_samples', 'N/A')}\n\n"
    
    # Universal patterns
    if 'universal_patterns' in results:
        doc += "## Universal Patterns\n\n"
        universal = results['universal_patterns']
        
        if len(universal) > 0:
            for pattern_name, pattern_data in list(universal.items())[:5]:
                freq = pattern_data.get('frequency', 0)
                win_rate = pattern_data.get('win_rate', 0)
                doc += f"- **{pattern_name}**: {freq:.1%} frequency, {win_rate:.1%} win rate\n"
        else:
            doc += "No universal patterns detected.\n"
        doc += "\n"
    
    # Domain-specific patterns
    if 'domain_patterns' in results:
        doc += "## Domain-Specific Patterns\n\n"
        domain_patterns = results['domain_patterns']
        
        if len(domain_patterns) > 0:
            doc += f"Discovered {len(domain_patterns)} unique patterns:\n\n"
            for pattern_name, pattern_data in list(domain_patterns.items())[:5]:
                freq = pattern_data.get('frequency', 0)
                doc += f"- **{pattern_name}**: {freq:.1%} frequency\n"
        else:
            doc += "No domain-specific patterns discovered.\n"
        doc += "\n"
    
    # Similar domains
    if 'similar_domains' in results:
        doc += "## Structurally Similar Domains\n\n"
        similar = results['similar_domains']
        
        if len(similar) > 0:
            for domain, similarity in similar[:3]:
                doc += f"- **{domain}**: {similarity:.0%} similar\n"
        doc += "\n"
    
    # Story frequency
    if 'frequency_analysis' in results:
        doc += "## Story Frequency Analysis\n\n"
        freq = results['frequency_analysis']
        doc += f"- Predicted frequency: {freq.get('predicted_frequency', 0):.1%}\n"
        doc += f"- Observed frequency: {freq.get('observed_frequency', 0):.1%}\n"
        doc += f"- Meets expectations: {'Yes' if freq.get('meets_expectations', False) else 'No'}\n\n"
    
    # Trends
    if 'trends' in results:
        doc += "## Emerging Trends\n\n"
        trends = results['trends']
        
        if len(trends) > 0:
            for trend in trends[:5]:
                doc += f"- **{trend['pattern']}**: {trend['type']} ({trend['strength']:.2f} strength)\n"
        else:
            doc += "No significant trends detected.\n"
        doc += "\n"
    
    # Footer
    doc += "---\n\n"
    doc += f"*Auto-generated documentation*\n"
    
    return doc


def generate_all_documentation():
    """Generate documentation for all domains."""
    print("="*80)
    print("GENERATING DOMAIN DOCUMENTATION")
    print("="*80)
    
    registry = get_domain_registry()
    domains_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'domains'
    
    generated = 0
    
    for domain in registry.get_all_domains():
        print(f"\nGenerating docs for {domain.name}...")
        
        domain_dir = domains_dir / domain.name
        
        # Look for results
        result_files = list(domain_dir.glob('*results*.json'))
        
        if not result_files:
            print(f"  ⊙ No results found")
            continue
        
        # Load most recent
        latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_result) as f:
            results = json.load(f)
        
        # Generate documentation
        doc = generate_domain_documentation(domain.name, results)
        
        # Save
        doc_path = domain_dir / 'README.md'
        with open(doc_path, 'w') as f:
            f.write(doc)
        
        print(f"  ✓ Saved: {doc_path}")
        generated += 1
    
    print(f"\n{'='*80}")
    print(f"DOCUMENTATION COMPLETE: {generated} domains")
    print(f"{'='*80}")


if __name__ == '__main__':
    generate_all_documentation()

