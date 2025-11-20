#!/usr/bin/env python3
"""
Domain Integration Assistant

Complete automated workflow for integrating new domains into the framework.
Handles template parsing, relativity analysis, experiment generation, and results synthesis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.domain_integration.template_parser import DomainIntegrationTemplate
from src.domain_integration.relativity_analyzer import CrossDomainRelativityAnalyzer
from typing import Dict, Any


def integrate_new_domain(template_path_or_text: str):
    """
    Complete workflow for integrating a new domain.
    
    Parameters
    ----------
    template_path_or_text : str
        Path to template file or template text itself
    
    Returns
    -------
    integration_results : dict
        Complete integration analysis
    """
    print("\n" + "🌟" * 40)
    print("\n  DOMAIN INTEGRATION ASSISTANT")
    print("  Automated Workflow with Cross-Domain Relativity")
    print("\n" + "🌟" * 40 + "\n")
    
    # Step 1: Parse Template
    print("📋 Step 1: Parsing Domain Template...")
    print("=" * 60)
    
    parser = DomainIntegrationTemplate()
    
    # Load template
    if Path(template_path_or_text).exists():
        with open(template_path_or_text) as f:
            template_text = f.read()
    else:
        template_text = template_path_or_text
    
    domain_spec = parser.parse_template(template_text)
    
    print(f"✓ Domain: {domain_spec['name']}")
    print(f"✓ Type: {domain_spec['type']}")
    print(f"✓ Selected Dimensions: {', '.join(domain_spec['dimensions'])}")
    print()
    
    # Step 2: Analyze Relativity
    print("🔗 Step 2: Analyzing Cross-Domain Relativity...")
    print("=" * 60)
    
    relativity_map = parser.map_to_existing_domains(domain_spec)
    
    print("Domain Similarities:")
    for existing_domain, similarity_info in relativity_map['similarities'].items():
        print(f"  {existing_domain}: {similarity_info['score']:.2f} similarity")
        if similarity_info['shared_dimensions']:
            print(f"    Shared: {', '.join(similarity_info['shared_dimensions'])}")
    
    if relativity_map['novel_aspects']:
        print(f"\nNovel Aspects:")
        for aspect in relativity_map['novel_aspects']:
            print(f"  • {aspect}")
    print()
    
    # Step 3: Generate Hypotheses
    print("💡 Step 3: Generating Hypotheses...")
    print("=" * 60)
    
    experiment_plan = parser.generate_experiment_plan(domain_spec, relativity_map)
    
    print(f"Experiments to run: {len(experiment_plan['experiments_to_run'])}")
    print(f"Transformers to test: {', '.join(experiment_plan['transformers_to_test'])}")
    
    if experiment_plan['combinations_to_test']:
        print(f"Combinations to test:")
        for combo in experiment_plan['combinations_to_test']:
            print(f"  • {combo[0]} + {combo[1]}")
    
    if experiment_plan['comparisons']:
        print(f"\nComparisons:")
        for comp in experiment_plan['comparisons']:
            print(f"  • vs {comp['compare_to']}: {comp['reason']}")
    print()
    
    # Step 4: Run Experiments (if data provided)
    print("🧪 Step 4: Experiment Execution...")
    print("=" * 60)
    print("Note: Actual experiments require data samples")
    print("Template processed - ready for data-driven experimentation")
    print()
    
    # Step 5: Relativity Analysis (framework update)
    print("🔄 Step 5: Framework Update via Relativity...")
    print("=" * 60)
    
    relativity_analyzer = CrossDomainRelativityAnalyzer()
    
    # Register existing domains
    for domain_name, domain_info in parser.existing_domains.items():
        if 'results' in domain_info and domain_info['results']:
            relativity_analyzer.register_domain(domain_name, domain_info)
    
    # Predict performance based on most similar domain
    most_similar = max(
        relativity_map['similarities'].items(),
        key=lambda x: x[1]['score']
    )[0] if relativity_map['similarities'] else None
    
    if most_similar:
        predictions = relativity_analyzer.predict_performance(domain_spec, most_similar)
        print(f"Performance predictions (based on {most_similar}):")
        print(f"  Confidence: {predictions.get('confidence', 'unknown')}")
        if predictions.get('expected_best'):
            print(f"  Expected best: {', '.join(predictions['expected_best'])}")
        for reason in predictions.get('reasoning', []):
            print(f"  • {reason}")
    print()
    
    # Step 6: Generate Integration Summary
    print("📊 Step 6: Integration Summary...")
    print("=" * 60)
    
    summary = {
        'domain_specification': domain_spec,
        'relativity_analysis': relativity_map,
        'experiment_plan': experiment_plan,
        'predictions': predictions if most_similar else {},
        'status': 'template_processed',
        'next_steps': [
            '1. Provide data samples matching template specification',
            '2. Run automated experiments',
            '3. Analyze results',
            '4. Update framework understanding',
            '5. Document insights and cross-domain effects'
        ]
    }
    
    print("✓ Template successfully processed")
    print("✓ Relativity mapped to existing domains")
    print("✓ Experiment plan generated")
    print("✓ Performance predictions made")
    print()
    print("📝 Next Steps:")
    for step in summary['next_steps']:
        print(f"  {step}")
    
    print("\n" + "✅" * 40)
    print("\n  TEMPLATE INTEGRATION COMPLETE")
    print("  Ready for data-driven experimentation")
    print("\n" + "✅" * 40 + "\n")
    
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate new domain into framework')
    parser.add_argument('template', help='Path to filled template or template text')
    
    args = parser.parse_args()
    
    try:
        results = integrate_new_domain(args.template)
    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage: python domain_integration_assistant.py DOMAIN_INTEGRATION_TEMPLATE.md")

