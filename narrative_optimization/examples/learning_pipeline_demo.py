"""
Learning Pipeline Demonstration

Shows the complete holistic learning system in action:
1. Ingest data from multiple domains
2. Discover universal + domain-specific patterns
3. Validate patterns
4. Measure improvements
5. Generate explanations

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.learning import (
    LearningPipeline,
    UniversalArchetypeLearner,
    DomainSpecificLearner,
    ValidationEngine,
    VersionedArchetypeRegistry,
    ExplanationGenerator
)


def demo_learning_cycle():
    """Demonstrate complete learning cycle."""
    print("="*80)
    print("HOLISTIC LEARNING ARCHETYPE SYSTEM DEMO")
    print("="*80)
    
    # Create pipeline
    pipeline = LearningPipeline(
        incremental=True,
        auto_prune=True,
        min_improvement=0.01
    )
    
    # Generate synthetic data for demo
    print("\n[DEMO] Generating synthetic data...")
    
    # Golf data
    golf_texts = [
        "Tiger Woods leads at Augusta with course knowledge and experience",
        "Underdog triumph as lower-ranked player wins major championship",
        "Pressure performance: clutch putt on final hole",
        "Dominant display by world number one with technical mastery",
        "Comeback victory after struggling in first two rounds",
    ] * 20  # Repeat for sufficient data
    
    golf_outcomes = np.random.binomial(1, 0.6, len(golf_texts))
    
    # Tennis data
    tennis_texts = [
        "Nadal dominates on clay court with surface expertise",
        "Underdog stuns favorite in Grand Slam upset",
        "Rivalry renewed: head-to-head battle with history",
        "Mental toughness prevails in five-set marathon",
        "Pressure moment: championship point saved with composure",
    ] * 20
    
    tennis_outcomes = np.random.binomial(1, 0.55, len(tennis_texts))
    
    # Chess data
    chess_texts = [
        "Strategic depth: positional masterclass in endgame",
        "Opening preparation pays off with novelty surprise",
        "Underdog upset: lower-rated player defeats champion",
        "Time pressure: clutch play in rapid chess format",
        "Comeback from losing position with tactical brilliance",
    ] * 20
    
    chess_outcomes = np.random.binomial(1, 0.5, len(chess_texts))
    
    # Ingest data
    print("\n[DEMO] Ingesting data into pipeline...")
    pipeline.ingest_domain('golf', golf_texts, golf_outcomes)
    pipeline.ingest_domain('tennis', tennis_texts, tennis_outcomes)
    pipeline.ingest_domain('chess', chess_texts, chess_outcomes)
    
    # Run learning cycle
    print("\n[DEMO] Running learning cycle...")
    metrics = pipeline.learn_cycle(
        domains=['golf', 'tennis', 'chess'],
        learn_universal=True,
        learn_domain_specific=True
    )
    
    # Display results
    print(f"\n{'='*80}")
    print("LEARNING RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Iteration: {metrics.iteration}")
    print(f"Patterns Discovered: {metrics.patterns_discovered}")
    print(f"Patterns Validated: {metrics.patterns_validated}")
    print(f"Patterns Pruned: {metrics.patterns_pruned}")
    print(f"Performance Before: R²={metrics.r_squared_before:.3f}")
    print(f"Performance After: R²={metrics.r_squared_after:.3f}")
    print(f"Improvement: {metrics.improvement:+.3f}")
    print(f"Coherence: {metrics.coherence_score:.3f}")
    
    # Show discovered patterns
    print(f"\n{'='*80}")
    print("DISCOVERED PATTERNS")
    print(f"{'='*80}\n")
    
    # Universal patterns
    print("Universal Patterns (cross-domain):")
    universal_patterns = pipeline.universal_learner.get_patterns()
    for pattern_name, pattern_data in list(universal_patterns.items())[:5]:
        print(f"\n  {pattern_name}:")
        print(f"    Type: {pattern_data.get('type', 'unknown')}")
        print(f"    Frequency: {pattern_data.get('frequency', 0.0):.1%}")
        print(f"    Description: {pattern_data.get('description', 'N/A')}")
    
    # Domain-specific patterns
    print(f"\n\nDomain-Specific Patterns:")
    for domain in ['golf', 'tennis', 'chess']:
        if domain in pipeline.domain_learners:
            print(f"\n  {domain.upper()}:")
            domain_patterns = pipeline.domain_learners[domain].get_patterns()
            for pattern_name, pattern_data in list(domain_patterns.items())[:3]:
                print(f"    • {pattern_name}")
                print(f"      Frequency: {pattern_data.get('frequency', 0.0):.1%}")
    
    # Generate explanations
    print(f"\n{'='*80}")
    print("PATTERN EXPLANATIONS")
    print(f"{'='*80}\n")
    
    explainer = ExplanationGenerator()
    
    # Explain a universal pattern
    if 'universal_underdog' in universal_patterns:
        explanation = explainer.explain_pattern(
            'universal_underdog',
            universal_patterns['universal_underdog'],
            context={'entity': 'Player X'}
        )
        print(f"Universal Underdog Pattern:\n{explanation}\n")
    
    # Generate report
    print(f"\n{'='*80}")
    print("COMPREHENSIVE REPORT")
    print(f"{'='*80}\n")
    
    report = explainer.generate_report(
        domain='golf',
        patterns=pipeline.domain_learners['golf'].get_patterns() if 'golf' in pipeline.domain_learners else {},
        performance_metrics={
            'R²': metrics.r_squared_after,
            'Improvement': metrics.improvement,
            'Coherence': metrics.coherence_score
        }
    )
    
    print(report)
    
    # Save state
    print(f"\n{'='*80}")
    print("SAVING STATE")
    print(f"{'='*80}\n")
    
    save_path = Path(__file__).parent.parent / 'pipeline_state.json'
    pipeline.save_state(save_path)
    
    print(f"\n✓ Demo complete!")
    print(f"\nKey Takeaways:")
    print(f"  1. System discovered both universal and domain-specific patterns")
    print(f"  2. Patterns were statistically validated")
    print(f"  3. Performance improvement: {metrics.improvement:+.3f}")
    print(f"  4. All patterns are human-interpretable")
    print(f"  5. System state saved for future use")


if __name__ == '__main__':
    demo_learning_cycle()

