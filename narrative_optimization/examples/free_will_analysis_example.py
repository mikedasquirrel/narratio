"""
Example: Free Will vs Determinism Narrative Analysis

Demonstrates complete pipeline for analyzing narratives for free will vs determinism.

Usage:
    python examples/free_will_analysis_example.py

Author: Narrative Integration System
Date: November 2025
"""

import sys
import os

# Add parent directory to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

try:
    from narrative_optimization.src.analysis.free_will_analyzer import NarrativeFreeWillAnalyzer
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.analysis.free_will_analyzer import NarrativeFreeWillAnalyzer


def main():
    """Run example analysis."""
    
    print("=" * 80)
    print("FREE WILL VS DETERMINISM NARRATIVE ANALYSIS")
    print("=" * 80)
    print()
    
    # Example narratives
    deterministic_story = """
    The prophecy had been written centuries ago. The ancient scrolls foretold 
    that the chosen one would come, and when the stars aligned, destiny would 
    be fulfilled. There was no escape from what was meant to be. The hero 
    tried to resist, but fate compelled him forward. Every action led inevitably 
    to the foretold conclusion. The outcome was predetermined, written in the 
    stars before time began.
    """
    
    free_will_story = """
    Sarah stood at the crossroads, uncertain which path to take. She could 
    choose to go left, or right, or turn back entirely. The decision was hers 
    alone. No prophecy guided her, no fate compelled her. She deliberated, 
    weighing her options carefully. Finally, she made her choice freely, 
    knowing that she could have chosen differently. The future was open, 
    contingent on her will.
    """
    
    mixed_story = """
    The old man knew that some things were inevitable - death, decay, the 
    passage of time. But within those constraints, he still had choices. 
    He could decide how to face his fate, what meaning to give his remaining 
    days. The structure was determined, but his agency within it was real. 
    He chose courage over despair, love over bitterness.
    """
    
    stories = [deterministic_story, free_will_story, mixed_story]
    story_names = ["Deterministic Story", "Free Will Story", "Mixed Story"]
    
    print("Initializing analyzer...")
    print("  - Loading SentenceTransformers...")
    print("  - Loading spaCy model...")
    print()
    
    # Initialize analyzer with configurable weights
    analyzer = NarrativeFreeWillAnalyzer(
        use_sentence_transformers=True,
        use_spacy=True,
        model_name='all-MiniLM-L6-v2',  # Fast model
        spacy_model='en_core_web_sm',
        extract_causal_graphs=True,
        track_observability=True,
        # Custom weights (optional - these are defaults)
        temporal_weight=0.30,
        semantic_weight=0.40,
        predictability_weight=0.30
    )
    
    print("Fitting analyzer on corpus...")
    analyzer.fit(stories)
    print("✓ Analyzer fitted\n")
    
    # Get interpretation
    print("CORPUS INTERPRETATION:")
    print("-" * 80)
    print(analyzer.get_interpretation())
    print()
    
    # Analyze each story
    print("=" * 80)
    print("INDIVIDUAL STORY ANALYSIS")
    print("=" * 80)
    print()
    
    for i, (story, name) in enumerate(zip(stories, story_names)):
        print(f"\n{name}")
        print("-" * 80)
        
        analysis = analyzer.analyze(story)
        
        print(f"\n📊 SCORES:")
        print(f"  Determinism Score: {analysis['determinism_score']:.3f} (0.0 = free will, 1.0 = deterministic)")
        print(f"  Agency Score: {analysis['agency_score']:.3f}")
        print(f"  Free Will Ratio: {analysis['free_will_ratio']:.3f}")
        print(f"  Inevitability Score: {analysis['inevitability_score']:.3f}")
        
        print(f"\n⏰ TEMPORAL ORIENTATION:")
        temporal = analysis['temporal_features']
        print(f"  Future: {temporal['future_orientation']:.3f}")
        print(f"  Past: {temporal['past_orientation']:.3f}")
        print(f"  Present: {temporal['present_orientation']:.3f}")
        
        print(f"\n🎭 SEMANTIC FIELDS:")
        semantic = analysis['semantic_features']
        print(f"  Fate Density: {semantic['fate_density']:.3f}")
        print(f"  Choice Density: {semantic['choice_density']:.3f}")
        print(f"  Agency Density: {semantic['agency_density']:.3f}")
        print(f"  Determinism Balance: {semantic['determinism_balance']:.3f}")
        
        print(f"\n📈 INFORMATION THEORY:")
        info = analysis['information_theory']
        print(f"  Entropy: {info['entropy']:.3f} (high = unpredictable)")
        print(f"  Predictability: {info['predictability']:.3f} (high = deterministic)")
        print(f"  Redundancy: {info['redundancy']:.3f}")
        
        if analysis['agency_analysis']['n_agents'] > 0:
            print(f"\n👤 AGENCY ANALYSIS:")
            agency = analysis['agency_analysis']
            print(f"  Agents (actors): {agency['n_agents']:.0f}")
            print(f"  Patients (acted upon): {agency['n_patients']:.0f}")
            print(f"  Free Will Score: {agency['free_will_score']:.3f}")
        
        print(f"\n🔗 CAUSAL STRUCTURE:")
        causal = analysis['causal_structure']
        print(f"  Path Dependency: {causal['path_dependency']:.3f}")
        print(f"  Branching Factor: {causal['branching_factor']:.3f}")
        print(f"  Deterministic Ratio: {causal['deterministic_ratio']:.3f}")
        
        print(f"\n👁️ OBSERVABILITY:")
        obs = analysis['observability']
        print(f"  Explicit Causality: {obs['explicit_ratio']:.3f}")
        print(f"  Hidden Causality: {obs['hidden_ratio']:.3f}")
        print(f"  Omniscient Perspective: {obs['omniscient_ratio']:.3f}")
    
    # Test structure-outcome prediction
    print("\n" + "=" * 80)
    print("STRUCTURE-OUTCOME PREDICTION TEST")
    print("=" * 80)
    print()
    
    beginning = "The hero was born under a dark star. The prophecy said he would bring destruction."
    ending_deterministic = "As foretold, the hero brought destruction. The prophecy was fulfilled."
    ending_surprise = "But the hero chose differently. He broke the prophecy and saved the world."
    
    print("Testing deterministic narrative (beginning → ending):")
    pred1 = analyzer.predict_outcome_from_structure(beginning, ending_deterministic)
    print(f"  Beginning-Ending Similarity: {pred1['beginning_ending_similarity']:.3f}")
    print(f"  Determinism from Structure: {pred1['determinism_from_structure']:.3f}")
    print(f"  Is Deterministic: {pred1['is_deterministic']}")
    
    print("\nTesting free will narrative (beginning → surprise ending):")
    pred2 = analyzer.predict_outcome_from_structure(beginning, ending_surprise)
    print(f"  Beginning-Ending Similarity: {pred2['beginning_ending_similarity']:.3f}")
    print(f"  Determinism from Structure: {pred2['determinism_from_structure']:.3f}")
    print(f"  Is Free Will: {pred2['is_free_will']}")
    
    # Nominative analysis
    print("\n" + "=" * 80)
    print("NOMINATIVE DETERMINISM ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze the deterministic story's nominative patterns
    print("Analyzing deterministic story:")
    nom_analysis = analyzer.analyze_nominative_determinism(deterministic_story)
    
    print(f"  Nominative Determinism Score: {nom_analysis['nominative_determinism_score']:.3f}")
    print("\n  Character Agency Scores:")
    for key, value in nom_analysis['character_agency_scores'].items():
        print(f"    {key}: {value:.3f}")
    
    print("\n  Naming Pattern Analysis:")
    for key, value in nom_analysis['naming_pattern_analysis'].items():
        print(f"    {key}: {value}")
    
    print("\n  Naming Features:")
    print(f"    Proper names: {nom_analysis['naming_features']['proper_name_density']:.3f}")
    print(f"    Generic labels: {nom_analysis['naming_features']['generic_label_ratio']:.3f}")
    print(f"    Deterministic titles: {nom_analysis['naming_features']['title_frequency']:.3f}")
    print(f"    Name consistency: {nom_analysis['naming_features']['name_consistency']:.3f}")
    
    # Analyze the free will story's nominative patterns
    print("\n\nAnalyzing free will story:")
    nom_analysis2 = analyzer.analyze_nominative_determinism(free_will_story)
    
    print(f"  Nominative Determinism Score: {nom_analysis2['nominative_determinism_score']:.3f}")
    print("\n  Character Agency Scores:")
    for key, value in nom_analysis2['character_agency_scores'].items():
        print(f"    {key}: {value:.3f}")
    
    print("\n  Naming Evolution:")
    print(f"    Pattern: {nom_analysis2['naming_pattern_analysis']['pattern']}")
    print(f"    Direction: {nom_analysis2['naming_pattern_analysis']['direction']}")
    
    # Clustering analysis
    print("\n" + "=" * 80)
    print("CLUSTERING BY DETERMINISM")
    print("=" * 80)
    print()
    
    clusters = analyzer.cluster_by_determinism(stories)
    
    print(f"High Determinism (>0.7): {clusters['high_determinism']['count']} stories")
    print(f"  Average Score: {clusters['high_determinism']['avg_score']:.3f}")
    print(f"  Stories: {[story_names[i] for i in clusters['high_determinism']['indices']]}")
    
    print(f"\nMedium Determinism (0.3-0.7): {clusters['medium_determinism']['count']} stories")
    print(f"  Average Score: {clusters['medium_determinism']['avg_score']:.3f}")
    print(f"  Stories: {[story_names[i] for i in clusters['medium_determinism']['indices']]}")
    
    print(f"\nLow Determinism (<0.3): {clusters['low_determinism']['count']} stories")
    print(f"  Average Score: {clusters['low_determinism']['avg_score']:.3f}")
    print(f"  Stories: {[story_names[i] for i in clusters['low_determinism']['indices']]}")
    
    # Compare fiction to reality
    print("\n" + "=" * 80)
    print("FICTION VS REALITY COMPARISON")
    print("=" * 80)
    print()
    
    fictional = """
    The wizard cast a spell that changed everything. Magic flowed through 
    the ancient runes, rewriting reality itself. The prophecy came true 
    because the stars aligned and fate demanded it.
    """
    
    reality = """
    The scientist conducted an experiment. The results followed from the 
    laws of physics. The outcome was determined by the initial conditions 
    and natural forces. Cause and effect operated predictably.
    """
    
    comparison = analyzer.compare_narrative_to_reality(fictional, reality)
    
    print(f"Semantic Similarity: {comparison['semantic_similarity']:.3f}")
    print(f"Structural Similarity: {comparison['structural_similarity']:.3f}")
    print(f"Determinism Similarity: {comparison['determinism_similarity']:.3f}")
    print(f"Maps to Reality: {comparison['maps_to_reality']}")
    print(f"Fiction Determinism: {comparison['fiction_determinism']:.3f}")
    print(f"Reality Determinism: {comparison['reality_determinism']:.3f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

