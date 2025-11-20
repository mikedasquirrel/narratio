"""
Meta-Nominative Selection Effects Analysis

PRIMARY TEST: Are researchers with name-field fit OVERREPRESENTED in matching research topics?
SECONDARY TEST: Do they report different effect sizes?

This tests INTEREST/SELECTION, not just bias in findings.
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def calculate_narrativity(researchers, papers):
    """
    Calculate п (narrativity) for meta-nominative domain.
    
    Narrativity = degree to which names/narrative matters vs pure chance.
    """
    print(f"\n{'='*80}")
    print("CALCULATING NARRATIVITY (п)")
    print(f"{'='*80}\n")
    
    # Factors that indicate narrativity:
    # 1. Topic diversity (more topics = more room for selection)
    # 2. Name-topic variance (are names distributed across topics or clustered?)
    # 3. Career choice flexibility (academia has HIGH flexibility)
    
    # Extract topics
    all_topics = set()
    for r in researchers.values():
        topics = r.get('topics_studied', [])
        all_topics.update(topics)
    
    topic_diversity = len(all_topics)
    
    print(f"[1/4] Topic diversity: {topic_diversity} unique topics")
    print(f"      Topics: {', '.join(sorted(list(all_topics))[:10])}")
    
    # Career flexibility (academia = high)
    career_flexibility = 0.85  # Academics choose topics freely
    
    print(f"\n[2/4] Career flexibility: {career_flexibility:.2f}")
    print(f"      (Academia allows free topic selection)")
    
    # Outcome variance (different researchers report vastly different effects)
    effect_sizes = [r.get('average_effect_size') for r in researchers.values() 
                   if r.get('average_effect_size') is not None]
    
    if len(effect_sizes) > 1:
        effect_variance = np.std(effect_sizes)
        print(f"\n[3/4] Effect size variance: {effect_variance:.3f}")
        print(f"      (Range: {min(effect_sizes):.3f} - {max(effect_sizes):.3f})")
    else:
        effect_variance = 0.2
    
    # Subjectivity (choosing research topics is highly subjective)
    subjectivity = 0.75
    
    print(f"\n[4/4] Subjectivity: {subjectivity:.2f}")
    print(f"      (Topic choice is highly subjective)")
    
    # Calculate narrativity
    # п = weighted combination of factors
    п = (
        (topic_diversity / 20) * 0.25 +  # Normalize to 0-1
        career_flexibility * 0.30 +
        (effect_variance / 0.3) * 0.20 +  # Normalize to 0-1
        subjectivity * 0.25
    )
    
    # Clamp to [0, 1]
    п = min(1.0, max(0.0, п))
    
    print(f"\n{'─'*80}")
    print(f"NARRATIVITY (п) = {п:.3f}")
    print(f"{'─'*80}")
    
    if п > 0.7:
        print("HIGH narrativity domain - names should matter significantly")
    elif п > 0.4:
        print("MODERATE narrativity - names have some influence")
    else:
        print("LOW narrativity - fundamentals dominate")
    
    return п


def test_selection_effects(researchers):
    """
    TEST: Are researchers with fitting names OVERREPRESENTED in matching topics?
    
    This is the PRIMARY nominative determinism test!
    """
    print(f"\n{'='*80}")
    print("PRIMARY TEST: SELECTION EFFECTS (INTEREST/CHOICE)")
    print(f"{'='*80}\n")
    
    print("Question: Are researchers with name-field fit OVERREPRESENTED")
    print("in research topics that match their names?\n")
    
    # Categorize researchers by fit level
    high_fit = []  # fit > 50
    medium_fit = []  # fit 20-50
    low_fit = []  # fit < 20
    
    for name, data in researchers.items():
        fit = data.get('name_field_fit', {}).get('overall_fit', 0)
        
        if fit > 50:
            high_fit.append((name, data))
        elif fit >= 20:
            medium_fit.append((name, data))
        else:
            low_fit.append((name, data))
    
    print(f"[1/5] Researcher categorization:")
    print(f"      High fit (>50): {len(high_fit)} researchers")
    print(f"      Medium fit (20-50): {len(medium_fit)} researchers")
    print(f"      Low fit (<20): {len(low_fit)} researchers")
    
    # Expected vs observed
    total_researchers = len(researchers)
    
    # If names DON'T matter: high-fit should be rare (random matching)
    # Expected high-fit: ~5-10% (by chance)
    expected_high_fit_pct = 0.075  # 7.5% by chance
    observed_high_fit_pct = len(high_fit) / total_researchers
    
    print(f"\n[2/5] Proportion with high name-field fit:")
    print(f"      Expected (by chance): {expected_high_fit_pct*100:.1f}%")
    print(f"      Observed (actual): {observed_high_fit_pct*100:.1f}%")
    
    # Chi-square test
    expected_high = total_researchers * expected_high_fit_pct
    expected_low = total_researchers * (1 - expected_high_fit_pct)
    observed_high = len(high_fit)
    observed_low = len(medium_fit) + len(low_fit)
    
    chi2, p_value = stats.chisquare(
        [observed_high, observed_low],
        [expected_high, expected_low]
    )
    
    print(f"\n[3/5] Chi-square test for overrepresentation:")
    print(f"      χ² = {chi2:.3f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        if observed_high_fit_pct > expected_high_fit_pct:
            print(f"      ✓ SIGNIFICANT OVERREPRESENTATION!")
            print(f"      → Researchers ARE drawn to matching topics!")
        else:
            print(f"      ✓ SIGNIFICANT UNDERREPRESENTATION")
            print(f"      → Researchers AVOID matching topics!")
    else:
        print(f"      ✗ No significant difference from chance")
        print(f"      → Names don't affect topic selection")
    
    # Effect size
    odds_ratio = (observed_high / observed_low) / (expected_high / expected_low)
    
    print(f"\n[4/5] Effect size:")
    print(f"      Odds ratio = {odds_ratio:.2f}")
    print(f"      (How much more likely to study matching topics)")
    
    # Examples
    print(f"\n[5/5] Examples of high name-field fit researchers:")
    for name, data in high_fit[:5]:
        fit = data['name_field_fit']['overall_fit']
        topics = ', '.join(data.get('topics_studied', []))
        print(f"      {name}: fit={fit:.1f} studying {topics}")
    
    return {
        'observed_pct': observed_high_fit_pct,
        'expected_pct': expected_high_fit_pct,
        'chi2': chi2,
        'p_value': p_value,
        'odds_ratio': odds_ratio
    }


def test_by_specific_topics(researchers):
    """
    Test specific name-topic pairs (e.g., Dennis → dentists).
    """
    print(f"\n{'='*80}")
    print("SPECIFIC NAME-TOPIC ANALYSIS")
    print(f"{'='*80}\n")
    
    # Define name-topic pairs to test
    pairs = [
        ('Dennis', 'dentist'),
        ('Laura', 'lawyer'),
        ('Doctor', 'doctor'),
        ('Baker', 'baker'),
        ('Research', 'research'),
        ('Science', 'scien'),  # partial match
    ]
    
    print("Testing classic nominative pairs:\n")
    
    results = []
    
    for name_part, topic_part in pairs:
        # Find researchers with this name studying this topic
        matches = []
        total_with_name = 0
        total_with_topic = 0
        
        for researcher_name, data in researchers.items():
            has_name = name_part.lower() in researcher_name.lower()
            topics = ' '.join(data.get('topics_studied', [])).lower()
            has_topic = topic_part.lower() in topics
            
            if has_name:
                total_with_name += 1
                if has_topic:
                    matches.append(researcher_name)
            
            if has_topic:
                total_with_topic += 1
        
        if total_with_name > 0:
            match_rate = len(matches) / total_with_name
            
            # Expected rate (by chance)
            expected_rate = total_with_topic / len(researchers)
            
            print(f"  {name_part} → {topic_part}:")
            print(f"    Researchers with '{name_part}': {total_with_name}")
            print(f"    Studying {topic_part}: {len(matches)} ({match_rate*100:.1f}%)")
            print(f"    Expected by chance: {expected_rate*100:.1f}%")
            
            if len(matches) > 0:
                print(f"    ✓ MATCHES FOUND: {', '.join(matches)}")
                
                # Fisher exact test for small samples
                if total_with_name >= 3:
                    contingency = [
                        [len(matches), total_with_name - len(matches)],
                        [total_with_topic - len(matches), 
                         len(researchers) - total_with_name - (total_with_topic - len(matches))]
                    ]
                    odds_ratio_specific, p_fisher = stats.fisher_exact(contingency)
                    print(f"    Fisher exact p = {p_fisher:.4f}")
            else:
                print(f"    ✗ No matches")
            
            print()
            
            results.append({
                'name': name_part,
                'topic': topic_part,
                'matches': len(matches),
                'total_with_name': total_with_name,
                'match_rate': match_rate,
                'expected_rate': expected_rate
            })
    
    return results


def main():
    """Run complete selection effects analysis."""
    print(f"\n{'='*80}")
    print("META-NOMINATIVE SELECTION EFFECTS ANALYSIS")
    print(f"{'='*80}")
    print("\nResearch Questions:")
    print("  1. Are researchers with fitting names OVERREPRESENTED in matching topics?")
    print("  2. What is the narrativity (п) of this domain?")
    print("  3. Do specific name-topic pairs show effects?")
    print(f"{'='*80}\n")
    
    # Load data
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative'
    
    print("Loading data...")
    with open(data_dir / 'researchers_metadata.json') as f:
        data = json.load(f)
        researchers = data['researchers']
    
    with open(data_dir / 'papers_consolidated.json') as f:
        data = json.load(f)
        papers = data['papers']
    
    print(f"✓ Loaded {len(researchers)} researchers and {len(papers)} papers\n")
    
    # Calculate narrativity
    п = calculate_narrativity(researchers, papers)
    
    # Test selection effects
    selection_results = test_selection_effects(researchers)
    
    # Test specific pairs
    pair_results = test_by_specific_topics(researchers)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")
    
    print(f"NARRATIVITY: п = {п:.3f}")
    print(f"  → This is a {'HIGH' if п > 0.7 else 'MODERATE' if п > 0.4 else 'LOW'} narrativity domain\n")
    
    print(f"SELECTION EFFECT:")
    print(f"  Observed high-fit: {selection_results['observed_pct']*100:.1f}%")
    print(f"  Expected by chance: {selection_results['expected_pct']*100:.1f}%")
    print(f"  Odds ratio: {selection_results['odds_ratio']:.2f}")
    print(f"  p-value: {selection_results['p_value']:.4f}")
    
    if selection_results['p_value'] < 0.05:
        if selection_results['odds_ratio'] > 1:
            print(f"\n  🔥 SIGNIFICANT: Researchers ARE drawn to name-matching topics!")
            print(f"  → Nominative determinism affects researchers themselves!")
        else:
            print(f"\n  ⚠️  SIGNIFICANT: Researchers AVOID name-matching topics!")
            print(f"  → Awareness leads to topic avoidance!")
    else:
        print(f"\n  ✗ NULL: No selection effect detected")
        print(f"  → Topic choice appears independent of names")
    
    # Save results
    output = {
        'narrativity': float(п),
        'selection_effects': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                            for k, v in selection_results.items()},
        'specific_pairs': pair_results
    }
    
    with open(data_dir / 'selection_analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved results to: {data_dir / 'selection_analysis_results.json'}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

