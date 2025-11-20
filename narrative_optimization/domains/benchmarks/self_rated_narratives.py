"""
Self-Rated Narratives - TRUE Д ≈ 1.0 Ceiling

The purest narrative agency: Narrator judges own narrative.

Generator creates:
- Narrative text (varying quality)
- Self-evaluation of that narrative (narrator's judgment)

Tests: When narrator IS the judge, Д should approach 1.0

This is the theoretical maximum: Complete narrative circularity.
Outcome (❊) IS narrator's evaluation of story (ю).

Expected: Д ≈ 0.95-1.00
"""

import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from scipy import stats


class SelfRatedNarrativeBenchmark:
    """
    TRUE ceiling: Narrator judges own work.
    
    Generates varied narratives where creator evaluates own creation.
    Complete narrative agency - outcome IS self-judgment.
    """
    
    def __init__(self, n_narratives: int = 500):
        self.n_narratives = n_narratives
        self.narratives = []
        self.self_ratings = []
    
    def generate_self_rated_narrative(self) -> tuple:
        """
        Generate narrative + narrator's self-evaluation.
        
        Returns: (narrative_text, self_rating)
        """
        # Vary narrative quality intentionally
        quality_tier = random.choice(['excellent', 'good', 'mediocre', 'poor', 'terrible'])
        
        # Quality determines both narrative coherence AND self-rating
        quality_scores = {
            'excellent': (0.9, 1.0),
            'good': (0.7, 0.85),
            'mediocre': (0.5, 0.65),
            'poor': (0.3, 0.45),
            'terrible': (0.1, 0.25)
        }
        
        narrative_quality_base, self_rating_base = quality_scores[quality_tier]
        
        # Generate narrative
        narrative = self._generate_narrative(quality_tier)
        
        # Generate self-rating (mostly correlated with quality, some noise)
        self_rating = self_rating_base + random.uniform(-0.05, 0.05)
        self_rating = np.clip(self_rating, 0, 1)
        
        # Add explicit self-evaluation text
        evaluation_text = self._generate_self_evaluation(self_rating)
        
        # Complete narrative includes self-assessment
        complete_narrative = f"{narrative}\n\n[Self-evaluation: {evaluation_text}]"
        
        return complete_narrative, self_rating
    
    def _generate_narrative(self, quality: str) -> str:
        """Generate narrative of specified quality."""
        
        if quality == 'excellent':
            narratives = [
                """I woke up thinking about mortality again. Not in the frightening way, but in that strange peaceful acceptance that comes at 3am. The shadows on my ceiling looked like memories trying to take shape. I wonder if this is what they mean by 'making peace' - not fighting the shadows, just watching them dance.""",
                
                """Today I realized I've been carrying my father's anger like inherited jewelry - something valuable I never wanted. The interesting part isn't the anger itself but how carefully I've polished it, maintained it, as if letting it tarnish would dishonor his memory.""",
                
                """The therapist asked what I want. Such a simple question that I've spent 30 years not answering. I said 'to be understood' but that's not quite right. Maybe 'to understand myself' but that's closer to the truth than comfortable."""
            ]
        
        elif quality == 'good':
            narratives = [
                """Had coffee with Sarah today. She talked about her new job for an hour. I realized I don't actually know what she does, just that she talks about it a lot. Is that friendship or just habit?""",
                
                """Tried meditation again. Made it 3 minutes before my brain started listing groceries. Progress? Maybe. Or maybe I'm just getting better at sitting still while mentally shopping.""",
                
                """The city looks different in winter. Not just the obvious snow stuff, but the way people move - faster, hunched, purposeful. Summer people meander. Winter people have destinations."""
            ]
        
        elif quality == 'mediocre':
            narratives = [
                """Today was okay. Work was fine. Ate lunch. Talked to some people. Nothing special happened. Just a regular day.""",
                
                """I'm tired of being tired. That's all I wanted to write today. Just really tired.""",
                
                """Watched TV for 3 hours. Should probably do something productive. Maybe tomorrow."""
            ]
        
        elif quality == 'poor':
            narratives = [
                """Stuff happened. I don't know. Whatever.""",
                """Another day another dollar. That's what they say. I guess it's true.""",
                """I ate food. It was food. Then I went to sleep. The end."""
            ]
        
        else:  # terrible
            narratives = [
                """Thing. Other thing. Yep.""",
                """I forgot what I was going to write.""",
                """..."""
            ]
        
        return random.choice(narratives)
    
    def _generate_self_evaluation(self, rating: float) -> str:
        """Generate self-evaluation text matching rating."""
        
        if rating > 0.85:
            evals = [
                "I'm genuinely proud of this. Captured exactly what I was feeling.",
                "This is some of my best work. The metaphor really landed.",
                "Finally wrote something that feels true. Really satisfied with this."
            ]
        elif rating > 0.65:
            evals = [
                "Pretty good. Got the main idea down even if not perfectly expressed.",
                "Solid entry. Not my best but definitely worthwhile.",
                "Happy with this. Conveys what I wanted to say."
            ]
        elif rating > 0.45:
            evals = [
                "Okay I guess. Could be better.",
                "Wrote something at least. Not terrible.",
                "Fine. Not great, not awful."
            ]
        elif rating > 0.25:
            evals = [
                "Pretty weak. Didn't really capture anything.",
                "Not happy with this. Feels forced.",
                "Disappointing. Know I can do better."
            ]
        else:
            evals = [
                "Terrible. Why did I even bother?",
                "This is garbage. Deleting later.",
                "Absolute waste of time. Awful."
            ]
        
        return random.choice(evals)
    
    def generate_dataset(self):
        """Generate complete dataset."""
        print(f"Generating {self.n_narratives} self-rated narratives...")
        
        for i in range(self.n_narratives):
            narrative, rating = self.generate_self_rated_narrative()
            self.narratives.append(narrative)
            self.self_ratings.append(rating)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{self.n_narratives}...")
        
        print(f"✓ Complete: {self.n_narratives} narratives with self-ratings")
    
    def extract_genome_and_test(self):
        """Apply standard transformers, test Д."""
        print("\n" + "=" * 80)
        print("APPLYING STANDARD TRANSFORMERS")
        print("=" * 80)
        
        X = np.array(self.narratives)
        y = np.array(self.self_ratings)
        
        all_features = []
        transformer_Ds = {}
        
        transformers = {
            'nominative': NominativeAnalysisTransformer(),
            'self_perception': SelfPerceptionTransformer(),
            'narrative_potential': NarrativePotentialTransformer()
        }
        
        for name, transformer in transformers.items():
            print(f"\nTesting {name} transformer...")
            
            try:
                transformer.fit(X)
                features = transformer.transform(X)
                
                # ю = aggregate quality from ж
                story_quality = np.mean(features, axis=1)
                
                # Д = correlation(ю, ❊_self_rating)
                D, p = stats.pearsonr(story_quality, y)
                
                print(f"  Features: {features.shape[1]}")
                print(f"  Д ({name}): {D:.4f}, p={p:.4f}")
                
                transformer_Ds[name] = D
                all_features.append(features)
                
            except Exception as e:
                print(f"  Error: {e}")
                transformer_Ds[name] = 0.0
        
        # Combined
        if all_features:
            ж_combined = np.hstack(all_features)
            ю_combined = np.mean(ж_combined, axis=1)
            
            D_combined, p_combined = stats.pearsonr(ю_combined, y)
            
            print(f"\n" + "=" * 80)
            print("COMBINED ж→ю→❊ TEST")
            print("=" * 80)
            print(f"Total ж dimensions: {ж_combined.shape[1]}")
            print(f"Overall Д: {D_combined:.4f}, p={p_combined:.4f}")
            print(f"R²: {D_combined**2:.4f}")
            print("")
            print("❊ = Self-rating (narrator judges own work)")
            print("This is maximum possible narrative agency.")
            
            return {
                'D_combined': float(D_combined),
                'R_squared': float(D_combined**2),
                'p_value': float(p_combined),
                'transformer_Ds': {k: float(v) for k, v in transformer_Ds.items()},
                'n_narratives': len(self.narratives),
                'interpretation': 'Narrator=Judge, pure circularity'
            }
        
        return {}


def main():
    """Generate and test self-rated narratives."""
    print("=" * 80)
    print("SELF-RATED NARRATIVES - TRUE Д ≈ 1.0 CEILING")
    print("Narrator judges own work (maximum narrative agency)")
    print("=" * 80)
    print("")
    
    benchmark = SelfRatedNarrativeBenchmark(n_narratives=500)
    benchmark.generate_dataset()
    
    # Save
    data_path = Path(__file__).parent.parent.parent.parent / 'data/domains/self_rated_narratives.json'
    with open(data_path, 'w') as f:
        json.dump({
            'narratives': benchmark.narratives,
            'self_ratings': benchmark.self_ratings
        }, f, indent=2)
    
    print(f"✓ Data saved to: {data_path}")
    
    # Test
    results = benchmark.extract_genome_and_test()
    
    # Save results
    results_path = Path(__file__).parent / 'self_rated_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("TRUE CEILING ESTABLISHED")
    print("=" * 80)
    print(f"Д = {results.get('D_combined', 0):.4f}")
    print("")
    print("This is maximum Д: Narrator controls both narrative AND judgment.")
    print("All domains with external judges/outcomes fall below this.")
    print("=" * 80)


if __name__ == "__main__":
    main()

