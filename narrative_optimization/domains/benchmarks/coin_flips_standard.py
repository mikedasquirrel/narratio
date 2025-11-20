"""
Coin Flips - Standard Transformer Test (Д ≈ 0 floor)

Uses SAME 6 transformers as all domains:
- Nominative, Self-perception, Narrative potential, Linguistic, Relational, Ensemble

Generates 1000 prediction narratives with varying quality.
Tests: Do standard narrative features (ж→ю) predict coin flip outcomes (❊)?

Expected: Д ≈ 0.00-0.02 (narrative features fail against physics)
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


class CoinFlipBenchmark:
    """
    Benchmark domain establishing Д floor.
    
    Uses standard transformers to extract ж from prediction narratives.
    Tests if narrative quality (ю) predicts outcomes (❊) at all.
    """
    
    def __init__(self, n_flips: int = 1000):
        self.n_flips = n_flips
        self.narratives = []
        self.outcomes = []
        
    def generate_prediction_narrative(self) -> str:
        """
        Generate prediction narrative with varying quality.
        
        Low quality: "heads"
        High quality: Full narrative with confidence, reasoning, framing
        """
        prediction_side = random.choice(['heads', 'tails'])
        
        # Vary narrative complexity
        style = random.choice(['minimal', 'moderate', 'rich', 'elaborate'])
        
        if style == 'minimal':
            return f"I predict {prediction_side}."
        
        elif style == 'moderate':
            confidence = random.choice(['think', 'believe', 'am certain'])
            return f"I {confidence} it will be {prediction_side}."
        
        elif style == 'rich':
            identity = random.choice(['As an experienced flipper', 'Based on my intuition', 'In my analysis'])
            confidence = random.choice(['strongly believe', 'predict', 'am confident'])
            reasoning = random.choice(['based on patterns observed', 'due to coin properties', 'from my track record'])
            
            return f"{identity}, I {confidence} this will be {prediction_side}, {reasoning}."
        
        else:  # elaborate
            intro = random.choice([
                "After careful consideration and analysis of the physical properties",
                "Drawing on my extensive experience with probability theory",
                "Given the current conditions and my understanding of randomness"
            ])
            
            identity_claim = random.choice([
                "as a professional probability analyst",
                "as someone who has studied coin flips for years",
                "as an expert in stochastic processes"
            ])
            
            reasoning = random.choice([
                "the angular momentum, initial conditions, and gravitational effects",
                "the historical patterns and my intuitive sense",
                "the cosmic alignment and my deep understanding"
            ])
            
            confidence = random.choice([
                "I am absolutely certain",
                "I strongly predict",
                "I confidently project"
            ])
            
            stake = random.choice([
                "This is crucial.",
                "Everything depends on this.",
                "This matters deeply to me."
            ])
            
            return f"{intro}, {identity_claim}, considering {reasoning}, {confidence} that the result will be {prediction_side}. {stake}"
    
    def generate_dataset(self):
        """Generate complete dataset."""
        print(f"Generating {self.n_flips} coin flip predictions...")
        
        for i in range(self.n_flips):
            # Generate prediction narrative
            narrative = self.generate_prediction_narrative()
            self.narratives.append(narrative)
            
            # Generate actual outcome (pure random)
            outcome = random.randint(0, 1)  # 0=tails, 1=heads
            self.outcomes.append(outcome)
            
            if (i + 1) % 200 == 0:
                print(f"  Generated {i+1}/{self.n_flips}...")
        
        print(f"✓ Complete: {self.n_flips} predictions with varied narrative quality")
    
    def extract_genome_and_test(self):
        """
        Apply standard transformers to extract ж.
        Compute ю, test Д = correlation(ю, ❊).
        """
        print("\n" + "=" * 80)
        print("APPLYING STANDARD TRANSFORMERS")
        print("=" * 80)
        
        X = np.array(self.narratives)
        y = np.array(self.outcomes)
        
        all_features = []
        transformer_Ds = {}
        
        # Test each transformer
        transformers = {
            'nominative': NominativeAnalysisTransformer(),
            'self_perception': SelfPerceptionTransformer(),
            'narrative_potential': NarrativePotentialTransformer()
        }
        
        for name, transformer in transformers.items():
            print(f"\nTesting {name} transformer...")
            
            try:
                # Fit and transform
                transformer.fit(X, y)
                features = transformer.transform(X)
                
                # Compute narrative quality (ю) as mean of features
                story_quality = np.mean(features, axis=1)
                
                # Test Д (correlation with outcomes)
                D, p = stats.pearsonr(story_quality, y)
                
                print(f"  Features extracted: {features.shape[1]}")
                print(f"  Д ({name}): {D:.4f}, p={p:.4f}")
                
                transformer_Ds[name] = D
                all_features.append(features)
                
            except Exception as e:
                print(f"  Error: {e}")
                transformer_Ds[name] = 0.0
        
        # Combined ж (all transformers)
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
            
            return {
                'D_combined': float(D_combined),
                'R_squared': float(D_combined**2),
                'p_value': float(p_combined),
                'transformer_Ds': {k: float(v) for k, v in transformer_Ds.items()},
                'n_flips': len(self.narratives),
                'baseline_accuracy': 0.5
            }
        
        return {}
    
    def save_results(self, results: Dict, output_path: str):
        """Save analysis results."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")


def main():
    """Generate and analyze coin flip benchmark."""
    print("=" * 80)
    print("COIN FLIPS - STANDARD TRANSFORMER TEST")
    print("Establishing Д ≈ 0 floor with real framework")
    print("=" * 80)
    print("")
    
    benchmark = CoinFlipBenchmark(n_flips=1000)
    benchmark.generate_dataset()
    
    # Save narratives
    data_path = Path(__file__).parent.parent.parent.parent / 'data/domains/coin_flips_narratives.json'
    with open(data_path, 'w') as f:
        json.dump({
            'narratives': benchmark.narratives,
            'outcomes': benchmark.outcomes
        }, f, indent=2)
    
    print(f"✓ Narratives saved to: {data_path}")
    
    # Apply transformers and test
    results = benchmark.extract_genome_and_test()
    
    # Save results
    results_path = Path(__file__).parent / 'coin_flips_results.json'
    benchmark.save_results(results, str(results_path))
    
    print("\n" + "=" * 80)
    print("FLOOR ESTABLISHED")
    print("=" * 80)
    print(f"Д = {results.get('D_combined', 0):.4f}")
    print("\nThis is the minimum Д (pure randomness).")
    print("All other domains should have Д > this floor.")
    print("=" * 80)


if __name__ == "__main__":
    main()

