"""
Coin Flipper Domain - Д ≈ 0 Benchmark

Tests MAXIMUM narrative variables against pure randomness.
Establishes floor: Can ANY narrative dimension affect a coin flip?

Narrative variables tested (40+):
- Prediction confidence, complexity, framing
- Flipper identity, credentials, superstitions
- Coin history, name, description
- Context (stakes, audience, location, timing)
- Meta-narratives (self-awareness, bias admission)
- Emotional investment
- Reasoning narratives
- Comparison framing
"""

import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


class CoinFlipGenerator:
    """
    Generates coin flips with MAXIMUM narrative variation.
    
    Tests limits of narrativity: Can ANY narrative affect outcome?
    Expected: Д ≈ 0.001 (all narrative variables fail)
    """
    
    def __init__(self, n_flips: int = 1000):
        self.n_flips = n_flips
        self.flips = []
        
    def generate_flip_with_full_narrative(self, flip_id: int) -> Dict:
        """
        Generate one flip with MAXIMUM narrative dimensions.
        
        Tests every conceivable narrative variable.
        """
        # Actual outcome (pure randomness)
        outcome = random.choice([0, 1])  # 0=tails, 1=heads
        prediction = random.choice([0, 1])
        
        # Generate exhaustive narrative
        narrative = {
            # PREDICTION NARRATIVE
            'prediction_text': self._generate_prediction_narrative(prediction),
            'prediction_confidence': random.uniform(0, 1),
            'prediction_complexity': random.randint(1, 10),  # Words used
            'certainty_markers': random.choice(['definitely', 'probably', 'maybe', 'possibly', 'perhaps']),
            'emotional_investment': random.choice(['desperately need', 'hope for', 'expect', 'predict', 'guess']),
            'temporal_framing': random.choice(['will be', 'is going to be', 'shall be', 'must be']),
            
            # REASONING NARRATIVE
            'reasoning_type': random.choice(['physics', 'intuition', 'divine', 'superstition', 'pattern', 'random']),
            'reasoning_complexity': random.randint(0, 5),  # Paragraphs of explanation
            'uses_statistics': random.choice([True, False]),
            'cites_past_flips': random.choice([True, False]),
            'mythological_framing': random.choice([True, False]),
            
            # FLIPPER NARRATIVE
            'flipper_identity': random.choice(['professional', 'amateur', 'expert', 'novice', 'lucky person']),
            'flipper_credentials': random.choice([None, 'PhD in randomness', '1000 flips experience', 'Vegas dealer']),
            'flipper_demographics': {
                'age_mentioned': random.choice([True, False]),
                'gender_mentioned': random.choice([True, False]),
                'background_story': random.choice([True, False])
            },
            'flipper_superstitions': random.choice([None, 'always heads', 'full moon method', 'lucky technique']),
            'flipper_success_rate_claim': random.uniform(0.3, 0.9),
            
            # COIN NARRATIVE
            'coin_named': random.choice([True, False]),
            'coin_name': random.choice(['Lucky Penny', 'Zeus Coin', 'Old Quarter', 'The Flipper']),
            'coin_history': random.choice([None, 'family heirloom', 'found on street', 'from Vegas', 'ancient origin']),
            'coin_description_quality': random.uniform(0, 1),
            'coin_physical_narrative': random.choice(['shiny new', 'worn old', 'pristine mint', 'battle-scarred']),
            'coin_weight_mentioned': random.choice([True, False]),
            'coin_mythology': random.choice([None, 'heads is Zeus', 'tails is underworld', 'blessed coin']),
            
            # CONTEXT NARRATIVE
            'stakes_described': random.choice(['life-changing', 'important', 'casual', 'meaningless']),
            'audience_mentioned': random.choice([None, 'millions watching', 'alone', 'with friends', 'for science']),
            'location_described': random.choice([None, 'sacred ground', 'parking lot', 'laboratory', 'bedroom']),
            'timing_narrative': random.choice([None, 'full moon', 'lucky hour', 'Tuesday', 'solstice']),
            'weather_mentioned': random.choice([True, False]),
            'cosmic_alignment_claimed': random.choice([True, False]),
            
            # META-NARRATIVE
            'self_awareness': random.choice(['knows its random', 'believes in control', 'agnostic', 'superstitious']),
            'bias_admission': random.choice([None, 'I always pick heads', 'I want tails', 'No preference']),
            'pattern_claim': random.choice([None, 'been on heads streak', 'alternating pattern', 'no pattern']),
            'contradiction': random.choice([True, False]),  # "It's random but I think..."
            'humility': random.choice(['very humble', 'confident', 'arrogant', 'uncertain']),
            
            # COMPARATIVE FRAMING
            'compares_to_past': random.choice([True, False]),
            'compares_to_others': random.choice([True, False]),
            'references_statistics': random.choice([True, False]),
            'claims_special_knowledge': random.choice([True, False]),
            
            # LINGUISTIC FEATURES
            'vocabulary_level': random.uniform(0, 1),
            'sentence_complexity': random.randint(1, 5),
            'uses_metaphor': random.choice([True, False]),
            'uses_alliteration': random.choice([True, False]),
            'rhythm_quality': random.uniform(0, 1),
            
            # EMOTIONAL NARRATIVE
            'expresses_anxiety': random.choice([True, False]),
            'expresses_hope': random.choice([True, False]),
            'expresses_indifference': random.choice([True, False]),
            'emotional_complexity': random.randint(0, 5)
        }
        
        flip_record = {
            'flip_id': flip_id,
            'prediction': prediction,
            'outcome': outcome,
            'correct': int(prediction == outcome),
            'narrative': narrative,
            'prediction_narrative_text': self._construct_full_narrative(prediction, narrative)
        }
        
        return flip_record
    
    def _generate_prediction_narrative(self, prediction: int) -> str:
        """Generate actual text prediction with narrative."""
        side = "heads" if prediction == 1 else "tails"
        templates = [
            f"I predict {side}",
            f"It will definitely be {side}",
            f"My intuition says {side}",
            f"Based on my experience, {side} is coming",
            f"The cosmic forces align for {side}",
            f"I have a feeling about {side}",
        ]
        return random.choice(templates)
    
    def _construct_full_narrative(self, prediction: int, narrative: Dict) -> str:
        """Construct complete narrative text from all variables."""
        side = "heads" if prediction == 1 else "tails"
        
        parts = []
        
        # Identity
        if narrative['flipper_identity']:
            parts.append(f"As a {narrative['flipper_identity']} coin flipper")
        
        # Credentials
        if narrative['flipper_credentials']:
            parts.append(f"with credentials in {narrative['flipper_credentials']}")
        
        # Coin description
        if narrative['coin_named']:
            parts.append(f"I will flip {narrative['coin_name']}, a {narrative['coin_physical_narrative']} coin")
        
        # History
        if narrative['coin_history']:
            parts.append(f"which is {narrative['coin_history']}")
        
        # Prediction with confidence
        parts.append(f"I {narrative['certainty_markers']} {narrative['emotional_investment']} that it {narrative['temporal_framing']} {side}")
        
        # Reasoning
        parts.append(f"because of {narrative['reasoning_type']}")
        
        # Meta
        if narrative['self_awareness'] == 'knows its random':
            parts.append("even though I know it's random")
        
        # Stakes
        parts.append(f"This is {narrative['stakes_described']}")
        
        return ". ".join(parts) + "."
    
    def generate_all_flips(self) -> List[Dict]:
        """Generate complete dataset."""
        print(f"Generating {self.n_flips} coin flips with MAXIMUM narrative variation...")
        
        for i in range(self.n_flips):
            flip = self.generate_flip_with_full_narrative(i)
            self.flips.append(flip)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{self.n_flips} flips...")
        
        print(f"✓ Complete: {self.n_flips} flips with 40+ narrative dimensions each")
        
        return self.flips
    
    def save_dataset(self, output_path: str):
        """Save complete dataset."""
        with open(output_path, 'w') as f:
            json.dump(self.flips, f, indent=2)
        
        print(f"✓ Saved to: {output_path}")
    
    def compute_statistics(self) -> Dict:
        """Compute dataset statistics."""
        if not self.flips:
            return {}
        
        accuracy = np.mean([f['correct'] for f in self.flips])
        
        # Test each narrative variable
        narrative_effects = {}
        
        for key in self.flips[0]['narrative'].keys():
            # Extract values for this narrative variable
            try:
                values = [f['narrative'][key] for f in self.flips]
                outcomes = [f['correct'] for f in self.flips]
                
                # If numeric, test correlation
                if isinstance(values[0], (int, float)):
                    from scipy import stats
                    r, p = stats.pearsonr(values, outcomes)
                    narrative_effects[key] = {'r': r, 'p': p}
                elif isinstance(values[0], bool):
                    # Test if boolean affects accuracy
                    true_acc = np.mean([outcomes[i] for i in range(len(values)) if values[i]])
                    false_acc = np.mean([outcomes[i] for i in range(len(values)) if not values[i]])
                    narrative_effects[key] = {'diff': true_acc - false_acc}
            except:
                pass
        
        return {
            'total_flips': len(self.flips),
            'overall_accuracy': accuracy,
            'expected_accuracy': 0.5,
            'deviation': accuracy - 0.5,
            'narrative_effects': narrative_effects
        }


def main():
    """Generate coin flip benchmark domain."""
    print("=" * 80)
    print("COIN FLIP DOMAIN - Д ≈ 0 BENCHMARK")
    print("Testing limits: Can ANY narrative variable affect outcome?")
    print("=" * 80)
    print("")
    
    generator = CoinFlipGenerator(n_flips=1000)
    flips = generator.generate_all_flips()
    
    # Save
    output_path = Path(__file__).parent.parent.parent.parent / 'data/domains/coin_flips_benchmark.json'
    generator.save_dataset(str(output_path))
    
    # Analyze
    stats = generator.compute_statistics()
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total flips: {stats['total_flips']}")
    print(f"Accuracy: {stats['overall_accuracy']:.3f}")
    print(f"Expected: {stats['expected_accuracy']:.3f}")
    print(f"Deviation: {stats['deviation']:.4f}")
    
    print("\nNarrative effects (top correlations):")
    effects = [(k, v.get('r', 0)) for k, v in stats['narrative_effects'].items() if 'r' in v]
    effects.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for key, r in effects[:10]:
        print(f"  {key}: r={r:.4f}")
    
    print("\n" + "=" * 80)
    print("EXPECTED: ALL correlations ≈ 0.00 (narrative has zero causal power)")
    print("If ANY r > 0.10, narrative has found a crack in physics")
    print("=" * 80)


if __name__ == "__main__":
    main()

