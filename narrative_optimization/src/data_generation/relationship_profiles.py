"""
Relationship Profile Data Generator

Generate synthetic dating profiles with controlled narrative dimensions
for testing ensemble, relational, and potential transformers.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import random


class RelationshipProfileGenerator:
    """
    Generate synthetic relationship profiles with controlled narrative patterns.
    
    Tests whether ensemble diversity, relational complementarity, and narrative
    potential predict compatibility better than content alone.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Narrative element pools
        self.interests = [
            'hiking', 'reading', 'cooking', 'travel', 'music', 'art', 'sports',
            'photography', 'gaming', 'yoga', 'meditation', 'dancing', 'writing',
            'films', 'theater', 'volunteering', 'technology', 'science', 'nature',
            'fitness', 'foodie', 'adventure', 'learning', 'creativity', 'animals'
        ]
        
        self.values = [
            'honesty', 'kindness', 'ambition', 'humor', 'intelligence', 'creativity',
            'loyalty', 'independence', 'empathy', 'authenticity', 'curiosity',
            'passion', 'growth', 'balance', 'adventure', 'stability', 'spontaneity'
        ]
        
        self.traits_positive = [
            'optimistic', 'thoughtful', 'adventurous', 'caring', 'ambitious',
            'creative', 'authentic', 'passionate', 'curious', 'confident',
            'warm', 'genuine', 'driven', 'easygoing', 'thoughtful'
        ]
        
        self.future_goals = [
            'build a career', 'travel the world', 'start a family', 'learn new skills',
            'make a difference', 'achieve work-life balance', 'pursue passions',
            'grow personally', 'create meaningful connections', 'explore possibilities'
        ]
    
    def generate_profile(
        self,
        ensemble_diversity: float,  # 0-1, how varied are interests
        growth_mindset: float,  # 0-1, future vs past orientation
        agency_level: float,  # 0-1, active vs passive voice
        narrative_style: str = 'balanced'  # 'analytical', 'emotional', 'balanced'
    ) -> str:
        """
        Generate a profile with controlled narrative dimensions.
        
        Parameters
        ----------
        ensemble_diversity : float
            0 (narrow interests) to 1 (broad interests)
        growth_mindset : float
            0 (past-focused) to 1 (future-focused)
        agency_level : float
            0 (passive) to 1 (active)
        narrative_style : str
            Communication style
        
        Returns
        -------
        profile : str
            Generated profile text
        """
        profile_parts = []
        
        # 1. Opening (voice and agency)
        if agency_level > 0.7:
            openers = [
                "I'm passionate about",
                "I love exploring",
                "I actively pursue",
                "I'm building a life around"
            ]
        elif agency_level > 0.4:
            openers = [
                "I enjoy",
                "I'm interested in",
                "I appreciate",
                "I value"
            ]
        else:
            openers = [
                "I'm someone who likes",
                "You'll find me",
                "I'm drawn to",
                "I tend to enjoy"
            ]
        
        opener = random.choice(openers)
        
        # 2. Interests (ensemble diversity)
        n_interests = int(3 + ensemble_diversity * 5)  # 3-8 interests
        selected_interests = random.sample(self.interests, n_interests)
        
        interests_text = f"{opener} {', '.join(selected_interests[:-1])}, and {selected_interests[-1]}."
        profile_parts.append(interests_text)
        
        # 3. Values and traits
        n_values = int(2 + ensemble_diversity * 3)
        selected_values = random.sample(self.values, n_values)
        
        values_text = f"I value {', '.join(selected_values)}."
        profile_parts.append(values_text)
        
        # 4. Self-description with growth mindset
        trait = random.choice(self.traits_positive)
        
        if growth_mindset > 0.7:
            # Future-oriented, growth language
            self_desc = random.choice([
                f"I'm {trait} and always looking to grow and learn.",
                f"I'm becoming more {trait} as I develop and evolve.",
                f"I'm {trait} and excited about future possibilities.",
                f"I'm continuously developing my {trait} side."
            ])
        elif growth_mindset > 0.4:
            # Balanced
            self_desc = f"I'm {trait} and open to new experiences."
        else:
            # Past-focused, static
            self_desc = random.choice([
                f"I've always been {trait}.",
                f"I'm {trait}, just like I've always been.",
                f"I remain {trait}."
            ])
        
        profile_parts.append(self_desc)
        
        # 5. Future goals (narrative potential)
        if growth_mindset > 0.6:
            n_goals = 2
            selected_goals = random.sample(self.future_goals, n_goals)
            
            goal_text = f"Looking forward to {selected_goals[0]} and {selected_goals[1]}."
            profile_parts.append(goal_text)
        
        # 6. What seeking (relational)
        if narrative_style == 'analytical':
            seeking = "Seeking someone who shares similar interests and values."
        elif narrative_style == 'emotional':
            seeking = "Hoping to find a genuine connection with someone special."
        else:
            seeking = "Looking for someone to share adventures and build meaningful experiences together."
        
        profile_parts.append(seeking)
        
        return " ".join(profile_parts)
    
    def generate_compatibility_label(
        self,
        profile_a_params: Dict[str, float],
        profile_b_params: Dict[str, float]
    ) -> float:
        """
        Generate compatibility score based on narrative complementarity.
        
        Rules for compatibility:
        - Ensemble: Moderate diversity difference (not too similar, not too different)
        - Growth: Aligned future orientation (both high or both moderate)
        - Agency: Can be complementary or similar
        
        Returns compatibility score 0-1
        """
        # Ensemble complementarity (optimal difference around 0.3-0.4)
        ensemble_diff = abs(profile_a_params['ensemble_diversity'] - profile_b_params['ensemble_diversity'])
        ensemble_score = 1 - abs(ensemble_diff - 0.35) / 0.65  # Peak at 0.35 difference
        ensemble_score = max(0, min(1, ensemble_score))
        
        # Growth alignment (both should be similar - either both high or both low)
        growth_diff = abs(profile_a_params['growth_mindset'] - profile_b_params['growth_mindset'])
        growth_score = 1 - growth_diff
        
        # Agency complementarity (can be different)
        agency_diff = abs(profile_a_params['agency_level'] - profile_b_params['agency_level'])
        agency_score = 0.7 + 0.3 * (1 - agency_diff)  # Bonus for similarity, but not required
        
        # Overall compatibility (weighted combination)
        compatibility = (
            0.4 * ensemble_score +
            0.4 * growth_score +
            0.2 * agency_score
        )
        
        return compatibility
    
    def generate_dataset(
        self,
        n_profiles: int = 500,
        n_pairs: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate complete dataset of profiles and compatibility labels.
        
        Parameters
        ----------
        n_profiles : int
            Number of individual profiles to generate
        n_pairs : int
            Number of profile pairs with compatibility labels
        
        Returns
        -------
        dataset : dict
            {
                'profiles': [profile_texts],
                'profile_params': [parameters used],
                'pairs': [(idx_a, idx_b)],
                'compatibility': [scores],
                'binary_compatible': [labels]
            }
        """
        print(f"Generating {n_profiles} profiles...")
        
        profiles = []
        profile_params = []
        
        for i in range(n_profiles):
            # Random parameters
            params = {
                'ensemble_diversity': np.random.beta(2, 2),  # Centered around 0.5
                'growth_mindset': np.random.beta(3, 2),  # Skewed toward high
                'agency_level': np.random.beta(2, 2),
                'narrative_style': random.choice(['analytical', 'emotional', 'balanced'])
            }
            
            profile = self.generate_profile(**params)
            profiles.append(profile)
            profile_params.append(params)
        
        print(f"✓ Generated {n_profiles} profiles")
        print(f"Generating {n_pairs} pairs with compatibility labels...")
        
        pairs = []
        compatibility_scores = []
        
        for _ in range(n_pairs):
            idx_a = np.random.randint(0, n_profiles)
            idx_b = np.random.randint(0, n_profiles)
            
            if idx_a == idx_b:
                continue
            
            params_a = profile_params[idx_a]
            params_b = profile_params[idx_b]
            
            compatibility = self.generate_compatibility_label(params_a, params_b)
            
            pairs.append((idx_a, idx_b))
            compatibility_scores.append(compatibility)
        
        # Binary labels (compatible if score > 0.6)
        binary_compatible = [1 if score > 0.6 else 0 for score in compatibility_scores]
        
        print(f"✓ Generated {len(pairs)} pairs")
        print(f"  Compatible pairs: {sum(binary_compatible)} ({sum(binary_compatible)/len(binary_compatible)*100:.1f}%)")
        
        return {
            'profiles': profiles,
            'profile_params': profile_params,
            'pairs': pairs,
            'compatibility': compatibility_scores,
            'binary_compatible': binary_compatible
        }
    
    def save_dataset(self, dataset: Dict[str, Any], output_dir: str):
        """Save generated dataset."""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save profiles
        with open(output_path / 'profiles.txt', 'w') as f:
            for profile in dataset['profiles']:
                f.write(profile + '\n')
        
        # Save metadata
        metadata = {
            'n_profiles': len(dataset['profiles']),
            'n_pairs': len(dataset['pairs']),
            'compatibility_stats': {
                'mean': float(np.mean(dataset['compatibility'])),
                'std': float(np.std(dataset['compatibility'])),
                'min': float(np.min(dataset['compatibility'])),
                'max': float(np.max(dataset['compatibility']))
            },
            'compatible_rate': sum(dataset['binary_compatible']) / len(dataset['binary_compatible'])
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save pairs and labels
        np.save(output_path / 'pairs.npy', np.array(dataset['pairs']))
        np.save(output_path / 'compatibility_scores.npy', np.array(dataset['compatibility']))
        np.save(output_path / 'binary_labels.npy', np.array(dataset['binary_compatible']))
        
        print(f"✓ Dataset saved to {output_path}")


if __name__ == '__main__':
    # Demo
    print("Relationship Profile Generator Demo\n")
    
    generator = RelationshipProfileGenerator()
    
    # Generate example profiles
    print("Example Profiles:\n")
    
    profile1 = generator.generate_profile(
        ensemble_diversity=0.7,
        growth_mindset=0.8,
        agency_level=0.9,
        narrative_style='balanced'
    )
    print("High diversity, growth-oriented, high agency:")
    print(profile1)
    print()
    
    profile2 = generator.generate_profile(
        ensemble_diversity=0.3,
        growth_mindset=0.3,
        agency_level=0.4,
        narrative_style='analytical'
    )
    print("Low diversity, past-focused, moderate agency:")
    print(profile2)
    print()
    
    # Generate full dataset
    print("Generating complete dataset...")
    dataset = generator.generate_dataset(n_profiles=200, n_pairs=500)
    
    generator.save_dataset(dataset, 'data/synthetic/relationships_generated')
    
    print("\n✅ Relationship dataset generation complete!")

