"""
Three-Force Calculator for All Domains

Calculates:
- ة (nominative gravity) - Pull toward name-matching
- θ (awareness resistance) - Free will resistance
- λ (fundamental constraints) - Physics/training barriers

For every domain in our registry.
"""

import json
from pathlib import Path
from typing import Dict
import numpy as np


class ThreeForceCalculator:
    """Calculate three forces for all domains."""
    
    def __init__(self):
        """Initialize with domain characteristics."""
        # Domain-specific data for calculation
        self.domain_data = self._initialize_domain_data()
    
    def _initialize_domain_data(self) -> Dict:
        """Initialize known data about each domain."""
        return {
            'crypto': {
                'narrativity': 0.65,
                'observed_r': 0.65,
                'observed_bridge': 0.423,
                'time_period': '2017-2018',
                
                # For θ calculation
                'population_education': 0.60,  # Mix of educated/retail
                'field_studies_nominative': 0.20,  # Some awareness
                'name_obviousness': 0.90,  # Very obvious (names ARE the brand)
                'social_cost': 0.10,  # No stigma to name-matching
                
                # For λ calculation
                'training_years': 0.0,  # Zero training needed
                'aptitude_threshold': 0.15,  # Basic tech literacy
                'economic_barrier': 0.10,  # Small investment
                
                # For ة calculation
                'branding_importance': 0.95,  # Names are everything
                'semantic_field_richness': 0.85,  # Many name-value associations
                'cultural_salience': 0.80,  # Names culturally important
            },
            
            'nba': {
                'narrativity': 0.15,
                'observed_r': 0.20,
                'observed_bridge': 0.018,
                
                # θ
                'population_education': 0.50,
                'field_studies_nominative': 0.40,  # Athletes know "bulletin board"
                'name_obviousness': 0.60,  # Team narratives noticeable
                'social_cost': 0.20,  # Some stigma to "just narratives"
                
                # λ
                'training_years': 1.0,  # Lifetime of training
                'aptitude_threshold': 0.95,  # Elite physical talent
                'economic_barrier': 0.30,
                
                # ة
                'branding_importance': 0.50,  # Team brands matter
                'semantic_field_richness': 0.40,  # Some team name patterns
                'cultural_salience': 0.60,  # Sports culture strong
            },
            
            'oscars': {
                'narrativity': 0.88,
                'observed_r': 0.95,  # Cast names near-perfect
                'observed_bridge': 0.837,
                
                # θ
                'population_education': 0.90,  # Academy members highly educated
                'field_studies_nominative': 0.30,  # Some awareness
                'name_obviousness': 0.95,  # Star power is obvious
                'social_cost': 0.05,  # NO stigma - judging prestige IS the task
                
                # λ
                'training_years': 0.80,  # Film school/experience
                'aptitude_threshold': 0.70,  # Talent required
                'economic_barrier': 0.50,  # High production costs
                
                # ة
                'branding_importance': 0.98,  # Names/prestige are the point
                'semantic_field_richness': 0.90,  # Rich star system
                'cultural_salience': 0.95,  # Hollywood prestige culture
                
                # Special flag
                'is_prestige_domain': True  # Equation flips
            },
            
            'aviation': {
                'narrativity': 0.05,
                'observed_r': 0.00,
                'observed_bridge': 0.000,
                
                # θ
                'population_education': 0.95,  # Pilots/engineers highly trained
                'field_studies_nominative': 0.00,
                'name_obviousness': 0.20,  # Names shouldn't matter
                'social_cost': 0.50,  # HIGH stigma to name-based thinking
                
                # λ
                'training_years': 0.90,  # Extensive training
                'aptitude_threshold': 0.90,  # High technical skills
                'economic_barrier': 0.70,  # Very expensive
                
                # ة
                'branding_importance': 0.10,  # Names barely matter
                'semantic_field_richness': 0.05,  # Few name-safety associations
                'cultural_salience': 0.10,  # Technical culture
            },
            
            'hurricanes': {
                'narrativity': 0.50,  # Can't control hurricane, but can control evacuation
                'observed_r': 0.47,
                'observed_bridge': 0.235,
                
                # θ
                'population_education': 0.40,  # General public
                'field_studies_nominative': 0.05,  # Very low awareness
                'name_obviousness': 0.30,  # Gender bias not obvious
                'social_cost': 0.10,
                
                # λ
                'training_years': 0.00,  # No training for evacuation choice
                'aptitude_threshold': 0.10,  # Just need to decide
                'economic_barrier': 0.30,  # Cost of evacuating
                
                # ة
                'branding_importance': 0.70,  # Name gender affects perception
                'semantic_field_richness': 0.65,  # Masculine/feminine associations
                'cultural_salience': 0.75,  # Gender stereotypes strong
            },
            
            'imdb_movies': {
                'narrativity': 0.60,
                'observed_r': 0.28,
                'observed_bridge': 0.084,
                
                # θ
                'population_education': 0.50,  # General audience
                'field_studies_nominative': 0.15,  # Some awareness
                'name_obviousness': 0.70,  # Titles/stars obvious
                'social_cost': 0.15,
                
                # λ
                'training_years': 0.50,  # Film school helps
                'aptitude_threshold': 0.60,  # Talent matters
                'economic_barrier': 0.60,  # High production costs
                
                # ة
                'branding_importance': 0.80,  # Titles and stars matter
                'semantic_field_richness': 0.70,  # Rich genre conventions
                'cultural_salience': 0.85,  # Movie culture strong
            },
            
            'startups': {
                'narrativity': 0.62,
                'observed_r': 0.36,
                'observed_bridge': 0.223,
                
                # θ
                'population_education': 0.85,  # VCs highly educated
                'field_studies_nominative': 0.30,  # Some awareness
                'name_obviousness': 0.75,  # Brand naming obvious
                'social_cost': 0.20,  # Some stigma to "just story"
                
                # λ
                'training_years': 0.30,  # No formal training required
                'aptitude_threshold': 0.60,  # Business acumen
                'economic_barrier': 0.40,  # Can bootstrap
                
                # ة
                'branding_importance': 0.85,  # Company names matter
                'semantic_field_richness': 0.75,  # Rich business metaphors
                'cultural_salience': 0.80,  # Startup culture strong
            },
            
            'mental_health': {
                'narrativity': 0.75,
                'observed_r': None,  # To be measured
                
                # θ
                'population_education': 0.70,  # Mental health professionals
                'field_studies_nominative': 0.40,  # Growing awareness
                'name_obviousness': 0.85,  # Disorder names very obvious
                'social_cost': 0.40,  # Moderate stigma
                
                # λ
                'training_years': 0.70,  # PhD/Medical degree
                'aptitude_threshold': 0.60,
                'economic_barrier': 0.50,
                
                # ة
                'branding_importance': 0.80,  # Disorder names stigmatizing
                'semantic_field_richness': 0.75,
                'cultural_salience': 0.85,
            },
            
            'tennis': {
                'narrativity': 0.25,
                'observed_r': 0.55,
                'observed_bridge': 0.138,
                
                # θ
                'population_education': 0.50,
                'field_studies_nominative': 0.30,
                'name_obviousness': 0.50,
                'social_cost': 0.20,
                
                # λ
                'training_years': 0.90,  # Lifetime training
                'aptitude_threshold': 0.90,  # Elite talent
                'economic_barrier': 0.60,  # Expensive sport
                
                # ة
                'branding_importance': 0.60,  # Player brands matter
                'semantic_field_richness': 0.45,
                'cultural_salience': 0.70,
            },
            
            'nfl': {
                'narrativity': 0.20,
                'observed_r': 0.25,
                'observed_bridge': 0.030,
                
                # θ
                'population_education': 0.45,
                'field_studies_nominative': 0.40,
                'name_obviousness': 0.65,
                'social_cost': 0.25,
                
                # λ
                'training_years': 0.95,
                'aptitude_threshold': 0.95,
                'economic_barrier': 0.40,
                
                # ة
                'branding_importance': 0.55,
                'semantic_field_richness': 0.50,
                'cultural_salience': 0.75,
            },
            
            'golf': {
                'narrativity': 0.30,
                'observed_r': None,
                
                # θ
                'population_education': 0.60,
                'field_studies_nominative': 0.25,
                'name_obviousness': 0.45,
                'social_cost': 0.15,
                
                # λ
                'training_years': 0.85,
                'aptitude_threshold': 0.85,
                'economic_barrier': 0.70,  # Expensive sport
                
                # ة
                'branding_importance': 0.50,
                'semantic_field_richness': 0.40,
                'cultural_salience': 0.65,
            },
            
            'ufc': {
                'narrativity': 0.35,
                'observed_r': None,
                
                # θ
                'population_education': 0.35,
                'field_studies_nominative': 0.35,
                'name_obviousness': 0.70,
                'social_cost': 0.15,
                
                # λ
                'training_years': 0.85,
                'aptitude_threshold': 0.92,
                'economic_barrier': 0.40,
                
                # ة
                'branding_importance': 0.70,  # Fighter persona
                'semantic_field_richness': 0.55,
                'cultural_salience': 0.75,
            },
        }
    
    def calculate_awareness(self, domain_name: str) -> float:
        """
        Calculate θ (awareness resistance) for a domain.
        
        Formula:
        θ = education × [field_studies + obviousness] × social_cost
        """
        data = self.domain_data.get(domain_name, {})
        
        education = data.get('population_education', 0.50)
        field_studies = data.get('field_studies_nominative', 0.30)
        obviousness = data.get('name_obviousness', 0.50)
        social_cost = data.get('social_cost', 0.20)
        
        θ = education * (field_studies + obviousness) / 2 * (1 + social_cost)
        
        # Clamp to [0, 1]
        θ = min(1.0, max(0.0, θ))
        
        return θ
    
    def calculate_constraints(self, domain_name: str) -> float:
        """
        Calculate λ (fundamental constraints) for a domain.
        
        Formula:
        λ = training_years/10 + aptitude_threshold + economic_barrier
        Then normalize to [0, 1]
        """
        data = self.domain_data.get(domain_name, {})
        
        training = data.get('training_years', 0.50)
        aptitude = data.get('aptitude_threshold', 0.50)
        economic = data.get('economic_barrier', 0.30)
        
        λ = (training + aptitude + economic) / 3
        
        # Clamp to [0, 1]
        λ = min(1.0, max(0.0, λ))
        
        return λ
    
    def calculate_nominative_gravity(self, domain_name: str) -> float:
        """
        Calculate ة (nominative gravity) for a domain.
        
        Formula:
        ة = п × [branding + semantic_richness + cultural_salience] / 3
        """
        data = self.domain_data.get(domain_name, {})
        
        п = data.get('narrativity', 0.50)
        branding = data.get('branding_importance', 0.50)
        semantic = data.get('semantic_field_richness', 0.50)
        cultural = data.get('cultural_salience', 0.50)
        
        ة = п * (branding + semantic + cultural) / 3
        
        # Clamp to [0, 1]
        ة = min(1.0, max(0.0, ة))
        
        return ة
    
    def calculate_all_forces(self, domain_name: str) -> Dict:
        """Calculate all three forces for a domain."""
        θ = self.calculate_awareness(domain_name)
        λ = self.calculate_constraints(domain_name)
        ة = self.calculate_nominative_gravity(domain_name)
        
        data = self.domain_data.get(domain_name, {})
        is_prestige = data.get('is_prestige_domain', False)
        
        # Calculate predicted bridge
        if is_prestige:
            Д_predicted = ة + θ - λ  # Awareness amplifies
        else:
            Д_predicted = ة - θ - λ  # Awareness suppresses
        
        # Clamp to [0, 1]
        Д_predicted = max(0.0, Д_predicted)
        
        # Get observed
        Д_observed = data.get('observed_bridge', None)
        
        # Calculate error
        if Д_observed is not None:
            model_error = abs(Д_predicted - Д_observed)
        else:
            model_error = None
        
        # Determine dominant force
        if ة > θ + λ:
            dominant_force = 'ة_dominates'
        elif θ > ة:
            dominant_force = 'θ_dominates'
        elif λ > ة:
            dominant_force = 'λ_dominates'
        else:
            dominant_force = 'equilibrium'
        
        return {
            'domain': domain_name,
            'nominative_gravity': float(ة),
            'awareness_resistance': float(θ),
            'fundamental_constraints': float(λ),
            'predicted_bridge': float(Д_predicted),
            'observed_bridge': Д_observed,
            'model_error': float(model_error) if model_error is not None else None,
            'dominant_force': dominant_force,
            'is_prestige_domain': is_prestige,
            'equation_used': 'ة + θ - λ' if is_prestige else 'ة - θ - λ'
        }
    
    def calculate_all_domains(self) -> Dict[str, Dict]:
        """Calculate forces for all domains."""
        print(f"\n{'='*80}")
        print("THREE-FORCE CALCULATION FOR ALL DOMAINS")
        print(f"{'='*80}\n")
        
        results = {}
        
        for domain_name in self.domain_data.keys():
            print(f"Calculating {domain_name}...")
            forces = self.calculate_all_forces(domain_name)
            results[domain_name] = forces
            
            # Print summary
            print(f"  ة={forces['nominative_gravity']:.3f}, "
                  f"θ={forces['awareness_resistance']:.3f}, "
                  f"λ={forces['fundamental_constraints']:.3f}")
            
            obs_str = f"{forces['observed_bridge']:.3f}" if forces['observed_bridge'] is not None else 'N/A'
            print(f"  Д_pred={forces['predicted_bridge']:.3f}, Д_obs={obs_str}")
            
            if forces['model_error'] is not None:
                print(f"  Error={forces['model_error']:.3f}, Force: {forces['dominant_force']}")
            else:
                print(f"  Force: {forces['dominant_force']}")
            print()
        
        # Summary statistics
        print(f"{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}\n")
        
        errors = [r['model_error'] for r in results.values() if r['model_error'] is not None]
        
        if errors:
            print(f"Model accuracy across {len(errors)} domains:")
            print(f"  Mean absolute error: {np.mean(errors):.3f}")
            print(f"  Median error: {np.median(errors):.3f}")
            print(f"  Max error: {np.max(errors):.3f}")
            print(f"  Min error: {np.min(errors):.3f}")
            
            if np.mean(errors) < 0.15:
                print(f"\n  ✓ EXCELLENT MODEL FIT (MAE < 0.15)")
            elif np.mean(errors) < 0.25:
                print(f"\n  ✓ GOOD MODEL FIT (MAE < 0.25)")
            else:
                print(f"\n  ⚠️  MODEL NEEDS REFINEMENT (MAE > 0.25)")
        
        # Force distribution
        force_counts = {}
        for r in results.values():
            force_counts[r['dominant_force']] = force_counts.get(r['dominant_force'], 0) + 1
        
        print(f"\nDominant force distribution:")
        for force, count in sorted(force_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {force}: {count} domains")
        
        return results
    
    def save_results(self, results: Dict, output_path: Path = None):
        """Save three-force analysis results."""
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent / 'data' / 'three_force_analysis.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved three-force analysis to: {output_path}")


def main():
    """Calculate three forces for all domains."""
    calculator = ThreeForceCalculator()
    results = calculator.calculate_all_domains()
    calculator.save_results(results)
    
    print(f"\n{'='*80}")
    print("✓ THREE-FORCE CALCULATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

