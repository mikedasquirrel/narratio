"""
Process ALL Existing Domains with ALL Archetype Transformers

Applies the complete archetype transformer suite to every existing domain:
- NBA, NFL, Tennis, Golf, UFC, etc. (Sports)
- Movies, Oscars (Entertainment)
- Startups, Crypto (Business)
- Mental Health (Medical)
- Hurricanes, Dinosaurs (Natural/Educational)
- WWE (Prestige)

Generates comprehensive archetype profiles for cross-domain comparison.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transformers.archetypes import (
    HeroJourneyTransformer,
    CharacterArchetypeTransformer,
    PlotArchetypeTransformer,
    StructuralBeatTransformer,
    ThematicArchetypeTransformer
)


class ComprehensiveDomainProcessor:
    """
    Process all existing domains with all archetype transformers.
    """
    
    def __init__(self):
        self.transformers = {
            'hero_journey': HeroJourneyTransformer(),
            'character': CharacterArchetypeTransformer(),
            'plot': PlotArchetypeTransformer(),
            'structural': StructuralBeatTransformer(),
            'thematic': ThematicArchetypeTransformer()
        }
        
        self.domain_results = {}
        
        # Map of domain files to load
        self.domain_files = {
            'nba': 'data/domains/nba_enriched_1000.json',
            'nfl': 'data/domains/nfl_complete_dataset.json',
            'tennis': 'data/domains/tennis_complete_dataset.json',
            'golf': 'data/domains/golf_with_narratives.json',
            'ufc': 'data/domains/ufc_with_narratives.json',
            'mlb': 'data/domains/mlb_complete_dataset.json',
            'movies': 'data/domains/imdb_movies_complete.json',
            'oscars': 'data/domains/oscar_nominees_complete.json',
            'startups': 'data/domains/startups_verified.json',
            'crypto': 'crypto_enriched_narratives.json',
            'mental_health': 'mental_health_complete_200_disorders.json',
            'hurricanes': 'data/domains/hurricanes/hurricane_complete_dataset.json',
            'dinosaurs': 'data/domains/dinosaurs/dinosaur_complete_dataset.json',
            'poker': 'data/domains/poker/poker_tournament_dataset_with_narratives.json',
            'boxing': 'data/domains/boxing/boxing_fights_complete.json'
        }
    
    def load_domain_data(self, domain_name: str, file_path: str) -> Tuple[List[str], np.ndarray]:
        """Load narratives and outcomes from domain file."""
        path = Path(file_path)
        
        if not path.exists():
            print(f"   ⏳ {domain_name}: File not found")
            return None, None
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            # Extract narratives and outcomes (domain-specific logic)
            narratives, outcomes = self._extract_narratives_outcomes(data, domain_name)
            
            if narratives and len(narratives) > 0:
                print(f"   ✅ {domain_name}: {len(narratives)} narratives loaded")
                return narratives, outcomes
            else:
                print(f"   ⏳ {domain_name}: No narratives found")
                return None, None
                
        except Exception as e:
            print(f"   ❌ {domain_name}: Error loading - {e}")
            return None, None
    
    def _extract_narratives_outcomes(self, data: Dict, domain: str) -> Tuple[List[str], np.ndarray]:
        """Extract narratives and outcomes from domain-specific format."""
        narratives = []
        outcomes = []
        
        # Domain-specific extraction
        if domain == 'nba':
            if isinstance(data, list):
                narratives = [g['narrative'] for g in data if 'narrative' in g]
                outcomes = np.array([1 if g['won'] else 0 for g in data if 'narrative' in g])
        
        elif domain == 'nfl':
            if 'games' in data:
                narratives = [g.get('narrative', '') for g in data['games'] if g.get('narrative')]
                outcomes = np.array([g.get('won', 0) for g in data['games'] if g.get('narrative')])
        
        elif domain == 'movies':
            if 'movies' in data:
                narratives = [m.get('plot_summary', '') or m.get('summary', '') 
                             for m in data['movies'] if m.get('plot_summary') or m.get('summary')]
                outcomes = np.array([m.get('box_office', 0) for m in data['movies'] 
                                    if m.get('plot_summary') or m.get('summary')])
        
        elif domain == 'startups':
            if isinstance(data, list):
                narratives = [s.get('product_story', '') or s.get('description', '') 
                             for s in data if s.get('product_story') or s.get('description')]
                outcomes = np.array([s.get('funded', 0) for s in data 
                                    if s.get('product_story') or s.get('description')])
        
        elif domain in ['tennis', 'golf', 'ufc', 'mlb', 'poker', 'boxing']:
            # Sports domains
            if isinstance(data, dict) and 'matches' in data:
                narratives = [m.get('narrative', '') for m in data['matches'] if m.get('narrative')]
                outcomes = np.array([m.get('won', 0) for m in data['matches'] if m.get('narrative')])
            elif isinstance(data, list):
                narratives = [m.get('narrative', '') for m in data if m.get('narrative')]
                outcomes = np.array([m.get('won', 0) for m in data if m.get('narrative')])
        
        elif domain == 'mental_health':
            narratives = [d.get('description', '') for d in data if isinstance(d, dict) and d.get('description')]
            outcomes = np.array([d.get('severity', 0.5) for d in data if isinstance(d, dict) and d.get('description')])
        
        elif domain == 'hurricanes':
            if 'hurricanes' in data:
                narratives = [h.get('narrative', '') for h in data['hurricanes'] if h.get('narrative')]
                outcomes = np.array([h.get('deaths', 0) for h in data['hurricanes'] if h.get('narrative')])
        
        elif domain == 'dinosaurs':
            if 'dinosaurs' in data:
                narratives = [d.get('description', '') for d in data['dinosaurs'] if d.get('description')]
                outcomes = np.array([d.get('popularity', 0) for d in data['dinosaurs'] if d.get('description')])
        
        return narratives, outcomes
    
    def process_domain(self, domain_name: str, narratives: List[str], outcomes: np.ndarray) -> Dict:
        """Process single domain with all archetype transformers."""
        print(f"\n   Processing {domain_name} with all transformers...")
        
        results = {
            'domain': domain_name,
            'sample_size': len(narratives),
            'transformers': {}
        }
        
        # Apply each transformer
        for transformer_name, transformer in self.transformers.items():
            try:
                print(f"      {transformer_name}...", end='')
                
                transformer.fit(narratives)
                features = transformer.transform(narratives)
                
                # Extract key metrics
                if transformer_name == 'hero_journey':
                    results['transformers']['hero_journey'] = {
                        'mean_completion': float(features[:, 2].mean()),
                        'stages_present_mean': float(np.mean([sum(features[i, :17] > 0.5) for i in range(len(features))])),
                        'follows_journey_pct': float(sum(features[:, 2] > 0.60) / len(features))
                    }
                
                elif transformer_name == 'character':
                    jung_scores = features[:, :12]
                    jung_names = ['innocent', 'orphan', 'warrior', 'caregiver', 'explorer', 
                                 'rebel', 'lover', 'creator', 'jester', 'sage', 'magician', 'ruler']
                    dominant = [jung_names[i] for i in np.argmax(jung_scores, axis=1)]
                    
                    from collections import Counter
                    dist = Counter(dominant)
                    
                    results['transformers']['character'] = {
                        'mean_clarity': float(features[:, 12].mean()),
                        'dominant_archetype': dist.most_common(1)[0][0],
                        'archetype_distribution': dict(dist.most_common(5))
                    }
                
                elif transformer_name == 'plot':
                    booker_scores = features[:, :7]
                    booker_names = ['overcoming_monster', 'rags_to_riches', 'quest', 'voyage_and_return',
                                   'comedy', 'tragedy', 'rebirth']
                    dominant = [booker_names[i] for i in np.argmax(booker_scores, axis=1)]
                    
                    from collections import Counter
                    dist = Counter(dominant)
                    
                    results['transformers']['plot'] = {
                        'plot_purity': float(features[:, 7].mean()),
                        'dominant_plot': dist.most_common(1)[0][0],
                        'plot_distribution': dict(dist.most_common())
                    }
                
                elif transformer_name == 'structural':
                    results['transformers']['structural'] = {
                        'mean_beat_adherence': float(features[:, 15].mean()),
                        'mean_structure_quality': float(features[:, -1].mean())
                    }
                
                elif transformer_name == 'thematic':
                    mythos_scores = features[:, :4]
                    mythos_names = ['comedy', 'romance', 'tragedy', 'irony']
                    dominant = [mythos_names[i] for i in np.argmax(mythos_scores, axis=1)]
                    
                    from collections import Counter
                    dist = Counter(dominant)
                    
                    results['transformers']['thematic'] = {
                        'mythos_purity': float(features[:, 4].mean()),
                        'dominant_mythos': dist.most_common(1)[0][0],
                        'mythos_distribution': dict(dist.most_common())
                    }
                
                # Save feature matrix
                feature_dir = Path('narrative_optimization/data/archetype_features')
                feature_dir.mkdir(parents=True, exist_ok=True)
                
                np.savez(
                    feature_dir / f'{domain_name}_{transformer_name}_features.npz',
                    features=features,
                    feature_names=transformer.get_feature_names()
                )
                
                print(f" ✓")
                
            except Exception as e:
                print(f" ✗ Error: {e}")
                results['transformers'][transformer_name] = {'error': str(e)}
        
        return results
    
    def run_comprehensive_processing(self) -> None:
        """Process all domains with all transformers."""
        print("="*70)
        print("COMPREHENSIVE ARCHETYPE PROCESSING")
        print("="*70)
        print("Applying ALL archetype transformers to ALL existing domains")
        print("="*70)
        
        print("\nLoading domains...")
        
        for domain_name, file_path in self.domain_files.items():
            narratives, outcomes = self.load_domain_data(domain_name, file_path)
            
            if narratives is not None and len(narratives) >= 10:
                results = self.process_domain(domain_name, narratives, outcomes)
                self.domain_results[domain_name] = results
        
        # Save comprehensive results
        print("\n" + "="*70)
        print("SAVING COMPREHENSIVE RESULTS")
        print("="*70)
        
        output_path = Path('narrative_optimization/results/comprehensive_archetype_analysis.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.domain_results, f, indent=2, default=str)
        
        print(f"\n✅ Comprehensive results saved: {output_path}")
        
        # Generate summary
        self._generate_summary()
    
    def _generate_summary(self) -> None:
        """Generate cross-domain summary."""
        print("\n" + "="*70)
        print("CROSS-DOMAIN ARCHETYPE SUMMARY")
        print("="*70)
        
        print(f"\nDomains processed: {len(self.domain_results)}")
        
        # Journey completion by domain
        print("\nHero's Journey Completion by Domain:")
        journey_data = []
        for domain, results in self.domain_results.items():
            if 'hero_journey' in results['transformers']:
                completion = results['transformers']['hero_journey']['mean_completion']
                journey_data.append((domain, completion))
        
        journey_data.sort(key=lambda x: x[1], reverse=True)
        for domain, completion in journey_data:
            print(f"   {domain}: {completion:.1%}")
        
        # Dominant archetypes by domain
        print("\nDominant Character Archetype by Domain:")
        for domain, results in self.domain_results.items():
            if 'character' in results['transformers']:
                archetype = results['transformers']['character']['dominant_archetype']
                print(f"   {domain}: {archetype}")
        
        # Dominant plots by domain
        print("\nDominant Plot Type by Domain:")
        for domain, results in self.domain_results.items():
            if 'plot' in results['transformers']:
                plot = results['transformers']['plot']['dominant_plot']
                print(f"   {domain}: {plot}")
        
        # Dominant mythoi by domain
        print("\nDominant Mythos by Domain:")
        for domain, results in self.domain_results.items():
            if 'thematic' in results['transformers']:
                mythos = results['transformers']['thematic']['dominant_mythos']
                print(f"   {domain}: {mythos}")
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"\nTotal domains analyzed: {len(self.domain_results)}")
        print(f"Total feature matrices saved: {len(self.domain_results) * 5}")
        print("\nResults available at:")
        print("   - narrative_optimization/results/comprehensive_archetype_analysis.json")
        print("   - narrative_optimization/data/archetype_features/ (feature matrices)")
        print("="*70)


def main():
    """Run comprehensive processing."""
    processor = ComprehensiveDomainProcessor()
    processor.run_comprehensive_processing()


if __name__ == '__main__':
    main()

