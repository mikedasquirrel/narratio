"""
Universal Multi-Domain Researcher Collector

Collects researchers across ALL academic domains to test gravitational similarity
hypothesis: Are people with name-field fit overrepresented in matching careers?

Integrates with our narrative framework:
- п (narrativity) = career choice subjectivity per domain
- ж (genome) = 524 features from 29 transformers  
- ю (story quality) = name-field fit score
- Д (bridge) = actual career selection effect
"""

import sys
from pathlib import Path
import json
import time
import requests
from collections import defaultdict
from typing import List, Dict, Optional
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


class UniversalFieldTaxonomy:
    """
    Complete taxonomy of all research fields and occupations.
    Maps fields to keywords for matching with names.
    """
    
    def __init__(self):
        """Initialize comprehensive field taxonomy."""
        
        # Academic domains with subfields and keywords
        self.fields = {
            # LIFE SCIENCES
            'biology': {
                'subfields': ['marine_biology', 'botany', 'zoology', 'genetics', 'molecular_biology', 
                             'ecology', 'evolution', 'microbiology', 'cell_biology'],
                'keywords': ['bio', 'life', 'organ', 'cell', 'gene', 'species', 'evolution', 'ecology'],
                'narrativity': 0.60  # Moderate - requires training but topic choice flexible
            },
            'marine_biology': {
                'subfields': ['oceanography', 'marine_ecology', 'ichthyology'],
                'keywords': ['marine', 'ocean', 'sea', 'fish', 'aqua', 'coral', 'reef', 'tide'],
                'narrativity': 0.65
            },
            
            # PHYSICAL SCIENCES
            'physics': {
                'subfields': ['astrophysics', 'quantum', 'nuclear', 'particle', 'condensed_matter'],
                'keywords': ['phys', 'quantum', 'particle', 'energy', 'force', 'field', 'wave'],
                'narrativity': 0.55  # Lower - highly technical
            },
            'astronomy': {
                'subfields': ['astrophysics', 'cosmology', 'planetary_science'],
                'keywords': ['astro', 'star', 'planet', 'galaxy', 'cosmos', 'space', 'celestial'],
                'narrativity': 0.70  # Higher - more romantic/narrative
            },
            'chemistry': {
                'subfields': ['organic', 'inorganic', 'physical', 'analytical', 'biochemistry'],
                'keywords': ['chem', 'molecule', 'compound', 'reaction', 'synthesis', 'element'],
                'narrativity': 0.50
            },
            
            # EARTH SCIENCES
            'geology': {
                'subfields': ['mineralogy', 'petrology', 'volcanology', 'seismology'],
                'keywords': ['geo', 'earth', 'rock', 'mineral', 'volcano', 'quake', 'stone'],
                'narrativity': 0.58
            },
            'meteorology': {
                'subfields': ['climatology', 'atmospheric_science'],
                'keywords': ['weather', 'climate', 'atmosphere', 'storm', 'wind', 'rain', 'cloud'],
                'narrativity': 0.62
            },
            
            # COMPUTER & MATHEMATICAL SCIENCES  
            'computer_science': {
                'subfields': ['AI', 'machine_learning', 'security', 'networks', 'databases', 'graphics'],
                'keywords': ['comput', 'code', 'algorithm', 'program', 'software', 'data', 'cyber'],
                'narrativity': 0.65
            },
            'mathematics': {
                'subfields': ['algebra', 'geometry', 'topology', 'analysis', 'statistics', 'number_theory'],
                'keywords': ['math', 'number', 'equation', 'theorem', 'proof', 'calculus', 'algebra'],
                'narrativity': 0.45  # Very technical
            },
            
            # MEDICAL & HEALTH
            'medicine': {
                'subfields': ['cardiology', 'neurology', 'oncology', 'surgery', 'pediatrics', 'psychiatry'],
                'keywords': ['medic', 'doctor', 'physician', 'clinic', 'hospital', 'patient', 'cure', 'heal'],
                'narrativity': 0.40  # Lower - requires specific training
            },
            'dentistry': {
                'subfields': ['orthodontics', 'periodontics', 'oral_surgery'],
                'keywords': ['dent', 'tooth', 'oral', 'mouth', 'orthodont'],
                'narrativity': 0.45
            },
            'neuroscience': {
                'subfields': ['cognitive', 'behavioral', 'molecular', 'computational'],
                'keywords': ['neuro', 'brain', 'mind', 'cognit', 'neural', 'synapse'],
                'narrativity': 0.58
            },
            
            # PSYCHOLOGY & SOCIAL SCIENCES
            'psychology': {
                'subfields': ['clinical', 'cognitive', 'social', 'developmental', 'experimental'],
                'keywords': ['psych', 'mind', 'behavior', 'cognit', 'emotion', 'mental'],
                'narrativity': 0.68
            },
            'sociology': {
                'subfields': ['social_theory', 'demography', 'criminology'],
                'keywords': ['social', 'society', 'culture', 'group', 'community'],
                'narrativity': 0.72
            },
            'economics': {
                'subfields': ['micro', 'macro', 'finance', 'behavioral'],
                'keywords': ['econom', 'market', 'trade', 'money', 'finance', 'capital'],
                'narrativity': 0.55
            },
            
            # HUMANITIES
            'philosophy': {
                'subfields': ['ethics', 'metaphysics', 'epistemology', 'logic'],
                'keywords': ['philos', 'ethics', 'moral', 'logic', 'reason', 'wisdom'],
                'narrativity': 0.80  # Very high - subjective
            },
            'linguistics': {
                'subfields': ['syntax', 'semantics', 'phonetics', 'psycholinguistics'],
                'keywords': ['language', 'linguistic', 'grammar', 'syntax', 'word', 'speech'],
                'narrativity': 0.70
            },
            'literature': {
                'subfields': ['poetry', 'fiction', 'criticism', 'theory'],
                'keywords': ['liter', 'book', 'novel', 'poet', 'story', 'writing'],
                'narrativity': 0.85  # Very high
            },
            
            # ENGINEERING
            'engineering': {
                'subfields': ['mechanical', 'electrical', 'civil', 'chemical', 'aerospace'],
                'keywords': ['engineer', 'design', 'build', 'construct', 'mechanical', 'system'],
                'narrativity': 0.48
            },
            
            # LAW & GOVERNANCE
            'law': {
                'subfields': ['criminal', 'corporate', 'constitutional', 'international'],
                'keywords': ['law', 'legal', 'court', 'justice', 'attorney', 'lawyer', 'judge'],
                'narrativity': 0.62
            },
            
            # ARTS
            'art': {
                'subfields': ['painting', 'sculpture', 'photography', 'digital'],
                'keywords': ['art', 'paint', 'draw', 'create', 'visual', 'aesthetic'],
                'narrativity': 0.90  # Extremely high
            },
            'music': {
                'subfields': ['composition', 'performance', 'theory', 'musicology'],
                'keywords': ['music', 'sound', 'melody', 'harmony', 'rhythm', 'song'],
                'narrativity': 0.88
            }
        }
    
    def get_field_keywords(self, field: str) -> List[str]:
        """Get all keywords for a field."""
        if field in self.fields:
            return self.fields[field]['keywords']
        return []
    
    def get_narrativity(self, field: str) -> float:
        """Get п (narrativity) for a field."""
        if field in self.fields:
            return self.fields[field]['narrativity']
        return 0.60  # Default moderate
    
    def infer_field(self, keywords: List[str], title: str = "") -> str:
        """Infer field from keywords and title."""
        text = ' '.join(keywords + [title]).lower()
        
        # Score each field
        scores = {}
        for field, data in self.fields.items():
            score = sum(1 for kw in data['keywords'] if kw in text)
            if score > 0:
                scores[field] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return 'general'
    
    def get_all_fields(self) -> List[str]:
        """Get list of all fields."""
        return list(self.fields.keys())


class MultiDomainResearcherCollector:
    """Collect researchers from multiple domains via multiple APIs."""
    
    def __init__(self):
        """Initialize collector."""
        self.taxonomy = UniversalFieldTaxonomy()
        self.researchers = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Bot 1.0)'
        })
    
    def collect_from_pubmed_by_field(self, field: str, max_results: int = 100) -> List[Dict]:
        """
        Collect researchers from PubMed for a specific field.
        
        Args:
            field: Research field (e.g., 'biology', 'physics')
            max_results: Maximum papers to retrieve
            
        Returns:
            List of researcher dictionaries
        """
        print(f"\n[PubMed] Collecting {field} researchers...")
        
        # Get keywords for this field
        keywords = self.taxonomy.get_field_keywords(field)
        
        if not keywords:
            print(f"  No keywords defined for {field}, skipping")
            return []
        
        # Build query
        query_terms = ' OR '.join([f'"{kw}"[Title/Abstract]' for kw in keywords[:3]])  # Use top 3 keywords
        
        # PubMed search
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query_terms,
            'retmax': max_results,
            'retmode': 'json',
            'email': 'research@universal.study'
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            print(f"  Found {len(pmids)} papers")
            
            if not pmids:
                return []
            
            # Fetch details
            time.sleep(0.34)  # Rate limit
            
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids[:50]),  # Limit to 50 for speed
                'retmode': 'xml',
                'email': 'research@universal.study'
            }
            
            response = self.session.get(fetch_url, params=params, timeout=60)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            researchers_found = []
            
            for article in root.findall('.//PubmedArticle'):
                article_elem = article.find('.//Article')
                if article_elem is None:
                    continue
                
                # Extract authors
                for author_elem in article_elem.findall('.//Author'):
                    lastname = author_elem.find('.//LastName')
                    forename = author_elem.find('.//ForeName')
                    
                    if lastname is not None and forename is not None:
                        researcher = {
                            'name': f"{forename.text} {lastname.text}",
                            'first_name': forename.text,
                            'last_name': lastname.text,
                            'field': field,
                            'narrativity': self.taxonomy.get_narrativity(field),
                            'source': 'pubmed',
                            'papers': 1  # Will aggregate later
                        }
                        researchers_found.append(researcher)
            
            print(f"  Extracted {len(researchers_found)} researchers")
            
            return researchers_found
            
        except Exception as e:
            print(f"  Error collecting from PubMed: {e}")
            return []
    
    def collect_multi_domain(self, fields_to_collect: List[str] = None, 
                            papers_per_field: int = 50) -> Dict[str, List[Dict]]:
        """
        Collect researchers from multiple domains.
        
        Args:
            fields_to_collect: List of fields, or None for all
            papers_per_field: Papers to collect per field
            
        Returns:
            Dictionary mapping field to researchers
        """
        print(f"\n{'='*80}")
        print("UNIVERSAL MULTI-DOMAIN RESEARCHER COLLECTION")
        print(f"{'='*80}\n")
        
        if fields_to_collect is None:
            fields_to_collect = self.taxonomy.get_all_fields()[:10]  # Limit to 10 fields for demo
        
        print(f"Collecting from {len(fields_to_collect)} fields:")
        print(f"  {', '.join(fields_to_collect)}\n")
        
        all_researchers = defaultdict(list)
        
        total = len(fields_to_collect)
        for idx, field in enumerate(fields_to_collect, 1):
            print(f"[{idx}/{total}] Processing {field}...")
            
            researchers = self.collect_from_pubmed_by_field(field, papers_per_field)
            
            if researchers:
                all_researchers[field].extend(researchers)
                print(f"  ✓ {len(researchers)} researchers from {field}")
            else:
                print(f"  ✗ No researchers found for {field}")
            
            # Rate limiting
            time.sleep(1)
        
        # Aggregate and deduplicate
        unique_researchers = self._deduplicate_researchers(all_researchers)
        
        print(f"\n{'='*80}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total researchers: {len(unique_researchers)}")
        print(f"Fields covered: {len(all_researchers)}")
        print(f"{'='*80}\n")
        
        self.researchers = unique_researchers
        
        return all_researchers
    
    def _deduplicate_researchers(self, field_researchers: Dict[str, List[Dict]]) -> List[Dict]:
        """Deduplicate researchers across fields."""
        seen = {}
        unique = []
        
        for field, researchers in field_researchers.items():
            for r in researchers:
                name = r['name'].lower().strip()
                
                if name not in seen:
                    seen[name] = r
                    unique.append(r)
                else:
                    # Same person in multiple fields
                    seen[name]['papers'] = seen[name].get('papers', 1) + 1
        
        return unique
    
    def save_researchers(self, output_path: Path = None):
        """Save collected researchers."""
        if output_path is None:
            output_path = Path(__file__).parent.parent / 'data' / 'researchers_multi_domain.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'total_researchers': len(self.researchers),
                'collection_date': time.strftime('%Y-%m-%d'),
                'researchers': self.researchers
            }, f, indent=2)
        
        print(f"✓ Saved {len(self.researchers)} researchers to: {output_path}")


def main():
    """Collect researchers from multiple domains."""
    collector = MultiDomainResearcherCollector()
    
    # Start with high-interest fields
    fields = [
        'biology', 'marine_biology', 'dentistry', 'medicine',
        'psychology', 'physics', 'chemistry', 'law',
        'computer_science', 'mathematics'
    ]
    
    collector.collect_multi_domain(fields, papers_per_field=30)
    collector.save_researchers()


if __name__ == "__main__":
    main()

