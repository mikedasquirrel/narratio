"""
Narrative Taxonomy System

Treats each comparison as a unique "organism" in the narrative ecosystem with:
- DNA (feature vector)
- Species classification (domain/subdomain)
- Gravitational mass (narrative weight)
- Evolutionary fitness (prediction quality)
- Genetic relationships (similarity to other comparisons)

Builds a living atlas of narrative species that grows with each comparison.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime
import json
import hashlib


class NarrativeOrganism:
    """
    Represents a single comparison as a biological organism in the narrative ecosystem.
    """
    
    def __init__(
        self,
        organism_id: str,
        text_a: str,
        text_b: str,
        comparison_data: Dict,
        narrative_weight: float,
        domain_classification: Dict,
        timestamp: Optional[str] = None
    ):
        self.organism_id = organism_id
        self.text_a = text_a
        self.text_b = text_b
        self.comparison_data = comparison_data
        self.narrative_weight = narrative_weight  # Gravitational mass
        self.domain_classification = domain_classification
        self.timestamp = timestamp or datetime.now().isoformat()
        
        # Extract DNA (feature vector)
        self.dna = self._extract_dna()
        
        # Classification
        self.kingdom = domain_classification.get('primary_domain', ['unknown', 'unknown'])[0]
        self.phylum = domain_classification.get('primary_domain', ['unknown', 'unknown'])[1]
        self.species_name = self._generate_species_name()
    
    def _extract_dna(self) -> np.ndarray:
        """Extract complete feature vector (DNA sequence) from comparison."""
        dna_sequence = []
        
        if 'transformers' in self.comparison_data:
            for transformer_name in ['nominative', 'self_perception', 'narrative_potential', 
                                    'linguistic', 'relational', 'ensemble']:
                if transformer_name in self.comparison_data['transformers']:
                    features_a = self.comparison_data['transformers'][transformer_name]['text_a']['features']
                    features_b = self.comparison_data['transformers'][transformer_name]['text_b']['features']
                    
                    # DNA = concatenated features from both texts
                    dna_sequence.extend(features_a)
                    dna_sequence.extend(features_b)
        
        return np.array(dna_sequence)
    
    def _generate_species_name(self) -> str:
        """Generate binomial nomenclature for this narrative organism."""
        # Genus (domain) + Species (subdomain + weight class)
        genus = self.kingdom.capitalize()
        
        # Species descriptor based on weight
        if self.narrative_weight >= 2.0:
            weight_class = "magnatus"  # Great/mighty
        elif self.narrative_weight >= 1.5:
            weight_class = "fortis"    # Strong
        elif self.narrative_weight >= 1.0:
            weight_class = "communis"  # Common
        else:
            weight_class = "minor"     # Small
        
        species = f"{self.phylum}_{weight_class}".replace('_', '')
        
        return f"{genus} {species}"
    
    def get_genome_summary(self) -> Dict[str, Any]:
        """Get summary of this organism's genetic makeup."""
        return {
            'genome_length': len(self.dna),
            'mean_expression': float(np.mean(self.dna)),
            'std_expression': float(np.std(self.dna)),
            'active_genes': int(np.sum(np.abs(self.dna) > 0.1)),  # Genes with significant expression
            'dominant_traits': self._identify_dominant_traits()
        }
    
    def _identify_dominant_traits(self) -> List[str]:
        """Identify dominant traits (highest-expressing features)."""
        traits = []
        
        if 'transformers' in self.comparison_data:
            for transformer_name, transformer_data in self.comparison_data['transformers'].items():
                if transformer_data.get('difference', 0) > 2.0:
                    traits.append(f"high_{transformer_name}")
        
        return traits
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize organism to dictionary."""
        return {
            'organism_id': self.organism_id,
            'species_name': self.species_name,
            'kingdom': self.kingdom,
            'phylum': self.phylum,
            'narrative_weight': self.narrative_weight,
            'timestamp': self.timestamp,
            'text_a_preview': self.text_a[:100],
            'text_b_preview': self.text_b[:100],
            'genome_summary': self.get_genome_summary(),
            'domain_classification': self.domain_classification
        }


class NarrativeTaxonomy:
    """
    Manages the taxonomy of all narrative organisms.
    
    Provides:
    - Classification system
    - Phylogenetic relationships
    - DNA overlap calculation
    - Gravitational clustering
    - Evolutionary tracking
    """
    
    def __init__(self):
        self.organisms = {}  # organism_id -> NarrativeOrganism
        self.taxonomy_tree = self._initialize_taxonomy_tree()
    
    def _initialize_taxonomy_tree(self) -> Dict:
        """Initialize hierarchical taxonomy structure."""
        return {
            'kingdoms': {},  # domain -> phyla
            'gravitational_clusters': {},  # weight_range -> organisms
            'temporal_lineages': {}  # date -> organisms
        }
    
    def add_organism(
        self,
        text_a: str,
        text_b: str,
        comparison_data: Dict,
        narrative_weight: float,
        domain_classification: Dict
    ) -> NarrativeOrganism:
        """Add new organism to taxonomy."""
        # Generate unique ID
        organism_id = self._generate_organism_id(text_a, text_b)
        
        # Create organism
        organism = NarrativeOrganism(
            organism_id,
            text_a,
            text_b,
            comparison_data,
            narrative_weight,
            domain_classification
        )
        
        # Store
        self.organisms[organism_id] = organism
        
        # Update taxonomy tree
        self._update_taxonomy_tree(organism)
        
        return organism
    
    def _generate_organism_id(self, text_a: str, text_b: str) -> str:
        """Generate unique ID for organism based on texts."""
        combined = f"{text_a}|{text_b}|{datetime.now().isoformat()}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _update_taxonomy_tree(self, organism: NarrativeOrganism):
        """Update taxonomy tree with new organism."""
        # Add to kingdom
        kingdom = organism.kingdom
        if kingdom not in self.taxonomy_tree['kingdoms']:
            self.taxonomy_tree['kingdoms'][kingdom] = {}
        
        phylum = organism.phylum
        if phylum not in self.taxonomy_tree['kingdoms'][kingdom]:
            self.taxonomy_tree['kingdoms'][kingdom][phylum] = []
        
        self.taxonomy_tree['kingdoms'][kingdom][phylum].append(organism.organism_id)
        
        # Add to gravitational cluster
        weight_range = self._get_weight_range(organism.narrative_weight)
        if weight_range not in self.taxonomy_tree['gravitational_clusters']:
            self.taxonomy_tree['gravitational_clusters'][weight_range] = []
        
        self.taxonomy_tree['gravitational_clusters'][weight_range].append(organism.organism_id)
        
        # Add to temporal lineage
        date = organism.timestamp[:10]  # YYYY-MM-DD
        if date not in self.taxonomy_tree['temporal_lineages']:
            self.taxonomy_tree['temporal_lineages'][date] = []
        
        self.taxonomy_tree['temporal_lineages'][date].append(organism.organism_id)
    
    def _get_weight_range(self, weight: float) -> str:
        """Classify weight into range."""
        if weight >= 2.0:
            return "supermassive"
        elif weight >= 1.5:
            return "massive"
        elif weight >= 1.0:
            return "standard"
        elif weight >= 0.5:
            return "light"
        else:
            return "minimal"
    
    def calculate_dna_overlap(
        self,
        organism_id_1: str,
        organism_id_2: str
    ) -> Dict[str, Any]:
        """
        Calculate genetic overlap between two organisms.
        
        Returns:
        - overlap_percentage: 0-100%
        - shared_traits: List of common features
        - divergent_traits: Features unique to each
        - evolutionary_distance: Genetic distance metric
        """
        org1 = self.organisms.get(organism_id_1)
        org2 = self.organisms.get(organism_id_2)
        
        if not org1 or not org2:
            return {'error': 'Organism not found'}
        
        # Calculate DNA similarity
        dna1 = org1.dna
        dna2 = org2.dna
        
        # Align DNA (pad shorter sequence)
        max_len = max(len(dna1), len(dna2))
        dna1_padded = np.pad(dna1, (0, max_len - len(dna1)), mode='constant')
        dna2_padded = np.pad(dna2, (0, max_len - len(dna2)), mode='constant')
        
        # Calculate similarity metrics
        # 1. Cosine similarity (overall alignment)
        cosine_sim = np.dot(dna1_padded, dna2_padded) / (
            np.linalg.norm(dna1_padded) * np.linalg.norm(dna2_padded) + 1e-10
        )
        
        # 2. Euclidean distance (genetic distance)
        euclidean_dist = np.linalg.norm(dna1_padded - dna2_padded)
        
        # 3. Correlation (pattern similarity)
        correlation = np.corrcoef(dna1_padded, dna2_padded)[0, 1]
        
        # Convert to overlap percentage (0-100)
        overlap_percentage = ((cosine_sim + 1) / 2) * 100  # Scale -1,1 to 0,100
        
        # Check domain overlap
        same_kingdom = org1.kingdom == org2.kingdom
        same_phylum = org1.phylum == org2.phylum
        
        # Domain similarity bonus
        if same_kingdom and same_phylum:
            domain_bonus = 20
        elif same_kingdom:
            domain_bonus = 10
        else:
            domain_bonus = 0
        
        # Gravitational similarity
        gravity_diff = abs(org1.narrative_weight - org2.narrative_weight)
        gravity_similarity = max(0, 1 - gravity_diff / 3.0) * 10
        
        # Total overlap with domain and gravity factors
        total_overlap = min(100, overlap_percentage + domain_bonus + gravity_similarity)
        
        # Classify relationship
        if total_overlap >= 90:
            relationship = "Nearly identical (same species)"
        elif total_overlap >= 75:
            relationship = "Very close (same genus)"
        elif total_overlap >= 60:
            relationship = "Related (same family)"
        elif total_overlap >= 40:
            relationship = "Distant relatives (same order)"
        elif total_overlap >= 25:
            relationship = "Distantly related (same class)"
        else:
            relationship = "Unrelated (different kingdoms)"
        
        return {
            'overlap_percentage': float(total_overlap),
            'dna_similarity': float(cosine_sim),
            'genetic_distance': float(euclidean_dist),
            'pattern_correlation': float(correlation),
            'same_kingdom': same_kingdom,
            'same_phylum': same_phylum,
            'domain_bonus': domain_bonus,
            'gravity_similarity': float(gravity_similarity),
            'relationship': relationship,
            'organism_1': {
                'id': organism_id_1,
                'species': org1.species_name,
                'weight': org1.narrative_weight
            },
            'organism_2': {
                'id': organism_id_2,
                'species': org2.species_name,
                'weight': org2.narrative_weight
            }
        }
    
    def find_nearest_relatives(
        self,
        organism_id: str,
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """Find N nearest relatives to given organism."""
        target_org = self.organisms.get(organism_id)
        if not target_org:
            return []
        
        similarities = []
        
        for other_id, other_org in self.organisms.items():
            if other_id == organism_id:
                continue
            
            overlap = self.calculate_dna_overlap(organism_id, other_id)
            similarities.append({
                'organism_id': other_id,
                'species_name': other_org.species_name,
                'overlap_percentage': overlap['overlap_percentage'],
                'relationship': overlap['relationship'],
                'kingdom': other_org.kingdom,
                'phylum': other_org.phylum
            })
        
        # Sort by overlap
        similarities.sort(key=lambda x: x['overlap_percentage'], reverse=True)
        
        return similarities[:n]
    
    def get_phylogenetic_tree(self) -> Dict[str, Any]:
        """Generate phylogenetic tree structure."""
        tree = {
            'name': 'Narrative Life',
            'children': []
        }
        
        # Build tree from kingdoms down
        for kingdom, phyla in self.taxonomy_tree['kingdoms'].items():
            kingdom_node = {
                'name': f'Kingdom: {kingdom}',
                'type': 'kingdom',
                'children': []
            }
            
            for phylum, organism_ids in phyla.items():
                phylum_node = {
                    'name': f'Phylum: {phylum}',
                    'type': 'phylum',
                    'children': []
                }
                
                # Group by gravitational mass
                mass_groups = {}
                for org_id in organism_ids:
                    org = self.organisms[org_id]
                    mass_class = self._get_weight_range(org.narrative_weight)
                    if mass_class not in mass_groups:
                        mass_groups[mass_class] = []
                    mass_groups[mass_class].append(org)
                
                for mass_class, orgs in mass_groups.items():
                    mass_node = {
                        'name': f'Mass: {mass_class}',
                        'type': 'mass_class',
                        'children': []
                    }
                    
                    for org in orgs:
                        organism_node = {
                            'name': org.species_name,
                            'type': 'organism',
                            'id': org.organism_id,
                            'weight': org.narrative_weight,
                            'size': org.narrative_weight * 10  # For visualization
                        }
                        mass_node['children'].append(organism_node)
                    
                    phylum_node['children'].append(mass_node)
                
                kingdom_node['children'].append(phylum_node)
            
            tree['children'].append(kingdom_node)
        
        return tree
    
    def get_gravitational_network(self) -> Dict[str, Any]:
        """
        Generate force-directed graph data for gravitational clustering.
        
        Nodes = organisms (size = narrative weight)
        Edges = DNA overlap (thickness = similarity)
        """
        nodes = []
        links = []
        
        organism_list = list(self.organisms.values())
        
        # Create nodes
        for org in organism_list:
            nodes.append({
                'id': org.organism_id,
                'name': org.species_name,
                'species': org.species_name,
                'kingdom': org.kingdom,
                'phylum': org.phylum,
                'weight': org.narrative_weight,
                'size': org.narrative_weight * 20,  # Visual size
                'color': self._get_kingdom_color(org.kingdom)
            })
        
        # Create links (only for significant overlap)
        for i, org1 in enumerate(organism_list):
            for org2 in organism_list[i+1:]:
                overlap = self.calculate_dna_overlap(org1.organism_id, org2.organism_id)
                
                if overlap['overlap_percentage'] > 30:  # Only show significant connections
                    links.append({
                        'source': org1.organism_id,
                        'target': org2.organism_id,
                        'value': overlap['overlap_percentage'],
                        'distance': 100 - overlap['overlap_percentage']  # Closer if more similar
                    })
        
        return {
            'nodes': nodes,
            'links': links
        }
    
    def _get_kingdom_color(self, kingdom: str) -> str:
        """Get color for kingdom."""
        colors = {
            'sports': '#06b6d4',  # cyan
            'products': '#9333ea',  # purple
            'profiles': '#ec4899',  # pink
            'brands': '#10b981',  # emerald
            'locations': '#f59e0b',  # amber
            'content': '#8b5cf6',  # violet
            'text': '#6b7280'  # gray
        }
        return colors.get(kingdom, '#6b7280')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ecosystem statistics."""
        total_organisms = len(self.organisms)
        
        if total_organisms == 0:
            return {'total_organisms': 0}
        
        # Kingdom distribution
        kingdom_counts = {}
        for org in self.organisms.values():
            kingdom_counts[org.kingdom] = kingdom_counts.get(org.kingdom, 0) + 1
        
        # Gravitational distribution
        gravity_distribution = {}
        for org in self.organisms.values():
            mass_class = self._get_weight_range(org.narrative_weight)
            gravity_distribution[mass_class] = gravity_distribution.get(mass_class, 0) + 1
        
        # Average genome length
        avg_genome_length = np.mean([len(org.dna) for org in self.organisms.values()])
        
        return {
            'total_organisms': total_organisms,
            'kingdoms': len(self.taxonomy_tree['kingdoms']),
            'kingdom_distribution': kingdom_counts,
            'gravitational_distribution': gravity_distribution,
            'avg_genome_length': float(avg_genome_length),
            'temporal_span': {
                'earliest': min(org.timestamp for org in self.organisms.values()),
                'latest': max(org.timestamp for org in self.organisms.values())
            }
        }
    
    def export_ecosystem(self) -> Dict[str, Any]:
        """Export complete ecosystem data."""
        return {
            'organisms': [org.to_dict() for org in self.organisms.values()],
            'taxonomy_tree': self.taxonomy_tree,
            'phylogenetic_tree': self.get_phylogenetic_tree(),
            'gravitational_network': self.get_gravitational_network(),
            'statistics': self.get_statistics()
        }


# Global taxonomy instance
narrative_taxonomy = NarrativeTaxonomy()

