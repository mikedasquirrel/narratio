"""
Genre Relationship Mapper

Maps domains as literary genres with DNA overlap indicating relatedness.
Like subgenres in literature - some domains are closer than others.

Competition sports cluster together (NBA, NFL, MMA)
Products cluster separately
Profiles cluster separately

DNA overlap reveals genre family structure.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


class GenreRelationshipMapper:
    """
    Maps domain relationships as literary genre taxonomy.
    
    Hypothesis: Domains with similar DNA are like subgenres.
    - NBA ↔ NFL: High overlap (same genre family: team sports)
    - NBA ↔ Wine: Low overlap (different genre families)
    """
    
    def __init__(self):
        # Define genre hierarchy
        self.genre_tree = {
            'Competition': {
                'Sports': {
                    'Team_Sports': ['nba', 'nfl', 'mlb', 'soccer'],
                    'Individual_Sports': ['tennis', 'golf', 'swimming'],
                    'Combat_Sports': ['mma', 'boxing', 'wrestling']
                },
                'E_Sports': {
                    'MOBA': ['league', 'dota'],
                    'FPS': ['csgo', 'valorant'],
                    'Strategy': ['starcraft']
                },
                'Business': {
                    'Startups': ['tech_competition'],
                    'Markets': ['stock_competition']
                }
            },
            'Experience': {
                'Products': {
                    'Tech': ['smartphones', 'laptops'],
                    'Consumer': ['appliances', 'goods'],
                    'Food_Beverage': ['wine', 'beer', 'restaurants']
                },
                'Services': {
                    'Hospitality': ['hotels', 'airbnb'],
                    'Dining': ['restaurants', 'cafes']
                }
            },
            'Relationship': {
                'Personal': {
                    'Dating': ['romantic_matching'],
                    'Friendship': ['social_matching']
                },
                'Professional': {
                    'Hiring': ['job_matching'],
                    'Collaboration': ['team_formation']
                }
            },
            'Achievement': {
                'Personal_Growth': ['wellness', 'education'],
                'Recognition': ['awards', 'rankings']
            }
        }
        
        # Expected α ranges by genre family
        self.genre_alpha_ranges = {
            'Competition': (0.20, 0.40),      # Character-driven
            'Experience': (0.30, 0.60),        # Hybrid
            'Relationship': (0.15, 0.35),      # Highly narrative
            'Achievement': (0.25, 0.45)        # Character development
        }
    
    def get_genre_path(self, domain: str, subdomain: str) -> List[str]:
        """
        Get full genre path for a domain.
        
        Example: ('sports', 'nba') → ['Competition', 'Sports', 'Team_Sports', 'nba']
        """
        # Search tree
        for family, family_content in self.genre_tree.items():
            for genre, genre_content in family_content.items():
                if isinstance(genre_content, dict):
                    for subgenre, domains in genre_content.items():
                        if subdomain in domains:
                            return [family, genre, subgenre, subdomain]
        
        return ['Unknown', domain, subdomain]
    
    def calculate_genre_distance(
        self,
        path1: List[str],
        path2: List[str]
    ) -> int:
        """
        Calculate taxonomic distance between genres.
        
        Returns levels of separation:
        - 0: Same domain
        - 1: Same subgenre
        - 2: Same genre
        - 3: Same family
        - 4: Different families
        """
        # Find deepest common ancestor
        common_depth = 0
        for i in range(min(len(path1), len(path2))):
            if path1[i] == path2[i]:
                common_depth = i + 1
            else:
                break
        
        # Distance = total depth - common depth
        total_depth = len(path1) + len(path2)
        distance = total_depth - 2 * common_depth
        
        return distance
    
    def predict_dna_overlap(
        self,
        domain1: Tuple[str, str],
        domain2: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Predict DNA overlap based on genre relationship.
        
        Hypothesis: Closer genres → higher DNA overlap
        """
        path1 = self.get_genre_path(domain1[0], domain1[1])
        path2 = self.get_genre_path(domain2[0], domain2[1])
        
        distance = self.calculate_genre_distance(path1, path2)
        
        # Predict overlap based on distance
        if distance == 0:
            expected_overlap = (95, 100)
            relationship = "Same domain"
        elif distance == 1:
            expected_overlap = (80, 95)
            relationship = "Same subgenre (very close)"
        elif distance == 2:
            expected_overlap = (65, 80)
            relationship = "Same genre (related)"
        elif distance == 3:
            expected_overlap = (45, 65)
            relationship = "Same family (distant)"
        elif distance == 4:
            expected_overlap = (30, 45)
            relationship = "Related families"
        else:
            expected_overlap = (10, 30)
            relationship = "Different families (unrelated)"
        
        return {
            'expected_overlap_range': expected_overlap,
            'taxonomic_distance': distance,
            'relationship': relationship,
            'genre_path_1': path1,
            'genre_path_2': path2,
            'shared_ancestor': path1[:distance] if distance > 0 else path1
        }
    
    def predict_alpha_similarity(
        self,
        domain1: Tuple[str, str],
        domain2: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Predict if domains have similar α parameters based on genre.
        
        Hypothesis: Same genre family → similar α
        """
        path1 = self.get_genre_path(domain1[0], domain1[1])
        path2 = self.get_genre_path(domain2[0], domain2[1])
        
        # Get genre families
        family1 = path1[0] if len(path1) > 0 else 'Unknown'
        family2 = path2[0] if len(path2) > 0 else 'Unknown'
        
        # Same family = similar α expected
        if family1 == family2:
            alpha_range1 = self.genre_alpha_ranges.get(family1, (0.3, 0.6))
            
            return {
                'similar_alpha_expected': True,
                'family': family1,
                'expected_alpha_range': alpha_range1,
                'reasoning': f'Both in {family1} family, should have similar α'
            }
        else:
            return {
                'similar_alpha_expected': False,
                'family_1': family1,
                'family_2': family2,
                'alpha_range_1': self.genre_alpha_ranges.get(family1, (0.3, 0.6)),
                'alpha_range_2': self.genre_alpha_ranges.get(family2, (0.3, 0.6)),
                'reasoning': f'{family1} vs {family2} - different families, different α expected'
            }
    
    def build_genre_matrix(
        self,
        organisms: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build DNA overlap matrix organized by genre.
        
        Shows clustering: sports organisms cluster, products cluster separately.
        """
        # Group organisms by genre
        genre_groups = {}
        
        for org_id, organism in organisms.items():
            domain = (organism.get('kingdom', 'unknown'), 
                     organism.get('phylum', 'unknown'))
            path = self.get_genre_path(domain[0], domain[1])
            genre_key = '/'.join(path[:2])  # Family/Genre
            
            if genre_key not in genre_groups:
                genre_groups[genre_key] = []
            genre_groups[genre_key].append(org_id)
        
        # Calculate inter-genre overlap
        genre_overlap_matrix = {}
        
        for genre1, organisms1 in genre_groups.items():
            genre_overlap_matrix[genre1] = {}
            for genre2, organisms2 in genre_groups.items():
                # Sample organisms from each genre
                sample_overlap = []
                
                for org1_id in organisms1[:5]:  # Sample max 5
                    for org2_id in organisms2[:5]:
                        if org1_id != org2_id:
                            # Would calculate actual DNA overlap here
                            # For now, predict based on genre distance
                            pass
                
                # Use genre distance to predict overlap
                paths1 = [self.get_genre_path(*self._get_domain(organisms[org_id])) 
                         for org_id in organisms1[:1] if org_id in organisms]
                paths2 = [self.get_genre_path(*self._get_domain(organisms[org_id])) 
                         for org_id in organisms2[:1] if org_id in organisms]
                
                if paths1 and paths2:
                    dist = self.calculate_genre_distance(paths1[0], paths2[0])
                    expected = self.predict_dna_overlap(
                        self._path_to_domain(paths1[0]),
                        self._path_to_domain(paths2[0])
                    )
                    genre_overlap_matrix[genre1][genre2] = expected['expected_overlap_range'][0]
        
        return {
            'genre_groups': {k: len(v) for k, v in genre_groups.items()},
            'overlap_matrix': genre_overlap_matrix
        }
    
    def _get_domain(self, organism: Dict) -> Tuple[str, str]:
        """Extract domain tuple from organism."""
        return (organism.get('kingdom', 'unknown'), organism.get('phylum', 'unknown'))
    
    def _path_to_domain(self, path: List[str]) -> Tuple[str, str]:
        """Convert genre path back to domain tuple."""
        if len(path) >= 4:
            return (path[1].lower(), path[3])
        return ('unknown', 'unknown')


def create_genre_mapper():
    """Factory function."""
    return GenreRelationshipMapper()

