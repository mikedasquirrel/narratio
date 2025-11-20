"""
Multi-Domain Detector

Detects the "collage of domains" present in text analysis.
Text can span multiple domains simultaneously.
"""

from typing import List, Tuple, Set, Dict, Any
import re


class MultiDomainDetector:
    """
    Detects all domains and sub-domains present in text.
    
    Handles "collage" - text spanning multiple domains:
    - "Lakers championship legacy" → Sports (NBA) + Achievement (cultural)
    - "iPhone innovative design" → Products (tech) + Innovation (business)
    """
    
    def __init__(self):
        """Initialize domain detection patterns."""
        
        self.domain_patterns = {
            'sports': {
                'nba': ['lakers', 'celtics', 'warriors', 'nba', 'basketball', 'lebron', 'curry', 'championship ring'],
                'nfl': ['patriots', 'cowboys', 'nfl', 'football', 'super bowl', 'quarterback'],
                'mlb': ['yankees', 'dodgers', 'mlb', 'baseball', 'world series'],
                'soccer': ['premier league', 'champions league', 'football', 'soccer', 'world cup'],
                'tennis': ['wimbledon', 'grand slam', 'tennis', 'federer', 'nadal'],
                'generic': ['team', 'game', 'win', 'lose', 'coach', 'player', 'season', 'match']
            },
            
            'products': {
                'tech': ['iphone', 'android', 'smartphone', 'device', 'specs', 'features', 'technology'],
                'consumer': ['product', 'quality', 'price', 'value', 'warranty', 'customer'],
                'automotive': ['car', 'vehicle', 'engine', 'performance', 'mpg'],
                'generic': ['buy', 'purchase', 'cost', 'benefit', 'compare']
            },
            
            'profiles': {
                'dating': ['looking for', 'seeking', 'connection', 'relationship', 'partner', 'date'],
                'professional': ['experience', 'skills', 'career', 'professional', 'resume', 'qualified'],
                'social': ['interests', 'hobbies', 'personality', 'values', 'lifestyle'],
                'generic': ['i am', 'i have', 'i enjoy', 'i love', 'about me']
            },
            
            'brands': {
                'corporate': ['mission', 'values', 'company', 'organization', 'business'],
                'marketing': ['brand', 'message', 'positioning', 'audience', 'market'],
                'social_impact': ['sustainability', 'responsibility', 'community', 'impact'],
                'generic': ['we believe', 'we are', 'our vision', 'our mission']
            },
            
            'locations': {
                'cities': ['city', 'urban', 'downtown', 'neighborhood', 'metro'],
                'venues': ['stadium', 'arena', 'park', 'facility', 'venue'],
                'generic': ['place', 'location', 'area', 'region']
            },
            
            'content': {
                'academic': ['research', 'study', 'paper', 'journal', 'academic'],
                'news': ['reported', 'according to', 'sources', 'breaking', 'news'],
                'creative': ['story', 'narrative', 'character', 'plot', 'creative'],
                'generic': ['article', 'text', 'content', 'writing']
            }
        }
    
    def detect_domains(self, text_a: str, text_b: str) -> List[Tuple[str, str]]:
        """
        Detect all domains present in the text pair.
        
        Parameters
        ----------
        text_a : str
            First text
        text_b : str
            Second text
        
        Returns
        -------
        domains : list of (domain, subdomain) tuples
            All detected domains (can be multiple - "collage")
        """
        combined_text = (text_a + " " + text_b).lower()
        
        detected = set()
        
        for domain, subdomains in self.domain_patterns.items():
            for subdomain, keywords in subdomains.items():
                # Check if any keywords present
                matches = sum(1 for keyword in keywords if keyword in combined_text)
                
                if matches >= 2:  # Need at least 2 keyword matches
                    detected.add((domain, subdomain))
                elif matches == 1 and len(keywords) < 5:  # Strong specific keywords
                    detected.add((domain, subdomain))
        
        # If no specific subdomain but generic found, use generic
        domains_list = list(detected)
        
        # If nothing detected, default to generic text
        if not domains_list:
            domains_list = [('text', 'generic')]
        
        return domains_list
    
    def get_primary_domain(self, domains: List[Tuple[str, str]]) -> Tuple[str, str]:
        """
        Get primary domain from collage.
        
        When multiple domains detected, identify the most prominent.
        """
        if not domains:
            return ('text', 'generic')
        
        # Priority order (most specific first)
        priority = ['nba', 'nfl', 'mlb', 'tech', 'dating', 'professional', 'corporate']
        
        for subdomain in priority:
            for domain, sub in domains:
                if sub == subdomain:
                    return (domain, sub)
        
        # Return first if no priority match
        return domains[0]
    
    def analyze_domain_collage(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """
        Full analysis of domain collage in text pair.
        
        Returns:
        - All domains detected
        - Primary domain
        - Domain complexity (how many domains)
        - Confidence in each
        """
        detected = self.detect_domains(text_a, text_b)
        primary = self.get_primary_domain(detected)
        
        return {
            'all_domains': detected,
            'primary_domain': primary,
            'domain_count': len(detected),
            'is_multi_domain': len(detected) > 1,
            'complexity': 'single' if len(detected) == 1 else 'collage',
            'domains_str': ', '.join([f"{d}/{s}" for d, s in detected])
        }

