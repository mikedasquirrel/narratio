"""
Domain Expertise Tracker

"The analyzer should integrate the degree of knowledge/familiarity it has
with the collage of domains that arise in the particular analysis"

Tracks what domains we've trained on, how much data we have,
and calibrates confidence accordingly.
"""

from typing import Dict, List, Any, Tuple
import json
from pathlib import Path


class DomainExpertiseTracker:
    """
    Tracks system's expertise across domains and sub-domains.
    
    Maintains registry of:
    - What domains we've trained on
    - How many training samples
    - Model performance in each domain
    - Confidence scores
    """
    
    def __init__(self):
        """Initialize domain expertise registry."""
        
        self.expertise_registry = {
            'sports': {
                'nba': {
                    'training_samples': 10749,
                    'test_samples': 1230,
                    'features': 134,
                    'accuracy': 0.589,
                    'source': '11,979 real NBA games (2014-2024)',
                    'confidence': 0.85,  # High confidence
                    'model_path': 'models/nba_optimized.pkl',
                    'last_updated': '2025-11-10',
                    'notes': 'Real data from NBA.com API, narrative + nominative features'
                },
                'generic_sports': {
                    'training_samples': 500,
                    'accuracy': 0.55,
                    'confidence': 0.45,  # Moderate
                    'notes': 'Limited generic sports data'
                }
            },
            
            'text_classification': {
                'news': {
                    'training_samples': 400,
                    'features': 114,
                    'accuracy': 0.68,
                    'source': '20newsgroups dataset',
                    'confidence': 0.65,
                    'notes': 'Standard text classification benchmark'
                }
            },
            
            'products': {
                'generic': {
                    'training_samples': 0,
                    'confidence': 0.0,
                    'notes': 'No product training data yet'
                }
            },
            
            'profiles': {
                'generic': {
                    'training_samples': 0,
                    'confidence': 0.0,
                    'notes': 'No profile training data yet'
                }
            },
            
            'brands': {
                'generic': {
                    'training_samples': 0,
                    'confidence': 0.0,
                    'notes': 'No brand training data yet'
                }
            }
        }
    
    def assess_expertise(self, detected_domains: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Assess system's expertise for detected domains.
        
        Parameters
        ----------
        detected_domains : list of (domain, subdomain) tuples
            E.g., [('sports', 'nba'), ('achievement', 'psychology')]
        
        Returns
        -------
        expertise : dict
            Overall confidence, domain breakdown, can_predict flag
        """
        domain_confidences = {}
        domain_details = {}
        
        for domain, subdomain in detected_domains:
            # Look up expertise
            if domain in self.expertise_registry:
                if subdomain in self.expertise_registry[domain]:
                    expertise = self.expertise_registry[domain][subdomain]
                    confidence = expertise['confidence']
                    samples = expertise.get('training_samples', 0)
                else:
                    # Check for generic domain expertise
                    if 'generic' in self.expertise_registry[domain]:
                        expertise = self.expertise_registry[domain]['generic']
                        confidence = expertise['confidence'] * 0.7  # Penalty for not exact match
                        samples = expertise.get('training_samples', 0)
                    else:
                        confidence = 0.0
                        samples = 0
                        expertise = {}
            else:
                confidence = 0.0
                samples = 0
                expertise = {}
            
            domain_key = f"{domain}/{subdomain}"
            domain_confidences[domain_key] = confidence
            domain_details[domain_key] = {
                'confidence': confidence,
                'training_samples': samples,
                'accuracy': expertise.get('accuracy', 0.0),
                'notes': expertise.get('notes', 'Unknown domain')
            }
        
        # Overall confidence = weighted average (higher sample counts weight more)
        if domain_confidences:
            # Weight by training samples
            total_samples = sum(domain_details[d]['training_samples'] for d in domain_details)
            
            if total_samples > 0:
                overall_confidence = sum(
                    domain_details[d]['confidence'] * domain_details[d]['training_samples']
                    for d in domain_details
                ) / total_samples
            else:
                overall_confidence = sum(domain_confidences.values()) / len(domain_confidences)
        else:
            overall_confidence = 0.0
        
        return {
            'overall_confidence': float(overall_confidence),
            'can_predict': overall_confidence > 0.30,
            'confidence_level': self._get_confidence_level(overall_confidence),
            'domain_breakdown': domain_details,
            'domains_detected': detected_domains,
            'total_training_samples': sum(d['training_samples'] for d in domain_details.values()),
            'recommendation': self._get_recommendation(overall_confidence)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level."""
        if confidence > 0.75:
            return "HIGH - Extensive training data available"
        elif confidence > 0.50:
            return "MODERATE - Some training data available"
        elif confidence > 0.30:
            return "LOW - Limited training data"
        else:
            return "VERY LOW - Insufficient training data"
    
    def _get_recommendation(self, confidence: float) -> str:
        """Provide recommendation based on confidence."""
        if confidence > 0.75:
            return "System is well-trained in this domain. Predictions are reliable."
        elif confidence > 0.50:
            return "System has moderate expertise. Predictions should be taken as educated guesses."
        elif confidence > 0.30:
            return "System has limited expertise. Predictions are speculative. Use with caution."
        else:
            return "System lacks training data for this domain. Cannot make reliable predictions. Consider this a new domain."
    
    def update_expertise(self, domain: str, subdomain: str, new_data: Dict):
        """Update expertise registry when new training data is added."""
        if domain not in self.expertise_registry:
            self.expertise_registry[domain] = {}
        
        if subdomain not in self.expertise_registry[domain]:
            self.expertise_registry[domain][subdomain] = {
                'training_samples': 0,
                'confidence': 0.0
            }
        
        # Update
        self.expertise_registry[domain][subdomain].update(new_data)
        
        # Recalculate confidence based on sample size
        samples = self.expertise_registry[domain][subdomain]['training_samples']
        
        if samples > 10000:
            confidence = 0.95
        elif samples > 5000:
            confidence = 0.85
        elif samples > 1000:
            confidence = 0.70
        elif samples > 100:
            confidence = 0.50
        else:
            confidence = 0.30
        
        self.expertise_registry[domain][subdomain]['confidence'] = confidence
    
    def get_expertise_report(self) -> str:
        """Generate human-readable expertise report."""
        report = "DOMAIN EXPERTISE REPORT\n"
        report += "="*70 + "\n\n"
        
        for domain, subdomains in self.expertise_registry.items():
            report += f"{domain.upper()}:\n"
            for subdomain, expertise in subdomains.items():
                samples = expertise.get('training_samples', 0)
                conf = expertise.get('confidence', 0.0)
                report += f"  • {subdomain}: {samples} samples, {conf:.0%} confidence\n"
            report += "\n"
        
        return report

