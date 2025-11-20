"""
Universal Analyzer - Enhanced with Framework 2.0

Complete narrative analysis with all new features:
- Instance-level π (π_effective)
- Blind Narratio (Β)
- Awareness amplification (θ_amp)
- Imperative gravity neighbors
- Cross-domain transfer predictions
- Concurrent narrative analysis

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.story_instance import StoryInstance
from config.domain_config import DomainConfig
from analysis.complexity_scorer import ComplexityScorer
from analysis.blind_narratio_calculator import BlindNarratioCalculator
from transformers.awareness_amplification import AwarenessAmplificationTransformer
from physics.imperative_gravity import ImperativeGravityCalculator
from analysis.multi_stream_narrative_processor import MultiStreamNarrativeProcessor
from learning.cross_domain_transfer import CrossDomainTransferLearner


class UniversalAnalyzerEnhanced:
    """
    Complete narrative analysis with Framework 2.0 enhancements.
    
    Analyzes narratives with:
    - All 60+ transformers
    - Instance-specific π_effective
    - Blind Narratio calculation
    - Awareness amplification detection
    - Imperative gravity neighbors
    - Cross-domain transfer learning
    - Concurrent narrative streams
    """
    
    def __init__(
        self,
        all_domain_configs: Optional[Dict[str, DomainConfig]] = None,
        repository: Optional[Any] = None
    ):
        """
        Initialize enhanced analyzer.
        
        Parameters
        ----------
        all_domain_configs : dict, optional
            All domain configurations
        repository : InstanceRepository, optional
            Instance repository for cross-domain queries
        """
        self.domain_configs = all_domain_configs or self._load_default_configs()
        self.repository = repository
        
        # Initialize components
        self.complexity_scorer = None
        self.blind_narratio_calc = BlindNarratioCalculator()
        self.awareness_transformer = AwarenessAmplificationTransformer()
        self.imperative_calculator = ImperativeGravityCalculator(self.domain_configs)
        self.stream_processor = MultiStreamNarrativeProcessor()
        
        # Transfer learner (if repository available)
        if self.repository:
            self.transfer_learner = CrossDomainTransferLearner(
                self.repository,
                self.domain_configs,
                self.imperative_calculator
            )
        else:
            self.transfer_learner = None
        
        # Fit awareness transformer (use default patterns)
        self._fit_awareness_transformer()
    
    def analyze_narrative_complete(
        self,
        narrative_text: str,
        narrative_id: Optional[str] = None,
        domain: Optional[str] = None,
        outcome: Optional[float] = None,
        include_transfer: bool = True,
        include_streams: bool = True
    ) -> Dict[str, Any]:
        """
        Complete analysis with all Framework 2.0 features.
        
        Parameters
        ----------
        narrative_text : str
            Narrative to analyze
        narrative_id : str, optional
            Identifier
        domain : str, optional
            Domain (will auto-detect if None)
        outcome : float, optional
            Known outcome
        include_transfer : bool
            Include cross-domain transfer learning
        include_streams : bool
            Include concurrent narrative analysis
        
        Returns
        -------
        dict
            Complete analysis results
        """
        if narrative_id is None:
            narrative_id = f"narrative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Detect domain if not provided
        if domain is None:
            domain = self._detect_domain(narrative_text)
        
        # Get domain config
        config = self.domain_configs.get(domain)
        if config is None:
            config = DomainConfig('generic')
        
        # Create StoryInstance
        instance = StoryInstance(
            instance_id=narrative_id,
            domain=domain,
            narrative_text=narrative_text,
            outcome=outcome
        )
        
        # Calculate complexity
        if self.complexity_scorer is None:
            self.complexity_scorer = ComplexityScorer(domain=domain)
        
        complexity = self.complexity_scorer.calculate_complexity(instance, narrative_text)
        
        # Calculate π_effective
        pi_eff = config.calculate_effective_pi(complexity)
        instance.pi_effective = pi_eff
        instance.pi_domain_base = config.get_pi()
        
        # Extract awareness amplification features
        awareness_features = self.awareness_transformer.transform([narrative_text])[0]
        instance.theta_amplification = awareness_features[-1]  # Aggregate score
        instance.awareness_features = dict(zip(
            self.awareness_transformer.get_feature_names(),
            awareness_features
        ))
        
        # Calculate Blind Narratio
        blind_narratio = self.blind_narratio_calc.calculate_instance_blind_narratio(instance)
        
        # Find imperative gravity neighbors
        all_domains = list(self.domain_configs.keys())
        imperative_neighbors = self.imperative_calculator.find_gravitational_neighbors(
            instance,
            all_domains,
            n_neighbors=5,
            exclude_same_domain=True
        )
        
        # Add to instance
        for neighbor_domain, force in imperative_neighbors:
            instance.add_imperative_gravity(neighbor_domain, "domain_aggregate", force)
        
        # Concurrent narrative analysis (if requested and text long enough)
        stream_analysis = None
        if include_streams and len(narrative_text) > 500:
            try:
                stream_features = self.stream_processor.extract_stream_features_for_genome(
                    narrative_text,
                    narrative_id
                )
                instance.genome_concurrent = stream_features
                instance.stream_count = int(stream_features[0] * 100)  # Approximate
                
                # Get full stream analysis
                stream_analysis = self.stream_processor.discover_streams(
                    narrative_text,
                    narrative_id
                )
            except Exception as e:
                print(f"  Warning: Stream analysis failed: {e}")
                stream_analysis = {'error': str(e)}
        
        # Cross-domain transfer prediction (if repository available)
        transfer_prediction = None
        if include_transfer and self.transfer_learner and self.repository:
            # Simplified prediction for demo
            # In production, would use actual trained model
            base_prediction = instance.story_quality if instance.story_quality else 0.5
            
            try:
                transfer_result = self.transfer_learner.predict_with_transfer(
                    instance,
                    base_prediction,
                    n_neighbors=3
                )
                transfer_prediction = transfer_result
            except Exception as e:
                print(f"  Warning: Transfer prediction failed: {e}")
        
        # Compile complete results
        result = {
            'instance': {
                'id': instance.instance_id,
                'domain': instance.domain,
                'text_length': len(narrative_text),
                'analyzed_at': datetime.now().isoformat()
            },
            'complexity': {
                'overall': float(complexity),
                'factors': instance.complexity_factors
            },
            'narrativity': {
                'pi_base': float(instance.pi_domain_base),
                'pi_effective': float(instance.pi_effective),
                'pi_variation': float(instance.pi_effective - instance.pi_domain_base),
                'interpretation': self._interpret_pi_variation(instance.pi_effective, instance.pi_domain_base)
            },
            'blind_narratio': {
                'value': float(blind_narratio) if not np.isinf(blind_narratio) else 'infinity',
                'interpretation': self._interpret_blind_narratio(blind_narratio)
            },
            'awareness': {
                'theta_amplification': float(instance.theta_amplification),
                'top_features': self._get_top_awareness_features(instance.awareness_features, n=5),
                'interpretation': self._interpret_awareness(instance.theta_amplification)
            },
            'imperative_gravity': {
                'neighbors': [
                    {
                        'domain': domain,
                        'force': float(force),
                        'explanation': self.imperative_calculator.explain_gravitational_pull(instance, domain)
                    }
                    for domain, force in imperative_neighbors[:3]
                ],
                'strongest_pull': imperative_neighbors[0][0] if imperative_neighbors else None
            },
            'concurrent_narratives': stream_analysis if stream_analysis else {'note': 'Not analyzed (text too short or disabled)'},
            'cross_domain_prediction': transfer_prediction if transfer_prediction else {'note': 'Not available (repository required)'}
        }
        
        return result
    
    def _load_default_configs(self) -> Dict[str, DomainConfig]:
        """Load default set of domain configs."""
        default_domains = [
            'golf', 'tennis', 'chess', 'boxing', 'wwe', 'oscars',
            'nba', 'nfl', 'supreme_court', 'startups', 'movies'
        ]
        
        configs = {}
        for domain in default_domains:
            try:
                configs[domain] = DomainConfig(domain)
            except:
                pass
        
        return configs
    
    def _fit_awareness_transformer(self):
        """Fit awareness transformer on default corpus."""
        default_corpus = [
            "I know this is my moment.",
            "Everything is on the line.",
            "The story of my life.",
            "Just another day.",
            "For everyone who believed in me."
        ]
        self.awareness_transformer.fit(default_corpus)
    
    def _detect_domain(self, narrative_text: str) -> str:
        """Simple domain detection (can be enhanced)."""
        text_lower = narrative_text.lower()
        
        # Simple keyword matching
        if any(word in text_lower for word in ['golf', 'masters', 'putt', 'course']):
            return 'golf'
        elif any(word in text_lower for word in ['court', 'justice', 'constitutional', 'precedent']):
            return 'supreme_court'
        elif any(word in text_lower for word in ['tennis', 'serve', 'volley', 'wimbledon']):
            return 'tennis'
        elif any(word in text_lower for word in ['oscar', 'academy award', 'film']):
            return 'oscars'
        elif any(word in text_lower for word in ['startup', 'founder', 'vc', 'funding']):
            return 'startups'
        else:
            return 'generic'
    
    def _interpret_pi_variation(self, pi_eff: float, pi_base: float) -> str:
        """Interpret π variation."""
        diff = pi_eff - pi_base
        
        if abs(diff) < 0.05:
            return "Typical complexity - π at domain baseline"
        elif diff > 0.15:
            return "High complexity - narrative matters significantly more than typical"
        elif diff > 0.05:
            return "Moderately complex - narrative matters somewhat more"
        elif diff < -0.15:
            return "Low complexity - evidence/rules dominate more than typical"
        elif diff < -0.05:
            return "Simple instance - less narrative influence than typical"
        else:
            return "Near baseline complexity"
    
    def _interpret_blind_narratio(self, beta: float) -> str:
        """Interpret Blind Narratio value."""
        if np.isinf(beta):
            return "Pure determinism - no free will resistance detected"
        elif beta > 2.0:
            return "High determinism - deterministic forces dominate strongly"
        elif beta > 1.2:
            return "Determinism-favored - deterministic forces somewhat stronger"
        elif beta > 0.8:
            return "Balanced - deterministic and free will forces in equilibrium"
        elif beta > 0.5:
            return "Free will-favored - conscious choice somewhat stronger"
        else:
            return "High free will - conscious agency dominates"
    
    def _interpret_awareness(self, theta_amp: float) -> str:
        """Interpret awareness amplification level."""
        if theta_amp > 0.7:
            return "High awareness - strong amplification of narrative potential likely"
        elif theta_amp > 0.4:
            return "Moderate awareness - some amplification expected"
        elif theta_amp > 0.2:
            return "Low awareness - minimal amplification"
        else:
            return "No detected awareness - baseline prediction"
    
    def _get_top_awareness_features(self, awareness_dict: Dict, n: int = 5) -> List[Dict]:
        """Get top N awareness features."""
        if not awareness_dict:
            return []
        
        # Sort by value
        sorted_features = sorted(
            awareness_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'feature': name, 'value': float(value)}
            for name, value in sorted_features[:n]
            if value > 0.01
        ]


def analyze_narrative_framework_2(
    narrative_text: str,
    domain: Optional[str] = None,
    outcome: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convenience function for complete Framework 2.0 analysis.
    
    Parameters
    ----------
    narrative_text : str
        Narrative to analyze
    domain : str, optional
        Domain (auto-detected if None)
    outcome : float, optional
        Known outcome
    
    Returns
    -------
    dict
        Complete analysis results
    """
    analyzer = UniversalAnalyzerEnhanced()
    return analyzer.analyze_narrative_complete(
        narrative_text,
        domain=domain,
        outcome=outcome
    )


if __name__ == '__main__':
    # Example usage
    test_narrative = """
    Tiger Woods' historic comeback at the 2019 Masters Championship 
    was one of golf's greatest redemption stories. After years of 
    injuries and personal struggles, the 43-year-old showed remarkable 
    mental toughness and clutch performance under championship pressure 
    to win his fifth Masters title.
    """
    
    result = analyze_narrative_framework_2(test_narrative, domain='golf')
    
    print("\nFramework 2.0 Analysis Results:")
    print(f"  π_effective: {result['narrativity']['pi_effective']:.3f}")
    print(f"  Complexity: {result['complexity']['overall']:.3f}")
    print(f"  Blind Narratio: {result['blind_narratio']['value']}")
    print(f"  Awareness Amplification: {result['awareness']['theta_amplification']:.3f}")
    print(f"  Top imperative neighbor: {result['imperative_gravity']['strongest_pull']}")

