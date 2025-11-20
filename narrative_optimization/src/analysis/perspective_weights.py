"""
Perspective Weight Schemas

Define weight schemes for different narrative perspectives.
Each perspective emphasizes different aspects of narrative quality.
"""

from typing import Dict
from enum import Enum


class NarrativePerspective(str, Enum):
    """Narrative perspectives for quality calculation"""
    DIRECTOR = "director"  # Creator's intent, vision, execution
    AUDIENCE = "audience"  # Viewer engagement, emotion, accessibility
    CRITIC = "critic"  # Craft, innovation, cultural significance
    CHARACTER = "character"  # Character development, authenticity, arc
    CULTURAL = "cultural"  # Cultural resonance, relevance, zeitgeist
    META = "meta"  # Self-awareness, genre play, innovation
    AUTHORITY = "authority"  # Leadership, strategic vision (sports: coach, business: CEO)
    STAR = "star"  # Individual excellence, heroism (sports: players, movies: actors)
    COLLECTIVE = "collective"  # Team/organization identity, collective goals
    SUPPORTING = "supporting"  # Ensemble, role fulfillment, depth


class PerspectiveWeightSchemas:
    """
    Weight schemas for different narrative perspectives.
    
    Each perspective emphasizes different transformer categories:
    - Director: Intent, vision, execution, craft
    - Audience: Engagement, emotion, accessibility, entertainment
    - Critic: Innovation, craft, cultural significance, technical excellence
    - Character: Development, authenticity, arc, depth
    - Cultural: Resonance, relevance, zeitgeist, social impact
    - Meta: Self-awareness, genre play, innovation, reflexivity
    """
    
    @staticmethod
    def get_director_weights(п: float) -> Dict[str, float]:
        """
        Director perspective: Emphasizes intent, vision, execution.
        
        Focus:
        - Framing (how story is presented)
        - Authenticity (genuine vision)
        - Expertise (technical craft)
        - Conflict (narrative structure)
        """
        if п < 0.3:
            return {
                'framing': 0.25,
                'conflict': 0.20,
                'suspense': 0.15,
                'statistical': 0.15,
                'expertise': 0.10,
                'linguistic': 0.10,
                'authenticity': 0.05
            }
        elif п > 0.7:
            return {
                'framing': 0.20,
                'authenticity': 0.18,
                'narrative_potential': 0.15,
                'emotional_semantic': 0.12,
                'expertise': 0.10,
                'nominative': 0.10,
                'self_perception': 0.08,
                'cultural_context': 0.07
            }
        else:
            return {
                'framing': 0.18,
                'conflict': 0.15,
                'authenticity': 0.15,
                'narrative_potential': 0.12,
                'expertise': 0.10,
                'emotional_semantic': 0.10,
                'linguistic': 0.08,
                'statistical': 0.07,
                'cultural_context': 0.05
            }
    
    @staticmethod
    def get_audience_weights(п: float) -> Dict[str, float]:
        """
        Audience perspective: Emphasizes engagement, emotion, accessibility.
        
        Focus:
        - Emotional resonance
        - Engagement markers
        - Accessibility (linguistic simplicity)
        - Entertainment value
        """
        if п < 0.3:
            return {
                'emotional_semantic': 0.25,
                'conflict': 0.20,
                'suspense': 0.18,
                'linguistic': 0.15,  # Simplicity matters
                'statistical': 0.12,
                'ensemble': 0.10
            }
        elif п > 0.7:
            return {
                'emotional_semantic': 0.25,
                'narrative_potential': 0.18,
                'self_perception': 0.15,
                'authenticity': 0.12,
                'nominative': 0.10,
                'linguistic': 0.10,
                'cultural_context': 0.10
            }
        else:
            return {
                'emotional_semantic': 0.22,
                'conflict': 0.15,
                'narrative_potential': 0.15,
                'linguistic': 0.12,
                'authenticity': 0.12,
                'suspense': 0.10,
                'ensemble': 0.08,
                'statistical': 0.06
            }
    
    @staticmethod
    def get_critic_weights(п: float) -> Dict[str, float]:
        """
        Critic perspective: Emphasizes craft, innovation, cultural significance.
        
        Focus:
        - Technical excellence
        - Innovation
        - Cultural significance
        - Craftsmanship
        """
        if п < 0.3:
            return {
                'expertise': 0.25,
                'linguistic': 0.20,
                'conflict': 0.15,
                'suspense': 0.15,
                'statistical': 0.12,
                'framing': 0.13
            }
        elif п > 0.7:
            return {
                'expertise': 0.20,
                'cultural_context': 0.18,
                'authenticity': 0.15,
                'emotional_semantic': 0.12,
                'nominative': 0.12,
                'narrative_potential': 0.10,
                'linguistic': 0.08,
                'self_perception': 0.05
            }
        else:
            return {
                'expertise': 0.20,
                'cultural_context': 0.18,
                'authenticity': 0.15,
                'conflict': 0.12,
                'emotional_semantic': 0.12,
                'linguistic': 0.10,
                'narrative_potential': 0.08,
                'statistical': 0.05
            }
    
    @staticmethod
    def get_character_weights(п: float) -> Dict[str, float]:
        """
        Character perspective: Emphasizes development, authenticity, arc.
        
        Focus:
        - Character depth
        - Authenticity
        - Development arc
        - Self-perception
        """
        if п < 0.3:
            return {
                'self_perception': 0.30,
                'authenticity': 0.25,
                'narrative_potential': 0.20,
                'linguistic': 0.15,
                'ensemble': 0.10
            }
        elif п > 0.7:
            return {
                'self_perception': 0.25,
                'nominative': 0.20,
                'authenticity': 0.18,
                'narrative_potential': 0.15,
                'emotional_semantic': 0.12,
                'phonetic': 0.05,
                'social_status': 0.05
            }
        else:
            return {
                'self_perception': 0.22,
                'authenticity': 0.20,
                'narrative_potential': 0.18,
                'nominative': 0.15,
                'emotional_semantic': 0.12,
                'linguistic': 0.08,
                'ensemble': 0.05
            }
    
    @staticmethod
    def get_cultural_weights(п: float) -> Dict[str, float]:
        """
        Cultural perspective: Emphasizes resonance, relevance, zeitgeist.
        
        Focus:
        - Cultural context
        - Relevance
        - Social impact
        - Zeitgeist alignment
        """
        if п < 0.3:
            return {
                'cultural_context': 0.35,
                'linguistic': 0.20,
                'statistical': 0.15,
                'ensemble': 0.15,
                'conflict': 0.15
            }
        elif п > 0.7:
            return {
                'cultural_context': 0.30,
                'nominative': 0.20,
                'emotional_semantic': 0.15,
                'authenticity': 0.12,
                'narrative_potential': 0.10,
                'social_status': 0.08,
                'phonetic': 0.05
            }
        else:
            return {
                'cultural_context': 0.28,
                'emotional_semantic': 0.18,
                'nominative': 0.15,
                'authenticity': 0.12,
                'narrative_potential': 0.12,
                'linguistic': 0.10,
                'ensemble': 0.05
            }
    
    @staticmethod
    def get_meta_weights(п: float) -> Dict[str, float]:
        """
        Meta perspective: Emphasizes self-awareness, genre play, innovation.
        
        Focus:
        - Self-awareness
        - Genre subversion
        - Innovation
        - Reflexivity
        """
        if п < 0.3:
            return {
                'framing': 0.30,
                'conflict': 0.20,
                'suspense': 0.18,
                'expertise': 0.15,
                'linguistic': 0.17
            }
        elif п > 0.7:
            return {
                'authenticity': 0.22,
                'narrative_potential': 0.20,
                'framing': 0.18,
                'emotional_semantic': 0.15,
                'cultural_context': 0.12,
                'nominative': 0.08,
                'expertise': 0.05
            }
        else:
            return {
                'framing': 0.22,
                'authenticity': 0.20,
                'narrative_potential': 0.18,
                'conflict': 0.15,
                'emotional_semantic': 0.12,
                'cultural_context': 0.08,
                'expertise': 0.05
            }
    
    @staticmethod
    def get_authority_weights(п: float) -> Dict[str, float]:
        """
        Authority perspective: Leadership, strategic vision (coach, CEO, director).
        
        Focus:
        - Strategic vision
        - Leadership markers
        - Decision-making
        - Control and authority
        """
        if п < 0.3:
            return {
                'expertise': 0.30,
                'framing': 0.25,
                'conflict': 0.20,
                'statistical': 0.15,
                'linguistic': 0.10
            }
        elif п > 0.7:
            return {
                'expertise': 0.25,
                'authenticity': 0.20,
                'narrative_potential': 0.18,
                'framing': 0.15,
                'nominative': 0.12,
                'self_perception': 0.10
            }
        else:
            return {
                'expertise': 0.25,
                'framing': 0.20,
                'authenticity': 0.18,
                'narrative_potential': 0.15,
                'conflict': 0.12,
                'linguistic': 0.10
            }
    
    @staticmethod
    def get_star_weights(п: float) -> Dict[str, float]:
        """
        Star perspective: Individual excellence, heroism (players, actors).
        
        Focus:
        - Individual excellence
        - Hero narratives
        - Star power
        - Personal arc
        """
        if п < 0.3:
            return {
                'self_perception': 0.35,
                'narrative_potential': 0.25,
                'authenticity': 0.20,
                'linguistic': 0.20
            }
        elif п > 0.7:
            return {
                'self_perception': 0.28,
                'nominative': 0.22,
                'narrative_potential': 0.18,
                'authenticity': 0.15,
                'emotional_semantic': 0.12,
                'phonetic': 0.05
            }
        else:
            return {
                'self_perception': 0.25,
                'narrative_potential': 0.22,
                'authenticity': 0.18,
                'nominative': 0.15,
                'emotional_semantic': 0.12,
                'linguistic': 0.08
            }
    
    @staticmethod
    def get_collective_weights(п: float) -> Dict[str, float]:
        """
        Collective perspective: Team/organization identity, collective goals.
        
        Focus:
        - Ensemble effects
        - Team chemistry
        - Collective identity
        - Group dynamics
        """
        if п < 0.3:
            return {
                'ensemble': 0.35,
                'relational': 0.25,
                'statistical': 0.20,
                'linguistic': 0.20
            }
        elif п > 0.7:
            return {
                'ensemble': 0.30,
                'relational': 0.25,
                'nominative': 0.18,
                'narrative_potential': 0.12,
                'authenticity': 0.10,
                'cultural_context': 0.05
            }
        else:
            return {
                'ensemble': 0.28,
                'relational': 0.22,
                'narrative_potential': 0.18,
                'nominative': 0.15,
                'authenticity': 0.12,
                'linguistic': 0.05
            }
    
    @staticmethod
    def get_supporting_weights(п: float) -> Dict[str, float]:
        """
        Supporting perspective: Ensemble, role fulfillment, depth.
        
        Focus:
        - Role fulfillment
        - Supporting narrative
        - Ensemble depth
        - Complementary features
        """
        if п < 0.3:
            return {
                'ensemble': 0.40,
                'relational': 0.30,
                'linguistic': 0.30
            }
        elif п > 0.7:
            return {
                'ensemble': 0.35,
                'relational': 0.28,
                'nominative': 0.15,
                'authenticity': 0.12,
                'narrative_potential': 0.10
            }
        else:
            return {
                'ensemble': 0.32,
                'relational': 0.25,
                'narrative_potential': 0.18,
                'nominative': 0.15,
                'authenticity': 0.10
            }
    
    @classmethod
    def get_weights(cls, perspective: NarrativePerspective, п: float) -> Dict[str, float]:
        """
        Get weight schema for a perspective.
        
        Parameters
        ----------
        perspective : NarrativePerspective
            Perspective to get weights for
        п : float
            Domain narrativity
            
        Returns
        -------
        weights : dict
            Transformer name -> weight mapping
        """
        method_map = {
            NarrativePerspective.DIRECTOR: cls.get_director_weights,
            NarrativePerspective.AUDIENCE: cls.get_audience_weights,
            NarrativePerspective.CRITIC: cls.get_critic_weights,
            NarrativePerspective.CHARACTER: cls.get_character_weights,
            NarrativePerspective.CULTURAL: cls.get_cultural_weights,
            NarrativePerspective.META: cls.get_meta_weights,
            NarrativePerspective.AUTHORITY: cls.get_authority_weights,
            NarrativePerspective.STAR: cls.get_star_weights,
            NarrativePerspective.COLLECTIVE: cls.get_collective_weights,
            NarrativePerspective.SUPPORTING: cls.get_supporting_weights,
        }
        
        method = method_map.get(perspective)
        if method is None:
            raise ValueError(f"Unknown perspective: {perspective}")
        
        return method(п)

