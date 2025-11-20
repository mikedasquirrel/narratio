"""
Domain-Specific Archetype Registry

Defines domain-specific Ξ (golden narratio) patterns discovered from analyses.
Each domain has its own archetypal perfection that defines "great story" in that context.

Author: Narrative Integration System
Date: November 2025
"""

from typing import Dict, List, Any

DOMAIN_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    'golf': {
        'archetype_patterns': {
            'mental_game': ['all mental', 'between the ears', 'choking', 'clutch', 'composure', 'psychological'],
            'elite_skill': ['world-class', 'elite level', 'years of training', 'tour level', 'championship caliber'],
            'course_mastery': ['course knowledge', 'familiar with layout', 'previous winner', 'course record'],
            'pressure_performance': ['Sunday pressure', 'final round', 'made the putt', 'held his nerve'],
            'veteran_wisdom': ['experience', 'been here before', 'knows what it takes', 'veteran presence']
        },
        'nominative_richness_requirement': 30,  # proper nouns needed
        'archetype_weights': {
            'mental_game': 0.30,
            'elite_skill': 0.25,
            'course_mastery': 0.20,
            'pressure_performance': 0.15,
            'veteran_wisdom': 0.10
        },
        'pi': 0.70,
        'theta_range': (0.50, 0.65),
        'lambda_range': (0.65, 0.75)
    },
    
    'boxing': {
        'archetype_patterns': {
            'fighting_style': ['orthodox', 'southpaw', 'counter-puncher', 'brawler', 'technical boxer'],
            'physical_dominance': ['knockout power', 'devastating', 'overwhelming', 'dominant performance'],
            'underdog_story': ['underdog', 'counted out', 'proved doubters wrong', 'comeback'],
            'ring_generalship': ['controlled the ring', 'dictated pace', 'ring IQ', 'tactical'],
            'warrior_spirit': ['heart', 'toughness', 'refused to quit', 'warrior mentality']
        },
        'nominative_richness_requirement': 15,
        'archetype_weights': {
            'physical_dominance': 0.35,
            'fighting_style': 0.25,
            'warrior_spirit': 0.20,
            'ring_generalship': 0.12,
            'underdog_story': 0.08
        },
        'pi': 0.743,
        'theta_range': (0.80, 0.95),  # Very high - this is the suppression problem
        'lambda_range': (0.40, 0.50)
    },
    
    'nba': {
        'archetype_patterns': {
            'team_chemistry': ['chemistry', 'playing together', 'unselfish', 'ball movement', 'trust'],
            'star_narrative': ['MVP', 'all-star', 'carried the team', 'took over', 'clutch performance'],
            'momentum': ['win streak', 'hot streak', 'rolling', 'confidence', 'on fire'],
            'coaching': ['game plan', 'adjustments', 'coaching', 'system', 'execution'],
            'matchup_advantage': ['matchup', 'exploited', 'advantage', 'favorable', 'mismatch']
        },
        'nominative_richness_requirement': 25,  # 10 players per team
        'archetype_weights': {
            'star_narrative': 0.30,
            'team_chemistry': 0.25,
            'momentum': 0.20,
            'coaching': 0.15,
            'matchup_advantage': 0.10
        },
        'pi': 0.49,
        'theta_range': (0.25, 0.35),
        'lambda_range': (0.70, 0.80)
    },
    
    'wwe': {
        'archetype_patterns': {
            'character_arc': ['character development', 'storyline', 'arc', 'journey', 'transformation'],
            'betrayal_redemption': ['betrayed', 'turned on', 'redemption', 'vindication', 'revenge'],
            'meta_awareness': ['kayfabe', 'work rate', 'psychology', 'selling', 'smart marks'],
            'long_term_payoff': ['years in the making', 'payoff', 'culmination', 'historic', 'legacy'],
            'crowd_reaction': ['pop', 'heat', 'over', 'reaction', 'engagement']
        },
        'nominative_richness_requirement': 20,
        'archetype_weights': {
            'character_arc': 0.30,
            'long_term_payoff': 0.25,
            'betrayal_redemption': 0.20,
            'meta_awareness': 0.15,
            'crowd_reaction': 0.10
        },
        'pi': 0.974,
        'theta_range': (0.50, 0.60),  # High awareness but amplifies (prestige)
        'lambda_range': (0.45, 0.55),
        'prestige_domain': True  # Uses inverted equation
    },
    
    'tennis': {
        'archetype_patterns': {
            'mental_toughness': ['mental strength', 'composure', 'focus', 'concentration', 'clutch'],
            'surface_mastery': ['clay court', 'grass court', 'hard court', 'surface', 'adapted'],
            'rivalry': ['rivalry', 'head-to-head', 'history', 'nemesis', 'matchup'],
            'grand_slam_pressure': ['grand slam', 'major', 'championship point', 'title', 'legacy'],
            'physical_conditioning': ['fitness', 'endurance', 'conditioning', 'stamina', 'physical']
        },
        'nominative_richness_requirement': 25,
        'archetype_weights': {
            'mental_toughness': 0.30,
            'grand_slam_pressure': 0.25,
            'surface_mastery': 0.20,
            'rivalry': 0.15,
            'physical_conditioning': 0.10
        },
        'pi': 0.75,
        'theta_range': (0.48, 0.55),
        'lambda_range': (0.75, 0.85)
    },
    
    'chess': {
        'archetype_patterns': {
            'strategic_depth': ['strategy', 'positional', 'tactical', 'calculation', 'planning'],
            'opening_theory': ['opening', 'preparation', 'theory', 'book', 'novelty'],
            'endgame_mastery': ['endgame', 'technique', 'precise', 'conversion', 'winning'],
            'time_pressure': ['time trouble', 'blitz', 'rapid', 'clock', 'pressure'],
            'psychological': ['psychology', 'mind games', 'intimidation', 'confidence', 'mental']
        },
        'nominative_richness_requirement': 20,
        'archetype_weights': {
            'strategic_depth': 0.30,
            'opening_theory': 0.25,
            'endgame_mastery': 0.20,
            'time_pressure': 0.15,
            'psychological': 0.10
        },
        'pi': 0.78,
        'theta_range': (0.40, 0.50),
        'lambda_range': (0.70, 0.80)
    },
    
    'oscars': {
        'archetype_patterns': {
            'campaign_narrative': ['campaign', 'awards season', 'for your consideration', 'buzz', 'momentum'],
            'cultural_moment': ['cultural', 'relevant', 'timely', 'resonance', 'zeitgeist'],
            'emotional_resonance': ['emotional', 'moving', 'powerful', 'touching', 'heartfelt'],
            'technical_excellence': ['cinematography', 'direction', 'acting', 'screenplay', 'production'],
            'prestige_factors': ['prestigious', 'esteemed', 'acclaimed', 'revered', 'honored']
        },
        'nominative_richness_requirement': 30,
        'archetype_weights': {
            'campaign_narrative': 0.30,
            'cultural_moment': 0.25,
            'emotional_resonance': 0.20,
            'technical_excellence': 0.15,
            'prestige_factors': 0.10
        },
        'pi': 0.82,
        'theta_range': (0.35, 0.45),
        'lambda_range': (0.60, 0.70),
        'prestige_domain': True
    },
    
    'crypto': {
        'archetype_patterns': {
            'innovation': ['innovative', 'revolutionary', 'breakthrough', 'cutting-edge', 'next-gen'],
            'decentralization': ['decentralized', 'peer-to-peer', 'autonomous', 'trustless', 'permissionless'],
            'community': ['community', 'ecosystem', 'adoption', 'network', 'users'],
            'technical_legitimacy': ['blockchain', 'protocol', 'consensus', 'cryptography', 'security'],
            'use_case_clarity': ['use case', 'application', 'utility', 'purpose', 'solution']
        },
        'nominative_richness_requirement': 15,
        'archetype_weights': {
            'innovation': 0.30,
            'decentralization': 0.25,
            'community': 0.20,
            'technical_legitimacy': 0.15,
            'use_case_clarity': 0.10
        },
        'pi': 0.32,
        'theta_range': (0.35, 0.40),
        'lambda_range': (0.55, 0.60)
    },
    
    'mental_health': {
        'archetype_patterns': {
            'clinical_framing': ['clinical', 'diagnostic', 'symptom', 'syndrome', 'disorder'],
            'phonetic_severity': ['harsh sounds', 'phonetic', 'pronunciation', 'sound pattern'],
            'treatment_seeking': ['treatment', 'therapy', 'intervention', 'care', 'support'],
            'stigma_association': ['stigma', 'stereotyped', 'prejudice', 'discrimination', 'bias'],
            'severity_indicator': ['severe', 'mild', 'moderate', 'chronic', 'acute']
        },
        'nominative_richness_requirement': 20,
        'archetype_weights': {
            'clinical_framing': 0.30,
            'phonetic_severity': 0.25,
            'treatment_seeking': 0.20,
            'stigma_association': 0.15,
            'severity_indicator': 0.10
        },
        'pi': 0.55,
        'theta_range': (0.60, 0.65),
        'lambda_range': (0.55, 0.65)
    },
    
    'startups': {
        'archetype_patterns': {
            'market_fit': ['market fit', 'product-market', 'demand', 'need', 'problem'],
            'innovation': ['innovative', 'disruptive', 'breakthrough', 'novel', 'unique'],
            'execution': ['execution', 'delivery', 'implementation', 'results', 'traction'],
            'team_quality': ['team', 'founders', 'talent', 'expertise', 'experience'],
            'scalability': ['scalable', 'growth', 'expansion', 'scale', 'potential']
        },
        'nominative_richness_requirement': 20,
        'archetype_weights': {
            'market_fit': 0.30,
            'innovation': 0.25,
            'execution': 0.20,
            'team_quality': 0.15,
            'scalability': 0.10
        },
        'pi': 0.76,
        'theta_range': (0.50, 0.55),
        'lambda_range': (0.40, 0.45)
    },
    
    'hurricanes': {
        'archetype_patterns': {
            'naming_patterns': ['name', 'hurricane', 'storm', 'tropical', 'cyclone'],
            'threat_perception': ['threat', 'danger', 'risk', 'warning', 'alert'],
            'gender_association': ['feminine', 'masculine', 'gender', 'male', 'female'],
            'severity_indicators': ['category', 'intensity', 'strength', 'force', 'power'],
            'geographic_context': ['coastal', 'landfall', 'region', 'area', 'location']
        },
        'nominative_richness_requirement': 10,
        'archetype_weights': {
            'naming_patterns': 0.30,
            'threat_perception': 0.25,
            'gender_association': 0.20,
            'severity_indicators': 0.15,
            'geographic_context': 0.10
        },
        'pi': 0.30,
        'theta_range': (0.05, 0.10),
        'lambda_range': (0.10, 0.15)
    },
    
    'housing': {
        'archetype_patterns': {
            'numerology': ['number', 'thirteen', '13', 'unlucky', 'superstition'],
            'cultural_superstition': ['superstition', 'belief', 'tradition', 'cultural', 'taboo'],
            'street_valence': ['street', 'avenue', 'boulevard', 'way', 'road'],
            'address_prestige': ['prestigious', 'exclusive', 'desirable', 'prime', 'elite'],
            'location_association': ['location', 'neighborhood', 'area', 'district', 'zone']
        },
        'nominative_richness_requirement': 5,
        'archetype_weights': {
            'numerology': 0.40,
            'cultural_superstition': 0.25,
            'street_valence': 0.15,
            'address_prestige': 0.12,
            'location_association': 0.08
        },
        'pi': 0.92,
        'theta_range': (0.30, 0.40),
        'lambda_range': (0.05, 0.10)
    },
    
    # ============================================================================
    # CLASSICAL ARCHETYPE DOMAINS
    # Added: November 13, 2025
    # Theory-guided empirical discovery of narrative archetypes
    # ============================================================================
    
    'classical_literature': {
        'archetype_patterns': {
            'hero_journey_elements': ['quest', 'journey', 'transformation', 'trial', 'ordeal', 'return'],
            'character_depth': ['complex', 'multifaceted', 'developed', 'psychological', 'nuanced'],
            'thematic_resonance': ['theme', 'meaning', 'universal', 'truth', 'insight', 'wisdom'],
            'literary_craft': ['metaphor', 'symbolism', 'imagery', 'prose', 'style', 'voice'],
            'moral_framework': ['moral', 'ethical', 'virtue', 'vice', 'justice', 'redemption']
        },
        'nominative_richness_requirement': 25,  # Character names
        'archetype_weights': {
            'hero_journey_elements': 0.30,
            'character_depth': 0.25,
            'thematic_resonance': 0.20,
            'literary_craft': 0.15,
            'moral_framework': 0.10
        },
        'pi': 0.72,
        'theta_range': (0.45, 0.60),  # Varies widely by period
        'lambda_range': (0.50, 0.65),
        'classical_theory_expectations': {
            'campbell_journey_completion': 0.70,
            'jung_archetype_clarity': 0.65,
            'booker_plot_purity': 0.60,
            'aristotelian_quality': 0.60
        }
    },
    
    'mythology': {
        'archetype_patterns': {
            'hero_archetype': ['hero', 'champion', 'warrior', 'chosen', 'destined', 'brave'],
            'divine_intervention': ['god', 'goddess', 'divine', 'blessed', 'cursed', 'prophecy'],
            'quest_structure': ['quest', 'journey', 'seek', 'find', 'retrieve', 'achieve'],
            'supernatural_elements': ['magic', 'enchanted', 'spell', 'supernatural', 'mystical'],
            'moral_lesson': ['taught', 'learned', 'lesson', 'wisdom', 'truth', 'moral']
        },
        'nominative_richness_requirement': 35,  # Zeus, Odin, heroes, places
        'archetype_weights': {
            'hero_archetype': 0.30,
            'divine_intervention': 0.25,
            'quest_structure': 0.20,
            'supernatural_elements': 0.15,
            'moral_lesson': 0.10
        },
        'pi': 0.89,
        'theta_range': (0.20, 0.30),  # Low - straight mythic telling
        'lambda_range': (0.25, 0.35),  # Low - gods can do anything
        'classical_theory_expectations': {
            'campbell_journey_completion': 0.88,  # HIGHEST - Campbell's source
            'jung_archetype_clarity': 0.92,  # Pure archetypes
            'booker_plot_purity': 0.80,  # Clear plot types
            'frye_romance': 0.70  # Romance mythos dominant
        }
    },
    
    'scripture_parables': {
        'archetype_patterns': {
            'moral_teaching': ['teach', 'lesson', 'learn', 'understand', 'wisdom', 'truth'],
            'parable_structure': ['story', 'parable', 'told', 'said', 'example', 'illustrate'],
            'allegorical_depth': ['represent', 'symbolize', 'metaphor', 'meaning', 'signify'],
            'spiritual_truth': ['spiritual', 'soul', 'faith', 'belief', 'divine', 'sacred'],
            'practical_application': ['therefore', 'thus', 'so', 'apply', 'practice', 'live']
        },
        'nominative_richness_requirement': 15,  # Moderate - often symbolic
        'archetype_weights': {
            'moral_teaching': 0.35,
            'parable_structure': 0.25,
            'allegorical_depth': 0.20,
            'spiritual_truth': 0.12,
            'practical_application': 0.08
        },
        'pi': 0.81,
        'theta_range': (0.35, 0.45),  # Moderate - lessons are conscious
        'lambda_range': (0.55, 0.65),  # Moderate - teaching format
        'classical_theory_expectations': {
            'parable_completeness': 0.85,  # Setup → crisis → resolution → lesson
            'moral_clarity': 0.88,
            'memorability': 0.80,
            'universal_applicability': 0.75
        }
    },
    
    'film_extended': {
        'archetype_patterns': {
            'beat_adherence': ['catalyst', 'midpoint', 'all is lost', 'finale', 'turning point'],
            'visual_storytelling': ['shows', 'visual', 'see', 'appears', 'looks', 'reveals'],
            'character_psychology': ['wants', 'needs', 'fears', 'desires', 'struggles', 'internal'],
            'emotional_payoff': ['emotional', 'satisfying', 'cathartic', 'powerful', 'moving'],
            'pacing_rhythm': ['fast', 'slow', 'builds', 'tension', 'release', 'rhythm']
        },
        'nominative_richness_requirement': 20,  # Character names
        'archetype_weights': {
            'beat_adherence': 0.30,
            'character_psychology': 0.25,
            'visual_storytelling': 0.20,
            'emotional_payoff': 0.15,
            'pacing_rhythm': 0.10
        },
        'pi': 0.68,
        'theta_range': (0.40, 0.50),
        'lambda_range': (0.55, 0.65),
        'classical_theory_expectations': {
            'save_the_cat_beats': 0.75,  # Hollywood formula
            'hero_journey_completion': 0.72,  # High for blockbusters
            'three_act_structure': 0.85,
            'refusal_of_call_emphasis': 0.70  # Higher than mythology
        }
    },
    
    'music_narrative': {
        'archetype_patterns': {
            'lyrical_storytelling': ['story', 'tale', 'narrative', 'told', 'happened', 'once'],
            'emotional_arc': ['feel', 'emotion', 'heart', 'soul', 'love', 'pain', 'joy'],
            'character_voice': ['I', 'me', 'my', 'you', 'we', 'he', 'she'],
            'narrative_clarity': ['clear', 'understand', 'story', 'plot', 'sequence', 'then'],
            'universal_themes': ['love', 'loss', 'hope', 'dream', 'life', 'death', 'time']
        },
        'nominative_richness_requirement': 12,  # Artist, place names
        'archetype_weights': {
            'emotional_arc': 0.35,  # PRIMARY in music
            'lyrical_storytelling': 0.25,
            'character_voice': 0.20,
            'universal_themes': 0.12,
            'narrative_clarity': 0.08
        },
        'pi': 0.58,
        'theta_range': (0.35, 0.45),
        'lambda_range': (0.50, 0.60),
        'classical_theory_expectations': {
            'narrative_clarity': 0.60,  # Lower due to time constraints
            'emotional_resonance': 0.85,  # Higher - music's strength
            'journey_elements': 0.45,  # Compressed
            'concept_album_cohesion': 0.75  # For concept albums
        }
    },
    
    'stage_drama': {
        'archetype_patterns': {
            'dramatic_conflict': ['conflict', 'tension', 'struggle', 'oppose', 'battle', 'clash'],
            'tragic_elements': ['tragedy', 'tragic', 'hubris', 'downfall', 'flaw', 'catastrophe'],
            'aristotelian_unity': ['unity', 'focused', 'single', 'coherent', 'consistent'],
            'dialogue_quality': ['dialogue', 'speech', 'words', 'said', 'spoke', 'voice'],
            'cathartic_impact': ['catharsis', 'pity', 'fear', 'emotional', 'purge', 'release']
        },
        'nominative_richness_requirement': 20,  # Character names
        'archetype_weights': {
            'dramatic_conflict': 0.30,
            'tragic_elements': 0.25,
            'aristotelian_unity': 0.20,
            'dialogue_quality': 0.15,
            'cathartic_impact': 0.10
        },
        'pi': 0.76,
        'theta_range': (0.50, 0.65),  # Higher in modern drama
        'lambda_range': (0.70, 0.80),  # High - stage constraints
        'classical_theory_expectations': {
            'aristotelian_adherence': 0.75,  # Should validate Aristotle
            'five_act_structure': 0.70,  # Shakespeare
            'unity_of_action': 0.85,
            'peripeteia_anagnorisis': 0.75,  # Reversal + recognition
            'greek_tragedy_highest': True  # Highest Aristotelian scores
        }
    }
}


def get_generic_archetype() -> Dict[str, Any]:
    """
    Generic archetype for domains without specific configuration.
    Uses balanced patterns that can apply across multiple contexts.
    """
    return {
        'archetype_patterns': {
            'quality': ['excellent', 'outstanding', 'superior', 'exceptional', 'remarkable'],
            'achievement': ['achieved', 'accomplished', 'successful', 'victory', 'triumph'],
            'challenge': ['challenge', 'difficult', 'demanding', 'tough', 'hard'],
            'skill': ['skilled', 'talented', 'proficient', 'expert', 'mastery'],
            'narrative': ['story', 'journey', 'progression', 'development', 'arc']
        },
        'nominative_richness_requirement': 15,
        'archetype_weights': {
            'quality': 0.25,
            'achievement': 0.25,
            'challenge': 0.20,
            'skill': 0.20,
            'narrative': 0.10
        },
        'pi': 0.50,
        'theta_range': (0.40, 0.60),
        'lambda_range': (0.40, 0.60),
        'prestige_domain': False
    }

