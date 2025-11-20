"""
Basketball-Specific Narrative Transformer

Domain-optimized features for college/pro basketball.
Let the formula discover which narrative dimensions dominate.

Features (60 total):
- School tradition (founding, championships, legendary coaches)
- Tournament narrative weight (single elimination, rounds)
- Upset potential (underdog narrative strength)
- Coach pedigree (coaching tree, tournament history)
- Player composition (senior leadership vs freshmen)
- Rivalry intensity (conference, historical)
- Momentum within tournament (games within days)
- Court/crowd advantage (home, altitude, student section)
- Archetype features (powerhouse, cinderella, sleeper)
- Temporal factors (season timing, tournament round)
"""

from typing import List
import numpy as np
import re
from ..transformers.base import NarrativeTransformer

class BasketballNarrativeTransformer(NarrativeTransformer):
    """Domain-optimized transformer for basketball narratives."""
    
    def __init__(self):
        super().__init__(
            narrative_id="basketball_specific",
            description="Basketball-optimized narrative features"
        )
        
        # Basketball-specific lexicon
        self.tradition_markers = [
            'historic', 'legendary', 'tradition', 'championship', 'dynasty', 'powerhouse',
            'legacy', 'storied', 'blue blood', 'perennial', 'established', 'elite program'
        ]
        
        self.cinderella_markers = [
            'underdog', 'upset', 'cinderella', 'dark horse', 'surprise', 'unlikely',
            'overlooked', 'underestimated', 'giant killer', 'bracket buster'
        ]
        
        self.momentum_markers = [
            'hot', 'streak', 'rolling', 'surging', 'momentum', 'unstoppable',
            'rhythm', 'flow', 'clicking', 'firing', 'peaking', 'cruising'
        ]
        
        self.pressure_markers = [
            'elimination', 'must-win', 'do-or-die', 'final', 'championship',
            'pressure', 'clutch', 'crunch time', 'high stakes', 'winner take all'
        ]
        
        self.coach_markers = [
            'coach', 'coaching', 'hall of fame', 'legendary coach', 'tournament veteran',
            'championship coach', 'experienced', 'strategic', 'tactical genius'
        ]
        
        self.experience_markers = [
            'senior', 'veteran', 'experienced', 'battle-tested', 'proven',
            'tournament experience', 'been there before', 'leadership'
        ]
        
        self.youth_markers = [
            'freshman', 'young', 'inexperienced', 'first time', 'debut',
            'new', 'untested', 'learning', 'developing'
        ]
        
        self.home_court_markers = [
            'home', 'home court', 'crowd', 'fans', 'student section', 'hostile environment',
            'altitude', 'loud', 'atmosphere', 'fortress', 'advantage'
        ]
        
        self.rivalry_markers = [
            'rival', 'rivalry', 'conference', 'hatred', 'bad blood', 'history',
            'grudge match', 'always competitive', 'battle', 'clash'
        ]
    
    def fit(self, X, y=None):
        """Fit on basketball narratives."""
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Extract 60 basketball-specific features."""
        self._validate_fitted()
        return np.array([self._extract_basketball_features(text) for text in X])
    
    def _extract_basketball_features(self, text: str) -> np.ndarray:
        """Extract 60 domain-optimized basketball features."""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = max(1, len(words))
        
        features = []
        
        # === TRADITION / LEGACY (10 features) ===
        tradition_count = sum(1 for m in self.tradition_markers if m in text_lower)
        features.append(tradition_count / word_count)  # 1. Tradition density
        
        tradition_strength = min(10, tradition_count)
        features.append(tradition_strength / 10)  # 2. Tradition strength
        
        # School age proxy (mentions of years, decades)
        year_mentions = len(re.findall(r'\b\d{2,4}\b', text))
        features.append(year_mentions)  # 3. Historical depth
        
        # Championship mentions
        champ_mentions = text_lower.count('championship') + text_lower.count('title')
        features.append(champ_mentions)  # 4. Championship legacy
        
        # Blue blood indicators
        blue_blood = 1.0 if any(m in text_lower for m in ['duke', 'kentucky', 'north carolina', 'kansas', 'ucla']) else 0.0
        features.append(blue_blood)  # 5. Blue blood program
        
        # Powerhouse archetype
        powerhouse = 1.0 if tradition_count >= 3 else 0.5 if tradition_count >= 1 else 0.0
        features.append(powerhouse)  # 6. Powerhouse classification
        
        # Legacy weight (for context weighting)
        legacy_weight = 1.0 + (tradition_count * 0.5) + (champ_mentions * 0.3)
        features.append(min(3.0, legacy_weight))  # 7. Legacy narrative weight
        
        # Pedigree mentions (coach, program history)
        pedigree = sum(1 for m in ['pedigree', 'lineage', 'tradition', 'heritage'] if m in text_lower)
        features.append(pedigree)  # 8. Pedigree markers
        
        # Sustained excellence (consistent success language)
        sustained = sum(1 for m in ['consistent', 'always', 'perennial', 'every year'] if m in text_lower)
        features.append(sustained)  # 9. Sustained excellence
        
        # Overall tradition score
        tradition_score = (features[0] + features[1] + features[5]) / 3
        features.append(tradition_score)  # 10. Tradition composite
        
        # === UPSET / CINDERELLA (10 features) ===
        cinderella_count = sum(1 for m in self.cinderella_markers if m in text_lower)
        features.append(cinderella_count / word_count)  # 11. Cinderella density
        
        upset_potential = min(10, cinderella_count)
        features.append(upset_potential / 10)  # 12. Upset narrative strength
        
        underdog_explicit = 1.0 if 'underdog' in text_lower or 'lower seed' in text_lower else 0.0
        features.append(underdog_explicit)  # 13. Explicit underdog
        
        seed_differential = len(re.findall(r'#?\d+\s*seed', text_lower))
        features.append(seed_differential)  # 14. Seed mentions (narrative awareness)
        
        giant_killer = 1.0 if 'giant killer' in text_lower or 'upset specialist' in text_lower else 0.0
        features.append(giant_killer)  # 15. Giant killer archetype
        
        surprise_factor = sum(1 for m in ['surprise', 'shocking', 'unexpected', 'stunned'] if m in text_lower)
        features.append(surprise_factor)  # 16. Surprise narrative
        
        belief_language = sum(1 for m in ['believe', 'dream', 'destiny', 'magic', 'miracle'] if m in text_lower)
        features.append(belief_language / word_count)  # 17. Belief narrative
        
        momentum_narrative = sum(1 for m in ['momentum', 'riding high', 'unstoppable', 'hot'] if m in text_lower)
        features.append(momentum_narrative)  # 18. Momentum in tournament
        
        cinderella_weight = 1.0 + (cinderella_count * 0.8)
        features.append(min(3.0, cinderella_weight))  # 19. Cinderella narrative weight
        
        upset_composite = (features[11] + features[12] + features[17]) / 3
        features.append(upset_composite)  # 20. Upset narrative composite
        
        # === TOURNAMENT CONTEXT (10 features) ===
        round_markers = {
            'first round': 1.0, 'round of 64': 1.0, 'round of 32': 1.2,
            'sweet sixteen': 1.5, 'elite eight': 2.0, 'final four': 3.0, 'championship': 4.0
        }
        tournament_round = 1.0
        for marker, weight in round_markers.items():
            if marker in text_lower:
                tournament_round = max(tournament_round, weight)
        features.append(tournament_round)  # 21. Tournament round weight
        
        elimination_explicit = 1.0 if 'elimination' in text_lower or 'season on the line' in text_lower else 0.0
        features.append(elimination_explicit)  # 22. Elimination game
        
        march_madness = 1.0 if 'march madness' in text_lower or 'big dance' in text_lower else 0.0
        features.append(march_madness)  # 23. March Madness context
        
        selection_sunday = 1.0 if 'selection' in text_lower or 'bubble' in text_lower else 0.0
        features.append(selection_sunday)  # 24. Selection implications
        
        tournament_stakes = tournament_round * (1.0 + elimination_explicit + march_madness)
        features.append(min(5.0, tournament_stakes))  # 25. Total tournament stakes
        
        bracket_language = sum(1 for m in ['bracket', 'upset pick', 'bracket buster'] if m in text_lower)
        features.append(bracket_language)  # 26. Bracket narrative
        
        single_elimination = 1.0 if 'single elimination' in text_lower or 'one and done' in text_lower else 0.5
        features.append(single_elimination)  # 27. Single elimination awareness
        
        win_or_go_home = 1.0 if 'win or go home' in text_lower or 'season ends' in text_lower else 0.0
        features.append(win_or_go_home)  # 28. Ultimate stakes language
        
        tournament_veteran = sum(1 for m in ['tournament veteran', 'been here before', 'experience'] if m in text_lower)
        features.append(tournament_veteran)  # 29. Tournament experience
        
        overall_tournament_narrative = (tournament_stakes + bracket_language + features[27]) / 3
        features.append(overall_tournament_narrative)  # 30. Tournament narrative composite
        
        # === MOMENTUM / FORM (10 features) ===
        momentum_count = sum(1 for m in self.momentum_markers if m in text_lower)
        features.append(momentum_count / word_count)  # 31. Momentum density
        
        hot_language = sum(1 for m in ['hot', 'fire', 'unstoppable', 'rolling'] if m in text_lower)
        features.append(hot_language)  # 32. Hot streak narrative
        
        cold_language = sum(1 for m in ['cold', 'struggling', 'slump', 'off'] if m in text_lower)
        features.append(cold_language)  # 33. Cold streak
        
        momentum_direction = hot_language / max(1, hot_language + cold_language)
        features.append(momentum_direction)  # 34. Momentum direction
        
        recent_wins = sum(1 for m in ['won last', 'winning streak', '3 straight', '5 in a row'] if m in text_lower)
        features.append(recent_wins)  # 35. Recent win mentions
        
        form_language = sum(1 for m in ['form', 'playing well', 'peak', 'best basketball'] if m in text_lower)
        features.append(form_language)  # 36. Form narrative
        
        confidence_markers = sum(1 for m in ['confident', 'belief', 'swagger', 'momentum'] if m in text_lower)
        features.append(confidence_markers)  # 37. Confidence narrative
        
        rhythm_markers = sum(1 for m in ['rhythm', 'flow', 'clicking', 'sync'] if m in text_lower)
        features.append(rhythm_markers)  # 38. Team rhythm
        
        momentum_weight = 1.0 + (momentum_direction * 0.5)
        features.append(momentum_weight)  # 39. Momentum narrative weight
        
        momentum_composite = (features[31] + features[34] + features[36]) / 3
        features.append(momentum_composite)  # 40. Momentum composite
        
        # === COACHING / LEADERSHIP (10 features) ===
        coach_count = sum(1 for m in self.coach_markers if m in text_lower)
        features.append(coach_count / word_count)  # 41. Coach mentions
        
        hall_of_fame_coach = 1.0 if 'hall of fame coach' in text_lower or 'legendary coach' in text_lower else 0.0
        features.append(hall_of_fame_coach)  # 42. HOF coach
        
        tournament_coaching = sum(1 for m in ['tournament experience', 'final four', 'championship coach'] if m in text_lower)
        features.append(tournament_coaching)  # 43. Tournament coaching pedigree
        
        coaching_tree = sum(1 for m in ['coaching tree', 'disciple', 'protégé'] if m in text_lower)
        features.append(coaching_tree)  # 44. Coaching lineage
        
        experience_count = sum(1 for m in self.experience_markers if m in text_lower)
        features.append(experience_count / word_count)  # 45. Experience density
        
        youth_count = sum(1 for m in self.youth_markers if m in text_lower)
        features.append(youth_count / word_count)  # 46. Youth density
        
        experience_ratio = experience_count / max(1, experience_count + youth_count)
        features.append(experience_ratio)  # 47. Experience vs youth
        
        senior_leadership = sum(1 for m in ['senior', 'fifth year', 'graduate student'] if m in text_lower)
        features.append(senior_leadership)  # 48. Senior leadership
        
        coaching_weight = 1.0 + (tournament_coaching * 0.3) + (hall_of_fame_coach * 0.5)
        features.append(min(2.0, coaching_weight))  # 49. Coaching narrative weight
        
        leadership_composite = (features[41] + features[45] + features[48]) / 3
        features.append(leadership_composite)  # 50. Leadership composite
        
        # === RIVALRY / MATCHUP (10 features) ===
        rivalry_count = sum(1 for m in self.rivalry_markers if m in text_lower)
        features.append(rivalry_count / word_count)  # 51. Rivalry density
        
        conference_game = 1.0 if 'conference' in text_lower else 0.0
        features.append(conference_game)  # 52. Conference rivalry
        
        historical_matchup = sum(1 for m in ['history', 'always', 'every year', 'tradition'] if m in text_lower)
        features.append(historical_matchup)  # 53. Historical rivalry
        
        intensity_markers = sum(1 for m in ['intense', 'heated', 'bitter', 'hate', 'bad blood'] if m in text_lower)
        features.append(intensity_markers)  # 54. Intensity narrative
        
        rivalry_weight = 1.0 + (rivalry_count * 1.0) + (conference_game * 0.5)
        features.append(min(2.5, rivalry_weight))  # 55. Rivalry narrative weight
        
        matchup_importance = sum(1 for m in ['important', 'crucial', 'key', 'pivotal'] if m in text_lower)
        features.append(matchup_importance)  # 56. Matchup importance
        
        revenge_narrative = 1.0 if 'revenge' in text_lower or 'rematch' in text_lower else 0.0
        features.append(revenge_narrative)  # 57. Revenge game
        
        home_court_count = sum(1 for m in self.home_court_markers if m in text_lower)
        features.append(home_court_count / word_count)  # 58. Home court narrative
        
        rivalry_composite = (features[51] + features[54] + features[56]) / 3
        features.append(rivalry_composite)  # 59. Rivalry composite
        
        # === OVERALL BASKETBALL NARRATIVE WEIGHT (1 feature) ===
        # Composite of all basketball-specific factors
        basketball_narrative_weight = (
            features[6] +   # Legacy weight
            features[19] +  # Cinderella weight
            features[24] +  # Tournament stakes
            features[39] +  # Momentum weight
            features[49] +  # Coaching weight
            features[54]    # Rivalry weight
        ) / 6
        features.append(basketball_narrative_weight)  # 60. Overall basketball narrative weight
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return all 60 feature names."""
        return [
            # Tradition (10)
            'tradition_density', 'tradition_strength', 'historical_depth', 'championship_legacy',
            'blue_blood', 'powerhouse_archetype', 'legacy_weight', 'pedigree_markers',
            'sustained_excellence', 'tradition_composite',
            # Upset/Cinderella (10)
            'cinderella_density', 'upset_narrative', 'underdog_explicit', 'seed_mentions',
            'giant_killer', 'surprise_narrative', 'belief_language', 'momentum_narrative',
            'cinderella_weight', 'upset_composite',
            # Tournament (10)
            'tournament_round', 'elimination_game', 'march_madness', 'selection_implications',
            'tournament_stakes', 'bracket_narrative', 'single_elimination', 'win_or_go_home',
            'tournament_veteran', 'tournament_composite',
            # Momentum (10)
            'momentum_density', 'hot_streak', 'cold_streak', 'momentum_direction',
            'recent_wins', 'form_language', 'confidence_narrative', 'rhythm_markers',
            'momentum_weight', 'momentum_composite',
            # Coaching (10)
            'coach_mentions', 'hof_coach', 'tournament_coaching', 'coaching_tree',
            'experience_density', 'youth_density', 'experience_ratio', 'senior_leadership',
            'coaching_weight', 'leadership_composite',
            # Rivalry/Matchup (10)
            'rivalry_density', 'conference_game', 'historical_rivalry', 'intensity_narrative',
            'rivalry_weight', 'matchup_importance', 'revenge_game', 'home_court_narrative',
            'rivalry_composite', 'overall_basketball_weight'
        ]

