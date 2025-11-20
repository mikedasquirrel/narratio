"""
Character Archetype Transformer

Detects Jung's 12 primary archetypes + Propp's 7 spheres of action.
Measures archetype clarity, character complexity, and archetypal pairing.

Based on:
- Carl Jung's "Archetypes and the Collective Unconscious" (1959)
- Vladimir Propp's "Morphology of the Folktale" (1928)
- Christopher Vogler's 8 character archetypes

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

from ..base import NarrativeTransformer


class CharacterArchetypeTransformer(NarrativeTransformer):
    """
    Detect and measure character archetypes in narratives.
    
    Extracts ~55 features:
    - 12 Jung archetype detection scores
    - 8 Vogler archetype detection scores
    - 7 Propp sphere detection scores
    - Archetype clarity score
    - Character complexity metrics
    - Shadow projection strength
    - Archetypal pairing quality
    - Character arc completeness
    
    Supports hybrid approach:
    - Theoretical weights from Jung/Propp
    - Empirical weights learned from data
    - Validation of classical theory
    """
    
    def __init__(self, use_learned_weights=False, learned_weights=None):
        """
        Initialize Character Archetype Transformer.
        
        Args:
            use_learned_weights: Use empirically learned weights
            learned_weights: Dict mapping archetype names to importance weights
        """
        super().__init__(
            narrative_id="character_archetype",
            description="Jung's 12 archetypes + Propp's 7 spheres + Vogler's 8 roles"
        )
        
        self.use_learned_weights = use_learned_weights
        self.learned_weights = learned_weights or {}
        
        # Jung's 12 Primary Archetypes
        self.jung_archetypes = {
            # EGO TYPES (seeking paradise)
            'innocent': {
                'markers': [
                    'innocent', 'pure', 'naive', 'optimistic', 'trusting', 'faith',
                    'hopeful', 'simple', 'childlike', 'untainted', 'believing'
                ],
                'traits': ['optimism', 'trust', 'simplicity', 'hope'],
                'fears': ['abandonment', 'punishment', 'corruption'],
                'desires': ['safety', 'happiness', 'belonging'],
                'weight': 0.6
            },
            'orphan': {
                'markers': [
                    'orphan', 'everyman', 'ordinary', 'common', 'relatable', 'realistic',
                    'down to earth', 'pragmatic', 'survivor', 'left out', 'abandoned'
                ],
                'traits': ['empathy', 'realism', 'connection', 'belonging'],
                'fears': ['being left out', 'standing out'],
                'desires': ['belonging', 'acceptance', 'community'],
                'weight': 0.7
            },
            'warrior': {
                'markers': [
                    'hero', 'warrior', 'brave', 'courageous', 'fight', 'battle',
                    'protect', 'defend', 'sacrifice', 'duty', 'honor', 'champion'
                ],
                'traits': ['courage', 'discipline', 'competence', 'determination'],
                'fears': ['weakness', 'vulnerability', 'cowardice'],
                'desires': ['prove worth', 'mastery', 'victory'],
                'weight': 1.0
            },
            'caregiver': {
                'markers': [
                    'caregiver', 'nurture', 'protect', 'selfless', 'compassion', 'care',
                    'mother', 'guardian', 'help', 'support', 'giving', 'generous'
                ],
                'traits': ['compassion', 'generosity', 'selflessness', 'nurturing'],
                'fears': ['selfishness', 'ingratitude', 'unable to help'],
                'desires': ['help others', 'protect', 'care for'],
                'weight': 0.8
            },
            
            # SOUL TYPES (seeking connection)
            'explorer': {
                'markers': [
                    'explorer', 'freedom', 'discover', 'journey', 'wander', 'independent',
                    'autonomous', 'adventure', 'search', 'experience', 'authentic'
                ],
                'traits': ['independence', 'curiosity', 'ambition', 'authenticity'],
                'fears': ['conformity', 'emptiness', 'being trapped'],
                'desires': ['freedom', 'discovery', 'authentic life'],
                'weight': 0.7
            },
            'rebel': {
                'markers': [
                    'rebel', 'outlaw', 'revolution', 'break rules', 'disrupt', 'overthrow',
                    'defy', 'challenge', 'liberation', 'freedom fighter', 'radical'
                ],
                'traits': ['independence', 'disruption', 'revolution', 'liberation'],
                'fears': ['powerlessness', 'ineffectuality', 'compliance'],
                'desires': ['revolution', 'change', 'overturn status quo'],
                'weight': 0.8
            },
            'lover': {
                'markers': [
                    'lover', 'love', 'passion', 'romance', 'intimate', 'devoted',
                    'desire', 'pleasure', 'beauty', 'sensual', 'connected', 'heart'
                ],
                'traits': ['passion', 'appreciation', 'commitment', 'devotion'],
                'fears': ['being alone', 'unloved', 'unwanted'],
                'desires': ['intimacy', 'connection', 'experience pleasure'],
                'weight': 0.7
            },
            'creator': {
                'markers': [
                    'creator', 'artist', 'create', 'innovate', 'imagine', 'invent',
                    'build', 'design', 'vision', 'original', 'express', 'craft'
                ],
                'traits': ['creativity', 'imagination', 'innovation', 'vision'],
                'fears': ['mediocrity', 'stagnation', 'imitation'],
                'desires': ['create', 'enduring value', 'self-expression'],
                'weight': 0.7
            },
            
            # SELF TYPES (seeking order)
            'jester': {
                'markers': [
                    'jester', 'fool', 'trickster', 'humor', 'playful', 'fun', 'joke',
                    'comic', 'lighthearted', 'mischief', 'enjoy', 'laugh', 'amusing'
                ],
                'traits': ['humor', 'playfulness', 'joy', 'perspective'],
                'fears': ['boredom', 'being boring', 'meaninglessness'],
                'desires': ['joy', 'live in moment', 'enjoy life'],
                'weight': 0.6
            },
            'sage': {
                'markers': [
                    'sage', 'wise', 'wisdom', 'knowledge', 'truth', 'understand',
                    'analyze', 'philosopher', 'scholar', 'teacher', 'enlightened'
                ],
                'traits': ['wisdom', 'knowledge', 'truth-seeking', 'analysis'],
                'fears': ['ignorance', 'deception', 'misunderstanding'],
                'desires': ['truth', 'understanding', 'knowledge'],
                'weight': 0.9
            },
            'magician': {
                'markers': [
                    'magician', 'wizard', 'sorcerer', 'magic', 'transform', 'power',
                    'mystical', 'vision', 'catalyst', 'change', 'manipulate reality'
                ],
                'traits': ['transformation', 'vision', 'power', 'charisma'],
                'fears': ['unintended consequences', 'corruption'],
                'desires': ['transform reality', 'make dreams real', 'power'],
                'weight': 0.8
            },
            'ruler': {
                'markers': [
                    'ruler', 'king', 'queen', 'leader', 'control', 'command', 'authority',
                    'power', 'order', 'responsibility', 'govern', 'organize', 'control'
                ],
                'traits': ['leadership', 'control', 'responsibility', 'authority'],
                'fears': ['chaos', 'overthrow', 'loss of control'],
                'desires': ['control', 'order', 'prosperity', 'power'],
                'weight': 0.9
            }
        }
        
        # Vogler's 8 Archetypes (simplified, functional roles)
        self.vogler_archetypes = {
            'hero': {
                'markers': ['hero', 'protagonist', 'champion', 'chosen one', 'warrior'],
                'weight': 1.0
            },
            'mentor': {
                'markers': ['mentor', 'teacher', 'guide', 'master', 'wise', 'elder', 'advisor'],
                'weight': 0.9
            },
            'threshold_guardian': {
                'markers': ['guard', 'test', 'block', 'challenge', 'obstacle', 'gatekeeper'],
                'weight': 0.6
            },
            'herald': {
                'markers': ['herald', 'messenger', 'announce', 'call', 'summons', 'news'],
                'weight': 0.5
            },
            'shapeshifter': {
                'markers': ['mysterious', 'ambiguous', 'unclear', 'unpredictable', 'enigmatic', 'changeable'],
                'weight': 0.7
            },
            'shadow': {
                'markers': ['villain', 'enemy', 'antagonist', 'dark', 'evil', 'oppose', 'threat'],
                'weight': 1.0
            },
            'ally': {
                'markers': ['ally', 'friend', 'companion', 'sidekick', 'partner', 'support'],
                'weight': 0.7
            },
            'trickster': {
                'markers': ['trickster', 'mischief', 'chaos', 'disrupt', 'playful', 'unpredictable'],
                'weight': 0.6
            }
        }
        
        # Propp's 7 Spheres of Action
        self.propp_spheres = {
            'villain': {
                'markers': ['villain', 'antagonist', 'evil', 'harm', 'oppose', 'struggle', 'enemy'],
                'actions': ['harm', 'fight', 'pursue', 'combat'],
                'weight': 1.0
            },
            'donor': {
                'markers': ['donor', 'give', 'provide', 'grant', 'bestow', 'gift', 'test'],
                'actions': ['test', 'give magical agent', 'provide', 'prepare'],
                'weight': 0.8
            },
            'helper': {
                'markers': ['helper', 'assist', 'aid', 'support', 'rescue', 'save', 'help'],
                'actions': ['aid', 'rescue', 'solve', 'transport', 'assist'],
                'weight': 0.7
            },
            'princess': {
                'markers': ['princess', 'sought', 'goal', 'prize', 'beloved', 'rescued', 'won'],
                'actions': ['sought after', 'sets task', 'is won', 'marries hero'],
                'weight': 0.6
            },
            'dispatcher': {
                'markers': ['dispatcher', 'send', 'dispatch', 'command', 'request', 'assign'],
                'actions': ['send hero', 'dispatch on quest', 'request help'],
                'weight': 0.5
            },
            'hero': {
                'markers': ['hero', 'protagonist', 'seeker', 'quest', 'journey', 'achieve'],
                'actions': ['depart', 'quest', 'fight', 'win', 'marry', 'ascend'],
                'weight': 1.0
            },
            'false_hero': {
                'markers': ['false hero', 'imposter', 'pretender', 'claim', 'deceive', 'lie'],
                'actions': ['claim credit', 'deceive', 'pretend', 'exposed'],
                'weight': 0.4
            }
        }
        
        # Shadow projection patterns
        self.shadow_patterns = {
            'projection': ['opposite', 'mirror', 'dark side', 'reflection', 'embodies'],
            'repressed_traits': ['deny', 'repress', 'hide', 'suppress', 'refuse to see'],
            'integration': ['accept', 'integrate', 'realize', 'understand', 'embrace']
        }
        
        # Character arc patterns
        self.arc_patterns = {
            'change_arc': ['changed', 'transformed', 'became', 'grew', 'evolved'],
            'flat_arc': ['remained', 'unchanged', 'same', 'constant', 'steady'],
            'corruption_arc': ['corrupted', 'fell', 'turned', 'betrayed', 'lost way']
        }
    
    def fit(self, X: List[str], y=None):
        """Fit transformer (no-op for rule-based)."""
        self.is_fitted_ = True
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        Extract character archetype features.
        
        Args:
            X: List of narrative texts
            
        Returns:
            Feature matrix of shape (n_samples, ~55)
        """
        features = []
        for text in X:
            features.append(self._extract_archetype_features(text))
        return np.array(features)
    
    def _extract_archetype_features(self, text: str) -> np.ndarray:
        """Extract complete character archetype feature vector."""
        text_lower = text.lower()
        features = {}
        
        # 1. Jung's 12 archetypes (12 features)
        jung_scores = []
        for archetype_name, archetype_info in self.jung_archetypes.items():
            score = self._detect_jung_archetype(text_lower, archetype_info)
            jung_scores.append(score)
            features[f'jung_{archetype_name}'] = score
        
        # 2. Vogler's 8 archetypes (8 features)
        vogler_scores = []
        for archetype_name, archetype_info in self.vogler_archetypes.items():
            score = self._detect_vogler_archetype(text_lower, archetype_info)
            vogler_scores.append(score)
            features[f'vogler_{archetype_name}'] = score
        
        # 3. Propp's 7 spheres (7 features)
        propp_scores = []
        for sphere_name, sphere_info in self.propp_spheres.items():
            score = self._detect_propp_sphere(text_lower, sphere_info)
            propp_scores.append(score)
            features[f'propp_{sphere_name}'] = score
        
        # 4. Archetype clarity scores (3 features)
        features['jung_archetype_clarity'] = self._calculate_archetype_clarity(jung_scores)
        features['vogler_archetype_clarity'] = self._calculate_archetype_clarity(vogler_scores)
        features['propp_archetype_clarity'] = self._calculate_archetype_clarity(propp_scores)
        
        # 5. Character complexity (4 features)
        complexity_features = self._analyze_character_complexity(jung_scores, vogler_scores)
        features.update(complexity_features)
        
        # 6. Shadow projection (3 features)
        shadow_features = self._analyze_shadow_projection(text_lower, jung_scores, vogler_scores)
        features.update(shadow_features)
        
        # 7. Archetypal pairing (4 features)
        pairing_features = self._analyze_archetypal_pairing(jung_scores, vogler_scores, propp_scores)
        features.update(pairing_features)
        
        # 8. Character arc (3 features)
        arc_features = self._analyze_character_arc(text_lower)
        features.update(arc_features)
        
        # 9. Dominant archetypes (4 features)
        features['dominant_jung'] = np.argmax(jung_scores)
        features['dominant_vogler'] = np.argmax(vogler_scores)
        features['dominant_propp'] = np.argmax(propp_scores)
        features['archetype_strength'] = max(jung_scores + vogler_scores + propp_scores)
        
        # 10. Cross-system agreement (3 features)
        features['hero_consistency'] = self._calculate_hero_consistency(jung_scores, vogler_scores, propp_scores)
        features['mentor_consistency'] = self._calculate_mentor_consistency(jung_scores, vogler_scores, propp_scores)
        features['villain_consistency'] = self._calculate_villain_consistency(vogler_scores, propp_scores)
        
        return np.array(list(features.values()))
    
    def _detect_jung_archetype(self, text: str, archetype_info: Dict) -> float:
        """Detect Jung archetype presence and strength."""
        # Count markers
        marker_count = sum([text.count(marker) for marker in archetype_info['markers']])
        marker_score = min(1.0, marker_count / 5)
        
        # Check traits
        trait_count = sum([1 for trait in archetype_info['traits'] if trait in text])
        trait_score = trait_count / len(archetype_info['traits'])
        
        # Check fears (negative presence)
        fear_count = sum([1 for fear in archetype_info['fears'] if fear in text])
        fear_score = min(1.0, fear_count / len(archetype_info['fears']))
        
        # Check desires
        desire_count = sum([1 for desire in archetype_info['desires'] if desire in text])
        desire_score = min(1.0, desire_count / len(archetype_info['desires']))
        
        # Combined score
        score = (
            0.40 * marker_score +
            0.25 * trait_score +
            0.20 * fear_score +
            0.15 * desire_score
        )
        
        return score
    
    def _detect_vogler_archetype(self, text: str, archetype_info: Dict) -> float:
        """Detect Vogler archetype (simpler, functional roles)."""
        marker_count = sum([text.count(marker) for marker in archetype_info['markers']])
        return min(1.0, marker_count / 3)
    
    def _detect_propp_sphere(self, text: str, sphere_info: Dict) -> float:
        """Detect Propp sphere of action."""
        # Markers
        marker_count = sum([text.count(marker) for marker in sphere_info['markers']])
        marker_score = min(1.0, marker_count / 3)
        
        # Actions
        action_count = sum([1 for action in sphere_info['actions'] if action in text])
        action_score = action_count / len(sphere_info['actions'])
        
        return 0.60 * marker_score + 0.40 * action_score
    
    def _calculate_archetype_clarity(self, scores: List[float]) -> float:
        """
        Measure archetype clarity (clear vs ambiguous characters).
        
        High clarity = one dominant archetype
        Low clarity = multiple weak archetypes or equally strong
        """
        if max(scores) < 0.3:
            return 0.0  # No clear archetype
        
        # Entropy-based clarity
        scores_array = np.array(scores) + 1e-10
        probs = scores_array / scores_array.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(scores))
        
        # Invert: high entropy = low clarity
        clarity = 1 - (entropy / max_entropy)
        
        # Boost if there's a very dominant archetype
        if max(scores) > 0.7:
            clarity = min(1.0, clarity * 1.2)
        
        return clarity
    
    def _analyze_character_complexity(self, jung_scores: List[float], 
                                     vogler_scores: List[float]) -> Dict[str, float]:
        """Analyze character complexity and depth."""
        features = {}
        
        # Number of active archetypes (score > 0.3)
        jung_active = sum([1 for s in jung_scores if s > 0.3])
        vogler_active = sum([1 for s in vogler_scores if s > 0.3])
        
        features['jung_archetype_count'] = min(1.0, jung_active / 4)  # Normalize
        features['vogler_archetype_count'] = min(1.0, vogler_active / 3)
        
        # Character complexity (multiple archetypes = complex)
        features['character_complexity'] = (
            0.60 * features['jung_archetype_count'] +
            0.40 * features['vogler_archetype_count']
        )
        
        # Archetype mixing (contradictory archetypes present)
        # E.g., Warrior + Caregiver = complex, Warrior + Innocent = less common
        mixing_score = 0.0
        if jung_scores[2] > 0.4 and jung_scores[3] > 0.4:  # Warrior + Caregiver
            mixing_score += 0.3
        if jung_scores[5] > 0.4 and jung_scores[11] > 0.4:  # Rebel + Ruler
            mixing_score += 0.4
        
        features['archetype_mixing'] = min(1.0, mixing_score)
        
        return features
    
    def _analyze_shadow_projection(self, text: str, jung_scores: List[float],
                                   vogler_scores: List[float]) -> Dict[str, float]:
        """Analyze hero-shadow projection strength."""
        features = {}
        
        # Check for projection language
        projection_count = sum([text.count(p) for p in self.shadow_patterns['projection']])
        features['shadow_projection_language'] = min(1.0, projection_count / 2)
        
        # Hero-Shadow pairing strength (Vogler)
        hero_score = vogler_scores[0]  # Hero
        shadow_score = vogler_scores[5]  # Shadow
        features['hero_shadow_pairing'] = min(hero_score, shadow_score)
        
        # Repressed traits integration
        repressed_count = sum([text.count(r) for r in self.shadow_patterns['repressed_traits']])
        integration_count = sum([text.count(i) for i in self.shadow_patterns['integration']])
        
        features['shadow_integration'] = min(1.0, integration_count / (repressed_count + 1))
        
        return features
    
    def _analyze_archetypal_pairing(self, jung_scores: List[float],
                                   vogler_scores: List[float],
                                   propp_scores: List[float]) -> Dict[str, float]:
        """Analyze archetypal pairing quality."""
        features = {}
        
        # Hero-Mentor pairing
        jung_warrior = jung_scores[2]
        jung_sage = jung_scores[9]
        vogler_hero = vogler_scores[0]
        vogler_mentor = vogler_scores[1]
        
        features['hero_mentor_pairing'] = (
            0.50 * min(jung_warrior, jung_sage) +
            0.50 * min(vogler_hero, vogler_mentor)
        )
        
        # Hero-Shadow pairing (already computed, but consistent)
        features['hero_shadow_present'] = float(vogler_scores[0] > 0.5 and vogler_scores[5] > 0.5)
        
        # Complete cast (multiple archetypes present)
        jung_diversity = len([s for s in jung_scores if s > 0.3])
        vogler_diversity = len([s for s in vogler_scores if s > 0.4])
        
        features['archetype_diversity'] = (jung_diversity / 12 + vogler_diversity / 8) / 2
        
        # Balanced ensemble (archetypes evenly distributed)
        jung_variance = np.var(jung_scores)
        vogler_variance = np.var(vogler_scores)
        features['ensemble_balance'] = 1 - (jung_variance + vogler_variance) / 2
        
        return features
    
    def _analyze_character_arc(self, text: str) -> Dict[str, float]:
        """Analyze character transformation arc."""
        features = {}
        
        # Change arc
        change_count = sum([text.count(marker) for marker in self.arc_patterns['change_arc']])
        features['change_arc'] = min(1.0, change_count / 3)
        
        # Flat arc (character changes world, not self)
        flat_count = sum([text.count(marker) for marker in self.arc_patterns['flat_arc']])
        features['flat_arc'] = min(1.0, flat_count / 2)
        
        # Corruption arc
        corruption_count = sum([text.count(marker) for marker in self.arc_patterns['corruption_arc']])
        features['corruption_arc'] = min(1.0, corruption_count / 2)
        
        return features
    
    def _calculate_hero_consistency(self, jung: List[float], vogler: List[float], propp: List[float]) -> float:
        """Check if 'hero' archetype is consistent across systems."""
        jung_warrior = jung[2]  # Warrior/Hero
        vogler_hero = vogler[0]  # Hero
        propp_hero = propp[5]  # Hero
        
        # High consistency if all three agree
        return (jung_warrior + vogler_hero + propp_hero) / 3
    
    def _calculate_mentor_consistency(self, jung: List[float], vogler: List[float], propp: List[float]) -> float:
        """Check mentor consistency."""
        jung_sage = jung[9]  # Sage
        vogler_mentor = vogler[1]  # Mentor
        propp_donor = propp[1]  # Donor (similar to mentor)
        
        return (jung_sage + vogler_mentor + propp_donor) / 3
    
    def _calculate_villain_consistency(self, vogler: List[float], propp: List[float]) -> float:
        """Check villain consistency."""
        vogler_shadow = vogler[5]  # Shadow
        propp_villain = propp[0]  # Villain
        
        return (vogler_shadow + propp_villain) / 2
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Jung
        for name in self.jung_archetypes.keys():
            names.append(f'jung_{name}')
        
        # Vogler
        for name in self.vogler_archetypes.keys():
            names.append(f'vogler_{name}')
        
        # Propp
        for name in self.propp_spheres.keys():
            names.append(f'propp_{name}')
        
        # Aggregate features
        names.extend([
            'jung_archetype_clarity',
            'vogler_archetype_clarity',
            'propp_archetype_clarity',
            'jung_archetype_count',
            'vogler_archetype_count',
            'character_complexity',
            'archetype_mixing',
            'shadow_projection_language',
            'hero_shadow_pairing',
            'shadow_integration',
            'hero_mentor_pairing',
            'hero_shadow_present',
            'archetype_diversity',
            'ensemble_balance',
            'change_arc',
            'flat_arc',
            'corruption_arc',
            'dominant_jung',
            'dominant_vogler',
            'dominant_propp',
            'archetype_strength',
            'hero_consistency',
            'mentor_consistency',
            'villain_consistency'
        ])
        
        return names
    
    def learn_weights_from_data(self, X: List[str], y: np.ndarray,
                                method='correlation') -> Dict[str, float]:
        """
        Learn empirical archetype importance weights.
        
        Discovers which archetypes actually predict success in this domain.
        """
        from scipy.stats import pearsonr
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        from sklearn.linear_model import Ridge
        
        features = self.transform(X)
        feature_names = self.get_feature_names()
        
        # Focus on archetype features (not derived features)
        archetype_indices = {
            name: i for i, name in enumerate(feature_names)
            if (name.startswith('jung_') or name.startswith('vogler_') or name.startswith('propp_'))
            and not any(x in name for x in ['clarity', 'count', 'consistency'])
        }
        
        learned_weights = {}
        
        if method == 'correlation':
            for name, idx in archetype_indices.items():
                corr, _ = pearsonr(features[:, idx], y)
                learned_weights[name] = abs(corr)
        
        elif method == 'mutual_info':
            archetype_features = features[:, list(archetype_indices.values())]
            if len(np.unique(y)) <= 10:
                mi_scores = mutual_info_classif(archetype_features, y)
            else:
                mi_scores = mutual_info_regression(archetype_features, y)
            
            mi_scores = mi_scores / (mi_scores.max() + 1e-10)
            for i, name in enumerate(archetype_indices.keys()):
                learned_weights[name] = mi_scores[i]
        
        elif method == 'regression':
            archetype_features = features[:, list(archetype_indices.values())]
            model = Ridge(alpha=1.0)
            model.fit(archetype_features, y)
            coeffs = np.abs(model.coef_)
            coeffs = coeffs / (coeffs.max() + 1e-10)
            
            for i, name in enumerate(archetype_indices.keys()):
                learned_weights[name] = coeffs[i]
        
        self.learned_weights = learned_weights
        self.use_learned_weights = True
        
        return learned_weights
    
    def compare_theoretical_vs_empirical(self) -> Dict:
        """Compare Jung/Propp theoretical weights to empirical weights."""
        if not self.learned_weights:
            raise ValueError("Must call learn_weights_from_data first")
        
        comparison = {'archetypes': {}, 'summary': {}}
        
        # Compare Jung archetypes
        for name, info in self.jung_archetypes.items():
            full_name = f'jung_{name}'
            theoretical = info['weight']
            empirical = self.learned_weights.get(full_name, 0.5)
            deviation = theoretical - empirical
            
            comparison['archetypes'][full_name] = {
                'theoretical_weight': theoretical,
                'empirical_weight': empirical,
                'deviation': deviation,
                'jung_overvalued': deviation > 0.2,
                'jung_undervalued': deviation < -0.2
            }
        
        # Summary
        deviations = [a['deviation'] for a in comparison['archetypes'].values()]
        comparison['summary'] = {
            'mean_absolute_deviation': np.mean(np.abs(deviations)),
            'theory_validated': np.mean(np.abs(deviations)) < 0.15,
            'most_overvalued': max(comparison['archetypes'].items(), key=lambda x: x[1]['deviation'])[0],
            'most_undervalued': min(comparison['archetypes'].items(), key=lambda x: x[1]['deviation'])[0]
        }
        
        return comparison


# Convenience functions
def discover_archetype_patterns(texts: List[str], outcomes: np.ndarray,
                               method='correlation') -> Dict:
    """
    Discover which character archetypes actually predict success.
    
    Returns learned weights and validation of Jung/Propp theory.
    """
    transformer = CharacterArchetypeTransformer()
    transformer.fit(texts)
    
    learned_weights = transformer.learn_weights_from_data(texts, outcomes, method)
    comparison = transformer.compare_theoretical_vs_empirical()
    
    return {
        'learned_weights': learned_weights,
        'theoretical_validation': comparison,
        'transformer': transformer
    }

