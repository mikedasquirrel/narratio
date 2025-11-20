"""
Plot Archetype Transformer

Detects Booker's 7 basic plots + Polti's 36 dramatic situations.
Measures plot type, situation complexity, and narrative structure.

Based on:
- Christopher Booker's "The Seven Basic Plots" (2004)
- Georges Polti's "The Thirty-Six Dramatic Situations" (1895)

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

from ..base import NarrativeTransformer


class PlotArchetypeTransformer(NarrativeTransformer):
    """
    Detect and measure plot archetypes in narratives.
    
    Extracts ~50 features:
    - 7 Booker plot type scores
    - 36 Polti situation scores (condensed to 10 categories)
    - Plot purity vs blending
    - Conflict complexity
    - Resolution quality
    
    Supports hybrid approach for empirical discovery.
    """
    
    def __init__(self, use_learned_weights=False, learned_weights=None):
        super().__init__(
            narrative_id="plot_archetype",
            description="Booker's 7 plots + Polti's 36 situations"
        )
        
        self.use_learned_weights = use_learned_weights
        self.learned_weights = learned_weights or {}
        
        # Booker's 7 Basic Plots
        self.booker_plots = {
            'overcoming_monster': {
                'markers': [
                    'monster', 'threat', 'danger', 'evil', 'menace', 'battle',
                    'defeat', 'triumph', 'victory', 'saved', 'destroyed evil'
                ],
                'stages': [
                    'anticipation', 'dream stage', 'frustration', 
                    'nightmare', 'miraculous escape'
                ],
                'structure': 'threat → prepare → struggle → near defeat → victory',
                'weight': 1.0
            },
            'rags_to_riches': {
                'markers': [
                    'poor', 'poverty', 'humble', 'lowly', 'oppressed', 'nothing',
                    'opportunity', 'success', 'wealth', 'prosperity', 'triumph',
                    'proved worth', 'earned', 'deserved'
                ],
                'stages': [
                    'initial wretchedness', 'out into world', 'initial success',
                    'central crisis', 'independence'
                ],
                'structure': 'low → opportunity → success → threat → proved worth',
                'weight': 0.9
            },
            'quest': {
                'markers': [
                    'quest', 'journey', 'search', 'seek', 'find', 'goal',
                    'companions', 'traveled', 'reached', 'achieved', 'returned'
                ],
                'stages': [
                    'the call', 'the journey', 'arrival and frustration',
                    'final ordeal', 'the goal'
                ],
                'structure': 'call → journey → arrive → ordeal → goal achieved',
                'weight': 1.0
            },
            'voyage_and_return': {
                'markers': [
                    'strange world', 'different', 'foreign', 'unknown', 'magical',
                    'wonderful', 'fascination', 'threat emerged', 'trapped',
                    'escape', 'return home', 'back', 'safe'
                ],
                'stages': [
                    'anticipation and fall', 'initial fascination', 'frustration',
                    'nightmare', 'thrilling escape'
                ],
                'structure': 'enter strange → fascination → threat → nightmare → escape',
                'weight': 0.8
            },
            'comedy': {
                'markers': [
                    'confusion', 'misunderstanding', 'mistaken', 'mix-up', 'chaos',
                    'revealed', 'truth', 'realize', 'clarity', 'marriage',
                    'united', 'celebration', 'harmony', 'resolved'
                ],
                'stages': [
                    'establishment', 'complication', 'confusion',
                    'resolution', 'union'
                ],
                'structure': 'normal → complications → chaos → revelation → union',
                'weight': 0.8
            },
            'tragedy': {
                'markers': [
                    'ambitious', 'hubris', 'pride', 'flaw', 'fatal', 'downfall',
                    'fell', 'destroyed', 'ruined', 'death', 'catastrophe',
                    'inevitable', 'doomed', 'too late'
                ],
                'stages': [
                    'anticipation', 'dream stage', 'frustration',
                    'nightmare', 'destruction'
                ],
                'structure': 'desire → success → opposition → collapse → death',
                'weight': 1.0
            },
            'rebirth': {
                'markers': [
                    'cursed', 'enchanted', 'spell', 'trapped', 'dark', 'shadow',
                    'imprisoned', 'hopeless', 'redeemed', 'transformed',
                    'reborn', 'awakened', 'freed', 'saved'
                ],
                'stages': [
                    'fall under shadow', 'recession', 'imprisonment',
                    'nightmare', 'redemption'
                ],
                'structure': 'curse → trapped → darkness → despair → redemption',
                'weight': 0.9
            }
        }
        
        # Polti's 36 situations (grouped into 10 categories for efficiency)
        self.polti_categories = {
            'power_dynamics': {
                'situations': ['supplication', 'deliverance', 'vengeance', 'disaster', 'revolt', 'daring_enterprise'],
                'markers': [
                    'power', 'authority', 'beg', 'plead', 'mercy', 'rescue', 'save',
                    'revenge', 'vengeance', 'disaster', 'defeat', 'revolt', 'overthrow',
                    'dare', 'bold', 'challenge'
                ],
                'weight': 0.9
            },
            'kinship_conflict': {
                'situations': ['vengeance_kindred', 'enmity_kinsmen', 'rivalry_kinsmen', 
                              'slaying_unrecognized', 'sacrifice_kindred', 'dishonor_loved_one'],
                'markers': [
                    'family', 'brother', 'sister', 'father', 'mother', 'son', 'daughter',
                    'betray', 'conflict', 'rivalry', 'compete', 'sacrifice', 'dishonor'
                ],
                'weight': 0.8
            },
            'love_and_passion': {
                'situations': ['murderous_adultery', 'crimes_of_love', 'adultery',
                              'obstacles_to_love', 'enemy_loved', 'mistaken_jealousy'],
                'markers': [
                    'love', 'passion', 'adultery', 'affair', 'forbidden', 'crime_of_passion',
                    'obstacle', 'prevent', 'jealous', 'jealousy', 'rival', 'competing'
                ],
                'weight': 0.7
            },
            'pursuit_escape': {
                'situations': ['pursuit', 'abduction', 'loss_of_loved'],
                'markers': [
                    'pursue', 'chase', 'hunt', 'flee', 'escape', 'abduct', 'kidnap',
                    'captured', 'taken', 'lost', 'missing', 'search_for'
                ],
                'weight': 0.7
            },
            'knowledge_mystery': {
                'situations': ['enigma', 'obtaining', 'fatal_imprudence', 'erroneous_judgment'],
                'markers': [
                    'mystery', 'enigma', 'puzzle', 'solve', 'discover', 'secret',
                    'mistake', 'error', 'judgment', 'wrong', 'misunderstood'
                ],
                'weight': 0.6
            },
            'moral_ideal': {
                'situations': ['self_sacrifice_ideal', 'rivalry_superior_inferior', 
                              'ambition', 'remorse'],
                'markers': [
                    'sacrifice', 'ideal', 'principle', 'belief', 'ambition', 'power',
                    'remorse', 'guilt', 'regret', 'moral', 'conscience'
                ],
                'weight': 0.8
            },
            'madness': {
                'situations': ['madness'],
                'markers': [
                    'mad', 'insane', 'madness', 'crazy', 'unstable', 'mental',
                    'breakdown', 'delusion', 'paranoia'
                ],
                'weight': 0.6
            },
            'divine_fate': {
                'situations': ['conflict_with_god'],
                'markers': [
                    'god', 'gods', 'divine', 'fate', 'destiny', 'prophecy',
                    'curse', 'blessed', 'damn', 'supernatural'
                ],
                'weight': 0.7
            },
            'sacrifice': {
                'situations': ['necessity_sacrificing_loved', 'all_sacrificed_passion'],
                'markers': [
                    'sacrifice', 'must give up', 'no choice', 'forced to',
                    'lose', 'give up everything', 'pay price'
                ],
                'weight': 0.8
            },
            'recovery': {
                'situations': ['recovery_of_lost'],
                'markers': [
                    'recover', 'found', 'reunion', 'return', 'restore',
                    'reclaim', 'get back'
                ],
                'weight': 0.6
            }
        }
    
    def fit(self, X: List[str], y=None):
        """Fit transformer."""
        self.is_fitted_ = True
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """Extract plot archetype features."""
        features = []
        for text in X:
            features.append(self._extract_plot_features(text))
        return np.array(features)
    
    def _extract_plot_features(self, text: str) -> np.ndarray:
        """Extract complete plot archetype vector."""
        text_lower = text.lower()
        segments = self._split_into_segments(text, 20)
        features = {}
        
        # 1. Booker's 7 plots (7 features)
        booker_scores = []
        for plot_name, plot_info in self.booker_plots.items():
            score = self._detect_booker_plot(text_lower, segments, plot_info)
            booker_scores.append(score)
            features[f'booker_{plot_name}'] = score
        
        # 2. Polti categories (10 features)
        polti_scores = []
        for category_name, category_info in self.polti_categories.items():
            score = self._detect_polti_category(text_lower, category_info)
            polti_scores.append(score)
            features[f'polti_{category_name}'] = score
        
        # 3. Plot purity (2 features)
        features['booker_plot_purity'] = self._calculate_plot_purity(booker_scores)
        features['polti_situation_diversity'] = self._calculate_situation_diversity(polti_scores)
        
        # 4. Dominant plot (2 features)
        features['dominant_booker_plot'] = np.argmax(booker_scores)
        features['booker_plot_strength'] = max(booker_scores)
        
        # 5. Plot blending (3 features)
        blend_features = self._analyze_plot_blending(booker_scores)
        features.update(blend_features)
        
        # 6. Conflict complexity (3 features)
        conflict_features = self._analyze_conflict_complexity(polti_scores)
        features.update(conflict_features)
        
        # 7. Story structure quality (5 features)
        structure_features = self._analyze_structure_quality(text_lower, segments, booker_scores)
        features.update(structure_features)
        
        # 8. Resolution type (3 features)
        resolution_features = self._analyze_resolution(text_lower, segments)
        features.update(resolution_features)
        
        # 9. Cross-plot consistency (2 features)
        features['quest_vs_monster_similarity'] = abs(booker_scores[2] - booker_scores[0])
        features['comedy_vs_tragedy_opposition'] = abs(booker_scores[4] - booker_scores[5])
        
        # 10. Emotional trajectory alignment (3 features)
        trajectory_features = self._analyze_emotional_trajectory(segments, booker_scores)
        features.update(trajectory_features)
        
        return np.array(list(features.values()))
    
    def _split_into_segments(self, text: str, n: int) -> List[str]:
        """Split text into n segments."""
        words = text.split()
        segment_size = max(1, len(words) // n)
        segments = []
        for i in range(n):
            start = i * segment_size
            end = start + segment_size if i < n-1 else len(words)
            segments.append(' '.join(words[start:end]).lower())
        return segments
    
    def _detect_booker_plot(self, text: str, segments: List[str], 
                           plot_info: Dict) -> float:
        """Detect Booker plot type."""
        # Marker presence
        marker_count = sum([text.count(m) for m in plot_info['markers']])
        marker_score = min(1.0, marker_count / 8)
        
        # Stage detection (check if stages appear in sequence)
        stage_scores = []
        for i, stage in enumerate(plot_info['stages']):
            # Look for stage in appropriate region
            region_start = int((i / len(plot_info['stages'])) * len(segments))
            region_end = int(((i + 1) / len(plot_info['stages'])) * len(segments))
            region_text = ' '.join(segments[region_start:region_end])
            
            # Simple stage presence (in real impl, would use more sophisticated detection)
            stage_present = float(any(word in region_text for word in stage.split()))
            stage_scores.append(stage_present)
        
        stage_score = np.mean(stage_scores)
        
        # Combined
        score = 0.60 * marker_score + 0.40 * stage_score
        return score
    
    def _detect_polti_category(self, text: str, category_info: Dict) -> float:
        """Detect Polti situation category."""
        marker_count = sum([text.count(m) for m in category_info['markers']])
        return min(1.0, marker_count / 6)
    
    def _calculate_plot_purity(self, scores: List[float]) -> float:
        """Measure plot purity (single dominant plot vs blended)."""
        if max(scores) < 0.3:
            return 0.0
        
        # Entropy-based
        scores_array = np.array(scores) + 1e-10
        probs = scores_array / scores_array.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(scores))
        
        purity = 1 - (entropy / max_entropy)
        return purity
    
    def _calculate_situation_diversity(self, scores: List[float]) -> float:
        """Measure diversity of dramatic situations."""
        active_situations = sum([1 for s in scores if s > 0.3])
        return active_situations / len(scores)
    
    def _analyze_plot_blending(self, booker_scores: List[float]) -> Dict[str, float]:
        """Analyze how plots blend together."""
        features = {}
        
        # Count active plots
        active_plots = sum([1 for s in booker_scores if s > 0.4])
        features['plot_blending_count'] = min(1.0, active_plots / 3)
        
        # Specific combinations
        # Quest + Monster (common: hero's journey)
        features['quest_monster_blend'] = min(booker_scores[2], booker_scores[0])
        
        # Comedy + Romance (common: rom-com)
        features['comedy_romance_blend'] = min(booker_scores[4], booker_scores[1])
        
        return features
    
    def _analyze_conflict_complexity(self, polti_scores: List[float]) -> Dict[str, float]:
        """Analyze conflict complexity."""
        features = {}
        
        # Number of conflict types
        active_conflicts = sum([1 for s in polti_scores if s > 0.3])
        features['conflict_types_count'] = min(1.0, active_conflicts / 5)
        
        # Conflict intensity (max score)
        features['conflict_intensity'] = max(polti_scores) if polti_scores else 0.0
        
        # Layered conflicts (multiple moderate scores)
        moderate_conflicts = sum([1 for s in polti_scores if 0.3 < s < 0.7])
        features['layered_conflicts'] = min(1.0, moderate_conflicts / 4)
        
        return features
    
    def _analyze_structure_quality(self, text: str, segments: List[str],
                                   booker_scores: List[float]) -> Dict[str, float]:
        """Analyze narrative structure quality."""
        features = {}
        
        # Beginning strength (setup)
        beginning = ' '.join(segments[:4])
        setup_markers = ['introduce', 'establish', 'normal', 'world', 'before']
        features['beginning_strength'] = min(1.0, sum([beginning.count(m) for m in setup_markers]) / 3)
        
        # Middle strength (conflict)
        middle = ' '.join(segments[7:13])
        conflict_markers = ['conflict', 'struggle', 'challenge', 'problem', 'crisis']
        features['middle_strength'] = min(1.0, sum([middle.count(m) for m in conflict_markers]) / 3)
        
        # Ending strength (resolution)
        ending = ' '.join(segments[-4:])
        resolution_markers = ['resolved', 'ended', 'finally', 'concluded', 'peace']
        features['ending_strength'] = min(1.0, sum([ending.count(m) for m in resolution_markers]) / 3)
        
        # Balanced structure
        features['structure_balance'] = 1 - np.std([
            features['beginning_strength'],
            features['middle_strength'],
            features['ending_strength']
        ])
        
        # Overall structure quality
        features['structure_quality'] = (
            0.25 * features['beginning_strength'] +
            0.50 * features['middle_strength'] +
            0.25 * features['ending_strength']
        )
        
        return features
    
    def _analyze_resolution(self, text: str, segments: List[str]) -> Dict[str, float]:
        """Analyze ending/resolution type."""
        ending = ' '.join(segments[-5:])
        
        features = {}
        
        # Happy ending
        happy_markers = ['happy', 'joy', 'celebration', 'triumph', 'success', 'together']
        features['happy_ending'] = min(1.0, sum([ending.count(m) for m in happy_markers]) / 3)
        
        # Tragic ending
        tragic_markers = ['death', 'died', 'lost', 'tragedy', 'fell', 'destroyed']
        features['tragic_ending'] = min(1.0, sum([ending.count(m) for m in tragic_markers]) / 3)
        
        # Ambiguous ending
        ambiguous_markers = ['uncertain', 'unclear', 'maybe', 'perhaps', 'unknown']
        features['ambiguous_ending'] = min(1.0, sum([ending.count(m) for m in ambiguous_markers]) / 2)
        
        return features
    
    def _analyze_emotional_trajectory(self, segments: List[str],
                                     booker_scores: List[float]) -> Dict[str, float]:
        """Analyze emotional trajectory alignment with plot type."""
        features = {}
        
        # Simple sentiment per segment (positive/negative word counts)
        positive_words = ['good', 'happy', 'joy', 'success', 'love', 'hope', 'triumph']
        negative_words = ['bad', 'sad', 'pain', 'death', 'fear', 'loss', 'defeat']
        
        trajectory = []
        for seg in segments:
            pos_count = sum([seg.count(w) for w in positive_words])
            neg_count = sum([seg.count(w) for w in negative_words])
            sentiment = (pos_count - neg_count) / (pos_count + neg_count + 1)
            trajectory.append(sentiment)
        
        # Trajectory patterns
        features['emotional_volatility'] = np.std(trajectory)
        features['emotional_range'] = max(trajectory) - min(trajectory)
        features['net_emotional_change'] = trajectory[-1] - trajectory[0]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Booker plots
        for plot_name in self.booker_plots.keys():
            names.append(f'booker_{plot_name}')
        
        # Polti categories
        for category_name in self.polti_categories.keys():
            names.append(f'polti_{category_name}')
        
        # Aggregate features
        names.extend([
            'booker_plot_purity',
            'polti_situation_diversity',
            'dominant_booker_plot',
            'booker_plot_strength',
            'plot_blending_count',
            'quest_monster_blend',
            'comedy_romance_blend',
            'conflict_types_count',
            'conflict_intensity',
            'layered_conflicts',
            'beginning_strength',
            'middle_strength',
            'ending_strength',
            'structure_balance',
            'structure_quality',
            'happy_ending',
            'tragic_ending',
            'ambiguous_ending',
            'quest_vs_monster_similarity',
            'comedy_vs_tragedy_opposition',
            'emotional_volatility',
            'emotional_range',
            'net_emotional_change'
        ])
        
        return names
    
    def learn_weights_from_data(self, X: List[str], y: np.ndarray,
                                method='correlation') -> Dict[str, float]:
        """Learn empirical plot importance weights."""
        from scipy.stats import pearsonr
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        from sklearn.linear_model import Ridge
        
        features = self.transform(X)
        feature_names = self.get_feature_names()
        
        # Focus on plot type features
        plot_indices = {
            name: i for i, name in enumerate(feature_names)
            if name.startswith('booker_') or name.startswith('polti_')
        }
        
        learned_weights = {}
        
        if method == 'correlation':
            for name, idx in plot_indices.items():
                corr, _ = pearsonr(features[:, idx], y)
                learned_weights[name] = abs(corr)
        
        elif method == 'mutual_info':
            plot_features = features[:, list(plot_indices.values())]
            if len(np.unique(y)) <= 10:
                mi_scores = mutual_info_classif(plot_features, y)
            else:
                mi_scores = mutual_info_regression(plot_features, y)
            mi_scores = mi_scores / (mi_scores.max() + 1e-10)
            
            for i, name in enumerate(plot_indices.keys()):
                learned_weights[name] = mi_scores[i]
        
        elif method == 'regression':
            plot_features = features[:, list(plot_indices.values())]
            model = Ridge(alpha=1.0)
            model.fit(plot_features, y)
            coeffs = np.abs(model.coef_)
            coeffs = coeffs / (coeffs.max() + 1e-10)
            
            for i, name in enumerate(plot_indices.keys()):
                learned_weights[name] = coeffs[i]
        
        self.learned_weights = learned_weights
        self.use_learned_weights = True
        
        return learned_weights
    
    def compare_theoretical_vs_empirical(self) -> Dict:
        """Compare Booker's theoretical weights to empirical weights."""
        if not self.learned_weights:
            raise ValueError("Must call learn_weights_from_data first")
        
        comparison = {'plots': {}, 'summary': {}}
        
        for plot_name, plot_info in self.booker_plots.items():
            full_name = f'booker_{plot_name}'
            theoretical = plot_info['weight']
            empirical = self.learned_weights.get(full_name, 0.5)
            deviation = theoretical - empirical
            
            comparison['plots'][full_name] = {
                'theoretical_weight': theoretical,
                'empirical_weight': empirical,
                'deviation': deviation,
                'booker_overvalued': deviation > 0.2,
                'booker_undervalued': deviation < -0.2
            }
        
        deviations = [p['deviation'] for p in comparison['plots'].values()]
        comparison['summary'] = {
            'mean_absolute_deviation': np.mean(np.abs(deviations)),
            'theory_validated': np.mean(np.abs(deviations)) < 0.15,
            'most_overvalued': max(comparison['plots'].items(), 
                                  key=lambda x: x[1]['deviation'])[0],
            'most_undervalued': min(comparison['plots'].items(), 
                                   key=lambda x: x[1]['deviation'])[0]
        }
        
        return comparison


# Convenience function
def discover_plot_patterns(texts: List[str], outcomes: np.ndarray,
                          method='correlation') -> Dict:
    """
    Discover which plot types actually predict success.
    
    Returns learned weights and validation of Booker's theory.
    """
    transformer = PlotArchetypeTransformer()
    transformer.fit(texts)
    
    learned_weights = transformer.learn_weights_from_data(texts, outcomes, method)
    comparison = transformer.compare_theoretical_vs_empirical()
    
    return {
        'learned_weights': learned_weights,
        'theoretical_validation': comparison,
        'transformer': transformer
    }

