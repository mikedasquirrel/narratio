"""
Thematic Archetype Transformer

Detects Frye's four mythoi (Comedy, Romance, Tragedy, Irony/Satire).
Maps to θ/λ phase space and moral frameworks.

Based on:
- Northrop Frye's "Anatomy of Criticism" (1957)

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
from typing import List, Dict
from ..base import NarrativeTransformer


class ThematicArchetypeTransformer(NarrativeTransformer):
    """
    Detect Frye's four mythoi and thematic patterns.
    Extracts ~25 features for thematic analysis.
    """
    
    def __init__(self, use_learned_weights=False, learned_weights=None):
        super().__init__(
            narrative_id="thematic_archetype",
            description="Frye's four mythoi + moral frameworks"
        )
        
        self.use_learned_weights = use_learned_weights
        self.learned_weights = learned_weights or {}
        
        # Frye's Four Mythoi
        self.frye_mythoi = {
            'comedy': {
                'markers': ['confusion', 'misunderstanding', 'mix-up', 'revealed', 'truth',
                           'marriage', 'united', 'celebration', 'harmony', 'resolved', 'together'],
                'trajectory': 'chaos → revelation → union',
                'season': 'spring',
                'weight': 0.8
            },
            'romance': {
                'markers': ['quest', 'adventure', 'hero', 'noble', 'brave', 'good', 'evil',
                           'triumph', 'victory', 'magical', 'exotic', 'ideal'],
                'trajectory': 'adventure → triumph',
                'season': 'summer',
                'weight': 1.0
            },
            'tragedy': {
                'markers': ['hubris', 'pride', 'flaw', 'fatal', 'downfall', 'fell',
                           'death', 'catastrophe', 'inevitable', 'doomed', 'too late'],
                'trajectory': 'peak → catastrophe',
                'season': 'autumn',
                'weight': 1.0
            },
            'irony': {
                'markers': ['futile', 'meaningless', 'absurd', 'system', 'trapped',
                           'powerless', 'ambiguous', 'unclear', 'unresolved', 'cycle'],
                'trajectory': 'ambiguous/cyclical',
                'season': 'winter',
                'weight': 0.7
            }
        }
        
        # Moral frameworks
        self.moral_frameworks = {
            'good_vs_evil': ['good', 'evil', 'virtue', 'sin', 'right', 'wrong'],
            'redemption': ['redemption', 'forgive', 'atone', 'save', 'redeem'],
            'justice': ['justice', 'fair', 'deserve', 'karma', 'balance'],
            'sacrifice': ['sacrifice', 'give up', 'price', 'cost', 'loss'],
            'corruption': ['corrupt', 'fall', 'betray', 'lost way', 'tainted']
        }
    
    def fit(self, X: List[str], y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        features = []
        for text in X:
            features.append(self._extract_thematic_features(text))
        return np.array(features)
    
    def _extract_thematic_features(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        segments = self._split_text(text, 20)
        features = {}
        
        # 1. Frye's four mythoi (4 features)
        mythos_scores = []
        for mythos_name, mythos_info in self.frye_mythoi.items():
            score = self._detect_mythos(text_lower, segments, mythos_info)
            mythos_scores.append(score)
            features[f'frye_{mythos_name}'] = score
        
        # 2. Mythos purity (1 feature)
        features['mythos_purity'] = self._calculate_purity(mythos_scores)
        
        # 3. Dominant mythos (1 feature)
        features['dominant_mythos'] = np.argmax(mythos_scores)
        
        # 4. Moral frameworks (5 features)
        for framework_name, framework_markers in self.moral_frameworks.items():
            count = sum([text_lower.count(m) for m in framework_markers])
            features[f'moral_{framework_name}'] = min(1.0, count / 3)
        
        # 5. Emotional trajectory (3 features)
        trajectory = self._analyze_emotional_trajectory(segments)
        features.update(trajectory)
        
        # 6. Seasonal alignment (4 features)
        features['spring_alignment'] = mythos_scores[0]  # Comedy
        features['summer_alignment'] = mythos_scores[1]  # Romance
        features['autumn_alignment'] = mythos_scores[2]  # Tragedy
        features['winter_alignment'] = mythos_scores[3]  # Irony
        
        # 7. Thematic clarity (1 feature)
        features['thematic_clarity'] = max(mythos_scores)
        
        # 8. θ/λ estimates (2 features)
        features['estimated_theta'] = self._estimate_theta(mythos_scores)
        features['estimated_lambda'] = self._estimate_lambda(mythos_scores)
        
        return np.array(list(features.values()))
    
    def _split_text(self, text: str, n: int) -> List[str]:
        words = text.split()
        seg_size = max(1, len(words) // n)
        return [' '.join(words[i*seg_size:(i+1)*seg_size]).lower() for i in range(n)]
    
    def _detect_mythos(self, text: str, segments: List[str], mythos_info: Dict) -> float:
        """Detect mythos presence."""
        marker_count = sum([text.count(m) for m in mythos_info['markers']])
        marker_score = min(1.0, marker_count / 8)
        
        # Check trajectory pattern
        trajectory_score = 0.5  # Default
        # This would be more sophisticated in full implementation
        
        return 0.70 * marker_score + 0.30 * trajectory_score
    
    def _calculate_purity(self, scores: List[float]) -> float:
        """Calculate mythos purity (single dominant vs mixed)."""
        if max(scores) < 0.3:
            return 0.0
        
        scores_array = np.array(scores) + 1e-10
        probs = scores_array / scores_array.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(scores))
        
        return 1 - (entropy / max_entropy)
    
    def _analyze_emotional_trajectory(self, segments: List[str]) -> Dict[str, float]:
        """Analyze emotional trajectory."""
        positive = ['good', 'happy', 'joy', 'love', 'hope', 'success']
        negative = ['bad', 'sad', 'pain', 'fear', 'loss', 'death']
        
        trajectory = []
        for seg in segments:
            pos = sum([seg.count(w) for w in positive])
            neg = sum([seg.count(w) for w in negative])
            sentiment = (pos - neg) / (pos + neg + 1)
            trajectory.append(sentiment)
        
        return {
            'emotional_volatility': np.std(trajectory),
            'emotional_range': max(trajectory) - min(trajectory),
            'net_emotional_change': trajectory[-1] - trajectory[0]
        }
    
    def _estimate_theta(self, mythos_scores: List[float]) -> float:
        """Estimate θ (awareness) from mythos."""
        # Irony/Satire = high θ, Romance = low θ
        comedy, romance, tragedy, irony = mythos_scores
        return (irony * 0.85 + tragedy * 0.55 + comedy * 0.30 + romance * 0.20)
    
    def _estimate_lambda(self, mythos_scores: List[float]) -> float:
        """Estimate λ (constraints) from mythos."""
        # Tragedy = high λ, Romance = low λ
        comedy, romance, tragedy, irony = mythos_scores
        return (tragedy * 0.75 + comedy * 0.50 + romance * 0.30 + irony * 0.50)
    
    def get_feature_names(self) -> List[str]:
        names = []
        for mythos in self.frye_mythoi.keys():
            names.append(f'frye_{mythos}')
        names.extend(['mythos_purity', 'dominant_mythos'])
        for framework in self.moral_frameworks.keys():
            names.append(f'moral_{framework}')
        names.extend(['emotional_volatility', 'emotional_range', 'net_emotional_change'])
        names.extend(['spring_alignment', 'summer_alignment', 'autumn_alignment', 'winter_alignment'])
        names.extend(['thematic_clarity', 'estimated_theta', 'estimated_lambda'])
        return names
    
    def learn_weights_from_data(self, X: List[str], y: np.ndarray, method='correlation') -> Dict[str, float]:
        """Learn empirical mythos importance weights."""
        from scipy.stats import pearsonr
        
        features = self.transform(X)
        feature_names = self.get_feature_names()
        learned_weights = {}
        
        for i, name in enumerate(feature_names):
            if name.startswith('frye_'):
                corr, _ = pearsonr(features[:, i], y)
                learned_weights[name] = abs(corr)
        
        self.learned_weights = learned_weights
        self.use_learned_weights = True
        return learned_weights

