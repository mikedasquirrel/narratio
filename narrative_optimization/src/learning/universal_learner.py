"""
Universal Archetype Learner

Learns patterns that transfer across ALL domains.
Examples: "underdog story", "comeback", "rivalry", "pressure performance"

These are the meta-narratives that appear regardless of domain.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re


class UniversalArchetypeLearner:
    """
    Learns universal (cross-domain) archetype patterns.
    
    Universal patterns are those that appear across multiple domains
    and have predictive power regardless of context.
    
    Examples:
    - "Underdog story": lower ranked entity wins
    - "Comeback": trailing entity recovers
    - "Rivalry": repeated matchups with history
    - "Pressure moment": high-stakes situation
    - "Dominance": consistent superiority
    
    These patterns transcend domain boundaries.
    """
    
    def __init__(self):
        self.patterns = {}  # pattern_name -> pattern_data
        self.validated_patterns = {}
        self.pattern_history = []  # tracking evolution
        
    def discover_patterns(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        n_patterns: int = 10,
        min_frequency: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Discover universal patterns from multi-domain data.
        
        Uses clustering + frequency analysis to find recurring themes.
        
        Parameters
        ----------
        texts : list of str
            Texts from ALL domains
        outcomes : ndarray
            Outcomes
        n_patterns : int
            Number of patterns to discover
        min_frequency : float
            Minimum pattern frequency
        
        Returns
        -------
        dict
            Discovered patterns
        """
        print(f"  Discovering universal patterns from {len(texts)} texts...")
        
        if len(texts) < 20:
            print(f"  ⚠ Too few texts ({len(texts)}), skipping discovery")
            return {}
        
        discovered = {}
        
        # 1. Discover semantic themes via clustering
        semantic_patterns = self._discover_semantic_themes(texts, outcomes, n_patterns)
        discovered.update(semantic_patterns)
        
        # 2. Discover narrative archetypes (underdog, comeback, etc.)
        narrative_patterns = self._discover_narrative_archetypes(texts, outcomes)
        discovered.update(narrative_patterns)
        
        # 3. Discover emotional/tonal patterns
        emotional_patterns = self._discover_emotional_patterns(texts, outcomes)
        discovered.update(emotional_patterns)
        
        # Filter by frequency
        filtered = {}
        for pattern_name, pattern_data in discovered.items():
            frequency = pattern_data.get('frequency', 0.0)
            if frequency >= min_frequency:
                filtered[pattern_name] = pattern_data
        
        self.patterns = filtered
        return filtered
    
    def _discover_semantic_themes(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        n_themes: int
    ) -> Dict[str, Dict]:
        """Discover semantic themes via clustering."""
        try:
            # Create embeddings
            vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 3),
                min_df=2
            )
            embeddings = vectorizer.fit_transform(texts).toarray()
            
            # Find optimal number of clusters
            optimal_k = min(n_themes, len(texts) // 10, 15)
            if optimal_k < 2:
                return {}
            
            # Cluster
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Extract theme from each cluster
            themes = {}
            for cluster_id in range(optimal_k):
                cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
                cluster_outcomes = outcomes[clusters == cluster_id]
                
                if len(cluster_texts) < 5:
                    continue
                
                # Extract keywords
                keywords = self._extract_theme_keywords(cluster_texts)
                
                # Calculate predictive power
                cluster_win_rate = np.mean(cluster_outcomes) if len(cluster_outcomes) > 0 else 0.5
                overall_win_rate = np.mean(outcomes)
                lift = abs(cluster_win_rate - overall_win_rate)
                
                theme_name = f"universal_theme_{cluster_id + 1}"
                themes[theme_name] = {
                    'type': 'semantic_theme',
                    'keywords': keywords,
                    'frequency': len(cluster_texts) / len(texts),
                    'win_rate': cluster_win_rate,
                    'lift': lift,
                    'sample_size': len(cluster_texts)
                }
            
            return themes
            
        except Exception as e:
            print(f"  ⚠ Error discovering semantic themes: {e}")
            return {}
    
    def _discover_narrative_archetypes(
        self,
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """Discover classic narrative archetypes."""
        archetypes = {}
        
        # Underdog pattern
        underdog_patterns = ['underdog', 'lower ranked', 'outsider', 'upset', 'surprising']
        underdog_matches = [
            i for i, text in enumerate(texts)
            if any(pattern in text.lower() for pattern in underdog_patterns)
        ]
        
        if len(underdog_matches) >= 10:
            underdog_outcomes = outcomes[underdog_matches]
            archetypes['universal_underdog'] = {
                'type': 'narrative_archetype',
                'keywords': underdog_patterns,
                'frequency': len(underdog_matches) / len(texts),
                'win_rate': np.mean(underdog_outcomes),
                'lift': abs(np.mean(underdog_outcomes) - np.mean(outcomes)),
                'sample_size': len(underdog_matches),
                'description': 'Underdog story: lower-ranked or unexpected contender'
            }
        
        # Comeback pattern
        comeback_patterns = ['comeback', 'recovered', 'rallied', 'fought back', 'resilient']
        comeback_matches = [
            i for i, text in enumerate(texts)
            if any(pattern in text.lower() for pattern in comeback_patterns)
        ]
        
        if len(comeback_matches) >= 10:
            comeback_outcomes = outcomes[comeback_matches]
            archetypes['universal_comeback'] = {
                'type': 'narrative_archetype',
                'keywords': comeback_patterns,
                'frequency': len(comeback_matches) / len(texts),
                'win_rate': np.mean(comeback_outcomes),
                'lift': abs(np.mean(comeback_outcomes) - np.mean(outcomes)),
                'sample_size': len(comeback_matches),
                'description': 'Comeback narrative: recovery from deficit'
            }
        
        # Rivalry pattern
        rivalry_patterns = ['rivalry', 'rematch', 'history', 'previous', 'head-to-head']
        rivalry_matches = [
            i for i, text in enumerate(texts)
            if any(pattern in text.lower() for pattern in rivalry_patterns)
        ]
        
        if len(rivalry_matches) >= 10:
            rivalry_outcomes = outcomes[rivalry_matches]
            archetypes['universal_rivalry'] = {
                'type': 'narrative_archetype',
                'keywords': rivalry_patterns,
                'frequency': len(rivalry_matches) / len(texts),
                'win_rate': np.mean(rivalry_outcomes),
                'lift': abs(np.mean(rivalry_outcomes) - np.mean(outcomes)),
                'sample_size': len(rivalry_matches),
                'description': 'Rivalry narrative: repeated competition with history'
            }
        
        # Dominance pattern
        dominance_patterns = ['dominant', 'undefeated', 'streak', 'unstoppable', 'overwhelming']
        dominance_matches = [
            i for i, text in enumerate(texts)
            if any(pattern in text.lower() for pattern in dominance_patterns)
        ]
        
        if len(dominance_matches) >= 10:
            dominance_outcomes = outcomes[dominance_matches]
            archetypes['universal_dominance'] = {
                'type': 'narrative_archetype',
                'keywords': dominance_patterns,
                'frequency': len(dominance_matches) / len(texts),
                'win_rate': np.mean(dominance_outcomes),
                'lift': abs(np.mean(dominance_outcomes) - np.mean(outcomes)),
                'sample_size': len(dominance_matches),
                'description': 'Dominance narrative: consistent superiority'
            }
        
        # Pressure moment pattern
        pressure_patterns = ['pressure', 'crucial', 'critical', 'decisive', 'clutch', 'high stakes']
        pressure_matches = [
            i for i, text in enumerate(texts)
            if any(pattern in text.lower() for pattern in pressure_patterns)
        ]
        
        if len(pressure_matches) >= 10:
            pressure_outcomes = outcomes[pressure_matches]
            archetypes['universal_pressure'] = {
                'type': 'narrative_archetype',
                'keywords': pressure_patterns,
                'frequency': len(pressure_matches) / len(texts),
                'win_rate': np.mean(pressure_outcomes),
                'lift': abs(np.mean(pressure_outcomes) - np.mean(outcomes)),
                'sample_size': len(pressure_matches),
                'description': 'Pressure moment: high-stakes critical situation'
            }
        
        return archetypes
    
    def _discover_emotional_patterns(
        self,
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """Discover emotional/tonal patterns."""
        emotional = {}
        
        # Positive emotion
        positive_patterns = ['confident', 'excited', 'motivated', 'inspired', 'passionate']
        positive_matches = [
            i for i, text in enumerate(texts)
            if any(pattern in text.lower() for pattern in positive_patterns)
        ]
        
        if len(positive_matches) >= 10:
            positive_outcomes = outcomes[positive_matches]
            emotional['universal_positive_emotion'] = {
                'type': 'emotional_pattern',
                'keywords': positive_patterns,
                'frequency': len(positive_matches) / len(texts),
                'win_rate': np.mean(positive_outcomes),
                'lift': abs(np.mean(positive_outcomes) - np.mean(outcomes)),
                'sample_size': len(positive_matches),
                'description': 'Positive emotional tone'
            }
        
        # Negative emotion
        negative_patterns = ['doubt', 'struggled', 'concerned', 'worried', 'frustrated']
        negative_matches = [
            i for i, text in enumerate(texts)
            if any(pattern in text.lower() for pattern in negative_patterns)
        ]
        
        if len(negative_matches) >= 10:
            negative_outcomes = outcomes[negative_matches]
            emotional['universal_negative_emotion'] = {
                'type': 'emotional_pattern',
                'keywords': negative_patterns,
                'frequency': len(negative_matches) / len(texts),
                'win_rate': np.mean(negative_outcomes),
                'lift': abs(np.mean(negative_outcomes) - np.mean(outcomes)),
                'sample_size': len(negative_matches),
                'description': 'Negative emotional tone'
            }
        
        return emotional
    
    def _extract_theme_keywords(self, texts: List[str], n_keywords: int = 5) -> List[str]:
        """Extract representative keywords for a theme."""
        # Simple: most frequent meaningful words
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            all_words.extend(words)
        
        # Filter stop words
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were',
            'their', 'there', 'would', 'could', 'should', 'about'
        }
        meaningful = [w for w in all_words if w not in stop_words]
        
        word_counts = Counter(meaningful)
        return [word for word, count in word_counts.most_common(n_keywords)]
    
    def get_patterns(self) -> Dict[str, Dict]:
        """Get all discovered patterns."""
        return self.patterns
    
    def get_validated_patterns(self) -> Dict[str, Dict]:
        """Get only validated patterns."""
        return self.validated_patterns
    
    def set_validated_patterns(self, patterns: Dict[str, Dict]):
        """Set validated patterns (after validation)."""
        self.validated_patterns = patterns
    
    def get_pattern_description(self, pattern_name: str) -> str:
        """Get human-readable description of a pattern."""
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            desc = pattern.get('description', '')
            keywords = pattern.get('keywords', [])
            freq = pattern.get('frequency', 0.0)
            win_rate = pattern.get('win_rate', 0.5)
            
            if desc:
                return f"{desc} (freq={freq:.1%}, win_rate={win_rate:.1%})"
            else:
                return f"Pattern: {', '.join(keywords[:3])} (freq={freq:.1%}, win_rate={win_rate:.1%})"
        return "Unknown pattern"

