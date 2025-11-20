"""
Narrative Context Weighting System

"Each game is a story. Story weight is by context. Better stories win over time."

Not all games are equal narratively. A championship game carries more narrative
weight than a regular season game. Rivalry games have different weight than
random matchups. The narrative's CONTEXT determines its predictive power.
"""

from typing import Dict, List, Any
import numpy as np
import re


class NarrativeContextWeighter:
    """
    Assigns narrative weights based on story context.
    
    Theory: Better stories win, but story importance varies by context.
    Not all games have equal narrative significance.
    """
    
    def __init__(self):
        """Initialize context detection patterns."""
        
        # High narrative weight indicators
        self.high_stakes_patterns = {
            'championship': 2.5,
            'playoff': 2.0,
            'finals': 2.5,
            'elimination': 2.3,
            'series': 1.8,
            'must-win': 2.0,
            'do or die': 2.2
        }
        
        self.rivalry_patterns = {
            'rivalry': 1.8,
            'historic': 1.6,
            'matchup': 1.4,
            'showdown': 1.5,
            'battle': 1.4
        }
        
        self.momentum_patterns = {
            'streak': 1.6,
            'momentum': 1.5,
            'surge': 1.6,
            'run': 1.4,
            'hot': 1.5,
            'rolling': 1.5
        }
        
        self.significance_patterns = {
            'defining': 1.7,
            'pivotal': 1.6,
            'crucial': 1.6,
            'critical': 1.6,
            'important': 1.4
        }
        
        # Low narrative weight indicators
        self.low_weight_patterns = {
            'regular season': 0.8,
            'routine': 0.7,
            'early season': 0.7,
            'meaningless': 0.5
        }
    
    def compute_narrative_weight(self, narrative: str, context: Dict = None) -> float:
        """
        Compute narrative weight for a game based on story context.
        
        Parameters
        ----------
        narrative : str
            Game/team narrative text
        context : dict
            Additional context (date, standings, etc.)
        
        Returns
        -------
        weight : float
            Narrative weight (0.5 to 3.0)
            1.0 = baseline, >1.0 = more important, <1.0 = less important
        """
        narrative_lower = narrative.lower()
        
        weight = 1.0  # Baseline
        
        # Check high-stakes patterns
        for pattern, multiplier in self.high_stakes_patterns.items():
            if pattern in narrative_lower:
                weight = max(weight, multiplier)
        
        # Check rivalry patterns
        for pattern, multiplier in self.rivalry_patterns.items():
            if pattern in narrative_lower:
                weight = max(weight, multiplier)
        
        # Check momentum patterns
        for pattern, multiplier in self.momentum_patterns.items():
            if pattern in narrative_lower:
                weight = max(weight, multiplier)
        
        # Check significance patterns
        for pattern, multiplier in self.significance_patterns.items():
            if pattern in narrative_lower:
                weight = max(weight, multiplier)
        
        # Check low-weight patterns
        for pattern, multiplier in self.low_weight_patterns.items():
            if pattern in narrative_lower:
                weight = min(weight, multiplier)
        
        # Contextual adjustments
        if context:
            # Late season games matter more
            if context.get('games_remaining', 82) < 20:
                weight *= 1.2
            
            # Close standings matter more
            if context.get('standings_gap', 10) < 3:
                weight *= 1.15
            
            # Playoff implications
            if context.get('playoff_implications', False):
                weight *= 1.5
        
        # Clamp to reasonable range
        return max(0.5, min(3.0, weight))
    
    def compute_temporal_weights(self, games: List[Dict]) -> np.ndarray:
        """
        Compute temporal narrative weights: later games matter more.
        
        Theory: "Better stories win over time, better ones over longer periods"
        
        Parameters
        ----------
        games : list of dict
            Ordered games (chronological)
        
        Returns
        -------
        weights : np.ndarray
            Temporal weights (increases over time)
        """
        n = len(games)
        
        # Linear increase: early games weight 0.8, late games weight 1.2
        # Theory: narrative effects compound over season
        weights = np.linspace(0.8, 1.2, n)
        
        return weights
    
    def compute_cumulative_narrative_strength(self, narratives: List[str]) -> np.ndarray:
        """
        Compute cumulative narrative strength over time.
        
        Theory: "Better stories win over longer periods"
        Narratives compound - each game adds to or subtracts from team story.
        
        Parameters
        ----------
        narratives : list of str
            Team narratives in chronological order
        
        Returns
        -------
        cumulative_strength : np.ndarray
            Running narrative strength score
        """
        strengths = []
        running_score = 0.0
        
        for narrative in narratives:
            # Assess narrative quality
            quality = self._assess_narrative_quality(narrative)
            
            # Update cumulative score (with decay)
            running_score = running_score * 0.95 + quality * 0.05
            
            strengths.append(running_score)
        
        return np.array(strengths)
    
    def _assess_narrative_quality(self, narrative: str) -> float:
        """
        Assess intrinsic quality of narrative.
        
        "Better stories win" - what makes a story better?
        """
        score = 0.5  # Baseline
        
        # Coherence indicators
        if len(narrative) > 100:
            score += 0.1  # Detailed stories
        
        # Confidence markers
        confidence_words = ['will', 'championship', 'dominant', 'strong', 'confident']
        score += sum(0.05 for word in confidence_words if word in narrative.lower())
        
        # Forward momentum
        future_words = ['will', 'going to', 'next', 'future', 'ahead']
        score += sum(0.03 for word in future_words if word in narrative.lower())
        
        # Achievement language
        achievement_words = ['won', 'victory', 'success', 'champion', 'leader']
        score += sum(0.04 for word in achievement_words if word in narrative.lower())
        
        return min(1.0, max(0.0, score))


class MultiHorizonNarrativeTester:
    """
    Tests narrative effects across multiple time horizons.
    
    Theory: "Better stories win over time, better ones over longer periods perhaps"
    
    Tests if narrative prediction improves with:
    - Single game
    - 5-game window
    - 10-game window
    - Season-long
    """
    
    def __init__(self):
        self.horizons = {
            'immediate': 1,    # Next game only
            'short': 5,        # Next 5 games
            'medium': 10,      # Next 10 games
            'season': 82       # Full season
        }
    
    def test_across_horizons(self, games: List[Dict], predictions: np.ndarray) -> Dict[str, float]:
        """
        Test prediction accuracy across different time horizons.
        
        Parameters
        ----------
        games : list
            Chronological games with outcomes
        predictions : np.ndarray
            Model predictions
        
        Returns
        -------
        accuracies : dict
            Accuracy for each time horizon
        """
        results = {}
        
        for horizon_name, window_size in self.horizons.items():
            accuracies = []
            
            for i in range(0, len(games) - window_size, window_size):
                window_games = games[i:i+window_size]
                window_preds = predictions[i:i+window_size]
                
                # Average prediction over window
                avg_pred = np.mean(window_preds)
                
                # Actual outcome over window
                wins = sum(1 for g in window_games if g.get('won', False))
                win_rate = wins / len(window_games)
                
                # Check if prediction matches trend
                correct = (avg_pred > 0.5 and win_rate > 0.5) or (avg_pred < 0.5 and win_rate < 0.5)
                accuracies.append(1.0 if correct else 0.0)
            
            results[horizon_name] = np.mean(accuracies) if accuracies else 0.0
        
        return results


class StoryOutcomeAnalyzer:
    """
    Analyzes what makes stories win.
    
    "Many things go into a story's outcome"
    
    Not just features - the INTERACTION of:
    - Context (what's at stake)
    - Quality (how well-told)
    - Timing (when in season)
    - Momentum (recent trajectory)
    - Stakes (what matters)
    """
    
    def analyze_winning_stories(self, games: List[Dict]) -> Dict[str, Any]:
        """
        Identify characteristics of winning narratives.
        
        Parameters
        ----------
        games : list
            Games with narratives and outcomes
        
        Returns
        -------
        analysis : dict
            Patterns in winning vs losing stories
        """
        winning_narratives = []
        losing_narratives = []
        
        for game in games:
            if not game.get('narrative'):
                continue
            
            if game.get('won', False):
                winning_narratives.append(game['narrative'])
            else:
                losing_narratives.append(game['narrative'])
        
        analysis = {
            'winning_patterns': self._extract_patterns(winning_narratives),
            'losing_patterns': self._extract_patterns(losing_narratives),
            'discriminating_features': self._find_discriminators(winning_narratives, losing_narratives)
        }
        
        return analysis
    
    def _extract_patterns(self, narratives: List[str]) -> Dict:
        """Extract common patterns in narrative set."""
        all_text = ' '.join(narratives).lower()
        
        words = re.findall(r'\b\w+\b', all_text)
        from collections import Counter
        common = Counter(words).most_common(20)
        
        return {
            'common_words': common,
            'avg_length': np.mean([len(n) for n in narratives]),
            'avg_confidence_markers': np.mean([
                sum(1 for word in ['champion', 'strong', 'confident', 'dominant'] if word in n.lower())
                for n in narratives
            ])
        }
    
    def _find_discriminators(self, winning: List[str], losing: List[str]) -> List[str]:
        """Find words that appear more in winning narratives."""
        from collections import Counter
        
        win_words = Counter(' '.join(winning).lower().split())
        lose_words = Counter(' '.join(losing).lower().split())
        
        discriminators = []
        for word, win_count in win_words.most_common(50):
            lose_count = lose_words.get(word, 0)
            
            if win_count > lose_count * 1.5:  # Appears 50% more in winning
                discriminators.append(word)
        
        return discriminators[:10]

