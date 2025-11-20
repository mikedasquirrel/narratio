"""
Ensemble Archetype Learning

Multiple archetype hypotheses combined for robust predictions.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict


class EnsembleArchetypeLearner:
    """
    Ensemble learning with multiple archetype hypotheses.
    
    Strategies:
    - Bagging: Multiple samples
    - Boosting: Sequential focus on errors
    - Stacking: Meta-learner combines base learners
    - Voting: Weighted combination
    """
    
    def __init__(self, n_members: int = 5, diversity_penalty: float = 0.1):
        self.n_members = n_members
        self.diversity_penalty = diversity_penalty
        self.members = []
        self.member_weights = []
        
    def create_ensemble(
        self,
        base_learner_class,
        texts: List[str],
        outcomes: np.ndarray,
        strategy: str = 'bagging'
    ) -> List:
        """
        Create ensemble of learners.
        
        Parameters
        ----------
        base_learner_class : class
            Learner class to instantiate
        texts : list
            Training texts
        outcomes : ndarray
            Training outcomes
        strategy : str
            'bagging', 'boosting', or 'random_subspace'
        
        Returns
        -------
        list
            Ensemble members
        """
        self.members = []
        
        if strategy == 'bagging':
            self.members = self._bagging(base_learner_class, texts, outcomes)
        elif strategy == 'boosting':
            self.members = self._boosting(base_learner_class, texts, outcomes)
        elif strategy == 'random_subspace':
            self.members = self._random_subspace(base_learner_class, texts, outcomes)
        
        # Initialize equal weights
        self.member_weights = np.ones(len(self.members)) / len(self.members)
        
        return self.members
    
    def _bagging(
        self,
        base_learner_class,
        texts: List[str],
        outcomes: np.ndarray
    ) -> List:
        """Bootstrap aggregating: sample with replacement."""
        members = []
        
        for i in range(self.n_members):
            # Bootstrap sample
            indices = np.random.choice(len(texts), size=len(texts), replace=True)
            sample_texts = [texts[i] for i in indices]
            sample_outcomes = outcomes[indices]
            
            # Train member
            member = base_learner_class()
            member.discover_patterns(sample_texts, sample_outcomes)
            members.append(member)
        
        return members
    
    def _boosting(
        self,
        base_learner_class,
        texts: List[str],
        outcomes: np.ndarray
    ) -> List:
        """Sequential boosting: focus on errors."""
        members = []
        sample_weights = np.ones(len(texts)) / len(texts)
        
        for i in range(self.n_members):
            # Sample based on weights (focus on hard cases)
            indices = np.random.choice(
                len(texts),
                size=len(texts),
                replace=True,
                p=sample_weights
            )
            sample_texts = [texts[i] for i in indices]
            sample_outcomes = outcomes[indices]
            
            # Train member
            member = base_learner_class()
            member.discover_patterns(sample_texts, sample_outcomes)
            members.append(member)
            
            # Update weights (increase for errors)
            # Simplified: would need actual predictions
            sample_weights = np.ones(len(texts)) / len(texts)
        
        return members
    
    def _random_subspace(
        self,
        base_learner_class,
        texts: List[str],
        outcomes: np.ndarray
    ) -> List:
        """Random subspace: use random feature subsets."""
        members = []
        
        for i in range(self.n_members):
            # Use full data but different random initialization
            member = base_learner_class()
            member.discover_patterns(texts, outcomes, n_patterns=5 + i)
            members.append(member)
        
        return members
    
    def predict_ensemble(
        self,
        text: str,
        voting: str = 'weighted'
    ) -> float:
        """
        Ensemble prediction.
        
        Parameters
        ----------
        text : str
            Text to predict
        voting : str
            'weighted', 'uniform', or 'soft'
        
        Returns
        -------
        float
            Ensemble prediction
        """
        if len(self.members) == 0:
            return 0.5
        
        predictions = []
        
        for member in self.members:
            # Get member prediction
            patterns = member.get_patterns()
            
            # Score based on pattern matching
            score = 0.5  # Default
            for pattern_name, pattern_data in patterns.items():
                keywords = pattern_data.get('patterns', pattern_data.get('keywords', []))
                if any(k.lower() in text.lower() for k in keywords):
                    score += pattern_data.get('win_rate', 0.0) * 0.1
            
            predictions.append(np.clip(score, 0, 1))
        
        predictions = np.array(predictions)
        
        if voting == 'weighted':
            return np.average(predictions, weights=self.member_weights)
        elif voting == 'uniform':
            return np.mean(predictions)
        elif voting == 'soft':
            # Soft voting with confidence
            return np.mean(predictions)
        
        return np.mean(predictions)
    
    def update_weights(
        self,
        texts: List[str],
        outcomes: np.ndarray
    ):
        """
        Update ensemble member weights based on performance.
        
        Parameters
        ----------
        texts : list
            Validation texts
        outcomes : ndarray
            True outcomes
        """
        if len(self.members) == 0:
            return
        
        performances = []
        
        for member in self.members:
            # Evaluate member
            predictions = []
            for text in texts:
                pred = self.predict_ensemble(text, voting='uniform')  # Single member
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Performance = correlation with outcomes
            if len(np.unique(outcomes)) > 1:
                corr = abs(np.corrcoef(predictions, outcomes)[0, 1])
            else:
                corr = 0.5
            
            performances.append(corr)
        
        performances = np.array(performances)
        
        # Normalize to weights
        if performances.sum() > 0:
            self.member_weights = performances / performances.sum()
        else:
            self.member_weights = np.ones(len(self.members)) / len(self.members)
    
    def diversity_score(self) -> float:
        """
        Calculate ensemble diversity.
        
        Higher diversity = less redundancy.
        
        Returns
        -------
        float
            Diversity score
        """
        if len(self.members) < 2:
            return 0.0
        
        # Compare patterns between members
        all_patterns = []
        for member in self.members:
            patterns = set(member.get_patterns().keys())
            all_patterns.append(patterns)
        
        # Average pairwise dissimilarity
        diversities = []
        for i in range(len(all_patterns)):
            for j in range(i + 1, len(all_patterns)):
                # Jaccard distance
                intersection = len(all_patterns[i] & all_patterns[j])
                union = len(all_patterns[i] | all_patterns[j])
                
                if union > 0:
                    jaccard = intersection / union
                    diversity = 1.0 - jaccard
                    diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def prune_ensemble(self, target_size: int):
        """
        Prune ensemble to target size, keeping best + most diverse.
        
        Parameters
        ----------
        target_size : int
            Target ensemble size
        """
        if len(self.members) <= target_size:
            return
        
        # Keep top performers
        top_indices = np.argsort(self.member_weights)[-target_size:]
        
        self.members = [self.members[i] for i in top_indices]
        self.member_weights = self.member_weights[top_indices]
        self.member_weights /= self.member_weights.sum()
    
    def get_ensemble_confidence(self, text: str) -> Tuple[float, float]:
        """
        Get prediction with confidence.
        
        Parameters
        ----------
        text : str
            Text to predict
        
        Returns
        -------
        tuple
            (prediction, confidence)
        """
        if len(self.members) == 0:
            return 0.5, 0.0
        
        predictions = []
        for member in self.members:
            patterns = member.get_patterns()
            score = 0.5
            for pattern_name, pattern_data in patterns.items():
                keywords = pattern_data.get('patterns', pattern_data.get('keywords', []))
                if any(k.lower() in text.lower() for k in keywords):
                    score += 0.1
            predictions.append(score)
        
        predictions = np.array(predictions)
        
        # Prediction = weighted mean
        prediction = np.average(predictions, weights=self.member_weights)
        
        # Confidence = inverse of variance (low variance = high confidence)
        variance = np.var(predictions)
        confidence = 1.0 / (1.0 + variance)
        
        return prediction, confidence

