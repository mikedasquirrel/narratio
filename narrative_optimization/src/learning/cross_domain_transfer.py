"""
Cross-Domain Transfer Learning

Uses imperative gravity to transfer patterns from structurally
similar domains to improve predictions.

Process:
1. For each instance, identify top-N imperative neighbors (other domains)
2. Query similar instances from neighbor domains
3. Extract transferable patterns
4. Weight by imperative gravity force
5. Ensemble with domain-specific predictions

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.story_instance import StoryInstance
from data.instance_repository import InstanceRepository
from physics.imperative_gravity import ImperativeGravityCalculator
from config.domain_config import DomainConfig


class CrossDomainTransferLearner:
    """
    Transfer learning across structurally similar domains.
    
    Uses imperative gravity to identify which domains to learn from
    and how much to weight their contributions.
    """
    
    def __init__(
        self,
        repository: InstanceRepository,
        domain_configs: Dict[str, DomainConfig],
        imperative_calculator: ImperativeGravityCalculator
    ):
        """
        Initialize transfer learner.
        
        Parameters
        ----------
        repository : InstanceRepository
            Instance repository
        domain_configs : dict
            All domain configurations
        imperative_calculator : ImperativeGravityCalculator
            Calculator for finding neighbors
        """
        self.repository = repository
        self.domain_configs = domain_configs
        self.imperative_calculator = imperative_calculator
        
        # Transfer models by domain pair
        self.transfer_models: Dict[Tuple[str, str], Any] = {}
        
        # Transfer effectiveness tracking
        self.transfer_results: Dict[str, Dict] = {}
    
    def predict_with_transfer(
        self,
        instance: StoryInstance,
        domain_model_prediction: float,
        n_neighbors: int = 5,
        ensemble_weight: float = 0.4
    ) -> Dict[str, Any]:
        """
        Make prediction using domain model + cross-domain transfer.
        
        Parameters
        ----------
        instance : StoryInstance
            Instance to predict for
        domain_model_prediction : float
            Prediction from domain-specific model
        n_neighbors : int
            Number of neighbor domains to use
        ensemble_weight : float
            Weight for cross-domain component (0-1)
            Final = (1-w)*domain + w*cross_domain
        
        Returns
        -------
        dict
            {
                'prediction': float,
                'domain_component': float,
                'cross_domain_component': float,
                'neighbors_used': list,
                'weights': dict
            }
        """
        # Find imperative neighbors
        all_domains = list(self.domain_configs.keys())
        neighbors = self.imperative_calculator.find_gravitational_neighbors(
            instance,
            all_domains,
            n_neighbors=n_neighbors,
            exclude_same_domain=True
        )
        
        if not neighbors:
            # No neighbors found, use domain model only
            return {
                'prediction': domain_model_prediction,
                'domain_component': domain_model_prediction,
                'cross_domain_component': domain_model_prediction,
                'neighbors_used': [],
                'weights': {}
            }
        
        # Get similar instances from neighbor domains
        cross_domain_predictions = []
        neighbor_weights = []
        neighbors_used = []
        
        for neighbor_domain, force in neighbors:
            # Find similar instances in neighbor domain
            similar_instances = self._find_similar_instances(
                instance,
                neighbor_domain
            )
            
            if similar_instances:
                # Calculate prediction from this neighbor domain
                neighbor_prediction = self._predict_from_neighbors(
                    instance,
                    similar_instances
                )
                
                cross_domain_predictions.append(neighbor_prediction)
                neighbor_weights.append(force)
                neighbors_used.append({
                    'domain': neighbor_domain,
                    'force': float(force),
                    'n_similar': len(similar_instances),
                    'contribution': float(neighbor_prediction)
                })
        
        # Ensemble cross-domain predictions
        if cross_domain_predictions:
            # Weight by force magnitude
            weights_array = np.array(neighbor_weights)
            weights_normalized = weights_array / weights_array.sum()
            
            cross_domain_prediction = np.average(
                cross_domain_predictions,
                weights=weights_normalized
            )
        else:
            # No valid neighbors, use domain model
            cross_domain_prediction = domain_model_prediction
        
        # Final ensemble
        final_prediction = (
            (1 - ensemble_weight) * domain_model_prediction +
            ensemble_weight * cross_domain_prediction
        )
        
        return {
            'prediction': float(final_prediction),
            'domain_component': float(domain_model_prediction),
            'cross_domain_component': float(cross_domain_prediction),
            'neighbors_used': neighbors_used,
            'ensemble_weight': ensemble_weight,
            'n_neighbors': len(neighbors_used)
        }
    
    def _find_similar_instances(
        self,
        query_instance: StoryInstance,
        target_domain: str,
        n_similar: int = 10
    ) -> List[StoryInstance]:
        """
        Find similar instances in target domain.
        
        Parameters
        ----------
        query_instance : StoryInstance
            Query instance
        target_domain : str
            Domain to search in
        n_similar : int
            Number of similar instances
        
        Returns
        -------
        list of StoryInstance
            Similar instances from target domain
        """
        # Get all instances from target domain
        target_instances = self.repository.get_instances_by_domain(target_domain)
        
        if not target_instances:
            return []
        
        # Query by structural similarity
        pi_range = (query_instance.pi_effective - 0.15, query_instance.pi_effective + 0.15)
        
        similar = self.repository.query_by_structure(
            pi_range=pi_range,
            exclude_domain=query_instance.domain
        )
        
        # Filter to target domain
        similar = [s for s in similar if s.domain == target_domain]
        
        # If we have genomes, can do more precise similarity
        if query_instance.genome_full is not None and len(similar) > 0:
            # Calculate cosine similarities
            similarities = []
            for candidate in similar:
                if candidate.genome_full is not None:
                    sim = self._cosine_similarity(
                        query_instance.genome_full,
                        candidate.genome_full
                    )
                    similarities.append((candidate, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar = [inst for inst, sim in similarities[:n_similar]]
        
        return similar[:n_similar]
    
    def _predict_from_neighbors(
        self,
        query_instance: StoryInstance,
        neighbor_instances: List[StoryInstance]
    ) -> float:
        """
        Make prediction based on neighbor instances.
        
        Uses K-nearest neighbors approach with story quality weighting.
        
        Parameters
        ----------
        query_instance : StoryInstance
            Query instance
        neighbor_instances : list of StoryInstance
            Similar instances from neighbor domain
        
        Returns
        -------
        float
            Predicted outcome
        """
        # Get neighbor outcomes and qualities
        outcomes = []
        qualities = []
        
        for neighbor in neighbor_instances:
            if neighbor.outcome is not None:
                outcomes.append(neighbor.outcome)
                quality = neighbor.story_quality if neighbor.story_quality else 0.5
                qualities.append(quality)
        
        if not outcomes:
            return 0.5  # Default
        
        # Weight by story quality (better stories more informative)
        qualities = np.array(qualities)
        outcomes = np.array(outcomes)
        
        if qualities.sum() > 0:
            weights = qualities / qualities.sum()
            prediction = np.average(outcomes, weights=weights)
        else:
            prediction = np.mean(outcomes)
        
        return float(prediction)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        # Ensure same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        # Calculate
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Scale to [0, 1]
        return (similarity + 1) / 2
    
    def evaluate_transfer_effectiveness(
        self,
        test_instances: List[StoryInstance],
        domain_predictions: np.ndarray,
        true_outcomes: np.ndarray,
        n_neighbors: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate how much transfer learning improves predictions.
        
        Parameters
        ----------
        test_instances : list of StoryInstance
            Test instances
        domain_predictions : ndarray
            Predictions from domain model only
        true_outcomes : ndarray
            Actual outcomes
        n_neighbors : int
            Number of neighbors for transfer
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        # Make predictions with transfer
        transfer_predictions = []
        
        for instance, domain_pred in zip(test_instances, domain_predictions):
            result = self.predict_with_transfer(
                instance,
                domain_pred,
                n_neighbors=n_neighbors
            )
            transfer_predictions.append(result['prediction'])
        
        transfer_predictions = np.array(transfer_predictions)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Domain-only metrics
        domain_mse = mean_squared_error(true_outcomes, domain_predictions)
        domain_mae = mean_absolute_error(true_outcomes, domain_predictions)
        
        # With transfer metrics
        transfer_mse = mean_squared_error(true_outcomes, transfer_predictions)
        transfer_mae = mean_absolute_error(true_outcomes, transfer_predictions)
        
        # Improvement
        mse_improvement = (domain_mse - transfer_mse) / domain_mse * 100
        mae_improvement = (domain_mae - transfer_mae) / domain_mae * 100
        
        return {
            'domain_only': {
                'mse': float(domain_mse),
                'mae': float(domain_mae)
            },
            'with_transfer': {
                'mse': float(transfer_mse),
                'mae': float(transfer_mae)
            },
            'improvement': {
                'mse_percent': float(mse_improvement),
                'mae_percent': float(mae_improvement)
            },
            'n_test': len(test_instances),
            'n_neighbors': n_neighbors
        }

