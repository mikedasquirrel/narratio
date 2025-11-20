"""
Relational Feature Computation Utilities

Compute differential, ratio, and interaction features between competitors.
These functions enable RELATIONAL narrative analysis (Home vs Away, Player A vs Player B).

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any


def compute_differential(
    entity_a_features: np.ndarray,
    entity_b_features: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute simple differential features: A - B
    
    Args:
        entity_a_features: Features for entity A (e.g., home team)
        entity_b_features: Features for entity B (e.g., away team)
        feature_names: Names of the input features
        
    Returns:
        differential_features: A - B for each feature
        differential_names: Feature names with '_diff' suffix
    """
    differential = entity_a_features - entity_b_features
    diff_names = [f"{name}_diff" for name in feature_names]
    return differential, diff_names


def compute_ratios(
    entity_a_features: np.ndarray,
    entity_b_features: np.ndarray,
    feature_names: List[str],
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute ratio features: A / B (with small epsilon to avoid division by zero)
    
    Args:
        entity_a_features: Features for entity A
        entity_b_features: Features for entity B
        feature_names: Names of the input features
        epsilon: Small value to avoid division by zero
        
    Returns:
        ratio_features: A / B for each feature
        ratio_names: Feature names with '_ratio' suffix
    """
    # Replace any NaN with 0
    entity_a_features = np.nan_to_num(entity_a_features, nan=0.0, posinf=0.0, neginf=0.0)
    entity_b_features = np.nan_to_num(entity_b_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Add epsilon to denominator to avoid division by zero
    # If denominator is 0, set it to epsilon
    denominator = np.where(
        np.abs(entity_b_features) < epsilon,
        epsilon,
        entity_b_features
    )
    
    ratios = entity_a_features / denominator
    
    # Clip extreme ratios to avoid numerical issues
    ratios = np.clip(ratios, -1000, 1000)
    
    # Final NaN check
    ratios = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)
    
    ratio_names = [f"{name}_ratio" for name in feature_names]
    return ratios, ratio_names


def compute_interactions(
    entity_a_features: np.ndarray,
    entity_b_features: np.ndarray,
    feature_names: List[str],
    interaction_pairs: Optional[List[Tuple[int, int]]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute interaction features: A_i * B_j for key feature pairs
    
    If interaction_pairs is None, computes element-wise A * B.
    If interaction_pairs provided, computes specific cross-interactions.
    
    Args:
        entity_a_features: Features for entity A
        entity_b_features: Features for entity B
        feature_names: Names of the input features
        interaction_pairs: Optional list of (i, j) index pairs to interact
        
    Returns:
        interaction_features: A * B interactions
        interaction_names: Feature names describing the interaction
    """
    if interaction_pairs is None:
        # Element-wise multiplication
        interactions = entity_a_features * entity_b_features
        interaction_names = [f"{name}_interaction" for name in feature_names]
    else:
        # Specific cross-interactions
        interactions = []
        interaction_names = []
        for i, j in interaction_pairs:
            interactions.append(entity_a_features[i] * entity_b_features[j])
            interaction_names.append(f"{feature_names[i]}_x_{feature_names[j]}")
        interactions = np.array(interactions)
    
    return interactions, interaction_names


def compute_relational_features(
    entity_a_features: np.ndarray,
    entity_b_features: np.ndarray,
    feature_names: List[str],
    compute_diff: bool = True,
    compute_ratio: bool = True,
    compute_interaction: bool = True,
    interaction_pairs: Optional[List[Tuple[int, int]]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all relational features: differential, ratios, and interactions
    
    Args:
        entity_a_features: Features for entity A (shape: [n_features])
        entity_b_features: Features for entity B (shape: [n_features])
        feature_names: Names of the input features
        compute_diff: Whether to compute differentials
        compute_ratio: Whether to compute ratios
        compute_interaction: Whether to compute interactions
        interaction_pairs: Optional specific pairs for interaction
        
    Returns:
        relational_features: Concatenated relational features
        relational_names: Names of all relational features
    """
    all_features = []
    all_names = []
    
    if compute_diff:
        diff_feats, diff_names = compute_differential(
            entity_a_features, entity_b_features, feature_names
        )
        all_features.append(diff_feats)
        all_names.extend(diff_names)
    
    if compute_ratio:
        ratio_feats, ratio_names = compute_ratios(
            entity_a_features, entity_b_features, feature_names
        )
        all_features.append(ratio_feats)
        all_names.extend(ratio_names)
    
    if compute_interaction:
        interact_feats, interact_names = compute_interactions(
            entity_a_features, entity_b_features, feature_names,
            interaction_pairs
        )
        all_features.append(interact_feats)
        all_names.extend(interact_names)
    
    # Concatenate all relational features
    if all_features:
        relational_features = np.concatenate(all_features)
    else:
        relational_features = np.array([])
    
    return relational_features, all_names


def compute_narrative_intensity(
    game_data: Dict[str, Any],
    domain_structure: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute narrative intensity metrics based on domain structure
    
    Args:
        game_data: Dictionary containing game/match data
        domain_structure: Domain structure schema loaded from JSON
        
    Returns:
        intensity_metrics: Dictionary of computed intensity features
    """
    intensity_metrics = {}
    
    # Get available intensity indicators from domain schema
    intensity_indicators = domain_structure.get('intensity_indicators', [])
    
    # Compute each available intensity metric
    for indicator in intensity_indicators:
        if indicator == 'comeback_occurred':
            # Check if there was a comeback (domain-specific logic)
            intensity_metrics['comeback_flag'] = float(
                game_data.get('comeback_occurred', False)
            )
            
        elif indicator == 'comeback_deficit_points':
            intensity_metrics['comeback_deficit'] = float(
                game_data.get('largest_deficit_overcome', 0)
            )
            
        elif indicator == 'final_margin':
            intensity_metrics['final_margin'] = float(
                abs(game_data.get('final_score_diff', 0))
            )
            
        elif indicator == 'lead_changes_count':
            intensity_metrics['lead_changes'] = float(
                game_data.get('lead_changes', 0)
            )
            
        elif indicator == 'overtime':
            intensity_metrics['overtime_flag'] = float(
                game_data.get('overtime', False)
            )
            
        elif indicator == 'close_sets_count':
            intensity_metrics['close_sets'] = float(
                game_data.get('close_sets_count', 0)
            )
            
        elif indicator == 'match_duration_minutes':
            intensity_metrics['match_duration'] = float(
                game_data.get('match_duration_minutes', 0)
            )
    
    return intensity_metrics


def compute_hierarchical_context(
    game_data: Dict[str, Any],
    domain_structure: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract hierarchical position and context from domain structure
    
    Args:
        game_data: Dictionary containing game/match data
        domain_structure: Domain structure schema loaded from JSON
        
    Returns:
        context_features: Dictionary of hierarchical context features
    """
    context_features = {}
    
    # Get game context type if available
    game_context_types = domain_structure.get('game_context_types', [])
    context_type = game_data.get('context_type', 'regular')
    
    # One-hot encode context type
    for ctx_type in game_context_types:
        context_features[f'context_{ctx_type}'] = float(
            context_type == ctx_type
        )
    
    # Season position (if applicable)
    if 'season_structure' in domain_structure:
        season_struct = domain_structure['season_structure']
        if 'regular_season_games' in season_struct:
            total_games = season_struct['regular_season_games']
            game_number = game_data.get('game_number_in_season', 1)
            context_features['season_progress'] = game_number / total_games
    
    # Temporal coupling type
    temporal_coupling = domain_structure.get('temporal_coupling', 'isolated')
    context_features['sequential_coupling'] = float(
        temporal_coupling == 'sequential'
    )
    context_features['episodic_coupling'] = float(
        temporal_coupling == 'episodic'
    )
    context_features['continuous_coupling'] = float(
        temporal_coupling == 'continuous'
    )
    
    return context_features


def pair_competitors(
    data: List[Dict[str, Any]],
    domain_structure: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Pair competitor data for relational analysis
    
    For team sports: pairs home/away teams in same game
    For individual sports: pairs two competitors in same match
    
    Args:
        data: List of individual entity records
        domain_structure: Domain structure schema
        
    Returns:
        paired_data: List of paired records with both competitors
    """
    relational_type = domain_structure.get('relational_structure', '1v1')
    
    if relational_type in ['team_vs_team', '1v1']:
        # Group by game_id to pair competitors
        game_groups = {}
        for record in data:
            game_id = record.get('game_id', record.get('match_id'))
            if game_id:
                if game_id not in game_groups:
                    game_groups[game_id] = []
                game_groups[game_id].append(record)
        
        # Create paired records
        paired_data = []
        for game_id, records in game_groups.items():
            if len(records) >= 2:
                # Identify home/away or player1/player2
                home_record = None
                away_record = None
                
                for record in records:
                    if record.get('home_game', False) or record.get('is_home', False):
                        home_record = record
                    else:
                        away_record = record
                
                # If we have both sides, create paired record
                if home_record and away_record:
                    paired_record = {
                        'game_id': game_id,
                        'entity_a': home_record,
                        'entity_b': away_record,
                        'outcome': home_record.get('won', home_record.get('home_won', 0))
                    }
                    paired_data.append(paired_record)
        
        return paired_data
    
    else:
        # For vs_market or other structures, return as-is
        return data

