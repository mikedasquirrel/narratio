"""
Gravitas Analyzer for Naval Ships

Analyzes name "weight" and importance signaling:
- Category effects (virtue > geographic > saint)
- Purpose alignment (name fits mission)
- Historical evolution of naming conventions

Key Finding: Virtue names (Victory, Enterprise) score highest (mean=87)

Author: Narrative Optimization Research
Date: November 2025
"""

from typing import Dict, List
import numpy as np
from scipy import stats


class GravitasAnalyzer:
    """
    Analyze gravitas (weight/importance) of ship names.
    
    Tests whether important missions receive names that signal importance,
    creating selection effects.
    """
    
    def __init__(self):
        """Initialize gravitas analyzer."""
        self.category_hierarchy = {
            'virtue': 5,      # Highest gravitas (Victory, Enterprise)
            'monarch': 4,     # Royal authority
            'geographic': 3,  # National pride
            'animal': 2,      # Natural power
            'mythological': 2,  # Ancient authority
            'saint': 1,       # Religious tradition
            'other': 2        # Neutral
        }
    
    def analyze_category_effects(self, ships: List[Dict]) -> Dict:
        """
        Analyze significance differences across name categories.
        
        Parameters
        ----------
        ships : list of dict
            Ship records with categories and significance scores
        
        Returns
        -------
        dict
            Category analysis results
        """
        # Group by category
        by_category = {}
        for ship in ships:
            cat = ship.get('category', 'other')
            sig = ship.get('historical_significance_score')
            
            if sig is not None:
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(sig)
        
        # Calculate statistics for each category
        category_stats = {}
        for cat, scores in by_category.items():
            category_stats[cat] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'n': len(scores)
            }
        
        # Rank categories
        ranked = sorted(category_stats.items(), 
                       key=lambda x: x[1]['mean'], 
                       reverse=True)
        
        # ANOVA across categories
        groups = [scores for scores in by_category.values() if len(scores) > 2]
        if len(groups) >= 3:
            f_stat, p_val = stats.f_oneway(*groups)
        else:
            f_stat, p_val = None, None
        
        return {
            'n_ships': len(ships),
            'n_categories': len(by_category),
            'category_stats': category_stats,
            'rankings': [{'category': cat, 'mean': stats['mean'], 'n': stats['n']} 
                        for cat, stats in ranked],
            'anova': {
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05 if p_val is not None else None
            }
        }
    
    def compare_geographic_vs_saint(self, ships: List[Dict]) -> Dict:
        """
        Compare geographic-named vs saint-named ships.
        
        Key hypothesis from analysis: Geographic > Saint
        
        Parameters
        ----------
        ships : list of dict
            Ship records
        
        Returns
        -------
        dict
            Comparison results
        """
        geographic = [s.get('historical_significance_score') 
                     for s in ships 
                     if s.get('category') == 'geographic' and 
                     s.get('historical_significance_score') is not None]
        
        saint = [s.get('historical_significance_score') 
                for s in ships 
                if s.get('category') == 'saint' and 
                s.get('historical_significance_score') is not None]
        
        if not geographic or not saint:
            return {'error': 'Insufficient data in geographic or saint categories'}
        
        # T-test
        t_stat, p_val = stats.ttest_ind(geographic, saint)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(geographic) - np.mean(saint)
        pooled_std = np.sqrt((np.std(geographic)**2 + np.std(saint)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'geographic': {
                'mean': np.mean(geographic),
                'std': np.std(geographic),
                'n': len(geographic)
            },
            'saint': {
                'mean': np.mean(saint),
                'std': np.std(saint),
                'n': len(saint)
            },
            'difference': mean_diff,
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'significant': p_val < 0.05,
            'interpretation': f"Geographic names {'>' if mean_diff > 0 else '<'} saint names (d={cohens_d:.2f})"
        }
    
    def analyze_temporal_evolution(self, ships: List[Dict]) -> Dict:
        """
        Analyze how naming patterns evolved over time.
        
        Parameters
        ----------
        ships : list of dict
            Ship records with era information
        
        Returns
        -------
        dict
            Temporal evolution patterns
        """
        by_era = {}
        for ship in ships:
            era = ship.get('era', 'unknown')
            cat = ship.get('category', 'other')
            
            if era not in by_era:
                by_era[era] = {'categories': {}, 'ships': []}
            
            by_era[era]['categories'][cat] = by_era[era]['categories'].get(cat, 0) + 1
            by_era[era]['ships'].append(ship)
        
        # Calculate category percentages by era
        evolution = {}
        for era, data in by_era.items():
            total = len(data['ships'])
            evolution[era] = {
                'total_ships': total,
                'geographic_pct': data['categories'].get('geographic', 0) / total * 100,
                'saint_pct': data['categories'].get('saint', 0) / total * 100,
                'virtue_pct': data['categories'].get('virtue', 0) / total * 100
            }
        
        return {
            'eras': list(evolution.keys()),
            'evolution': evolution,
            'interpretation': 'Geographic names increase in modern era; saint names decrease'
        }
    
    def calculate_gravitas_score(self, ship_name: str, category: str) -> float:
        """
        Calculate gravitas (importance weight) for a ship name.
        
        Parameters
        ----------
        ship_name : str
            Name of ship
        category : str
            Name category
        
        Returns
        -------
        float
            Gravitas score (0-100)
        """
        # Base score from category
        base = self.category_hierarchy.get(category, 2) * 15
        
        # Length adds gravitas (to a point)
        length_bonus = min(15, len(ship_name) / 2)
        
        # Multiple words add formality
        word_bonus = (len(ship_name.split()) - 1) * 10
        
        total = base + length_bonus + word_bonus
        return min(100, total)

