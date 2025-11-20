"""
Ensemble Nominative Analyzer
BREAKTHROUGH: Analyze the complete NAME ENSEMBLE, not just individuals
Theory: Collective nominative harmony predicts team/group success beyond individual effects
Expected: 5-15% additional explained variance from ensemble coherence
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class EnsembleNominativeAnalyzer:
    """
    Analyze name ensembles as collective linguistic units
    The GESTALT of nominative determinism
    """
    
    def __init__(self):
        """Initialize ensemble analyzer"""
        pass
    
    def calculate_ensemble_metrics(self, individual_scores: List[float],
                                  individual_features: List[Dict]) -> Dict:
        """
        Calculate ensemble-level metrics from individual name scores
        
        Args:
            individual_scores: List of individual linguistic scores
            individual_features: List of feature dicts for each individual
            
        Returns:
            Complete ensemble analysis
        """
        if len(individual_scores) < 3:
            return {'error': 'Need at least 3 members for ensemble analysis'}
        
        # METRIC 1: Collective Statistics
        mean_score = np.mean(individual_scores)
        median_score = np.median(individual_scores)
        std_score = np.std(individual_scores)
        min_score = np.min(individual_scores)
        max_score = np.max(individual_scores)
        
        # METRIC 2: Ensemble Coherence (low variance = harmonic)
        variance = np.var(individual_scores)
        coefficient_of_variation = (std_score / mean_score) if mean_score > 0 else 0
        
        # Normalize variance to coherence score (0-100)
        # Low variance = high coherence
        coherence_score = max(0, 100 - (variance / 4))
        
        if coherence_score > 80:
            coherence_class = 'HIGH_COHERENCE'
            coherence_bonus = 1.25
        elif coherence_score > 65:
            coherence_class = 'MODERATE_COHERENCE'
            coherence_bonus = 1.12
        elif coherence_score > 50:
            coherence_class = 'LOW_COHERENCE'
            coherence_bonus = 1.0
        else:
            coherence_class = 'DISCORDANT'
            coherence_bonus = 0.92
        
        # METRIC 3: Star Dominance Effect
        # Do top performers elevate or isolate from team?
        top3_scores = sorted(individual_scores, reverse=True)[:3]
        top3_mean = np.mean(top3_scores)
        
        star_differential = top3_mean - mean_score
        
        if star_differential > 20:
            # Superstars far above team avg
            star_effect = 'DOMINANT_STARS'
            star_multiplier = 1.35  # Stars can carry team
        elif star_differential > 10:
            star_effect = 'STRONG_STARS'
            star_multiplier = 1.20
        elif star_differential > 5:
            star_effect = 'BALANCED_STARS'
            star_multiplier = 1.08
        else:
            star_effect = 'ENSEMBLE_BALANCED'
            star_multiplier = 1.15  # No single star, collective strength
        
        # METRIC 4: Phonetic Harmony
        # Do names share similar phonetic patterns?
        if individual_features:
            harshness_values = [f.get('harshness', 50) for f in individual_features]
            memorability_values = [f.get('memorability', 50) for f in individual_features]
            
            harsh_variance = np.var(harshness_values)
            mem_variance = np.var(memorability_values)
            
            # Low variance in features = phonetically harmonic
            phonetic_harmony = 100 - ((harsh_variance + mem_variance) / 8)
            phonetic_harmony = max(0, min(100, phonetic_harmony))
            
            if phonetic_harmony > 75:
                harmony_multiplier = 1.20
            elif phonetic_harmony > 60:
                harmony_multiplier = 1.10
            else:
                harmony_multiplier = 1.0
        else:
            phonetic_harmony = 50
            harmony_multiplier = 1.0
        
        # METRIC 5: Ensemble Range (diversity)
        score_range = max_score - min_score
        
        if score_range > 40:
            diversity = 'HIGH_DIVERSITY'
            # High diversity can be good (specialists) or bad (weak links)
            diversity_effect = 0.95
        elif score_range > 25:
            diversity = 'MODERATE_DIVERSITY'
            diversity_effect = 1.0
        else:
            diversity = 'HOMOGENEOUS'
            diversity_effect = 1.08  # Consistency bonus
        
        # CALCULATE ENSEMBLE SCORE
        # Not just mean - includes coherence, stars, harmony
        ensemble_score = (
            mean_score *
            coherence_bonus *
            star_multiplier *
            harmony_multiplier *
            diversity_effect
        )
        
        ensemble_score = min(ensemble_score, 100)
        
        return {
            'ensemble_size': len(individual_scores),
            'collective_stats': {
                'mean': round(mean_score, 2),
                'median': round(median_score, 2),
                'std': round(std_score, 2),
                'min': round(min_score, 2),
                'max': round(max_score, 2),
                'range': round(score_range, 2)
            },
            'coherence': {
                'variance': round(variance, 2),
                'coefficient_of_variation': round(coefficient_of_variation, 3),
                'coherence_score': round(coherence_score, 2),
                'classification': coherence_class,
                'bonus_multiplier': coherence_bonus
            },
            'star_effect': {
                'top3_mean': round(top3_mean, 2),
                'differential': round(star_differential, 2),
                'effect_type': star_effect,
                'multiplier': star_multiplier
            },
            'phonetic_harmony': {
                'harmony_score': round(phonetic_harmony, 2),
                'multiplier': harmony_multiplier
            },
            'diversity': {
                'score_range': round(score_range, 2),
                'classification': diversity,
                'effect': diversity_effect
            },
            'ensemble_score': round(ensemble_score, 2),
            'improvement_over_mean': round((ensemble_score / mean_score - 1) * 100, 1)
        }
    
    def compare_ensembles(self, ensemble1: Dict, ensemble2: Dict) -> Dict:
        """
        Compare two ensembles (e.g., Team A vs Team B, Horse entry 1 vs 2)
        
        Args:
            ensemble1: First ensemble analysis
            ensemble2: Second ensemble analysis
            
        Returns:
            Comparative analysis
        """
        score1 = ensemble1['ensemble_score']
        score2 = ensemble2['ensemble_score']
        
        differential = score1 - score2
        
        # Compare coherence
        coherence1 = ensemble1['coherence']['coherence_score']
        coherence2 = ensemble2['coherence']['coherence_score']
        coherence_advantage = coherence1 - coherence2
        
        # Compare star power
        star1 = ensemble1['star_effect']['top3_mean']
        star2 = ensemble2['star_effect']['top3_mean']
        star_advantage = star1 - star2
        
        # Predict winner
        if differential > 15:
            prediction = 'ENSEMBLE_1_STRONG'
            confidence = min(85, 65 + differential)
        elif differential > 8:
            prediction = 'ENSEMBLE_1_MODERATE'
            confidence = 70
        elif differential < -15:
            prediction = 'ENSEMBLE_2_STRONG'
            confidence = min(85, 65 + abs(differential))
        elif differential < -8:
            prediction = 'ENSEMBLE_2_MODERATE'
            confidence = 70
        else:
            prediction = 'TOSS_UP'
            confidence = 55
        
        return {
            'ensemble1_score': score1,
            'ensemble2_score': score2,
            'differential': round(differential, 2),
            'coherence_advantage': round(coherence_advantage, 2),
            'star_advantage': round(star_advantage, 2),
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': self._generate_comparison_reasoning(
                differential, coherence_advantage, star_advantage
            )
        }
    
    def _generate_comparison_reasoning(self, diff: float, coh: float, star: float) -> str:
        """Generate reasoning for ensemble comparison"""
        factors = []
        
        if abs(diff) > 15:
            factors.append(f"{'Ensemble 1' if diff > 0 else 'Ensemble 2'} has {abs(diff):.0f}-point overall advantage")
        
        if abs(coh) > 10:
            factors.append(f"{'Ensemble 1' if coh > 0 else 'Ensemble 2'} has {abs(coh):.0f}-point coherence advantage")
        
        if abs(star) > 12:
            factors.append(f"{'Ensemble 1' if star > 0 else 'Ensemble 2'} has superior star power ({abs(star):.0f} points)")
        
        if not factors:
            return "Ensembles are evenly matched"
        
        return "; ".join(factors)
    
    def calculate_ensemble_synergy(self, individual_features: List[Dict]) -> Dict:
        """
        Calculate how well ensemble members' names work together
        Do they create synergy or discord?
        
        Args:
            individual_features: List of linguistic feature dicts
            
        Returns:
            Synergy analysis
        """
        if len(individual_features) < 2:
            return {'synergy_score': 50}
        
        # Extract feature arrays
        harshness = [f.get('harshness', 50) for f in individual_features]
        syllables = [f.get('syllables', 2.5) for f in individual_features]
        memorability = [f.get('memorability', 50) for f in individual_features]
        
        # SYNERGY 1: Phonetic Clustering
        # Do ensemble members have similar phonetic patterns?
        harsh_cv = np.std(harshness) / np.mean(harshness) if np.mean(harshness) > 0 else 0
        
        if harsh_cv < 0.15:  # Low variation = harmonic
            phonetic_synergy = 1.25
        elif harsh_cv < 0.25:
            phonetic_synergy = 1.10
        else:
            phonetic_synergy = 1.0
        
        # SYNERGY 2: Complementary Diversity
        # Mix of harsh and memorable can be strategic
        harsh_range = max(harshness) - min(harshness)
        
        if 20 < harsh_range < 35:  # Goldilocks diversity
            diversity_synergy = 1.15  # Good mix of power and finesse
        elif harsh_range < 20:
            diversity_synergy = 1.08  # Very similar (consistent)
        else:
            diversity_synergy = 0.95  # Too diverse (identity crisis)
        
        # SYNERGY 3: Syllabic Rhythm
        # Do names flow together or clash?
        syll_variance = np.var(syllables)
        
        if syll_variance < 0.5:  # Similar length = rhythmic
            rhythm_synergy = 1.12
        else:
            rhythm_synergy = 1.0
        
        # Total synergy
        total_synergy = phonetic_synergy * diversity_synergy * rhythm_synergy
        synergy_score = 50 + ((total_synergy - 1.0) * 50)
        
        return {
            'phonetic_clustering': {
                'harsh_cv': round(harsh_cv, 3),
                'synergy': phonetic_synergy
            },
            'complementary_diversity': {
                'harsh_range': round(harsh_range, 2),
                'synergy': diversity_synergy
            },
            'syllabic_rhythm': {
                'syll_variance': round(syll_variance, 3),
                'synergy': rhythm_synergy
            },
            'total_synergy_multiplier': round(total_synergy, 3),
            'synergy_score': round(synergy_score, 2),
            'verdict': 'SYNERGISTIC' if total_synergy > 1.15 else 'HARMONIC' if total_synergy > 1.05 else 'NEUTRAL'
        }


if __name__ == "__main__":
    # Test ensemble analyzer
    logging.basicConfig(level=logging.INFO)
    
    analyzer = EnsembleNominativeAnalyzer()
    
    print("="*80)
    print("ENSEMBLE NOMINATIVE ANALYSIS")
    print("="*80)
    
    # Test 1: High coherence ensemble (Argentina-style)
    print("\n1. HIGH COHERENCE ENSEMBLE (Argentina-style)")
    print("-" * 80)
    
    argentina_scores = [92, 88, 85, 82, 78, 75, 72, 70, 68, 65, 64]
    argentina_features = [
        {'harshness': 75, 'syllables': 2.5, 'memorability': 92},
        {'harshness': 72, 'syllables': 3.0, 'memorability': 88},
        {'harshness': 70, 'syllables': 2.8, 'memorability': 85},
        {'harshness': 73, 'syllables': 2.5, 'memorability': 82},
        {'harshness': 71, 'syllables': 2.7, 'memorability': 78},
        {'harshness': 69, 'syllables': 2.6, 'memorability': 75},
        {'harshness': 72, 'syllables': 2.4, 'memorability': 72},
        {'harshness': 68, 'syllables': 2.8, 'memorability': 70},
        {'harshness': 70, 'syllables': 2.5, 'memorability': 68},
        {'harshness': 67, 'syllables': 2.6, 'memorability': 65},
        {'harshness': 66, 'syllables': 2.7, 'memorability': 64}
    ]
    
    ensemble1 = analyzer.calculate_ensemble_metrics(argentina_scores, argentina_features)
    synergy1 = analyzer.calculate_ensemble_synergy(argentina_features)
    
    print(f"Mean Score: {ensemble1['collective_stats']['mean']}")
    print(f"Coherence: {ensemble1['coherence']['classification']}")
    print(f"Coherence Bonus: {ensemble1['coherence']['bonus_multiplier']}×")
    print(f"Star Effect: {ensemble1['star_effect']['effect_type']}")
    print(f"Star Multiplier: {ensemble1['star_effect']['multiplier']}×")
    print(f"Ensemble Score: {ensemble1['ensemble_score']}")
    print(f"Improvement over mean: +{ensemble1['improvement_over_mean']}%")
    print(f"\nSynergy: {synergy1['verdict']}")
    print(f"Total Synergy: {synergy1['total_synergy_multiplier']}×")
    
    # Test 2: Low coherence (one superstar, weak team)
    print("\n2. LOW COHERENCE ENSEMBLE (Superstar + Weak Team)")
    print("-" * 80)
    
    weak_scores = [95, 48, 45, 42, 40, 38, 36, 35, 33, 32, 30]
    weak_features = [
        {'harshness': 85, 'syllables': 2.0, 'memorability': 95},
        {'harshness': 45, 'syllables': 3.5, 'memorability': 48},
        {'harshness': 42, 'syllables': 3.8, 'memorability': 45},
        {'harshness': 40, 'syllables': 3.6, 'memorability': 42},
        {'harshness': 38, 'syllables': 3.9, 'memorability': 40},
        {'harshness': 43, 'syllables': 3.7, 'memorability': 38},
        {'harshness': 41, 'syllables': 3.5, 'memorability': 36},
        {'harshness': 39, 'syllables': 3.8, 'memorability': 35},
        {'harshness': 37, 'syllables': 3.6, 'memorability': 33},
        {'harshness': 36, 'syllables': 3.9, 'memorability': 32},
        {'harshness': 35, 'syllables': 4.0, 'memorability': 30}
    ]
    
    ensemble2 = analyzer.calculate_ensemble_metrics(weak_scores, weak_features)
    synergy2 = analyzer.calculate_ensemble_synergy(weak_features)
    
    print(f"Mean Score: {ensemble2['collective_stats']['mean']}")
    print(f"Coherence: {ensemble2['coherence']['classification']}")
    print(f"Coherence Bonus: {ensemble2['coherence']['bonus_multiplier']}×")
    print(f"Star Effect: {ensemble2['star_effect']['effect_type']}")
    print(f"Ensemble Score: {ensemble2['ensemble_score']}")
    print(f"Improvement over mean: {ensemble2['improvement_over_mean']:+.1f}%")
    
    # Test 3: Compare ensembles
    print("\n3. ENSEMBLE COMPARISON")
    print("-" * 80)
    
    comparison = analyzer.compare_ensembles(ensemble1, ensemble2)
    
    print(f"Ensemble 1: {comparison['ensemble1_score']}")
    print(f"Ensemble 2: {comparison['ensemble2_score']}")
    print(f"Differential: {comparison['differential']:+.1f}")
    print(f"Prediction: {comparison['prediction']}")
    print(f"Confidence: {comparison['confidence']}%")
    print(f"Reasoning: {comparison['reasoning']}")
    
    print("\n" + "="*80)
    print("✅ ENSEMBLE ANALYSIS OPERATIONAL")
    print("="*80)

