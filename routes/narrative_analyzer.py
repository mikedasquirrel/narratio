"""
Universal Narrative Analysis Engine
Route for analyzing ANY narrative and providing comprehensive insights.

This is the flagship feature - showcasing all 55 transformers across all domains.
"""

from flask import Blueprint, render_template, request, jsonify, session
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from typing import Dict, List, Any, Tuple
import traceback

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization' / 'src'))

from src.transformers.universal_hybrid import UniversalHybridTransformer
from src.transformers.transformer_selector import TransformerSelector
import pickle

narrative_analyzer_bp = Blueprint('narrative_analyzer', __name__)


class UniversalNarrativeAnalyzer:
    """
    Analyzes ANY narrative using the complete transformer framework.
    """
    
    def __init__(self):
        self.transformer = UniversalHybridTransformer(
            extract_text_features=True,
            extract_numeric_features=True,
            extract_categorical_features=True,
            use_advanced_nlp=True,
            max_text_features=100,
            max_numeric_features=50
        )
        self.selector = TransformerSelector()
        
        # Domain formulas for prediction (from DOMAIN_STATUS.md)
        self.domain_formulas = {
            'startups': {'pi': 0.76, 'delta': 0.223, 'r': 0.980, 'kappa': 0.29},
            'movies': {'pi': 0.65, 'delta': 0.026, 'r': 0.04, 'kappa': 0.65},
            'nba': {'pi': 0.49, 'delta': 0.034, 'r': 0.055, 'kappa': 0.62},
            'nfl': {'pi': 0.57, 'delta': 0.034, 'r': -0.016, 'kappa': 0.57},
            'character': {'pi': 0.85, 'delta': 0.617, 'r': 0.73, 'kappa': 0.85},
            'self_rated': {'pi': 0.95, 'delta': 0.564, 'r': 0.59, 'kappa': 0.95},
            'oscars': {'pi': 0.75, 'delta': 0.68, 'r': 0.68, 'kappa': 0.91},
            'golf': {'pi': 0.70, 'delta': 0.012, 'r': 0.017, 'kappa': 0.70},
            'poker': {'pi': 0.60, 'delta': 0.05, 'r': 0.08, 'kappa': 0.625},
            'wwe': {'pi': 0.88, 'delta': 0.75, 'r': 0.85, 'kappa': 0.88},
        }
        
        # Narrative patterns catalog
        self.patterns = {
            'hero_journey': {
                'name': 'Hero\'s Journey',
                'markers': ['hero', 'journey', 'quest', 'challenge', 'transformation', 'return'],
                'strength_threshold': 0.6
            },
            'underdog': {
                'name': 'Underdog Story',
                'markers': ['underdog', 'overcome', 'unlikely', 'against odds', 'disadvantage', 'upset'],
                'strength_threshold': 0.5
            },
            'rivalry': {
                'name': 'Rivalry',
                'markers': ['rival', 'vs', 'compete', 'battle', 'clash', 'face off'],
                'strength_threshold': 0.5
            },
            'redemption': {
                'name': 'Redemption Arc',
                'markers': ['redemption', 'comeback', 'second chance', 'return', 'rebuild', 'recover'],
                'strength_threshold': 0.5
            },
            'origin_story': {
                'name': 'Origin Story',
                'markers': ['origin', 'begin', 'start', 'found', 'early', 'first'],
                'strength_threshold': 0.5
            },
            'dynasty': {
                'name': 'Dynasty/Legacy',
                'markers': ['dynasty', 'legacy', 'champion', 'dominance', 'reign', 'history'],
                'strength_threshold': 0.6
            },
            'momentum': {
                'name': 'Momentum',
                'markers': ['momentum', 'streak', 'hot', 'rolling', 'surge', 'unstoppable'],
                'strength_threshold': 0.4
            },
            'conflict': {
                'name': 'Conflict/Tension',
                'markers': ['conflict', 'tension', 'drama', 'struggle', 'fight', 'challenge'],
                'strength_threshold': 0.5
            }
        }
    
    def analyze(self, text: str, context: Dict = None) -> Dict[str, Any]:
        """
        Analyze narrative and return comprehensive insights.
        
        Parameters
        ----------
        text : str
            The narrative to analyze
        context : dict, optional
            Additional context (domain hint, numeric data, etc.)
        
        Returns
        -------
        results : dict
            Complete analysis results
        """
        try:
            # Prepare input
            if context:
                input_data = {'text': text, **context}
            else:
                input_data = text
            
            # Fit and transform
            self.transformer.fit([input_data])
            features = self.transformer.transform([input_data])[0]
            
            # Calculate narrative quality score (ю)
            narrative_quality = self._calculate_narrative_quality(features, text)
            
            # Detect patterns
            patterns = self._detect_patterns(text)
            
            # Cross-domain predictions
            domain_predictions = self._predict_cross_domain(features, narrative_quality)
            
            # Feature breakdown
            feature_breakdown = self._analyze_features(features, text)
            
            # Recommendations
            recommendations = self._generate_recommendations(narrative_quality, patterns, feature_breakdown)
            
            # Archetype matching
            archetypes = self._match_archetypes(patterns, text)
            
            return {
                'narrative_quality': narrative_quality,
                'patterns': patterns,
                'domain_predictions': domain_predictions,
                'feature_breakdown': feature_breakdown,
                'recommendations': recommendations,
                'archetypes': archetypes,
                'text_stats': self._get_text_stats(text),
                'success': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _calculate_narrative_quality(self, features: np.ndarray, text: str) -> Dict[str, float]:
        """
        Calculate narrative quality score (ю).
        
        Uses weighted combination of features to produce 0-1 score.
        """
        # Text complexity features
        words = text.split()
        unique_ratio = len(set(words)) / (len(words) + 1)
        
        # Feature statistics
        feature_mean = float(np.mean(features))
        feature_std = float(np.std(features))
        feature_max = float(np.max(features))
        
        # Composite quality score (normalized)
        quality_raw = (
            feature_mean * 0.3 +
            feature_std * 0.2 +
            feature_max * 0.2 +
            unique_ratio * 0.3
        )
        
        # Normalize to 0-1
        quality = np.clip(quality_raw, 0, 1)
        
        # Component scores
        components = {
            'overall': float(quality),
            'complexity': float(unique_ratio),
            'richness': float(feature_mean),
            'variation': float(feature_std),
            'peak_strength': float(feature_max),
            'percentile': self._quality_to_percentile(quality)
        }
        
        return components
    
    def _quality_to_percentile(self, quality: float) -> int:
        """Convert quality score to percentile."""
        # Map 0-1 quality to 0-100 percentile
        # Using sigmoid-like distribution (most around 50th)
        percentile = int(100 * (1 / (1 + np.exp(-10 * (quality - 0.5)))))
        return np.clip(percentile, 1, 99)
    
    def _detect_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect universal narrative patterns in text.
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        
        detected = []
        
        for pattern_id, pattern_info in self.patterns.items():
            # Count marker matches
            marker_matches = sum(1 for marker in pattern_info['markers'] if marker in text_lower)
            
            # Calculate strength
            strength = marker_matches / len(pattern_info['markers'])
            
            if strength >= pattern_info['strength_threshold']:
                detected.append({
                    'id': pattern_id,
                    'name': pattern_info['name'],
                    'strength': float(strength),
                    'confidence': float(np.clip(strength * 1.2, 0, 1)),
                    'markers_found': marker_matches,
                    'markers_total': len(pattern_info['markers'])
                })
        
        # Sort by strength
        detected.sort(key=lambda x: x['strength'], reverse=True)
        
        return detected
    
    def _predict_cross_domain(self, features: np.ndarray, quality: Dict) -> List[Dict[str, Any]]:
        """
        Predict success across different domains using domain formulas.
        """
        predictions = []
        
        quality_score = quality['overall']
        
        for domain, formula in self.domain_formulas.items():
            # Calculate domain-specific prediction
            # Uses narrativity (π), correlation (r), and quality (ю)
            pi = formula['pi']
            r = formula['r']
            delta = formula['delta']
            
            # Prediction = π × quality × (1 + r)
            # This approximates "does narrative matter here?"
            base_prediction = pi * quality_score * (1 + abs(r))
            
            # Normalize to probability
            probability = np.clip(base_prediction, 0.1, 0.95)
            
            # Percentile based on probability
            percentile = int(probability * 100)
            
            # Success likelihood
            if delta > 0.5:
                success_verdict = "Very Likely" if probability > 0.7 else "Likely" if probability > 0.5 else "Possible"
            elif delta > 0.1:
                success_verdict = "Possible" if probability > 0.6 else "Uncertain"
            else:
                success_verdict = "Uncertain - Domain Constrained"
            
            predictions.append({
                'domain': domain,
                'domain_name': self._get_domain_display_name(domain),
                'probability': float(probability),
                'percentile': percentile,
                'success_verdict': success_verdict,
                'narrativity': float(pi),
                'narrative_matters': delta > 0.5,
                'explanation': self._get_domain_explanation(domain, probability, delta)
            })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return predictions
    
    def _get_domain_display_name(self, domain: str) -> str:
        """Get human-readable domain name."""
        names = {
            'startups': 'Startup Funding',
            'movies': 'Movie Success',
            'nba': 'NBA Game Outcome',
            'nfl': 'NFL Game Outcome',
            'character': 'Character Perception',
            'self_rated': 'Self-Perception',
            'oscars': 'Oscar Win',
            'golf': 'Golf Tournament',
            'poker': 'Poker Tournament',
            'wwe': 'WWE Storyline'
        }
        return names.get(domain, domain.title())
    
    def _get_domain_explanation(self, domain: str, probability: float, delta: float) -> str:
        """Generate explanation for domain prediction."""
        if delta > 0.5:
            return f"Narrative matters significantly in this domain (Δ={delta:.2f}). Your story quality directly influences success probability."
        elif delta > 0.1:
            return f"Narrative has moderate influence (Δ={delta:.2f}). Story quality matters but other factors dominate."
        else:
            return f"Domain is constrained (Δ={delta:.3f}). Narrative creates perception gaps but doesn't control outcomes."
    
    def _analyze_features(self, features: np.ndarray, text: str) -> Dict[str, Any]:
        """
        Break down feature contributions.
        """
        # Top features by magnitude
        top_indices = np.argsort(np.abs(features))[-10:][::-1]
        
        top_features = []
        for idx in top_indices:
            value = float(features[idx])
            top_features.append({
                'index': int(idx),
                'value': value,
                'normalized': float(value / (np.max(np.abs(features)) + 1e-6)),
                'name': f'Feature {idx}',
                'category': self._categorize_feature_index(idx)
            })
        
        # Feature categories
        categories = {
            'text': 0,
            'semantic': 0,
            'structural': 0,
            'numerical': 0
        }
        
        # Simple categorization based on index
        for idx, value in enumerate(features):
            if idx < 20:
                categories['text'] += abs(value)
            elif idx < 40:
                categories['semantic'] += abs(value)
            elif idx < 60:
                categories['structural'] += abs(value)
            else:
                categories['numerical'] += abs(value)
        
        # Normalize
        total = sum(categories.values())
        if total > 0:
            categories = {k: float(v/total) for k, v in categories.items()}
        
        return {
            'top_features': top_features,
            'category_breakdown': categories,
            'feature_count': len(features),
            'feature_density': float(np.mean(np.abs(features))),
            'feature_sparsity': float(np.sum(features == 0) / len(features))
        }
    
    def _categorize_feature_index(self, idx: int) -> str:
        """Categorize feature by index."""
        if idx < 20:
            return 'Text'
        elif idx < 40:
            return 'Semantic'
        elif idx < 60:
            return 'Structural'
        else:
            return 'Numerical'
    
    def _generate_recommendations(
        self,
        quality: Dict,
        patterns: List[Dict],
        features: Dict
    ) -> List[Dict[str, str]]:
        """
        Generate actionable recommendations for improving narrative.
        """
        recommendations = []
        
        quality_score = quality['overall']
        
        # Quality-based recommendations
        if quality_score < 0.4:
            recommendations.append({
                'type': 'quality',
                'priority': 'high',
                'title': 'Strengthen Overall Narrative Quality',
                'description': 'Your narrative quality is below average. Focus on adding more specific details, vivid language, and clear structure.',
                'action': 'Add concrete examples, sensory details, and emotional resonance.'
            })
        
        if quality['complexity'] < 0.3:
            recommendations.append({
                'type': 'complexity',
                'priority': 'medium',
                'title': 'Increase Vocabulary Diversity',
                'description': 'Your narrative uses repetitive language. Vary your word choice for more engaging reading.',
                'action': 'Use synonyms and more descriptive language.'
            })
        
        # Pattern-based recommendations
        if len(patterns) == 0:
            recommendations.append({
                'type': 'pattern',
                'priority': 'high',
                'title': 'Add Recognizable Narrative Patterns',
                'description': 'No clear narrative archetypes detected. Consider incorporating classic story patterns like hero\'s journey, underdog, or rivalry.',
                'action': 'Frame your story using a recognizable narrative arc.'
            })
        elif len(patterns) == 1:
            recommendations.append({
                'type': 'pattern',
                'priority': 'low',
                'title': 'Consider Additional Narrative Layers',
                'description': f'Only one pattern detected ({patterns[0]["name"]}). Adding complementary patterns can deepen the narrative.',
                'action': 'Layer multiple narrative themes for complexity.'
            })
        
        # Feature-based recommendations
        text_strength = features['category_breakdown'].get('text', 0)
        semantic_strength = features['category_breakdown'].get('semantic', 0)
        
        if text_strength < 0.2:
            recommendations.append({
                'type': 'feature',
                'priority': 'medium',
                'title': 'Enhance Textual Richness',
                'description': 'Your narrative lacks textual depth. Add more descriptive passages.',
                'action': 'Expand descriptions and add more context.'
            })
        
        if semantic_strength < 0.2:
            recommendations.append({
                'type': 'feature',
                'priority': 'medium',
                'title': 'Strengthen Emotional Resonance',
                'description': 'Semantic features are weak. Add more emotional language and thematic depth.',
                'action': 'Use emotionally charged language and clear themes.'
            })
        
        # Always add a positive recommendation if quality is good
        if quality_score > 0.6:
            recommendations.append({
                'type': 'strength',
                'priority': 'info',
                'title': 'Strong Foundation',
                'description': f'Your narrative has strong fundamentals ({quality["percentile"]}th percentile). Minor refinements can make it excellent.',
                'action': 'Polish specific weak areas while maintaining strengths.'
            })
        
        return recommendations
    
    def _match_archetypes(self, patterns: List[Dict], text: str) -> List[Dict[str, Any]]:
        """
        Match narrative to classic archetypes.
        """
        archetypes = [
            {
                'name': 'The Hero',
                'description': 'Protagonist who overcomes challenges through courage and growth',
                'keywords': ['hero', 'protagonist', 'champion', 'overcome', 'victory'],
                'match_score': 0.0
            },
            {
                'name': 'The Underdog',
                'description': 'Unlikely victor fighting against overwhelming odds',
                'keywords': ['underdog', 'unlikely', 'overcome', 'surprise', 'upset'],
                'match_score': 0.0
            },
            {
                'name': 'The Mentor',
                'description': 'Wise guide who enables transformation in others',
                'keywords': ['mentor', 'guide', 'teach', 'wisdom', 'experience'],
                'match_score': 0.0
            },
            {
                'name': 'The Rival',
                'description': 'Matched opponent who drives excellence',
                'keywords': ['rival', 'competition', 'match', 'equal', 'opponent'],
                'match_score': 0.0
            },
            {
                'name': 'The Collective',
                'description': 'Group achieving more than sum of parts',
                'keywords': ['team', 'together', 'collective', 'group', 'unity'],
                'match_score': 0.0
            }
        ]
        
        text_lower = text.lower()
        
        for archetype in archetypes:
            matches = sum(1 for kw in archetype['keywords'] if kw in text_lower)
            archetype['match_score'] = float(matches / len(archetype['keywords']))
        
        # Sort by match score
        archetypes.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Only return top 3 with score > 0
        return [a for a in archetypes[:3] if a['match_score'] > 0]
    
    def _get_text_stats(self, text: str) -> Dict[str, Any]:
        """Get basic text statistics."""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': float(np.mean([len(w) for w in words]) if words else 0),
            'unique_words': len(set(words)),
            'vocabulary_richness': float(len(set(words)) / (len(words) + 1))
        }


# Global analyzer instance
analyzer = UniversalNarrativeAnalyzer()


@narrative_analyzer_bp.route('/analyze')
def analyze_page():
    """Landing page for narrative analysis."""
    return render_template('narrative_analyzer/landing.html')


@narrative_analyzer_bp.route('/api/analyze', methods=['POST'])
def analyze_api():
    """API endpoint for narrative analysis."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text']
        context = data.get('context', {})
        
        # Validate text length
        if len(text) < 50:
            return jsonify({
                'success': False,
                'error': 'Text too short (minimum 50 characters)'
            }), 400
        
        if len(text) > 10000:
            return jsonify({
                'success': False,
                'error': 'Text too long (maximum 10,000 characters)'
            }), 400
        
        # Analyze
        results = analyzer.analyze(text, context)
        
        if not results['success']:
            return jsonify(results), 500
        
        # Store in session for results page
        session['last_analysis'] = {
            'text': text[:500],  # Store truncated for session
            'results': results
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@narrative_analyzer_bp.route('/api/compare', methods=['POST'])
@narrative_analyzer_bp.route('/api/comprehensive_compare', methods=['POST'])  # Alias
def compare_api():
    """
    API endpoint for comparing two narratives in one call.
    More efficient than calling /api/analyze twice.
    
    Accepts: text1, text2, context1 (optional), context2 (optional)
    Returns: Both analysis results plus comparison metadata
    """
    try:
        data = request.get_json()
        
        if not data or 'text1' not in data or 'text2' not in data:
            return jsonify({
                'success': False,
                'error': 'Both text1 and text2 required'
            }), 400
        
        text1 = data['text1']
        text2 = data['text2']
        
        # Validate both texts
        errors = []
        if len(text1) < 50:
            errors.append('Text1 too short (minimum 50 characters)')
        if len(text2) < 50:
            errors.append('Text2 too short (minimum 50 characters)')
        if len(text1) > 10000:
            errors.append('Text1 too long (maximum 10,000 characters)')
        if len(text2) > 10000:
            errors.append('Text2 too long (maximum 10,000 characters)')
        
        if errors:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Analyze both
        result1 = analyzer.analyze(text1, data.get('context1', {}))
        result2 = analyzer.analyze(text2, data.get('context2', {}))
        
        if not result1['success'] or not result2['success']:
            return jsonify({
                'success': False,
                'error': 'Analysis failed',
                'result1': result1,
                'result2': result2
            }), 500
        
        # Determine winner
        q1 = float(result1['narrative_quality']['overall'])
        q2 = float(result2['narrative_quality']['overall'])
        winner = 1 if q1 > q2 else 2
        quality_diff = abs(q1 - q2)
        
        # Convert numpy types to native Python types for JSON serialization
        import json
        result1_json = json.loads(json.dumps(result1, default=lambda x: float(x) if hasattr(x, 'item') else x))
        result2_json = json.loads(json.dumps(result2, default=lambda x: float(x) if hasattr(x, 'item') else x))
        
        return jsonify({
            'success': True,
            'result1': result1_json,
            'result2': result2_json,
            'comparison': {
                'winner': int(winner),
                'quality_difference': float(quality_diff),
                'quality_difference_percent': float(quality_diff * 100),
                'winner_name': f'Narrative {winner}'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@narrative_analyzer_bp.route('/results')
def results_page():
    """Results visualization page."""
    # Get from session
    analysis = session.get('last_analysis')
    
    if not analysis:
        return render_template('narrative_analyzer/no_results.html')
    
    return render_template(
        'narrative_analyzer/results.html',
        text_preview=analysis['text'],
        results=analysis['results']
    )


@narrative_analyzer_bp.route('/compare')
def compare_page():
    """Comparison mode page."""
    return render_template('narrative_analyzer/compare.html')


@narrative_analyzer_bp.route('/api')
def api_docs():
    """API documentation page."""
    return render_template('narrative_analyzer/api_docs.html')


@narrative_analyzer_bp.route('/examples')
def examples_page():
    """Example narratives page."""
    examples = [
        {
            'title': 'Startup Pitch',
            'text': 'We are revolutionizing the way people interact with AI. Our founding team has 20 years of combined experience from Google and Stanford. We have traction with 10,000 users and $500K ARR. Our vision is to democratize AI for everyone.',
            'domain': 'startups'
        },
        {
            'title': 'Game Preview',
            'text': 'The Lakers face the Celtics tonight in a historic rivalry matchup. LeBron James returns after missing three games, facing a young Celtics team on a 7-game win streak. Both teams desperate for playoff positioning.',
            'domain': 'sports'
        },
        {
            'title': 'Movie Synopsis',
            'text': 'A young programmer discovers she has the ability to manipulate reality through code. As corporations hunt her, she must decide whether to use her power to liberate humanity or protect those she loves.',
            'domain': 'movies'
        }
    ]
    
    return render_template('narrative_analyzer/examples.html', examples=examples)

