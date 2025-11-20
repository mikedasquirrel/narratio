"""
Real-time narrative analysis routes with AI-powered insights
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for
import sys
from pathlib import Path
import numpy as np
import json
import os

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

# OpenAI Integration
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False
    openai_client = None
    print("Warning: OpenAI not installed. AI features disabled.")

from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.relational import RelationalValueTransformer
from src.transformers.nominative import NominativeAnalysisTransformer
from src.analysis.contextual_analyzer import ContextualNarrativeAnalyzer
from src.domain.expertise_tracker import DomainExpertiseTracker
from src.domain.domain_detector import MultiDomainDetector
from src.domain.confidence_calibrator import ConfidenceCalibrator

# NEW: Import enhanced analysis modules
from src.analysis.conversation_manager import conversation_manager, ConversationManager
from src.analysis.question_generator import create_question_generator
from src.analysis.narrative_weighter import create_narrative_weighter
from src.analysis.scenario_predictor import create_scenario_predictor
from src.analysis.feature_impact import create_feature_impact_analyzer
from src.analysis.temporal_analyzer import create_temporal_analyzer

# NEW: Import taxonomy system
from src.taxonomy.narrative_taxonomy import narrative_taxonomy

analysis_bp = Blueprint('analysis', __name__)

# Initialize domain expertise tracking
expertise_tracker = DomainExpertiseTracker()
domain_detector = MultiDomainDetector()
confidence_calibrator = ConfidenceCalibrator()

# Initialize enhanced analysis modules
question_generator = create_question_generator(openai_client if AI_ENABLED else None)
narrative_weighter = create_narrative_weighter()
scenario_predictor = create_scenario_predictor()
feature_impact_analyzer = create_feature_impact_analyzer()
temporal_analyzer = create_temporal_analyzer()

@analysis_bp.route('/')
def analyzer():
    """Main analyzer landing page - redirects to compare tool."""
    return redirect(url_for('analysis.compare'))

@analysis_bp.route('/compare')
def compare():
    """Comprehensive comparison tool with domain-aware prediction."""
    return render_template('compare.html')

@analysis_bp.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze text with selected transformers and identify missing context."""
    data = request.get_json()
    text = data.get('text', '')
    transformers = data.get('transformers', [])
    domain = data.get('domain', 'general')  # NEW: domain parameter
    known_context = data.get('context', {})  # NEW: known contextual variables
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    results = {}
    
    # NEW: Contextual analysis
    contextual_analyzer = ContextualNarrativeAnalyzer()
    contextual_analysis = contextual_analyzer.analyze_with_context(text, domain, known_context)
    
    try:
        # Fit dummy data for transformers that need it
        dummy_texts = [text, "Sample text for fitting.", "Another sample."]
        
        if 'ensemble' in transformers:
            transformer = EnsembleNarrativeTransformer(n_top_terms=20)
            transformer.fit(dummy_texts)
            features = transformer.transform([text])[0]
            results['ensemble'] = {
                'features': features.tolist(),
                'interpretation': transformer.get_narrative_report()['interpretation']
            }
        
        if 'linguistic' in transformers:
            transformer = LinguisticPatternsTransformer()
            transformer.fit(dummy_texts)
            features = transformer.transform([text])[0]
            results['linguistic'] = {
                'features': features.tolist(),
                'interpretation': transformer.get_narrative_report()['interpretation']
            }
        
        if 'self_perception' in transformers:
            transformer = SelfPerceptionTransformer()
            transformer.fit(dummy_texts)
            features = transformer.transform([text])[0]
            results['self_perception'] = {
                'features': features.tolist(),
                'interpretation': transformer.get_narrative_report()['interpretation']
            }
        
        if 'potential' in transformers:
            transformer = NarrativePotentialTransformer()
            transformer.fit(dummy_texts)
            features = transformer.transform([text])[0]
            results['potential'] = {
                'features': features.tolist(),
                'interpretation': transformer.get_narrative_report()['interpretation']
            }
        
        if 'relational' in transformers:
            transformer = RelationalValueTransformer()
            transformer.fit(dummy_texts)
            features = transformer.transform([text])[0]
            results['relational'] = {
                'features': features.tolist(),
                'interpretation': transformer.get_narrative_report()['interpretation']
            }
        
        if 'nominative' in transformers:
            transformer = NominativeAnalysisTransformer()
            transformer.fit(dummy_texts)
            features = transformer.transform([text])[0]
            results['nominative'] = {
                'features': features.tolist(),
                'interpretation': transformer.get_narrative_report()['interpretation']
            }
        
        return jsonify({
            'success': True,
            'results': results,
            'text_length': len(text),
            'word_count': len(text.split()),
            'contextual_analysis': {
                'missing_variables': contextual_analysis['missing_variables'],
                'follow_up_questions': contextual_analysis['follow_up_questions'],
                'recommendations': contextual_analysis['contextual_recommendations']
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/comprehensive_compare', methods=['POST'])
def comprehensive_compare():
    """
    Comprehensive comparison running ALL transformers and extracting detailed insights.
    Intelligently handles any input type (sports, profiles, products, etc.)
    Includes user's comparative question for contextualized AI analysis.
    """
    data = request.get_json()
    text_a = data.get('text_a', '')
    text_b = data.get('text_b', '')
    question = data.get('question', '').strip()
    
    if not text_a or not text_b:
        return jsonify({'error': 'Both texts required'}), 400
    
    try:
        # Prepare corpus for fitting
        corpus = [text_a, text_b, "Sample text for fitting.", "Another sample for training."]
        
        comparison_results = {
            'metadata': {
                'text_a_length': len(text_a),
                'text_b_length': len(text_b),
                'text_a_words': len(text_a.split()),
                'text_b_words': len(text_b.split()),
            },
            'transformers': {}
        }
        
        # 1. Nominative Analysis
        nom = NominativeAnalysisTransformer(track_proper_nouns=True, track_categories=True)
        nom.fit(corpus)
        nom_a = nom.transform([text_a])[0]
        nom_b = nom.transform([text_b])[0]
        
        comparison_results['transformers']['nominative'] = {
            'text_a': {
                'features': nom_a.tolist(),
                'semantic_field_profile': nom.get_semantic_field_profile(text_a)
            },
            'text_b': {
                'features': nom_b.tolist(),
                'semantic_field_profile': nom.get_semantic_field_profile(text_b)
            },
            'interpretation': nom._generate_interpretation(),
            'difference': float(np.linalg.norm(nom_a - nom_b)),
            'feature_names': _get_nominative_feature_names()
        }
        
        # 2. Self-Perception
        sp = SelfPerceptionTransformer(track_attribution=True, track_growth=True, track_coherence=True)
        sp.fit(corpus)
        sp_a = sp.transform([text_a])[0]
        sp_b = sp.transform([text_b])[0]
        
        comparison_results['transformers']['self_perception'] = {
            'text_a': {'features': sp_a.tolist()},
            'text_b': {'features': sp_b.tolist()},
            'interpretation': sp._generate_interpretation(),
            'difference': float(np.linalg.norm(sp_a - sp_b)),
            'feature_names': _get_self_perception_feature_names()
        }
        
        # 3. Narrative Potential
        np_trans = NarrativePotentialTransformer(track_modality=True, track_flexibility=True, track_arc_position=True)
        np_trans.fit(corpus)
        np_a = np_trans.transform([text_a])[0]
        np_b = np_trans.transform([text_b])[0]
        
        comparison_results['transformers']['narrative_potential'] = {
            'text_a': {'features': np_a.tolist()},
            'text_b': {'features': np_b.tolist()},
            'interpretation': np_trans._generate_interpretation(),
            'difference': float(np.linalg.norm(np_a - np_b)),
            'feature_names': _get_narrative_potential_feature_names()
        }
        
        # 4. Linguistic Patterns
        ling = LinguisticPatternsTransformer(track_evolution=True, n_segments=3)
        ling.fit(corpus)
        ling_a = ling.transform([text_a])[0]
        ling_b = ling.transform([text_b])[0]
        
        comparison_results['transformers']['linguistic'] = {
            'text_a': {'features': ling_a.tolist()},
            'text_b': {'features': ling_b.tolist()},
            'interpretation': ling._generate_interpretation(),
            'difference': float(np.linalg.norm(ling_a - ling_b)),
            'feature_names': _get_linguistic_feature_names()
        }
        
        # 5. Relational Value
        rel = RelationalValueTransformer(n_features=50, complementarity_threshold=0.3)
        rel.fit(corpus)
        rel_a = rel.transform([text_a])[0]
        rel_b = rel.transform([text_b])[0]
        
        comparison_results['transformers']['relational'] = {
            'text_a': {'features': rel_a.tolist()},
            'text_b': {'features': rel_b.tolist()},
            'interpretation': rel._generate_interpretation(),
            'difference': float(np.linalg.norm(rel_a - rel_b)),
            'feature_names': _get_relational_feature_names()
        }
        
        # 6. Ensemble Effects
        ens = EnsembleNarrativeTransformer(n_top_terms=30, network_metrics=True, diversity_metrics=True)
        ens.fit(corpus)
        ens_a = ens.transform([text_a])[0]
        ens_b = ens.transform([text_b])[0]
        
        comparison_results['transformers']['ensemble'] = {
            'text_a': {'features': ens_a.tolist()},
            'text_b': {'features': ens_b.tolist()},
            'interpretation': ens._generate_interpretation(),
            'difference': float(np.linalg.norm(ens_a - ens_b)),
            'top_pairs': ens.get_top_ensemble_pairs(n=10) if hasattr(ens, 'get_top_ensemble_pairs') else [],
            'feature_names': _get_ensemble_feature_names()
        }
        
        # Calculate overall similarity
        all_diffs = [t['difference'] for t in comparison_results['transformers'].values()]
        comparison_results['overall_similarity'] = 1 / (1 + np.mean(all_diffs))
        comparison_results['most_different_dimension'] = max(
            comparison_results['transformers'].items(), 
            key=lambda x: x[1]['difference']
        )[0]
        comparison_results['most_similar_dimension'] = min(
            comparison_results['transformers'].items(), 
            key=lambda x: x[1]['difference']
        )[0]
        
        # DOMAIN EXPERTISE ASSESSMENT
        domain_collage = domain_detector.analyze_domain_collage(text_a, text_b)
        expertise_assessment = expertise_tracker.assess_expertise(domain_collage['all_domains'])
        
        comparison_results['domain_expertise'] = {
            'domains_detected': domain_collage,
            'expertise': expertise_assessment,
            'can_predict': expertise_assessment['can_predict'],
            'confidence_level': expertise_assessment['confidence_level']
        }
        
        # PREDICTION (if we have expertise)
        if expertise_assessment['can_predict']:
            # Simple prediction based on differential
            # In production, use domain-specific model
            all_diffs = [t['difference'] for t in comparison_results['transformers'].values()]
            avg_diff = np.mean(all_diffs)
            
            # Raw prediction: larger differences favor text A
            raw_prediction_a = 0.5 + (avg_diff / 10) * 0.5
            raw_prediction_a = max(0.1, min(0.9, raw_prediction_a))
            
            # Calibrate based on domain expertise
            calibrated = confidence_calibrator.calibrate_prediction(
                raw_prediction_a,
                expertise_assessment
            )
            
            comparison_results['prediction'] = {
                'text_a_probability': calibrated['calibrated_prediction'],
                'text_b_probability': 1 - calibrated['calibrated_prediction'],
                'winner': 'Text A' if calibrated['calibrated_prediction'] > 0.5 else 'Text B',
                'confidence': calibrated['prediction_confidence'],
                'confidence_level': calibrated['confidence_level'],
                'can_predict': calibrated['can_predict'],
                'reasoning': calibrated['reasoning'],
                'based_on_training': expertise_assessment['total_training_samples']
            }
        else:
            comparison_results['prediction'] = {
                'can_predict': False,
                'message': f"Insufficient training data. System has {expertise_assessment['total_training_samples']} examples in this domain.",
                'recommendation': "I can compare these texts but cannot reliably predict outcomes. This is a domain I'm not trained on."
            }
        
        # Generate AI insights if enabled
        if AI_ENABLED:
            try:
                ai_insights = generate_ai_insights(text_a, text_b, comparison_results, question)
                comparison_results['ai_insights'] = ai_insights
            except Exception as e:
                comparison_results['ai_insights'] = {
                    'error': f'AI analysis unavailable: {str(e)}',
                    'comparison_type': 'unknown',
                    'key_insights': [],
                    'direct_answer': f'AI analysis failed: {str(e)}'
                }
        else:
            comparison_results['ai_insights'] = {
                'error': 'AI features disabled',
                'comparison_type': 'unknown',
                'key_insights': [],
                'direct_answer': 'AI integration not available'
            }
        
        # ============================================================================
        # NEW: ENHANCED ANALYSIS ADDITIONS
        # ============================================================================
        
        # 1. Narrative Context Weighting
        try:
            narrative_weighting = narrative_weighter.compute_narrative_weight(
                text_a, text_b, question, {}
            )
            comparison_results['narrative_weighting'] = narrative_weighting
        except Exception as e:
            print(f"Narrative weighting error: {e}")
            comparison_results['narrative_weighting'] = {'weight': 1.0, 'level': 'MODERATE'}
        
        # 2. Multi-Scenario Predictions
        try:
            scenarios = scenario_predictor.generate_scenarios(
                comparison_results,
                comparison_results,
                comparison_results.get('narrative_weighting', {}).get('weight', 1.0),
                {}
            )
            comparison_results['scenarios'] = scenarios
        except Exception as e:
            print(f"Scenario generation error: {e}")
            comparison_results['scenarios'] = None
        
        # 3. Feature Impact Analysis
        try:
            feature_impact = feature_impact_analyzer.analyze_feature_impact(
                comparison_results,
                comparison_results
            )
            comparison_results['feature_impact'] = feature_impact
        except Exception as e:
            print(f"Feature impact error: {e}")
            comparison_results['feature_impact'] = None
        
        # 4. Temporal Dynamics
        try:
            temporal_dynamics = temporal_analyzer.analyze_temporal_dynamics(
                comparison_results,
                comparison_results,
                comparison_results.get('narrative_weighting', {}).get('weight', 1.0)
            )
            comparison_results['temporal_dynamics'] = temporal_dynamics
        except Exception as e:
            print(f"Temporal analysis error: {e}")
            comparison_results['temporal_dynamics'] = None
        
        # 5. Generate initial follow-up questions (for potential conversation)
        try:
            primary_domain = domain_collage['primary_domain'][0]
            follow_up_questions = question_generator.generate_questions(
                text_a, text_b, primary_domain,
                comparison_results, question, {}, max_questions=5
            )
            comparison_results['suggested_questions'] = [q['question'] for q in follow_up_questions]
        except Exception as e:
            print(f"Question generation error: {e}")
            comparison_results['suggested_questions'] = []
        
        # 6. Add to Narrative Taxonomy (biology system)
        try:
            organism = narrative_taxonomy.add_organism(
                text_a,
                text_b,
                comparison_results,
                comparison_results.get('narrative_weighting', {}).get('weight', 1.0),
                domain_collage
            )
            comparison_results['organism_id'] = organism.organism_id
            comparison_results['species_name'] = organism.species_name
        except Exception as e:
            print(f"Taxonomy addition error: {e}")
            comparison_results['organism_id'] = None
        
        return jsonify({
            'success': True,
            'comparison': comparison_results
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


def _get_nominative_feature_names():
    """Feature names for nominative analysis."""
    semantic_fields = ['motion', 'cognition', 'emotion', 'perception', 'communication', 
                      'creation', 'change', 'possession', 'existence', 'social']
    base_features = [f'{field}_density' for field in semantic_fields]
    base_features += ['dominant_field', 'semantic_entropy', 'proper_noun_density', 
                     'proper_noun_diversity', 'proper_noun_repetition', 'category_usage',
                     'category_diversity', 'identity_markers', 'comparison_usage',
                     'naming_consistency', 'specificity', 'categorical_thinking',
                     'field_balance', 'identity_construction']
    return base_features


def _get_self_perception_feature_names():
    """Feature names for self-perception."""
    return ['fp_singular', 'fp_plural', 'self_focus_ratio', 'positive_attribution',
            'negative_attribution', 'attribution_balance', 'self_confidence',
            'growth_orientation', 'stasis_orientation', 'growth_mindset',
            'aspirational_density', 'descriptive_density', 'aspirational_ratio',
            'high_agency', 'low_agency', 'agency_score', 'identity_coherence',
            'self_complexity', 'self_awareness', 'self_transformation', 'self_positioning']


def _get_narrative_potential_feature_names():
    """Feature names for narrative potential."""
    return ['future_tense', 'future_intention', 'future_orientation', 'possibility_modals',
            'potential_words', 'possibility_score', 'growth_verbs', 'change_words',
            'growth_mindset', 'flexibility', 'rigidity', 'flexibility_ratio',
            'possibility_words', 'constraint_words', 'net_possibility',
            'beginning_phase', 'middle_phase', 'resolution_phase', 'dominant_arc',
            'conditional_language', 'alternative_language', 'openness_score',
            'temporal_breadth', 'actualization', 'momentum']


def _get_linguistic_feature_names():
    """Feature names for linguistic patterns."""
    base = ['first_person', 'second_person', 'third_person', 'voice_entropy',
            'past_tense', 'present_tense', 'future_tense', 'temporal_entropy',
            'active_voice', 'passive_voice', 'agency_ratio', 'sentiment',
            'emotional_intensity', 'subordination', 'relativization', 'modality',
            'complexity_score']
    evolution = ['voice_trend', 'temporal_trend', 'complexity_trend',
                'voice_variability', 'temporal_variability', 'complexity_variability',
                'voice_consistency', 'temporal_consistency', 'complexity_consistency']
    return base + evolution


def _get_relational_feature_names():
    """Feature names for relational value."""
    return ['internal_complementarity', 'relational_density', 'synergy_gini',
            'complementarity_potential', 'value_ratio', 'relational_entropy',
            'complementarity_balance', 'synergistic_peaks', 'relational_coherence']


def _get_ensemble_feature_names():
    """Feature names for ensemble effects."""
    base = ['ensemble_size', 'cooccurrence_density', 'diversity_entropy']
    network = ['avg_centrality', 'max_centrality', 'centrality_std', 'avg_betweenness']
    coherence = ['ensemble_coherence', 'ensemble_reach']
    return base + network + coherence


def generate_ai_insights(text_a, text_b, comparison_data, user_question=''):
    """
    Use OpenAI to generate intelligent insights about the comparison.
    
    Analyzes all transformer outputs and provides:
    - Comparison type detection
    - Direct answer to user's question
    - Key narrative insights
    - Important feature identification
    - Implications and recommendations
    """
    # Check if AI is available
    if not AI_ENABLED or openai_client is None:
        return {
            'error': 'OpenAI client not initialized',
            'comparison_type': 'unknown',
            'key_insights': [],
            'important_features': [],
            'direct_answer': 'AI features unavailable'
        }
    
    # Prepare transformer summary for AI
    transformer_summary = {}
    for name, tdata in comparison_data['transformers'].items():
        transformer_summary[name] = {
            'difference': tdata['difference'],
            'key_features': _extract_key_features(name, tdata)
        }
    
    # Extract semantic field profiles
    semantic_fields_a = json.dumps(comparison_data['transformers']['nominative']['text_a']['semantic_field_profile'], indent=2)
    semantic_fields_b = json.dumps(comparison_data['transformers']['nominative']['text_b']['semantic_field_profile'], indent=2)
    
    # Build task 2 based on whether question exists
    if user_question:
        task2 = f'ANSWER THE USER\'S QUESTION: "{user_question}" - Provide a direct, evidence-based answer using narrative analysis'
        answer_field = '"direct_answer": "Direct answer to user\'s question with narrative evidence",'
    else:
        task2 = "Determine the implicit comparison intent"
        answer_field = '"presumed_intent": "What user likely wants to know",'
    
    # Build comprehensive prompt
    prompt = f"""You are an expert narrative analyst. Analyze this text comparison using our 6-dimensional narrative framework.

TEXT A: "{text_a[:800]}"

TEXT B: "{text_b[:800]}"

TRANSFORMER OUTPUTS:

1. NOMINATIVE ANALYSIS (naming & categorization):
   - Difference score: {comparison_data['transformers']['nominative']['difference']:.2f}
   - Semantic fields A: {semantic_fields_a}
   - Semantic fields B: {semantic_fields_b}

2. SELF-PERCEPTION (identity & agency):
   - Difference score: {comparison_data['transformers']['self_perception']['difference']:.2f}

3. NARRATIVE POTENTIAL (openness & growth):
   - Difference score: {comparison_data['transformers']['narrative_potential']['difference']:.2f}
   
4. LINGUISTIC PATTERNS (voice & temporality):
   - Difference score: {comparison_data['transformers']['linguistic']['difference']:.2f}

5. RELATIONAL VALUE (complementarity):
   - Difference score: {comparison_data['transformers']['relational']['difference']:.2f}

6. ENSEMBLE EFFECTS (diversity & networks):
   - Difference score: {comparison_data['transformers']['ensemble']['difference']:.2f}

TASKS:
1. Identify the comparison type (sports, products, profiles, brands, ideas, places, other)
2. {task2}
3. Generate 4-5 key narrative insights that are specific and non-obvious
4. Identify the 5-8 most important distinguishing features and explain WHY they matter
5. Provide actionable implications

OUTPUT FORMAT (JSON):
{{
  "comparison_type": "string (e.g., 'sports_teams', 'product_comparison', 'user_profiles')",
  "comparison_category": "string (one word description)",
  "confidence": 0.0-1.0,
  {answer_field}
  "reasoning": "Brief explanation of the answer using specific narrative features",
  "key_insights": [
    "Specific insight about a striking narrative difference...",
    "Another meaningful pattern discovered...",
    ...
  ],
  "important_features": [
    {{
      "feature": "motion_density",
      "transformer": "nominative",
      "text_a_value": 0.15,
      "text_b_value": 0.08,
      "importance": 0.95,
      "explanation": "Text A uses 2x more motion language, suggesting action-oriented framing"
    }}
  ],
  "narrative_themes": {{
    "text_a": "One sentence capturing A's narrative essence",
    "text_b": "One sentence capturing B's narrative essence"
  }},
  "implications": "What this comparison reveals about narrative strategies and effects",
  "recommendations": "Actionable insights for understanding or using these narratives"
}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert narrative analyst specializing in computational narrative analysis. You understand nominative patterns, linguistic structures, self-perception markers, narrative potential indicators, relational dynamics, and ensemble effects. Provide precise, insightful analysis grounded in the data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        ai_response = response.choices[0].message.content
        
        # Try to parse JSON from response
        try:
            # Find JSON in response (might be wrapped in markdown)
            if '```json' in ai_response:
                json_str = ai_response.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_response:
                json_str = ai_response.split('```')[1].split('```')[0].strip()
            else:
                json_str = ai_response.strip()
            
            insights = json.loads(json_str)
            return insights
        except json.JSONDecodeError:
            # If JSON parsing fails, return structured error
            return {
                'comparison_type': 'analysis_error',
                'key_insights': [ai_response[:500]],
                'important_features': [],
                'implications': 'AI provided unstructured response',
                'raw_response': ai_response
            }
    
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'comparison_type': 'unknown',
            'key_insights': [],
            'important_features': [],
            'direct_answer': f'AI encountered an error: {str(e)}'
        }
        print("AI Insights Error:", error_details)  # Server-side logging
        return error_details


def _extract_key_features(transformer_name, tdata):
    """Extract the most significant features from transformer output."""
    features_a = np.array(tdata['text_a']['features'])
    features_b = np.array(tdata['text_b']['features'])
    feature_names = tdata['feature_names']
    
    # Calculate differences
    diffs = np.abs(features_a - features_b)
    
    # Get top 5 most different features
    top_indices = np.argsort(diffs)[-5:][::-1]
    
    key_features = []
    for idx in top_indices:
        if idx < len(feature_names):
            key_features.append({
                'name': feature_names[idx],
                'value_a': float(features_a[idx]),
                'value_b': float(features_b[idx]),
                'difference': float(diffs[idx])
            })
    
    return key_features


# ============================================================================
# NEW: INTERACTIVE ANALYSIS ENDPOINTS
# ============================================================================

@analysis_bp.route('/api/interactive_start', methods=['POST'])
def interactive_start():
    """
    Start interactive analysis conversation.
    Generates initial insights and follow-up questions.
    """
    data = request.get_json()
    text_a = data.get('text_a', '')
    text_b = data.get('text_b', '')
    question = data.get('question', '').strip()
    
    if not text_a or not text_b:
        return jsonify({'error': 'Both texts required'}), 400
    
    try:
        # Run comprehensive comparison first
        comparison_response = comprehensive_compare()
        comparison_result = comparison_response.get_json()
        
        if not comparison_result.get('success'):
            return comparison_result, 500
        
        comparison_data = comparison_result['comparison']
        
        # Start conversation
        conversation = conversation_manager.start_conversation(
            text_a, text_b, question, comparison_data
        )
        
        # Detect domain
        domain_collage = domain_detector.analyze_domain_collage(text_a, text_b)
        primary_domain = domain_collage['primary_domain'][0]
        
        # Compute narrative weight
        narrative_weighting = narrative_weighter.compute_narrative_weight(
            text_a, text_b, question, {}
        )
        
        # Generate follow-up questions
        questions = question_generator.generate_questions(
            text_a, text_b, primary_domain,
            comparison_data, question, {}, max_questions=5
        )
        
        # Add first turn to conversation
        question_texts = [q['question'] for q in questions]
        conversation_manager.add_turn(
            conversation['id'],
            question_texts,
            None,
            None
        )
        
        return jsonify({
            'success': True,
            'conversation_id': conversation['id'],
            'initial_analysis': comparison_data,
            'narrative_weighting': narrative_weighting,
            'follow_up_questions': questions,
            'should_continue': True,
            'message': 'Initial analysis complete. Answer questions to refine predictions.'
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@analysis_bp.route('/api/interactive_refine', methods=['POST'])
def interactive_refine():
    """
    Refine analysis based on user responses to questions.
    Can ask additional questions or provide final refined analysis.
    """
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    responses = data.get('responses', {})  # {question: answer}
    
    if not conversation_id:
        return jsonify({'error': 'conversation_id required'}), 400
    
    try:
        # Get conversation
        conversation = conversation_manager.get_conversation(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Get accumulated context
        context = conversation_manager.get_context(conversation_id)
        
        # Add responses to conversation
        last_turn = conversation['turns'][-1] if conversation['turns'] else {}
        last_questions = last_turn.get('questions', [])
        
        conversation_manager.add_turn(
            conversation_id,
            last_questions,
            responses,
            None
        )
        
        # Get updated context
        updated_context = conversation_manager.get_context(conversation_id)
        
        # Recompute narrative weight with new context
        text_a = conversation['texts']['text_a']
        text_b = conversation['texts']['text_b']
        question = conversation['initial_question']
        
        narrative_weighting = narrative_weighter.compute_narrative_weight(
            text_a, text_b, question, updated_context
        )
        
        # Get comparison data
        comparison_data = conversation['comparison_data']
        
        # Generate scenarios with updated context
        scenarios = scenario_predictor.generate_scenarios(
            comparison_data,
            comparison_data,
            narrative_weighting['weight'],
            updated_context
        )
        
        # Generate feature impact analysis
        feature_impact = feature_impact_analyzer.analyze_feature_impact(
            comparison_data,
            comparison_data
        )
        
        # Generate temporal dynamics
        temporal_dynamics = temporal_analyzer.analyze_temporal_dynamics(
            comparison_data,
            comparison_data,
            narrative_weighting['weight']
        )
        
        # Check if should continue conversation
        should_continue = conversation_manager.should_continue_conversation(conversation_id)
        
        additional_questions = []
        if should_continue:
            # Generate more questions if needed
            domain_collage = domain_detector.analyze_domain_collage(text_a, text_b)
            primary_domain = domain_collage['primary_domain'][0]
            
            additional_questions = question_generator.generate_questions(
                text_a, text_b, primary_domain,
                comparison_data, question, updated_context, max_questions=3
            )
        
        # Mark conversation complete if done
        if not should_continue or not additional_questions:
            conversation_manager.mark_complete(conversation_id)
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'refined_analysis': {
                'narrative_weighting': narrative_weighting,
                'scenarios': scenarios,
                'feature_impact': feature_impact,
                'temporal_dynamics': temporal_dynamics
            },
            'context': updated_context,
            'additional_questions': [q['question'] for q in additional_questions] if additional_questions else [],
            'should_continue': should_continue and len(additional_questions) > 0,
            'conversation_complete': not should_continue or not additional_questions,
            'message': 'Analysis refined with your input.' if not should_continue else 'Additional questions generated.'
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@analysis_bp.route('/api/scenario_analysis', methods=['POST'])
def scenario_analysis():
    """
    Generate multi-scenario predictions with detailed breakdowns.
    """
    data = request.get_json()
    text_a = data.get('text_a', '')
    text_b = data.get('text_b', '')
    question = data.get('question', '').strip()
    context = data.get('context', {})
    
    if not text_a or not text_b:
        return jsonify({'error': 'Both texts required'}), 400
    
    try:
        # Get base comparison (reuse comprehensive_compare logic)
        # Create temporary request context with data
        from flask import Request
        import io
        
        # Run comprehensive comparison
        comparison_response = comprehensive_compare()
        comparison_result = comparison_response.get_json()
        
        if not comparison_result.get('success'):
            return comparison_result, 500
        
        comparison_data = comparison_result['comparison']
        
        # Compute narrative weight
        narrative_weighting = narrative_weighter.compute_narrative_weight(
            text_a, text_b, question, context
        )
        
        # Generate scenarios
        scenarios = scenario_predictor.generate_scenarios(
            comparison_data,
            comparison_data,
            narrative_weighting['weight'],
            context
        )
        
        # Generate feature impact
        feature_impact = feature_impact_analyzer.analyze_feature_impact(
            comparison_data,
            comparison_data
        )
        
        # Generate temporal dynamics
        temporal_dynamics = temporal_analyzer.analyze_temporal_dynamics(
            comparison_data,
            comparison_data,
            narrative_weighting['weight']
        )
        
        # Betting/decision recommendation (if applicable)
        betting_recommendation = None
        if narrative_weighting['weight'] > 1.0:
            # Calculate model edge (simplified)
            base_pred = comparison_data.get('prediction', {})
            if base_pred.get('can_predict'):
                prob_a = base_pred['text_a_probability']
                edge = abs(prob_a - 0.5)
                confidence = base_pred.get('confidence', 0.5)
                
                betting_recommendation = narrative_weighter.should_bet_on_comparison(
                    narrative_weighting['weight'],
                    confidence,
                    edge
                )
        
        return jsonify({
            'success': True,
            'narrative_weighting': narrative_weighting,
            'scenarios': scenarios,
            'feature_impact': feature_impact,
            'temporal_dynamics': temporal_dynamics,
            'betting_recommendation': betting_recommendation,
            'base_comparison': comparison_data
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

