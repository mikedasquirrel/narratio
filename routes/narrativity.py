"""
Narrativity Spectrum Route

Visualizes domains across the narrativity spectrum (0=circumscribed to 1=open).
Shows the fundamental organizing principle for narrative taxonomy.
"""

from flask import Blueprint, render_template, jsonify
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

from src.theory.narrativity_spectrum import NarrivityAnalyzer

narrativity_bp = Blueprint('narrativity', __name__)


@narrativity_bp.route('/narrativity/spectrum')
def spectrum_view():
    """Main narrativity spectrum visualization."""
    return render_template('narrativity_spectrum.html')


@narrativity_bp.route('/api/narrativity/spectrum-data')
def get_spectrum_data():
    """Get complete spectrum data for all domains."""
    
    analyzer = NarrivityAnalyzer()
    
    # Analyze all domains
    domains_to_analyze = {
        'die_roll': "Rolling a 6-sided die. Outcome determined by physics. Six possible outcomes. No actor choice. Instant event.",
        
        'coin_flip': "Flipping a coin. Binary outcome determined by physics and initial conditions. No agency once flipped. Instant result.",
        
        'chess_move': "Player chooses move in chess. Highly constrained by rules. Discrete options enumerable. Strategic agency within rules. Performance measured objectively.",
        
        'basketball_game': "Team sport with 5 players per side, 48 minutes, rules-constrained. Players have agency within rules. Performance measured but narrative emerges. Temporal unfolding.",
        
        'startup_pitch': "Founders describe product to investors. Format partially constrained (deck, time). Creative freedom in framing. Participant narrates creation. Outcomes measured.",
        
        'job_interview': "Candidate presents experience and fit. Format constrained (resume, Q&A). High interpretation by observers. Participant narrates performance. Social norms apply.",
        
        'therapy_session': "Patient discusses thoughts and feelings. Open format beyond time. Internal states narrated by participant. High interpretive freedom. Can go anywhere.",
        
        'marriage_proposal': "Person asks partner to marry. Format flexible (can be anything). High emotional stakes. Internal states + interaction. Participant narrates internal + performance.",
        
        'diary_entry': "Personal writing for self. No external constraints. Writer chooses everything. Internal states narrated by participant for self. Maximum freedom.",
        
        'dream': "Subconscious narrative generation during sleep. No conscious control. High interpretive freedom. Internal states narrated by unconscious. Completely unconstrained format."
    }
    
    spectrum_data = []
    
    for domain_name, description in domains_to_analyze.items():
        measure = analyzer.analyze_domain_narrativity(domain_name, description)
        
        spectrum_data.append({
            'name': domain_name.replace('_', ' ').title(),
            'narrativity': float(measure.narrativity_score),
            'alpha_predicted': float(measure.alpha_prediction),
            'structural_openness': float(measure.structural_openness),
            'temporal_freedom': float(measure.temporal_freedom),
            'actor_agency': float(measure.actor_agency),
            'observer_interpretation': float(measure.observer_interpretation),
            'format_flexibility': float(measure.format_flexibility),
            'narrator': measure.narrator_type.value,
            'narrated': measure.narrated_type.value,
            'coupling': float(measure.narrator_narrated_coupling),
            'superficial_potential': float(measure.superficial_potential),
            'actual_potential': float(measure.actual_potential),
            'potential_gap': float(measure.potential_gap),
            'degrees_of_freedom': measure.degrees_of_freedom,
            'key_features': measure.narrative_features_prediction
        })
    
    # Sort by narrativity
    spectrum_data.sort(key=lambda x: x['narrativity'])
    
    return jsonify({'domains': spectrum_data})


@narrativity_bp.route('/api/narrativity/correlation-data')
def get_correlation_data():
    """Get narrativity-α correlation data."""
    return jsonify({
        'correlation': -0.961,
        'relationship': 'inverse',
        'formula': 'α ≈ 1.0 - Narrativity',
        'interpretation': 'Low narrativity (constrained) → High α (plot). High narrativity (open) → Low α (character).'
    })


@narrativity_bp.route('/api/narrativity/domain-detail/<domain_name>')
def get_domain_detail(domain_name):
    """Get detailed narrativity analysis for specific domain."""
    analyzer = NarrivityAnalyzer()
    
    # Would load actual domain description from database
    # For now, return structure
    return jsonify({
        'domain': domain_name,
        'analysis': 'Full narrativity breakdown',
        'components': {
            'structural_openness': 0.5,
            'temporal_freedom': 0.5,
            'actor_agency': 0.5,
            'observer_interpretation': 0.5,
            'format_flexibility': 0.5
        }
    })

