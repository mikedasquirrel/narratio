"""
Variables Page Route

Complete variable reference with plain English and mathematical definitions.
"""

from flask import Blueprint, render_template

variables_bp = Blueprint('variables', __name__)


@variables_bp.route('/variables')
def variables_reference():
    """Complete variable reference page."""
    
    variables = {
        'organism_level': [
            {
                'symbol': 'ж',
                'name': 'Genome / DNA',
                'plain_english': 'The complete information genome for an instance - ALL structured data, relationships, context, and features that determine outcome. NOT just text.',
                'mathematical': 'ж_i ∈ ℝ^k where k ≈ 50-350 features (depends on domain)',
                'example': 'NHL Game: ж = [home_team, away_team, rest_advantage: +2, is_rivalry: true, goalie_quality, betting_odds, cup_history_diff: -7, + 50 more fields]',
                'extracted_by': '50+ universal transformers: nominative, temporal, structural, relational, contextual. See docs/NARRATIVE_DEFINITION.md'
            },
            {
                'symbol': 'ю',
                'name': 'Story Quality',
                'plain_english': 'Aggregate quality score computed from genome. How good is the narrative based on its features?',
                'mathematical': 'ю_i = Σ w_k × ж_k where weights depend on π (domain openness), ю ∈ [0,1]',
                'example': 'Movie with strong structure + cast + timing = ю=0.82 (high). Weak plot + poor timing = ю=0.31 (low)',
                'computed_by': 'Weighted aggregation of genome features. High π domains → weight character/names more. Low π → weight structure/plot more.'
            },
            {
                'symbol': '❊',
                'name': 'Outcome / The Star',
                'plain_english': 'Did they succeed? The actual measured outcome we\'re trying to predict.',
                'mathematical': '❊_i ∈ {0,1} for binary (win/loss) or ℝ for continuous (citations, revenue)',
                'example': 'NHL: ❊=1 (home won), ❊=0 (away won). Supreme Court: ❊=citation_count. Startups: ❊=funding_success',
                'determined_by': 'External reality - market forces, physical performance, judgments, randomness'
            },
            {
                'symbol': 'μ',
                'name': 'Mass',
                'plain_english': 'Importance/stakes of this instance. Higher stakes create stronger gravitational effects.',
                'mathematical': 'μ_i = stakes × context_multiplier, typically μ ∈ [0.5, 3.0]',
                'example': 'NHL playoff game: μ=2.5. Regular season: μ=1.0. Rivalry: μ=1.5',
                'used_for': 'Gravitational calculations - high mass instances have stronger pull on similar instances'
            }
        ],
        'domain_level': [
            {
                'symbol': 'π',
                'name': 'Narrativity / Potential',
                'plain_english': 'Domain openness - how much freedom exists for narratives to vary? Constrained by physics/rules or open for interpretation?',
                'mathematical': 'π ∈ [0,1], π = 0.30×structural + 0.20×temporal + 0.25×agency + 0.15×interpretation + 0.10×format',
                'example': 'Lottery π=0.04 (physics), NBA π=0.49 (skill), Supreme Court π=0.52 (evidence+narrative), Movies π=0.65 (content), Golf π=0.70 (mental)',
                'determines': 'Feature weighting in ю calculation. Prediction potential. Whether Δ can be high.',
                'validated_range': '0.30 (Hurricanes) to 0.70 (Golf) across validated domains'
            },
            {
                'symbol': 'Δ',
                'name': 'Narrative Agency / Advantage',
                'plain_english': 'THE KEY VARIABLE: How much does narrative actually matter in predicting outcomes? The measurable advantage narratives provide.',
                'mathematical': 'Δ = π × |r| × κ (main formula) OR Δ = r_narrative - r_baseline (advantage formulation)',
                'example': 'NHL: Δ=0.306, NFL: Δ varies by context, Movies: strong patterns with 0.40 median effect',
                'interpretation': 'Threshold: Δ/π > 0.5 means narrative matters significantly. Use |r| because positive AND negative correlations both indicate narrative matters.',
                'status': 'PRODUCTION VALIDATED - Core metric across all domains. See analysis/EXECUTIVE_SUMMARY_BACKTEST.md'
            },
            {
                'symbol': 'r',
                'name': 'Correlation',
                'plain_english': 'How strongly does story quality (ю) correlate with outcomes (❊)? Can be positive or negative (both meaningful).',
                'mathematical': 'r = Pearson correlation(ю, ❊), r ∈ [-1, 1]',
                'example': 'Supreme Court: r=0.785 (strong positive). Use |r| in Δ formula.',
                'interpretation': 'Positive: better stories → better outcomes. Negative: stories indicate role (underdog narratives). Both show narrative matters!'
            },
            {
                'symbol': 'κ',
                'name': 'Coupling',
                'plain_english': 'How coupled is the narrator to the narrated? Does the storyteller also judge outcomes?',
                'mathematical': 'κ ∈ [0, 1], κ=1 when narrator=judge (self-rated), κ<1 when external judgment',
                'example': 'Self-rated: κ=1.0 (perfect coupling). Startups: κ=0.3 (external VCs judge). Sports: κ varies by awareness',
                'determines': 'Amplification of narrative effects - higher coupling means narratives matter more'
            }
        ],
        'gravitational': [
            {
                'symbol': 'ф',
                'name': 'Narrative Gravity',
                'plain_english': 'Story-based attraction. Instances with similar narrative structures cluster together.',
                'mathematical': 'ф(i,j) = (μ_i × μ_j × similarity_story(ю_i, ю_j)) / distance_narrative²',
                'creates': 'Story-based clusters: similar genres, arcs, themes, structures',
                'example': 'Movies cluster by genre. Sports cluster by game type (blowout vs close).'
            },
            {
                'symbol': 'ة',
                'name': 'Nominative Gravity / Ta Marbuta',
                'plain_english': 'Name-based attraction. Instances with similar names/brands cluster together. Also used as force in three-force model.',
                'mathematical': 'ة(i,j) = (μ_i × μ_j × similarity_name(names_i, names_j)) / distance_nominative²',
                'creates': 'Name-based clusters: phonetic groupings, semantic families, brand associations',
                'example': 'Hurricanes: feminine names cluster. Teams with similar brand prestige cluster. House #13 avoidance.'
            },
            {
                'symbol': 'ф_net',
                'name': 'Net Gravity',
                'plain_english': 'Combined gravitational pull from both narrative and nominative forces. Can create tension if pulling different directions.',
                'mathematical': 'ф_net = ф + ة (vector sum - can oppose)',
                'determines': 'Actual instance clustering, pattern formation, phylogenetic relationships in latent space'
            }
        ],
        'three_forces': [
            {
                'symbol': 'θ',
                'name': 'Theta / Awareness Resistance',
                'plain_english': 'Conscious resistance to narrative effects. Free will, skepticism, meta-awareness that counters narrative influence.',
                'mathematical': 'θ ∈ [0,1], θ = education × (field_awareness + obviousness) × social_cost',
                'example': 'Golf: θ=0.573 (high mental game awareness). NBA: θ=0.500 (baseline). Hurricanes: θ=0.376 (low public awareness)',
                'role': 'RESISTANCE force - higher θ suppresses narrative effects (except in prestige domains where it flips to amplification)',
                'discovered_by': 'Phase 7 transformer extraction - now measured across all domains'
            },
            {
                'symbol': 'λ',
                'name': 'Lambda / Fundamental Constraints',
                'plain_english': 'Physical, skill, or resource barriers that constrain outcomes regardless of narrative.',
                'mathematical': 'λ ∈ [0,1], λ = (training_required + aptitude_threshold + economic_barriers) / 3',
                'example': 'Golf: λ=0.689 (elite skill required). NBA: λ=0.500 (baseline). Mental Health: λ=0.508 (training needed)',
                'role': 'CONSTRAINT force - higher λ means physics/skill dominates over narrative',
                'discovered_by': 'Phase 7 transformer extraction - measures domain accessibility'
            }
        ],
        'theoretical': [
            {
                'symbol': 'Ξ',
                'name': 'The Golden Narratio',
                'plain_english': 'The theoretical perfect narrative. Universal archetypal pattern that optimal stories approximate.',
                'mathematical': 'Ξ ∈ ℝ^k, theoretical ideal estimated from winners: Ξ ≈ mean(ж_winners)',
                'status': 'Cannot directly observe. Inferred from patterns in successful instances.',
                'hypothesis': 'Better stories approximate Ξ within domain constraints (π, θ, λ). Distance from Ξ predicts success probability.'
            }
        ]
    }
    
    return render_template('variables.html', variables=variables)

