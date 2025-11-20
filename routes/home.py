"""Home route - CLEANED to show only validated discoveries."""

from flask import Blueprint, render_template

home_bp = Blueprint('home', __name__)

@home_bp.route('/')
def index():
    """Clean home page showing framework and validated domains."""
    
    # Framework story links (NEW - Complete three-force model)
    framework_links = [
        {
            'name': 'The Framework Story',
            'description': 'Complete narrative journey from "Do better stories win?" to three-realm model',
            'url': '/framework-story',
            'highlight': True
        },
        {
            'name': 'Three-Force Explorer',
            'description': 'Interactive analysis of ة, θ, λ across all domains',
            'url': '/framework-explorer'
        },
        {
            'name': 'Three Realms Visual',
            'description': 'Beautiful visualization of where physics, meaning, and consciousness meet',
            'url': '/three-realms'
        },
        {
            'name': 'Quick Reference',
            'description': 'One-page guide with all variables and equations',
            'url': '/framework-quickref'
        }
    ]
    
    # Key Statistics (Updated November 17, 2025 - VALIDATED ONLY)
    stats = {
        'total_domains': 9,  # NHL, NFL, NBA, Golf, Hurricanes, Movies, IMDB, Supreme Court, Startups
        'total_entities': '16,183',  # Validated entities: 15925 + 258 startups
        'spectrum_min': 0.30,  # Hurricanes (lowest validated)
        'spectrum_max': 0.76,  # Startups (highest validated)
        'validated_sports': 4,  # NHL, NFL, NBA, Golf
        'validated_betting': 3,  # NHL, NFL, NBA
        'top_roi': '32.5%',  # NHL
        'best_win_rate': '69.4%',  # NHL
        'investor_ready': True,  # Investor package complete
        'games_tested': 4294  # Total holdout games tested
    }
    
    # Featured Discoveries (VALIDATED ONLY - Nov 17, 2025)
    featured = [
        {
            'name': 'NHL Betting (32.5% ROI)',
            'metric': 'Win Rate = 69.4%',
            'highlight': 'PRODUCTION READY',
            'finding': '2,779 games validated on 2024-25 holdout data. 32.5% ROI at ≥65% confidence. Meta-Ensemble + GBM models with 79 features.',
            'url': '/nhl',
            'color': 'green'
        },
        {
            'name': 'NFL Betting (27.3% ROI)',
            'metric': 'Win Rate = 66.7%',
            'highlight': 'CONTEXTUAL EDGE',
            'finding': 'QB Edge + Home Dog pattern validated on 2024 holdout. Market inefficiency in contrarian contexts.',
            'url': '/nfl',
            'color': 'green'
        },
        {
            'name': 'Supreme Court (r = 0.785)',
            'metric': 'R² = 61.6%',
            'highlight': 'LEGAL NARRATIVE',
            'finding': 'Narrative quality predicts citation impact. 26 opinions analyzed. Moderately narrative (π=0.52) in evidence-based system.',
            'url': '/supreme-court',
            'color': 'purple'
        },
        {
            'name': 'Movies (20 Patterns)',
            'metric': 'Effect = 0.40 median',
            'highlight': 'ENTERTAINMENT',
            'finding': '2,000 films with 20 distinct narrative patterns. Strong effects (-0.899 to +0.631). Plot structure matters.',
            'url': '/movie-results',
            'color': 'orange'
        },
        {
            'name': 'Golf (7% Effect)',
            'metric': '20 Patterns',
            'highlight': 'INDIVIDUAL SPORT',
            'finding': '5,000 tournaments analyzed. π=0.70 with moderate narrative effects. Mental game awareness present.',
            'url': '/golf-results',
            'color': 'orange'
        },
        {
            'name': 'Hurricanes (Dual π)',
            'metric': 'Storm 0.30 / Response 0.68',
            'highlight': 'NOMINATIVE EFFECTS',
            'finding': '819 hurricanes. Name effects on evacuation confirmed. Validates nominative determinism in life/death decisions.',
            'url': '/hurricanes-results',
            'color': 'purple'
        },
        {
            'name': 'Startups (π=0.76)',
            'metric': 'Effect ~0.13',
            'highlight': 'MARGINAL',
            'finding': '258 companies. 4 patterns found (expected 21). Small sample limits conclusions. Needs larger dataset.',
            'url': '/startups-results',
            'color': 'orange'
        }
    ]
    
    # Validated domains ONLY (Nov 17, 2025 - 9 domains validated)
    domains = [
        {
            'name': 'Golf',
            'metric': 'π = 0.70',
            'stats': '5,000 tournaments, 20 patterns, median effect 0.07',
            'key_finding': 'Moderate narrative effects in individual sport. Mental game awareness detected.',
            'url': '/golf-results'
        },
        {
            'name': 'Movies / IMDB',
            'metric': 'π = 0.65',
            'stats': '2,000 films, 20 patterns, median effect 0.40',
            'key_finding': 'Strong narrative effects in entertainment. Success pattern: +12.8%, Failure pattern: -17.6%.',
            'url': '/movie-results'
        },
        {
            'name': 'NFL',
            'metric': 'Win Rate = 66.7%',
            'stats': 'π=0.57, ROI 27.3%, validated 2024',
            'key_finding': 'QB Edge + Home Dog pattern. Contrarian market inefficiency. PRODUCTION BETTING SYSTEM.',
            'url': '/nfl-results'
        },
        {
            'name': 'Supreme Court',
            'metric': 'r = 0.785',
            'stats': 'π=0.52, Δ=0.306, R²=61.6%',
            'key_finding': 'Narrative quality predicts citations. Moderate narrativity in adversarial legal system.',
            'url': '/supreme-court'
        },
        {
            'name': 'NBA',
            'metric': 'Win Rate = 54.5%',
            'stats': 'π=0.49, ROI 7.6%, validated 2023-24',
            'key_finding': 'Elite Team + Close Game pattern. Small edge in efficient market. VALIDATED BETTING SYSTEM.',
            'url': '/nba-results'
        },
        {
            'name': 'Hurricanes',
            'metric': 'Dual π',
            'stats': 'Storm 0.30 / Response 0.68, n=819',
            'key_finding': 'Name effects on evacuation. Validates nominative determinism in life/death decisions.',
            'url': '/hurricanes-results'
        },
        {
            'name': 'NHL',
            'metric': 'Win Rate = 69.4%',
            'stats': 'π=0.52, ROI 32.5%, validated 2024-25',
            'key_finding': 'Highest validated ROI. 79 features. PRODUCTION BETTING SYSTEM.',
            'url': '/nhl'
        },
        {
            'name': 'Startups',
            'metric': 'Effect = 0.13',
            'stats': 'π=0.76, n=258, 4 patterns ⚠️',
            'key_finding': 'Small sample, too few patterns. MARGINAL - needs larger dataset for robust validation.',
            'url': '/startups-results'
        }
    ]
    
    return render_template('home_clean.html', 
                          domains=domains, 
                          framework_links=framework_links,
                          stats=stats,
                          featured=featured)
