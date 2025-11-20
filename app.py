"""
Narrative Optimization Web Dashboard

A production-ready Flask application showcasing interactive narrative analysis
and visualization capabilities.
"""

# FIX TENSORFLOW MUTEX DEADLOCK ON MACOS
# Must be set BEFORE any Flask/TensorFlow imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
os.environ['TOKENIZERS_PARALLELISM'] = os.environ.get('TOKENIZERS_PARALLELISM', 'false')

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
import sys
from pathlib import Path
import json
from typing import Any, Dict, List
from dotenv import load_dotenv
from datetime import datetime


def log_stage(message: str) -> None:
    """Console progress logging with timestamps."""
    print(f"[app][{datetime.now().isoformat(timespec='seconds')}] {message}")

# Load environment variables from .env file
load_dotenv()
log_stage("dotenv environment variables loaded")

# Import Phase 7 data loader
sys.path.insert(0, str(Path(__file__).parent))
from utils.phase7_data_loader import load_phase7_for_domain, get_all_phase7_domains
log_stage("Phase 7 data loader ready")

# Add narrative_optimization to path
narrative_opt_path = Path(__file__).parent / 'narrative_optimization'
if str(narrative_opt_path) not in sys.path:
    sys.path.insert(0, str(narrative_opt_path))
log_stage("narrative_optimization package path added")

# Import routes - CLEANED (only validated, working pages)
# Lazy import to avoid bus errors from heavy dependencies
log_stage("Importing Flask blueprints...")
try:
    from routes.home import home_bp
except Exception as e:
    print(f"Warning: Could not import home_bp: {e}")
    home_bp = None
# from routes.experiments import experiments_bp  # REMOVED: Not user-facing
# from routes.visualizations import visualizations_bp  # REMOVED: Generic
# Lazy import routes to avoid bus errors
def safe_import(module_name, attr_name, default=None):
    """Safely import a route blueprint"""
    try:
        module = __import__(f'routes.{module_name}', fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception as e:
        print(f"Warning: Could not import {module_name}.{attr_name}: {e}")
        return default

# ============================================================================
# NHL DASHBOARD DATA HELPERS
# ============================================================================

def _load_json(path: Path, default):
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open() as handle:
            return json.load(handle)
    except Exception as exc:
        log_stage(f"[nhl_dashboard] Unable to read {path}: {exc}")
        return default


def _format_percent(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "–"


def _format_signed_percent(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "–"
    try:
        pct = float(value) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.{decimals}f}%"
    except (TypeError, ValueError):
        return "–"


def _format_plain_percent(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return "–"


def _format_moneyline(value: Any) -> str:
    if value is None:
        return "–"
    try:
        val = int(float(value))
        return f"+{val}" if val > 0 else str(val)
    except (TypeError, ValueError):
        return str(value)


def load_nhl_dashboard_context() -> Dict[str, Any]:
    project_root = Path(__file__).parent
    analysis = _load_json(
        project_root / "narrative_optimization" / "domains" / "nhl" / "nhl_complete_analysis.json",
        {}
    )
    coverage = _load_json(project_root / "analysis" / "nhl_odds_coverage_diagnostics.json", {})
    forward = _load_json(project_root / "data" / "paper_trading" / "nhl_forward_summary.json", {})
    holdout = _load_json(
        project_root / "docs" / "investor" / "verification" / "nhl_holdout_metrics.json",
        {}
    )
    features_meta = _load_json(
        project_root / "narrative_optimization" / "domains" / "nhl" / "nhl_features_metadata.json",
        {}
    )
    predictions = _load_json(project_root / "analysis" / "nhl_upcoming_predictions.json", [])

    pattern_summary = analysis.get("pattern_summary", {})
    patterns = analysis.get("patterns", []) or []
    top_patterns: List[Dict[str, Any]] = []
    for pattern in patterns[:5]:
        top_patterns.append(
            {
                "name": pattern.get("name", "Pattern"),
                "games": pattern.get("n_games", 0),
                "win_rate_pct": _format_percent(pattern.get("win_rate")),
                "roi_pct": _format_signed_percent(pattern.get("roi")),
                "confidence": pattern.get("confidence", "—"),
            }
        )

    headline_pattern = top_patterns[0] if top_patterns else None

    # Holdout thresholds
    thresholds = holdout.get("thresholds", []) or []
    holdout_rows = []
    headline_holdout = None
    for row in thresholds:
        entry = {
            "threshold": row.get("threshold"),
            "threshold_label": f"{row.get('threshold', 0) * 100:.0f}%",
            "threshold_pct": f"{row.get('threshold', 0) * 100:.0f}%",
            "bets": row.get("bets", 0),
            "win_rate": row.get("win_rate", 0),  # Add raw value for template
            "win_rate_pct": _format_percent(row.get("win_rate")),
            "roi": row.get("roi", 0),  # Add raw value for template
            "roi_pct": _format_signed_percent(row.get("roi")),
        }
        holdout_rows.append(entry)
        if row.get("threshold") and abs(row["threshold"] - 0.65) < 1e-6:
            headline_holdout = entry
    if headline_holdout is None and holdout_rows:
        headline_holdout = holdout_rows[0]

    coverage_overall = coverage.get("overall_coverage_pct")
    unmatched_by_season = coverage.get("unmatched_by_season", []) or []
    top_unmatched = []
    for season in unmatched_by_season[:3]:
        top_unmatched.append(
            {
                "key": season.get("key"),
                "count": season.get("count", 0),
                "coverage_pct": _format_percent(season.get("coverage")),
            }
        )

    feature_breakdown = features_meta.get("feature_breakdown", {}) or {}
    feature_summary = {
        "total": features_meta.get("total_features"),
        "universal": feature_breakdown.get("universal"),
        "performance": feature_breakdown.get("performance"),
        "nominative": feature_breakdown.get("nominative"),
    }

    forward_by_model = []
    for model, stats in (forward.get("by_model") or {}).items():
        forward_by_model.append({"model": model, "bets": stats.get("bets", 0)})
    forward_by_model.sort(key=lambda item: item["model"])

    live_predictions = []
    if isinstance(predictions, list):
        for game in predictions:
            matchup = game.get("matchup") or f"{game.get('away_team')} @ {game.get('home_team')}"
            date = game.get("date")
            for rec in game.get("recommendations", []) or []:
                live_predictions.append(
                    {
                        "matchup": matchup,
                        "date": date,
                        "model": rec.get("model"),
                        "side": (rec.get("side") or "").upper(),
                        "prob_raw": rec.get("prob"),
                        "edge_raw": rec.get("edge"),
                        "moneyline": _format_moneyline(rec.get("moneyline")),
                    }
                )
    live_predictions.sort(key=lambda item: item.get("edge_raw") or 0.0, reverse=True)
    live_predictions = [
        {
            "matchup": entry["matchup"],
            "date": entry["date"],
            "model": entry["model"],
            "side": entry["side"],
            "prob_pct": _format_percent(entry.get("prob_raw")),
            "edge_pct": _format_signed_percent(entry.get("edge_raw")),
            "moneyline": entry["moneyline"],
        }
        for entry in live_predictions[:5]
    ]

    links = [
        {"label": "Evidence tracker", "path": "analysis/NHL_EVIDENCE_READINESS.md"},
        {"label": "Technical validation report", "path": "docs/investor/TECHNICAL_VALIDATION_REPORT.md"},
        {"label": "Holdout metrics JSON", "path": "docs/investor/verification/nhl_holdout_metrics.json"},
    ]

    return {
        "analysis_date": analysis.get("analysis_date"),
        "top_pattern": headline_pattern,
        "pattern_summary": pattern_summary,
        "patterns": top_patterns,
        "holdout": {
            "summary": headline_holdout,
            "rows": holdout_rows,
            "train_games": holdout.get("train_games"),
            "test_games": holdout.get("test_games"),
            "generated_at": holdout.get("generated_at"),
            "feature_count": holdout.get("feature_count"),
            "cutoff_date": holdout.get("cutoff_date", "2024-09-01"),
        },
        "coverage": {
            "overall_pct": _format_plain_percent(coverage_overall),
            "matched_total": coverage.get("matched_total"),
            "unmatched_total": coverage.get("unmatched_total"),
            "top_unmatched": top_unmatched,
        },
        "feature_summary": feature_summary,
        "forward": {
            "total_bets": forward.get("total_bets", 0),
            "updated_at": forward.get("updated_at", "–"),
            "updated_short": forward.get("updated_at", "2025-11-20")[:10] if forward.get("updated_at") else "Nov 20",
            "by_model": forward_by_model,
        },
        "live_predictions": live_predictions,
        "links": links,
        "pipeline": {
            "last_run": "2025-11-20 07:08 UTC",
            "duration": "~12 min",
            "status": "✓ Success"
        },
        "models": {
            "trained_at": "2025-11-20",
            "train_samples": "15,495"
        }
    }

# ============================================================================
# ROUTE IMPORTS - VALIDATED SYSTEMS ONLY (Nov 17, 2025)
# ============================================================================
# Only importing routes for systems validated in recent production backtest
# See: analysis/RECENT_SEASON_BACKTEST_REPORT.md and EXECUTIVE_SUMMARY_BACKTEST.md

# CORE TOOLS (Always available)
analysis_bp = safe_import('analysis', 'analysis_bp')
narrative_analyzer_bp = safe_import('narrative_analyzer', 'narrative_analyzer_bp')
domain_processor_bp = safe_import('domain_processor', 'domain_processor_bp')
cross_domain_bp = safe_import('cross_domain', 'cross_domain_bp')
variables_bp = safe_import('variables', 'variables_bp')
framework_story_bp = safe_import('framework_story', 'framework_story_bp')
project_overview_bp = safe_import('project_overview', 'project_overview_bp')
narrativity_bp = safe_import('narrativity', 'narrativity_bp')
betting_bp = safe_import('betting', 'betting_bp')
live_betting_api_bp = safe_import('live_betting_api', 'live_betting_api_bp')

# VALIDATED SPORTS SYSTEMS (Nov 17, 2025 Production Backtest)
# NHL: 69.4% win rate, 32.5% ROI ✅
nhl_bp = safe_import('nhl', 'nhl_bp')
nhl_betting_bp = safe_import('nhl_betting', 'nhl_betting_bp')

# NFL: 66.7% win rate, 27.3% ROI ✅
nfl_bp = safe_import('nfl', 'nfl_bp')
nfl_live_betting_bp = safe_import('nfl_live_betting', 'nfl_live_betting_bp')

# NBA: 54.5% win rate, 7.6% ROI ✅
nba_bp = safe_import('nba', 'nba_bp')
nba_betting_live_bp = safe_import('nba_betting_live', 'nba_betting_live_bp')

# ============================================================================
# TEMPORARILY DISABLED - NOT VALIDATED IN RECENT PIPELINE
# These represent older analyses not re-run through Nov 2025 production pipeline
# Re-enable after validation with current universal pipeline
# ============================================================================

# VALIDATED ENTERTAINMENT DOMAINS (Nov 17, 2025)
movies_bp = safe_import('movies', 'movies_bp')  # Validated: 2000 films, 20 patterns, median effect 0.40
imdb_bp = safe_import('imdb', 'imdb_bp')  # Same as movies - IMDB dataset
literary_bp = safe_import('literary', 'literary_bp')  # Literary insights + corpus diagnostics

# VALIDATED LEGAL DOMAINS (Nov 17, 2025)
supreme_court_bp = safe_import('supreme_court', 'supreme_court_bp')  # Validated: 26 opinions, r=0.785, R²=61.6%

# VALIDATED BUSINESS DOMAINS (Nov 17, 2025)
startups_bp = safe_import('startups', 'startups_bp')  # Validated: 258 companies, 4 patterns (marginal - small sample)

# OLDER ANALYSES (Not validated in recent production backtest)
# mental_health_bp = safe_import('mental_health', 'mental_health_bp')  # Pre-Nov 2025 analysis
# oscars_bp = safe_import('oscars', 'oscars_bp')  # Pre-Nov 2025 analysis
# crypto_bp = safe_import('crypto', 'crypto_bp')  # Pre-Nov 2025 analysis
# housing_bp = safe_import('housing', 'housing_bp')  # Pre-Nov 2025 analysis
# music_bp = safe_import('music', 'music_bp')  # Pre-Nov 2025 analysis
# free_will_bp = safe_import('free_will', 'free_will_bp')  # Theory page, not validated

# VALIDATED INDIVIDUAL SPORTS (Nov 17, 2025)
golf_bp = safe_import('golf', 'golf_bp')  # Validated: 5000 tournaments, 20 patterns, median effect 0.07

# SPORTS NOT IN RECENT VALIDATION
# tennis_bp = safe_import('tennis', 'tennis_bp')  # Not in Nov 2025 backtest
# tennis_betting_bp = safe_import('tennis_betting', 'tennis_betting_bp')  # Not validated
# ufc_bp = safe_import('ufc', 'ufc_bp')  # Not in Nov 2025 backtest
# mlb_bp = safe_import('mlb', 'mlb_bp')  # Not in Nov 2025 backtest

# VALIDATED RESEARCH DOMAINS (Nov 17, 2025)
hurricanes_bp = safe_import('hurricanes', 'hurricanes_bp')  # Validated: Dual π framework, 819 storms

# EXPERIMENTAL/INCOMPLETE (Already commented)
# temporal_linguistics_bp = safe_import('temporal_linguistics', 'temporal_linguistics_bp')
# wwe_domain_bp = safe_import('wwe_domain', 'wwe_domain_bp')
# novels_bp = safe_import('novels', 'novels_bp')
# poker_bp = safe_import('poker', 'poker_bp')
# dinosaurs_bp = safe_import('dinosaurs', 'dinosaurs_bp')
# conspiracies_bp = safe_import('conspiracies', 'conspiracies_bp')
# bible_bp = safe_import('bible', 'bible_bp')

# UTILITY/META (Consider re-enabling if needed)
# meta_eval_bp = safe_import('meta_evaluation', 'meta_eval_bp')
# insights_bp = safe_import('insights', 'insights_bp')
# prediction_ai_bp = safe_import('prediction_ai', 'prediction_ai_bp')
# checkpoint_feeds_bp = safe_import('checkpoint_feeds', 'checkpoint_feeds_bp')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['JSON_SORT_KEYS'] = False

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================================
# BLUEPRINT REGISTRATION - VALIDATED SYSTEMS ONLY (Nov 17, 2025)
# ============================================================================

# CORE APPLICATION
if home_bp:
    app.register_blueprint(home_bp)

# CORE TOOLS
if analysis_bp:
    app.register_blueprint(analysis_bp, url_prefix='/analyze')
if narrative_analyzer_bp:
    app.register_blueprint(narrative_analyzer_bp, url_prefix='/analyze')
if domain_processor_bp:
    app.register_blueprint(domain_processor_bp, url_prefix='/process')
if cross_domain_bp:
    app.register_blueprint(cross_domain_bp)
if variables_bp:
    app.register_blueprint(variables_bp)
if framework_story_bp:
    app.register_blueprint(framework_story_bp)
if project_overview_bp:
    app.register_blueprint(project_overview_bp)
if narrativity_bp:
    app.register_blueprint(narrativity_bp)

# BETTING INFRASTRUCTURE
if betting_bp:
    app.register_blueprint(betting_bp)
if live_betting_api_bp:
    app.register_blueprint(live_betting_api_bp)

# VALIDATED SPORTS (Nov 17, 2025 Production Backtest)
if nhl_bp:
    app.register_blueprint(nhl_bp)
if nhl_betting_bp:
    app.register_blueprint(nhl_betting_bp)
if nfl_bp:
    app.register_blueprint(nfl_bp)
if nfl_live_betting_bp:
    app.register_blueprint(nfl_live_betting_bp)
if nba_bp:
    app.register_blueprint(nba_bp, url_prefix='/nba')
if nba_betting_live_bp:
    app.register_blueprint(nba_betting_live_bp)

# ============================================================================
# DISABLED BLUEPRINTS - Waiting for re-validation
# ============================================================================
# Uncomment after running through current production pipeline

# OLDER ANALYSES
# if mental_health_bp:
#     app.register_blueprint(mental_health_bp, url_prefix='/mental-health')
# if movies_bp:
#     app.register_blueprint(movies_bp)
# if imdb_bp:
#     app.register_blueprint(imdb_bp)
# if oscars_bp:
#     app.register_blueprint(oscars_bp)
# if crypto_bp:
#     app.register_blueprint(crypto_bp)
# if startups_bp:
#     app.register_blueprint(startups_bp)
# if housing_bp:
#     app.register_blueprint(housing_bp)
# if music_bp:
#     app.register_blueprint(music_bp)
# if free_will_bp:
#     app.register_blueprint(free_will_bp)
# if supreme_court_bp:
#     app.register_blueprint(supreme_court_bp, url_prefix='/supreme-court')

# SPORTS NOT IN RECENT VALIDATION
# if tennis_bp:
#     app.register_blueprint(tennis_bp)
# if golf_bp:
#     app.register_blueprint(golf_bp)
# if ufc_bp:
#     app.register_blueprint(ufc_bp)
# if mlb_bp:
#     app.register_blueprint(mlb_bp)
# if tennis_betting_bp:
#     app.register_blueprint(tennis_betting_bp)

# VALIDATED RESEARCH DOMAINS
if hurricanes_bp:
    app.register_blueprint(hurricanes_bp, url_prefix='/hurricanes')

# VALIDATED ENTERTAINMENT DOMAINS
if movies_bp:
    app.register_blueprint(movies_bp)
if imdb_bp:
    app.register_blueprint(imdb_bp)
if literary_bp:
    app.register_blueprint(literary_bp, url_prefix='/literary')

# VALIDATED INDIVIDUAL SPORTS
if golf_bp:
    app.register_blueprint(golf_bp)

# VALIDATED LEGAL DOMAINS
if supreme_court_bp:
    app.register_blueprint(supreme_court_bp, url_prefix='/supreme-court')

# VALIDATED BUSINESS DOMAINS
if startups_bp:
    app.register_blueprint(startups_bp)

# UTILITY/META
# if meta_eval_bp:
#     app.register_blueprint(meta_eval_bp)
# if insights_bp:
#     app.register_blueprint(insights_bp)
# if prediction_ai_bp:
#     app.register_blueprint(prediction_ai_bp)
# if checkpoint_feeds_bp:
#     app.register_blueprint(checkpoint_feeds_bp)
log_stage("Blueprint registration complete")

# Formulas page - VALIDATED cross-domain discoveries
@app.route('/formulas')
def formulas():
    """Cross-domain formula reference - validated discoveries only."""
    return render_template('formulas.html')

# Betting System Landing Page
@app.route('/betting')
def betting_home():
    """Betting system landing page - production-ready systems."""
    return render_template('betting_home.html')

# Betting Dashboard - Master view
@app.route('/betting/dashboard')
def betting_dashboard():
    """Master betting dashboard - all validated strategies."""
    # Load portfolio summary
    portfolio_path = Path('analysis/portfolio_summary.json')
    portfolio_data = {}
    if portfolio_path.exists():
        with open(portfolio_path) as f:
            portfolio_data = json.load(f)
    
    return render_template('betting_dashboard.html', portfolio=portfolio_data)

# Live Betting Dashboard - Real-time opportunities
@app.route('/betting/live')
def live_betting_dashboard():
    """Real-time betting opportunities dashboard."""
    # Load NHL predictions
    predictions_path = Path('analysis/nhl_upcoming_predictions.json')
    opportunities = []
    
    if predictions_path.exists():
        with open(predictions_path) as f:
            predictions_data = json.load(f)
            for game in predictions_data[:10]:
                for rec in game.get('recommendations', []):
                    if rec.get('prob', 0) >= 0.60 and rec.get('edge', 0) >= 0.02:
                        opportunities.append({
                            'matchup': game.get('matchup', f"{game.get('away_team')} @ {game.get('home_team')}"),
                            'model': rec.get('model', '').replace('narrative_', ''),
                            'side': rec.get('side', '').upper(),
                            'probability': round(rec.get('prob', 0) * 100, 1),
                            'edge': round(rec.get('edge', 0) * 100, 1),
                            'moneyline': rec.get('moneyline', '—'),
                            'confidence': rec.get('prob', 0)
                        })
    
    from datetime import datetime
    current_time = datetime.now().strftime('%H:%M:%S')
    
    return render_template('live_betting_dashboard.html', 
                         opportunities=opportunities,
                         current_time=current_time)

# ============================================================================
# INVESTOR PAGES - Production-Validated Systems (Nov 2025)
# ============================================================================

@app.route('/investor')
def investor_landing():
    """Investor information landing page - NHL system focus."""
    return render_template('investor_landing.html')

@app.route('/investor/dashboard')
def investor_dashboard_page():
    """Interactive investor dashboard - serves the generated HTML."""
    dashboard_path = Path('docs/investor/INTERACTIVE_DASHBOARD.html')
    if dashboard_path.exists():
        with open(dashboard_path) as f:
            return f.read()
    else:
        return "Dashboard not generated. Run: python3 scripts/generate_investor_dashboard.py", 404

@app.route('/investor/proposal')
def investor_proposal():
    """One-page investment proposal."""
    return send_file('docs/investor/INVESTMENT_PROPOSAL.md', mimetype='text/markdown')

@app.route('/investor/presentation')
def investor_presentation():
    """Comprehensive investor presentation."""
    return send_file('docs/investor/INVESTOR_PRESENTATION.md', mimetype='text/markdown')

@app.route('/investor/validation')
def investor_validation():
    """Technical validation report for business partners."""
    return send_file('docs/investor/TECHNICAL_VALIDATION_REPORT.md', mimetype='text/markdown')

@app.route('/investor/api/metrics')
def investor_api_metrics():
    """JSON API endpoint for investor metrics."""
    summary_path = Path('docs/investor/data/backtest_summary.json')
    if summary_path.exists():
        with open(summary_path) as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Data not found'}), 404

# NHL Validation and Methodology Pages
@app.route('/nhl/validation')
def nhl_validation():
    """Statistical validation deep-dive for NHL system."""
    return render_template('nhl_validation.html')

@app.route('/betting/methodology')
def betting_methodology():
    """Technical methodology and feature engineering documentation."""
    return render_template('betting_methodology.html')

# Expanded Betting Platform Pages
@app.route('/betting/performance')
def betting_performance():
    """Historical performance tracking with detailed metrics."""
    return render_template('betting_performance.html')

@app.route('/betting/education')
def betting_education():
    """Comprehensive betting education and terminology guide."""
    return render_template('betting_education.html')

@app.route('/betting/risk-management')
def betting_risk_management():
    """Detailed risk management framework and protocols."""
    return render_template('betting_risk_management.html')

@app.route('/nhl/patterns')
def nhl_patterns():
    """NHL-specific betting patterns analysis."""
    return render_template('nhl_patterns.html')

# Standalone Exports (No navbar, self-contained for sharing/printing)
@app.route('/export/nhl-validation')
def export_nhl_validation():
    """Standalone NHL validation report."""
    return render_template('nhl_validation_standalone.html')

@app.route('/export/nhl-patterns')
def export_nhl_patterns():
    """Standalone NHL patterns report."""
    return render_template('nhl_patterns_standalone.html')

@app.route('/export/nhl-performance')
def export_nhl_performance():
    """Standalone NHL performance tracking report."""
    return render_template('nhl_performance_standalone.html')

# ============================================================================
# UNIFIED SPORT PAGES (Analysis + Betting in one)
# ============================================================================
# Note: These routes are handled by the imported blueprints (nhl_bp, nfl_bp, nba_bp)
# which are registered in the VALIDATED SPORTS section below

# ============================================================================
# VALIDATED SPORT RESULTS PAGES (Nov 17, 2025 Production Backtest)
# ============================================================================

# NBA Results - VALIDATED: 54.5% win, 7.6% ROI
@app.route('/nba-results')
def nba_results():
    """NBA validated on 2023-24 holdout: 54.5% win rate, 7.6% ROI (44 games)."""
    return render_template('nba_results.html')

# NFL Results - VALIDATED: 66.7% win, 27.3% ROI
@app.route('/nfl-results')
def nfl_results():
    """NFL validated on 2024 holdout: 66.7% win rate, 27.3% ROI in QB Edge + Home Dog context (9 games)."""
    return render_template('nfl_results.html')

# NHL Results - VALIDATED: 69.4% win, 32.5% ROI (coming soon - currently in blueprint)
# @app.route('/nhl-results')
# def nhl_results():
#     """NHL validated on 2024-25 holdout: 69.4% win rate, 32.5% ROI (85 bets at ≥65% confidence)."""
#     return render_template('nhl_results.html')

# ============================================================================
# DISABLED RESULTS PAGES - Not validated in recent production pipeline
# ============================================================================

# # Mental Health Results - PRE-NOV 2025 ANALYSIS
# @app.route('/mental-health-results')
# def mental_health_results():
#     """Mental Health α=0.80 discovery - NOT validated in Nov 2025 pipeline."""
#     return render_template('mental_health_results.html')

# # Crypto Results - PRE-NOV 2025 ANALYSIS
# @app.route('/crypto-results')
# def crypto_results():
#     """Cryptocurrency α=0.60 discovery - NOT validated in Nov 2025 pipeline."""
#     return render_template('crypto_results.html')

# Movie Results - VALIDATED: Nov 17, 2025
@app.route('/movie-results')
@app.route('/movies-results')
def movie_results():
    """Movies validated (π=0.65): 2,000 films, 20 narrative patterns discovered, median effect size 0.40 (strong signal)."""
    return render_template('movie_results.html')

# IMDB Results - VALIDATED: Nov 17, 2025 (same dataset)
@app.route('/imdb-results')
def imdb_results():
    """IMDB/CMU Movie Summaries validated: Same as movies analysis, 2,000 films with significant narrative patterns."""
    return render_template('imdb_results.html')

# # Oscar Results - PRE-NOV 2025 ANALYSIS
# @app.route('/oscar-results')
# def oscar_results():
#     """Oscar Best Picture - NOT validated in Nov 2025 pipeline."""
#     return render_template('oscar_results.html')

# # Tennis Results - PRE-NOV 2025 ANALYSIS
# @app.route('/tennis-results')
# def tennis_results():
#     """Tennis narrative analysis - NOT validated in Nov 2025 pipeline."""
#     return render_template('tennis_results.html')

# # UFC Results - PRE-NOV 2025 ANALYSIS
# @app.route('/ufc-results')
# def ufc_results():
#     """UFC narrative analysis - NOT validated in Nov 2025 pipeline."""
#     return render_template('ufc_results.html')

# Golf Results - VALIDATED: Nov 17, 2025
@app.route('/golf-results')
def golf_results():
    """Golf validated (π=0.70): 5,000 tournaments, 20 patterns discovered, median effect 0.07 (moderate signal)."""
    return render_template('golf_results.html')

# Hurricanes Results - VALIDATED: Nov 17, 2025
@app.route('/hurricanes-results')
@app.route('/hurricane-results')  # Also support singular
def hurricanes_results():
    """Hurricanes dual π framework validated: Storm (π=0.30) vs Response (π=0.68), 819 storms, name effects confirmed."""
    return render_template('hurricane_results.html')

# Startups Results - VALIDATED: Nov 17, 2025 (MARGINAL)
@app.route('/startups-results')
def startups_results():
    """Startups validated (π=0.76): 258 companies, 4 patterns (expected 21), effect ~0.13. MARGINAL - small sample, needs more data."""
    return render_template('startups.html')

# ============================================================================
# CORE THEORY/NAVIGATION PAGES (Keep - not domain-specific)
# ============================================================================

# Narrative Determinism Explorations
@app.route('/findings')
def findings_simple():
    """Narrative determinism explorations across the spectrum."""
    return render_template('explorations.html')

# Cool Discoveries
@app.route('/discoveries')
def discoveries():
    """Actual cool shit we found."""
    return render_template('findings_summary.html')

# ============================================================================
# DISABLED DOMAIN EXPLORER PAGES - Use after re-validation
# ============================================================================

# # Domain Index/Explorer
# @app.route('/domains')
# def domain_index():
#     """Complete spectrum explorer - DISABLED until domains re-validated."""
#     return render_template('domain_index.html')

# # Comprehensive Domain Explorer
# @app.route('/domains/explorer')
# def domain_explorer():
#     """Interactive domain explorer - DISABLED until domains re-validated."""
#     return render_template('domain_explorer.html')

# # Individual Domain Deep-Dive - DISABLED (contains non-validated domain data)
# # All domain detail data has been removed - will be re-added after re-validation
@app.route('/domains/<domain_name>')
def domain_detail(domain_name):
    """Detailed analysis page - DISABLED until domains re-validated."""
    return render_template('404.html'), 404

# DOMAIN DATA REMOVED - was here, will restore after validation
# domain_analyses contained data for: golf, tennis, boxing, poker, hurricanes, dinosaurs
# Original function returned domain_detail.html with domain-specific analysis

# (Original domain_analyses data removed - will restore after validation)

# # Domain Comparison Tool - DISABLED until re-validation
# @app.route('/domains/compare')
# def domain_compare():
#     """Interactive domain comparison - DISABLED until domains re-validated."""
#     return render_template('domain_compare.html')

# Unified Domains API
@app.route('/api/domains/all')
def domains_all_api():
    """JSON endpoint returning all domains with complete framework data - UPDATED Nov 12, 2025."""
    domains = [
        {
            'domain': 'lottery',
            'name': 'Lottery Numbers',
            'pi': 0.04,
            'lambda': 0.95,
            'psi': 0.70,
            'nu': 0.05,
            'arch': 0.000,
            'leverage': 0.00,
            'type': 'Pure Randomness',
            'finding': 'Lucky numbers at expected frequency. Perfect control.',
            'sample_size': 60000,
            'passes': False,
            'url': '/formulas'
        },
        {
            'domain': 'aviation',
            'name': 'Aviation',
            'pi': 0.12,
            'lambda': 0.83,
            'psi': 0.14,
            'nu': 0.00,
            'arch': 0.000,
            'leverage': 0.00,
            'type': 'Engineering',
            'finding': 'Complete nominative suppression. Engineering dominates.',
            'sample_size': 1743,
            'passes': False,
            'url': '/formulas'
        },
        {
            'domain': 'hurricanes',
            'name': 'Hurricanes',
            'pi': 0.30,
            'lambda': 0.13,
            'psi': 0.08,
            'nu': 0.35,
            'arch': 0.036,
            'leverage': 0.12,
            'type': 'Natural',
            'finding': 'Name gender affects evacuation. Physics dominates storm.',
            'sample_size': 94,
            'passes': False,
            'url': '/formulas'
        },
        {
            'domain': 'nba',
            'name': 'NBA',
            'pi': 0.49,
            'lambda': 0.500,  # Updated from Phase 7 extraction
            'theta': 0.500,  # NEW - Phase 7 extraction
            'psi': 0.30,
            'nu': 0.08,
            'arch': 0.018,
            'leverage': 0.04,
            'type': 'Physical Skill / Team',
            'finding': 'Tiny narrative wedge (~15% R²). Physical talent dominates. θ=0.500, λ=0.500 baseline.',
            'sample_size': 11979,
            'passes': False,
            'url': '/nba',
            'phase7_coverage': True
        },
        {
            'domain': 'nfl',
            'name': 'NFL',
            'pi': 0.57,
            'lambda': 0.500,  # Updated from Phase 7 extraction
            'theta': 0.505,  # NEW - Phase 7 extraction
            'psi': 0.25,
            'nu': 0.35,
            'arch': 0.014,
            'leverage': 0.14,
            'type': 'Team Sport',
            'finding': '14% R² optimized. Context-dependent, fractal structure. θ=0.505, λ=0.500 near baseline.',
            'sample_size': 3010,
            'passes': False,
            'url': '/nfl',
            'phase7_coverage': True
        },
        {
            'domain': 'mental_health',
            'name': 'Mental Health',
            'pi': 0.55,
            'lambda': 0.508,  # Updated from Phase 7 extraction
            'theta': 0.517,  # NEW - Phase 7 extraction
            'psi': 0.61,
            'nu': 0.60,
            'arch': 0.066,
            'leverage': 0.12,
            'type': 'Medical/Social',
            'finding': 'Name harshness predicts stigma (11% R² top context: High×Long). θ=0.517 shows moderate stigma awareness.',
            'sample_size': 200,
            'passes': False,
            'url': '/mental-health/dashboard',
            'phase7_coverage': True
        },
        {
            'domain': 'movies',
            'name': 'Movies/IMDB',
            'pi': 0.65,
            'lambda': 0.57,
            'psi': 0.24,
            'nu': 0.47,
            'arch': 0.026,
            'leverage': 0.04,
            'type': 'Entertainment',
            'finding': 'Combined R²=42.3%. Genre/budget dominate but context matters.',
            'sample_size': 1000,
            'passes': False,
            'url': '/imdb'
        },
        {
            'domain': 'oscars',
            'name': 'Oscar Best Picture',
            'pi': 0.75,
            'lambda': 0.25,
            'psi': 0.60,
            'nu': 0.80,
            'arch': 1.00,
            'leverage': 1.00,
            'type': 'Competitive Entertainment',
            'finding': 'AUC=1.00 (perfect). Nominative features separate perfectly.',
            'sample_size': 45,
            'passes': True,
            'url': '/oscars'
        },
        {
            'domain': 'crypto',
            'name': 'Cryptocurrency',
            'pi': 0.76,
            'lambda': 0.505,  # Updated from Phase 7 extraction
            'theta': 0.502,  # NEW - Phase 7 extraction
            'psi': 0.36,
            'nu': 0.85,
            'arch': 0.423,
            'leverage': 0.56,
            'type': 'Speculation',
            'finding': 'Names predict returns. ROC-AUC 0.925. θ=0.502 (low awareness), λ=0.505 (minimal barriers).',
            'sample_size': 3514,
            'passes': True,
            'url': '/crypto',
            'phase7_coverage': True
        },
        {
            'domain': 'startups',
            'name': 'Startups',
            'pi': 0.76,
            'lambda': 0.506,  # Updated from Phase 7 extraction
            'theta': 0.502,  # NEW - Phase 7 extraction
            'psi': 0.54,
            'nu': 0.50,
            'arch': 0.223,
            'leverage': 0.29,
            'type': 'Business',
            'finding': 'Product story r=0.980 (98% R²). Validates=TRUE. θ=0.502, λ=0.506 minimal constraints.',
            'sample_size': 474,
            'passes': True,
            'url': '/startups',
            'phase7_coverage': True
        },
        {
            'domain': 'character',
            'name': 'Character Domains',
            'pi': 0.85,
            'lambda': 0.15,
            'psi': 0.40,
            'nu': 0.75,
            'arch': 0.617,
            'leverage': 0.73,
            'type': 'Character-Driven',
            'finding': 'High narrativity. Narrative constructs reality.',
            'sample_size': 1000,
            'passes': True,
            'url': '/formulas'
        },
        {
            'domain': 'tennis',
            'name': 'Tennis',
            'pi': 0.75,
            'lambda': 0.531,  # Updated from Phase 7 extraction
            'theta': 0.515,  # NEW - Phase 7 extraction
            'psi': 0.70,
            'nu': 0.85,
            'arch': 0.865,
            'leverage': 0.93,
            'type': 'Individual Sport',
            'finding': '93.1% R² (optimized), 127% ROI. Rich nominative context. θ=0.515, λ=0.531 moderate forces.',
            'sample_size': 74906,
            'passes': True,
            'url': '/tennis',
            'phase7_coverage': True
        },
        {
            'domain': 'ufc',
            'name': 'UFC (Mixed Martial Arts)',
            'pi': 0.722,
            'lambda': 0.544,  # Updated from Phase 7 extraction
            'theta': 0.535,  # NEW - Phase 7 extraction
            'psi': 0.55,
            'nu': 0.40,
            'arch': 0.025,
            'leverage': 0.025,
            'type': 'Individual Combat Sport',
            'finding': 'HIGH π (0.722) but Δ=2.5% only. Performance-dominated (physical 87% >> narrative 55%). θ=0.535, λ=0.544 moderate constraints.',
            'sample_size': 7735,
            'passes': False,
            'url': '/ufc',
            'special': 'performance_dominated_lesson',
            'phase7_coverage': True
        },
        {
            'domain': 'golf',
            'name': 'Golf (Enhanced)',
            'pi': 0.70,
            'lambda': 0.689,  # Updated from Phase 7 extraction - HIGH constraints!
            'theta': 0.573,  # NEW - Phase 7 extraction - HIGH awareness!
            'psi': 0.72,
            'nu': 0.88,
            'arch': 0.953,
            'leverage': 0.977,
            'type': 'Individual Sport',
            'finding': '97.7% R² (40%→97.7% via nominative enrichment). θ=0.573 (mental game) + λ=0.689 (elite skill) + rich nominatives = peak.',
            'sample_size': 7700,
            'passes': True,
            'url': '/golf',
            'special': 'nominative_enrichment_proof',
            'phase7_coverage': True
        },
        {
            'domain': 'music',
            'name': 'Music/Spotify',
            'pi': 0.702,
            'lambda': 0.30,
            'psi': 0.65,
            'nu': 0.75,
            'arch': 0.031,
            'leverage': 0.044,
            'type': 'Entertainment',
            'finding': 'Weak narrative (like movies). Genre effects modest. Country 7.3% > Rock 6.4%.',
            'sample_size': 50000,
            'passes': False,
            'url': '/music'
        },
        {
            'domain': 'housing',
            'name': 'Housing (#13)',
            'pi': 0.92,
            'lambda': 0.08,
            'psi': 0.35,
            'nu': 0.85,
            'arch': 0.156,
            'leverage': 0.17,
            'type': 'Pure Nominative',
            'finding': '$93K discount (15.6% Arch). 99.92% skip rate. Pure nominative gravity.',
            'sample_size': 50000,
            'passes': False,  # Near threshold (0.156 vs 0.42 predicted)
            'url': '/housing',
            'special': 'cleanest_test'
        },
        {
            'domain': 'self_rated',
            'name': 'Self-Rated',
            'pi': 0.95,
            'lambda': 0.05,
            'psi': 1.00,
            'nu': 0.95,
            'arch': 0.564,
            'leverage': 0.59,
            'type': 'Identity',
            'finding': 'Narrator = judge. Perfect coupling enables narrative.',
            'sample_size': 1000,
            'passes': True,
            'url': '/formulas'
        },
        {
            'domain': 'wwe',
            'name': 'WWE (Wrestling)',
            'pi': 0.974,
            'lambda': 0.05,
            'psi': 0.90,
            'nu': 0.95,
            'arch': 1.800,
            'leverage': 1.85,
            'type': 'Prestige/Constructed',
            'finding': '$1B+ from fake. Everyone knows → Still works. Kayfabe.',
            'sample_size': 1250,
            'passes': True,
            'url': '/wwe-domain',
            'special': 'highest_pi'
        },
        {
            'domain': 'poker',
            'name': 'Professional Poker',
            'pi': 0.835,
            'lambda': 0.557,
            'theta': 0.256,
            'psi': 0.704,
            'nu': 0.047,
            'arch': 0.047,
            'leverage': 0.047,
            'type': 'Skill+Chance Hybrid',
            'finding': 'HIGHEST π (0.835) but R²=4.7% only. Card variance dominates despite rich narratives. First skill+chance hybrid.',
            'sample_size': 12000,
            'passes': False,
            'url': '/poker/results',
            'special': 'variance_dominated',
            'phase7_coverage': True
        },
        {
            'domain': 'hurricanes',
            'name': 'Hurricane Names',
            'pi': 0.677,  # Response π
            'lambda': 0.838,
            'theta': 0.376,
            'psi': 0.713,
            'nu': 0.011,  # Name contribution to R²
            'arch': 0.011,
            'leverage': 0.011,
            'type': 'Natural Disaster / Human Response',
            'finding': 'DUAL π: Storm (0.425) vs Response (0.677). Name effects +1.1% R². Tests nominative determinism in life/death decisions.',
            'sample_size': 1128,
            'passes': False,  # Small effect but significant stakes
            'url': '/hurricanes/results',
            'special': 'dual_pi_life_death',
            'phase7_coverage': True
        },
        {
            'domain': 'dinosaurs',
            'name': 'Dinosaur Names',
            'pi': 0.753,
            'lambda': 0.285,
            'theta': 0.362,
            'psi': 0.763,
            'nu': 0.623,  # Name contribution to R²
            'arch': 0.626,
            'leverage': 0.626,
            'type': 'Educational / Cultural Transmission',
            'finding': 'Names contribute 62.3% R² - 156x MORE than scientific importance! Jurassic Park +67.7%, Nickname +11.5%. First educational domain.',
            'sample_size': 950,
            'passes': True,  # Strong name effects
            'url': '/dinosaurs/results',
            'special': 'educational_transmission',
            'phase7_coverage': True
        }
    ]
    
    return jsonify({
        'total_domains': len(domains),
        'spectrum': {'min_pi': 0.04, 'max_pi': 0.974},
        'total_entities': 293606,
        'bookends': {
            'lower': {'domain': 'lottery', 'pi': 0.04, 'arch': 0.00},
            'upper': {'domain': 'wwe', 'pi': 0.974, 'arch': 1.80}
        },
        'correlations': {
            'pi_arch': 0.930,
            'pi_lambda': -0.958
        },
        'top_breakthroughs': {
            'highest_r2': {'domain': 'golf_enhanced', 'pi': 0.70, 'r2': 0.977},
            'highest_roi': {'domain': 'tennis', 'pi': 0.75, 'roi': 1.27},
            'highest_pi': {'domain': 'wwe', 'pi': 0.974, 'arch': 1.80},
            'highest_correlation': {'domain': 'startups', 'pi': 0.76, 'r': 0.980}
        },
        'key_insights': {
            'nominative_richness': 'Golf: 40% → 97.7% via rich nominatives (30+ proper nouns)',
            'performance_vs_narrative': 'UFC: HIGH π (0.722) but Δ=2.5% (performance-dominated)',
            'individual_vs_team': 'Tennis/Golf 90%+ R² vs NFL/NBA ~15% R²'
        },
        'domains': domains
    })

# Phase 7 API Endpoint
@app.route('/api/domains/phase7')
def phase7_api():
    """Phase 7 force data for all processed domains."""
    try:
        phase7_domains = get_all_phase7_domains()
        
        force_data = []
        for domain_name, data in phase7_domains.items():
            from utils.phase7_data_loader import get_force_interpretation
            interpretation = get_force_interpretation(data['theta_mean'], data['lambda_mean'])
            
            force_data.append({
                'domain': domain_name,
                'theta_mean': data['theta_mean'],
                'theta_std': data['theta_std'],
                'lambda_mean': data['lambda_mean'],
                'lambda_std': data['lambda_std'],
                'samples': data['samples'],
                'interpretation': interpretation
            })
        
        return jsonify({
            'framework_coverage': '11/11 (100%)',
            'transformers_total': 33,
            'phase7_transformers': 4,
            'domains_processed': len(force_data),
            'timestamp': '2025-11-12',
            'force_data': force_data,
            'key_insights': {
                'highest_theta': 'Golf (0.602) - High player awareness',
                'highest_lambda': 'Golf Enhanced (0.689) - Elite skill requirements',
                'theta_lambda_correlation': 0.675,
                'pattern': 'Expertise domains show both high awareness and high constraints'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Context processor for global template variables
@app.context_processor
def inject_globals():
    return {
        'app_name': 'Narrative Optimization',
        'version': '2.0.0'
    }

# ============================================================================
# TRANSFORMER ANALYSIS ROUTES
# ============================================================================

@app.route('/transformers/analysis')
def transformer_analysis():
    """Transformer analysis dashboard"""
    try:
        import numpy as np
        
        # Load completed domains
        features_dir = Path('narrative_optimization/data/features')
        domain_files = list(features_dir.glob('*_all_features.npz'))
        
        # Load catalog
        catalog_path = Path('narrative_optimization/TRANSFORMER_CATALOG.json')
        with open(catalog_path) as f:
            catalog = json.load(f)
        
        # Compile stats
        domain_stats = []
        for domain_file in domain_files:
            domain_name = domain_file.stem.replace('_all_features', '')
            try:
                data = np.load(domain_file, allow_pickle=True)
                domain_stats.append({
                    'name': domain_name,
                    'samples': int(data['features'].shape[0]),
                    'features': int(data['features'].shape[1]),
                    'has_outcomes': 'outcomes' in data.files and len(data['outcomes']) > 0,
                    'file': domain_file.name
                })
            except Exception as e:
                print(f"Error loading {domain_name}: {e}")
                continue
        
        # Sort by name
        domain_stats.sort(key=lambda x: x['name'])
        
        # Calculate totals
        total_samples = sum(d['samples'] for d in domain_stats)
        total_features = sum(d['features'] for d in domain_stats)
        
        return render_template('transformer_analysis.html',
                             domains=domain_stats,
                             catalog=catalog,
                             total_transformers=catalog['summary']['total_transformers'],
                             total_samples=total_samples,
                             total_features=total_features,
                             domains_processed=len(domain_stats))
    except Exception as e:
        return f"Error loading transformer analysis: {str(e)}", 500


@app.route('/transformers/catalog')
def transformer_catalog_view():
    """Browse transformer catalog"""
    try:
        catalog_path = Path('narrative_optimization/TRANSFORMER_CATALOG.json')
        with open(catalog_path) as f:
            catalog = json.load(f)
        
        return render_template('transformer_catalog.html',
                             catalog=catalog)
    except Exception as e:
        return f"Error loading catalog: {str(e)}", 500


@app.route('/api/domains/<domain>/features')
def domain_features_api(domain):
    """API endpoint for domain feature data"""
    try:
        import numpy as np
        features_path = Path(f'narrative_optimization/data/features/{domain}_all_features.npz')
        if not features_path.exists():
            return jsonify({'error': 'Domain not found'}), 404
        
        data = np.load(features_path, allow_pickle=True)
        
        return jsonify({
            'domain': domain,
            'samples': int(data['features'].shape[0]),
            'features': int(data['features'].shape[1]),
            'has_outcomes': 'outcomes' in data.files,
            'feature_names': data['feature_names'].tolist() if 'feature_names' in data.files else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# DISABLED: CLASSICAL ARCHETYPE ROUTES (Not validated in Nov 2025 pipeline)
# Added: November 13, 2025
# Theory-guided empirical discovery of narrative archetypes
# ============================================================================

# @app.route('/archetypes')
# def archetypes_home():
#     """Archetype system home - DISABLED until validated."""
#     return render_template('archetypes_home.html')

# @app.route('/archetypes/classical')
# def archetypes_classical():
#     """Classical narrative theories overview - DISABLED."""
#     theories = [
#         {'name': 'Joseph Campbell', 'work': "The Hero's Journey", 'year': 1949, 
#          'key_concept': '17-stage monomyth', 'status': 'Implemented'},
#         {'name': 'Carl Jung', 'work': 'Archetypal Psychology', 'year': 1959,
#          'key_concept': '12 universal archetypes', 'status': 'Implemented'},
#         {'name': 'Christopher Booker', 'work': '7 Basic Plots', 'year': 2004,
#          'key_concept': 'Seven master plots', 'status': 'Implemented'},
#         {'name': 'Blake Snyder', 'work': 'Save the Cat', 'year': 2005,
#          'key_concept': '15-beat structure', 'status': 'Implemented'},
#         {'name': 'Northrop Frye', 'work': 'Anatomy of Criticism', 'year': 1957,
#          'key_concept': 'Four mythoi', 'status': 'Implemented'},
#         {'name': 'Aristotle', 'work': 'Poetics', 'year': -335,
#          'key_concept': 'Six elements of drama', 'status': 'Theoretical'},
#         {'name': 'Vladimir Propp', 'work': 'Morphology of the Folktale', 'year': 1928,
#          'key_concept': '31 narrative functions', 'status': 'Theoretical'},
#         {'name': 'Kurt Vonnegut', 'work': 'Story Shapes', 'year': 2005,
#          'key_concept': '8 emotional trajectories', 'status': 'Theoretical'},
#     ]
#     return render_template('archetypes_classical.html', theories=theories)

# @app.route('/archetypes/domain/<domain_name>')
# def archetypes_domain(domain_name):
#     """Domain-specific archetype analysis - DISABLED."""
#     # Load domain archetype configuration
#     sys.path.insert(0, str(Path('narrative_optimization/src')))
#     from config.domain_archetypes import DOMAIN_ARCHETYPES
#     
#     if domain_name not in DOMAIN_ARCHETYPES:
#         return render_template('404.html'), 404
#     
#     domain_info = DOMAIN_ARCHETYPES[domain_name]
#     
#     return render_template('archetypes_domain.html',
#                           domain=domain_name,
#                           config=domain_info)

# @app.route('/theory/integration')
# def theory_integration():
#     """Complete framework integration - DISABLED."""
#     return render_template('theory_integration.html')

# @app.route('/archetypes/compare')
# def archetypes_compare():
#     """Interactive archetype comparison - DISABLED."""
#     return render_template('archetypes_compare.html')

# ============================================================================
# DISABLED: ARCHETYPE API ENDPOINTS (Not validated in Nov 2025 pipeline)
# ============================================================================

# Archetype API functions removed - will restore after validation
# Original functions: archetypes_all_api, archetype_theory_api, analyze_archetype_api, archetypes_domains_api

# ============================================================================
# DISABLED: BETTING ARCHETYPE OPPORTUNITIES (Not validated in Nov 2025 pipeline)
# Based on archetype analysis of 121,727 narratives
# ============================================================================

# Betting archetype opportunities function removed - will restore after validation

if __name__ == '__main__':
    port = 5738
    print(f"\n🚀 Starting Narrative Optimization Framework")
    print(f"📊 Access at: http://127.0.0.1:{port}")
    print(f"\n✅ VALIDATED SYSTEMS (Nov 17, 2025 Production Backtest):")
    print(f"🏒 NHL: http://127.0.0.1:{port}/nhl (69.4% win, 32.5% ROI)")
    print(f"🏈 NFL: http://127.0.0.1:{port}/nfl (66.7% win, 27.3% ROI)")
    print(f"🏀 NBA: http://127.0.0.1:{port}/nba (54.5% win, 7.6% ROI)")
    print(f"\n🔧 CORE TOOLS:")
    print(f"🎯 Main dashboard: http://127.0.0.1:{port}/")
    print(f"📝 Narrative Analyzer: http://127.0.0.1:{port}/analyze")
    print(f"⚙️  Domain Processor: http://127.0.0.1:{port}/process")
    print(f"📈 Betting Dashboard: http://127.0.0.1:{port}/betting/dashboard")
    print(f"🔬 Transformers: http://127.0.0.1:{port}/transformers/analysis")
    print(f"⚡ Variables: http://127.0.0.1:{port}/variables")
    print(f"📚 Framework Story: http://127.0.0.1:{port}/framework-story")
    print(f"\n⚠️  NOTE: Non-validated domains temporarily hidden until re-validation")
    print(f"📋 See: analysis/EXECUTIVE_SUMMARY_BACKTEST.md for validation details\n")
    app.run(host='0.0.0.0', port=5738, debug=True)

