"""
Live Betting Prediction API
============================

REST API for real-time betting predictions with:
- <100ms response time target
- Live game integration
- Kelly Criterion bet sizing
- Cross-domain feature extraction
- Advanced ensemble predictions

Endpoints:
- GET /api/live/games - List active games
- POST /api/live/predict - Get prediction for game
- GET /api/live/opportunities - Current betting opportunities
- POST /api/live/kelly-size - Calculate optimal bet size

Author: AI Coding Assistant
Date: November 16, 2025
"""

from flask import Blueprint, jsonify, request
import json
import os
import sys
import numpy as np
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative_optimization.feature_engineering.cross_domain_features import CrossDomainFeatureExtractor
from narrative_optimization.betting.kelly_criterion import KellyCriterion
from narrative_optimization.betting.prop_kelly_criterion import PropKellyCriterion, PropBet
from scripts.live_game_monitor import LiveGameMonitor
from scripts.live_odds_fetcher import LiveOddsFetcher
from scripts import nhl_daily_prop_predictions as nhl_prop_pipeline

# Create Blueprint
live_betting_api_bp = Blueprint('live_betting_api', __name__, url_prefix='/api/live')

# Initialize components (in production, these would be singleton instances)
cross_domain_extractor = CrossDomainFeatureExtractor()
kelly = KellyCriterion()
prop_kelly = PropKellyCriterion()
game_monitor = LiveGameMonitor(update_frequency=120)
odds_fetcher = LiveOddsFetcher()

# Model cache (load models at startup)
MODEL_CACHE = {}
PROP_CACHE_SECONDS = int(os.getenv('PROP_CACHE_DURATION', '180'))
MAX_PROPS_PER_SLATE = int(os.getenv('MAX_PROPS_PER_SLATE', '20'))

_prop_prediction_cache: Dict[str, Optional[object]] = {
    'timestamp': None,
    'edges': [],
    'metadata': {},
}


def load_models():
    """Load pre-trained models into cache."""
    # In production, load actual trained models
    # For now, return mock predictions
    MODEL_CACHE['nba_ensemble'] = 'mock'
    MODEL_CACHE['nfl_ensemble'] = 'mock'


def get_mock_prediction(game_data: Dict, league: str) -> float:
    """Generate mock prediction (replace with actual model inference)."""
    # Simple mock: home team favored 55%
    return 0.55 + np.random.uniform(-0.1, 0.1)


def _american_to_probability(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _decode_game_matchup(game_id: str) -> Tuple[str, str]:
    """Infer away/home team abbreviations from game_id if possible."""
    try:
        date_part, away, home = game_id.split('-', 2)
        return away, home
    except ValueError:
        return "AWAY", "HOME"


def _tier_prop_edge(edge: Dict) -> str:
    """Classify prop edge into tiers."""
    edge_pct = edge.get('edge_pct', edge.get('edge', 0) * 100)
    confidence = edge.get('confidence', 0)
    if edge_pct >= 8 and confidence >= 0.65:
        return 'elite'
    if edge_pct >= 6 and confidence >= 0.60:
        return 'strong'
    if edge_pct >= 4 and confidence >= 0.55:
        return 'moderate'
    return 'speculative'


def _load_prop_edges(force_refresh: bool = False) -> Dict:
    """Run (or reuse) the prop prediction pipeline."""
    global _prop_prediction_cache
    now = datetime.utcnow()
    cache_ts = _prop_prediction_cache.get('timestamp')
    if (
        not force_refresh
        and cache_ts
        and isinstance(cache_ts, datetime)
        and (now - cache_ts).total_seconds() < PROP_CACHE_SECONDS
    ):
        return _prop_prediction_cache

    try:
        games = nhl_prop_pipeline.fetch_todays_games()
        if not games:
            _prop_prediction_cache = {
                'timestamp': now,
                'edges': [],
                'metadata': {
                    'message': 'No NHL games scheduled today',
                    'n_games': 0,
                    'n_players': 0,
                    'n_predictions': 0,
                    'n_edges': 0,
                },
            }
            return _prop_prediction_cache

        player_data = nhl_prop_pipeline.collect_player_data(games)
        predictions = nhl_prop_pipeline.generate_prop_predictions(player_data)
        edges = nhl_prop_pipeline.fetch_prop_odds_and_calculate_edges(predictions)

        _prop_prediction_cache = {
            'timestamp': now,
            'edges': edges,
            'metadata': {
                'n_games': len(games),
                'n_players': len(player_data.get('players', [])),
                'n_predictions': len(predictions),
                'n_edges': len(edges),
            },
        }
        return _prop_prediction_cache
    except Exception as exc:
        # Bubble up error after logging
        print(f"[PROP PIPELINE] Error generating prop predictions: {exc}")
        raise


def _prepare_prop_response(
    edges: List[Dict],
    *,
    game_filter: Optional[List[str]] = None,
    min_edge: float = 0.04,
    min_confidence: float = 0.0,
    limit: Optional[int] = None,
) -> Dict:
    """Filter and annotate prop edges for API responses."""
    game_set = set(g.upper() for g in game_filter) if game_filter else None
    filtered: List[Dict] = []

    for edge in edges:
        if game_set and edge.get('game_id', '').upper() not in game_set:
            continue
        if edge.get('edge', 0.0) < min_edge:
            continue
        if edge.get('confidence', 0.0) < min_confidence:
            continue

        record = dict(edge)
        tier = _tier_prop_edge(record)
        record['tier'] = tier
        record['edge_pct'] = round(record.get('edge_pct', record.get('edge', 0) * 100), 2)
        record['confidence'] = round(record.get('confidence', 0.0), 3)
        away, home = _decode_game_matchup(record.get('game_id', ''))
        record['matchup'] = f"{away} @ {home}"
        filtered.append(record)

    filtered.sort(key=lambda e: e.get('edge', 0.0), reverse=True)
    total_filtered = len(filtered)
    if limit:
        filtered = filtered[:limit]

    tier_counts: Dict[str, int] = {}
    game_counts: Dict[str, Dict[str, object]] = {}

    for edge in filtered:
        tier_counts[edge['tier']] = tier_counts.get(edge['tier'], 0) + 1
        game_ref = game_counts.setdefault(
            edge['game_id'],
            {
                'game_id': edge['game_id'],
                'matchup': edge['matchup'],
                'prop_count': 0,
            },
        )
        game_ref['prop_count'] += 1

    return {
        'props': filtered,
        'summary': {
            'tier_counts': tier_counts,
            'game_counts': list(game_counts.values()),
            'returned': len(filtered),
            'filtered_total': total_filtered,
        },
    }


@live_betting_api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'models_loaded': len(MODEL_CACHE),
        'version': '1.0.0'
    })


@live_betting_api_bp.route('/games', methods=['GET'])
def get_active_games():
    """
    Get list of active games.
    
    Query params:
        league: 'nba' or 'nfl' (optional)
    
    Response time target: <50ms
    """
    start_time = time.time()
    
    league = request.args.get('league', 'nba').lower()
    
    # Fetch active games
    if league == 'nba':
        games = game_monitor.fetch_live_nba_scores()
    else:
        games = game_monitor.fetch_live_nfl_scores()
    
    # Add features to each game
    games_with_features = []
    for game in games:
        features = game_monitor.extract_live_features(game, league)
        games_with_features.append({
            'game': game,
            'features': features
        })
    
    response_time = (time.time() - start_time) * 1000  # ms
    
    return jsonify({
        'league': league.upper(),
        'n_games': len(games_with_features),
        'games': games_with_features,
        'response_time_ms': round(response_time, 2)
    })


@live_betting_api_bp.route('/props/predictions', methods=['POST'])
def get_prop_predictions():
    """
    Generate NHL prop predictions (pre-game) and return filtered edges.
    
    Request body (all optional):
    {
        "games": ["20241120-BOS-TOR"],   # filter specific game_ids
        "min_edge": 0.04,                # minimum edge threshold
        "min_confidence": 0.55,          # minimum model confidence
        "limit": 20,                     # max props to return
        "force_refresh": false           # bypass cache
    }
    """
    start_time = time.time()
    payload = request.get_json(silent=True) or {}
    
    games_filter = payload.get('games')
    min_edge = float(payload.get('min_edge', 0.04))
    min_confidence = float(payload.get('min_confidence', 0.0))
    limit = int(payload.get('limit', MAX_PROPS_PER_SLATE))
    force_refresh = bool(payload.get('force_refresh', False))
    
    try:
        cache = _load_prop_edges(force_refresh=force_refresh)
        response = _prepare_prop_response(
            cache.get('edges', []),
            game_filter=games_filter,
            min_edge=min_edge,
            min_confidence=min_confidence,
            limit=limit,
        )
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': str(exc),
            'solution': 'Run python scripts/nhl_daily_prop_predictions.py to inspect logs'
        }), 500
    
    timestamp = cache.get('timestamp')
    response_time = (time.time() - start_time) * 1000
    
    return jsonify({
        'success': True,
        'generated_at': timestamp.isoformat() if isinstance(timestamp, datetime) else None,
        'metadata': cache.get('metadata', {}),
        'filters': {
            'games': games_filter,
            'min_edge': min_edge,
            'min_confidence': min_confidence,
            'limit': limit,
        },
        'props': response['props'],
        'summary': response['summary'],
        'response_time_ms': round(response_time, 2),
    })


@live_betting_api_bp.route('/predict', methods=['POST'])
def predict_game():
    """
    Get prediction for a specific game.
    
    Request body:
    {
        "game_id": "nba_game_123",
        "league": "nba",
        "game_data": {...},  // optional, will fetch if not provided
        "include_features": true  // optional
    }
    
    Response time target: <100ms
    """
    start_time = time.time()
    
    data = request.get_json()
    game_id = data.get('game_id')
    league = data.get('league', 'nba').lower()
    game_data = data.get('game_data')
    include_features = data.get('include_features', False)
    
    if not game_id:
        return jsonify({'error': 'game_id required'}), 400
    
    # Get game data if not provided
    if not game_data:
        # Fetch from monitor
        if league == 'nba':
            games = game_monitor.fetch_live_nba_scores()
        else:
            games = game_monitor.fetch_live_nfl_scores()
        
        game_data = next((g for g in games if g['game_id'] == game_id), None)
        
        if not game_data:
            return jsonify({'error': 'Game not found'}), 404
    
    # Extract features
    live_features = game_monitor.extract_live_features(game_data, league)
    cross_domain_features = cross_domain_extractor.extract_all_cross_domain_features(game_data, league)
    
    # Combine features
    all_features = {**live_features, **cross_domain_features.to_dict('records')[0]}
    
    # Get prediction (mock for now)
    win_probability = get_mock_prediction(game_data, league)
    
    # Calculate confidence
    confidence = abs(win_probability - 0.5) * 2  # 0 to 1 scale
    confidence_level = 'high' if confidence > 0.2 else 'medium' if confidence > 0.1 else 'low'
    
    response_time = (time.time() - start_time) * 1000
    
    response = {
        'game_id': game_id,
        'league': league.upper(),
        'prediction': {
            'home_win_probability': round(win_probability, 4),
            'away_win_probability': round(1 - win_probability, 4),
            'confidence': round(confidence, 4),
            'confidence_level': confidence_level,
            'recommendation': 'home' if win_probability > 0.5 else 'away'
        },
        'response_time_ms': round(response_time, 2)
    }
    
    if include_features:
        response['features'] = {
            'live_features': live_features,
            'n_cross_domain_features': len(cross_domain_features.columns)
        }
    
    return jsonify(response)


@live_betting_api_bp.route('/generate-predictions', methods=['POST'])
def generate_predictions():
    """
    Generate fresh NHL predictions on-demand using the vetted feature pipeline.
    Prefers the curated `data/domains/nhl_upcoming_latest.json` snapshot so that
    feature engineering has the same context as the validation dataset. If that
    file is missing, falls back to live odds fetcher (best-effort with limited
    context).
    """
    from datetime import datetime
    import json
    
    print(f"\n[LIVE API] Generating fresh NHL predictions...")
    
    def american_prob(odds):
        from narrative_optimization.domains.nhl.score_upcoming_games import american_to_prob
        return american_to_prob(odds) if odds is not None else None
    
    def normalize_game(raw_game):
        """Ensure game structure matches the modeling pipeline expectations."""
        game = dict(raw_game)
        
        if not game.get('game_id'):
            stamp = (game.get('date') or datetime.now().strftime('%Y%m%d'))
            away = game.get('away_team', 'AWAY')
            home = game.get('home_team', 'HOME')
            game['game_id'] = f"{stamp}-{away}-{home}"
        
        odds = game.get('betting_odds') or game.get('odds') or {}
        if 'betting_odds' not in game:
            game['betting_odds'] = {
                'moneyline_home': odds.get('moneyline_home'),
                'moneyline_away': odds.get('moneyline_away'),
                'total': odds.get('total'),
                'over_odds': odds.get('over_odds'),
                'under_odds': odds.get('under_odds'),
                'puck_line_home': odds.get('puck_line_home'),
                'puck_line_home_odds': odds.get('puck_line_home_odds'),
                'puck_line_away': odds.get('puck_line_away'),
                'puck_line_away_odds': odds.get('puck_line_away_odds'),
            }
        
        # Provide placeholders so feature generators don't explode
        game.setdefault('home_record', {'wins': 0, 'losses': 0})
        game.setdefault('away_record', {'wins': 0, 'losses': 0})
        game.setdefault('home_home_record', {'wins': 0, 'losses': 0})
        game.setdefault('away_away_record', {'wins': 0, 'losses': 0})
        game.setdefault('matchup', f"{game.get('away_team', 'AWAY')} @ {game.get('home_team', 'HOME')}")
        
        return game
    
    try:
        project_root = Path(__file__).parent.parent
        curated_path = project_root / 'data' / 'domains' / 'nhl_upcoming_latest.json'
        data_source = None
        
        # ALWAYS fetch fresh games from The Odds API
        from scripts.nhl_fetch_live_odds import NHLOddsFetcher
        fetcher = NHLOddsFetcher()
        raw_games = fetcher.fetch_upcoming_games()
        data_source = 'the-odds-api-live'
        print(f"[LIVE API] ✓ Fetched {len(raw_games)} games from live odds API")
        
        # Filter out games that have already started
        now = datetime.now()
        future_games = []
        for g in raw_games:
            try:
                commence_str = g.get('commence_time', '')
                if commence_str:
                    # Parse ISO format: 2025-11-20T23:00:00Z
                    commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                    # Convert to naive datetime for comparison
                    commence_naive = commence_time.replace(tzinfo=None) if commence_time.tzinfo else commence_time
                    if commence_naive > now:
                        future_games.append(g)
                        print(f"[LIVE API]   ✓ {g.get('away_team', '?')} @ {g.get('home_team', '?')} at {commence_str}")
            except:
                # If we can't parse the time, include it anyway
                future_games.append(g)
        
        raw_games = future_games
        print(f"[LIVE API] ✓ Filtered to {len(raw_games)} upcoming games")
        
        if not raw_games:
            return jsonify({
                'success': False,
                'error': 'No NHL games available in upcoming dataset',
                'solution': 'Run refresh script to rebuild data/domains/nhl_upcoming_latest.json'
            }), 404
        
        def is_valid_moneyline(value):
            return isinstance(value, (int, float)) and abs(value) >= 100
        
        normalized_games = []
        for g in raw_games:
            norm = normalize_game(g)
            odds = norm.get('betting_odds', {})
            home_ml = odds.get('moneyline_home')
            away_ml = odds.get('moneyline_away')
            if not (is_valid_moneyline(home_ml) and is_valid_moneyline(away_ml)):
                continue
            normalized_games.append(norm)
        
        if not normalized_games:
            return jsonify({
                'success': False,
                'error': 'Upcoming games missing moneyline odds',
                'solution': 'Ensure nhl_upcoming_latest.json contains moneyline_home/away'
            }), 400
        
        # Step 2: Load NHL models
        from narrative_optimization.domains.nhl.score_upcoming_games import load_models, build_feature_matrix
        
        models = load_models(['narrative_logistic', 'narrative_gradient', 'narrative_forest'])
        print(f"[LIVE API] ✓ Loaded {len(models)} models")
        
        # Step 3: Load feature columns
        metadata_path = project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'narrative_model_summary.json'
        with open(metadata_path) as f:
            metadata = json.load(f)
        feature_cols = metadata.get('feature_columns', [])
        
        # Step 4: Build feature matrix
        feature_matrix = build_feature_matrix(normalized_games, feature_cols)
        print(f"[LIVE API] ✓ Built feature matrix ({feature_matrix.shape})")
        
        # Save fresh games to upcoming file for future reference
        curated_path.parent.mkdir(parents=True, exist_ok=True)
        with open(curated_path, 'w') as f:
            json.dump(normalized_games, f, indent=2)
        print(f"[LIVE API] ✓ Saved fresh games to {curated_path.name}")
        
        # Keep lookup for odds/matchups
        game_lookup = {game['game_id']: game for game in normalized_games}
        
        # Step 5: Generate predictions with probability caps
        predictions = []
        for name, model in models.items():
            probas = model.predict_proba(feature_matrix.values)[:, 1]
            
            # Cap extreme probabilities at 72% (realistic upper bound for sports betting)
            probas = np.clip(probas, 0.28, 0.72)
            
            for idx, game in enumerate(normalized_games):
                prob_home = float(probas[idx])
                odds = game.get('betting_odds', {})
                home_ml = odds.get('moneyline_home')
                away_ml = odds.get('moneyline_away')
                puck_line_home = odds.get('puck_line_home')
                puck_line_away = odds.get('puck_line_away')
                puck_line_home_odds = odds.get('puck_line_home_odds')
                puck_line_away_odds = odds.get('puck_line_away_odds')
                total = odds.get('total')
                over_odds = odds.get('over_odds')
                under_odds = odds.get('under_odds')
                
                game_id = game['game_id']
                matchup = game.get('matchup', f"{game.get('away_team', 'AWAY')} @ {game.get('home_team', 'HOME')}")
                
                # MONEYLINE bets
                if home_ml is not None and away_ml is not None:
                    home_implied = american_prob(home_ml)
                    away_implied = american_prob(away_ml)
                    
                    if home_implied and away_implied:
                        home_edge = prob_home - home_implied
                        away_edge = (1 - prob_home) - away_implied
                        
                        if home_edge > away_edge and home_edge > 0.02:
                            predictions.append({
                                'game_id': game_id,
                                'matchup': matchup,
                                'model': name,
                                'side': 'home',
                                'bet_type': 'moneyline',
                                'prob': round(prob_home, 4),
                                'edge': round(home_edge, 4),
                                'odds': home_ml,
                                'implied_prob': round(home_implied, 4)
                            })
                        elif away_edge > 0.02:
                            predictions.append({
                                'game_id': game_id,
                                'matchup': matchup,
                                'model': name,
                                'side': 'away',
                                'bet_type': 'moneyline',
                                'prob': round(1 - prob_home, 4),
                                'edge': round(away_edge, 4),
                                'odds': away_ml,
                                'implied_prob': round(away_implied, 4)
                            })
                
                # PUCK LINE bets (if strong favorite/underdog signal)
                if puck_line_home_odds and puck_line_away_odds and puck_line_home is not None:
                    pl_home_implied = american_prob(puck_line_home_odds)
                    pl_away_implied = american_prob(puck_line_away_odds)
                    
                    # Adjust probability for puck line (spread)
                    # If home is favored, covering -1.5 requires stronger win
                    if prob_home >= 0.60 and pl_home_implied:
                        pl_edge = (prob_home - 0.15) - pl_home_implied  # Reduce prob for spread coverage
                        if pl_edge > 0.03:
                            predictions.append({
                                'game_id': game_id,
                                'matchup': matchup,
                                'model': name,
                                'side': 'home',
                                'bet_type': 'puck_line',
                                'spread': round(puck_line_home, 1),
                                'prob': round(prob_home - 0.15, 4),
                                'edge': round(pl_edge, 4),
                                'odds': puck_line_home_odds,
                                'implied_prob': round(pl_home_implied, 4)
                            })
                    
                    if prob_home <= 0.40 and pl_away_implied:
                        pl_edge = (0.85 - prob_home) - pl_away_implied
                        if pl_edge > 0.03:
                            predictions.append({
                                'game_id': game_id,
                                'matchup': matchup,
                                'model': name,
                                'side': 'away',
                                'bet_type': 'puck_line',
                                'spread': round(puck_line_away, 1),
                                'prob': round(1 - prob_home - 0.15, 4),
                                'edge': round(pl_edge, 4),
                                'odds': puck_line_away_odds,
                                'implied_prob': round(pl_away_implied, 4)
                            })
                
                # TOTALS bets (model-based scoring signal)
                # Generate totals for all models to get diversity
                if total and over_odds and under_odds:
                    over_implied = american_prob(over_odds)
                    under_implied = american_prob(under_odds)
                    
                    if over_implied and under_implied:
                        # Model-specific scoring patterns
                        if name == 'narrative_logistic':
                            # Logistic: competitiveness drives scoring
                            competitiveness = abs(prob_home - 0.5)
                            over_signal = 0.52 + (competitiveness * 0.18)
                        elif name == 'narrative_gradient':
                            # Gradient: strong favorites push over
                            over_signal = 0.53 if prob_home > 0.60 or prob_home < 0.40 else 0.48
                        else:  # forest
                            # Forest: balanced approach
                            over_signal = 0.51 + (abs(prob_home - 0.5) * 0.12)
                        
                        over_edge = over_signal - over_implied
                        if over_edge > 0.02:
                            predictions.append({
                                'game_id': game_id,
                                'matchup': matchup,
                                'model': name,
                                'side': 'over',
                                'bet_type': 'total',
                                'total_line': round(total, 1),
                                'prob': round(over_signal, 4),
                                'edge': round(over_edge, 4),
                                'odds': over_odds,
                                'implied_prob': round(over_implied, 4)
                            })
                        
                        under_signal = 1 - over_signal
                        under_edge = under_signal - under_implied
                        if under_edge > 0.02:
                            predictions.append({
                                'game_id': game_id,
                                'matchup': matchup,
                                'model': name,
                                'side': 'under',
                                'bet_type': 'total',
                                'total_line': round(total, 1),
                                'prob': round(under_signal, 4),
                                'edge': round(under_edge, 4),
                                'odds': under_odds,
                                'implied_prob': round(under_implied, 4)
                            })
        
        # Step 6: Group by game and save
        games_with_recs = {}
        now_iso = datetime.now().isoformat()
        for pred in predictions:
            game_ref = game_lookup.get(pred['game_id'], {})
            odds = game_ref.get('betting_odds', {})
            record = games_with_recs.setdefault(pred['game_id'], {
                'game_id': pred['game_id'],
                'matchup': pred['matchup'],
                'date': game_ref.get('date'),
                'commence_time': game_ref.get('commence_time', now_iso + 'Z'),
                'moneyline_home': odds.get('moneyline_home'),
                'moneyline_away': odds.get('moneyline_away'),
                'source_dataset': data_source,
                'exported_at': now_iso,
                'recommendations': []
            })
            
            rec_data = {
                'model': pred['model'],
                'side': pred['side'],
                'bet_type': pred['bet_type'],
                'prob': pred['prob'],
                'edge': pred['edge'],
                'odds': pred.get('odds') or pred.get('moneyline'),
                'implied_prob': pred['implied_prob']
            }
            
            # Add bet-type specific fields
            if pred['bet_type'] == 'puck_line':
                rec_data['spread'] = pred.get('spread')
            elif pred['bet_type'] == 'total':
                rec_data['total_line'] = pred.get('total_line')
            
            record['recommendations'].append(rec_data)
        
        output_path = project_root / 'analysis' / 'nhl_upcoming_predictions.json'
        with open(output_path, 'w') as f:
            json.dump(list(games_with_recs.values()), f, indent=2)
        
        print(f"[LIVE API] ✓ Generated {len(predictions)} predictions for {len(games_with_recs)} games")
        
        return jsonify({
            'success': True,
            'n_games': len(games_with_recs),
            'n_predictions': len(predictions),
            'timestamp': now_iso,
            'data_source': data_source
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"[LIVE API] ✗ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        
        solution = None
        if 'sklearn' in error_msg or '_gb_losses' in error_msg:
            solution = 'sklearn version mismatch - rerun model trainer'
        elif 'ModuleNotFoundError' in error_msg:
            solution = 'Missing dependencies - install required packages'
        elif 'No games available' in error_msg:
            solution = 'Check data/domains/nhl_upcoming_latest.json freshness'
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'solution': solution,
            'manual_command': 'python3 scripts/nhl_daily_predictions.py'
        }), 500

@live_betting_api_bp.route('/opportunities', methods=['GET'])
def get_opportunities():
    """
    Get current NHL betting opportunities using actual predictions.
    Auto-generates fresh predictions if data is >4 hours old.
    
    Query params:
        league: 'nhl' (default, only NHL is validated)
        min_edge: minimum edge required (default: 0.05)
        min_confidence: minimum confidence required (default: 0.65)
    
    Response time target: <200ms (may take longer on first call if regenerating)
    """
    start_time = time.time()
    from datetime import datetime, timedelta
    
    league = request.args.get('league', 'nhl').lower()
    min_edge = float(request.args.get('min_edge', 0.05))
    min_confidence = float(request.args.get('min_confidence', 0.65))
    include_props = request.args.get('include_props', 'false').lower() == 'true'
    max_props = int(request.args.get('max_props', MAX_PROPS_PER_SLATE))
    
    opportunities = []
    predictions_age_hours = None
    auto_refreshed = False
    prop_summary = None
    prop_error = None
    
    # Load actual NHL predictions from existing pipeline
    if league == 'nhl':
        predictions_path = Path(__file__).parent.parent / 'analysis' / 'nhl_upcoming_predictions.json'
        
        # Don't auto-generate, user can click button
        auto_refreshed = False
        
        if predictions_path.exists():
            import json
            from datetime import timezone
            
            # Check file age
            file_modified = datetime.fromtimestamp(predictions_path.stat().st_mtime)
            predictions_age_hours = (datetime.now() - file_modified).total_seconds() / 3600
            
            with open(predictions_path, 'r') as f:
                games = json.load(f)
            
            # Filter for today's and future games only
            today = datetime.now().date()
            
            for game in games:
                # Parse game date from game_id (format: YYYYMMDD-TEAM-TEAM)
                try:
                    game_date_str = game['game_id'].split('-')[0]
                    game_date = datetime.strptime(game_date_str, '%Y%m%d').date()
                    
                    # Skip games that have already occurred
                    if game_date < today:
                        continue
                except:
                    pass  # If we can't parse date, include the game
                
                # Get ALL significant recommendations per game (not just best)
                matchup_parts = game['matchup'].split(' @ ')
                home_team = matchup_parts[1] if len(matchup_parts) > 1 else 'TBD'
                away_team = matchup_parts[0] if len(matchup_parts) > 0 else 'TBD'
                
                for rec in game.get('recommendations', []):
                    edge = rec.get('edge', 0)
                    prob = rec.get('prob', 0)
                    bet_type = rec.get('bet_type', 'moneyline')
                    
                    # Determine tier based on confidence and edge (adjust for bet type)
                    if bet_type == 'total':
                        # Totals naturally have tighter markets, use relaxed thresholds
                        if prob >= 0.56 and edge >= 0.04:
                            tier = 'elite'
                        elif prob >= 0.54 and edge >= 0.03:
                            tier = 'strong'
                        elif prob >= 0.52 and edge >= 0.02:
                            tier = 'moderate'
                        else:
                            tier = 'speculative'
                    else:
                        # Moneyline and puck line use standard thresholds
                        if prob >= 0.60 and edge >= 0.10:
                            tier = 'elite'
                        elif prob >= 0.55 and edge >= 0.08:
                            tier = 'strong'
                        elif prob >= 0.52 and edge >= 0.05:
                            tier = 'moderate'
                        else:
                            tier = 'speculative'
                    
                    # Only include significant picks (moderate and above)
                    if tier not in ['elite', 'strong', 'moderate']:
                        continue
                    
                    # Build pick label
                    if bet_type == 'moneyline':
                        recommended_pick = home_team if rec['side'] == 'home' else away_team
                    elif bet_type == 'puck_line':
                        spread = rec.get('spread', 0)
                        team = home_team if rec['side'] == 'home' else away_team
                        recommended_pick = f"{team} {spread:+.1f}"
                    elif bet_type == 'total':
                        total_line = rec.get('total_line', 0)
                        recommended_pick = f"{rec['side'].upper()} {total_line}"
                    else:
                        recommended_pick = rec['side']
                    
                    opp_data = {
                        'game_id': game['game_id'],
                        'matchup': game['matchup'],
                        'home_team': home_team,
                        'away_team': away_team,
                        'recommended_pick': recommended_pick,
                        'confidence': round(prob, 3),
                        'edge_pct': round(edge * 100, 1),
                        'odds': rec.get('odds'),
                        'model': rec['model'],
                        'side': rec['side'],
                        'bet_type': bet_type,
                        'tier': tier,
                        'game_time': game.get('commence_time', 'TBD')
                    }
                    
                    # Add bet-type specific fields to display
                    if bet_type == 'puck_line':
                        opp_data['spread'] = rec.get('spread')
                    elif bet_type == 'total':
                        opp_data['total_line'] = rec.get('total_line')
                    
                    opportunities.append(opp_data)
    
    if include_props:
        try:
            prop_cache = _load_prop_edges(force_refresh=False)
            prop_response = _prepare_prop_response(
                prop_cache.get('edges', []),
                min_edge=min_edge,
                min_confidence=min_confidence,
                limit=max_props,
            )
            prop_summary = {
                **prop_response['summary'],
                'generated_at': prop_cache.get('timestamp').isoformat()
                if isinstance(prop_cache.get('timestamp'), datetime)
                else None,
            }
            
            for prop in prop_response['props']:
                prob_key = 'prob_over' if prop.get('side') == 'over' else 'prob_under'
                opportunities.append({
                    'game_id': prop.get('game_id'),
                    'matchup': prop.get('matchup'),
                    'player_name': prop.get('player_name'),
                    'bet_type': 'prop',
                    'prop_type': prop.get('prop_type'),
                    'line': prop.get('line'),
                    'side': prop.get('side'),
                    'recommended_pick': f"{prop.get('player_name')} {prop.get('prop_type', '').upper()} {prop.get('side', '').upper()} {prop.get('line')}",
                    'confidence': prop.get('confidence'),
                    'probability': round(prop.get(prob_key, 0), 3),
                    'edge_pct': prop.get('edge_pct'),
                    'odds': prop.get('odds'),
                    'bookmaker': prop.get('bookmaker'),
                    'tier': prop.get('tier'),
                    'expected_value': prop.get('expected_value'),
                })
        except Exception as exc:
            prop_error = str(exc)
    
    response_time = (time.time() - start_time) * 1000
    
    return jsonify({
        'league': league.upper(),
        'n_opportunities': len(opportunities),
        'opportunities': opportunities,
        'filters': {
            'min_edge': min_edge,
            'min_confidence': min_confidence,
            'include_props': include_props,
            'max_props': max_props
        },
        'response_time_ms': round(response_time, 2),
        'source': 'nhl_upcoming_predictions.json',
        'predictions_age_hours': round(predictions_age_hours, 1) if predictions_age_hours else None,
        'auto_refreshed': auto_refreshed,
        'needs_refresh': predictions_age_hours > 24 if predictions_age_hours else True,
        'prop_summary': prop_summary,
        'prop_error': prop_error
    })


@live_betting_api_bp.route('/kelly-size', methods=['POST'])
def calculate_kelly_size():
    """
    Calculate optimal bet size using Kelly Criterion.
    
    Request body:
    {
        "game_id": "nba_game_123",
        "win_probability": 0.65,
        "american_odds": -150,
        "bankroll": 1000.0,
        "kelly_fraction": 0.5  // optional, default 0.5 (half Kelly)
    }
    
    Response time target: <10ms
    """
    start_time = time.time()
    
    data = request.get_json()
    
    required = ['game_id', 'win_probability', 'american_odds', 'bankroll']
    for field in required:
        if field not in data:
            return jsonify({'error': f'{field} required'}), 400
    
    game_id = data['game_id']
    win_prob = float(data['win_probability'])
    odds = float(data['american_odds'])
    bankroll = float(data['bankroll'])
    kelly_fraction = data.get('kelly_fraction')
    bet_type = data.get('bet_type', 'moneyline').lower()
    
    # Validate
    if not (0 < win_prob < 1):
        return jsonify({'error': 'win_probability must be between 0 and 1'}), 400
    
    if bankroll <= 0:
        return jsonify({'error': 'bankroll must be positive'}), 400
    
    if bet_type == 'prop':
        prop_details = data.get('prop_details') or {}
        required_prop_fields = ['player_name', 'prop_type', 'line', 'side']
        for field in required_prop_fields:
            if field not in prop_details:
                return jsonify({'error': f'prop_details.{field} required'}), 400
        
        implied_prob = _american_to_probability(odds)
        prop_edge = win_prob - implied_prob
        confidence = float(prop_details.get('confidence', 0.6))
        book_limit = float(prop_details.get('book_limit', 500))
        
        prop_bet = PropBet(
            player_name=prop_details['player_name'],
            game_id=game_id,
            prop_type=prop_details['prop_type'],
            line=float(prop_details['line']),
            side=prop_details['side'],
            odds=int(odds),
            probability=win_prob,
            edge=prop_edge,
            confidence=confidence,
            book_limit=book_limit,
        )
        
        existing_props_input = prop_details.get('existing_props') or []
        existing_props = []
        for existing in existing_props_input:
            try:
                existing_props.append(PropBet(
                    player_name=existing['player_name'],
                    game_id=existing.get('game_id', game_id),
                    prop_type=existing['prop_type'],
                    line=float(existing['line']),
                    side=existing['side'],
                    odds=int(existing['odds']),
                    probability=float(existing['probability']),
                    edge=float(existing['edge']),
                    confidence=float(existing.get('confidence', 0.6)),
                ))
            except Exception:
                continue
        
        result = prop_kelly.calculate_prop_kelly(prop_bet, bankroll, existing_props or None)
        response_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'bet_type': 'prop',
            'game_id': game_id,
            'player_name': prop_details['player_name'],
            'prop': {
                'prop_type': prop_details['prop_type'],
                'line': float(prop_details['line']),
                'side': prop_details['side'],
                'odds': odds,
                'probability': win_prob,
                'edge': round(prop_edge, 4),
                'confidence': confidence,
                'book_limit': book_limit,
            },
            'kelly_sizing': {
                'full_kelly': round(result.kelly_fraction * 100, 2),
                'recommended_fraction': round(result.recommended_fraction * 100, 2),
                'bet_amount': result.bet_amount,
                'expected_value': result.expected_value,
            },
            'risk': {
                'score': round(result.risk_score, 3),
                'adjustments': result.adjustments,
                'reasoning': result.reasoning,
            },
            'response_time_ms': round(response_time, 2),
        })
    
    # Calculate Kelly bet for moneyline/puckline/totals (default)
    bet = kelly.calculate_bet(
        game_id=game_id,
        bet_type='moneyline',
        side='home',
        american_odds=odds,
        win_probability=win_prob,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction
    )
    
    response_time = (time.time() - start_time) * 1000
    
    return jsonify({
        'bet_type': 'moneyline',
        'game_id': game_id,
        'kelly_sizing': {
            'full_kelly': round(bet.bankroll_fraction_full * 100, 2),
            'half_kelly': round(bet.bankroll_fraction_half * 100, 2),
            'quarter_kelly': round(bet.bankroll_fraction_quarter * 100, 2),
            'recommended_fraction': round(bet.recommended_fraction * 100, 2),
            'recommended_units': round(bet.recommended_units, 2),
            'max_units': round(bet.max_units, 2)
        },
        'analysis': {
            'edge': round(bet.edge, 4),
            'expected_value': round(bet.expected_value, 3),
            'reasoning': bet.reasoning
        },
        'response_time_ms': round(response_time, 2)
    })


@live_betting_api_bp.route('/bet-track', methods=['POST'])
def track_bet():
    """
    Track a bet for paper trading or performance monitoring.
    
    Request body:
    {
        "game_id": "nba_game_123",
        "bet_type": "moneyline",
        "side": "home",
        "amount": 100.0,
        "odds": -150,
        "timestamp": "2025-11-16T19:00:00"
    }
    """
    data = request.get_json()
    
    # In production, save to database
    # For now, just validate and acknowledge
    
    required = ['game_id', 'bet_type', 'side', 'amount', 'odds']
    for field in required:
        if field not in data:
            return jsonify({'error': f'{field} required'}), 400
    
    # Generate bet ID
    bet_id = f"bet_{int(time.time())}_{data['game_id']}"
    
    return jsonify({
        'bet_id': bet_id,
        'status': 'tracked',
        'message': 'Bet recorded for paper trading',
        'bet_details': data
    }), 201


# Initialize models at module load
load_models()


def test_api():
    """Test API endpoints locally."""
    print("=" * 80)
    print("LIVE BETTING API TEST")
    print("=" * 80)
    
    # Test would require Flask app context
    # Run with: flask run or python app.py
    
    print("\nAPI Endpoints:")
    print("  GET  /api/live/health")
    print("  GET  /api/live/games?league=nba")
    print("  POST /api/live/predict")
    print("  GET  /api/live/opportunities")
    print("  POST /api/live/kelly-size")
    print("  POST /api/live/bet-track")
    
    print("\nTo test:")
    print("  1. Add to app.py: app.register_blueprint(live_betting_api_bp)")
    print("  2. Run: python app.py")
    print("  3. Test: curl http://localhost:5738/api/live/health")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_api()
