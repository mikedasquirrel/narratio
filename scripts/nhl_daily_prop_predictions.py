"""
NHL Daily Prop Predictions

Generates player prop betting recommendations for today's NHL games.
Integrates with the narrative optimization framework to find edges.

Workflow:
1. Fetch today's NHL games
2. Collect player data for top players
3. Extract narrative features
4. Run prop models
5. Fetch current prop odds
6. Calculate edges
7. Save recommendations

Run daily before games start.

Author: Prop Betting System
Date: November 20, 2024
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.nhl_player_data_collector import NHLPlayerDataCollector
from narrative_optimization.src.transformers.sports.nhl_player_performance import NHLPlayerPerformanceTransformer
from narrative_optimization.domains.nhl.nhl_prop_models import NHLPropModelSuite
from scripts.nhl_fetch_prop_odds import NHLPropOddsFetcher
from scripts.nhl_fetch_live_odds import NHLOddsFetcher


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'=' * 80}")
    print(f"{text}")
    print(f"{'=' * 80}")


def fetch_todays_games() -> List[Dict]:
    """Fetch today's NHL games"""
    print_header("FETCHING TODAY'S NHL GAMES")
    
    # Use the main odds fetcher
    odds_fetcher = NHLOddsFetcher()
    games = odds_fetcher.fetch_upcoming_games()
    
    # Filter to today only
    today = datetime.now().strftime('%Y-%m-%d')
    todays_games = []
    
    for game in games:
        if game.get('commence_time', '').startswith(today):
            todays_games.append(game)
            
    print(f"✓ Found {len(todays_games)} games today")
    
    for game in todays_games:
        print(f"  {game['away_team']} @ {game['home_team']} - {game['commence_time']}")
        
    return todays_games


def collect_player_data(games: List[Dict]) -> Dict:
    """Collect player data for all games"""
    print_header("COLLECTING PLAYER DATA")
    
    collector = NHLPlayerDataCollector()
    all_player_data = []
    
    for game in games:
        home = game['home_team']
        away = game['away_team']
        game_id = f"{datetime.now().strftime('%Y%m%d')}-{away}-{home}"
        
        print(f"\nCollecting data for {away} @ {home}...")
        
        # Collect top players
        players = collector.collect_players_for_game(home, away, top_n_players=8)
        
        # Add game context
        for side in ['home', 'away']:
            for player in players[side]:
                player['game_id'] = game_id
                player['is_home_game'] = (side == 'home')
                player['opponent'] = away if side == 'home' else home
                player['is_division_rival'] = False  # Would need division data
                player['is_national_tv'] = False  # Would need TV schedule
                player['days_rest'] = 1  # Would need schedule analysis
                
                all_player_data.append(player)
                
        print(f"  ✓ Collected {len(players['home']) + len(players['away'])} players")
        
    return {
        'players': all_player_data,
        'games': games
    }


def generate_prop_predictions(player_data: Dict) -> List[Dict]:
    """Generate prop predictions using narrative features"""
    print_header("GENERATING PROP PREDICTIONS")
    
    # Initialize components
    transformer = NHLPlayerPerformanceTransformer()
    prop_suite = NHLPropModelSuite()
    
    # Load trained models
    model_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'nhl' / 'models' / 'props'
    
    # For demo, we'll use mock models
    print("⚠️  Using mock models for demo (would load real trained models)")
    
    # Extract features
    print("\nExtracting narrative features...")
    player_features = transformer.transform(player_data['players'])
    print(f"✓ Extracted {player_features.shape[1]} features for {len(player_data['players'])} players")
    
    # Generate predictions (mock for now)
    predictions = []
    
    for i, player in enumerate(player_data['players']):
        # Mock predictions based on narrative features
        features = player_features[i]
        
        # Star power drives goal props
        star_power = features[0]  # First feature is star name value
        momentum = features[5]   # Scoring surge feature
        
        # Goals
        goal_prob_o05 = 0.35 + star_power * 0.3 + momentum * 0.2
        goal_prob_o15 = 0.15 + star_power * 0.2 + momentum * 0.15
        
        # Assists
        assist_prob_o05 = 0.40 + star_power * 0.2
        
        # Shots
        shot_base = 2.5 + star_power * 2.0
        shot_prob_o25 = 0.5 + (shot_base - 2.5) * 0.1
        shot_prob_o35 = 0.3 + (shot_base - 3.5) * 0.1
        
        # Points
        point_prob_o05 = min(0.95, goal_prob_o05 + assist_prob_o05 - 0.2)
        point_prob_o15 = goal_prob_o15 + assist_prob_o05 * 0.3
        
        # Add predictions
        for prop_type, line, prob in [
            ('goals', 0.5, goal_prob_o05),
            ('goals', 1.5, goal_prob_o15),
            ('assists', 0.5, assist_prob_o05),
            ('shots', 2.5, shot_prob_o25),
            ('shots', 3.5, shot_prob_o35),
            ('points', 0.5, point_prob_o05),
            ('points', 1.5, point_prob_o15),
        ]:
            predictions.append({
                'player_name': player['player_name'],
                'player_id': player['player_id'],
                'game_id': player['game_id'],
                'position': player['position'],
                'prop_type': prop_type,
                'line': line,
                'prob_over': np.clip(prob, 0.05, 0.95),
                'prob_under': np.clip(1 - prob, 0.05, 0.95),
                'confidence': abs(prob - 0.5) * 2,
                'star_power': star_power,
                'momentum': momentum,
            })
            
    print(f"\n✓ Generated {len(predictions)} prop predictions")
    
    # Show sample
    print("\nSample predictions:")
    for pred in predictions[:5]:
        print(f"  {pred['player_name']} {pred['prop_type']} o{pred['line']}: "
              f"{pred['prob_over']:.3f} ({pred['confidence']:.3f} conf)")
        
    return predictions


def fetch_prop_odds_and_calculate_edges(predictions: List[Dict]) -> List[Dict]:
    """Fetch current prop odds and calculate edges"""
    print_header("CALCULATING PROP EDGES")
    
    odds_fetcher = NHLPropOddsFetcher()
    
    # Fetch all props
    print("\nFetching current prop odds...")
    all_props = odds_fetcher.fetch_all_nhl_props()
    
    if not all_props:
        print("⚠️  No prop odds available - using mock data")
        # Generate mock odds
        all_props = _generate_mock_prop_odds(predictions)
        
    # Calculate edges
    edges = []
    
    for game_id, game_props in all_props.items():
        # Get predictions for this game
        game_predictions = [p for p in predictions if p['game_id'] in game_props.get('game_id', '')]
        
        if not game_predictions:
            continue
            
        # Match predictions to odds
        for pred in game_predictions:
            # Find matching prop odds
            matching_props = [
                prop for prop in game_props.get('props', [])
                if prop['player_name'] == pred['player_name']
                and prop['market'].replace('player_', '').replace('_over_under', '') == pred['prop_type']
                and prop['line'] == pred['line']
            ]
            
            if not matching_props:
                continue
                
            # Get best odds
            over_prop = next((p for p in matching_props if p['side'] == 'over'), None)
            under_prop = next((p for p in matching_props if p['side'] == 'under'), None)
            
            if over_prop and under_prop:
                # Calculate implied probabilities
                over_implied = odds_fetcher.american_to_probability(over_prop['odds'])
                under_implied = odds_fetcher.american_to_probability(under_prop['odds'])
                
                # Calculate edges
                over_edge = pred['prob_over'] - over_implied
                under_edge = pred['prob_under'] - under_implied
                
                # Pick best side
                if over_edge > under_edge and over_edge > 0.03:  # 3% minimum edge
                    edge = {
                        **pred,
                        'side': 'over',
                        'odds': over_prop['odds'],
                        'implied_prob': over_implied,
                        'edge': over_edge,
                        'edge_pct': over_edge * 100,
                        'bookmaker': over_prop['bookmaker'],
                        'expected_value': over_edge * _calculate_payout(over_prop['odds']),
                    }
                    edges.append(edge)
                elif under_edge > 0.03:
                    edge = {
                        **pred,
                        'side': 'under',
                        'odds': under_prop['odds'],
                        'implied_prob': under_implied,
                        'edge': under_edge,
                        'edge_pct': under_edge * 100,
                        'bookmaker': under_prop['bookmaker'],
                        'expected_value': under_edge * _calculate_payout(under_prop['odds']),
                    }
                    edges.append(edge)
                    
    # Sort by edge
    edges.sort(key=lambda x: x['edge'], reverse=True)
    
    print(f"\n✓ Found {len(edges)} positive edge props")
    
    return edges


def _generate_mock_prop_odds(predictions: List[Dict]) -> Dict:
    """Generate mock prop odds for testing"""
    mock_odds = {}
    
    # Group by game
    games = {}
    for pred in predictions:
        if pred['game_id'] not in games:
            games[pred['game_id']] = []
        games[pred['game_id']].append(pred)
        
    for game_id, game_preds in games.items():
        props = []
        
        for pred in game_preds:
            # Generate slightly off odds to create edges
            true_prob = pred['prob_over']
            
            # Add some noise and vig
            book_prob = true_prob + np.random.normal(0, 0.05)
            book_prob = np.clip(book_prob, 0.1, 0.9)
            
            # Convert to American odds with vig
            vig = 0.05
            over_prob_with_vig = book_prob + vig/2
            under_prob_with_vig = (1 - book_prob) + vig/2
            
            # Convert to American
            if over_prob_with_vig >= 0.5:
                over_odds = -int(over_prob_with_vig / (1 - over_prob_with_vig) * 100)
            else:
                over_odds = int((1 - over_prob_with_vig) / over_prob_with_vig * 100)
                
            if under_prob_with_vig >= 0.5:
                under_odds = -int(under_prob_with_vig / (1 - under_prob_with_vig) * 100)
            else:
                under_odds = int((1 - under_prob_with_vig) / under_prob_with_vig * 100)
                
            props.extend([
                {
                    'player_name': pred['player_name'],
                    'market': f"player_{pred['prop_type']}_over_under",
                    'line': pred['line'],
                    'side': 'over',
                    'odds': over_odds,
                    'bookmaker': 'DraftKings',
                },
                {
                    'player_name': pred['player_name'],
                    'market': f"player_{pred['prop_type']}_over_under",
                    'line': pred['line'],
                    'side': 'under',
                    'odds': under_odds,
                    'bookmaker': 'DraftKings',
                }
            ])
            
        mock_odds[f"mock_{game_id}"] = {
            'game_id': game_id,
            'props': props
        }
        
    return mock_odds


def _calculate_payout(odds: int) -> float:
    """Calculate payout for American odds"""
    if odds > 0:
        return odds / 100
    else:
        return 100 / abs(odds)


def save_predictions(edges: List[Dict], output_dir: Path):
    """Save prop predictions to file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Save full predictions
    output_file = output_dir / 'nhl_prop_predictions.json'
    
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'n_predictions': len(edges),
        'predictions': edges,
        'summary': {
            'by_prop_type': {},
            'by_tier': {},
            'top_edges': edges[:10]
        }
    }
    
    # Summarize by prop type
    for edge in edges:
        prop_type = edge['prop_type']
        if prop_type not in output_data['summary']['by_prop_type']:
            output_data['summary']['by_prop_type'][prop_type] = {
                'count': 0,
                'avg_edge': 0
            }
        output_data['summary']['by_prop_type'][prop_type]['count'] += 1
        
    # Calculate averages
    for prop_type in output_data['summary']['by_prop_type']:
        type_edges = [e for e in edges if e['prop_type'] == prop_type]
        if type_edges:
            output_data['summary']['by_prop_type'][prop_type]['avg_edge'] = \
                np.mean([e['edge_pct'] for e in type_edges])
                
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n✓ Saved predictions to {output_file}")
    
    # Also save timestamped version
    archive_file = output_dir / f'nhl_props_{timestamp}.json'
    with open(archive_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def display_top_props(edges: List[Dict], n: int = 20):
    """Display top prop bets"""
    print_header(f"TOP {n} PROP BETS")
    
    tiers = {'elite': '🔥', 'strong': '💪', 'moderate': '📊', 'speculative': '🎲'}
    
    for i, edge in enumerate(edges[:n]):
        # Calculate tier
        if edge['edge_pct'] >= 8 and edge['confidence'] >= 0.65:
            tier = 'elite'
        elif edge['edge_pct'] >= 6 and edge['confidence'] >= 0.60:
            tier = 'strong'
        elif edge['edge_pct'] >= 4 and edge['confidence'] >= 0.55:
            tier = 'moderate'
        else:
            tier = 'speculative'
            
        tier_icon = tiers[tier]
        
        print(f"\n{i+1}. {tier_icon} {edge['player_name']} - {edge['prop_type'].upper()} "
              f"{edge['side'].upper()} {edge['line']}")
        print(f"   Edge: {edge['edge_pct']:.1f}% | "
              f"Our prob: {edge['prob_over' if edge['side'] == 'over' else 'prob_under']:.1%} | "
              f"Implied: {edge['implied_prob']:.1%}")
        print(f"   Odds: {edge['odds']:+d} @ {edge['bookmaker']} | "
              f"EV: ${edge['expected_value']:.2f} per $1")
        print(f"   Confidence: {edge['confidence']:.3f} | "
              f"Star power: {edge.get('star_power', 0):.2f}")


def main():
    """Main execution"""
    print("\nNHL DAILY PROP PREDICTIONS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Fetch today's games
    games = fetch_todays_games()
    
    if not games:
        print("\n⚠️  No NHL games today")
        return
        
    # Collect player data
    player_data = collect_player_data(games)
    
    # Generate predictions
    predictions = generate_prop_predictions(player_data)
    
    # Calculate edges
    edges = fetch_prop_odds_and_calculate_edges(predictions)
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'analysis'
    save_predictions(edges, output_dir)
    
    # Display top props
    display_top_props(edges)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total props analyzed: {len(predictions)}")
    print(f"Props with edge: {len(edges)}")
    print(f"Average edge: {np.mean([e['edge_pct'] for e in edges]):.1f}%")
    
    # By prop type
    print("\nBy prop type:")
    prop_types = {}
    for edge in edges:
        pt = edge['prop_type']
        if pt not in prop_types:
            prop_types[pt] = []
        prop_types[pt].append(edge['edge_pct'])
        
    for prop_type, edge_pcts in prop_types.items():
        print(f"  {prop_type}: {len(edge_pcts)} props, {np.mean(edge_pcts):.1f}% avg edge")
        
    print("\n✓ Daily prop prediction complete")
    

if __name__ == "__main__":
    main()
