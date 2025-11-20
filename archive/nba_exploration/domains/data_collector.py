"""
NBA Data Collection Module

Collects game data, team narratives, and betting information
for narrative-driven prediction analysis.
"""

from typing import List, Dict, Any, Optional
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


class NBADataCollector:
    """
    Collects NBA game data including narratives and betting information.
    
    For MVP/Demo: Uses synthetic realistic data
    For Production: Would integrate with NBA API, Basketball Reference, etc.
    """
    
    def __init__(self, seasons: Optional[List[int]] = None):
        """
        Initialize NBA data collector.
        
        Parameters
        ----------
        seasons : list of int, optional
            Seasons to collect (e.g., [2015, 2016, 2017])
            If None, defaults to last 15 seasons
        """
        self.seasons = seasons or list(range(2010, 2025))
        self.teams = self._get_nba_teams()
        
    def _get_nba_teams(self) -> Dict[str, Dict]:
        """Get NBA team information with narrative profiles."""
        return {
            'LAL': {
                'name': 'Los Angeles Lakers',
                'city': 'Los Angeles',
                'conference': 'West',
                'narrative': 'Legendary franchise with championship pedigree and star power',
                'identity_markers': ['champions', 'historic', 'elite', 'legendary']
            },
            'BOS': {
                'name': 'Boston Celtics',
                'city': 'Boston',
                'conference': 'East',
                'narrative': 'Storied tradition with most championships in NBA history',
                'identity_markers': ['tradition', 'championship', 'pride', 'legacy']
            },
            'GSW': {
                'name': 'Golden State Warriors',
                'city': 'San Francisco',
                'conference': 'West',
                'narrative': 'Modern dynasty with innovative play style and championship success',
                'identity_markers': ['dynasty', 'innovative', 'modern', 'championship']
            },
            'PHI': {
                'name': 'Philadelphia 76ers',
                'city': 'Philadelphia',
                'conference': 'East',
                'narrative': 'Young team building towards championship contention',
                'identity_markers': ['rising', 'potential', 'building', 'emerging']
            },
            'MIL': {
                'name': 'Milwaukee Bucks',
                'city': 'Milwaukee',
                'conference': 'East',
                'narrative': 'Championship caliber team led by elite superstar',
                'identity_markers': ['champion', 'dominant', 'elite', 'powerful']
            },
            'BKN': {
                'name': 'Brooklyn Nets',
                'city': 'Brooklyn',
                'conference': 'East',
                'narrative': 'Star-powered team seeking championship glory',
                'identity_markers': ['star-powered', 'ambitious', 'talented', 'driven']
            },
            'MIA': {
                'name': 'Miami Heat',
                'city': 'Miami',
                'conference': 'East',
                'narrative': 'Competitive culture built on toughness and discipline',
                'identity_markers': ['culture', 'tough', 'disciplined', 'competitive']
            },
            'DEN': {
                'name': 'Denver Nuggets',
                'city': 'Denver',
                'conference': 'West',
                'narrative': 'Ascending team with MVP-level talent and teamwork',
                'identity_markers': ['ascending', 'teamwork', 'talented', 'growing']
            },
            'PHX': {
                'name': 'Phoenix Suns',
                'city': 'Phoenix',
                'conference': 'West',
                'narrative': 'Resurgent franchise with championship aspirations',
                'identity_markers': ['resurgent', 'aspiring', 'improving', 'hungry']
            },
            'DAL': {
                'name': 'Dallas Mavericks',
                'city': 'Dallas',
                'conference': 'West',
                'narrative': 'Veteran leadership with championship experience',
                'identity_markers': ['veteran', 'experienced', 'championship', 'leader']
            }
        }
    
    def fetch_games(self, include_narratives: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch game data for configured seasons.
        
        Parameters
        ----------
        include_narratives : bool
            Whether to include narrative text data
        
        Returns
        -------
        games : list of dict
            Game data with narratives
        """
        games = []
        team_ids = list(self.teams.keys())
        
        for season in self.seasons:
            # ~82 games per team, ~1230 total games per season
            # For demo, generate representative sample
            n_games = 200  # Sample size
            
            for game_id in range(n_games):
                # Random matchup
                home_team = random.choice(team_ids)
                away_team = random.choice([t for t in team_ids if t != home_team])
                
                # Generate realistic game data
                game = self._generate_game(season, game_id, home_team, away_team, include_narratives)
                games.append(game)
        
        return games
    
    def _generate_game(self, season: int, game_id: int, home_team: str, away_team: str, include_narratives: bool) -> Dict:
        """Generate realistic game data with narratives."""
        home_info = self.teams[home_team]
        away_info = self.teams[away_team]
        
        # Simulate outcome (with some narrative influence)
        home_narrative_strength = self._compute_narrative_strength(home_info)
        away_narrative_strength = self._compute_narrative_strength(away_info)
        
        # Add home court advantage
        home_advantage = 0.1
        home_win_prob = 0.5 + (home_narrative_strength - away_narrative_strength) * 0.3 + home_advantage
        
        home_wins = random.random() < home_win_prob
        
        # Generate scores
        base_score = random.randint(95, 115)
        margin = random.randint(1, 20)
        
        if home_wins:
            home_score = base_score + margin // 2
            away_score = base_score - margin // 2
        else:
            home_score = base_score - margin // 2
            away_score = base_score + margin // 2
        
        # Generate betting line (correlated with but not perfectly aligned with narrative)
        betting_line = (home_narrative_strength - away_narrative_strength) * 10 + random.uniform(-3, 3)
        
        game_data = {
            'game_id': f"{season}_{game_id}",
            'season': season,
            'date': datetime(season, random.randint(10, 12), random.randint(1, 28)).isoformat(),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'home_wins': home_wins,
            'margin': abs(home_score - away_score),
            'betting_line': round(betting_line, 1),
            'total': home_score + away_score
        }
        
        if include_narratives:
            game_data['home_narrative'] = self._generate_team_narrative(home_info, season, True)
            game_data['away_narrative'] = self._generate_team_narrative(away_info, season, False)
            game_data['matchup_narrative'] = self._generate_matchup_narrative(home_info, away_info)
        
        return game_data
    
    def _compute_narrative_strength(self, team_info: Dict) -> float:
        """Compute rough narrative strength score (0-1) based on team identity."""
        championship_words = ['champion', 'champions', 'championship', 'dynasty', 'elite', 'legendary']
        narrative = team_info['narrative'].lower()
        
        strength = sum(1 for word in championship_words if word in narrative)
        strength += len(team_info['identity_markers']) * 0.1
        
        return min(strength / 5, 1.0)  # Normalize to 0-1
    
    def _generate_team_narrative(self, team_info: Dict, season: int, is_home: bool) -> str:
        """Generate realistic team narrative for a game."""
        templates = [
            f"The {team_info['name']} {random.choice(['enter tonight with', 'bring', 'come into this game with', 'face off with'])} {random.choice(['momentum', 'confidence', 'determination', 'energy'])}. {team_info['narrative']}. {'Playing at home gives them an edge.' if is_home else 'On the road, they look to prove themselves.'} {random.choice(['Their recent performance shows', 'The team has been', 'They are currently'])} {random.choice(['improving', 'dominating', 'competitive', 'fighting hard'])}.",
            
            f"{team_info['name']} {'have' if is_home else 'bring'} {random.choice(['championship aspirations', 'playoff intensity', 'winning mentality', 'competitive fire'])} to tonight's matchup. {team_info['narrative']}. The {random.choice(['coaching staff', 'locker room', 'organization', 'fanbase'])} {random.choice(['believes in', 'expects', 'anticipates', 'demands'])} victory.",
            
            f"With {random.choice(['strong execution', 'focused intensity', 'championship pedigree', 'elite talent'])}, the {team_info['name']} aim to {random.choice(['dominate', 'win convincingly', 'control the game', 'establish superiority'])}. {team_info['narrative']}.",
        ]
        
        return random.choice(templates)
    
    def _generate_matchup_narrative(self, home_info: Dict, away_info: Dict) -> str:
        """Generate narrative framing for the matchup."""
        templates = [
            f"This matchup pits the {home_info['name']} against the {away_info['name']} in what promises to be {random.choice(['an intense battle', 'a competitive showdown', 'a must-watch game', 'a pivotal contest'])}.",
            
            f"When the {home_info['name']} host the {away_info['name']}, {random.choice(['championship implications', 'playoff positioning', 'pride', 'momentum'])} are on the line.",
            
            f"The {away_info['name']} travel to face the {home_info['name']} in a game that could {random.choice(['define their season', 'shift momentum', 'prove their worth', 'establish dominance'])}."
        ]
        
        return random.choice(templates)
    
    def split_train_test(self, games: List[Dict], test_every_nth: int = 10) -> tuple:
        """
        Split games temporally: exclude every Nth season for testing.
        
        Parameters
        ----------
        games : list of dict
            All game data
        test_every_nth : int
            Test on every Nth season (default: 10)
        
        Returns
        -------
        train_games, test_games : tuple of lists
            Training and testing splits
        """
        train_games = []
        test_games = []
        
        for game in games:
            season = game['season']
            # Determine if this season is a test season
            # Season 2010 → 201 → 201 % 10 = 1 (train)
            # Season 2020 → 202 → 202 % 10 = 2 (train)
            # But we want actual every 10th in sequence
            
            # Get season index (0-based from first season)
            season_index = season - min(self.seasons)
            
            if (season_index + 1) % test_every_nth == 0:
                test_games.append(game)
            else:
                train_games.append(game)
        
        return train_games, test_games
    
    def get_team_season_narrative(self, team_id: str, season: int) -> Dict[str, Any]:
        """
        Get comprehensive narrative for a team in a specific season.
        
        Parameters
        ----------
        team_id : str
            Team identifier (e.g., 'LAL')
        season : int
            Season year
        
        Returns
        -------
        narrative_data : dict
            Comprehensive team narrative information
        """
        if team_id not in self.teams:
            return {}
        
        team = self.teams[team_id]
        
        # Generate season-specific narrative
        season_narrative = f"""
        The {team['name']} enter the {season} season {random.choice(['with high expectations', 'looking to prove themselves', 'as championship contenders', 'seeking to improve'])}.
        
        {team['narrative']}
        
        Key strengths: {random.choice(['Offensive firepower', 'Defensive intensity', 'Veteran leadership', 'Young talent', 'Team chemistry'])} and {random.choice(['clutch performance', 'depth', 'coaching', 'home court advantage'])}.
        
        {random.choice(['They aim to compete for a championship.', 'Playoff contention is the goal.', 'Building for the future remains the focus.', 'Win-now mentality drives the organization.'])}
        """
        
        return {
            'team_id': team_id,
            'team_name': team['name'],
            'season': season,
            'narrative': season_narrative.strip(),
            'identity_markers': team['identity_markers'],
            'conference': team['conference'],
            'estimated_wins': random.randint(35, 60),  # Realistic range
            'playoff_odds': random.uniform(0.3, 0.9)
        }
    
    def save_to_json(self, games: List[Dict], filepath: str):
        """Save collected games to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(games, f, indent=2)
    
    def load_from_json(self, filepath: str) -> List[Dict]:
        """Load games from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


class NBAAPICollector(NBADataCollector):
    """
    Extended collector that would integrate with real NBA APIs.
    
    For production use, this would connect to:
    - NBA Stats API
    - Basketball Reference
    - ESPN API
    - Betting data providers
    
    Currently implements synthetic data for demonstration.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
    
    def fetch_real_games(self, season: int) -> List[Dict]:
        """
        Fetch real game data from NBA API.
        
        Note: For production, implement actual API calls here.
        For demo, returns synthetic data.
        """
        # TODO: Implement real API integration
        # from nba_api.stats.endpoints import leaguegamefinder
        # games = leaguegamefinder.LeagueGameFinder(season_nullable=season).get_data_frames()
        
        return self.fetch_games()  # Use synthetic for now
    
    def fetch_team_narratives_from_web(self, team_id: str, season: int) -> str:
        """
        Scrape team narratives from sports media websites.
        
        Note: For production, implement web scraping here.
        """
        # TODO: Implement web scraping
        # - ESPN team pages
        # - Basketball Reference team pages
        # - Team official websites
        # - Sports news articles
        
        return self.get_team_season_narrative(team_id, season)['narrative']

