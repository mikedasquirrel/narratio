"""
Universal Domain Registry

Central registry for ALL narrative domains.
Makes adding new domains TRIVIAL - just add entry here.

Each domain specifies:
- Where data lives
- How to extract narratives
- How to extract outcomes
- Estimated π (narrativity)
- Expected pattern count

Adding new domain = 5 lines of config.

Author: Narrative Optimization Framework
Date: November 2025
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable, Iterable
import json
import numpy as np
from textwrap import shorten

from narrative_optimization.domains.hurricanes.narrative_extractor import (
    extract_hurricane_narratives,
)
from narrative_optimization.domains.nhl.checkpoint_narratives import (
    build_nhl_checkpoint_snapshots,
)
from narrative_optimization.domains.supreme_court.narrative_extractor import (
    extract_supreme_court_narratives,
)
from narrative_optimization.domains.stereotropes.narrative_extractor import (
    extract_stereotropes,
)
from narrative_optimization.domains.ml_research.narrative_extractor import (
    extract_research_narratives,
)
from narrative_optimization.domains.wikiplots.narrative_extractor import (
    extract_wikiplots,
)


def extract_wwe_data(data):
    """Extract WWE storyline narratives from CSV data."""
    import csv
    from pathlib import Path
    
    # data parameter is ignored for CSV; we load directly from file
    csv_path = Path('narrative_optimization/domains/wwe/data/wwe_storylines.csv')
    if not csv_path.exists():
        return [], np.array([]), 0
    
    narratives = []
    outcomes = []
    total_count = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            # Synthesize narrative from structured data
            storyline_type = row['storyline_type'].replace('_', ' ').title()
            participants = row['participants']
            duration = row['duration_weeks']
            
            narrative = (
                f"{storyline_type} storyline featuring {participants}. "
                f"Duration: {duration} weeks. "
                f"Character quality: {float(row['character_quality']):.2f}, "
                f"Plot quality: {float(row['plot_quality']):.2f}, "
                f"Promo quality: {float(row['promo_quality']):.2f}, "
                f"Star power: {float(row['star_power']):.2f}. "
                f"Narrative quality score: {float(row['narrative_quality_yu']):.3f}."
            )
            
            narratives.append(narrative)
            outcomes.append(float(row['engagement']))
    
    return narratives, np.array(outcomes), total_count


def extract_novels_data(data):
    """Extract novel narratives from JSON data."""
    narratives = []
    outcomes = []
    
    for item in data:
        # Use full_narrative if available, otherwise plot_summary
        narrative = item.get('full_narrative') or item.get('plot_summary', '')
        if not narrative or len(narrative) < 100:
            continue
        
        outcome = item.get('critical_acclaim_score')
        if outcome is None:
            continue
        
        narratives.append(narrative)
        outcomes.append(float(outcome))
    
    return narratives, np.array(outcomes), len(data)


def extract_nonfiction_data(data):
    """Extract nonfiction book narratives from JSON data."""
    narratives = []
    outcomes = []
    
    for item in data:
        # Use full_narrative if available, otherwise description
        narrative = item.get('full_narrative') or item.get('description', '')
        if not narrative or len(narrative) < 100:
            continue
        
        outcome = item.get('critical_acclaim_score')
        if outcome is None:
            continue
        
        narratives.append(narrative)
        outcomes.append(float(outcome))
    
    return narratives, np.array(outcomes), len(data)


def extract_tennis_data(data):
    """Extract tennis match narratives from structured data."""
    narratives = []
    outcomes = []
    total_count = len(data) if data else 0
    
    for item in data:
        try:
            # Extract player data
            p1 = item.get('player1', {})
            p2 = item.get('player2', {})
            p1_name = p1.get('name', 'Unknown')
            p2_name = p2.get('name', 'Unknown')
            
            # Skip if missing key data
            if not p1_name or not p2_name:
                continue
            
            # Build rich narrative
            tournament = item.get('tournament', 'Unknown')
            surface = item.get('surface', 'unknown')
            round_name = item.get('round', 'unknown')
            level = item.get('level', 'unknown')
            
            narrative = f"Tennis match: {p1_name} vs {p2_name} at {tournament} ({level})."
            narrative += f" Surface: {surface}. Round: {round_name}."
            
            # Add rankings
            p1_rank = p1.get('ranking')
            p2_rank = p2.get('ranking')
            if p1_rank and p2_rank:
                narrative += f" Rankings: {p1_name} #{p1_rank}, {p2_name} #{p2_rank}."
                rank_diff = abs(p1_rank - p2_rank)
                if rank_diff > 50:
                    narrative += f" Significant ranking gap: {rank_diff} positions."
            
            # Add seeds if available
            p1_seed = p1.get('seed')
            p2_seed = p2.get('seed')
            if p1_seed or p2_seed:
                seeds = []
                if p1_seed:
                    seeds.append(f"{p1_name} seeded #{p1_seed}")
                if p2_seed:
                    seeds.append(f"{p2_name} seeded #{p2_seed}")
                narrative += f" {', '.join(seeds)}."
            
            # Add head-to-head if available
            h2h = item.get('head_to_head', {})
            if h2h and isinstance(h2h, dict):
                p1_wins = h2h.get('player1_wins', 0)
                p2_wins = h2h.get('player2_wins', 0)
                if p1_wins + p2_wins > 0:
                    narrative += f" Head-to-head: {p1_name} {p1_wins}-{p2_wins} {p2_name}."
            
            # Add player attributes
            p1_hand = p1.get('hand')
            p2_hand = p2.get('hand')
            if p1_hand and p2_hand:
                narrative += f" {p1_name} plays {p1_hand}-handed, {p2_name} plays {p2_hand}-handed."
            
            # Add context if available
            context = item.get('context', '')
            if context and len(context) > 20:
                narrative += f" Context: {context[:200]}"
            
            if len(narrative) < 100:
                continue
            
            # Outcome
            outcome = 1 if item.get('player1_won') else 0
            
            narratives.append(narrative)
            outcomes.append(outcome)
        except Exception as e:
            continue
    
    return narratives, np.array(outcomes), total_count


def extract_mlb_data(data):
    """Extract MLB game narratives from structured data."""
    narratives = []
    outcomes = []
    total_count = len(data) if data else 0
    
    for item in data:
        try:
            # Use pre-built narrative if available
            narrative = item.get('narrative', '')
            if not narrative or len(narrative) < 100:
                # Build narrative from structured data
                home_team = item.get('home_team', {}).get('name', 'Unknown')
                away_team = item.get('away_team', {}).get('name', 'Unknown')
                venue = item.get('venue', {}).get('name', 'Unknown')
                
                narrative = f"MLB game: {away_team} at {home_team} ({venue})."
                
                # Add team records
                home_record = item.get('home_team', {}).get('record', {})
                away_record = item.get('away_team', {}).get('record', {})
                if home_record and away_record:
                    narrative += f" {home_team} ({home_record.get('wins')}-{home_record.get('losses')})"
                    narrative += f" vs {away_team} ({away_record.get('wins')}-{away_record.get('losses')})."
                
                # Add context if available
                context = item.get('context', '')
                if context and len(context) > 20:
                    narrative += f" {context}"
            
            if len(narrative) < 50:
                continue
            
            # Extract outcome
            outcome_data = item.get('outcome', {})
            winner = outcome_data.get('winner', '')
            
            if winner == 'home':
                outcome = 1
            elif winner == 'away':
                outcome = 0
            else:
                continue  # Skip ties/unknown
            
            narratives.append(narrative)
            outcomes.append(outcome)
        except Exception as e:
            continue
    
    return narratives, np.array(outcomes), total_count


def extract_ufc_data(data):
    """Extract UFC fight narratives from JSON data with rich stats."""
    narratives = []
    outcomes = []
    total_count = len(data) if data else 0
    
    for idx, item in enumerate(data):
        try:
            # Extract fighter data
            r_fighter = item.get('R_fighter', 'Unknown')
            b_fighter = item.get('B_fighter', 'Unknown')
            r_nickname = item.get('R_nickname', '')
            b_nickname = item.get('B_nickname', '')
            winner = item.get('Winner', '')
            
            # Skip if missing key data
            if not r_fighter or not b_fighter or not winner:
                continue
            
            # Build rich narrative
            r_desc = f"{r_fighter} '{r_nickname}'" if r_nickname else r_fighter
            b_desc = f"{b_fighter} '{b_nickname}'" if b_nickname else b_fighter
            
            weight_class = item.get('weight_class', 'unknown')
            title_bout = item.get('title_bout', False)
            event = item.get('event', 'UFC Event')
            
            if title_bout:
                narrative = f"UFC Title Fight ({weight_class}): {r_desc} vs {b_desc} at {event}."
            else:
                narrative = f"UFC {weight_class} bout: {r_desc} vs {b_desc} at {event}."
            
            # Add records and streaks
            r_wins = item.get('R_wins')
            r_losses = item.get('R_losses')
            b_wins = item.get('B_wins')
            b_losses = item.get('B_losses')
            if r_wins is not None and r_losses is not None:
                narrative += f" {r_fighter} record: {r_wins}-{r_losses}."
            if b_wins is not None and b_losses is not None:
                narrative += f" {b_fighter} record: {b_wins}-{b_losses}."
            
            # Add win streaks
            r_streak = item.get('R_current_win_streak')
            b_streak = item.get('B_current_win_streak')
            try:
                if r_streak and float(r_streak) > 2:
                    narrative += f" {r_fighter} on {int(float(r_streak))}-fight win streak."
            except (ValueError, TypeError):
                pass
            try:
                if b_streak and float(b_streak) > 2:
                    narrative += f" {b_fighter} on {int(float(b_streak))}-fight win streak."
            except (ValueError, TypeError):
                pass
            
            # Add physical attributes
            r_reach = item.get('R_Reach_cms')
            b_reach = item.get('B_Reach_cms')
            try:
                if r_reach and b_reach:
                    r_reach_f = float(r_reach)
                    b_reach_f = float(b_reach)
                    reach_diff = abs(r_reach_f - b_reach_f)
                    if reach_diff > 10:
                        longer = r_fighter if r_reach_f > b_reach_f else b_fighter
                        narrative += f" {longer} has {int(reach_diff)}cm reach advantage."
            except (ValueError, TypeError):
                pass
            
            # Add stance
            r_stance = item.get('R_Stance')
            b_stance = item.get('B_Stance')
            if r_stance and b_stance:
                if r_stance != b_stance:
                    narrative += f" Stance matchup: {r_fighter} ({r_stance}) vs {b_fighter} ({b_stance})."
            
            # Add fighting style stats
            r_sig_str = item.get('R_avg_SIG_STR_pct')
            b_sig_str = item.get('B_avg_SIG_STR_pct')
            try:
                if r_sig_str and b_sig_str:
                    narrative += f" Striking accuracy: {r_fighter} {float(r_sig_str):.1f}%, {b_fighter} {float(b_sig_str):.1f}%."
            except (ValueError, TypeError):
                pass
            
            r_td = item.get('R_avg_TD_pct')
            b_td = item.get('B_avg_TD_pct')
            try:
                if r_td and b_td:
                    narrative += f" Takedown success: {r_fighter} {float(r_td):.1f}%, {b_fighter} {float(b_td):.1f}%."
            except (ValueError, TypeError):
                pass
            
            r_sub = item.get('R_avg_SUB_ATT')
            b_sub = item.get('B_avg_SUB_ATT')
            try:
                if r_sub and b_sub:
                    r_sub_f = float(r_sub)
                    b_sub_f = float(b_sub)
                    if r_sub_f > 1.0 or b_sub_f > 1.0:
                        if r_sub_f > b_sub_f + 0.5:
                            narrative += f" {r_fighter} submission threat ({r_sub_f:.1f} attempts/fight)."
                        elif b_sub_f > r_sub_f + 0.5:
                            narrative += f" {b_fighter} submission threat ({b_sub_f:.1f} attempts/fight)."
            except (ValueError, TypeError):
                pass
            
            # Add odds
            r_odds = item.get('R_odds')
            b_odds = item.get('B_odds')
            if r_odds and b_odds:
                narrative += f" Betting odds: {r_fighter} {r_odds}, {b_fighter} {b_odds}."
            
            # Outcome: 1 if Red fighter won, 0 if Blue fighter won
            if winner == r_fighter:
                outcome = 1
            elif winner == b_fighter:
                outcome = 0
            else:
                continue  # Skip draws/no contests
            
            # Only check length after building full narrative
            if len(narrative) < 50:  # Lower threshold since we have rich stats
                continue
            
            narratives.append(narrative)
            outcomes.append(outcome)
        except Exception as e:
            continue
    
    return narratives, np.array(outcomes), total_count
from narrative_optimization.domains.nfl.checkpoint_narratives import (
    build_nfl_checkpoint_snapshots,
)
from narrative_optimization.domains.supreme_court.checkpoint_narratives import (
    build_supreme_court_checkpoint_snapshots,
)
from narrative_optimization.domain_targets import (
    build_nhl_game_win_row,
    build_nhl_roi_row,
    build_nfl_game_win_row,
    build_nfl_roi_row,
    build_nba_game_win_row,
    build_nba_margin_row,
    build_mlb_game_win_row,
    build_golf_win_row,
    build_startup_success_row,
    build_startup_funding_row,
    build_supreme_court_outcome_row,
    build_supreme_court_citation_row,
    build_wikiplots_impact_row,
    build_stereotropes_impact_row,
    build_ml_research_impact_row,
    build_cmu_movies_revenue_row,
)


@dataclass
class TargetConfig:
    """
    Declarative target definition for a domain.

    Each target represents a distinct prediction objective (e.g., game win,
    ATS cover, ROI percent, citation count).
    """

    name: str
    scope: str = "game"
    outcome_type: str = "binary"
    description: str = ""
    builder: Optional[Callable[..., Any]] = None
    enabled: bool = True
    requires_checkpoints: bool = False
    requires_odds: bool = False
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DomainConfig:
    """Configuration for a single domain."""
    
    def __init__(
        self,
        name: str,
        data_path: str,
        narrative_field: str,
        outcome_field: str,
        estimated_pi: float,
        description: str = "",
        outcome_type: str = 'continuous',
        min_narrative_length: int = 50,
        custom_extractor: Optional[Callable] = None,
        checkpoint_schema: Optional[List[str]] = None,
        checkpoint_fields: Optional[List[str]] = None,
        checkpoint_description: str = "",
        checkpoint_builder: Optional[
            Callable[[Iterable[Dict], Optional[str], Optional[int]], List[Dict]]
        ] = None,
        targets: Optional[List[TargetConfig]] = None,
    ):
        """
        Define domain configuration.
        
        Parameters
        ----------
        name : str
            Domain identifier (e.g., 'movies', 'nba')
        data_path : str
            Path to JSON data file
        narrative_field : str
            Field containing narrative text (e.g., 'plot_summary', 'description')
            Can be list of fields to try: ['field1', 'field2']
        outcome_field : str
            Field containing outcome (e.g., 'success_score', 'won')
        estimated_pi : float
            Domain's estimated narrativity (0-1)
        description : str
            Human-readable description
        outcome_type : str
            'binary' or 'continuous'
        min_narrative_length : int
            Minimum characters for valid narrative
        custom_extractor : callable, optional
            Custom function: extract_data(data) -> (narratives, outcomes)
        """
        self.name = name
        self.data_path = Path(data_path)
        self.narrative_field = narrative_field if isinstance(narrative_field, list) else [narrative_field]
        self.outcome_field = outcome_field
        self.estimated_pi = estimated_pi
        self.description = description
        self.outcome_type = outcome_type
        self.min_narrative_length = min_narrative_length
        self.custom_extractor = custom_extractor
        self.checkpoint_schema = checkpoint_schema or []
        self.checkpoint_fields = checkpoint_fields or []
        self.checkpoint_description = checkpoint_description
        self.checkpoint_builder = checkpoint_builder
        self.targets: List[TargetConfig] = targets or []
        self._legacy_target: Optional[TargetConfig] = None

    # ------------------------------------------------------------------ #
    # Target helpers
    # ------------------------------------------------------------------ #

    def has_targets(self) -> bool:
        """Whether the domain declares explicit TargetConfig entries."""
        return any(target.enabled for target in self.targets)

    def get_targets(self) -> List[TargetConfig]:
        """
        Return the enabled targets for the domain.
        Falls back to a synthetic legacy target when none are defined.
        """
        if self.has_targets():
            return [target for target in self.targets if target.enabled]
        return [self._get_legacy_target()]

    def get_primary_target(self) -> TargetConfig:
        """Convenience accessor for the first enabled target."""
        return self.get_targets()[0]

    def find_target(self, name: str) -> Optional[TargetConfig]:
        """Return the enabled target with the provided name, if present."""
        for target in self.get_targets():
            if target.name == name:
                return target
        return None

    def _get_legacy_target(self) -> TargetConfig:
        if self._legacy_target is None:
            self._legacy_target = TargetConfig(
                name=self.outcome_field or "legacy_target",
                scope="legacy",
                outcome_type=self.outcome_type,
                description="Auto-generated fallback target mirroring outcome_field.",
                builder=None,
                enabled=True,
            )
        return self._legacy_target
    
    def load_and_extract(self) -> Tuple[List, np.ndarray, int]:
        """
        Load data and extract narratives + outcomes.
        
        CRITICAL CHANGE (Nov 17, 2025):
        Returns STRUCTURED GENOMES (dicts) when available, not just text.
        
        Returns
        -------
        narratives : list of (str OR dict)
            - If item has rich structured data: returns full dict (THE GENOME)
            - If item is mainly text: returns text string
        outcomes : ndarray
        total_count : int (total items in dataset)
        """
        data = self._load_raw_records()
        
        # Use custom extractor if provided
        if self.custom_extractor:
            return self.custom_extractor(data)
        
        narratives = []
        outcomes = []
        
        for item in data:
            # Get outcome first (required either way)
            outcome = item.get(self.outcome_field)
            if outcome is None:
                continue
            
            # Check if this is a rich structured genome or mainly text
            has_rich_structure = self._has_rich_genome(item)
            
            if has_rich_structure:
                # For rich genomes: Use existing text field if available, or synthesize
                # This gives transformers text to work with while preserving genome structure
                
                # Try to get existing narrative text first
                narrative_text = None
                for field in self.narrative_field:
                    text = item.get(field, '')
                    if text and isinstance(text, str) and len(text) >= 50:
                        narrative_text = text
                        break
                
                # If no text, synthesize from genome
                if not narrative_text:
                    narrative_text = self._synthesize_text_from_genome(item)
                
                # CRITICAL: For now, pass TEXT but mark that genome is available
                # Future: Update transformers to extract from genome directly
                narrative = narrative_text
                
                # Store reference to full genome in the text (for future genome extractors)
                # This is a transitional approach
                
                if len(narrative_text) < 20:  # Too short
                    continue
                    
            else:
                # Extract text narrative (legacy behavior for text-only domains)
                narrative = None
                for field in self.narrative_field:
                    narrative = item.get(field, '')
                    if narrative and len(str(narrative)) >= self.min_narrative_length:
                        break
                
                # Skip if no valid text found
                if not narrative or len(str(narrative)) < self.min_narrative_length:
                    continue
            
            narratives.append(narrative)
            outcomes.append(float(outcome) if self.outcome_type == 'continuous' else int(outcome))
        
        return narratives, np.array(outcomes), len(data)
    
    def _has_rich_genome(self, item: Dict) -> bool:
        """
        Determine if item has rich structured genome or is mainly text.
        
        Rich genome = multiple structured fields beyond just text description.
        """
        if not isinstance(item, dict):
            return False
        
        # Count non-text, non-outcome fields
        rich_fields = []
        for key, value in item.items():
            # Skip narrative text fields and outcome
            if key in self.narrative_field or key == self.outcome_field:
                continue
            
            # Count structured data fields
            if isinstance(value, (int, float, bool, dict, list)):
                rich_fields.append(key)
            elif isinstance(value, str) and len(value) < 50:  # Short strings are categories, not narratives
                rich_fields.append(key)
        
        # If more than 5 rich structured fields, treat as genome
        return len(rich_fields) >= 5
    
    def _synthesize_text_from_genome(self, genome: Dict) -> str:
        """
        Synthesize text description from structured genome for embedding purposes.
        
        This allows embedders to work with genome dicts while preserving structure.
        """
        # Extract key structured values and format as text
        parts = []
        
        # Add domain-agnostic fields
        for key, value in genome.items():
            if key == self.outcome_field:
                continue
            
            # Format structured data as readable text
            if isinstance(value, (int, float)):
                parts.append(f"{key}:{value}")
            elif isinstance(value, str) and len(value) < 100:
                parts.append(f"{key}:{value}")
            elif isinstance(value, dict) and len(value) < 10:
                # Small dicts - include
                parts.append(f"{key}:{value}")
        
        # Join into synthetic narrative
        return " ".join(parts[:50])  # Limit to avoid huge strings
    
    def _load_raw_records(self) -> List[Dict]:
        """Load and flatten raw data records."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        with open(self.data_path) as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            flattened = []
            for value in data.values():
                if isinstance(value, list):
                    flattened.extend(value)
                elif value:
                    flattened.append(value)
            data = flattened
        
        if not isinstance(data, list):
            raise ValueError(f"Unexpected data format for {self.name}: {type(data)}")
        
        return data
    
    def get_raw_records(self) -> List[Dict]:
        """Public accessor for raw domain records."""
        return self._load_raw_records()
    
    def supports_checkpoints(self) -> bool:
        """Whether the domain exposes checkpoint snapshots."""
        return bool(self.checkpoint_builder and self.checkpoint_schema)
    
    def generate_checkpoint_snapshots(
        self,
        checkpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Generate structured checkpoint snapshots for the domain."""
        if not self.supports_checkpoints():
            raise ValueError(f"Domain '{self.name}' does not expose checkpoints.")
        
        records = self._load_raw_records()
        return self.checkpoint_builder(records, checkpoint, limit)


# ==============================================================================
# DOMAIN REGISTRY - ADD NEW DOMAINS HERE
# ==============================================================================

def _bool_phrase(value: bool, true_text: str, false_text: str) -> str:
    return true_text if value else false_text


def build_nhl_narrative(game: dict) -> str:
    """Synthesize predictive (pre-game) narrative text for NHL games from structured fields."""
    home = game.get('home_team', 'Home Team')
    away = game.get('away_team', 'Away Team')
    season = game.get('season', 'Unknown season')
    date = game.get('date', 'Unknown date')
    venue = game.get('venue', 'unknown venue')
    game_type = game.get('game_type', 'regular')
    playoff = _bool_phrase(game.get('is_playoff', False), 'playoff matchup', 'regular season meeting')
    rivalry = _bool_phrase(game.get('is_rivalry', False), 'rivalry game', 'non-rivalry game')
    goalies = f"Projected goalies: {home} leaning {game.get('home_goalie', 'unknown')} vs {away} leaning {game.get('away_goalie', 'unknown')}."

    context = game.get('temporal_context', {}) or {}
    rest_adv = context.get('rest_advantage')
    rest_phrase = f"Rest advantage: {rest_adv:+} days" if rest_adv is not None else "Rest advantage data unavailable"
    b2b_home = _bool_phrase(context.get('home_back_to_back', False), f"{home} on back-to-back", f"{home} rested")
    b2b_away = _bool_phrase(context.get('away_back_to_back', False), f"{away} on back-to-back", f"{away} rested")
    record_diff = context.get('record_differential')
    record_phrase = f"Record differential {record_diff:+.2f}" if record_diff is not None else "Record differential unknown"

    outcome_flag = game.get('home_won')
    if outcome_flag is None:
        outcome_phrase = "Outcome pending"
    else:
        outcome_phrase = f"Last result: {home} {'won' if outcome_flag else 'lost'}"

    odds = game.get('betting_odds', {}) or {}
    odds_phrase = f"Betting odds: {home} {odds.get('moneyline_home', odds.get('home_moneyline', 'n/a'))} vs {away} {odds.get('moneyline_away', odds.get('away_moneyline', 'n/a'))}."

    narrative = (
        f"{season} {playoff} ({game_type}) preview for {date} at {venue}: {home} host {away}. "
        f"{rivalry.capitalize()}. {goalies} "
        f"Prep signals: {rest_phrase}; {b2b_home}, {b2b_away}. {record_phrase}. {odds_phrase} {outcome_phrase}."
    )

    return shorten(narrative, width=600, placeholder='…')


def extract_nhl_data(data):
    """Custom extractor for NHL narrative processing."""
    if isinstance(data, dict):
        games = data.get('games', [])
    else:
        games = data

    narratives = []
    outcomes = []

    for game in games:
        try:
            narratives.append(build_nhl_narrative(game))
            outcomes.append(1 if game.get('home_won') else 0)
        except Exception:
            continue

    return narratives, np.array(outcomes), len(games)


def _resolve_mental_health_records(data):
    """Normalize mental health dataset structure to a list of disorder records."""
    if isinstance(data, dict):
        if 'disorders' in data and isinstance(data['disorders'], list):
            return data['disorders']
        if 'high_severity_disorders' in data and isinstance(data['high_severity_disorders'], list):
            return data['high_severity_disorders']
        # Fallback: flatten any list values
        records = []
        for value in data.values():
            if isinstance(value, list):
                records.extend(value)
        return records
    return data


def extract_mental_health_narratives(data):
    """Convert structured disorder metadata into textual narratives with stigma outcomes."""
    records = _resolve_mental_health_records(data) or []
    narratives = []
    outcomes = []
    
    for disorder in records:
        name = disorder.get('disorder_name')
        if not name:
            continue
        
        social = disorder.get('social_impact', {}) or {}
        stigma = (
            social.get('stigma_score')
            or disorder.get('stigma_score')
            or disorder.get('predicted_stigma')
        )
        try:
            stigma_value = float(stigma)
        except (TypeError, ValueError):
            continue
        
        phonetic = disorder.get('phonetic_analysis', {}) or {}
        clinical = disorder.get('clinical_outcomes', {}) or {}
        funding = disorder.get('research_funding', {}) or disorder.get('nih_funding', {}) or {}
        name_effect = disorder.get('name_impact_assessment', {}) or {}
        synchronicity = disorder.get('synchronicity_analysis', {}) or {}
        
        parts = [
            f"{name} ({disorder.get('dsm_code', 'unknown DSM')}/{disorder.get('icd10', 'unknown ICD')}) "
            f"carries a stigma score of {stigma_value:.2f}.",
        ]
        
        if phonetic:
            parts.append(
                f"The name sounds {phonetic.get('harshness_score', 'n/a')} harsh with "
                f"{phonetic.get('syllables', 'n/a')} syllables and origin "
                f"{phonetic.get('foreign_origin') or phonetic.get('name_origin') or 'unspecified'}."
            )
        
        if synchronicity:
            meaning = synchronicity.get('name_meaning')
            alignment = synchronicity.get('alignment')
            if meaning or alignment:
                parts.append(
                    f"Meaning/alignment: {meaning or 'unspecified'} ({alignment or 'alignment n/a'})."
                )
        
        if clinical:
            mortality = clinical.get('mortality_rate_per_100k')
            treatment = clinical.get('treatment_seeking_rate')
            delay = clinical.get('treatment_delay_months')
            clinical_bits = []
            if mortality is not None:
                clinical_bits.append(f"mortality {mortality} per 100k")
            if treatment is not None:
                clinical_bits.append(f"treatment seeking rate {treatment}")
            if delay is not None:
                clinical_bits.append(f"treatment delay {delay} months")
            if clinical_bits:
                parts.append("Clinical context: " + ", ".join(clinical_bits) + ".")
        
        if funding:
            nih = funding.get('nih_funding_millions_2023') or funding.get('total_funding_millions')
            articles = funding.get('pubmed_articles_total') or funding.get('article_count')
            funding_bits = []
            if nih is not None:
                funding_bits.append(f"NIH funding ${nih}M")
            if articles is not None:
                funding_bits.append(f"{articles} publications")
            if funding_bits:
                parts.append("Research scale: " + ", ".join(funding_bits) + ".")
        
        if name_effect:
            positives = name_effect.get('positive')
            negatives = name_effect.get('negative')
            net = name_effect.get('net_effect')
            summary = []
            if positives:
                summary.append(f"Positive: {positives}")
            if negatives:
                summary.append(f"Negative: {negatives}")
            if net:
                summary.append(f"Net impact: {net}")
            if summary:
                parts.append("Name impact assessment — " + " ".join(summary))
        
        narratives.append(" ".join(part.strip() for part in parts if part).strip())
        outcomes.append(stigma_value)
    
    return narratives, np.array(outcomes), len(records)


DOMAINS = {
    # ============================================================================
    # LEGAL DOMAINS - Testing narrative vs evidence in adversarial systems
    # ============================================================================
    
    'supreme_court': DomainConfig(
        name='supreme_court',
        data_path='data/domains/supreme_court_complete.json',
        narrative_field='majority_opinion',
        outcome_field='citation_count',
        estimated_pi=0.52,
        description='Supreme Court opinions - tests narrative vs evidence, π variance, adversarial dynamics',
        outcome_type='continuous',
        min_narrative_length=500,
        custom_extractor=extract_supreme_court_narratives,
        checkpoint_schema=["FILING", "ORAL", "CONFERENCE", "DECISION"],
        checkpoint_fields=[
            "win_probability_petitioner",
            "vote_margin",
            "citation_intensity",
            "complexity",
        ],
        checkpoint_description="Procedural checkpoints from cert grant to final decision",
        checkpoint_builder=build_supreme_court_checkpoint_snapshots,
        targets=[
            TargetConfig(
                name="petitioner_win",
                scope="case",
                outcome_type="binary",
                description="Whether the petitioner wins the case.",
                builder=build_supreme_court_outcome_row,
            ),
            TargetConfig(
                name="citation_intensity",
                scope="case",
                outcome_type="continuous",
                description="Citation count as continuous target.",
                builder=build_supreme_court_citation_row,
            ),
        ],
    ),

    'wikiplots': DomainConfig(
        name='wikiplots',
        data_path='data/literary_corpus/wikiplots_corpus.json',
        narrative_field='narrative',
        outcome_field='impact_score',
        estimated_pi=0.81,
        description='112k Wikipedia story plots',
        outcome_type='continuous',
        min_narrative_length=180,
        custom_extractor=extract_wikiplots,
        targets=[
            TargetConfig(
                name="impact_score",
                scope="story",
                outcome_type="continuous",
                description="WikiPlots narrative impact score.",
                builder=build_wikiplots_impact_row,
            )
        ],
    ),

    'stereotropes': DomainConfig(
        name='stereotropes',
        data_path='data/literary_corpus/stereotropes_corpus.json',
        narrative_field='narrative',
        outcome_field='impact_score',
        estimated_pi=0.78,
        description='Tropes + film/adjective narratives from Stereotropes',
        outcome_type='continuous',
        min_narrative_length=150,
        custom_extractor=extract_stereotropes,
        targets=[
            TargetConfig(
                name="impact_score",
                scope="story",
                outcome_type="continuous",
                description="Stereotropes impact score.",
                builder=build_stereotropes_impact_row,
            )
        ],
    ),

    'ml_research': DomainConfig(
        name='ml_research',
        data_path='data/literary_corpus/ml_research_corpus.json',
        narrative_field='narrative',
        outcome_field='impact_score',
        estimated_pi=0.74,
        description='Research/tutorial narratives from ML/DL/RL corpus',
        outcome_type='continuous',
        min_narrative_length=200,
        custom_extractor=extract_research_narratives,
        targets=[
            TargetConfig(
                name="impact_score",
                scope="paper",
                outcome_type="continuous",
                description="Research impact / readership score.",
                builder=build_ml_research_impact_row,
            )
        ],
    ),
    
    'cmu_movies': DomainConfig(
        name='cmu_movies',
        data_path='data/literary_corpus/cmu_movies_corpus.json',
        narrative_field='narrative',
        outcome_field='impact_score',
        estimated_pi=0.78,
        description='CMU Movie Summary Corpus (42k+ movie plots with metadata)',
        outcome_type='continuous',
        min_narrative_length=150,
        custom_extractor=extract_wikiplots,  # Same structure as wikiplots
        targets=[
            TargetConfig(
                name="box_office_signal",
                scope="story",
                outcome_type="continuous",
                description="Revenue/impact hybrid target for CMU movies.",
                builder=build_cmu_movies_revenue_row,
            )
        ],
    ),
    
    # ============================================================================
    # EXISTING DOMAINS
    # ============================================================================
    # ENTERTAINMENT DOMAINS
    
    'movies': DomainConfig(
        name='movies',
        data_path='data/domains/imdb_movies_complete.json',
        narrative_field='plot_summary',
        outcome_field='success_score',
        estimated_pi=0.65,
        description='Movie plot summaries with success scores',
        outcome_type='continuous',
        min_narrative_length=200
    ),
    
    'oscars': DomainConfig(
        name='oscars',
        data_path='data/domains/oscar_nominees_complete.json',
        narrative_field=['overview', 'plot_summary', 'tagline'],
        outcome_field='won_oscar',
        estimated_pi=0.88,
        description='Oscar Best Picture nominees',
        outcome_type='binary',
        min_narrative_length=60
    ),
    
    'music': DomainConfig(
        name='music',
        data_path='data/domains/spotify_songs.json',
        narrative_field='lyrics',
        outcome_field='popularity',
        estimated_pi=0.70,
        description='Song lyrics with popularity scores',
        outcome_type='continuous',
        min_narrative_length=100
    ),
    
    # SPORTS DOMAINS
    
    'nba': DomainConfig(
        name='nba',
        data_path='data/domains/nba_pregame_narratives.json',
        narrative_field=['pregame_narrative', 'narrative', 'description'],
        outcome_field='won',  # Fixed: data has 'won' not 'player1_won'
        estimated_pi=0.49,
        description='NBA game narratives',
        outcome_type='binary',
        targets=[
            TargetConfig(
                name="game_win",
                scope="game",
                outcome_type="binary",
                description="Whether the focus team wins the game.",
                builder=build_nba_game_win_row,
            ),
            TargetConfig(
                name="point_margin",
                scope="game",
                outcome_type="continuous",
                description="Final point differential as continuous target.",
                builder=build_nba_margin_row,
            ),
        ],
    ),
    
    'nfl': DomainConfig(
        name='nfl',
        data_path='data/domains/nfl_games_rich_narratives.json',
        narrative_field=['narrative', 'rich_narrative', 'description'],
        outcome_field='home_won',
        estimated_pi=0.57,
        description='NFL game narratives',
        outcome_type='binary',
        checkpoint_schema=['PREGAME', 'Q1', 'HALF', 'Q3', 'FINAL'],
        checkpoint_fields=[
            'home_score',
            'away_score',
            'score_differential',
            'win_probability_home',
            'spread',
        ],
        checkpoint_description='Quarterly synthetic snapshots derived from structured odds/context',
        checkpoint_builder=build_nfl_checkpoint_snapshots,
        targets=[
            TargetConfig(
                name="game_win",
                scope="game",
                outcome_type="binary",
                description="Whether the home team wins the game.",
                builder=build_nfl_game_win_row,
            ),
            TargetConfig(
                name="home_moneyline_roi",
                scope="bet",
                outcome_type="continuous",
                description="ROI of backing the home moneyline each game.",
                builder=build_nfl_roi_row,
                requires_odds=True,
            ),
        ],
    ),
    
    'mlb': DomainConfig(
        name='mlb',
        data_path='data/domains/mlb_complete_dataset.json',
        narrative_field='synthesized_narrative',
        outcome_field='won',
        estimated_pi=0.55,
        description='MLB game narratives with team records, venue, and context',
        outcome_type='binary',
        custom_extractor=extract_mlb_data,
        targets=[
            TargetConfig(
                name="game_win",
                scope="game",
                outcome_type="binary",
                description="Whether the home club wins the game.",
                builder=build_mlb_game_win_row,
            )
        ],
    ),
    
    'tennis': DomainConfig(
        name='tennis',
        data_path='data/domains/tennis_complete_dataset.json',
        narrative_field='synthesized_narrative',
        outcome_field='player1_won',
        estimated_pi=0.75,
        description='Tennis match narratives with surface, rankings, H2H, tournament context',
        outcome_type='binary',
        custom_extractor=extract_tennis_data
    ),
    
    'ufc': DomainConfig(
        name='ufc',
        data_path='data/domains/ufc_complete_dataset.json',
        narrative_field='synthesized_narrative',
        outcome_field='won',
        estimated_pi=0.68,
        description='UFC fight narratives with fighter stats and betting odds',
        outcome_type='binary',
        custom_extractor=extract_ufc_data
    ),
    
    'golf': DomainConfig(
        name='golf',
        data_path='data/domains/golf_with_narratives.json',
        narrative_field=['narrative', 'tournament_narrative', 'description'],
        outcome_field='won_tournament',
        estimated_pi=0.70,
        description='Golf tournament narratives',
        outcome_type='binary',
        targets=[
            TargetConfig(
                name="tournament_win",
                scope="event",
                outcome_type="binary",
                description="Golfer wins the event.",
                builder=build_golf_win_row,
            )
        ],
    ),

    'nhl': DomainConfig(
        name='nhl',
        data_path='data/domains/nhl_games_with_odds.json',
        narrative_field='synthetic_narrative',
        outcome_field='home_won',
        estimated_pi=0.52,
        description='NHL game narratives synthesized from structured context',
        outcome_type='binary',
        custom_extractor=lambda data: extract_nhl_data(data),
        checkpoint_schema=['P1', 'P2', 'FINAL'],
        checkpoint_fields=[
            'home_score',
            'away_score',
            'score_differential',
            'win_probability_home',
            'rest_advantage',
            'record_differential',
        ],
        checkpoint_description='Per-period synthetic summaries leveraging rest/context/betting signals',
        checkpoint_builder=build_nhl_checkpoint_snapshots,
        targets=[
            TargetConfig(
                name="game_win",
                scope="game",
                outcome_type="binary",
                description="Whether the home club wins the game.",
                builder=build_nhl_game_win_row,
            ),
            TargetConfig(
                name="home_moneyline_roi",
                scope="bet",
                outcome_type="continuous",
                description="ROI from betting home moneyline each game.",
                builder=build_nhl_roi_row,
                requires_odds=True,
            ),
        ],
    ),
    
    # BUSINESS DOMAINS
    
    'startups': DomainConfig(
        name='startups',
        data_path='data/domains/startups_real_data.json',
        narrative_field=['description_long', 'description_short', 'founding_team_narrative'],
        outcome_field='successful',
        estimated_pi=0.76,
        description='Startup pitches and descriptions',
        outcome_type='binary',
        min_narrative_length=60,
        targets=[
            TargetConfig(
                name="success",
                scope="company",
                outcome_type="binary",
                description="Startup achieves successful outcome (acquired/IPO).",
                builder=build_startup_success_row,
            ),
            TargetConfig(
                name="total_funding",
                scope="company",
                outcome_type="continuous",
                description="Total funding raised (USD).",
                builder=build_startup_funding_row,
            ),
        ],
    ),
    
    'crypto': DomainConfig(
        name='crypto',
        data_path='data/domains/crypto_complete.json',  # If exists
        narrative_field=['description', 'whitepaper_summary'],
        outcome_field='success_metric',
        estimated_pi=0.76,
        description='Cryptocurrency project descriptions',
        outcome_type='continuous'
    ),
    
    # MEDICAL DOMAINS
    
    'mental_health': DomainConfig(
        name='mental_health',
        data_path='narrative_optimization/domains/mental_health/data/integrated_disorders_complete.json',
        narrative_field=['disorder_name'],
        outcome_field='stigma_score',
        estimated_pi=0.55,
        description='Mental health disorder nomenclature and stigma outcomes',
        outcome_type='continuous',
        min_narrative_length=120,
        custom_extractor=extract_mental_health_narratives
    ),
    
    # NATURAL/SCIENCE DOMAINS
    
    'hurricanes': DomainConfig(
        name='hurricanes',
        data_path='data/domains/hurricanes/hurricane_dataset_with_name_analysis.json',
        narrative_field=['synthetic_narrative'],
        outcome_field='fatalities',
        estimated_pi=0.30,  # Physical event
        description='Hurricane event narratives',
        outcome_type='continuous',
        custom_extractor=extract_hurricane_narratives,
    ),
    
    'dinosaurs': DomainConfig(
        name='dinosaurs',
        data_path='data/domains/dinosaurs/dinosaur_complete_dataset.json',
        narrative_field=['description', 'narrative', 'wikipedia_text'],
        outcome_field='cultural_prominence',
        estimated_pi=0.75,
        description='Dinosaur species narratives',
        outcome_type='continuous'
    ),
    
    # ABSTRACT/CULTURAL DOMAINS
    
    'mythology': DomainConfig(
        name='mythology',
        data_path='data/domains/mythology/mythology_complete_dataset.json',
        narrative_field=['full_text', 'summary', 'narrative'],
        outcome_field='cultural_persistence',
        estimated_pi=0.85,
        description='Mythological narratives',
        outcome_type='continuous',
        min_narrative_length=200
    ),
    
    'poker': DomainConfig(
        name='poker',
        data_path='data/domains/poker/poker_tournament_dataset_with_narratives.json',
        narrative_field=['narrative', 'description', 'tournament_story'],
        outcome_field='won',
        estimated_pi=0.83,
        description='Poker tournament narratives',
        outcome_type='binary'
    ),
    
    'boxing': DomainConfig(
        name='boxing',
        data_path='data/domains/boxing/boxing_fights_complete.json',
        narrative_field=['narrative', 'fight_description', 'description'],
        outcome_field='won',
        estimated_pi=0.74,
        description='Boxing match narratives',
        outcome_type='binary'
    ),
    
    # NEW DOMAINS - Nov 2025
    
    'wwe': DomainConfig(
        name='wwe',
        data_path='narrative_optimization/domains/wwe/data/wwe_dummy.json',
        narrative_field='synthesized_narrative',
        outcome_field='engagement',
        estimated_pi=0.85,
        description='WWE storylines with character arcs, promos, and booking quality',
        outcome_type='continuous',
        min_narrative_length=100,
        custom_extractor=extract_wwe_data
    ),
    
    'novels': DomainConfig(
        name='novels',
        data_path='narrative_optimization/domains/novels/data/novels_dataset_expanded.json',
        narrative_field=['full_narrative', 'plot_summary'],
        outcome_field='critical_acclaim_score',
        estimated_pi=0.82,
        description='Literary fiction novels with plot summaries and critical acclaim',
        outcome_type='continuous',
        min_narrative_length=100,
        custom_extractor=extract_novels_data
    ),
    
    'nonfiction': DomainConfig(
        name='nonfiction',
        data_path='narrative_optimization/domains/nonfiction/data/nonfiction_dataset_expanded.json',
        narrative_field=['full_narrative', 'description'],
        outcome_field='critical_acclaim_score',
        estimated_pi=0.75,
        description='Nonfiction books with descriptions and critical acclaim',
        outcome_type='continuous',
        min_narrative_length=100,
        custom_extractor=extract_nonfiction_data
    ),
}


# ==============================================================================
# EASY DOMAIN ADDITION FUNCTIONS
# ==============================================================================

def add_new_domain(
    name: str,
    data_path: str,
    narrative_field: str,
    outcome_field: str,
    estimated_pi: float,
    **kwargs
):
    """
    Add new domain to registry.
    
    Example:
    --------
    add_new_domain(
        name='chess',
        data_path='data/domains/chess_games.json',
        narrative_field='game_narrative',
        outcome_field='won',
        estimated_pi=0.40,
        description='Chess game narratives',
        outcome_type='binary'
    )
    """
    config = DomainConfig(
        name=name,
        data_path=data_path,
        narrative_field=narrative_field,
        outcome_field=outcome_field,
        estimated_pi=estimated_pi,
        **kwargs
    )
    
    DOMAINS[name] = config
    print(f"✓ Added domain: {name} (π={estimated_pi:.2f})")
    
    return config


def list_available_domains():
    """List all registered domains and their status."""
    print(f"\n{'='*80}")
    print("REGISTERED DOMAINS")
    print(f"{'='*80}\n")
    
    available = []
    unavailable = []
    
    for name, config in sorted(DOMAINS.items(), key=lambda x: x[1].estimated_pi, reverse=True):
        exists = config.data_path.exists()
        status = '✓' if exists else '✗'
        
        line = f"{status} {name:<15s} (π={config.estimated_pi:.2f}): {config.description}"
        
        if exists:
            available.append((name, config))
            print(line)
        else:
            unavailable.append((name, config))
    
    if unavailable:
        print(f"\nNot yet available:")
        for name, config in unavailable:
            print(f"✗ {name:<15s} (π={config.estimated_pi:.2f}): {config.description}")
    
    print(f"\n{'='*80}")
    print(f"Available: {len(available)}/{len(DOMAINS)} domains")
    print(f"Total narratives: ~197K across available domains")
    print(f"{'='*80}\n")
    
    return available


def get_domain(name: str) -> Optional[DomainConfig]:
    """Get domain configuration by name."""
    return DOMAINS.get(name)


def load_domain_safe(name: str) -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[DomainConfig]]:
    """
    Safely load domain with error handling.
    
    Returns
    -------
    narratives : list of str or None
    outcomes : ndarray or None
    config : DomainConfig or None
    """
    config = get_domain(name)
    
    if config is None:
        print(f"✗ Domain '{name}' not registered")
        print(f"  Available: {list(DOMAINS.keys())}")
        return None, None, None
    
    try:
        narratives, outcomes, total = config.load_and_extract()
        
        print(f"✓ {name}: Loaded {len(narratives):,} narratives (from {total:,} total)")
        print(f"  π={config.estimated_pi:.2f}, outcome_type={config.outcome_type}")
        
        return narratives, outcomes, config
    
    except FileNotFoundError as e:
        print(f"✗ {name}: Data file not found")
        print(f"  Expected: {config.data_path}")
        return None, None, config
    
    except Exception as e:
        print(f"✗ {name}: Error loading - {str(e)[:100]}")
        return None, None, config


# ==============================================================================
# CHECKPOINT DATASETS
# ==============================================================================

def load_domain_checkpoints(
    name: str,
    checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict], DomainConfig]:
    """
    Generate checkpoint snapshots for a domain.
    
    Returns
    -------
    snapshots : list of dict
        Structured checkpoint payloads
    config : DomainConfig
        Domain configuration reference
    """
    config = get_domain(name)
    if config is None:
        raise ValueError(f"Domain '{name}' not registered")
    
    snapshots = config.generate_checkpoint_snapshots(checkpoint=checkpoint, limit=limit)
    return snapshots, config


# ==============================================================================
# BATCH OPERATIONS
# ==============================================================================

def get_domains_by_pi_range(min_pi: float = 0.0, max_pi: float = 1.0) -> List[DomainConfig]:
    """Get domains in π range."""
    return [config for config in DOMAINS.values() 
            if min_pi <= config.estimated_pi <= max_pi 
            and config.data_path.exists()]


def get_similar_domains(reference_domain: str, pi_tolerance: float = 0.10) -> List[DomainConfig]:
    """Get domains with similar π to reference."""
    ref_config = get_domain(reference_domain)
    if ref_config is None:
        return []
    
    ref_pi = ref_config.estimated_pi
    
    similar = []
    for name, config in DOMAINS.items():
        if name == reference_domain:
            continue
        
        if abs(config.estimated_pi - ref_pi) <= pi_tolerance and config.data_path.exists():
            similar.append(config)
    
    return sorted(similar, key=lambda c: abs(c.estimated_pi - ref_pi))


if __name__ == '__main__':
    # Show registered domains
    available = list_available_domains()
    
    # Example: Find domains similar to NBA
    print("\nDomains similar to NBA (π=0.49 ± 0.10):")
    similar = get_similar_domains('nba', pi_tolerance=0.10)
    for config in similar:
        print(f"  • {config.name} (π={config.estimated_pi:.2f})")
    
    # Example: Preview NHL checkpoint snapshots
    try:
        snapshots, _ = load_domain_checkpoints('nhl', limit=2)
        print(f"\nNHL checkpoint preview ({len(snapshots)} snapshots):")
        for snap in snapshots:
            print(f"  [{snap['checkpoint_id']}] {snap['narrative']}")
    except Exception as exc:
        print(f"\n⚠️  Unable to preview NHL checkpoints: {exc}")

