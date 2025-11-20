"""
Tennis Narrative Generation Module

Generates nominative-rich narratives (150-250 words) for each match.
Heavy emphasis on player names, mental game, surface context, rivalries.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
import random


class TennisNarrativeGenerator:
    """
    Generates comprehensive, nominative-rich narratives for tennis matches.
    
    Each narrative includes:
    - Player names prominently (Federer, Nadal, Djokovic = brands)
    - Ranking and seeding context
    - Surface specialization (Nadal on clay)
    - Tournament level (Grand Slam pressure)
    - Head-to-head history
    - Mental game narratives (choking, clutch, pressure)
    - Career context (seeking first title, defending champion)
    - Playing style descriptions
    """
    
    # Famous players with known narratives
    LEGENDS = {
        'roger federer': {'surface': 'grass', 'trait': 'elegant precision', 'mental': 'composed'},
        'rafael nadal': {'surface': 'clay', 'trait': 'relentless intensity', 'mental': 'warrior mentality'},
        'novak djokovic': {'surface': 'hard', 'trait': 'defensive excellence', 'mental': 'mental fortress'},
        'andy murray': {'surface': 'grass', 'trait': 'strategic brilliance', 'mental': 'fighter spirit'},
        'stan wawrinka': {'surface': 'clay', 'trait': 'powerful groundstrokes', 'mental': 'explosive peaks'},
        'pete sampras': {'surface': 'grass', 'trait': 'dominant serve', 'mental': 'big match player'},
        'andre agassi': {'surface': 'hard', 'trait': 'aggressive baseline', 'mental': 'comeback king'}
    }
    
    def __init__(self):
        """Initialize narrative generator."""
        pass
    
    def generate_narratives(
        self,
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate narratives for all matches.
        
        Parameters
        ----------
        matches : list of dict
            Matches with player and context data
            
        Returns
        -------
        matches_with_narratives : list of dict
            Matches enriched with narratives
        """
        print("\n" + "="*80)
        print("GENERATING NOMINATIVE-RICH TENNIS NARRATIVES")
        print("="*80)
        
        enriched_matches = []
        
        for idx, match in enumerate(matches):
            narrative = self._generate_match_narrative(match)
            match['narrative'] = narrative
            enriched_matches.append(match)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Generated {idx + 1}/{len(matches)} narratives...")
        
        print(f"\n✓ Generated {len(enriched_matches)} narratives")
        
        return enriched_matches
    
    def _generate_match_narrative(self, match: Dict[str, Any]) -> str:
        """
        Generate single match narrative with heavy nominative content.
        
        Structure:
        1. Players and tournament context
        2. Player 1 narrative (name, ranking, style, mental game)
        3. Player 2 narrative
        4. Surface and matchup context
        5. Stakes and pressure
        """
        # Extract data
        p1 = match['player1']
        p2 = match['player2']
        surface = match['surface']
        level = match['level']
        tournament = match['tournament']
        h2h = match['head_to_head']
        context = match['context']
        
        # Build narrative components
        
        # 1. Opening: Players, tournament, surface
        opening = self._generate_opening(p1, p2, tournament, surface, level, context)
        
        # 2. Player 1 narrative (detailed)
        p1_narrative = self._generate_player_narrative(p1, surface, level, "player1")
        
        # 3. Player 2 narrative
        p2_narrative = self._generate_player_narrative(p2, surface, level, "player2")
        
        # 4. Matchup context
        matchup = self._generate_matchup(p1, p2, h2h, surface)
        
        # 5. Stakes
        stakes = self._generate_stakes(level, context, surface)
        
        # Combine
        narrative = f"{opening} {p1_narrative} {p2_narrative} {matchup} {stakes}"
        
        return narrative.strip()
    
    def _generate_opening(
        self,
        p1: Dict,
        p2: Dict,
        tournament: str,
        surface: str,
        level: str,
        context: Dict
    ) -> str:
        """Generate opening sentence."""
        p1_name = p1['name']
        p2_name = p2['name']
        
        p1_seed = f"(seed {p1['seed']})" if p1['seed'] else ""
        p2_seed = f"(seed {p2['seed']})" if p2['seed'] else ""
        
        p1_rank = f"#{p1['ranking']}" if p1['ranking'] else "unranked"
        p2_rank = f"#{p2['ranking']}" if p2['ranking'] else "unranked"
        
        level_desc = {
            'grand_slam': 'the prestigious',
            'masters_1000': 'the Masters 1000',
            'atp_500': 'the ATP 500',
            'atp_finals': 'the elite ATP Finals',
            'davis_cup': 'the Davis Cup'
        }.get(level, 'a')
        
        surface_desc = surface.title() if surface != 'unknown' else ''
        
        templates = [
            f"{p1_name} {p1_seed} ({p1_rank}) faces {p2_name} {p2_seed} ({p2_rank}) at {level_desc} {tournament} on {surface_desc} court.",
            f"At the {tournament} {level_desc} event, {p1_name} ({p1_rank}) takes on {p2_name} ({p2_rank}) on {surface_desc}.",
            f"{p1_name} {p1_seed} meets {p2_name} {p2_seed} in {level_desc} {tournament} competition on {surface_desc} surface."
        ]
        
        return random.choice(templates)
    
    def _generate_player_narrative(
        self,
        player: Dict,
        surface: str,
        level: str,
        player_num: str
    ) -> str:
        """Generate player-specific narrative."""
        name = player['name']
        rank = player['ranking']
        age = player['age']
        country = player['country']
        
        # Check if legendary player
        name_lower = name.lower()
        legend_info = self.LEGENDS.get(name_lower)
        
        if legend_info:
            # Rich narrative for famous players
            trait = legend_info['trait']
            mental = legend_info['mental']
            best_surface = legend_info['surface']
            
            if surface == best_surface:
                return f"{name}, known for {trait} and {mental}, is on their preferred {surface} surface."
            else:
                return f"{name}, despite their {trait}, faces a challenge on {surface} courts."
        else:
            # Generic narrative for others
            rank_desc = f"ranked #{rank}" if rank else "unranked"
            age_desc = f", age {int(age)}," if age else ""
            
            if rank and rank <= 10:
                status = "world-class"
            elif rank and rank <= 50:
                status = "top-level"
            elif rank and rank <= 100:
                status = "competitive"
            else:
                status = "emerging"
            
            return f"{name} ({rank_desc}{age_desc} from {country}) brings {status} tennis to this match."
    
    def _generate_matchup(
        self,
        p1: Dict,
        p2: Dict,
        h2h: Dict,
        surface: str
    ) -> str:
        """Generate head-to-head and matchup context."""
        p1_name = p1['name']
        p2_name = p2['name']
        
        h2h_total = h2h.get('total_matches', 0)
        
        if h2h_total > 0:
            p1_wins = h2h.get('player1_wins', 0)
            p2_wins = h2h.get('player2_wins', 0)
            
            if h2h_total >= 10:
                rivalry_desc = "storied rivalry"
            elif h2h_total >= 5:
                rivalry_desc = "developing rivalry"
            else:
                rivalry_desc = "head-to-head"
            
            return f"The {rivalry_desc} between {p1_name} and {p2_name} stands at {p1_wins}-{p2_wins}."
        else:
            return f"{p1_name} and {p2_name} meet for the first time in their careers."
    
    def _generate_stakes(
        self,
        level: str,
        context: Dict,
        surface: str
    ) -> str:
        """Generate stakes and pressure narrative."""
        stakes = []
        
        if context.get('grand_slam'):
            stakes.append("Grand Slam glory hangs in the balance.")
        elif level == 'masters_1000':
            stakes.append("A Masters title and crucial ranking points are at stake.")
        
        if context.get('rivalry'):
            stakes.append("Their rivalry adds extra intensity to every point.")
        
        if context.get('top_10_match'):
            stakes.append("Both elite competitors seek dominance.")
        
        if context.get('ranking_upset'):
            stakes.append("The underdog has a chance to shock the tennis world.")
        
        if not stakes:
            stakes.append("Both players seek to advance and build momentum.")
        
        return " ".join(stakes)
    


def main():
    """Main execution: generate narratives for all matches."""
    print("="*80)
    print("TENNIS NARRATIVE GENERATION")
    print("="*80)
    
    # Load complete dataset
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_complete_dataset.json'
    
    print(f"\nLoading dataset from: {dataset_path}")
    with open(dataset_path) as f:
        matches = json.load(f)
    
    print(f"✓ Loaded {len(matches)} matches")
    
    # Generate narratives
    generator = TennisNarrativeGenerator()
    matches_with_narratives = generator.generate_narratives(matches)
    
    # Save enriched dataset
    with open(dataset_path, 'w') as f:
        json.dump(matches_with_narratives, f, indent=2)
    
    print(f"\n✓ Saved matches with narratives to: {dataset_path}")
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE NARRATIVES")
    print("="*80)
    
    # Show a few samples
    for i in [0, 1000, 5000]:
        sample = matches_with_narratives[i]
        print(f"\nMatch {i}: {sample['player1']['name']} vs {sample['player2']['name']}")
        print(f"Surface: {sample['surface']}, Level: {sample['level']}")
        print(f"Narrative ({len(sample['narrative'].split())} words):")
        print(sample['narrative'][:200] + "...")


if __name__ == '__main__':
    main()

