"""
UFC Fighter Narrative Generation

Generates NOMINATIVE-RICH narratives (200-300 words) for each UFC fight.
Maximum emphasis on fighter names, nicknames, personas, and stylistic clash.

Following INSTRUCTIONS lines 139-200:
- Fighter names repeated 15-20 times
- Nicknames prominently featured
- Persona and style clash emphasized
- Context and stakes highlighted
"""

import pandas as pd
import json
import random
from pathlib import Path
from typing import Dict, Any, List

class UFCNarrativeGenerator:
    """
    Generates comprehensive, nominative-rich narratives for UFC fights.
    
    Each narrative emphasizes:
    - Fighter full names and nicknames
    - Persona clash (trash talker vs silent, etc.)
    - Fighting styles and matchup dynamics
    - Records and recent form
    - Stakes (title, ranking, redemption)
    - Individual characteristics that create narrative
    """
    
    def __init__(self):
        """Initialize narrative templates."""
        self.fighting_styles = {
            'Striker': ['striking prowess', 'devastating power', 'precise technique', 'knockout ability'],
            'Grappler': ['ground dominance', 'submission expertise', 'grappling mastery', 'control'],
            'Wrestler': ['takedown threat', 'wrestling pedigree', 'control', 'ground-and-pound'],
            'Mixed': ['well-rounded skills', 'versatility', 'complete game', 'adaptability']
        }
        
        self.persona_descriptors = [
            'calculated', 'explosive', 'patient', 'aggressive', 'technical', 'wild',
            'cerebral', 'instinctive', 'methodical', 'unpredictable'
        ]
    
    def generate_narratives(self, csv_path: str, output_path: str) -> List[Dict[str, Any]]:
        """
        Generate narratives for all fights in dataset.
        
        Parameters
        ----------
        csv_path : str
            Path to UFC dataset CSV
        output_path : str
            Path to save narratives JSON
            
        Returns
        -------
        fights_with_narratives : list of dict
            Complete fight data with narratives
        """
        print("="*80)
        print("GENERATING NOMINATIVE-RICH UFC NARRATIVES")
        print("="*80)
        
        # Load dataset
        print(f"\nLoading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} fights")
        
        # Generate narratives
        print(f"\nGenerating narratives...")
        fights_with_narratives = []
        
        for idx, row in df.iterrows():
            narrative = self._generate_fight_narrative(row)
            
            # Create complete fight record
            fight = {
                'fight_id': row['fight_id'],
                'event_name': row['event'],
                'date': row['date'],
                'location': row['location'],
                'weight_class': row['weight_class'],
                'title_fight': bool(row['title_bout']),
                
                'fighter_a': {
                    'name': row['R_fighter'],
                    'nickname': row.get('R_nickname', ''),
                    'record': f"{row['R_wins']}-{row['R_losses']}-{row['R_draw']}",
                    'age': int(row['R_age']),
                    'height': float(row['R_Height_cms']),
                    'reach': float(row['R_Reach_cms']),
                    'stance': row['R_Stance'],
                    'win_streak': int(row.get('R_current_win_streak', 0)),
                    'sig_str_pct': float(row.get('R_avg_SIG_STR_pct', 50)),
                    'sub_att': float(row.get('R_avg_SUB_ATT', 1)),
                    'td_pct': float(row.get('R_avg_TD_pct', 40))
                },
                
                'fighter_b': {
                    'name': row['B_fighter'],
                    'nickname': row.get('B_nickname', ''),
                    'record': f"{row['B_wins']}-{row['B_losses']}-{row['B_draw']}",
                    'age': int(row['B_age']),
                    'height': float(row['B_Height_cms']),
                    'reach': float(row['B_Reach_cms']),
                    'stance': row['B_Stance'],
                    'win_streak': int(row.get('B_current_win_streak', 0)),
                    'sig_str_pct': float(row.get('B_avg_SIG_STR_pct', 50)),
                    'sub_att': float(row.get('B_avg_SUB_ATT', 1)),
                    'td_pct': float(row.get('B_avg_TD_pct', 40))
                },
                
                'result': {
                    'winner': 'fighter_a' if row['Winner'] == row['R_fighter'] else 'fighter_b',
                    'method': row['win_by'],
                    'round': int(row['last_round']),
                    'time': row['last_round_time'],
                    'finish': 'Decision' not in row['win_by']
                },
                
                'betting_odds': {
                    'moneyline_a': float(row['R_odds']),
                    'moneyline_b': float(row['B_odds']),
                    'favorite': 'fighter_a' if row['R_odds'] < row['B_odds'] else 'fighter_b'
                },
                
                'narrative': narrative
            }
            
            fights_with_narratives.append(fight)
            
            if (idx + 1) % 500 == 0:
                print(f"  Generated {idx + 1}/{len(df)} narratives...")
        
        # Save
        print(f"\nSaving narratives...")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(fights_with_narratives, f, indent=2)
        
        print(f"✓ Saved {len(fights_with_narratives)} narratives to: {output_file}")
        
        # Statistics
        total_words = sum(len(f['narrative'].split()) for f in fights_with_narratives)
        avg_words = total_words / len(fights_with_narratives)
        print(f"\nNarrative Statistics:")
        print(f"  - Average length: {avg_words:.0f} words")
        print(f"  - Total words: {total_words:,}")
        
        return fights_with_narratives
    
    def _generate_fight_narrative(self, row: pd.Series) -> str:
        """
        Generate single nominative-rich fight narrative.
        
        Target: 200-300 words with 15-20 name mentions.
        """
        # Extract fighter data
        name_a = row['R_fighter']
        nickname_a = row.get('R_nickname', '')
        record_a = f"{row['R_wins']}-{row['R_losses']}"
        
        name_b = row['B_fighter']
        nickname_b = row.get('B_nickname', '')
        record_b = f"{row['B_wins']}-{row['B_losses']}"
        
        weight_class = row['weight_class']
        is_title = row['title_bout']
        event = row['event']
        location = row.get('location', 'Las Vegas')
        
        # Infer styles
        style_a = self._infer_style_from_stats(
            row['R_avg_SIG_STR_pct'],
            row['R_avg_SUB_ATT'],
            row['R_avg_TD_pct']
        )
        style_b = self._infer_style_from_stats(
            row['B_avg_SIG_STR_pct'],
            row['B_avg_SUB_ATT'],
            row['B_avg_TD_pct']
        )
        
        # Build narrative with heavy name usage
        paragraphs = []
        
        # Paragraph 1: Introduction with names and records
        title_text = "title bout" if is_title else f"{weight_class} matchup"
        nick_a_text = f" '{nickname_a}'" if nickname_a else ""
        nick_b_text = f" '{nickname_b}'" if nickname_b else ""
        
        p1 = f"{name_a}{nick_a_text} ({record_a}) faces {name_b}{nick_b_text} ({record_b}) in a {title_text} at {event} in {location}. "
        
        # Add style context
        if nickname_a:
            p1 += f"{nickname_a}'s {random.choice(self.fighting_styles[style_a])} "
        else:
            p1 += f"{name_a.split()[-1]}'s {random.choice(self.fighting_styles[style_a])} "
        
        p1 += f"meets "
        
        if nickname_b:
            p1 += f"{nickname_b}'s {random.choice(self.fighting_styles[style_b])} "
        else:
            p1 += f"{name_b.split()[-1]}'s {random.choice(self.fighting_styles[style_b])} "
        
        p1 += f"in this highly anticipated clash."
        
        paragraphs.append(p1)
        
        # Paragraph 2: Individual fighter analysis with names
        win_streak_a = int(row.get('R_current_win_streak', 0))
        win_streak_b = int(row.get('B_current_win_streak', 0))
        
        p2 = f"{name_a} "
        if win_streak_a > 2:
            p2 += f"enters on a {win_streak_a}-fight winning streak, demonstrating {random.choice(self.persona_descriptors)} dominance. "
        else:
            p2 += f"brings {random.choice(self.persona_descriptors)} approach to the octagon. "
        
        # Add specific attribute
        if row['R_avg_SIG_STR_pct'] > 55:
            p2 += f"The {name_a.split()[-1]} striking accuracy of {row['R_avg_SIG_STR_pct']:.0f}% makes {name_a.split()[0]} exceptionally dangerous on the feet. "
        elif row['R_avg_SUB_ATT'] > 2:
            p2 += f"{name_a}'s submission threat ({row['R_avg_SUB_ATT']:.1f} attempts per fight) keeps opponents wary on the ground. "
        else:
            p2 += f"{name_a.split()[-1]}'s well-rounded skillset creates problems in all areas. "
        
        paragraphs.append(p2)
        
        # Paragraph 3: Opponent analysis with names
        p3 = f"Across the cage, {name_b} "
        if win_streak_b > 2:
            p3 += f"rides momentum with {win_streak_b} consecutive victories, showing {random.choice(self.persona_descriptors)} execution. "
        else:
            p3 += f"presents a {random.choice(self.persona_descriptors)} challenge. "
        
        # Add specific attribute
        if row['B_avg_SIG_STR_pct'] > 55:
            p3 += f"{nickname_b if nickname_b else name_b.split()[-1]}'s precision striking gives {name_b} a significant advantage in standup exchanges. "
        elif row['B_avg_SUB_ATT'] > 2:
            p3 += f"The submission game of {name_b} ({row['B_avg_SUB_ATT']:.1f} attempts per fight) threatens to end fights at any moment. "
        else:
            p3 += f"{name_b}'s versatility allows adaptation throughout the fight. "
        
        paragraphs.append(p3)
        
        # Paragraph 4: Matchup analysis and stakes with names
        p4 = "The stylistic matchup between "
        p4 += f"{name_a.split()[-1]} and {name_b.split()[-1]} "
        p4 += f"creates compelling drama. "
        
        # Style clash
        if style_a != style_b:
            p4 += f"{name_a}'s {style_a.lower()} approach contrasts sharply with {name_b}'s {style_b.lower()} style. "
        else:
            p4 += f"Both fighters favor {style_a.lower()} strategies, setting up a chess match. "
        
        # Stakes
        if is_title:
            p4 += f"Championship gold awaits the victor in this {weight_class} title bout. "
        else:
            p4 += f"A victory propels the winner toward title contention in the {weight_class} division. "
        
        # Final prediction setup
        reach_diff = abs(row['R_Reach_cms'] - row['B_Reach_cms'])
        if reach_diff > 5:
            longer_reach = name_a if row['R_Reach_cms'] > row['B_Reach_cms'] else name_b
            p4 += f"{longer_reach}'s reach advantage could prove decisive. "
        
        # Closing
        ref_a = nickname_a if nickname_a else name_a.split()[0]
        ref_b = nickname_b if nickname_b else name_b.split()[0]
        p4 += f"Can {ref_a} impose their will, or will {ref_b} shock the world?"
        
        paragraphs.append(p4)
        
        # Combine into narrative
        narrative = " ".join(paragraphs)
        
        return narrative
    
    def _infer_style_from_stats(self, sig_str_pct: float, sub_att: float, td_pct: float) -> str:
        """Infer fighting style from statistics."""
        if sig_str_pct > 52:
            return "Striker"
        elif sub_att > 2:
            return "Grappler"
        elif td_pct > 50:
            return "Wrestler"
        else:
            return "Mixed"

def main():
    """Generate narratives for massive UFC dataset."""
    generator = UFCNarrativeGenerator()
    
    # Generate from massive dataset
    csv_path = "data/domains/ufc_massive_dataset.csv"
    output_path = "data/domains/ufc_with_narratives.json"
    
    fights = generator.generate_narratives(csv_path, output_path)
    
    print("\n" + "="*80)
    print("NARRATIVE GENERATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated {len(fights)} nominative-rich narratives")
    print(f"\nNext step: Run analyze_ufc_complete.py to apply ALL transformers")

if __name__ == "__main__":
    main()

