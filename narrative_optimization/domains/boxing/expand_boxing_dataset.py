"""
Expand Boxing Dataset - Generate 5,000+ Realistic Fights

Expands initial 8 high-profile fights into comprehensive dataset:
- Multiple weight classes
- Various promotions
- Title fights and regular bouts
- Different fighter tiers
- Temporal coverage 2020-2024

Author: Narrative Integration System
Date: November 2025
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
data_dir = project_root / 'data' / 'domains' / 'boxing'


# Expanded fighter database
FIGHTER_DATABASE = {
    # Heavyweight
    'Tyson Fury': {'nationality': 'British', 'weight_class': 'Heavyweight', 'record': '34-0-1', 
                   'style': 'Boxer-puncher, unorthodox movement', 'narrative': 'The Gypsy King, comeback story, mental health advocate',
                   'achievements': ['WBC Heavyweight Champion', 'Lineal Champion'], 'reputation': 0.95, 'tier': 'elite'},
    'Oleksandr Usyk': {'nationality': 'Ukrainian', 'weight_class': 'Heavyweight', 'record': '21-0-0',
                       'style': 'Technical boxer, exceptional footwork', 'narrative': 'Undisputed cruiserweight, Olympic gold medalist',
                       'achievements': ['Undisputed Heavyweight Champion', 'Olympic Gold'], 'reputation': 0.93, 'tier': 'elite'},
    'Deontay Wilder': {'nationality': 'American', 'weight_class': 'Heavyweight', 'record': '43-3-1',
                       'style': 'Power puncher, devastating right hand', 'narrative': 'Bronze Bomber, knockout artist',
                       'achievements': ['Former WBC Champion', '42 KOs'], 'reputation': 0.88, 'tier': 'elite'},
    'Anthony Joshua': {'nationality': 'British', 'weight_class': 'Heavyweight', 'record': '28-3-0',
                       'style': 'Athletic boxer, combination puncher', 'narrative': 'AJ, Olympic gold medalist, two-time champion',
                       'achievements': ['Former Unified Champion', 'Olympic Gold'], 'reputation': 0.90, 'tier': 'elite'},
    'Andy Ruiz Jr': {'nationality': 'Mexican-American', 'weight_class': 'Heavyweight', 'record': '35-2-0',
                     'style': 'Power puncher, fast hands', 'narrative': 'Destroyer, upset specialist',
                     'achievements': ['Former Unified Champion'], 'reputation': 0.82, 'tier': 'contender'},
    'Joseph Parker': {'nationality': 'New Zealander', 'weight_class': 'Heavyweight', 'record': '33-3-0',
                      'style': 'Technical boxer, good movement', 'narrative': 'Former WBO champion',
                      'achievements': ['Former WBO Champion'], 'reputation': 0.80, 'tier': 'contender'},
    'Dillian Whyte': {'nationality': 'British', 'weight_class': 'Heavyweight', 'record': '29-3-0',
                      'style': 'Aggressive puncher, body attack', 'narrative': 'The Body Snatcher, tough contender',
                      'achievements': ['Former WBC Interim Champion'], 'reputation': 0.78, 'tier': 'contender'},
    'Luis Ortiz': {'nationality': 'Cuban', 'weight_class': 'Heavyweight', 'record': '33-3-0',
                   'style': 'Southpaw, technical boxer', 'narrative': 'King Kong, crafty veteran',
                   'achievements': ['Top Contender'], 'reputation': 0.75, 'tier': 'contender'},
    
    # Welterweight
    'Terence Crawford': {'nationality': 'American', 'weight_class': 'Welterweight', 'record': '40-0-0',
                         'style': 'Switch-hitter, technical master', 'narrative': 'Bud, undisputed welterweight, pound-for-pound #1',
                         'achievements': ['Undisputed Welterweight', 'P4P #1'], 'reputation': 0.92, 'tier': 'elite'},
    'Errol Spence Jr': {'nationality': 'American', 'weight_class': 'Welterweight', 'record': '28-1-0',
                        'style': 'Pressure fighter, body puncher', 'narrative': 'The Truth, unified champion',
                        'achievements': ['Former Unified Welterweight', '22 KOs'], 'reputation': 0.89, 'tier': 'elite'},
    'Manny Pacquiao': {'nationality': 'Filipino', 'weight_class': 'Welterweight', 'record': '62-8-2',
                       'style': 'Volume puncher, speed, angles', 'narrative': 'Pac-Man, eight-division champion, legend',
                       'achievements': ['8-Division Champion', 'Legend'], 'reputation': 0.91, 'tier': 'elite'},
    'Keith Thurman': {'nationality': 'American', 'weight_class': 'Welterweight', 'record': '30-1-0',
                      'style': 'Power boxer, good timing', 'narrative': 'One Time, former unified champion',
                      'achievements': ['Former Unified Welterweight'], 'reputation': 0.85, 'tier': 'contender'},
    'Danny Garcia': {'nationality': 'American', 'weight_class': 'Welterweight', 'record': '37-3-0',
                     'style': 'Counter-puncher, left hook', 'narrative': 'Swift, two-division champion',
                     'achievements': ['Two-Division Champion'], 'reputation': 0.84, 'tier': 'contender'},
    'Yordenis Ugas': {'nationality': 'Cuban', 'weight_class': 'Welterweight', 'record': '27-5-0',
                      'style': 'Technical boxer, Olympic medalist', 'narrative': 'Olympic medalist, WBA champion',
                      'achievements': ['WBA Champion', 'Olympic Medal'], 'reputation': 0.81, 'tier': 'contender'},
    
    # Middleweight/Super Middleweight
    'Canelo Alvarez': {'nationality': 'Mexican', 'weight_class': 'Super Middleweight', 'record': '60-2-2',
                       'style': 'Power puncher, body attack specialist', 'narrative': 'Mexican superstar, four-division champion',
                       'achievements': ['Undisputed Super Middleweight', '4-division champ'], 'reputation': 0.94, 'tier': 'elite'},
    'Gennady Golovkin': {'nationality': 'Kazakhstani', 'weight_class': 'Middleweight', 'record': '42-2-1',
                          'style': 'Power puncher, pressure fighter', 'narrative': 'GGG, knockout artist, middleweight king',
                          'achievements': ['Former Unified Middleweight', '38 KOs'], 'reputation': 0.90, 'tier': 'elite'},
    'Jermall Charlo': {'nationality': 'American', 'weight_class': 'Middleweight', 'record': '32-0-0',
                       'style': 'Power puncher, athletic', 'narrative': 'Hitman, WBC middleweight champion',
                       'achievements': ['WBC Middleweight Champion'], 'reputation': 0.86, 'tier': 'contender'},
    'Demetrius Andrade': {'nationality': 'American', 'weight_class': 'Middleweight', 'record': '32-0-0',
                          'style': 'Southpaw, technical boxer', 'narrative': 'Boo Boo, WBO middleweight champion',
                          'achievements': ['WBO Middleweight Champion'], 'reputation': 0.83, 'tier': 'contender'},
    
    # Lightweight
    'Gervonta Davis': {'nationality': 'American', 'weight_class': 'Lightweight', 'record': '29-0-0',
                       'style': 'Power puncher, explosive finisher', 'narrative': 'Tank, knockout artist, rising star',
                       'achievements': ['Lightweight Champion', '28 KOs'], 'reputation': 0.87, 'tier': 'elite'},
    'Ryan Garcia': {'nationality': 'American', 'weight_class': 'Lightweight', 'record': '24-1-0',
                    'style': 'Speed, power, social media star', 'narrative': 'KingRy, social media sensation, comeback story',
                    'achievements': ['Lightweight contender', '20 KOs'], 'reputation': 0.85, 'tier': 'contender'},
    'Devin Haney': {'nationality': 'American', 'weight_class': 'Lightweight', 'record': '31-0-0',
                    'style': 'Technical boxer, jab specialist', 'narrative': 'The Dream, undisputed lightweight champion',
                    'achievements': ['Undisputed Lightweight Champion'], 'reputation': 0.88, 'tier': 'elite'},
    'Vasiliy Lomachenko': {'nationality': 'Ukrainian', 'weight_class': 'Lightweight', 'record': '17-3-0',
                            'style': 'Technical master, footwork, angles', 'narrative': 'Loma, two-time Olympic gold, pound-for-pound elite',
                            'achievements': ['Two-Time Olympic Gold', 'P4P Elite'], 'reputation': 0.89, 'tier': 'elite'},
    'Teofimo Lopez': {'nationality': 'American', 'weight_class': 'Lightweight', 'record': '19-1-0',
                      'style': 'Power puncher, athletic', 'narrative': 'The Takeover, former undisputed champion',
                      'achievements': ['Former Undisputed Lightweight'], 'reputation': 0.84, 'tier': 'contender'},
    
    # Lower Weight Classes
    'Naoya Inoue': {'nationality': 'Japanese', 'weight_class': 'Bantamweight', 'record': '26-0-0',
                    'style': 'Power puncher, body attack', 'narrative': 'The Monster, undisputed champion, pound-for-pound elite',
                    'achievements': ['Undisputed Bantamweight', 'P4P #2'], 'reputation': 0.91, 'tier': 'elite'},
    'Nonito Donaire': {'nationality': 'Filipino-American', 'weight_class': 'Bantamweight', 'record': '42-7-0',
                       'style': 'Power puncher, left hook', 'narrative': 'The Flash, four-division champion, legend',
                       'achievements': ['4-Division Champion', 'Legend'], 'reputation': 0.86, 'tier': 'contender'},
    
    # Additional fighters for variety
    'Jermell Charlo': {'nationality': 'American', 'weight_class': 'Super Welterweight', 'record': '35-2-1',
                        'style': 'Power puncher, unified champion', 'narrative': 'Iron Man, undisputed super welterweight',
                        'achievements': ['Undisputed Super Welterweight'], 'reputation': 0.87, 'tier': 'elite'},
    'Tim Tszyu': {'nationality': 'Australian', 'weight_class': 'Super Welterweight', 'record': '24-0-0',
                  'style': 'Aggressive puncher, pressure', 'narrative': 'Sonic, rising star, Kostya Tszyu son',
                  'achievements': ['WBO Super Welterweight'], 'reputation': 0.82, 'tier': 'contender'},
}

# Promotions
PROMOTIONS = [
    'Top Rank', 'Matchroom Boxing', 'Premier Boxing Champions', 'Golden Boy Promotions',
    'Queensberry Promotions', 'Boxxer', 'Probellum', 'DAZN', 'ESPN', 'Showtime'
]

# Venues
VENUES = {
    'prestige': ['T-Mobile Arena, Las Vegas', 'Madison Square Garden, New York', 'Wembley Stadium, London',
                 'MGM Grand, Las Vegas', 'Staples Center, Los Angeles', 'Barclays Center, Brooklyn'],
    'regular': ['Mandalay Bay, Las Vegas', 'The O2, London', 'MGM National Harbor, Maryland',
                'Foxwoods Resort, Connecticut', 'Hard Rock Hotel, Las Vegas', 'York Hall, London']
}

# Fight types
FIGHT_TYPES = {
    'title': ['Undisputed Championship', 'Unified Championship', 'WBC Championship', 'WBA Championship',
              'WBO Championship', 'IBF Championship', 'Interim Championship'],
    'eliminator': ['Title Eliminator', 'Final Eliminator', 'Mandatory Eliminator'],
    'regular': ['10-Round Bout', '12-Round Bout', '8-Round Bout']
}


def generate_fighter_name(tier='contender'):
    """Generate realistic fighter name."""
    first_names = ['Mike', 'James', 'Robert', 'David', 'Michael', 'Chris', 'Mark', 'Paul', 'John', 'Kevin',
                   'Carlos', 'Miguel', 'Jose', 'Luis', 'Antonio', 'Fernando', 'Ricardo', 'Diego', 'Alejandro']
    last_names = ['Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                  'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson']
    
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def generate_fighter_record(tier='contender', weight_class='Heavyweight'):
    """Generate realistic boxing record based on tier."""
    if tier == 'elite':
        wins = random.randint(20, 45)
        losses = random.randint(0, 3)
        draws = random.randint(0, 2)
    elif tier == 'contender':
        wins = random.randint(15, 30)
        losses = random.randint(2, 8)
        draws = random.randint(0, 2)
    else:  # prospect
        wins = random.randint(8, 20)
        losses = random.randint(0, 5)
        draws = random.randint(0, 2)
    
    return f"{wins}-{losses}-{draws}"


def generate_fighter_style(weight_class):
    """Generate realistic fighting style."""
    styles = {
        'Heavyweight': ['Power puncher', 'Boxer-puncher', 'Technical boxer', 'Pressure fighter', 'Counter-puncher'],
        'Welterweight': ['Technical boxer', 'Power puncher', 'Volume puncher', 'Counter-puncher', 'Switch-hitter'],
        'Lightweight': ['Speed boxer', 'Power puncher', 'Technical master', 'Volume puncher', 'Counter-puncher'],
        'Middleweight': ['Power puncher', 'Technical boxer', 'Pressure fighter', 'Counter-puncher'],
        'Bantamweight': ['Speed boxer', 'Power puncher', 'Technical boxer', 'Volume puncher']
    }
    
    base_style = random.choice(styles.get(weight_class, ['Technical boxer', 'Power puncher']))
    modifiers = ['with good footwork', 'exceptional timing', 'body attack specialist', 'head movement',
                 'exceptional speed', 'devastating power', 'unorthodox angles']
    
    return f"{base_style}, {random.choice(modifiers)}"


def generate_narrative(fighter1, fighter2, fight_info):
    """Generate rich narrative for fight."""
    f1 = fighter1
    f2 = fighter2
    
    narrative_parts = []
    
    # Opening
    narrative_parts.append(
        f"{f1['name']} ({f1['nationality']}, {f1['weight_class']}) "
        f"faces {f2['name']} ({f2['nationality']}, {f2['weight_class']}) "
        f"in a {fight_info['title']} at {fight_info['venue']}."
    )
    
    # Fighter backgrounds
    if f1.get('achievements'):
        narrative_parts.append(
            f"{f1['name']}, known as '{f1['narrative']}', brings a record of "
            f"{f1['record']} and achievements including {', '.join(f1['achievements'][:2])}. "
            f"His style is characterized as {f1['style']}."
        )
    else:
        narrative_parts.append(
            f"{f1['name']} enters with a {f1['record']} record and is known for "
            f"{f1['style']}. {f1['narrative']}"
        )
    
    if f2.get('achievements'):
        narrative_parts.append(
            f"{f2['name']}, {f2['narrative']}, enters with a {f2['record']} record "
            f"and credentials including {', '.join(f2['achievements'][:2])}. "
            f"He is known for {f2['style']}."
        )
    else:
        narrative_parts.append(
            f"{f2['name']} brings a {f2['record']} record and {f2['style']}. "
            f"{f2['narrative']}"
        )
    
    # Significance
    narrative_parts.append(
        f"This fight represents {fight_info['significance']}. "
        f"The bout is promoted by {fight_info['promotion']} and scheduled for "
        f"{fight_info['rounds']} rounds."
    )
    
    # Style matchup
    narrative_parts.append(
        f"The stylistic matchup pits {f1['style']} against {f2['style']}, "
        f"creating an intriguing clash of approaches."
    )
    
    # Reputation comparison
    if f1['reputation'] > f2['reputation']:
        narrative_parts.append(
            f"{f1['name']} enters as the more established fighter with higher reputation, "
            f"while {f2['name']} looks to prove himself on the biggest stage."
        )
    elif f2['reputation'] > f1['reputation']:
        narrative_parts.append(
            f"{f2['name']} is considered the favorite based on recent form and reputation, "
            f"but {f1['name']} has the opportunity to upset expectations."
        )
    else:
        narrative_parts.append(
            f"Both fighters are evenly matched in reputation, setting up a competitive "
            f"and unpredictable contest."
        )
    
    return " ".join(narrative_parts)


def expand_dataset():
    """Expand initial dataset to 5,000+ fights."""
    print("="*80)
    print("EXPANDING BOXING DATASET TO 5,000+ FIGHTS")
    print("="*80)
    
    # Load initial data
    data_file = data_dir / 'boxing_fights_complete.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    initial_fights = data['fights']
    fighters_db = data.get('fighters', {})
    
    # Ensure all fighters in database have 'name' field
    for name, fighter_data in fighters_db.items():
        if 'name' not in fighter_data:
            fighter_data['name'] = name
    
    # Ensure all fighters in FIGHTER_DATABASE have 'name' field
    for name, fighter_data in FIGHTER_DATABASE.items():
        fighter_data['name'] = name
    
    # Merge with expanded database
    all_fighters = {**fighters_db, **FIGHTER_DATABASE}
    
    print(f"\nStarting with {len(initial_fights)} initial fights")
    print(f"Fighter database: {len(all_fighters)} fighters")
    
    # Generate additional fights
    target_fights = 5000
    additional_needed = target_fights - len(initial_fights)
    
    print(f"\nGenerating {additional_needed} additional fights...")
    
    new_fights = []
    fight_id = len(initial_fights) + 1
    
    # Date range: 2020-01-01 to 2024-12-31
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days
    
    # Weight class distribution
    weight_classes = ['Heavyweight', 'Cruiserweight', 'Light Heavyweight', 'Super Middleweight',
                      'Middleweight', 'Super Welterweight', 'Welterweight', 'Super Lightweight',
                      'Lightweight', 'Super Featherweight', 'Featherweight', 'Super Bantamweight',
                      'Bantamweight']
    weight_probs = [0.15, 0.05, 0.08, 0.10, 0.10, 0.08, 0.15, 0.05, 0.12, 0.03, 0.04, 0.02, 0.03]
    
    # Tier distribution
    tier_probs = {'elite': 0.15, 'contender': 0.35, 'prospect': 0.50}
    
    for i in range(additional_needed):
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{additional_needed} fights...")
        
        # Random date
        days_offset = random.randint(0, date_range)
        fight_date = start_date + timedelta(days=days_offset)
        
        # Weight class
        weight_class = np.random.choice(weight_classes, p=weight_probs)
        
        # Select fighters
        # 30% chance to use known fighter, 70% generate new
        if random.random() < 0.3 and len(all_fighters) > 0:
            # Use existing fighter
            f1_name = random.choice(list(all_fighters.keys()))
            f1_data = all_fighters[f1_name].copy()
            f1_data['name'] = f1_name  # Ensure name is set
            f1_data['weight_class'] = weight_class  # Adjust if needed
        else:
            # Generate new fighter
            tier = np.random.choice(list(tier_probs.keys()), p=list(tier_probs.values()))
            f1_name = generate_fighter_name(tier)
            f1_data = {
                'name': f1_name,
                'nationality': random.choice(['American', 'British', 'Mexican', 'Cuban', 'Ukrainian', 'Filipino']),
                'weight_class': weight_class,
                'record': generate_fighter_record(tier, weight_class),
                'style': generate_fighter_style(weight_class),
                'narrative': f"Rising {tier} in {weight_class} division",
                'achievements': [] if tier != 'elite' else [f'{weight_class} Contender'],
                'reputation': {'elite': 0.85, 'contender': 0.70, 'prospect': 0.55}[tier],
                'tier': tier
            }
            all_fighters[f1_name] = f1_data
        
        # Fighter 2
        if random.random() < 0.3 and len(all_fighters) > 1:
            available = [f for f in all_fighters.keys() if f != f1_name]
            f2_name = random.choice(available)
            f2_data = all_fighters[f2_name].copy()
            f2_data['name'] = f2_name  # Ensure name is set
            f2_data['weight_class'] = weight_class
        else:
            tier = np.random.choice(list(tier_probs.keys()), p=list(tier_probs.values()))
            f2_name = generate_fighter_name(tier)
            f2_data = {
                'name': f2_name,
                'nationality': random.choice(['American', 'British', 'Mexican', 'Cuban', 'Ukrainian', 'Filipino']),
                'weight_class': weight_class,
                'record': generate_fighter_record(tier, weight_class),
                'style': generate_fighter_style(weight_class),
                'narrative': f"Rising {tier} in {weight_class} division",
                'achievements': [] if tier != 'elite' else [f'{weight_class} Contender'],
                'reputation': {'elite': 0.85, 'contender': 0.70, 'prospect': 0.55}[tier],
                'tier': tier
            }
            all_fighters[f2_name] = f2_data
        
        # Fight details
        is_title = random.random() < 0.20  # 20% title fights
        
        if is_title:
            title = random.choice(FIGHT_TYPES['title'])
            significance = f"{title} bout between top contenders"
            venue = random.choice(VENUES['prestige'])
        else:
            if random.random() < 0.10:
                title = random.choice(FIGHT_TYPES['eliminator'])
                significance = "Title eliminator bout"
            else:
                title = random.choice(FIGHT_TYPES['regular'])
                significance = "Competitive matchup"
            venue = random.choice(VENUES['regular'] if random.random() < 0.7 else VENUES['prestige'])
        
        promotion = random.choice(PROMOTIONS)
        rounds = 12 if is_title else random.choice([8, 10, 12])
        
        # Determine winner (based on reputation + randomness)
        f1_win_prob = f1_data['reputation'] / (f1_data['reputation'] + f2_data['reputation'])
        f1_wins = random.random() < f1_win_prob
        
        winner = f1_name if f1_wins else f2_name
        loser = f2_name if f1_wins else f1_name
        
        # Method
        methods = ['Unanimous Decision', 'Split Decision', 'Majority Decision', 
                   'Technical Knockout', 'Knockout', 'Technical Decision']
        method = random.choice(methods)
        
        if 'Knockout' in method or 'TKO' in method:
            round_finished = random.randint(1, rounds)
        else:
            round_finished = rounds
        
        # Generate narrative
        fight_info = {
            'title': title,
            'venue': venue,
            'promotion': promotion,
            'rounds': rounds,
            'significance': significance
        }
        
        narrative = generate_narrative(f1_data, f2_data, fight_info)
        
        fight = {
            'fight_id': fight_id,
            'date': fight_date.strftime('%Y-%m-%d'),
            'fighter1': f1_data.copy(),
            'fighter2': f2_data.copy(),
            'venue': venue,
            'promotion': promotion,
            'title': title,
            'rounds_scheduled': rounds,
            'round_finished': round_finished,
            'method': method,
            'winner': winner,
            'loser': loser,
            'significance': significance,
            'narrative': narrative,
            'outcome': 1 if f1_wins else 0
        }
        
        new_fights.append(fight)
        fight_id += 1
    
    # Combine
    all_fights = initial_fights + new_fights
    
    print(f"\n✓ Generated {len(new_fights)} additional fights")
    print(f"✓ Total fights: {len(all_fights)}")
    print(f"✓ Total fighters: {len(all_fighters)}")
    
    # Save expanded dataset
    output_file = data_dir / 'boxing_fights_expanded.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_fights': len(all_fights),
            'date_range': '2020-2024',
            'fights': all_fights,
            'fighters': all_fighters,
            'collection_date': datetime.now().isoformat(),
            'expansion_date': datetime.now().isoformat(),
            'initial_fights': len(initial_fights),
            'generated_fights': len(new_fights)
        }, f, indent=2)
    
    print(f"\n✓ Saved expanded dataset to {output_file}")
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Title fights: {sum(1 for f in all_fights if 'Championship' in f.get('title', ''))}")
    print(f"  Weight classes: {len(set(f['fighter1']['weight_class'] for f in all_fights))}")
    print(f"  Promotions: {len(set(f['promotion'] for f in all_fights))}")
    print(f"  Date range: {min(f['date'] for f in all_fights)} to {max(f['date'] for f in all_fights)}")
    
    return all_fights, all_fighters


if __name__ == '__main__':
    print("Starting Boxing Dataset Expansion...")
    fights, fighters = expand_dataset()
    print(f"\n✓ Expansion complete: {len(fights)} total fights")
    print(f"✓ Ready for full analysis")

