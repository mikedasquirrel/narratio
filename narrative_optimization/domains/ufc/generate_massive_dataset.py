"""
Generate Massive UFC Dataset - 5,000+ Fights

Creates comprehensive UFC fight database with:
- Real fighter names (100+ fighters)
- Betting odds
- Fighter statistics
- Context (title fights, rivalries)
- 2014-2024 date range
"""

import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

# Comprehensive fighter database
FIGHTERS = [
    # Legends & All-Time Greats
    ("Conor McGregor", "The Notorious", "Lightweight"),
    ("Khabib Nurmagomedov", "The Eagle", "Lightweight"),
    ("Jon Jones", "Bones", "Light Heavyweight"),
    ("Daniel Cormier", "DC", "Light Heavyweight"),
    ("Anderson Silva", "The Spider", "Middleweight"),
    ("Georges St-Pierre", "GSP", "Welterweight"),
    ("Demetrious Johnson", "Mighty Mouse", "Flyweight"),
    ("Jose Aldo", "Junior", "Featherweight"),
    ("Dominick Cruz", "The Dominator", "Bantamweight"),
    ("Cain Velasquez", "", "Heavyweight"),
    
    # Current Champions & Top Contenders
    ("Israel Adesanya", "The Last Stylebender", "Middleweight"),
    ("Alex Pereira", "Poatan", "Light Heavyweight"),
    ("Jamahal Hill", "Sweet Dreams", "Light Heavyweight"),
    ("Islam Makhachev", "", "Lightweight"),
    ("Leon Edwards", "Rocky", "Welterweight"),
    ("Alexander Volkanovski", "The Great", "Featherweight"),
    ("Charles Oliveira", "Do Bronx", "Lightweight"),
    ("Justin Gaethje", "The Highlight", "Lightweight"),
    ("Sean O'Malley", "Suga", "Bantamweight"),
    ("Aljamain Sterling", "Funk Master", "Bantamweight"),
    ("Francis Ngannou", "The Predator", "Heavyweight"),
    ("Ciryl Gane", "Bon Gamin", "Heavyweight"),
    ("Tom Aspinall", "", "Heavyweight"),
    ("Stipe Miocic", "", "Heavyweight"),
    ("Jan Blachowicz", "", "Light Heavyweight"),
    ("Glover Teixeira", "", "Light Heavyweight"),
    ("Jiri Prochazka", "BJP", "Light Heavyweight"),
    
    # Lightweights
    ("Dustin Poirier", "The Diamond", "Lightweight"),
    ("Michael Chandler", "Iron", "Lightweight"),
    ("Tony Ferguson", "El Cucuy", "Lightweight"),
    ("Rafael dos Anjos", "RDA", "Lightweight"),
    ("Eddie Alvarez", "", "Lightweight"),
    ("Beneil Dariush", "", "Lightweight"),
    ("Arman Tsarukyan", "", "Lightweight"),
    ("Michael Chiesa", "Maverick", "Lightweight"),
    ("Dan Hooker", "The Hangman", "Lightweight"),
    ("Paul Felder", "The Irish Dragon", "Lightweight"),
    
    # Welterweights
    ("Kamaru Usman", "The Nigerian Nightmare", "Welterweight"),
    ("Colby Covington", "Chaos", "Welterweight"),
    ("Gilbert Burns", "Durinho", "Welterweight"),
    ("Belal Muhammad", "Remember the Name", "Welterweight"),
    ("Stephen Thompson", "Wonderboy", "Welterweight"),
    ("Jorge Masvidal", "Gamebred", "Welterweight"),
    ("Nate Diaz", "", "Welterweight"),
    ("Nick Diaz", "", "Welterweight"),
    ("Vicente Luque", "The Silent Assassin", "Welterweight"),
    ("Tyron Woodley", "The Chosen One", "Welterweight"),
    ("Robbie Lawler", "Ruthless", "Welterweight"),
    ("Shavkat Rakhmonov", "Nomad", "Welterweight"),
    
    # Middleweights
    ("Robert Whittaker", "The Reaper", "Middleweight"),
    ("Marvin Vettori", "The Italian Dream", "Middleweight"),
    ("Paulo Costa", "The Eraser", "Middleweight"),
    ("Jared Cannonier", "The Killa Gorilla", "Middleweight"),
    ("Derek Brunson", "", "Middleweight"),
    ("Kelvin Gastelum", "", "Middleweight"),
    ("Sean Strickland", "Tarzan", "Middleweight"),
    ("Dricus du Plessis", "Stillknocks", "Middleweight"),
    ("Yoel Romero", "Soldier of God", "Middleweight"),
    ("Chris Weidman", "The All-American", "Middleweight"),
    ("Luke Rockhold", "", "Middleweight"),
    
    # Featherweights
    ("Max Holloway", "Blessed", "Featherweight"),
    ("Brian Ortega", "T-City", "Featherweight"),
    ("Yair Rodriguez", "El Pantera", "Featherweight"),
    ("Calvin Kattar", "The Boston Finisher", "Featherweight"),
    ("Arnold Allen", "Almighty", "Featherweight"),
    ("Chan Sung Jung", "The Korean Zombie", "Featherweight"),
    ("Frankie Edgar", "The Answer", "Featherweight"),
    ("Ilia Topuria", "El Matador", "Featherweight"),
    ("Movsar Evloev", "", "Featherweight"),
    
    # Bantamweights
    ("Petr Yan", "No Mercy", "Bantamweight"),
    ("Cory Sandhagen", "The Sandman", "Bantamweight"),
    ("TJ Dillashaw", "", "Bantamweight"),
    ("Merab Dvalishvili", "The Machine", "Bantamweight"),
    ("Marlon Vera", "Chito", "Bantamweight"),
    ("Rob Font", "", "Bantamweight"),
    ("Song Yadong", "", "Bantamweight"),
    
    # Flyweights
    ("Deiveson Figueiredo", "Deus da Guerra", "Flyweight"),
    ("Brandon Moreno", "The Assassin Baby", "Flyweight"),
    ("Alexandre Pantoja", "The Cannibal", "Flyweight"),
    ("Brandon Royval", "Raw Dawg", "Flyweight"),
    ("Henry Cejudo", "The Messenger", "Flyweight"),
    
    # Women's Divisions
    ("Amanda Nunes", "The Lioness", "Women's Bantamweight"),
    ("Valentina Shevchenko", "Bullet", "Women's Flyweight"),
    ("Rose Namajunas", "Thug", "Women's Strawweight"),
    ("Weili Zhang", "Magnum", "Women's Strawweight"),
    ("Joanna Jedrzejczyk", "Joanna Champion", "Women's Strawweight"),
    ("Jessica Andrade", "Bate Estaca", "Women's Strawweight"),
    ("Holly Holm", "The Preacher's Daughter", "Women's Bantamweight"),
    ("Julianna Pena", "The Venezuelan Vixen", "Women's Bantamweight"),
    ("Alexa Grasso", "", "Women's Flyweight"),
    ("Carla Esparza", "Cookie Monster", "Women's Strawweight"),
]

def generate_massive_dataset(num_fights=5500):
    """Generate comprehensive UFC dataset."""
    print("="*80)
    print(f"GENERATING MASSIVE UFC DATASET - {num_fights} FIGHTS")
    print("="*80)
    
    fights = []
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2024, 11, 1)
    total_days = (end_date - start_date).days
    
    print(f"\nFighters in database: {len(FIGHTERS)}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"\nGenerating fights...")
    
    for i in range(num_fights):
        # Random fighters from same weight class when possible
        f1_idx = random.randint(0, len(FIGHTERS) - 1)
        fighter_a = FIGHTERS[f1_idx]
        
        # Try to match weight class
        same_weight = [f for f in FIGHTERS if f[2] == fighter_a[2] and f != fighter_a]
        if same_weight and random.random() > 0.3:
            fighter_b = random.choice(same_weight)
        else:
            f2_idx = random.randint(0, len(FIGHTERS) - 1)
            while f2_idx == f1_idx:
                f2_idx = random.randint(0, len(FIGHTERS) - 1)
            fighter_b = FIGHTERS[f2_idx]
        
        # Random date
        days_offset = random.randint(0, total_days)
        fight_date = start_date + timedelta(days=days_offset)
        
        # Random outcome
        winner_is_a = random.choice([True, False])
        
        # Betting odds (realistic distribution)
        odds_type = random.random()
        if odds_type < 0.4:  # Heavy favorite
            odds_a, odds_b = random.choice([(-300, +250), (-400, +320), (-250, +200)])
        elif odds_type < 0.7:  # Moderate favorite
            odds_a, odds_b = random.choice([(-180, +150), (-200, +170), (-150, +130)])
        else:  # Pick'em
            odds_a, odds_b = random.choice([(-110, -110), (+105, -115), (-105, +105)])
        
        # Randomly flip odds
        if random.random() > 0.5:
            odds_a, odds_b = odds_b, odds_a
        
        # Title fight (10% of fights)
        is_title = random.random() < 0.10
        
        # Win method
        methods = [
            ('KO/TKO', 0.35),
            ('Submission', 0.25),
            ('Decision - Unanimous', 0.25),
            ('Decision - Split', 0.10),
            ('Decision - Majority', 0.05)
        ]
        win_method = random.choices([m[0] for m in methods], weights=[m[1] for m in methods])[0]
        
        # Round and time
        if 'KO/TKO' in win_method or 'Submission' in win_method:
            last_round = random.choices([1,2,3,4,5], weights=[0.3,0.3,0.2,0.1,0.1])[0]
            minutes = random.randint(0, 4)
            seconds = random.randint(0, 59)
            time = f"{minutes}:{seconds:02d}"
        else:
            last_round = 3 if not is_title else 5
            time = "5:00"
        
        fight = {
            'fight_id': f'ufc_fight_{i:05d}',
            'R_fighter': fighter_a[0],
            'R_nickname': fighter_a[1],
            'B_fighter': fighter_b[0],
            'B_nickname': fighter_b[1],
            'Winner': fighter_a[0] if winner_is_a else fighter_b[0],
            'date': fight_date.strftime('%Y-%m-%d'),
            'year': fight_date.year,
            'R_odds': odds_a,
            'B_odds': odds_b,
            'R_wins': random.randint(8, 28),
            'R_losses': random.randint(0, 8),
            'R_draw': random.choice([0, 0, 0, 1]),
            'B_wins': random.randint(8, 28),
            'B_losses': random.randint(0, 8),
            'B_draw': random.choice([0, 0, 0, 1]),
            'R_age': random.randint(23, 39),
            'B_age': random.randint(23, 39),
            'R_Height_cms': random.randint(165, 200),
            'B_Height_cms': random.randint(165, 200),
            'R_Reach_cms': random.randint(165, 215),
            'B_Reach_cms': random.randint(165, 215),
            'R_Stance': random.choice(['Orthodox', 'Orthodox', 'Orthodox', 'Southpaw', 'Switch']),
            'B_Stance': random.choice(['Orthodox', 'Orthodox', 'Orthodox', 'Southpaw', 'Switch']),
            'weight_class': fighter_a[2],
            'title_bout': is_title,
            'win_by': win_method,
            'last_round': last_round,
            'last_round_time': time,
            'R_avg_SIG_STR_pct': round(random.uniform(30, 70), 2),
            'B_avg_SIG_STR_pct': round(random.uniform(30, 70), 2),
            'R_avg_SUB_ATT': round(random.uniform(0, 5), 2),
            'B_avg_SUB_ATT': round(random.uniform(0, 5), 2),
            'R_avg_TD_pct': round(random.uniform(10, 80), 2),
            'B_avg_TD_pct': round(random.uniform(10, 80), 2),
            'R_current_win_streak': random.randint(0, 12),
            'B_current_win_streak': random.randint(0, 12),
            'event': f"UFC {random.randint(150, 310)}" if random.random() > 0.3 else f"UFC Fight Night {random.randint(100, 250)}",
            'location': random.choice(['Las Vegas', 'Abu Dhabi', 'New York', 'Los Angeles', 'London', 'Brazil', 'Canada', 'Australia']),
        }
        
        fights.append(fight)
        
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_fights} fights...")
    
    df = pd.DataFrame(fights)
    
    # Save
    output_dir = Path("data/domains")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ufc_massive_dataset.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Generated {len(df)} fights")
    print(f"✓ Saved to: {output_file}")
    print(f"\nDataset Statistics:")
    print(f"  - Total fights: {len(df)}")
    print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  - Unique fighters: {len(set(df['R_fighter']) | set(df['B_fighter']))}")
    print(f"  - Title fights: {df['title_bout'].sum()} ({df['title_bout'].sum()/len(df)*100:.1f}%)")
    print(f"  - Weight classes: {df['weight_class'].nunique()}")
    print(f"  - Years: {sorted(df['year'].unique())}")
    
    # Win method distribution
    print(f"\nWin Methods:")
    for method, count in df['win_by'].value_counts().items():
        print(f"  - {method}: {count} ({count/len(df)*100:.1f}%)")
    
    return df, str(output_file)

def main():
    """Generate massive dataset."""
    df, filepath = generate_massive_dataset(5500)
    
    print("\n" + "="*80)
    print("MASSIVE DATASET COMPLETE!")
    print("="*80)
    print(f"\nDataset ready: {filepath}")
    print(f"\nNext steps:")
    print(f"1. Run: python generate_fighter_narratives.py")
    print(f"2. Run: python analyze_ufc_complete.py")
    print(f"3. Test narrativity (п) and correlation (r)")
    
    return df

if __name__ == "__main__":
    main()

