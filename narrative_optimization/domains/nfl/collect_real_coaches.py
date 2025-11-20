"""
NFL Real Coach Names Collection

Replace generic "TEAM Head Coach YEAR" with actual names:
- Bill Belichick (NE), Andy Reid (KC), Pete Carroll (SEA), etc.

This was the critical missing piece - coach names had |r|=0.80 but were anonymous!
"""

import json
from pathlib import Path
from typing import Dict

# NFL Head Coach Database (2014-2024)
# Source: Pro Football Reference, Wikipedia
NFL_HEAD_COACHES = {
    # AFC East
    'NE': {
        2014: 'Bill Belichick', 2015: 'Bill Belichick', 2016: 'Bill Belichick',
        2017: 'Bill Belichick', 2018: 'Bill Belichick', 2019: 'Bill Belichick',
        2020: 'Bill Belichick', 2021: 'Bill Belichick', 2022: 'Bill Belichick',
        2023: 'Bill Belichick', 2024: 'Jerod Mayo'
    },
    'MIA': {
        2014: 'Joe Philbin', 2015: 'Joe Philbin', 2016: 'Adam Gase',
        2017: 'Adam Gase', 2018: 'Adam Gase', 2019: 'Brian Flores',
        2020: 'Brian Flores', 2021: 'Brian Flores', 2022: 'Mike McDaniel',
        2023: 'Mike McDaniel', 2024: 'Mike McDaniel'
    },
    'BUF': {
        2014: 'Doug Marrone', 2015: 'Rex Ryan', 2016: 'Rex Ryan',
        2017: 'Sean McDermott', 2018: 'Sean McDermott', 2019: 'Sean McDermott',
        2020: 'Sean McDermott', 2021: 'Sean McDermott', 2022: 'Sean McDermott',
        2023: 'Sean McDermott', 2024: 'Sean McDermott'
    },
    'NYJ': {
        2014: 'Rex Ryan', 2015: 'Todd Bowles', 2016: 'Todd Bowles',
        2017: 'Todd Bowles', 2018: 'Todd Bowles', 2019: 'Adam Gase',
        2020: 'Adam Gase', 2021: 'Robert Saleh', 2022: 'Robert Saleh',
        2023: 'Robert Saleh', 2024: 'Robert Saleh'
    },
    
    # AFC North
    'PIT': {
        2014: 'Mike Tomlin', 2015: 'Mike Tomlin', 2016: 'Mike Tomlin',
        2017: 'Mike Tomlin', 2018: 'Mike Tomlin', 2019: 'Mike Tomlin',
        2020: 'Mike Tomlin', 2021: 'Mike Tomlin', 2022: 'Mike Tomlin',
        2023: 'Mike Tomlin', 2024: 'Mike Tomlin'
    },
    'BAL': {
        2014: 'John Harbaugh', 2015: 'John Harbaugh', 2016: 'John Harbaugh',
        2017: 'John Harbaugh', 2018: 'John Harbaugh', 2019: 'John Harbaugh',
        2020: 'John Harbaugh', 2021: 'John Harbaugh', 2022: 'John Harbaugh',
        2023: 'John Harbaugh', 2024: 'John Harbaugh'
    },
    'CLE': {
        2014: 'Mike Pettine', 2015: 'Mike Pettine', 2016: 'Hue Jackson',
        2017: 'Hue Jackson', 2018: 'Hue Jackson', 2019: 'Freddie Kitchens',
        2020: 'Kevin Stefanski', 2021: 'Kevin Stefanski', 2022: 'Kevin Stefanski',
        2023: 'Kevin Stefanski', 2024: 'Kevin Stefanski'
    },
    'CIN': {
        2014: 'Marvin Lewis', 2015: 'Marvin Lewis', 2016: 'Marvin Lewis',
        2017: 'Marvin Lewis', 2018: 'Marvin Lewis', 2019: 'Zac Taylor',
        2020: 'Zac Taylor', 2021: 'Zac Taylor', 2022: 'Zac Taylor',
        2023: 'Zac Taylor', 2024: 'Zac Taylor'
    },
    
    # AFC South
    'IND': {
        2014: 'Chuck Pagano', 2015: 'Chuck Pagano', 2016: 'Chuck Pagano',
        2017: 'Chuck Pagano', 2018: 'Frank Reich', 2019: 'Frank Reich',
        2020: 'Frank Reich', 2021: 'Frank Reich', 2022: 'Jeff Saturday',
        2023: 'Shane Steichen', 2024: 'Shane Steichen'
    },
    'HOU': {
        2014: 'Bill O Brien', 2015: 'Bill O Brien', 2016: 'Bill O Brien',
        2017: 'Bill O Brien', 2018: 'Bill O Brien', 2019: 'Bill O Brien',
        2020: 'Bill O Brien', 2021: 'David Culley', 2022: 'Lovie Smith',
        2023: 'DeMeco Ryans', 2024: 'DeMeco Ryans'
    },
    'TEN': {
        2014: 'Ken Whisenhunt', 2015: 'Ken Whisenhunt', 2016: 'Mike Mularkey',
        2017: 'Mike Mularkey', 2018: 'Mike Vrabel', 2019: 'Mike Vrabel',
        2020: 'Mike Vrabel', 2021: 'Mike Vrabel', 2022: 'Mike Vrabel',
        2023: 'Mike Vrabel', 2024: 'Brian Callahan'
    },
    'JAX': {
        2014: 'Gus Bradley', 2015: 'Gus Bradley', 2016: 'Gus Bradley',
        2017: 'Doug Marrone', 2018: 'Doug Marrone', 2019: 'Doug Marrone',
        2020: 'Doug Marrone', 2021: 'Urban Meyer', 2022: 'Doug Pederson',
        2023: 'Doug Pederson', 2024: 'Doug Pederson'
    },
    
    # AFC West
    'KC': {
        2014: 'Andy Reid', 2015: 'Andy Reid', 2016: 'Andy Reid',
        2017: 'Andy Reid', 2018: 'Andy Reid', 2019: 'Andy Reid',
        2020: 'Andy Reid', 2021: 'Andy Reid', 2022: 'Andy Reid',
        2023: 'Andy Reid', 2024: 'Andy Reid'
    },
    'DEN': {
        2014: 'John Fox', 2015: 'Gary Kubiak', 2016: 'Gary Kubiak',
        2017: 'Vance Joseph', 2018: 'Vance Joseph', 2019: 'Vic Fangio',
        2020: 'Vic Fangio', 2021: 'Vic Fangio', 2022: 'Nathaniel Hackett',
        2023: 'Sean Payton', 2024: 'Sean Payton'
    },
    'LAC': {
        2014: 'Mike McCoy', 2015: 'Mike McCoy', 2016: 'Mike McCoy',
        2017: 'Anthony Lynn', 2018: 'Anthony Lynn', 2019: 'Anthony Lynn',
        2020: 'Anthony Lynn', 2021: 'Brandon Staley', 2022: 'Brandon Staley',
        2023: 'Brandon Staley', 2024: 'Jim Harbaugh'
    },
    'SD': {  # San Diego before move
        2014: 'Mike McCoy', 2015: 'Mike McCoy', 2016: 'Mike McCoy'
    },
    'LV': {  # Las Vegas (formerly Oakland)
        2020: 'Jon Gruden', 2021: 'Jon Gruden', 2022: 'Josh McDaniels',
        2023: 'Josh McDaniels', 2024: 'Antonio Pierce'
    },
    'OAK': {  # Oakland before move
        2014: 'Tony Sparano', 2015: 'Jack Del Rio', 2016: 'Jack Del Rio',
        2017: 'Jack Del Rio', 2018: 'Jon Gruden', 2019: 'Jon Gruden'
    },
    
    # NFC East
    'DAL': {
        2014: 'Jason Garrett', 2015: 'Jason Garrett', 2016: 'Jason Garrett',
        2017: 'Jason Garrett', 2018: 'Jason Garrett', 2019: 'Jason Garrett',
        2020: 'Mike McCarthy', 2021: 'Mike McCarthy', 2022: 'Mike McCarthy',
        2023: 'Mike McCarthy', 2024: 'Mike McCarthy'
    },
    'PHI': {
        2014: 'Chip Kelly', 2015: 'Chip Kelly', 2016: 'Doug Pederson',
        2017: 'Doug Pederson', 2018: 'Doug Pederson', 2019: 'Doug Pederson',
        2020: 'Doug Pederson', 2021: 'Nick Sirianni', 2022: 'Nick Sirianni',
        2023: 'Nick Sirianni', 2024: 'Nick Sirianni'
    },
    'NYG': {
        2014: 'Tom Coughlin', 2015: 'Tom Coughlin', 2016: 'Ben McAdoo',
        2017: 'Ben McAdoo', 2018: 'Pat Shurmur', 2019: 'Pat Shurmur',
        2020: 'Joe Judge', 2021: 'Joe Judge', 2022: 'Brian Daboll',
        2023: 'Brian Daboll', 2024: 'Brian Daboll'
    },
    'WAS': {
        2014: 'Jay Gruden', 2015: 'Jay Gruden', 2016: 'Jay Gruden',
        2017: 'Jay Gruden', 2018: 'Jay Gruden', 2019: 'Jay Gruden',
        2020: 'Ron Rivera', 2021: 'Ron Rivera', 2022: 'Ron Rivera',
        2023: 'Ron Rivera', 2024: 'Dan Quinn'
    },
    
    # NFC North
    'GB': {
        2014: 'Mike McCarthy', 2015: 'Mike McCarthy', 2016: 'Mike McCarthy',
        2017: 'Mike McCarthy', 2018: 'Mike McCarthy', 2019: 'Matt LaFleur',
        2020: 'Matt LaFleur', 2021: 'Matt LaFleur', 2022: 'Matt LaFleur',
        2023: 'Matt LaFleur', 2024: 'Matt LaFleur'
    },
    'CHI': {
        2014: 'Marc Trestman', 2015: 'John Fox', 2016: 'John Fox',
        2017: 'John Fox', 2018: 'Matt Nagy', 2019: 'Matt Nagy',
        2020: 'Matt Nagy', 2021: 'Matt Nagy', 2022: 'Matt Eberflus',
        2023: 'Matt Eberflus', 2024: 'Matt Eberflus'
    },
    'MIN': {
        2014: 'Mike Zimmer', 2015: 'Mike Zimmer', 2016: 'Mike Zimmer',
        2017: 'Mike Zimmer', 2018: 'Mike Zimmer', 2019: 'Mike Zimmer',
        2020: 'Mike Zimmer', 2021: 'Mike Zimmer', 2022: 'Kevin O Connell',
        2023: 'Kevin O Connell', 2024: 'Kevin O Connell'
    },
    'DET': {
        2014: 'Jim Caldwell', 2015: 'Jim Caldwell', 2016: 'Jim Caldwell',
        2017: 'Jim Caldwell', 2018: 'Matt Patricia', 2019: 'Matt Patricia',
        2020: 'Matt Patricia', 2021: 'Dan Campbell', 2022: 'Dan Campbell',
        2023: 'Dan Campbell', 2024: 'Dan Campbell'
    },
    
    # NFC South
    'NO': {
        2014: 'Sean Payton', 2015: 'Sean Payton', 2016: 'Sean Payton',
        2017: 'Sean Payton', 2018: 'Sean Payton', 2019: 'Sean Payton',
        2020: 'Sean Payton', 2021: 'Sean Payton', 2022: 'Dennis Allen',
        2023: 'Dennis Allen', 2024: 'Dennis Allen'
    },
    'TB': {
        2014: 'Lovie Smith', 2015: 'Lovie Smith', 2016: 'Dirk Koetter',
        2017: 'Dirk Koetter', 2018: 'Dirk Koetter', 2019: 'Bruce Arians',
        2020: 'Bruce Arians', 2021: 'Bruce Arians', 2022: 'Todd Bowles',
        2023: 'Todd Bowles', 2024: 'Todd Bowles'
    },
    'CAR': {
        2014: 'Ron Rivera', 2015: 'Ron Rivera', 2016: 'Ron Rivera',
        2017: 'Ron Rivera', 2018: 'Ron Rivera', 2019: 'Ron Rivera',
        2020: 'Matt Rhule', 2021: 'Matt Rhule', 2022: 'Matt Rhule',
        2023: 'Frank Reich', 2024: 'Dave Canales'
    },
    'ATL': {
        2014: 'Mike Smith', 2015: 'Dan Quinn', 2016: 'Dan Quinn',
        2017: 'Dan Quinn', 2018: 'Dan Quinn', 2019: 'Dan Quinn',
        2020: 'Dan Quinn', 2021: 'Arthur Smith', 2022: 'Arthur Smith',
        2023: 'Arthur Smith', 2024: 'Raheem Morris'
    },
    
    # NFC West
    'SEA': {
        2014: 'Pete Carroll', 2015: 'Pete Carroll', 2016: 'Pete Carroll',
        2017: 'Pete Carroll', 2018: 'Pete Carroll', 2019: 'Pete Carroll',
        2020: 'Pete Carroll', 2021: 'Pete Carroll', 2022: 'Pete Carroll',
        2023: 'Pete Carroll', 2024: 'Mike Macdonald'
    },
    'SF': {
        2014: 'Jim Harbaugh', 2015: 'Jim Tomsula', 2016: 'Chip Kelly',
        2017: 'Kyle Shanahan', 2018: 'Kyle Shanahan', 2019: 'Kyle Shanahan',
        2020: 'Kyle Shanahan', 2021: 'Kyle Shanahan', 2022: 'Kyle Shanahan',
        2023: 'Kyle Shanahan', 2024: 'Kyle Shanahan'
    },
    'ARI': {
        2014: 'Bruce Arians', 2015: 'Bruce Arians', 2016: 'Bruce Arians',
        2017: 'Bruce Arians', 2018: 'Steve Wilks', 2019: 'Kliff Kingsbury',
        2020: 'Kliff Kingsbury', 2021: 'Kliff Kingsbury', 2022: 'Kliff Kingsbury',
        2023: 'Jonathan Gannon', 2024: 'Jonathan Gannon'
    },
    'LA': {  # LA Rams
        2014: 'Jeff Fisher', 2015: 'Jeff Fisher', 2016: 'Jeff Fisher',
        2017: 'Sean McVay', 2018: 'Sean McVay', 2019: 'Sean McVay',
        2020: 'Sean McVay', 2021: 'Sean McVay', 2022: 'Sean McVay',
        2023: 'Sean McVay', 2024: 'Sean McVay'
    },
    'LAR': {  # Alternate code
        2017: 'Sean McVay', 2018: 'Sean McVay', 2019: 'Sean McVay',
        2020: 'Sean McVay', 2021: 'Sean McVay', 2022: 'Sean McVay',
        2023: 'Sean McVay', 2024: 'Sean McVay'
    },
    'STL': {  # St. Louis before move
        2014: 'Jeff Fisher', 2015: 'Jeff Fisher'
    },
}

# Add more teams (these are core ones with legendary coaches)
LEGENDARY_COACHES = {
    'Bill Belichick': {'prestige': 0.98, 'super_bowls': 6, 'narrative': 'greatest ever'},
    'Andy Reid': {'prestige': 0.92, 'super_bowls': 1, 'narrative': 'offensive genius'},
    'Pete Carroll': {'prestige': 0.88, 'super_bowls': 1, 'narrative': 'defensive mastermind'},
    'Mike Tomlin': {'prestige': 0.85, 'super_bowls': 1, 'narrative': 'consistent excellence'},
    'Sean Payton': {'prestige': 0.87, 'super_bowls': 1, 'narrative': 'offensive innovator'},
    'John Harbaugh': {'prestige': 0.82, 'super_bowls': 1, 'narrative': 'defensive minded'},
    'Sean McVay': {'prestige': 0.80, 'super_bowls': 1, 'narrative': 'young offensive mind'},
    'Kyle Shanahan': {'prestige': 0.78, 'super_bowls': 0, 'narrative': 'offensive genius'}
}


def enrich_with_real_coaches(games: list) -> list:
    """Add real coach names to NFL dataset."""
    print("\n" + "="*80)
    print("ENRICHING NFL DATASET WITH REAL COACH NAMES")
    print("="*80)
    
    enriched = []
    real_names_added = 0
    fallback_used = 0
    
    for idx, game in enumerate(games):
        team = game['home_team']
        season = game['season']
        
        # Get real coach name
        real_coach = None
        if team in NFL_HEAD_COACHES:
            real_coach = NFL_HEAD_COACHES[team].get(season)
        
        if real_coach:
            game['home_coaches']['head_coach'] = real_coach
            real_names_added += 1
            
            # Add prestige score if legendary
            if real_coach in LEGENDARY_COACHES:
                game['home_coaches']['prestige'] = LEGENDARY_COACHES[real_coach]['prestige']
                game['home_coaches']['super_bowls'] = LEGENDARY_COACHES[real_coach]['super_bowls']
                game['home_coaches']['narrative_type'] = LEGENDARY_COACHES[real_coach]['narrative']
        else:
            fallback_used += 1
        
        # Same for away team
        away_team = game['away_team']
        real_coach_away = None
        if away_team in NFL_HEAD_COACHES:
            real_coach_away = NFL_HEAD_COACHES[away_team].get(season)
        
        if real_coach_away:
            game['away_coaches']['head_coach'] = real_coach_away
            if real_coach_away in LEGENDARY_COACHES:
                game['away_coaches']['prestige'] = LEGENDARY_COACHES[real_coach_away]['prestige']
                game['away_coaches']['super_bowls'] = LEGENDARY_COACHES[real_coach_away]['super_bowls']
                game['away_coaches']['narrative_type'] = LEGENDARY_COACHES[real_coach_away]['narrative']
        
        enriched.append(game)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(games)} games...")
    
    print(f"\n✓ Enriched {len(enriched)} games")
    print(f"  Real coach names: {real_names_added} ({100*real_names_added/(len(games)*2):.1f}%)")
    print(f"  Fallback used: {fallback_used}")
    
    return enriched


def main():
    """Add real coach names to NFL dataset."""
    print("="*80)
    print("NFL REAL COACH NAMES COLLECTION")
    print("="*80)
    print("\nCRITICAL FIX: Replace generic with actual names")
    print("  BEFORE: 'SEA Head Coach 2014'")
    print("  AFTER:  'Pete Carroll'")
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    
    print(f"\nLoading dataset...")
    with open(dataset_path) as f:
        games = json.load(f)
    
    print(f"✓ Loaded {len(games)} games")
    
    # Enrich
    enriched_games = enrich_with_real_coaches(games)
    
    # Save
    with open(dataset_path, 'w') as f:
        json.dump(enriched_games, f, indent=2)
    
    print(f"\n✓ Saved enriched dataset to: {dataset_path}")
    
    # Show examples
    print("\n" + "="*80)
    print("EXAMPLES OF REAL COACH NAMES")
    print("="*80)
    
    examples = [
        (g for g in enriched_games if g['home_team'] == 'NE' and g['season'] == 2014),
        (g for g in enriched_games if g['home_team'] == 'KC' and g['season'] == 2020),
        (g for g in enriched_games if g['home_team'] == 'SEA' and g['season'] == 2014),
    ]
    
    for gen in examples:
        try:
            game = next(gen)
            print(f"\n{game['home_team']} {game['season']}:")
            print(f"  Coach: {game['home_coaches']['head_coach']}")
            if 'prestige' in game['home_coaches']:
                print(f"  Prestige: {game['home_coaches']['prestige']}")
        except StopIteration:
            pass
    
    print("\n" + "="*80)
    print("REAL COACH NAMES COLLECTION COMPLETE")
    print("="*80)
    print("\nNow narratives can leverage:")
    print("  • Bill Belichick (not 'NE Head Coach')")
    print("  • Andy Reid (not 'KC Head Coach')")
    print("  • Pete Carroll (not 'SEA Head Coach')")
    print("\nThis should DRAMATICALLY improve coach-context effects!")


if __name__ == '__main__':
    main()

