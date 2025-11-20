"""
Oscar Best Picture - Complete Competitive Analysis

For each year: Winner + ALL nominees (gravitational field)

Extract EVERY nominative element:
- Film title (nominative properties)
- Director name (nominative gravity)
- Lead actors + character names (full cast nominative breakdown)
- Supporting actors + characters
- Setting locations (geographic nominatives)
- Key objects/symbols mentioned in plot (nominative anchors)
- Studio name
- Production company names

Then test: What distinguishes winner from nominees in gravitational narrative space?

This is RELATIONAL analysis - winner emerges from competitive field.
"""

import json
from pathlib import Path

# Real Oscar data - will collect complete nominative breakdowns
# Starting with recent years where data is most accessible

OSCAR_YEARS = [
    {
        "year": 2024,
        "winner": {
            "title": "Oppenheimer",
            "director": "Christopher Nolan",
            "lead_actors": [
                {"actor": "Cillian Murphy", "character": "J. Robert Oppenheimer"},
                {"actor": "Emily Blunt", "character": "Kitty Oppenheimer"},
                {"actor": "Matt Damon", "character": "Leslie Groves"},
                {"actor": "Robert Downey Jr.", "character": "Lewis Strauss"}
            ],
            "settings": ["Los Alamos", "New Mexico", "Princeton", "Washington DC"],
            "key_nominatives": ["Manhattan Project", "Trinity", "atomic bomb", "Berkeley"],
            "studio": "Universal Pictures",
            "plot_summary": "Biopic of physicist J. Robert Oppenheimer focusing on his role in developing the atomic bomb",
            "won": True
        },
        "nominees": [
            {
                "title": "Killers of the Flower Moon",
                "director": "Martin Scorsese",
                "lead_actors": [
                    {"actor": "Leonardo DiCaprio", "character": "Ernest Burkhart"},
                    {"actor": "Lily Gladstone", "character": "Mollie Kyle"},
                    {"actor": "Robert De Niro", "character": "William Hale"}
                ],
                "settings": ["Oklahoma", "Osage Nation", "1920s America"],
                "key_nominatives": ["Osage", "oil money", "FBI investigation"],
                "studio": "Apple/Paramount",
                "won": False
            },
            {
                "title": "Poor Things",
                "director": "Yorgos Lanthimos",
                "lead_actors": [
                    {"actor": "Emma Stone", "character": "Bella Baxter"},
                    {"actor": "Mark Ruffalo", "character": "Duncan Wedderburn"},
                    {"actor": "Willem Dafoe", "character": "Godwin Baxter"}
                ],
                "settings": ["Victorian London", "Lisbon", "Paris", "Alexandria"],
                "key_nominatives": ["Frankenstein-like", "resurrection", "liberation"],
                "studio": "Searchlight Pictures",
                "won": False
            }
            # Need to add remaining nominees: Barbie, American Fiction, Anatomy of a Fall, etc.
        ]
    },
    {
        "year": 2023,
        "winner": {
            "title": "Everything Everywhere All at Once",
            "director": "Dan Kwan, Daniel Scheinert",
            "lead_actors": [
                {"actor": "Michelle Yeoh", "character": "Evelyn Quan Wang"},
                {"actor": "Stephanie Hsu", "character": "Joy Wang / Jobu Tupaki"},
                {"actor": "Ke Huy Quan", "character": "Waymond Wang"},
                {"actor": "Jamie Lee Curtis", "character": "Deirdre Beaubeirdre"}
            ],
            "settings": ["Los Angeles", "Multiverse", "Laundromat", "IRS Office", "Infinite parallel universes"],
            "key_nominatives": ["Alpha Universe", "Bagel", "Raccacoonie", "Hot Dog Fingers Universe"],
            "studio": "A24",
            "plot_summary": "Chinese-American immigrant discovers multiverse while doing taxes, must save reality",
            "won": True
        },
        "nominees": [
            {
                "title": "The Banshees of Inisherin",
                "director": "Martin McDonagh",
                "lead_actors": [
                    {"actor": "Colin Farrell", "character": "Pádraic Súilleabháin"},
                    {"actor": "Brendan Gleeson", "character": "Colm Doherty"},
                    {"actor": "Kerry Condon", "character": "Siobhán Súilleabháin"}
                ],
                "settings": ["Inisherin Island", "Ireland", "1923"],
                "key_nominatives": ["Irish Civil War", "Banshee", "Folk music"],
                "studio": "Searchlight Pictures",
                "won": False
            }
            # Need: All Quiet Western Front, Elvis, Tár, Triangle of Sadness, etc.
        ]
    }
]

def collect_complete_data():
    """
    Collect complete Oscar data with EVERY nominative element.
    
    For full analysis need:
    - All 10 nominees per year (not just 2-3)
    - Complete cast lists (20+ actors per film)
    - Character names for ALL roles
    - Setting names (every location)
    - Plot summaries (real from IMDb/Wikipedia)
    - Genre tags
    - Runtime, budget if available
    """
    print("=" * 80)
    print("OSCAR BEST PICTURE - COMPLETE NOMINATIVE COLLECTION")
    print("=" * 80)
    print("\n⚠️  Need to research each year's complete nominee list")
    print("⚠️  Extract ALL cast (not just leads)")
    print("⚠️  Get ALL character names")
    print("⚠️  Map ALL settings")
    print("\nThis requires manual research per year from:")
    print("  - IMDb (cast/character lists)")
    print("  - Wikipedia (plot summaries)")
    print("  - Oscar.org (official nominee lists)")
    print("\nCurrent: 2 years, 5 films (partial data)")
    print("Target: 10+ years, 100+ films (complete nominative extraction)")
    
    output_path = Path(__file__).parent.parent.parent.parent / 'data/domains/oscar_nominees_partial.json'
    with open(output_path, 'w') as f:
        json.dump(OSCAR_YEARS, f, indent=2)
    
    print(f"\n✓ Saved partial dataset: {output_path}")
    print("\nTo complete:")
    print("  1. Research each year 2014-2024 (10 years)")
    print("  2. Get all 8-10 nominees per year")
    print("  3. Extract complete cast for each")
    print("  4. Get all character names")
    print("  5. Map settings/locations")
    print("  6. Then analyze: Winner vs nominees in nominative space")


if __name__ == "__main__":
    collect_complete_data()

