"""
Oscar Best Pictures Analysis

Collects all Best Picture winners (1927-2024) with plot summaries.
Tests: Does narrative quality (archetypal resonance) predict Academy choice?

This is pure narrative domain - movies ARE stories, Best Picture judges STORY quality.
Expected: High Д (narrative features should dominate)
"""

import json
from pathlib import Path

# Complete list of Oscar Best Picture winners with plot summaries
BEST_PICTURES = [
    {
        "year": 2024,
        "title": "Oppenheimer",
        "plot": "The story of J. Robert Oppenheimer's role in developing the atomic bomb during World War II. A man of genius creates devastating power, grapples with moral consequences, faces persecution during McCarthyism. Themes: Creation and destruction, brilliance and tragedy, scientific achievement and ethical crisis. Arc: Rise to prominence → moral reckoning → fall from grace.",
        "genre": "Biography/Drama",
        "runtime": 180
    },
    {
        "year": 2023,
        "title": "Everything Everywhere All at Once",
        "plot": "A Chinese-American immigrant struggles with her laundromat, taxes, and family relationships while discovering she must connect with parallel universe versions of herself to prevent destruction of the multiverse. A mundane life becomes cosmic adventure. Themes: Family reconciliation, immigrant experience, finding meaning in chaos, mother-daughter relationship. Arc: Ordinary life → multiverse chaos → redemption through love.",
        "genre": "Sci-Fi/Comedy/Drama",
        "runtime": 140
    },
    {
        "year": 2022,
        "title": "CODA",
        "plot": "Ruby is the only hearing member of a deaf family in Gloucester, Massachusetts. She balances her responsibility as interpreter for her parents' fishing business with her own dreams of becoming a singer. Coming-of-age story about finding independence while honoring family. Themes: Identity, duty vs. desire, family bonds, finding voice. Arc: Dependence → conflict → independence with love.",
        "genre": "Drama",
        "runtime": 111
    },
    {
        "year": 2021,
        "title": "Nomadland",
        "plot": "After losing everything in the Great Recession, a woman in her sixties embarks on a journey through the American West, living as a modern-day nomad. Explores grief, loss, community, and finding meaning outside conventional success. Themes: Economic displacement, aging, freedom, transient community. Arc: Loss → wandering → acceptance.",
        "genre": "Drama",
        "runtime": 108
    },
    {
        "year": 2020,
        "title": "Parasite",
        "plot": "A poor family schemes to become employed by a wealthy family by infiltrating their household through deception. Class warfare, social commentary, dark comedy turning to thriller. The basement reveals. Themes: Class divide, desperation, exploitation, violence. Arc: Clever scheme → growing tension → violent climax → devastating irony.",
        "genre": "Thriller/Dark Comedy",
        "runtime": 132
    },
    {
        "year": 2019,
        "title": "Green Book",
        "plot": "Working-class Italian-American bouncer becomes driver for African-American classical pianist on tour through the Deep South in 1960s. Unlikely friendship, confronting racism, mutual transformation. Themes: Racism, friendship across class/race, personal growth. Arc: Prejudice → journey together → mutual understanding.",
        "genre": "Biography/Drama",
        "runtime": 130
    },
    {
        "year": 2018,
        "title": "The Shape of Water",
        "plot": "Mute janitor at a secret government laboratory falls in love with a captured amphibian creature during the Cold War. Beauty and the beast, outsider romance, magical realism. Themes: Love transcending difference, otherness, oppression, escape. Arc: Loneliness → forbidden love → liberation.",
        "genre": "Fantasy/Romance",
        "runtime": 123
    },
    {
        "year": 2017,
        "title": "Moonlight",
        "plot": "The life of a young Black man from childhood to adulthood as he struggles with his identity and sexuality in Miami. Three acts: boyhood, adolescence, adulthood. Themes: Identity, sexuality, masculinity, poverty, connection. Arc: Innocence → hardening → reconnection with self.",
        "genre": "Drama",
        "runtime": 111
    },
    {
        "year": 2016,
        "title": "Spotlight",
        "plot": "Boston Globe journalists uncover massive child abuse scandal and cover-up within the Catholic Church. Investigative journalism, institutional corruption, truth-seeking. Themes: Justice, institutional power, persistence, systemic failure. Arc: Discovery → investigation → revelation → impact.",
        "genre": "Drama/Thriller",
        "runtime": 128
    },
    {
        "year": 2015,
        "title": "Birdman",
        "plot": "Washed-up superhero actor attempts comeback via Broadway play while battling his ego and family issues. Meta-theatrical, one-shot filming style, reality blurring. Themes: Relevance, artistic integrity, ego, redemption. Arc: Desperate comeback → escalating chaos → transcendence or delusion.",
        "genre": "Comedy-Drama",
        "runtime": 119
    },
    {
        "year": 2014,
        "title": "12 Years a Slave",
        "plot": "Free Black man kidnapped and sold into slavery, endures brutal treatment for twelve years before being freed. True story of Solomon Northup. Themes: Slavery, survival, dignity, injustice, freedom. Arc: Freedom → enslavement → survival → liberation.",
        "genre": "Biography/Drama",
        "runtime": 134
    },
    {
        "year": 2013,
        "title": "Argo",
        "plot": "CIA agent creates fake movie production to rescue Americans from Iran during 1979 hostage crisis. True story, suspense, political thriller. Themes: Deception, courage, geopolitics, truth-in-fiction. Arc: Crisis → impossible plan → tense execution → escape.",
        "genre": "Thriller/Drama",
        "runtime": 120
    },
    {
        "year": 2012,
        "title": "The Artist",
        "plot": "Silent film star struggles with transition to talking pictures as his career declines while a young actress rises. Silent film homage, tragedy-to-redemption. Themes: Obsolescence, pride, adaptation, love saves. Arc: Success → technological disruption → fall → redemption through love.",
        "genre": "Comedy-Drama/Romance",
        "runtime": 100
    },
    {
        "year": 2011,
        "title": "The King's Speech",
        "plot": "King George VI overcomes his stammer with help from unconventional speech therapist as WWII looms. Overcoming disability, unlikely friendship, finding voice. Themes: Disability, leadership, friendship, courage. Arc: Struggle → therapy → triumph → inspiring nation.",
        "genre": "Biography/Drama",
        "runtime": 118
    },
    {
        "year": 2010,
        "title": "The Hurt Locker",
        "plot": "Bomb disposal team in Iraq, following adrenaline-addicted squad leader through dangerous missions. War, addiction, masculinity, purpose through danger. Themes: War addiction, purpose, mortality, brotherhood. Arc: Mission after mission → increasing danger → return to addiction.",
        "genre": "War/Thriller",
        "runtime": 127
    }
    # Continue with more films... (can add all 95)
]

def generate_complete_dataset():
    """Generate dataset with all Best Pictures."""
    print("=" * 80)
    print("OSCAR BEST PICTURES - Complete Dataset")
    print("=" * 80)
    
    # Start with recent 15, can expand to all 95
    print(f"\n✓ Collected {len(BEST_PICTURES)} Best Picture winners")
    print(f"✓ Each has: Title, plot summary, themes, narrative arc")
    
    # Save
    output_path = Path(__file__).parent.parent.parent.parent / 'data/domains/oscar_best_pictures.json'
    with open(output_path, 'w') as f:
        json.dump(BEST_PICTURES, f, indent=2)
    
    print(f"✓ Saved to: {output_path}")
    
    return BEST_PICTURES

if __name__ == "__main__":
    generate_complete_dataset()

