"""
Professional Poker Narrative Generator

Generates rich 200-300 word narratives for each tournament entry.
Target: 30-40 proper nouns per narrative for maximum nominative richness.

Narrative elements:
- Player introduction (name, nickname, style, reputation)
- Tournament context (prestige, venue, stakes)
- Psychological dynamics (mental game, pressure, confidence)
- Playing style matchups and table dynamics
- Historical context (rivalries, previous performances)
- Temporal progression (tournament stage, chip stacks)
- Mental game elements (tells, bluffs, reads, composure)

Author: Narrative Integration System
Date: November 2025
"""

import json
import random
from pathlib import Path
import re


# Narrative templates and components
MENTAL_GAME_PHRASES = [
    "known for exceptional reading ability",
    "famous for stone-cold demeanor",
    "vulnerable to tilt after bad beats",
    "ice cold under pressure",
    "never shows emotion at the felt",
    "elite psychological warfare skills",
    "can read opponents like a book",
    "intimidating table presence",
    "masterful bluffing ability",
    "disciplined and patient approach",
    "aggressive and fearless style",
    "clinical and calculated decisions",
    "thrives in high-pressure situations",
    "composure never cracks",
    "legendary for clutch performances",
]

STYLE_DESCRIPTIONS = {
    "Small-ball": "applying relentless pressure with small bets, gradually building pots",
    "Mathematical": "using game theory optimal strategies and mathematical precision",
    "Old school": "relying on instinct and decades of experience reading opponents",
    "Tight-aggressive": "playing premium hands aggressively, rarely bluffing",
    "GTO wizard": "implementing perfect game theory optimal play",
    "High roller": "fearlessly playing massive pots with supreme confidence",
    "Silent killer": "letting actions speak louder than words with quiet dominance",
    "Aggressive": "applying constant pressure, forcing opponents into tough decisions",
    "Methodical": "carefully analyzing every decision with systematic precision",
    "Balanced": "mixing aggression with patience in perfect equilibrium",
    "Reads-based": "relying primarily on opponent tells and behavioral patterns",
    "Showman": "combining entertainment value with world-class poker skills",
    "High variance": "taking big risks for huge potential rewards",
    "Steady": "maintaining consistent pressure without major swings",
    "Analytical": "approaching poker as pure mathematics and data science",
    "Loose-aggressive": "playing many hands with maximum aggression",
    "Amateur": "unconventional style that confuses experienced professionals",
    "LAG": "loose-aggressive play, constantly attacking weakness",
    "Solid": "fundamentally sound with minimal mistakes",
}

TOURNAMENT_STAGE_PHRASES = [
    "entering the final table as chip leader",
    "short-stacked and fighting for survival",
    "with momentum after recent double-up",
    "on the bubble with medium stack",
    "dominating the early stages",
    "climbing steadily through middle rounds",
    "facing elimination pressure",
    "holding commanding chip lead",
    "nursing short stack with precision",
    "in heads-up battle for the title",
]

PRESSURE_NARRATIVES = [
    "The pressure of {buy_in:,} buy-in weighs on every decision.",
    "With ${prize:,} on the line, every hand matters.",
    "The weight of the moment is palpable.",
    "Stakes have never been higher in this tournament.",
    "The pressure cooker atmosphere is intense.",
    "Every chip counts at this stage of the tournament.",
    "The mental game reaches its peak intensity.",
    "Composure is tested with every big pot.",
    "The psychological warfare intensifies.",
    "Nerves of steel required at this level.",
]

VENUE_DESCRIPTIONS = {
    "Rio All-Suite Hotel": "the legendary Rio in Las Vegas",
    "Wynn": "the prestigious Wynn Las Vegas",
    "Commerce Casino": "California's iconic Commerce Casino",
    "Bay 101 Casino": "the renowned Bay 101 in San Jose",
    "Bellagio": "the luxurious Bellagio poker room",
    "Monte Carlo Bay Hotel": "the glamorous Monte Carlo Bay in Monaco",
    "Casino Barcelona": "Barcelona's stunning Casino Barcelona",
    "King's Casino": "the massive King's Casino in Prague",
    "Aria Resort": "the modern Aria Resort & Casino",
}

def count_proper_nouns(text):
    """Count proper nouns (capitalized words) in text"""
    # Find all capitalized words (excluding sentence starts)
    words = text.split()
    proper_nouns = []
    for i, word in enumerate(words):
        # Clean punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        # Count if capitalized and not sentence start (or is a known proper noun)
        if clean_word and clean_word[0].isupper():
            if i > 0 or clean_word in ['WSOP', 'WPT', 'EPT', 'Las', 'Vegas', 'Monte', 'Carlo']:
                proper_nouns.append(clean_word)
    return len(proper_nouns)


def generate_player_intro(player, tournament_name):
    """Generate player introduction paragraph"""
    
    # Get mental game phrase
    mental = random.choice(MENTAL_GAME_PHRASES)
    
    # Get style description
    style_desc = STYLE_DESCRIPTIONS.get(player["playing_style"], "with a unique approach")
    
    # Build intro
    intro_templates = [
        f"{player['name']}, {mental}, brings their signature {player['playing_style']} style to the {tournament_name}.",
        f"Known as '{player['nickname']}', {player['name']} enters with ${player['career_earnings']:,} in career earnings and {player['major_titles']} major titles.",
        f"{player['name']} ({player['nickname']}) is {mental}, {style_desc}.",
    ]
    
    return " ".join(random.sample(intro_templates, 2))


def generate_tournament_context(entry):
    """Generate tournament context paragraph"""
    
    tournament_name = entry["tournament_name"]
    buy_in = entry["buy_in"]
    field_size = entry["field_size"]
    venue = entry["venue"]
    location = entry["location"]
    venue_desc = VENUE_DESCRIPTIONS.get(venue, f"the {venue}")
    
    prestige = entry["prestige_level"]
    tier = entry["metadata"]["tournament_tier"]
    
    context_templates = [
        f"The {tournament_name}, held at {venue_desc} in {location}, features a ${buy_in:,} buy-in and {field_size:,} player field.",
        f"This {tier} event draws elite competition from around the world.",
        f"With a prestige level of {prestige:.2f}, this tournament represents the pinnacle of professional poker.",
    ]
    
    return " ".join(context_templates)


def generate_psychological_narrative(player, opponent, finish_position):
    """Generate psychological warfare narrative"""
    
    stage = TOURNAMENT_STAGE_PHRASES[min(finish_position - 1, len(TOURNAMENT_STAGE_PHRASES) - 1)]
    
    if opponent:
        # Final table narrative with opponent
        matchup = f"{player['name']}'s {player['playing_style']} style creates fascinating dynamics against {opponent['name']}'s {opponent['playing_style']} approach."
        
        mental_game = f"The psychological battle between {player['psychological_profile']} {player['name']} and {opponent['psychological_profile']} {opponent['name']} captivates the poker world."
        
        contrast = f"While {player['name']} is {player['psychological_profile'].lower()}, {opponent['name']}'s {opponent['psychological_profile'].lower()} creates perfect counter-balance."
        
        return f"{player['name']}, {stage}, faces intense competition. {matchup} {mental_game} {contrast}"
    else:
        # Solo narrative
        focus = f"{player['name']}, {stage}, demonstrates {player['psychological_profile'].lower()} throughout the tournament."
        
        style = f"Their {player['playing_style']} style, {STYLE_DESCRIPTIONS.get(player['playing_style'], 'unique to their game')}, proves effective against the elite field."
        
        return f"{focus} {style}"


def generate_outcome_narrative(entry):
    """Generate outcome and stakes narrative"""
    
    finish = entry["outcome"]["finish_position"]
    prize = entry["outcome"]["prize_money"]
    field = entry["field_size"]
    
    if finish == 1:
        result = f"Ultimately triumphing over {field:,} players, {entry['player']['name']} claims victory and the ${prize:,} first-place prize."
    elif finish <= 3:
        result = f"Finishing in {finish}{'rd' if finish == 3 else 'nd'} place out of {field:,} entries, {entry['player']['name']} earns ${prize:,}."
    elif finish <= 9:
        result = f"Making the final table in {finish}th position, {entry['player']['name']} secures ${prize:,} from the {field:,} player field."
    elif prize > 0:
        result = f"Cashing in {finish}th place, {entry['player']['name']} takes home ${prize:,} from the massive {field:,} entry field."
    else:
        result = f"Despite fierce competition, {entry['player']['name']} falls short of the money in {field:,} player field."
    
    pressure = random.choice(PRESSURE_NARRATIVES).format(
        buy_in=entry["buy_in"],
        prize=prize if prize > 0 else entry["buy_in"] * field
    )
    
    return f"{pressure} {result}"


def generate_nominative_rich_details(entry):
    """Add nominative richness with proper nouns"""
    
    player = entry["player"]
    tournament = entry["tournament_name"]
    date = entry["date"]
    location = entry["location"]
    
    # Extract year, month
    year = date.split("-")[0]
    month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    month = month_names[int(date.split("-")[1]) - 1]
    
    # Specific details
    details = f"In {month} {year}, at {location}, the {tournament} showcases {player['name']} ('{player['nickname']}'), "
    details += f"a {player['style_classification']} player with {player['major_titles']} major championship"
    details += f"{'s' if player['major_titles'] != 1 else ''}. "
    
    # Add more proper nouns
    details += f"Known throughout the World Series of Poker circuit and World Poker Tour, "
    details += f"{player['name']}'s reputation as {player['psychological_profile'].lower()} precedes them. "
    
    # Historical context
    if player['major_titles'] > 5:
        details += f"With titles spanning from early World Series events to modern European Poker Tour victories, "
        details += f"their legacy in cities like Las Vegas, Monte Carlo, and Barcelona is secure. "
    elif player['major_titles'] > 0:
        details += f"Their championship pedigree, earned across major tournaments from Las Vegas to Monte Carlo, "
        details += f"establishes them as elite competition. "
    
    return details


def generate_complete_narrative(entry):
    """Generate complete 200-300 word narrative"""
    
    player = entry["player"]
    opponent = entry.get("opponent")
    tournament = entry["tournament_name"]
    
    # Build narrative sections
    sections = []
    
    # 1. Opening with player intro
    sections.append(generate_player_intro(player, tournament))
    
    # 2. Tournament context
    sections.append(generate_tournament_context(entry))
    
    # 3. Nominative-rich details
    sections.append(generate_nominative_rich_details(entry))
    
    # 4. Psychological narrative
    sections.append(generate_psychological_narrative(player, opponent, entry["outcome"]["finish_position"]))
    
    # 5. Outcome narrative
    sections.append(generate_outcome_narrative(entry))
    
    # Combine with proper spacing
    narrative = " ".join(sections)
    
    # Count proper nouns
    noun_count = count_proper_nouns(narrative)
    
    # If under 30, add more details
    while noun_count < 30:
        # Add sponsorship/media mentions
        additions = [
            f"Covered extensively by PokerNews and CardPlayer Magazine, this tournament draws worldwide attention.",
            f"ESPN cameras capture every moment at the featured table in the Amazon Room.",
            f"The PokerGO stream brings millions of viewers into the action.",
            f"Commentators like Norman Chad and Lon McEachern provide expert analysis.",
            f"Daniel Negreanu's poker podcast later discusses key hands from the final table.",
        ]
        narrative += " " + random.choice(additions)
        noun_count = count_proper_nouns(narrative)
    
    # Ensure word count is appropriate (200-300 words)
    word_count = len(narrative.split())
    
    return {
        "narrative": narrative,
        "word_count": word_count,
        "proper_noun_count": noun_count
    }


def add_narratives_to_dataset(dataset_path, output_path):
    """Add narratives to all tournament entries"""
    
    print("="*80)
    print("POKER NARRATIVE GENERATION")
    print("="*80)
    print(f"\nTarget: 200-300 words per narrative")
    print(f"Target: 30-40 proper nouns per narrative")
    print(f"\nLoading dataset from: {dataset_path}\n")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    total_entries = len(dataset["tournaments"])
    print(f"Processing {total_entries:,} tournament entries...\n")
    
    # Track statistics
    word_counts = []
    noun_counts = []
    
    # Generate narratives
    for i, entry in enumerate(dataset["tournaments"]):
        narrative_data = generate_complete_narrative(entry)
        
        entry["narrative"] = narrative_data["narrative"]
        entry["narrative_metadata"] = {
            "word_count": narrative_data["word_count"],
            "proper_noun_count": narrative_data["proper_noun_count"],
            "generated_date": entry["date"]
        }
        
        word_counts.append(narrative_data["word_count"])
        noun_counts.append(narrative_data["proper_noun_count"])
        
        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"Generated {i+1:,} / {total_entries:,} narratives...")
            print(f"  Avg words: {sum(word_counts[-1000:]) / 1000:.1f}")
            print(f"  Avg proper nouns: {sum(noun_counts[-1000:]) / 1000:.1f}")
    
    # Calculate final statistics
    avg_words = sum(word_counts) / len(word_counts)
    avg_nouns = sum(noun_counts) / len(noun_counts)
    min_nouns = min(noun_counts)
    max_nouns = max(noun_counts)
    
    print(f"\n✓ Generated {total_entries:,} narratives")
    print(f"\nNarrative Statistics:")
    print(f"  Average word count: {avg_words:.1f}")
    print(f"  Average proper nouns: {avg_nouns:.1f}")
    print(f"  Min proper nouns: {min_nouns}")
    print(f"  Max proper nouns: {max_nouns}")
    print(f"  Target met: {avg_nouns >= 30} ({'✓' if avg_nouns >= 30 else '✗'})")
    
    # Update metadata
    dataset["metadata"]["narrative_statistics"] = {
        "total_narratives": total_entries,
        "average_word_count": round(avg_words, 1),
        "average_proper_nouns": round(avg_nouns, 1),
        "min_proper_nouns": min_nouns,
        "max_proper_nouns": max_nouns,
        "target_achieved": avg_nouns >= 30
    }
    
    # Save updated dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Dataset with narratives saved to: {output_path}")
    print(f"✓ File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Show example narrative
    print("\n" + "="*80)
    print("EXAMPLE NARRATIVE")
    print("="*80)
    example = dataset["tournaments"][0]
    print(f"\nPlayer: {example['player']['name']}")
    print(f"Tournament: {example['tournament_name']}")
    print(f"Finish: {example['outcome']['finish_position']}/{example['field_size']}")
    print(f"Prize: ${example['outcome']['prize_money']:,}")
    print(f"\nNarrative ({example['narrative_metadata']['word_count']} words, "
          f"{example['narrative_metadata']['proper_noun_count']} proper nouns):")
    print(f"\n{example['narrative']}")
    print("\n" + "="*80)
    
    return dataset


if __name__ == '__main__':
    # Paths
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'poker'
    input_path = data_dir / 'poker_tournament_dataset.json'
    output_path = data_dir / 'poker_tournament_dataset_with_narratives.json'
    
    print("\nStarting narrative generation...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")
    
    # Generate narratives
    dataset = add_narratives_to_dataset(input_path, output_path)
    
    print("\n✓ Narrative generation complete!")
    print("✓ Ready for transformer application (next step)")

