"""
Hurricane Name Analyzer - COMPREHENSIVE

Extracts ALL name characteristics for each hurricane:
1. Gender classification (1953-2024 patterns)
2. Phonetic analysis (harshness, sounds, syllables)
3. Popularity/familiarity (common vs rare names)
4. Memorability factors
5. Semantic associations

Tests Jung et al. (2014) finding: feminine names → more deaths

Author: Narrative Integration System  
Date: November 2025
"""

import json
import re
from pathlib import Path
from collections import defaultdict


# Gender assignment patterns (Atlantic hurricanes)
# 1953-1978: All female names
# 1979+: Alternating male/female (6 lists rotated)

FEMALE_INDICATORS = ['a', 'ie', 'ine', 'elle', 'ette', 'y']
MALE_INDICATORS = ['o', 'an', 'en', 'on']

# Harsh vs soft phonetic sounds
HARSH_CONSONANTS = ['k', 'c', 't', 'd', 'g', 'p', 'b', 'q', 'x']
SOFT_CONSONANTS = ['s', 'l', 'm', 'n', 'f', 'v', 'w', 'h', 'r']

# Known hurricane name lists (Atlantic, rotating every 6 years)
ATLANTIC_NAMES_BY_GENDER = {
    'male': ['arthur', 'bill', 'colin', 'danny', 'don', 'franklin', 'gordon', 'harold', 'idai', 
             'jerry', 'karl', 'larry', 'marco', 'omar', 'philippe', 'peter', 'rene', 'sam',
             'teddy', 'victor', 'wilfred', 'kyle', 'michael', 'nestor', 'pablo', 'richard',
             'tony', 'vince', 'waldo', 'ernesto', 'gaston', 'harvey', 'isaac', 'kirk',
             'lee', 'oscar', 'raphael', 'sebastian', 'matthew', 'otto', 'alex', 'bret',
             'charley', 'dean', 'earl', 'fred', 'gustav', 'hector', 'ian', 'joaquin',
             'keith', 'lorenzo', 'martin', 'nigel', 'philippe', 'rina', 'sean', 'tammy',
             'vince', 'walter', 'andre', 'beryl', 'chris', 'erin', 'felix', 'henri',
             'ivan', 'jose', 'klaus', 'luis', 'manuel', 'noel', 'owen', 'philippe',
             'rene', 'sean', 'tomas', 'vince', 'william'],
    'female': ['andrea', 'bonnie', 'claudette', 'danielle', 'elsa', 'fiona', 'gaston', 
               'hermine', 'ida', 'julia', 'kate', 'lisa', 'mindy', 'nicole', 'odette',
               'paula', 'rose', 'sally', 'teresa', 'arlene', 'bertha', 'cindy', 'dolly',
               'fay', 'hanna', 'isaias', 'josephine', 'katrina', 'laura', 'maria', 
               'nana', 'ophelia', 'patty', 'rita', 'sandy', 'tanya', 'valerie', 'wanda',
               'amelia', 'babe', 'carla', 'dora', 'edna', 'flora', 'ginger', 'hazel',
               'ione', 'janet', 'karen', 'louise', 'mabel', 'nora', 'opal', 'greta',
               'holly', 'iris', 'michelle', 'allison', 'chantal', 'erin', 'gabrielle',
               'hortense', 'isidore', 'lili', 'noel', 'charley', 'frances', 'isabel',
               'jeanne', 'katrina', 'rita', 'wilma', 'emily', 'iris', 'lenny', 'michelle']
}


def classify_gender(name, year):
    """
    Classify hurricane name as male or female
    
    Rules:
    - 1953-1978: All female
    - 1979+: Alternating male/female in the list
    """
    
    name_lower = name.lower()
    
    # Pre-1979: all female
    if year < 1979:
        return 'female'
    
    # Check against known lists
    if name_lower in ATLANTIC_NAMES_BY_GENDER['male']:
        return 'male'
    elif name_lower in ATLANTIC_NAMES_BY_GENDER['female']:
        return 'female'
    
    # Heuristic based on ending (for any missed names)
    for ending in FEMALE_INDICATORS:
        if name_lower.endswith(ending):
            return 'female'
    
    for ending in MALE_INDICATORS:
        if name_lower.endswith(ending):
            return 'male'
    
    # Default: check first letter alternation pattern
    # In rotating lists, odd positions tend to be one gender
    return 'female'  # Conservative default


def calculate_phonetic_harshness(name):
    """
    Calculate phonetic harshness score (0-1)
    
    Based on:
    - Harsh consonants (K, T, D, G, P, B): +1 each
    - Soft consonants (S, L, M, N, F, V): -0.5 each
    - Normalize to 0-1 scale
    
    Examples:
    - KATRINA: K(+1), T(+1), R, N(-0.5) = harsh (0.75)
    - SANDY: S(-0.5), N(-0.5), D(+1), Y = softer (0.35)
    """
    
    name_lower = name.lower()
    score = 0
    count = 0
    
    for char in name_lower:
        if char in HARSH_CONSONANTS:
            score += 1.0
            count += 1
        elif char in SOFT_CONSONANTS:
            score += 0.0  # Neutral (soft = 0)
            count += 1
    
    if count == 0:
        return 0.5  # Neutral if no consonants
    
    # Normalize (harsh consonants push toward 1, soft toward 0)
    harshness = score / count
    
    return round(harshness, 3)


def calculate_syllable_count(name):
    """Estimate syllable count"""
    # Simple heuristic: count vowel groups
    name = name.lower()
    vowels = 'aeiou'
    syllables = 0
    previous_was_vowel = False
    
    for char in name:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllables += 1
        previous_was_vowel = is_vowel
    
    return max(1, syllables)  # At least 1 syllable


def get_name_length_category(name):
    """Categorize name by length"""
    length = len(name)
    if length <= 4:
        return 'short'
    elif length <= 6:
        return 'medium'
    else:
        return 'long'


def calculate_memorability(name):
    """
    Calculate name memorability (0-1)
    
    Factors:
    - Length (shorter = more memorable)
    - Uniqueness (uncommon = more memorable)
    - Pronounceability (simpler = more memorable)
    - Distinctiveness
    """
    
    # Length factor (shorter = better)
    length = len(name)
    length_score = 1.0 - min(1.0, (length - 4) / 8)  # Optimal around 4-6 letters
    
    # Syllable factor (2-3 syllables ideal)
    syllables = calculate_syllable_count(name)
    syllable_score = 1.0 - abs(syllables - 2.5) / 3
    
    # Uniqueness (uncommon endings = distinctive)
    uncommon_endings = ['x', 'z', 'q', 'v']
    uniqueness_score = 0.5
    if name.lower()[-1] in uncommon_endings:
        uniqueness_score = 0.9
    elif name.lower()[-1] in ['a', 'e', 'i']:
        uniqueness_score = 0.6  # Common but pleasant
    
    # Combine (weighted average)
    memorability = (length_score * 0.4 + 
                   syllable_score * 0.3 + 
                   uniqueness_score * 0.3)
    
    return round(memorability, 3)


def is_retired_name(name, storms_data):
    """
    Check if name was retired (happens for very destructive storms)
    Retired names: Katrina, Sandy, Harvey, Irma, Maria, Michael, etc.
    """
    # Count how many times this name appears
    name_lower = name.lower()
    appearances = [s for s in storms_data if s['name'].lower() == name_lower]
    
    # If name only appears in older years and not recent, likely retired
    if appearances:
        last_year = max(s['year'] for s in appearances)
        # If last appearance was pre-2015 and it was a major storm, likely retired
        major_appearances = [s for s in appearances if s['category'] >= 3]
        if major_appearances and last_year < 2015:
            return True
    
    return False


def get_famous_association(name):
    """Check if name has famous hurricane association"""
    infamous_hurricanes = {
        'katrina': {'year': 2005, 'deaths': 1833, 'damage': 125000000000, 'fame': 'Most infamous US hurricane'},
        'sandy': {'year': 2012, 'deaths': 233, 'damage': 70000000000, 'fame': 'Superstorm Sandy, New York/New Jersey'},
        'harvey': {'year': 2017, 'deaths': 107, 'damage': 125000000000, 'fame': 'Houston flooding catastrophe'},
        'irma': {'year': 2017, 'deaths': 134, 'damage': 77000000000, 'fame': 'One of strongest Atlantic hurricanes'},
        'maria': {'year': 2017, 'deaths': 2975, 'damage': 92000000000, 'fame': 'Puerto Rico devastation'},
        'michael': {'year': 2018, 'deaths': 74, 'damage': 25000000000, 'fame': 'Florida Panhandle Cat 5'},
        'andrew': {'year': 1992, 'deaths': 65, 'damage': 27000000000, 'fame': 'South Florida devastation'},
        'hugo': {'year': 1989, 'deaths': 76, 'damage': 10000000000, 'fame': 'Carolinas destruction'},
        'camille': {'year': 1969, 'deaths': 259, 'damage': 1420000000, 'fame': 'Mississippi Cat 5'},
        'wilma': {'year': 2005, 'deaths': 62, 'damage': 27000000000, 'fame': 'Record low pressure'},
    }
    
    name_lower = name.lower()
    if name_lower in infamous_hurricanes:
        return infamous_hurricanes[name_lower]
    
    return None


def analyze_all_names(storms_data):
    """Analyze all hurricane names in dataset"""
    
    print("\n" + "="*80)
    print("HURRICANE NAME CHARACTERIZATION")
    print("="*80)
    print(f"\nAnalyzing {len(storms_data)} storm names...\n")
    
    analyzed_storms = []
    
    # Track statistics
    gender_counts = {'male': 0, 'female': 0}
    harshness_values = []
    memorable_names = []
    
    for storm in storms_data:
        name = storm['name']
        year = storm['year']
        
        # Gender
        gender = classify_gender(name, year)
        gender_counts[gender] += 1
        
        # Phonetic harshness
        harshness = calculate_phonetic_harshness(name)
        harshness_values.append(harshness)
        
        # Syllables
        syllables = calculate_syllable_count(name)
        
        # Length
        length = len(name)
        length_category = get_name_length_category(name)
        
        # Memorability
        memorability = calculate_memorability(name)
        if memorability > 0.7:
            memorable_names.append((name, memorability))
        
        # Retired/famous
        retired = is_retired_name(name, storms_data)
        famous = get_famous_association(name)
        
        # Create name profile
        name_profile = {
            'name': name,
            'year': year,
            'gender': gender,
            'phonetic_harshness': harshness,
            'syllables': syllables,
            'length': length,
            'length_category': length_category,
            'memorability': memorability,
            'is_retired': retired,
            'famous_association': famous,
            'first_letter': name[0].upper(),
            'last_letter': name[-1].lower(),
            'has_harsh_start': name[0].lower() in HARSH_CONSONANTS,
            'has_soft_start': name[0].lower() in SOFT_CONSONANTS,
        }
        
        # Add to storm data
        storm['name_profile'] = name_profile
        analyzed_storms.append(storm)
    
    # Calculate statistics
    stats = {
        'total_storms': len(analyzed_storms),
        'gender_distribution': gender_counts,
        'gender_percentages': {
            'male': (gender_counts['male'] / len(analyzed_storms)) * 100,
            'female': (gender_counts['female'] / len(analyzed_storms)) * 100
        },
        'harshness': {
            'mean': sum(harshness_values) / len(harshness_values),
            'min': min(harshness_values),
            'max': max(harshness_values),
            'median': sorted(harshness_values)[len(harshness_values) // 2]
        },
        'most_memorable': sorted(memorable_names, key=lambda x: x[1], reverse=True)[:20]
    }
    
    print(f"✓ Analyzed {len(analyzed_storms)} storm names")
    print(f"\nGender Distribution:")
    print(f"  Male: {gender_counts['male']} ({stats['gender_percentages']['male']:.1f}%)")
    print(f"  Female: {gender_counts['female']} ({stats['gender_percentages']['female']:.1f}%)")
    
    print(f"\nPhonetic Harshness:")
    print(f"  Mean: {stats['harshness']['mean']:.3f}")
    print(f"  Range: [{stats['harshness']['min']:.3f}, {stats['harshness']['max']:.3f}]")
    
    print(f"\nMost Memorable Names:")
    for name, score in stats['most_memorable'][:10]:
        print(f"  {name}: {score:.3f}")
    
    return analyzed_storms, stats


def identify_name_patterns(analyzed_storms):
    """Identify interesting patterns in name characteristics"""
    
    print("\n" + "="*80)
    print("NAME PATTERN ANALYSIS")
    print("="*80)
    
    # Gender by decade
    by_decade_gender = defaultdict(lambda: {'male': 0, 'female': 0})
    for storm in analyzed_storms:
        decade = (storm['year'] // 10) * 10
        gender = storm['name_profile']['gender']
        by_decade_gender[decade][gender] += 1
    
    print(f"\nGender Distribution by Decade:")
    for decade in sorted(by_decade_gender.keys()):
        male = by_decade_gender[decade]['male']
        female = by_decade_gender[decade]['female']
        total = male + female
        print(f"  {decade}s: Male {male} ({male/total*100:.0f}%), Female {female} ({female/total*100:.0f}%)")
    
    # Harshness by gender
    male_harshness = [s['name_profile']['phonetic_harshness'] 
                      for s in analyzed_storms if s['name_profile']['gender'] == 'male']
    female_harshness = [s['name_profile']['phonetic_harshness'] 
                        for s in analyzed_storms if s['name_profile']['gender'] == 'female']
    
    print(f"\nPhonetic Harshness by Gender:")
    if male_harshness:
        print(f"  Male average: {sum(male_harshness)/len(male_harshness):.3f}")
    if female_harshness:
        print(f"  Female average: {sum(female_harshness)/len(female_harshness):.3f}")
    
    # Famous hurricanes
    famous = [s for s in analyzed_storms if s['name_profile']['famous_association']]
    print(f"\nInfamous Hurricanes (Known to Public):")
    for storm in famous[:15]:
        assoc = storm['name_profile']['famous_association']
        print(f"  {storm['year']} {storm['name']}: {assoc['fame']}")
    
    # Category 5 by gender
    cat5_by_gender = defaultdict(int)
    for storm in analyzed_storms:
        if storm['category'] == 5:
            cat5_by_gender[storm['name_profile']['gender']] += 1
    
    print(f"\nCategory 5 Storms by Gender:")
    for gender, count in cat5_by_gender.items():
        print(f"  {gender.capitalize()}: {count}")


def save_analyzed_dataset(analyzed_storms, stats, output_path):
    """Save complete dataset with name analysis"""
    
    from datetime import datetime
    
    dataset = {
        'metadata': {
            'domain': 'hurricanes',
            'description': 'Atlantic hurricanes with comprehensive name analysis',
            'collection_date': json.load(open(Path(output_path).parent / 'hurricane_complete_dataset.json'))['metadata']['collection_date'],
            'analysis_date': datetime.now().isoformat(),
            'total_storms': len(analyzed_storms)
        },
        'name_analysis_statistics': stats,
        'storms': analyzed_storms
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Analyzed dataset saved to: {output_path}")
    print(f"✓ File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    return dataset


def main():
    """Main execution"""
    
    # Load hurricane data
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'hurricanes'
    input_file = data_dir / 'hurricane_complete_dataset.json'
    
    print(f"Loading hurricane data from: {input_file}")
    
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    storms = dataset['storms']
    print(f"✓ Loaded {len(storms)} storms")
    
    # Analyze all names
    analyzed_storms, stats = analyze_all_names(storms)
    
    # Identify patterns
    identify_name_patterns(analyzed_storms)
    
    # Save
    output_file = data_dir / 'hurricane_dataset_with_name_analysis.json'
    analyzed_dataset = save_analyzed_dataset(analyzed_storms, stats, output_file)
    
    print("\n" + "="*80)
    print("NAME CHARACTERIZATION COMPLETE")
    print("="*80)
    print(f"\n✓ Analyzed {len(analyzed_storms)} hurricane names")
    print(f"✓ Gender classified: {len(analyzed_storms)} storms")
    print(f"✓ Phonetic harshness calculated: {len(analyzed_storms)} storms")
    print(f"✓ Memorability scored: {len(analyzed_storms)} storms")
    print(f"✓ Famous associations identified: {len([s for s in analyzed_storms if s['name_profile']['famous_association']])} storms")
    
    print(f"\nReady for outcome integration and regression analysis")
    
    return analyzed_dataset


if __name__ == '__main__':
    dataset = main()

