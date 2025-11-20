"""
Poker-Specific Enriched Patterns for θ and λ Extraction

θ (Theta) - Awareness/Resistance Patterns:
Mental game awareness, psychological sophistication, self-awareness

λ (Lambda) - Fundamental Constraints Patterns:
Skill requirements, expertise barriers, training needed

Author: Narrative Integration System
Date: November 2025
"""

# θ (THETA) - Awareness/Resistance Patterns for Poker
# These indicate psychological awareness, mental game sophistication
POKER_THETA_PATTERNS = [
    # Core mental game awareness (20 patterns)
    "mental game",
    "psychology",
    "psychological warfare",
    "psychological battle",
    "psychological edge",
    "mental edge",
    "mental strength",
    "mental toughness",
    "mind games",
    "psychological dynamics",
    "mental warfare",
    "psychological pressure",
    "mental pressure",
    "psychological advantage",
    "mental advantage",
    "psychological skill",
    "psychological awareness",
    "mental awareness",
    "psychological sophistication",
    "mental sophistication",
    
    # Tells and reading (15 patterns)
    "tell",
    "tells",
    "physical tell",
    "behavioral tell",
    "timing tell",
    "read",
    "reading",
    "reading opponents",
    "read opponents",
    "opponent read",
    "table read",
    "reading ability",
    "reading skills",
    "read the table",
    "read the situation",
    
    # Bluffing awareness (12 patterns)
    "bluff",
    "bluffing",
    "semi-bluff",
    "pure bluff",
    "bluff detection",
    "detect bluff",
    "bluffing strategy",
    "bluff frequency",
    "bluff catcher",
    "hero call",
    "sick bluff",
    "audacious bluff",
    
    # Table image (10 patterns)
    "table image",
    "image",
    "reputation",
    "table presence",
    "intimidating",
    "intimidation",
    "table dynamics",
    "table awareness",
    "positional awareness",
    "stack awareness",
    
    # Composure and tilt (18 patterns)
    "composure",
    "tilt",
    "tilting",
    "tilted",
    "on tilt",
    "emotional control",
    "emotional discipline",
    "emotional game",
    "ice cold",
    "stone cold",
    "stone-cold",
    "poker face",
    "never shows emotion",
    "no emotion",
    "calm demeanor",
    "cool under pressure",
    "ice in veins",
    "nerves of steel",
    
    # Pressure handling (15 patterns)
    "pressure",
    "high pressure",
    "under pressure",
    "pressure situation",
    "pressure cooker",
    "handles pressure",
    "thrives under pressure",
    "clutch",
    "clutch performance",
    "clutch gene",
    "clutch player",
    "big moment",
    "big spot",
    "crucial decision",
    "pivotal moment",
    
    # Patience and discipline (12 patterns)
    "patience",
    "patient",
    "disciplined",
    "discipline",
    "wait for spots",
    "waiting for",
    "selective",
    "careful",
    "calculated",
    "methodical",
    "systematic",
    "deliberate",
    
    # Confidence and boldness (10 patterns)
    "confident",
    "confidence",
    "fearless",
    "bold",
    "aggressive mindset",
    "attacks weakness",
    "relentless",
    "unshakeable",
    "self-assured",
    "conviction",
]

# λ (LAMBDA) - Fundamental Constraints Patterns for Poker
# These indicate skill requirements, expertise barriers, training needed
POKER_LAMBDA_PATTERNS = [
    # Elite/world-class skill (20 patterns)
    "elite",
    "elite player",
    "elite competition",
    "elite field",
    "elite level",
    "world-class",
    "world-class player",
    "world-class skill",
    "championship level",
    "championship caliber",
    "top-tier",
    "highest level",
    "professional",
    "professional level",
    "premier",
    "premier player",
    "top player",
    "best in the world",
    "among the best",
    "legendary",
    
    # Skill and expertise (18 patterns)
    "skill",
    "skilled",
    "skillful",
    "expert",
    "expertise",
    "mastery",
    "master",
    "masterful",
    "exceptional",
    "extraordinary",
    "superior",
    "advanced",
    "sophisticated",
    "experienced",
    "veteran",
    "seasoned",
    "accomplished",
    "proven",
    
    # Mathematical/GTO (15 patterns)
    "mathematical",
    "mathematics",
    "game theory",
    "GTO",
    "game theory optimal",
    "optimal play",
    "optimal strategy",
    "mathematical precision",
    "mathematical edge",
    "calculated",
    "analytical",
    "data-driven",
    "systematic",
    "scientific",
    "probability",
    
    # Training and preparation (12 patterns)
    "years of experience",
    "decades of experience",
    "years of training",
    "extensive training",
    "preparation",
    "studied",
    "analyzed",
    "practice",
    "dedicated",
    "commitment",
    "work ethic",
    "poker IQ",
    
    # Technical skills (15 patterns)
    "technical",
    "technical skill",
    "fundamentals",
    "solid fundamentals",
    "hand reading",
    "range",
    "range construction",
    "bet sizing",
    "pot odds",
    "implied odds",
    "equity",
    "ICM",
    "chip EV",
    "stack-to-pot ratio",
    "position",
    
    # Competitive advantage (12 patterns)
    "edge",
    "skill edge",
    "competitive edge",
    "advantage",
    "skill advantage",
    "technical advantage",
    "strategic advantage",
    "tactical advantage",
    "superior play",
    "dominant",
    "crushing",
    "outplaying",
    
    # Tournament/cash game specific (10 patterns)
    "tournament skill",
    "cash game skill",
    "deep stack",
    "short stack",
    "ICM mastery",
    "bubble play",
    "final table",
    "heads-up",
    "multi-table",
    "multi-way",
    
    # Major accomplishments (8 patterns)
    "bracelet",
    "bracelets",
    "title",
    "titles",
    "championship",
    "championships",
    "career earnings",
    "major wins",
]


def get_poker_patterns():
    """Return poker-specific patterns for force extraction"""
    return {
        'theta': POKER_THETA_PATTERNS,
        'lambda': POKER_LAMBDA_PATTERNS
    }


def get_pattern_counts():
    """Get count of patterns"""
    return {
        'theta_patterns': len(POKER_THETA_PATTERNS),
        'lambda_patterns': len(POKER_LAMBDA_PATTERNS),
        'total_patterns': len(POKER_THETA_PATTERNS) + len(POKER_LAMBDA_PATTERNS)
    }


def save_patterns_to_json(output_path):
    """Save patterns to JSON file"""
    import json
    from pathlib import Path
    
    patterns = {
        'domain': 'professional_poker',
        'pattern_library': {
            'theta': {
                'name': 'Awareness/Resistance (θ)',
                'description': 'Mental game awareness, psychological sophistication, self-awareness in poker',
                'pattern_count': len(POKER_THETA_PATTERNS),
                'patterns': POKER_THETA_PATTERNS,
                'categories': {
                    'mental_game': POKER_THETA_PATTERNS[0:20],
                    'tells_reading': POKER_THETA_PATTERNS[20:35],
                    'bluffing': POKER_THETA_PATTERNS[35:47],
                    'table_image': POKER_THETA_PATTERNS[47:57],
                    'composure_tilt': POKER_THETA_PATTERNS[57:75],
                    'pressure': POKER_THETA_PATTERNS[75:90],
                    'patience': POKER_THETA_PATTERNS[90:102],
                    'confidence': POKER_THETA_PATTERNS[102:112]
                }
            },
            'lambda': {
                'name': 'Fundamental Constraints (λ)',
                'description': 'Skill requirements, expertise barriers, training needed in poker',
                'pattern_count': len(POKER_LAMBDA_PATTERNS),
                'patterns': POKER_LAMBDA_PATTERNS,
                'categories': {
                    'elite_skill': POKER_LAMBDA_PATTERNS[0:20],
                    'expertise': POKER_LAMBDA_PATTERNS[20:38],
                    'mathematical': POKER_LAMBDA_PATTERNS[38:53],
                    'training': POKER_LAMBDA_PATTERNS[53:65],
                    'technical': POKER_LAMBDA_PATTERNS[65:80],
                    'advantage': POKER_LAMBDA_PATTERNS[80:92],
                    'tournament_specific': POKER_LAMBDA_PATTERNS[92:102],
                    'accomplishments': POKER_LAMBDA_PATTERNS[102:110]
                }
            }
        },
        'total_patterns': len(POKER_THETA_PATTERNS) + len(POKER_LAMBDA_PATTERNS),
        'expected_forces': {
            'theta': 0.65,
            'lambda': 0.70,
            'rationale': 'Poker players have high awareness (θ=0.65) of mental game but still susceptible to narratives. High skill requirements (λ=0.70) create significant barriers.'
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"✓ Patterns saved to: {output_path}")
    return patterns


if __name__ == '__main__':
    from pathlib import Path
    
    print("="*80)
    print("POKER ENRICHED PATTERNS LIBRARY")
    print("="*80)
    
    counts = get_pattern_counts()
    
    print(f"\nθ (Awareness) Patterns: {counts['theta_patterns']}")
    print(f"λ (Constraints) Patterns: {counts['lambda_patterns']}")
    print(f"Total Patterns: {counts['total_patterns']}")
    
    print(f"\nθ Pattern Categories:")
    print(f"  - Mental game awareness (20)")
    print(f"  - Tells and reading (15)")
    print(f"  - Bluffing awareness (12)")
    print(f"  - Table image (10)")
    print(f"  - Composure and tilt (18)")
    print(f"  - Pressure handling (15)")
    print(f"  - Patience and discipline (12)")
    print(f"  - Confidence and boldness (10)")
    
    print(f"\nλ Pattern Categories:")
    print(f"  - Elite/world-class skill (20)")
    print(f"  - Skill and expertise (18)")
    print(f"  - Mathematical/GTO (15)")
    print(f"  - Training and preparation (12)")
    print(f"  - Technical skills (15)")
    print(f"  - Competitive advantage (12)")
    print(f"  - Tournament specific (10)")
    print(f"  - Major accomplishments (8)")
    
    # Save to JSON
    output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'poker' / 'poker_enriched_patterns.json'
    patterns = save_patterns_to_json(output_path)
    
    print(f"\n✓ Pattern library complete")
    print(f"✓ Ready for transformer application")

