"""
Comprehensive Vocabulary List for Temporal Linguistics

500+ words across 15 categories for robust cycle detection.
"""

COMPREHENSIVE_VOCABULARY = {
    'war_military': {
        'words': [
            # Weapons
            'sword', 'spear', 'arrow', 'musket', 'rifle', 'cannon', 'artillery', 'bayonet',
            'grenade', 'bomb', 'missile', 'torpedo', 'mine', 'tank', 'aircraft', 'drone',
            
            # Actions
            'battle', 'combat', 'warfare', 'conflict', 'siege', 'assault', 'raid', 'ambush',
            'invasion', 'retreat', 'surrender', 'victory', 'defeat', 'truce', 'ceasefire',
            
            # Personnel
            'soldier', 'warrior', 'trooper', 'infantry', 'cavalry', 'marine', 'sailor',
            'pilot', 'general', 'commander', 'veteran', 'recruit', 'mercenary',
            
            # Locations
            'battlefield', 'trench', 'bunker', 'fortress', 'garrison', 'barracks', 'front',
            
            # Concepts
            'strategy', 'tactics', 'campaign', 'offensive', 'defensive', 'morale',
            'casualties', 'prisoners', 'insurgent', 'guerrilla', 'terrorism'
        ],
        'expected_cycle': 75,
        'hypothesis': 'crisis_rhyming'
    },
    
    'economic_financial': {
        'words': [
            # Market terms
            'market', 'stock', 'bond', 'commodity', 'futures', 'option', 'derivative',
            'security', 'portfolio', 'index', 'exchange', 'trading', 'speculation',
            
            # Crises
            'crash', 'panic', 'crisis', 'bubble', 'collapse', 'meltdown', 'contagion',
            
            # Cycles
            'boom', 'bust', 'recession', 'depression', 'recovery', 'expansion', 'contraction',
            'inflation', 'deflation', 'stagnation', 'prosperity', 'austerity',
            
            # Banking
            'bank', 'credit', 'debt', 'loan', 'mortgage', 'interest', 'liquidity',
            'capital', 'asset', 'liability', 'equity', 'leverage', 'default', 'bankruptcy',
            
            # Money
            'currency', 'dollar', 'gold', 'silver', 'money', 'wealth', 'fortune', 'poverty'
        ],
        'expected_cycle': 25,
        'hypothesis': 'economic_cycle'
    },
    
    'technology': {
        'words': [
            # Communication
            'telegraph', 'telephone', 'radio', 'television', 'broadcast', 'cable',
            'satellite', 'fiber', 'wireless', 'cellular', 'smartphone',
            
            # Computing
            'computer', 'processor', 'memory', 'storage', 'disk', 'chip', 'circuit',
            'transistor', 'semiconductor', 'microprocessor', 'silicon',
            
            # Internet
            'internet', 'web', 'website', 'browser', 'email', 'network', 'server',
            'cloud', 'data', 'digital', 'virtual', 'cyber', 'online',
            
            # Modern tech
            'algorithm', 'software', 'hardware', 'database', 'application', 'platform',
            'interface', 'protocol', 'encryption', 'blockchain', 'cryptocurrency',
            
            # Metaphors
            'wire', 'tube', 'stream', 'portal', 'gateway', 'node', 'hub'
        ],
        'expected_cycle': 30,
        'hypothesis': 'tech_innovation'
    },
    
    'slang_approval': {
        'words': [
            # Pre-1950
            'swell', 'keen', 'dandy', 'nifty', 'peachy', 'neato',
            
            # 1950s-60s
            'groovy', 'hip', 'far out', 'boss', 'outta sight',
            
            # 1970s-80s
            'rad', 'tubular', 'gnarly', 'wicked', 'bad', 'fresh',
            
            # 1990s
            'phat', 'tight', 'dope', 'fly', 'def', 'all that',
            
            # 2000s-2010s
            'sick', 'beast', 'epic', 'legit', 'savage', 'fire', 'lit',
            
            # 2020s
            'slaps', 'bussin', 'valid', 'based', 'goated', 'hits different'
        ],
        'expected_cycle': 25,
        'hypothesis': 'generation_cycle'
    },
    
    'victorian_formal': {
        'words': [
            # Intensifiers
            'splendid', 'capital', 'excellent', 'superb', 'magnificent', 'marvelous',
            'wonderful', 'delightful', 'charming', 'gracious', 'elegant',
            
            # Negative
            'dreadful', 'frightful', 'ghastly', 'horrid', 'beastly', 'wretched',
            'deplorable', 'abominable', 'detestable', 'odious',
            
            # Modifiers
            'quite', 'rather', 'utterly', 'thoroughly', 'perfectly', 'exceedingly',
            'frightfully', 'dreadfully', 'awfully',
            
            # Nouns
            'gentleman', 'lady', 'chap', 'fellow', 'maiden', 'suitor', 'parlor',
            'governess', 'chambermaid'
        ],
        'expected_cycle': 120,
        'hypothesis': 'victorian_revival'
    },
    
    'emotion_psychological': {
        'words': [
            # Anxiety cluster
            'anxiety', 'worry', 'dread', 'fear', 'panic', 'terror', 'alarm', 'fright',
            'nervousness', 'apprehension', 'unease', 'distress', 'angst',
            
            # Depression cluster
            'depression', 'melancholy', 'sadness', 'gloom', 'despair', 'misery',
            'sorrow', 'grief', 'anguish', 'despondency', 'dejection',
            
            # Joy cluster
            'joy', 'happiness', 'delight', 'pleasure', 'bliss', 'ecstasy', 'jubilation',
            'elation', 'euphoria', 'contentment', 'satisfaction', 'cheer',
            
            # Anger cluster
            'anger', 'rage', 'fury', 'wrath', 'ire', 'indignation', 'outrage'
        ],
        'expected_cycle': None,
        'hypothesis': 'cultural_mood_cycles'
    },
    
    'social_movements': {
        'words': [
            'liberty', 'freedom', 'equality', 'justice', 'rights', 'democracy',
            'revolution', 'reform', 'protest', 'activism', 'rebellion', 'resistance',
            'suffrage', 'emancipation', 'liberation', 'oppression', 'tyranny',
            'solidarity', 'collective', 'movement', 'struggle', 'empowerment'
        ],
        'expected_cycle': 50,
        'hypothesis': 'social_awakening_cycles'
    },
    
    'scientific_terms': {
        'words': [
            'atom', 'molecule', 'cell', 'organism', 'evolution', 'species', 'gene',
            'protein', 'enzyme', 'bacteria', 'virus', 'chromosome', 'DNA',
            'theory', 'hypothesis', 'experiment', 'observation', 'evidence',
            'quantum', 'relativity', 'gravity', 'energy', 'matter', 'particle'
        ],
        'expected_cycle': None,
        'hypothesis': 'discovery_waves'
    },
    
    'architectural_spatial': {
        'words': [
            'palace', 'castle', 'manor', 'cottage', 'villa', 'mansion', 'estate',
            'tower', 'cathedral', 'temple', 'mosque', 'synagogue', 'church',
            'skyscraper', 'apartment', 'condo', 'loft', 'studio', 'penthouse',
            'courtyard', 'terrace', 'balcony', 'veranda', 'porch', 'patio'
        ],
        'expected_cycle': 100,
        'hypothesis': 'architectural_nostalgia'
    },
    
    'food_culinary': {
        'words': [
            'feast', 'banquet', 'supper', 'dinner', 'lunch', 'breakfast', 'brunch',
            'cuisine', 'recipe', 'gourmet', 'delicacy', 'delicious', 'savory',
            'roast', 'bake', 'fry', 'grill', 'steam', 'boil', 'simmer',
            'spice', 'herb', 'seasoning', 'flavor', 'taste', 'aroma', 'texture'
        ],
        'expected_cycle': None,
        'hypothesis': 'culinary_trends'
    },
    
    'fashion_style': {
        'words': [
            'fashion', 'style', 'elegant', 'chic', 'vogue', 'trendy', 'stylish',
            'garment', 'attire', 'costume', 'outfit', 'ensemble', 'wardrobe',
            'silk', 'velvet', 'linen', 'cotton', 'wool', 'leather',
            'hat', 'bonnet', 'cap', 'coat', 'cloak', 'dress', 'gown', 'suit'
        ],
        'expected_cycle': 30,
        'hypothesis': 'fashion_cycles'
    },
    
    'nature_environment': {
        'words': [
            'nature', 'wilderness', 'forest', 'jungle', 'desert', 'mountain', 'valley',
            'river', 'lake', 'ocean', 'sea', 'beach', 'island', 'coast',
            'tree', 'flower', 'plant', 'vegetation', 'foliage', 'meadow',
            'sky', 'cloud', 'rain', 'storm', 'thunder', 'lightning', 'wind',
            'season', 'spring', 'summer', 'autumn', 'winter', 'climate'
        ],
        'expected_cycle': None,
        'hypothesis': 'environmental_awareness'
    },
    
    'medicine_health': {
        'words': [
            'medicine', 'doctor', 'physician', 'surgeon', 'nurse', 'patient',
            'disease', 'illness', 'sickness', 'ailment', 'malady', 'affliction',
            'cure', 'treatment', 'remedy', 'therapy', 'healing', 'recovery',
            'hospital', 'clinic', 'infirmary', 'ward', 'surgery', 'operation',
            'diagnosis', 'symptom', 'prognosis', 'epidemic', 'pandemic', 'contagion'
        ],
        'expected_cycle': None,
        'hypothesis': 'medical_terminology_evolution'
    },
    
    'education_learning': {
        'words': [
            'education', 'school', 'college', 'university', 'academy', 'institute',
            'student', 'pupil', 'scholar', 'teacher', 'professor', 'tutor',
            'lesson', 'lecture', 'class', 'course', 'curriculum', 'syllabus',
            'learn', 'study', 'knowledge', 'wisdom', 'intelligence', 'understanding',
            'exam', 'test', 'grade', 'degree', 'diploma', 'certificate'
        ],
        'expected_cycle': None,
        'hypothesis': 'educational_discourse'
    },
    
    'art_aesthetic': {
        'words': [
            'art', 'painting', 'sculpture', 'drawing', 'portrait', 'landscape',
            'beauty', 'aesthetic', 'elegant', 'graceful', 'sublime', 'magnificent',
            'artist', 'painter', 'sculptor', 'creator', 'artisan', 'craftsman',
            'gallery', 'museum', 'exhibition', 'masterpiece', 'work', 'creation',
            'style', 'technique', 'composition', 'color', 'form', 'texture'
        ],
        'expected_cycle': 50,
        'hypothesis': 'artistic_movements'
    },
    
    'transportation': {
        'words': [
            'carriage', 'wagon', 'cart', 'coach', 'automobile', 'car', 'vehicle',
            'train', 'railway', 'locomotive', 'station', 'track',
            'ship', 'boat', 'vessel', 'steamship', 'cruise', 'yacht',
            'airplane', 'aircraft', 'jet', 'helicopter', 'aviation', 'flight',
            'journey', 'travel', 'voyage', 'trip', 'expedition', 'passenger'
        ],
        'expected_cycle': 40,
        'hypothesis': 'transportation_evolution'
    }
}


def get_all_words():
    """Get complete list of all words."""
    all_words = []
    for category in COMPREHENSIVE_VOCABULARY.values():
        all_words.extend(category['words'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in all_words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    return unique_words


def get_category_info():
    """Get summary information about categories."""
    info = {}
    total = 0
    
    for cat_name, cat_data in COMPREHENSIVE_VOCABULARY.items():
        word_count = len(cat_data['words'])
        info[cat_name] = {
            'count': word_count,
            'expected_cycle': cat_data['expected_cycle'],
            'hypothesis': cat_data['hypothesis']
        }
        total += word_count
    
    info['total'] = total
    return info


if __name__ == "__main__":
    words = get_all_words()
    info = get_category_info()
    
    print(f"Total unique words: {len(words)}")
    print(f"\nBy category:")
    for cat, data in sorted(info.items()):
        if cat != 'total':
            print(f"  {cat:25s} {data['count']:3d} words")
    
    print(f"\nTotal word count: {info['total']}")

