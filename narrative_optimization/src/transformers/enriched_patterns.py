"""
Enriched Pattern Dictionaries for θ (Awareness) and λ (Constraints)

Expands pattern recognition beyond baseline to capture domain-specific
awareness and constraint language. Addresses the problem of many domains
clustering at 0.50 baseline due to insufficient pattern coverage.

Author: Narrative Integration System
Date: November 2025
"""

from typing import Dict, List, Tuple
import numpy as np

from .base import NarrativeTransformer

# ============================================================================
# SPORTS DOMAIN PATTERNS
# ============================================================================

SPORTS_AWARENESS_PATTERNS = {
    # Mental game and psychological awareness
    'mental_game': [
        'mental game', 'mental toughness', 'mentally strong', 'mental edge',
        'psychological advantage', 'mind games', 'mental preparation',
        'mental approach', 'mental side', 'mentally prepared', 'mental fortitude',
        'between the ears', 'all mental', 'mental aspect', 'psychology',
        'mental battle', 'mental strength', 'mental component'
    ],
    
    # Pressure and clutch performance
    'pressure_awareness': [
        'pressure situation', 'under pressure', 'clutch', 'clutch performance',
        'pressure moment', 'high-pressure', 'handles pressure', 'pressure player',
        'big moment', 'crunch time', 'clutch gene', 'pressure cooker',
        'rises to occasion', 'performs under pressure', 'clutch ability',
        'pressure-packed', 'moment of truth', 'high-stakes'
    ],
    
    # Choking and failure awareness
    'failure_awareness': [
        'choke', 'choking', 'choked', 'mental lapse', 'lost composure',
        'crumbled under pressure', 'fell apart', 'couldn\'t handle',
        'nerves got to', 'succumbed to pressure', 'wilted', 'buckled',
        'mental breakdown', 'composure lost', 'lost focus', 'distracted'
    ],
    
    # Confidence and belief
    'confidence': [
        'confidence', 'confident', 'self-belief', 'believes in himself',
        'believes in herself', 'swagger', 'faith in ability', 'conviction',
        'self-assurance', 'certainty', 'belief system', 'trust in self',
        'doubts', 'self-doubt', 'questioning', 'lacks confidence'
    ],
    
    # Focus and concentration
    'focus': [
        'focused', 'concentration', 'concentrating', 'dialed in', 'locked in',
        'in the zone', 'zoned in', 'laser focus', 'tunnel vision', 'centered',
        'composed', 'poise', 'distracted', 'lost focus', 'unfocused',
        'attention', 'mindfulness', 'present', 'in the moment'
    ],
    
    # Momentum and narrative awareness
    'momentum': [
        'momentum', 'momentum shift', 'momentum swing', 'riding momentum',
        'building momentum', 'lost momentum', 'momentum builder',
        'swing momentum', 'momentum change', 'psychological momentum',
        'narrative', 'story', 'storyline', 'script', 'wrote the script'
    ],
    
    # Experience and wisdom
    'experience_awareness': [
        'experience', 'veteran', 'seasoned', 'been there before',
        'knows what it takes', 'championship experience', 'playoff experience',
        'big-game experience', 'learned from', 'matured', 'wisdom',
        'seen it all', 'battle-tested', 'proven', 'inexperienced',
        'first-timer', 'rookie', 'learning curve', 'green'
    ],
    
    # Intelligence and strategy
    'strategic_awareness': [
        'smart player', 'high IQ', 'basketball IQ', 'football IQ', 'game sense',
        'reads the game', 'anticipates', 'thinks the game', 'cerebral',
        'tactical', 'strategic', 'outsmart', 'outthink', 'chess match',
        'mental edge', 'savvy', 'crafty', 'intelligent play'
    ],
    
    # ====== COMBAT SPORTS SPECIFIC (UFC, Boxing, MMA) ======
    
    # Fight IQ and game planning
    'fight_intelligence': [
        'fight IQ', 'high fight IQ', 'game plan', 'game planning', 'strategy',
        'tactical approach', 'coaching difference', 'corner advice', 'coaching',
        'adjustments', 'made adjustments', 'adapt mid-fight', 'ring IQ',
        'octagon IQ', 'fight strategy', 'tactical fighter', 'smart fighter',
        'cerebral fighter', 'chess match', 'tactical battle', 'strategic fight'
    ],
    
    # Styles and matchups
    'styles_matchups': [
        'styles make fights', 'bad matchup', 'good matchup', 'style clash',
        'grappler vs striker', 'wrestler vs striker', 'boxer vs puncher',
        'reach advantage', 'reach disadvantage', 'southpaw advantage',
        'orthodox vs southpaw', 'stylistic advantage', 'style mismatch',
        'style matchup', 'favorable matchup', 'stylistic', 'style matters',
        'matchup problems', 'matchup advantage', 'stylistic edge'
    ],
    
    # Mental warfare and intimidation
    'mental_warfare': [
        'mental warfare', 'psychological warfare', 'mind games', 'intimidation',
        'intimidation factor', 'aura', 'mystique', 'fear factor', 'in his head',
        'in her head', 'psyched out', 'mental advantage', 'psychological edge',
        'intimidated', 'intimidating presence', 'mental game', 'trash talk',
        'gets in head', 'psychological advantage', 'mental dominance',
        'break them mentally', 'mental toughness wins', 'mental battle'
    ],
    
    # Championship experience and big moments
    'championship_experience': [
        'championship experience', 'title fight experience', 'big fight experience',
        'been there before', 'championship pedigree', 'championship caliber',
        'big fight feel', 'big stage', 'bright lights', 'championship moment',
        'title fight pressure', 'championship rounds', 'knows what it takes',
        'championship DNA', 'championship mentality', 'title fight jitters',
        'first title fight', 'championship debut', 'unproven at this level'
    ],
    
    # Technical awareness
    'technical_awareness': [
        'technical breakdown', 'technical analysis', 'Joe Rogan breakdown',
        'DC breakdown', 'commentator insight', 'technical fighter', 'technique',
        'technical game', 'technical skill', 'fundamentals', 'technical edge',
        'technically sound', 'technical mastery', 'technique wins', 'form',
        'technical proficiency', 'technician', 'technical approach',
        'fundamental soundness', 'technique matters'
    ],
    
    # Momentum and finish awareness
    'finish_momentum': [
        'finish', 'finishing ability', 'finisher', 'finish rate', 'knockout power',
        'KO power', 'one-punch power', 'puncher\'s chance', 'submission threat',
        'submission game', 'submission specialist', 'finish fights', 'finishing instinct',
        'goes for finish', 'killer instinct', 'finishing sequence', 'finish potential',
        'knockout artist', 'submission artist', 'dangerous finisher'
    ],
    
    # Physical attributes awareness
    'physical_attributes_combat': [
        'chin', 'iron chin', 'glass chin', 'questionable chin', 'can take a shot',
        'durability', 'durable', 'cardio', 'gas tank', 'conditioning', 'cardio issues',
        'gasses out', 'fades', 'cardio advantage', 'reach', 'reach advantage',
        'length', 'height advantage', 'speed', 'hand speed', 'power', 'knockout power'
    ],
    
    # Betting and odds awareness
    'betting_awareness': [
        'betting favorite', 'betting underdog', 'odds', 'betting line', 'favorite',
        'underdog', 'odds-on favorite', 'heavy favorite', 'upset potential',
        'sharp money', 'line movement', 'betting public', 'value bet',
        'oddsmakers', 'vegas odds', 'opening line', 'closing line', 'odds shift'
    ]
}

SPORTS_CONSTRAINT_PATTERNS = {
    # Physical elite status
    'elite_physical': [
        'world-class', 'elite athlete', 'elite level', 'top-tier', 'world-class athlete',
        'olympic caliber', 'professional level', 'highest level', 'elite status',
        'premier athlete', 'top athlete', 'elite performer', 'world class',
        'championship caliber', 'all-pro', 'all-star level', 'pro bowl'
    ],
    
    # Training and preparation
    'training_required': [
        'years of training', 'dedicated training', 'rigorous training',
        'countless hours', 'thousands of hours', '10,000 hours',
        'training regime', 'training program', 'preparation', 'conditioning',
        'practice', 'repetition', 'muscle memory', 'trained for years',
        'lifetime of training', 'developed over years'
    ],
    
    # Physical talent and gifts
    'natural_ability': [
        'natural talent', 'gifted', 'born with', 'innate ability', 'natural athlete',
        'physical gifts', 'natural ability', 'blessed with', 'naturally gifted',
        'genetic advantage', 'physical tools', 'raw talent', 'god-given',
        'freak athlete', 'exceptional physical', 'rare physical'
    ],
    
    # Physical requirements
    'physical_demands': [
        'physically demanding', 'physical specimen', 'athleticism required',
        'strength needed', 'speed required', 'endurance required',
        'physically gifted', 'athletic ability', 'physical prowess',
        'physical attributes', 'physical capabilities', 'athletic requirements',
        'physical demands', 'conditioning required', 'fitness level'
    ],
    
    # Skill development
    'skill_barriers': [
        'technical skill', 'fundamental skill', 'skilled execution',
        'requires mastery', 'skill level', 'technical proficiency',
        'execution skill', 'refined technique', 'technical ability',
        'skill development', 'years to master', 'difficult to master',
        'high skill ceiling', 'skill-intensive', 'technically demanding'
    ],
    
    # Competition level
    'competitive_level': [
        'highest level of competition', 'professional level', 'elite competition',
        'toughest competition', 'best in the world', 'world championship level',
        'olympic competition', 'premier competition', 'highest caliber',
        'professional sports', 'top-level competition', 'major league'
    ],
    
    # Physical limitations
    'physical_limits': [
        'physical limitation', 'physical ceiling', 'athletic limit',
        'can\'t teach height', 'can\'t teach speed', 'genetic ceiling',
        'physical constraint', 'physical barrier', 'natural limitation',
        'physical disadvantage', 'lacks athleticism', 'physical shortcoming',
        'limited physically', 'physical deficiency'
    ],
    
    # Performance determinism
    'performance_dominated': [
        'performance matters most', 'execution is everything', 'talent wins',
        'better team wins', 'skill determines', 'performance-based',
        'meritocratic', 'ability determines outcome', 'talent trumps',
        'objective performance', 'measurable performance', 'stats don\'t lie'
    ],
    
    # ====== COMBAT SPORTS SPECIFIC (UFC, Boxing, MMA) ======
    
    # Grappling/Wrestling expertise
    'grappling_expertise': [
        'Brazilian Jiu-Jitsu', 'BJJ', 'black belt', 'purple belt', 'brown belt',
        'grappling mastery', 'grappling expert', 'elite grappling', 'wrestling',
        'wrestler', 'Olympic wrestler', 'Division I wrestler', 'NCAA champion',
        'All-American wrestler', 'wrestling pedigree', 'wrestling background',
        'elite wrestler', 'world-class grappling', 'high-level grappling',
        'submission expert', 'submission game', 'ground game', 'grappling specialist'
    ],
    
    # Striking expertise
    'striking_expertise': [
        'elite striking', 'world-class striking', 'championship-level striking',
        'Muay Thai', 'kickboxing', 'boxing background', 'professional boxer',
        'striking mastery', 'striking specialist', 'knockout artist', 'heavy hands',
        'powerful striker', 'technical striker', 'precision striker', 'volume striker',
        'striking pedigree', 'striking background', 'stand-up game', 'striking game',
        'K-1 level', 'Glory kickboxing', 'striking credentials'
    ],
    
    # Training and preparation
    'combat_training': [
        'years of training', 'lifetime of training', 'trained since childhood',
        'full-time fighter', 'professional fighter', 'elite training', 'fight camp',
        'training camp', 'gym', 'team', 'Jackson-Wink', 'American Top Team',
        'Tristar', 'AKA', 'Team Alpha Male', 'training partners', 'coaching',
        'corner', 'elite coaching', 'world-class coaching', 'training facility',
        'preparation', 'fight preparation', 'camp preparation'
    ],
    
    # Physical requirements combat
    'combat_physical': [
        'physical specimen', 'athletic', 'athleticism', 'explosive', 'explosiveness',
        'power', 'speed', 'quickness', 'reflexes', 'reaction time', 'timing',
        'hand speed', 'foot speed', 'agility', 'coordination', 'balance',
        'strength', 'raw strength', 'physical strength', 'functional strength',
        'core strength', 'knockout power', 'one-punch power', 'punching power'
    ],
    
    # Durability and toughness
    'durability_toughness': [
        'chin', 'iron chin', 'durability', 'durable', 'tough', 'toughness',
        'warrior', 'heart', 'grit', 'will', 'never quit', 'can take punishment',
        'takes shots well', 'recovery', 'recovers well', 'resilient', 'resilience',
        'comes back', 'weathered storm', 'survived', 'tough as nails',
        'battle-tested', 'proven toughness', 'proven durability'
    ],
    
    # Cardio and conditioning
    'cardio_conditioning': [
        'cardio', 'gas tank', 'conditioning', 'endurance', 'stamina', 'fitness',
        'well-conditioned', 'elite cardio', 'championship cardio', 'five-round fighter',
        'deep gas tank', 'never tires', 'pace', 'pace pressure', 'high-pace',
        'cardio advantage', 'cardio for days', 'cardio issues', 'gasses', 'fades',
        'cardio problems', 'conditioning problems', 'slows down'
    ],
    
    # Fight-finishing ability
    'finishing_ability': [
        'finish rate', 'finishing ability', 'finisher', 'knockout rate', 'KO rate',
        'submission rate', 'finish percentage', 'rarely goes to decision',
        'always finishes', 'finish fights', 'knockout artist', 'submission specialist',
        'dangerous finisher', 'high finish rate', 'finishing threat', 'finishes early',
        'first-round finisher', 'early finisher', 'finish potential'
    ],
    
    # Experience and record
    'combat_experience': [
        'professional record', 'undefeated', 'unblemished record', 'win streak',
        'winning streak', 'consecutive wins', 'consecutive victories', 'experience',
        'veteran', 'seasoned fighter', 'been around', 'years in the game',
        'fights', 'professional fights', 'amateur fights', 'competition experience',
        'high-level competition', 'faced top competition', 'elite competition',
        'title fight experience', 'championship fights'
    ],
    
    # Technical skill requirements
    'technical_combat_skills': [
        'technical skill', 'technical ability', 'technique', 'fundamentals',
        'technical proficiency', 'skill level', 'elite-level skills', 'well-rounded',
        'complete fighter', 'no weaknesses', 'well-rounded game', 'complete skillset',
        'striking and grappling', 'all aspects', 'every area', 'technical mastery',
        'technical excellence', 'highly skilled', 'skillful', 'technical fighter'
    ],
    
    # Defense and protection
    'defensive_skills': [
        'defense', 'defensive ability', 'defensive skills', 'head movement',
        'footwork', 'distance management', 'range control', 'takedown defense',
        'TDD', 'submission defense', 'guard', 'defensive grappling', 'defensive wrestling',
        'hard to hit', 'elusive', 'slippery', 'defensive prowess', 'defensive mastery',
        'defensive sound', 'defensive fundamentals', 'protects himself', 'protects herself'
    ]
}

# ============================================================================
# MEDICAL/HEALTH DOMAIN PATTERNS
# ============================================================================

MEDICAL_AWARENESS_PATTERNS = {
    # Stigma and social awareness
    'stigma_awareness': [
        'stigma', 'stigmatized', 'social stigma', 'stigma attached',
        'carries stigma', 'stigma surrounding', 'stigmatizing', 'de-stigmatize',
        'reduce stigma', 'fight stigma', 'overcome stigma', 'label',
        'labeled as', 'negative perception', 'misconception', 'misunderstood'
    ],
    
    # Professional knowledge
    'clinical_awareness': [
        'clinically aware', 'clinical understanding', 'medical knowledge',
        'professional understanding', 'clinical insight', 'medical awareness',
        'healthcare awareness', 'clinical recognition', 'diagnostic awareness',
        'symptom awareness', 'condition awareness', 'disease awareness'
    ],
    
    # Public education and understanding
    'public_awareness': [
        'public awareness', 'awareness campaign', 'education about',
        'raising awareness', 'public understanding', 'awareness of condition',
        'widely known', 'commonly known', 'well-known condition',
        'recognizable', 'familiar condition', 'awareness movement'
    ],
    
    # Psychosocial impact awareness
    'impact_awareness': [
        'psychological impact', 'social impact', 'quality of life impact',
        'functional impact', 'disability impact', 'life impact',
        'daily living impact', 'social functioning', 'psychological burden',
        'emotional impact', 'mental health impact', 'wellbeing impact'
    ],
    
    # Name/label awareness
    'naming_awareness': [
        'diagnostic label', 'diagnosis carries', 'name implies', 'label suggests',
        'term conveys', 'language around', 'terminology', 'naming matters',
        'label affects', 'diagnosis means', 'called', 'referred to as',
        'medical terminology', 'clinical terminology', 'diagnostic terminology'
    ],
    
    # Treatment awareness
    'treatment_awareness': [
        'treatment available', 'treatable', 'treatment options', 'therapy available',
        'intervention available', 'management possible', 'can be treated',
        'responds to treatment', 'treatment-responsive', 'therapy-responsive',
        'medical intervention', 'therapeutic intervention'
    ]
}

MEDICAL_CONSTRAINT_PATTERNS = {
    # Professional qualification requirements
    'clinical_expertise': [
        'board certified', 'medical degree', 'clinical training', 'licensed',
        'certified', 'credentialed', 'qualified', 'medical training',
        'clinical experience', 'specialized training', 'professional training',
        'medical education', 'residency', 'fellowship', 'specialist',
        'expert in', 'trained in', 'certified in'
    ],
    
    # Evidence-based medicine
    'evidence_requirements': [
        'evidence-based', 'clinically proven', 'scientifically validated',
        'research-backed', 'empirically supported', 'evidence shows',
        'studies demonstrate', 'clinical trials', 'peer-reviewed',
        'validated treatment', 'proven treatment', 'established treatment',
        'FDA approved', 'clinically tested', 'rigorously tested'
    ],
    
    # Biological/physiological constraints
    'biological_basis': [
        'biological basis', 'physiological', 'neurological', 'genetic',
        'brain chemistry', 'biochemical', 'neurotransmitter', 'hormonal',
        'anatomical', 'organic', 'physical basis', 'biological cause',
        'medical basis', 'physiological cause', 'neurological basis',
        'genetic component', 'hereditary', 'biological mechanism'
    ],
    
    # Diagnostic requirements
    'diagnostic_criteria': [
        'diagnostic criteria', 'clinical criteria', 'DSM criteria',
        'ICD criteria', 'diagnostic standards', 'clinical diagnosis',
        'medical diagnosis', 'objective criteria', 'diagnostic threshold',
        'clinical threshold', 'criteria met', 'diagnosis requires',
        'symptoms must', 'diagnostic assessment', 'clinical assessment'
    ],
    
    # Treatment constraints
    'treatment_barriers': [
        'treatment barriers', 'access to care', 'cost of treatment',
        'specialized treatment', 'long-term treatment', 'intensive treatment',
        'requires medication', 'requires therapy', 'requires hospitalization',
        'treatment compliance', 'adherence required', 'treatment duration',
        'treatment availability', 'limited treatment', 'treatment options limited'
    ],
    
    # Medical expertise required
    'expertise_required': [
        'medical expertise', 'clinical expertise', 'specialist required',
        'expert opinion', 'professional judgment', 'clinical judgment',
        'medical professional', 'healthcare professional', 'trained professional',
        'specialized care', 'expert care', 'professional care',
        'requires specialist', 'specialist consultation', 'expert consultation'
    ],
    
    # Chronicity and severity
    'severity_constraints': [
        'chronic condition', 'severe condition', 'serious condition',
        'life-threatening', 'debilitating', 'disabling', 'progressive',
        'degenerative', 'persistent', 'long-term', 'lifelong',
        'requires ongoing', 'continuous care', 'long-term management',
        'severity level', 'degree of impairment', 'functional impairment'
    ]
}

# ============================================================================
# BUSINESS/STARTUP DOMAIN PATTERNS
# ============================================================================

BUSINESS_AWARENESS_PATTERNS = {
    # Market sophistication
    'market_awareness': [
        'market aware', 'market savvy', 'understands market', 'market knowledge',
        'market sophistication', 'market intelligence', 'competitive awareness',
        'market dynamics', 'market forces', 'market trends', 'market insight',
        'reads the market', 'market sense', 'market timing awareness'
    ],
    
    # Investor sophistication
    'investor_awareness': [
        'sophisticated investor', 'experienced investor', 'VC knows',
        'investor skepticism', 'due diligence', 'investor awareness',
        'seen it before', 'pattern recognition', 'red flags', 'investor savvy',
        'investment experience', 'institutional knowledge', 'deal experience',
        'investment sophistication', 'seasoned investor'
    ],
    
    # Narrative/pitch awareness
    'pitch_awareness': [
        'pitch matters', 'storytelling', 'narrative important', 'branding',
        'positioning', 'framing', 'messaging', 'story resonates',
        'compelling narrative', 'pitch quality', 'story matters',
        'narrative drives', 'perception matters', 'story sells',
        'narrative fit', 'story alignment'
    ],
    
    # Disruption narrative
    'disruption_narrative': [
        'disruption', 'disruptive', 'revolutionary', 'game-changer',
        'paradigm shift', 'transformative', 'innovative', 'breakthrough',
        'reimagine', 'reinvent', 'transform', 'disrupt the industry',
        'category creation', 'market disruption', 'innovative solution'
    ],
    
    # Vision and strategy awareness
    'strategic_awareness': [
        'strategic vision', 'long-term vision', 'strategic thinking',
        'vision matters', 'strategy important', 'strategic position',
        'competitive strategy', 'strategic advantage', 'vision clarity',
        'strategic direction', 'vision-driven', 'strategy-focused',
        'strategic insight', 'visionary', 'strategic plan'
    ],
    
    # Pivot and adaptation awareness
    'adaptation_awareness': [
        'pivot', 'pivoted', 'pivoting', 'adapt', 'adaptation', 'flexible',
        'nimble', 'agile', 'responsive', 'course correct', 'iterate',
        'iteration', 'learning', 'evolving', 'adjusting', 'refining',
        'market feedback', 'customer feedback', 'responsive to'
    ]
}

BUSINESS_CONSTRAINT_PATTERNS = {
    # Market fundamentals
    'market_fundamentals': [
        'product-market fit', 'market need', 'market demand', 'real demand',
        'viable market', 'market size', 'addressable market', 'TAM',
        'market opportunity', 'market gap', 'unmet need', 'pain point',
        'customer need', 'market validation', 'market proof'
    ],
    
    # Execution requirements
    'execution_barriers': [
        'execution matters', 'execution is key', 'execution required',
        'operational excellence', 'execution capability', 'deliver results',
        'proven execution', 'execution track record', 'execute successfully',
        'delivery capability', 'operational capability', 'execution skills',
        'can deliver', 'proven delivery'
    ],
    
    # Technical feasibility
    'technical_constraints': [
        'technically feasible', 'technical capability', 'technical expertise',
        'engineering required', 'technical challenge', 'technically difficult',
        'technical complexity', 'technical barriers', 'technical requirements',
        'build capability', 'development capability', 'technical team',
        'technical skills', 'engineering talent', 'technical foundation'
    ],
    
    # Capital requirements
    'capital_constraints': [
        'capital required', 'funding required', 'capital-intensive',
        'burn rate', 'runway', 'cash flow', 'financial sustainability',
        'path to profitability', 'unit economics', 'monetization',
        'revenue model', 'business model', 'economics work',
        'financial viability', 'sustainable business'
    ],
    
    # Competitive dynamics
    'competitive_barriers': [
        'competitive advantage', 'moat', 'defensibility', 'barriers to entry',
        'competitive position', 'incumbent advantage', 'network effects',
        'switching costs', 'competitive threat', 'competitive pressure',
        'market competition', 'competitive landscape', 'competitors',
        'competitive dynamics', 'market position'
    ],
    
    # Team and talent
    'team_requirements': [
        'team quality', 'founding team', 'team capability', 'team experience',
        'talent required', 'key hire', 'team strength', 'leadership team',
        'technical talent', 'domain expertise', 'team composition',
        'team execution', 'proven team', 'experienced team',
        'team track record', 'founder experience'
    ],
    
    # Regulatory and legal
    'regulatory_barriers': [
        'regulatory requirement', 'regulatory approval', 'compliance',
        'legal requirement', 'regulatory hurdle', 'regulatory risk',
        'regulated industry', 'licensing required', 'certification required',
        'regulatory environment', 'legal barriers', 'regulatory constraints',
        'regulatory approval needed', 'regulatory process'
    ],
    
    # Traction and validation
    'traction_required': [
        'traction', 'market traction', 'customer traction', 'growth',
        'user growth', 'revenue growth', 'metrics', 'KPIs', 'proven model',
        'validation', 'market validation', 'customer validation',
        'product validation', 'traction demonstrated', 'proof points',
        'early traction', 'momentum', 'traction metrics'
    ]
}

# ============================================================================
# ENTERTAINMENT DOMAIN PATTERNS
# ============================================================================

ENTERTAINMENT_AWARENESS_PATTERNS = {
    # Craft and artistry awareness
    'craft_awareness': [
        'storytelling', 'narrative craft', 'artistic vision', 'creative vision',
        'directorial vision', 'auteur', 'artistry', 'craftsmanship',
        'technical mastery', 'artistic merit', 'creative excellence',
        'story structure', 'narrative structure', 'character development',
        'plot development', 'storytelling ability', 'narrative skill'
    ],
    
    # Genre awareness
    'genre_awareness': [
        'genre expectations', 'genre conventions', 'genre subversion',
        'genre mastery', 'genre blending', 'genre-defining', 'genre knowledge',
        'genre understanding', 'genre tropes', 'genre standards',
        'genre innovation', 'genre tradition', 'genre formula'
    ],
    
    # Criticism and evaluation
    'critical_awareness': [
        'critical acclaim', 'critically praised', 'critics recognize',
        'critical reception', 'critical consensus', 'review aggregation',
        'metacritic', 'rotten tomatoes', 'critical evaluation',
        'critical assessment', 'critics note', 'critical perspective',
        'critical analysis', 'reviewer consensus', 'critical response'
    ],
    
    # Audience awareness
    'audience_awareness': [
        'audience expectation', 'audience appeal', 'audience engagement',
        'resonates with audience', 'audience response', 'audience reaction',
        'viewer response', 'audience connection', 'appeals to',
        'target audience', 'demographic appeal', 'broad appeal',
        'niche appeal', 'audience sophistication'
    ],
    
    # Cultural impact awareness
    'cultural_awareness': [
        'cultural impact', 'cultural significance', 'cultural moment',
        'zeitgeist', 'cultural relevance', 'culturally significant',
        'defines generation', 'cultural phenomenon', 'cultural commentary',
        'reflects culture', 'cultural context', 'cultural resonance',
        'social commentary', 'cultural conversation'
    ],
    
    # Prestige awareness
    'prestige_awareness': [
        'prestige', 'prestigious', 'award-worthy', 'oscar-worthy',
        'emmy-worthy', 'acclaim-worthy', 'prestige project', 'prestige film',
        'awards consideration', 'awards campaign', 'for your consideration',
        'prestige television', 'prestige drama', 'awards potential'
    ]
}

ENTERTAINMENT_CONSTRAINT_PATTERNS = {
    # Production quality requirements
    'production_quality': [
        'production value', 'production quality', 'production budget',
        'high production', 'production scale', 'production standards',
        'technical excellence', 'production resources', 'production capability',
        'production expertise', 'production team', 'production design',
        'cinematography', 'visual effects', 'sound design', 'editing quality'
    ],
    
    # Talent requirements
    'talent_barriers': [
        'star power', 'A-list talent', 'bankable stars', 'acting talent',
        'performance quality', 'ensemble cast', 'casting', 'talent level',
        'acclaimed actor', 'proven talent', 'directorial talent',
        'creative talent', 'top-tier talent', 'talent caliber',
        'performance caliber', 'acting ability'
    ],
    
    # Budget constraints
    'budget_requirements': [
        'budget', 'production budget', 'financing', 'financial backing',
        'studio backing', 'budget size', 'budget constraints',
        'expensive production', 'budget limitation', 'financial resources',
        'production costs', 'budget scale', 'well-funded', 'adequately funded',
        'budget sufficient', 'financial investment'
    ],
    
    # Commercial viability
    'commercial_requirements': [
        'box office', 'commercial appeal', 'marketability', 'commercial viability',
        'mainstream appeal', 'commercial success', 'market potential',
        'audience size', 'broad appeal', 'commercial prospects',
        'revenue potential', 'commercial value', 'market appeal',
        'commercial considerations', 'box office potential'
    ],
    
    # Distribution access
    'distribution_barriers': [
        'distribution', 'theatrical release', 'wide release', 'platform',
        'streaming platform', 'distribution deal', 'distributor',
        'release strategy', 'marketing budget', 'marketing support',
        'promotional support', 'distribution channels', 'exhibition',
        'theatrical distribution', 'platform release'
    ],
    
    # Industry dynamics
    'industry_constraints': [
        'industry standards', 'industry expectations', 'studio system',
        'industry norms', 'industry requirements', 'industry pressures',
        'studio notes', 'executive input', 'test screening',
        'focus group', 'industry politics', 'industry relationships',
        'studio backing', 'industry support', 'industry infrastructure'
    ],
    
    # Genre constraints
    'genre_limitations': [
        'genre constraints', 'genre limitations', 'genre expectations limit',
        'genre conventions constrain', 'genre formula', 'genre requirements',
        'genre standards', 'genre boundaries', 'genre restrictions',
        'genre mandates', 'genre rules', 'genre format'
    ]
}

# ============================================================================
# COMBINED DICTIONARY FOR EASY ACCESS
# ============================================================================

ENRICHED_PATTERNS = {
    'sports': {
        'theta': SPORTS_AWARENESS_PATTERNS,
        'lambda': SPORTS_CONSTRAINT_PATTERNS
    },
    'medical': {
        'theta': MEDICAL_AWARENESS_PATTERNS,
        'lambda': MEDICAL_CONSTRAINT_PATTERNS
    },
    'business': {
        'theta': BUSINESS_AWARENESS_PATTERNS,
        'lambda': BUSINESS_CONSTRAINT_PATTERNS
    },
    'entertainment': {
        'theta': ENTERTAINMENT_AWARENESS_PATTERNS,
        'lambda': ENTERTAINMENT_CONSTRAINT_PATTERNS
    }
}

def get_patterns_for_domain(domain_type, force_type='both'):
    """
    Get enriched patterns for a specific domain type.
    
    Args:
        domain_type: 'sports', 'medical', 'business', or 'entertainment'
        force_type: 'theta', 'lambda', or 'both'
    
    Returns:
        Dictionary of pattern categories and their patterns
    """
    if domain_type not in ENRICHED_PATTERNS:
        raise ValueError(f"Unknown domain type: {domain_type}")
    
    if force_type == 'both':
        return ENRICHED_PATTERNS[domain_type]
    elif force_type in ['theta', 'lambda']:
        return ENRICHED_PATTERNS[domain_type][force_type]
    else:
        raise ValueError(f"force_type must be 'theta', 'lambda', or 'both'")

def count_patterns(domain_type=None):
    """Count total patterns available."""
    if domain_type:
        theta_patterns = sum(len(patterns) for patterns in 
                            ENRICHED_PATTERNS[domain_type]['theta'].values())
        lambda_patterns = sum(len(patterns) for patterns in 
                             ENRICHED_PATTERNS[domain_type]['lambda'].values())
        return {
            'theta': theta_patterns,
            'lambda': lambda_patterns,
            'total': theta_patterns + lambda_patterns
        }
    else:
        # Count across all domains
        total = {}
        for dtype in ENRICHED_PATTERNS:
            total[dtype] = count_patterns(dtype)
        return total


class EnrichedPatternsTransformer(NarrativeTransformer):
    """
    Transformer that converts the enriched θ/λ pattern dictionaries into
    numerical features by counting pattern hits per category.
    """

    def __init__(
        self,
        domain_type: str = 'sports',
        force_type: str = 'both',
        normalize: bool = False,
    ):
        super().__init__(
            narrative_id='enriched_patterns',
            description='Counts θ/λ narrative pattern categories for a domain.'
        )
        self.domain_type = domain_type
        self.force_type = force_type
        self.normalize = normalize
        self._compiled_patterns: List[Tuple[str, List[str]]] = []
        self.feature_names_: List[str] = []

    def fit(self, X, y=None):
        # Build pattern catalog for requested domain/type
        domain_patterns = get_patterns_for_domain(
            self.domain_type,
            force_type=self.force_type
        )

        if self.force_type == 'both':
            iterator = domain_patterns.items()
        else:
            iterator = [(self.force_type, domain_patterns)]

        self._compiled_patterns = []
        self.feature_names_ = []

        for dimension, categories in iterator:
            for category, phrases in categories.items():
                feature_name = f"{self.domain_type}.{dimension}.{category}"
                self.feature_names_.append(feature_name)
                self._compiled_patterns.append((feature_name, phrases))

        self.metadata['domains'] = self.domain_type
        self.metadata['force_type'] = self.force_type
        self.metadata['feature_names'] = self.feature_names_
        self.metadata['total_patterns'] = len(self.feature_names_)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        n_samples = len(X)
        n_features = len(self._compiled_patterns)
        features = np.zeros((n_samples, n_features), dtype=float)

        for i, text in enumerate(X):
            lowered = text.lower()
            for j, (_, phrases) in enumerate(self._compiled_patterns):
                count = 0
                for phrase in phrases:
                    count += lowered.count(phrase.lower())
                features[i, j] = count

            if self.normalize:
                word_count = max(len(text.split()), 1)
                features[i, :] = features[i, :] / word_count

        return features

    def get_feature_names(self) -> List[str]:
        self._validate_fitted()
        return self.feature_names_

    def _generate_interpretation(self) -> str:
        return (
            f"Counts θ/λ enriched pattern categories for domain '{self.domain_type}' "
            f"({self.force_type}). Higher values indicate stronger awareness or "
            "constraint language for each category."
        )

if __name__ == '__main__':
    # Display pattern counts
    print("="*80)
    print("ENRICHED PATTERN DICTIONARY STATISTICS")
    print("="*80)
    
    counts = count_patterns()
    
    for domain_type, domain_counts in counts.items():
        print(f"\n{domain_type.upper()}:")
        print(f"  θ (Awareness) patterns: {domain_counts['theta']}")
        print(f"  λ (Constraints) patterns: {domain_counts['lambda']}")
        print(f"  Total: {domain_counts['total']}")
    
    grand_total = sum(d['total'] for d in counts.values())
    print(f"\n{'='*80}")
    print(f"GRAND TOTAL: {grand_total} patterns across all domains")
    print(f"{'='*80}")
    
    print("\nPattern categories by domain:")
    for dtype in ENRICHED_PATTERNS:
        print(f"\n{dtype.upper()} - θ categories:", 
              list(ENRICHED_PATTERNS[dtype]['theta'].keys()))
        print(f"{dtype.upper()} - λ categories:", 
              list(ENRICHED_PATTERNS[dtype]['lambda'].keys()))

