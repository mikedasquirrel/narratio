"""
Framing Transformer

Captures how narratives frame the same facts through different perspectives,
gain/loss orientation, victim/agent positioning, and metaphor families.

Research Foundation:
- Prospect Theory (Kahneman & Tversky): Losses loom larger than gains
- Framing effects: Same facts, different presentation = different decisions
- Agent vs. victim framing affects perceived control and outcomes
- Metaphors shape understanding (Lakoff & Johnson)

Universal across domains:
- Sports: Winners frame vs. losers frame ("We fought" vs. "They dominated")
- Products: Benefits focus vs. problem-solving frame
- Profiles: Growth journey vs. stable foundation
- Brands: Innovation vs. tradition framing
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string

# AI-powered semantic analysis (wavelike interpretation)
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.semantic_analyzer import get_semantic_analyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class FramingTransformer(NarrativeTransformer):
    """
    Analyzes narrative framing and perspective.
    
    Tests hypothesis that framing (how facts are presented) predicts
    outcomes independent of the facts themselves.
    
    Features extracted (12):
    - Gain vs. loss framing ratio
    - Problem vs. solution orientation
    - Positive vs. negative spin on same facts
    - Victim vs. agent framing
    - Glass half-full/empty markers
    - Opportunity vs. threat language
    - Metaphor family detection (war, journey, game, growth, machine)
    - Reframing indicators
    - Control vs. helplessness language
    - Optimism vs. pessimism markers
    
    Parameters
    ----------
    track_metaphors : bool
        Whether to detect and categorize metaphor families
    """
    
    def __init__(self, track_metaphors: bool = True, use_ai: bool = True, domain_config=None):
        super().__init__(
            narrative_id="framing",
            description="Narrative framing: wavelike interpretative analysis with AI-powered semantic understanding"
        )
        
        self.track_metaphors = track_metaphors
        self.use_ai = use_ai and AI_AVAILABLE
        self.semantic_analyzer = None
        self.domain_config = domain_config
        
        if self.use_ai:
            try:
                from config.api_config import OPENAI_API_KEY
                self.semantic_analyzer = get_semantic_analyzer(OPENAI_API_KEY)
            except:
                self.use_ai = False  # Fallback to keywords
        
        # Gain vs. Loss framing (Prospect Theory)
        self.gain_language = [
            'gain', 'win', 'earn', 'benefit', 'profit', 'advantage', 'bonus', 
            'reward', 'achieve', 'acquire', 'obtain', 'add', 'increase', 'grow',
            'improve', 'enhance', 'boost', 'elevate', 'positive', 'success'
        ]
        
        self.loss_language = [
            'lose', 'loss', 'cost', 'expense', 'sacrifice', 'give up', 'forfeit',
            'risk', 'danger', 'threat', 'decrease', 'reduce', 'decline', 'fall',
            'drop', 'negative', 'failure', 'miss', 'lack', 'absence'
        ]
        
        # Problem vs. Solution orientation
        self.problem_language = [
            'problem', 'issue', 'challenge', 'difficulty', 'obstacle', 'barrier',
            'struggle', 'trouble', 'concern', 'worry', 'flaw', 'defect', 'weakness',
            'limitation', 'constraint', 'complication', 'dilemma'
        ]
        
        self.solution_language = [
            'solution', 'answer', 'fix', 'resolve', 'address', 'handle', 'solve',
            'overcome', 'manage', 'deal with', 'remedy', 'cure', 'improve', 'enhance',
            'optimize', 'streamline', 'simplify', 'facilitate'
        ]
        
        # Victim vs. Agent framing
        self.victim_language = [
            'happened to', 'was done to', 'suffered', 'victim', 'affected by',
            'impacted by', 'subjected to', 'exposed to', 'vulnerable', 'helpless',
            'powerless', 'at the mercy of', 'couldn\'t', 'unable', 'forced'
        ]
        
        self.agent_language = [
            'I did', 'we did', 'chose to', 'decided', 'took action', 'made',
            'created', 'built', 'achieved', 'accomplished', 'controlled', 'directed',
            'managed', 'handled', 'led', 'initiated', 'drove', 'powered'
        ]
        
        # Optimism vs. Pessimism
        self.optimism_markers = [
            'hope', 'optimistic', 'positive', 'confident', 'expect success',
            'looking forward', 'excited', 'encouraged', 'promising', 'bright',
            'opportunity', 'potential', 'growth', 'progress', 'better'
        ]
        
        self.pessimism_markers = [
            'fear', 'pessimistic', 'negative', 'doubt', 'expect failure',
            'worried', 'concerned', 'anxious', 'discouraging', 'bleak',
            'threat', 'risk', 'decline', 'worse', 'deteriorating'
        ]
        
        # Opportunity vs. Threat framing
        self.opportunity_frame = [
            'opportunity', 'chance', 'possibility', 'potential', 'prospect',
            'opening', 'window', 'doorway', 'pathway', 'avenue'
        ]
        
        self.threat_frame = [
            'threat', 'danger', 'risk', 'hazard', 'peril', 'jeopardy',
            'menace', 'warning', 'alarm', 'concern'
        ]
        
        # Control vs. Helplessness
        self.control_language = [
            'control', 'manage', 'direct', 'steer', 'guide', 'regulate',
            'influence', 'shape', 'determine', 'decide', 'choose', 'command'
        ]
        
        self.helplessness_language = [
            'helpless', 'powerless', 'unable', 'can\'t control', 'at the mercy',
            'dependent', 'vulnerable', 'subject to', 'victim of', 'passive'
        ]
        
        # Reframing indicators
        self.reframing_markers = [
            'on the other hand', 'alternatively', 'different perspective', 'another way',
            'from this angle', 'looking at it', 'consider that', 'think of it as',
            'reframe', 'reconsider', 'rethink', 'different lens', 'new light'
        ]
        
        # Metaphor families (Lakoff & Johnson)
        self.metaphors = {
            'war': [
                'battle', 'fight', 'combat', 'struggle', 'attack', 'defend', 'victory',
                'defeat', 'enemy', 'ally', 'strategy', 'tactics', 'conquer', 'surrender',
                'warrior', 'soldier', 'campaign', 'offensive', 'defensive', 'frontline'
            ],
            'journey': [
                'journey', 'path', 'road', 'travel', 'destination', 'milestone', 'step',
                'progress', 'move forward', 'crossroads', 'detour', 'roadblock', 'guide',
                'navigate', 'direction', 'route', 'voyage', 'expedition', 'trek'
            ],
            'game': [
                'game', 'play', 'player', 'team', 'win', 'lose', 'score', 'compete',
                'competition', 'opponent', 'rules', 'fair play', 'level playing field',
                'move', 'strategy', 'tactics', 'gamble', 'bet', 'odds'
            ],
            'growth': [
                'grow', 'seed', 'plant', 'root', 'cultivate', 'nurture', 'blossom',
                'flourish', 'bloom', 'harvest', 'fruit', 'organic', 'natural', 'evolve',
                'develop', 'mature', 'sprout', 'branch', 'fertile'
            ],
            'machine': [
                'machine', 'mechanism', 'engine', 'system', 'process', 'operate', 'function',
                'efficient', 'optimize', 'streamline', 'automated', 'programmed', 'input',
                'output', 'processing', 'technical', 'precision', 'calibrate'
            ],
            'building': [
                'build', 'construct', 'foundation', 'structure', 'framework', 'scaffold',
                'architecture', 'design', 'blueprint', 'solid', 'stable', 'brick', 'layer',
                'support', 'pillar', 'base', 'construct', 'erect', 'assemble'
            ]
        }
    
    def fit(self, X, y=None):
        """
        Learn framing patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        # Analyze corpus-level framing patterns
        gain_counts = []
        loss_counts = []
        problem_counts = []
        solution_counts = []
        metaphor_distribution = {family: 0 for family in self.metaphors.keys()}
        
        for text in X:
            text_lower = text.lower()
            
            # Gain/loss balance
            gain_count = sum(1 for word in self.gain_language if word in text_lower)
            loss_count = sum(1 for word in self.loss_language if word in text_lower)
            gain_counts.append(gain_count)
            loss_counts.append(loss_count)
            
            # Problem/solution balance
            prob_count = sum(1 for word in self.problem_language if word in text_lower)
            sol_count = sum(1 for word in self.solution_language if word in text_lower)
            problem_counts.append(prob_count)
            solution_counts.append(sol_count)
            
            # Metaphor usage
            if self.track_metaphors:
                for family, words in self.metaphors.items():
                    count = sum(1 for word in words if word in text_lower)
                    metaphor_distribution[family] += count
        
        # Metadata
        self.metadata['avg_gain_count'] = np.mean(gain_counts) if gain_counts else 0
        self.metadata['avg_loss_count'] = np.mean(loss_counts) if loss_counts else 0
        self.metadata['avg_problem_count'] = np.mean(problem_counts) if problem_counts else 0
        self.metadata['avg_solution_count'] = np.mean(solution_counts) if solution_counts else 0
        self.metadata['metaphor_distribution'] = metaphor_distribution
        self.metadata['dominant_metaphor'] = max(metaphor_distribution.items(), key=lambda x: x[1])[0] if any(metaphor_distribution.values()) else None
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to framing features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array, shape (n_samples, 12)
            Framing feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_framing_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_framing_features(self, text: str) -> np.ndarray:
        """Extract all 12 framing features from text (AI-enhanced wavelike interpretation)."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words) if words else 1
        
        features = []
        
        # 1. Gain vs. loss framing ratio (AI-ENHANCED)
        if self.use_ai and self.semantic_analyzer:
            # Wavelike interpretation using AI
            gain_sim = self.semantic_analyzer.semantic_similarity(
                text[:500],
                "gain, benefit, profit, win, advantage, positive outcome, success, achievement",
                context="prospect theory framing"
            )
            loss_sim = self.semantic_analyzer.semantic_similarity(
                text[:500],
                "loss, cost, risk, sacrifice, negative outcome, failure, disadvantage",
                context="prospect theory framing"
            )
            total_gl = gain_sim + loss_sim
            gain_ratio = gain_sim / total_gl if total_gl > 0 else 0.5
            features.append(gain_ratio)
        else:
            # Fallback to keyword matching
            gain_count = sum(1 for word in self.gain_language if word in text_lower)
            loss_count = sum(1 for word in self.loss_language if word in text_lower)
            total_gl = gain_count + loss_count
            gain_ratio = gain_count / total_gl if total_gl > 0 else 0.5
            features.append(gain_ratio)
        
        # 2. Problem vs. solution orientation (AI-ENHANCED)
        if self.use_ai and self.semantic_analyzer:
            problem_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "problem, issue, challenge, difficulty, obstacle, concern",
                context="problem-solution framing")
            solution_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "solution, answer, fix, resolve, overcome, remedy",
                context="problem-solution framing")
            total_ps = problem_sim + solution_sim
            solution_ratio = solution_sim / total_ps if total_ps > 0 else 0.5
            features.append(solution_ratio)
        else:
            problem_count = sum(1 for word in self.problem_language if word in text_lower)
            solution_count = sum(1 for word in self.solution_language if word in text_lower)
            total_ps = problem_count + solution_count
            solution_ratio = solution_count / total_ps if total_ps > 0 else 0.5
            features.append(solution_ratio)
        
        # 3. Victim vs. agent framing (AI-ENHANCED)
        if self.use_ai and self.semantic_analyzer:
            victim_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "victim, helpless, acted upon, passive, vulnerable, powerless",
                context="agency framing")
            agent_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "agent, actor, control, active, powerful, choosing, deciding",
                context="agency framing")
            total_va = victim_sim + agent_sim
            agent_ratio = agent_sim / total_va if total_va > 0 else 0.5
            features.append(agent_ratio)
        else:
            victim_count = sum(text_lower.count(phrase) for phrase in self.victim_language)
            agent_count = sum(text_lower.count(phrase) for phrase in self.agent_language)
            total_va = victim_count + agent_count
            agent_ratio = agent_count / total_va if total_va > 0 else 0.5
            features.append(agent_ratio)
        
        # 4. Optimism vs. pessimism (AI-ENHANCED - highly context-dependent!)
        if self.use_ai and self.semantic_analyzer:
            optimism_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "optimistic, hopeful, positive, confident, encouraging, bright future",
                context="emotional outlook")
            pessimism_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "pessimistic, doubtful, negative, worried, discouraging, bleak",
                context="emotional outlook")
            total_op = optimism_sim + pessimism_sim
            optimism_ratio = optimism_sim / total_op if total_op > 0 else 0.5
            features.append(optimism_ratio)
        else:
            optimism_count = sum(1 for word in self.optimism_markers if word in text_lower)
            pessimism_count = sum(1 for word in self.pessimism_markers if word in text_lower)
            total_op = optimism_count + pessimism_count
            optimism_ratio = optimism_count / total_op if total_op > 0 else 0.5
            features.append(optimism_ratio)
        
        # 5. Opportunity vs. threat framing (AI-ENHANCED)
        if self.use_ai and self.semantic_analyzer:
            opportunity_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "opportunity, chance, possibility, potential, opening, prospect",
                context="opportunity framing")
            threat_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "threat, danger, risk, hazard, peril, menace, warning",
                context="threat framing")
            total_ot = opportunity_sim + threat_sim
            opportunity_ratio = opportunity_sim / total_ot if total_ot > 0 else 0.5
            features.append(opportunity_ratio)
        else:
            opportunity_count = sum(1 for word in self.opportunity_frame if word in text_lower)
            threat_count = sum(1 for word in self.threat_frame if word in text_lower)
            total_ot = opportunity_count + threat_count
            opportunity_ratio = opportunity_count / total_ot if total_ot > 0 else 0.5
            features.append(opportunity_ratio)
        
        # 6. Control vs. helplessness (AI-ENHANCED)
        if self.use_ai and self.semantic_analyzer:
            control_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "control, manage, direct, influence, command, power, agency",
                context="locus of control")
            helpless_sim = self.semantic_analyzer.semantic_similarity(
                text[:500], "helpless, powerless, unable, dependent, vulnerable, passive",
                context="locus of control")
            total_ch = control_sim + helpless_sim
            control_ratio = control_sim / total_ch if total_ch > 0 else 0.5
            features.append(control_ratio)
        else:
            control_count = sum(1 for word in self.control_language if word in text_lower)
            helpless_count = sum(1 for word in self.helplessness_language if word in text_lower)
            total_ch = control_count + helpless_count
            control_ratio = control_count / total_ch if total_ch > 0 else 0.5
            features.append(control_ratio)
        
        # 7. Overall framing valence (composite positive vs. negative)
        # Positive = gain + solution + agent + optimism + opportunity + control
        # Negative = loss + problem + victim + pessimism + threat + helplessness
        positive_total = gain_count + solution_count + agent_count + optimism_count + opportunity_count + control_count
        negative_total = loss_count + problem_count + victim_count + pessimism_count + threat_count + helpless_count
        total_valence = positive_total + negative_total
        positive_valence = positive_total / total_valence if total_valence > 0 else 0.5
        features.append(positive_valence)  # 0-1 overall framing tone
        
        # 8. Reframing density (explicit perspective shifts)
        reframing_count = sum(text_lower.count(phrase) for phrase in self.reframing_markers)
        features.append(reframing_count / word_count * 100)
        
        # 9-14. Metaphor family dominance (6 families)
        if self.track_metaphors:
            metaphor_counts = {}
            total_metaphors = 0
            for family, words in self.metaphors.items():
                count = sum(1 for word in words if word in text_lower)
                metaphor_counts[family] = count
                total_metaphors += count
            
            # Metaphor family percentages (which metaphor dominates)
            for family in ['war', 'journey', 'game', 'growth']:
                family_ratio = metaphor_counts[family] / total_metaphors if total_metaphors > 0 else 0
                features.append(family_ratio)
        else:
            features.extend([0, 0, 0, 0])  # Placeholder if not tracking
        
        return np.array(features)
    
    def _load_domain_patterns(self):
        """Load domain-specific framing patterns from config."""
        if not self.domain_config:
            return
        
        # Try to get domain-specific framing patterns
        domain_patterns = self.domain_config.get_domain_specific_patterns('framing')
        
        if domain_patterns:
            # Extend existing pattern lists with domain-specific ones
            if 'gain_language' in domain_patterns:
                self.gain_language.extend(domain_patterns['gain_language'])
            if 'problem_language' in domain_patterns:
                self.problem_language.extend(domain_patterns['problem_language'])
            if 'agent_language' in domain_patterns:
                self.agent_language.extend(domain_patterns['agent_language'])
    
    def get_feature_names(self) -> List[str]:
        """Return names of all features."""
        names = [
            'gain_vs_loss_ratio',
            'solution_vs_problem_ratio',
            'agent_vs_victim_ratio',
            'optimism_vs_pessimism',
            'opportunity_vs_threat',
            'control_vs_helplessness',
            'overall_positive_valence',
            'reframing_density',
            'metaphor_war',
            'metaphor_journey',
            'metaphor_game',
            'metaphor_growth'
        ]
        return names
    
    def interpret_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Interpret framing features in plain English.
        
        Parameters
        ----------
        features : array, shape (12,)
            Feature vector for one document
        
        Returns
        -------
        interpretation : dict
            Plain English interpretation
        """
        names = self.get_feature_names()
        
        interpretation = {
            'summary': self._generate_summary(features),
            'features': {},
            'insights': []
        }
        
        # Interpret each feature
        for i, (name, value) in enumerate(zip(names, features)):
            interpretation['features'][name] = {
                'value': float(value),
                'description': self._describe_feature(name, value)
            }
        
        # Generate insights
        if features[0] > 0.65:  # Gain-focused
            interpretation['insights'].append("Gain-focused framing - emphasizes benefits and wins")
        elif features[0] < 0.35:
            interpretation['insights'].append("Loss-avoidance framing - emphasizes risks and costs")
        
        if features[1] > 0.60:  # Solution-oriented
            interpretation['insights'].append("Solution-oriented - focuses on fixing problems")
        elif features[1] < 0.40:
            interpretation['insights'].append("Problem-focused - emphasizes difficulties")
        
        if features[2] > 0.60:  # Agent frame
            interpretation['insights'].append("Agent framing - portrays control and active choice")
        elif features[2] < 0.40:
            interpretation['insights'].append("Victim framing - portrays being acted upon")
        
        if features[6] > 0.60:  # Overall positive
            interpretation['insights'].append("OVERALL: Positive framing dominates")
        elif features[6] < 0.40:
            interpretation['insights'].append("OVERALL: Negative framing dominates")
        
        # Dominant metaphor
        metaphor_scores = list(features[8:12])
        if max(metaphor_scores) > 0.3:
            metaphor_names = ['war', 'journey', 'game', 'growth']
            dominant_metaphor = metaphor_names[np.argmax(metaphor_scores)]
            interpretation['insights'].append(f"Dominant metaphor: {dominant_metaphor.upper()}")
        
        return interpretation
    
    def _generate_summary(self, features: np.ndarray) -> str:
        """Generate plain English summary of framing."""
        gain_ratio = features[0]
        agent_ratio = features[2]
        optimism = features[3]
        valence = features[6]
        
        summary_parts = []
        
        # Overall tone
        if valence > 0.60:
            summary_parts.append("Positive framing")
        elif valence < 0.40:
            summary_parts.append("Negative framing")
        else:
            summary_parts.append("Balanced framing")
        
        # Gain/loss
        if gain_ratio > 0.60:
            summary_parts.append("emphasizing gains and benefits")
        elif gain_ratio < 0.40:
            summary_parts.append("emphasizing losses and risks")
        
        # Agency
        if agent_ratio > 0.60:
            summary_parts.append("with high agency/control")
        elif agent_ratio < 0.40:
            summary_parts.append("with low agency/victim positioning")
        
        # Outlook
        if optimism > 0.60:
            summary_parts.append("optimistic outlook")
        elif optimism < 0.40:
            summary_parts.append("pessimistic outlook")
        
        return ", ".join(summary_parts) + "."
    
    def _describe_feature(self, name: str, value: float) -> str:
        """Describe what a feature value means."""
        descriptions = {
            'gain_vs_loss_ratio': f"{'Gain-focused' if value > 0.6 else 'Loss-focused' if value < 0.4 else 'Balanced'} framing",
            'solution_vs_problem_ratio': f"{'Solution-oriented' if value > 0.6 else 'Problem-focused' if value < 0.4 else 'Balanced'} perspective",
            'agent_vs_victim_ratio': f"{'Agent' if value > 0.6 else 'Victim' if value < 0.4 else 'Mixed'} framing",
            'optimism_vs_pessimism': f"{'Optimistic' if value > 0.6 else 'Pessimistic' if value < 0.4 else 'Neutral'} outlook",
            'opportunity_vs_threat': f"{'Opportunity' if value > 0.6 else 'Threat' if value < 0.4 else 'Balanced'} framing",
            'control_vs_helplessness': f"{'High control' if value > 0.6 else 'Helpless' if value < 0.4 else 'Mixed'} positioning",
            'overall_positive_valence': f"{'Positive' if value > 0.6 else 'Negative' if value < 0.4 else 'Neutral'} overall tone",
            'reframing_density': f"{'High' if value > 2 else 'Moderate' if value > 0.5 else 'Low'} reframing frequency",
            'metaphor_war': f"{'High' if value > 0.4 else 'Some' if value > 0.1 else 'Minimal'} war metaphors",
            'metaphor_journey': f"{'High' if value > 0.4 else 'Some' if value > 0.1 else 'Minimal'} journey metaphors",
            'metaphor_game': f"{'High' if value > 0.4 else 'Some' if value > 0.1 else 'Minimal'} game metaphors",
            'metaphor_growth': f"{'High' if value > 0.4 else 'Some' if value > 0.1 else 'Minimal'} growth metaphors"
        }
        
        return descriptions.get(name, f"Value: {value:.2f}")

