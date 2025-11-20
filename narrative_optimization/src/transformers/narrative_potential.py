"""
Narrative Potential Transformer

Measures narrative openness, possibility, and developmental potential.
Tests whether future-oriented, possibility-rich narratives predict better outcomes.
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string
from .utils.input_validation import ensure_string_list, ensure_string


class NarrativePotentialTransformer(NarrativeTransformer):
    """
    Analyzes narrative potential, possibility space, AND stakes/urgency.
    
    NOW DOMAIN-ADAPTIVE: Adapts patterns based on domain configuration.
    
    Tests the hypothesis that narratives rich in possibility, open to change,
    and future-oriented predict better outcomes than closed, static narratives.
    Also captures stakes/urgency/criticality (Narrative Weighting Theory).
    
    Features extracted (35 total):
    - Future orientation (forward-looking language)
    - Possibility language (modal verbs, conditionals)
    - Growth mindset indicators
    - Narrative flexibility (openness to alternatives)
    - Developmental arc position (where in story are we?)
    - **NEW: Stakes/Urgency (10 features)**:
      * Urgency markers (critical, urgent, immediate)
      * High-stakes language (championship, must-win, decisive)
      * Consequence language (impact, ramifications)
      * Deadline/temporal pressure (running out, last chance)
      * Crisis language (emergency, catastrophe, dire)
      * Irreversibility markers (no turning back, permanent)
      * Opportunity stakes (once-in-lifetime, breakthrough)
      * Overall stakes intensity
      * Stakes valence (negative crisis vs positive opportunity)
      * Narrative weight estimator (0-3x multiplier)
    
    Parameters
    ----------
    domain_config : DomainConfig, optional
        Domain-specific configuration for adaptive patterns
    track_modality : bool
        Whether to track modal verbs and possibility language
    track_flexibility : bool
        Whether to measure narrative flexibility
    track_arc_position : bool
        Whether to infer developmental arc position
    """
    
    def __init__(
        self,
        domain_config = None,
        track_modality: bool = True,
        track_flexibility: bool = True,
        track_arc_position: bool = True
    ):
        super().__init__(
            narrative_id="narrative_potential",
            description="Narrative potential: openness, possibility, and developmental trajectory"
        )
        
        self.domain_config = domain_config
        self.track_modality = track_modality
        self.track_flexibility = track_flexibility
        self.track_arc_position = track_arc_position
        
        # Future orientation patterns
        self.future_tense = [r'\bwill\b', r'\bshall\b', r'\bgoing to\b', r'\bgonna\b']
        self.future_intentions = [r'\bplan to\b', r'\bintend to\b', r'\bhope to\b', r'\bexpect to\b', r'\baim to\b']
        
        # Possibility language
        self.possibility_modals = [r'\bcould\b', r'\bmight\b', r'\bmay\b', r'\bpossibly\b', r'\bperhaps\b', r'\bmaybe\b']
        self.potential_words = ['potential', 'possibility', 'opportunity', 'chance', 'option', 'alternative', 'choice']
        
        # Growth indicators
        self.growth_verbs = ['become', 'grow', 'develop', 'evolve', 'transform', 'progress', 'advance', 'improve', 'learn']
        self.change_words = ['change', 'shift', 'transition', 'move', 'adapt', 'adjust', 'modify', 'alter']
        
        # Flexibility indicators
        self.flexibility_words = ['flexible', 'adaptable', 'open', 'willing', 'ready', 'able', 'capable']
        self.rigidity_words = ['must', 'have to', 'need to', 'required', 'necessary', 'always', 'never', 'only']
        
        # Constraint vs possibility
        self.constraint_words = ['can\'t', 'cannot', 'couldn\'t', 'impossible', 'unable', 'limited', 'stuck', 'trapped']
        self.possibility_words = ['can', 'able', 'possible', 'feasible', 'achievable', 'attainable', 'reachable']
        
        # Developmental arc indicators
        self.beginning_markers = ['start', 'begin', 'first', 'new', 'initial', 'early', 'beginning', 'launch']
        self.middle_markers = ['currently', 'now', 'present', 'ongoing', 'during', 'while', 'process', 'developing']
        self.resolution_markers = ['complete', 'finish', 'end', 'final', 'conclusion', 'result', 'outcome', 'achieved']
        
        # **NEW: STAKES / URGENCY / CRITICALITY** (Narrative Weighting Theory)
        # Base patterns (domain-agnostic)
        base_urgency = ['urgent', 'critical', 'crucial', 'immediate', 'now', 'emergency', 'pressing', 'imperative', 'vital']
        base_high_stakes = ['championship', 'finals', 'playoff', 'elimination', 'must-win', 'do-or-die', 'decisive', 'pivotal', 'life-or-death', 'critical', 'make-or-break']
        base_consequence = ['consequence', 'result', 'outcome', 'impact', 'effect', 'ramifications', 'implications', 'repercussions']
        base_deadline = ['deadline', 'time', 'running out', 'limited time', 'countdown', 'last chance', 'final opportunity', 'now or never']
        base_crisis = ['crisis', 'disaster', 'catastrophe', 'emergency', 'dire', 'severe', 'critical situation', 'breaking point']
        base_irreversible = ['irreversible', 'permanent', 'final', 'no turning back', 'point of no return', 'once and for all', 'forever']
        base_opportunity = ['once-in-a-lifetime', 'historic', 'breakthrough', 'game-changer', 'defining moment', 'turning point']
        
        # Domain-specific adaptations if config provided
        if domain_config:
            # Get domain-specific patterns from config if available
            domain_potential = domain_config.get_domain_specific_patterns('potential')
            if domain_potential:
                # Extend base patterns with domain-specific ones
                base_high_stakes.extend(domain_potential)
        
        self.urgency_markers = base_urgency
        self.high_stakes = base_high_stakes
        self.consequence_language = base_consequence
        self.deadline_language = base_deadline
        self.crisis_language = base_crisis
        self.irreversible_language = base_irreversible
        self.opportunity_stakes = base_opportunity
    
    def fit(self, X, y=None):
        """
        Learn narrative potential patterns from corpus.
        
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
        
        # Corpus statistics
        corpus_stats = {
            'avg_future_orientation': 0,
            'avg_possibility': 0,
            'avg_growth_language': 0,
            'avg_flexibility': 0
        }
        
        for text in X:
            text_lower = text.lower()
            words = text_lower.split()
            n_words = len(words) + 1
            
            # Future orientation
            future_count = sum(len(re.findall(p, text_lower)) for p in self.future_tense + self.future_intentions)
            corpus_stats['avg_future_orientation'] += future_count / n_words
            
            # Possibility
            possibility_count = sum(len(re.findall(p, text_lower)) for p in self.possibility_modals)
            possibility_count += sum(1 for word in self.potential_words if word in text_lower)
            corpus_stats['avg_possibility'] += possibility_count / n_words
            
            # Growth
            growth_count = sum(1 for word in self.growth_verbs + self.change_words if word in text_lower)
            corpus_stats['avg_growth_language'] += growth_count / n_words
            
            # Flexibility
            flex_count = sum(1 for word in self.flexibility_words if word in text_lower)
            corpus_stats['avg_flexibility'] += flex_count / n_words
        
        # Average
        n_docs = len(X)
        for key in corpus_stats:
            corpus_stats[key] /= n_docs
        
        self.metadata['corpus_stats'] = corpus_stats
        self.metadata['n_documents'] = n_docs
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to narrative potential features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array
            Narrative potential feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_potential_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_potential_features(self, text: str) -> List[float]:
        """Extract narrative potential features from a document."""
        features = []
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) + 1
        
        # 1. Future Orientation
        future_tense_count = sum(len(re.findall(p, text_lower)) for p in self.future_tense)
        future_intention_count = sum(len(re.findall(p, text_lower)) for p in self.future_intentions)
        
        features.append(future_tense_count / n_words)  # Future tense density
        features.append(future_intention_count / n_words)  # Intentional future
        
        # Overall future orientation
        future_orientation = (future_tense_count + future_intention_count) / n_words
        features.append(future_orientation)
        
        # 2. Possibility Language (Modality)
        if self.track_modality:
            possibility_modal_count = sum(len(re.findall(p, text_lower)) for p in self.possibility_modals)
            potential_word_count = sum(1 for word in self.potential_words if word in text_lower)
            
            features.append(possibility_modal_count / n_words)  # Modal density
            features.append(potential_word_count / n_words)  # Potential word density
            
            # Possibility score
            possibility_score = (possibility_modal_count + potential_word_count) / n_words
            features.append(possibility_score)
        
        # 3. Growth & Change Language
        growth_count = sum(1 for word in self.growth_verbs if word in text_lower)
        change_count = sum(1 for word in self.change_words if word in text_lower)
        
        features.append(growth_count / n_words)  # Growth verb density
        features.append(change_count / n_words)  # Change word density
        
        # Growth mindset indicator
        growth_mindset = (growth_count + change_count) / n_words
        features.append(growth_mindset)
        
        # 4. Narrative Flexibility
        if self.track_flexibility:
            flex_count = sum(1 for word in self.flexibility_words if word in text_lower)
            rigid_count = sum(1 for word in self.rigidity_words if word in text_lower)
            
            features.append(flex_count / n_words)  # Flexibility language
            features.append(rigid_count / n_words)  # Rigidity language
            
            # Flexibility ratio
            flexibility_ratio = flex_count / (flex_count + rigid_count + 1)
            features.append(flexibility_ratio)
        
        # 5. Possibility vs Constraint
        constraint_count = sum(1 for word in self.constraint_words if word in text_lower)
        poss_word_count = sum(1 for word in self.possibility_words if word in text_lower)
        
        features.append(poss_word_count / n_words)  # Possibility words
        features.append(constraint_count / n_words)  # Constraint words
        
        # Net possibility (possibility - constraint)
        net_possibility = (poss_word_count - constraint_count) / n_words
        features.append(net_possibility)
        
        # 6. Developmental Arc Position
        if self.track_arc_position:
            beginning_count = sum(1 for word in self.beginning_markers if word in text_lower)
            middle_count = sum(1 for word in self.middle_markers if word in text_lower)
            resolution_count = sum(1 for word in self.resolution_markers if word in text_lower)
            
            total_arc_markers = beginning_count + middle_count + resolution_count + 1
            
            # Normalized arc position scores
            features.append(beginning_count / total_arc_markers)  # Beginning phase
            features.append(middle_count / total_arc_markers)  # Middle/development phase
            features.append(resolution_count / total_arc_markers)  # Resolution phase
            
            # Dominant arc position (which phase is strongest)
            arc_scores = [beginning_count, middle_count, resolution_count]
            dominant_arc = np.argmax(arc_scores) / 2.0  # 0, 0.5, or 1.0
            features.append(dominant_arc)
        
        # 7. Openness to Alternatives
        # Measured by conditional language and alternative expressions
        conditional_patterns = [r'\bif\b', r'\bunless\b', r'\bwhether\b', r'\beither\b', r'\bor\b']
        conditional_count = sum(len(re.findall(p, text_lower)) for p in conditional_patterns)
        
        features.append(conditional_count / n_words)  # Conditional language
        
        # Alternative language
        alternative_words = ['alternative', 'option', 'choice', 'different', 'another', 'other', 'instead', 'rather']
        alternative_count = sum(1 for word in alternative_words if word in text_lower)
        features.append(alternative_count / n_words)
        
        # Openness score (conditionals + alternatives)
        openness_score = (conditional_count + alternative_count) / n_words
        features.append(openness_score)
        
        # 8. Temporal Breadth
        # How much temporal range does the narrative cover?
        past_markers = ['was', 'were', 'had', 'did', 'ago', 'before', 'previously', 'earlier']
        present_markers = ['is', 'are', 'am', 'now', 'currently', 'today', 'present']
        future_markers = ['will', 'shall', 'going to', 'soon', 'later', 'future', 'next']
        
        past_count = sum(1 for word in past_markers if word in text_lower)
        present_count = sum(1 for word in present_markers if word in text_lower)
        future_markers_count = sum(1 for word in future_markers if word in text_lower)
        
        # Temporal breadth (how many time periods covered)
        time_periods_covered = sum([
            past_count > 0,
            present_count > 0,
            future_markers_count > 0
        ])
        features.append(time_periods_covered / 3.0)  # Normalized 0-1
        
        # 9. Potential Actualization Language
        # Language about turning potential into reality
        actualization_patterns = [r'\bmake\s+it\s+happen\b', r'\btake\s+action\b', r'\bdo\s+it\b', r'\bactually\b', r'\brealize\b']
        actualization_count = sum(len(re.findall(p, text_lower)) for p in actualization_patterns)
        features.append(actualization_count / n_words)
        
        # 10. Narrative Momentum
        # Is the narrative moving forward?
        forward_words = ['forward', 'ahead', 'progress', 'advance', 'next', 'continue', 'onward', 'moving']
        backward_words = ['back', 'return', 'regress', 'retreat', 'backward', 'previous', 'again']
        
        forward_count = sum(1 for word in forward_words if word in text_lower)
        backward_count = sum(1 for word in backward_words if word in text_lower)
        
        # Forward momentum score
        momentum = (forward_count - backward_count) / n_words
        features.append(momentum)
        
        # **NEW: 11-20. STAKES / URGENCY / CRITICALITY FEATURES**
        
        # 11. Urgency marker density
        urgency_count = sum(1 for word in self.urgency_markers if word in text_lower)
        features.append(urgency_count / n_words)
        
        # 12. High-stakes language density
        high_stakes_count = sum(1 for word in self.high_stakes if word in text_lower)
        features.append(high_stakes_count / n_words)
        
        # 13. Consequence language density
        consequence_count = sum(1 for word in self.consequence_language if word in text_lower)
        features.append(consequence_count / n_words)
        
        # 14. Deadline/temporal pressure
        deadline_count = sum(1 for word in self.deadline_language if word in text_lower)
        features.append(deadline_count / n_words)
        
        # 15. Crisis language density
        crisis_count = sum(1 for word in self.crisis_language if word in text_lower)
        features.append(crisis_count / n_words)
        
        # 16. Irreversibility markers
        irreversible_count = sum(1 for word in self.irreversible_language if word in text_lower)
        features.append(irreversible_count / n_words)
        
        # 17. Opportunity stakes (positive high-stakes)
        opportunity_count = sum(1 for word in self.opportunity_stakes if word in text_lower)
        features.append(opportunity_count / n_words)
        
        # 18. Overall stakes intensity (all stakes language combined)
        total_stakes = (urgency_count + high_stakes_count + consequence_count + 
                       deadline_count + crisis_count + irreversible_count + opportunity_count)
        features.append(total_stakes / n_words)
        
        # 19. Stakes type ratio (negative crisis vs positive opportunity)
        # Negative = urgency + crisis + deadline + irreversible
        # Positive = opportunity stakes
        negative_stakes = urgency_count + crisis_count + deadline_count + irreversible_count
        total_typed_stakes = negative_stakes + opportunity_count + 1
        stakes_valence = opportunity_count / total_typed_stakes
        features.append(stakes_valence)  # 0 = all negative, 1 = all positive, 0.5 = balanced
        
        # 20. Narrative weight estimator (proxy for context weighting)
        # Combines high-stakes + urgency + crisis for 0-3x multiplier estimate
        narrative_weight = min(3.0, (high_stakes_count * 2 + urgency_count + crisis_count) / 10.0)
        features.append(narrative_weight)
        
        return features
    
    def _generate_interpretation(self):
        """Generate human-readable interpretation."""
        corpus_stats = self.metadata.get('corpus_stats', {})
        
        interpretation = (
            "Narrative Potential & Stakes Analysis: Measures openness, possibility, developmental trajectory, "
            "AND narrative stakes/urgency/criticality. "
            f"Corpus averages - future orientation: {corpus_stats.get('avg_future_orientation', 0):.3f}, "
            f"possibility language: {corpus_stats.get('avg_possibility', 0):.3f}, "
            f"growth language: {corpus_stats.get('avg_growth_language', 0):.3f}. "
            "Features capture (35 total): future orientation (tense + intentions), possibility language "
            "(modals + potential words), growth & change language, narrative flexibility "
            "(vs rigidity), possibility vs constraint balance, developmental arc position "
            "(beginning/middle/resolution), openness to alternatives (conditionals), "
            "temporal breadth (time periods covered), actualization language (potential → reality), "
            "narrative momentum (forward vs backward), "
            "**PLUS stakes/urgency (urgency markers, high-stakes language, consequence language, "
            "deadline/temporal pressure, crisis language, irreversibility markers, opportunity stakes, "
            "overall stakes intensity, stakes valence, and narrative weight estimator 0-3x)**. "
            "Validates: (1) possibility-rich, future-oriented narratives with growth potential "
            "perform better than closed, static narratives, AND (2) high-stakes contexts amplify "
            "narrative effects (Narrative Weighting Theory)."
        )
        
        return interpretation

