"""
Narrativity Spectrum Framework

Formalizes the degree of narrative openness/constraint across domains.

Spectrum ranges from:
- 0.0: Totally circumscribed (die roll, physical laws)
- 1.0: Totally open (diary entry, creative writing)

Most domains fall between, with specific structural constraints that:
1. Limit possible narrative paths (domain constraints)
2. Create probabilistic tendencies (domain features)
3. Define narrator-narrated relationships (who tells what)
4. Establish superficial narrative potential (apparent freedom before unfolding)

This spectrum is MORE FUNDAMENTAL than α parameter:
- Narrativity measures: How much CAN the narrative vary?
- α parameter measures: Which narrative dimensions matter most?

Example:
- Die roll: Narrativity = 0.05 (outcome fully constrained by physics)
- Startup pitch: Narrativity = 0.65 (constrained by market, but creative freedom in framing)
- Diary entry: Narrativity = 0.95 (almost no constraints, person writes whatever)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class NarratorType(Enum):
    """Who tells the story."""
    OBJECTIVE_EXTERNAL = "objective"  # Physics, math (no narrator)
    PARTICIPANT = "participant"  # Actor in the scene narrates
    OBSERVER = "observer"  # Witness narrates
    OMNISCIENT = "omniscient"  # God's-eye view
    COLLECTIVE = "collective"  # Multiple narrators
    ALGORITHMIC = "algorithmic"  # System generates narrative


class NarratedType(Enum):
    """What is being narrated."""
    PHYSICAL_EVENT = "physical"  # Dice roll, chemical reaction
    PERFORMANCE = "performance"  # Game, execution, measurable
    INTERACTION = "interaction"  # Relationship, conversation
    INTERNAL_STATE = "internal"  # Thoughts, feelings, perceptions
    CREATION = "creation"  # Making something new
    TRAJECTORY = "trajectory"  # Life path, company growth


@dataclass
class NarrativeConstraint:
    """A structural constraint that limits narrative possibilities."""
    constraint_type: str  # 'physical', 'format', 'social', 'temporal', 'resource'
    strength: float  # 0-1, how strongly it constrains
    description: str
    examples: List[str]


@dataclass
class NarrativityMeasure:
    """Complete measure of domain's narrativity."""
    domain_name: str
    narrativity_score: float  # 0-1 (0=circumscribed, 1=open)
    
    # Components
    structural_openness: float  # How many possible paths?
    temporal_freedom: float  # Can narrative unfold over time?
    actor_agency: float  # How much choice do actors have?
    observer_interpretation: float  # How much interpretation possible?
    format_flexibility: float  # How flexible is the medium?
    
    # Narrative elements
    narrator_type: NarratorType
    narrated_type: NarratedType
    narrator_narrated_coupling: float  # How tightly coupled?
    
    # Constraints
    constraints: List[NarrativeConstraint]
    degrees_of_freedom: int  # How many narrative dimensions can vary?
    
    # Relationships
    superficial_potential: float  # Apparent freedom before unfolding
    actual_potential: float  # Real freedom after constraints applied
    potential_gap: float  # Difference (illusion of freedom)
    
    # Derived insights
    alpha_prediction: float  # Predicted α parameter
    narrative_features_prediction: List[str]  # Which features should matter
    
    def __repr__(self):
        return (
            f"{self.domain_name}: Narrativity={self.narrativity_score:.2f}, "
            f"α={self.alpha_prediction:.2f}, {self.narrator_type.value}→{self.narrated_type.value}"
        )


class NarrivityAnalyzer:
    """
    Analyzes and measures narrativity across the spectrum.
    
    Provides formal framework for understanding where domains fall
    from totally circumscribed to totally open.
    """
    
    def __init__(self):
        self.domain_narrativity_scores = {}
    
    def analyze_domain_narrativity(
        self,
        domain_name: str,
        domain_description: str,
        example_instances: Optional[List[str]] = None
    ) -> NarrativityMeasure:
        """
        Measure domain's position on narrativity spectrum.
        
        Parameters
        ----------
        domain_name : str
        domain_description : str
            Description of domain mechanics, constraints, format
        example_instances : List[str], optional
            Example instances from domain
            
        Returns
        -------
        NarrativityMeasure with complete analysis
        """
        # Extract components
        structural_openness = self._measure_structural_openness(domain_description)
        temporal_freedom = self._measure_temporal_freedom(domain_description)
        actor_agency = self._measure_actor_agency(domain_description)
        observer_interpretation = self._measure_observer_interpretation(domain_description)
        format_flexibility = self._measure_format_flexibility(domain_description)
        
        # Identify narrator and narrated types
        narrator_type = self._identify_narrator(domain_description)
        narrated_type = self._identify_narrated(domain_description)
        coupling = self._measure_narrator_narrated_coupling(narrator_type, narrated_type, domain_description)
        
        # Extract constraints
        constraints = self._extract_constraints(domain_description)
        degrees_of_freedom = self._count_degrees_of_freedom(constraints, structural_openness)
        
        # Calculate potentials
        superficial_potential = self._calculate_superficial_potential(
            structural_openness, format_flexibility
        )
        actual_potential = self._calculate_actual_potential(
            constraints, degrees_of_freedom
        )
        potential_gap = superficial_potential - actual_potential
        
        # Overall narrativity (weighted average of components)
        narrativity_score = self._calculate_narrativity_score(
            structural_openness,
            temporal_freedom,
            actor_agency,
            observer_interpretation,
            format_flexibility
        )
        
        # Predict α parameter from narrativity
        alpha_prediction = self._predict_alpha_from_narrativity(
            narrativity_score, narrator_type, narrated_type
        )
        
        # Predict which narrative features matter
        feature_prediction = self._predict_narrative_features(
            narrativity_score, alpha_prediction, narrated_type
        )
        
        measure = NarrativityMeasure(
            domain_name=domain_name,
            narrativity_score=narrativity_score,
            structural_openness=structural_openness,
            temporal_freedom=temporal_freedom,
            actor_agency=actor_agency,
            observer_interpretation=observer_interpretation,
            format_flexibility=format_flexibility,
            narrator_type=narrator_type,
            narrated_type=narrated_type,
            narrator_narrated_coupling=coupling,
            constraints=constraints,
            degrees_of_freedom=degrees_of_freedom,
            superficial_potential=superficial_potential,
            actual_potential=actual_potential,
            potential_gap=potential_gap,
            alpha_prediction=alpha_prediction,
            narrative_features_prediction=feature_prediction
        )
        
        self.domain_narrativity_scores[domain_name] = measure
        return measure
    
    def _measure_structural_openness(self, description: str) -> float:
        """
        How many possible narrative paths exist?
        
        0.0 = One outcome (die has 6 faces, fully enumerable)
        1.0 = Infinite outcomes (diary can go anywhere)
        """
        desc_lower = description.lower()
        
        # Indicators of constraint
        constraint_words = ['rule', 'must', 'law', 'fixed', 'determined', 'physics', 'one', 'specific']
        constraint_score = sum(1 for w in constraint_words if w in desc_lower) / len(constraint_words)
        
        # Indicators of openness
        openness_words = ['creative', 'free', 'choice', 'flexible', 'open', 'varied', 'diverse', 'any']
        openness_score = sum(1 for w in openness_words if w in desc_lower) / len(openness_words)
        
        # Calculate openness (inverse of constraints)
        if constraint_score + openness_score > 0:
            structural_openness = openness_score / (constraint_score + openness_score + 0.1)
        else:
            structural_openness = 0.5  # Default to moderate
        
        return np.clip(structural_openness, 0.0, 1.0)
    
    def _measure_temporal_freedom(self, description: str) -> float:
        """
        Can narrative unfold over time?
        
        0.0 = Single moment (die roll instant)
        1.0 = Extended duration (life story, company growth)
        """
        desc_lower = description.lower()
        
        # Temporal extent indicators
        instant_words = ['instant', 'moment', 'single', 'one-time', 'immediate']
        extended_words = ['over time', 'years', 'growth', 'evolution', 'journey', 'trajectory']
        
        instant_score = sum(1 for w in instant_words if w in desc_lower)
        extended_score = sum(1 for w in extended_words if w in desc_lower)
        
        if instant_score + extended_score > 0:
            temporal_freedom = extended_score / (instant_score + extended_score)
        else:
            temporal_freedom = 0.5
        
        return np.clip(temporal_freedom, 0.0, 1.0)
    
    def _measure_actor_agency(self, description: str) -> float:
        """
        How much choice do actors have?
        
        0.0 = No agency (physics determines all)
        1.0 = Complete agency (can do anything)
        """
        desc_lower = description.lower()
        
        # Agency indicators
        no_agency_words = ['determined', 'fixed', 'automatic', 'physics', 'law', 'random']
        high_agency_words = ['choose', 'decide', 'create', 'control', 'agency', 'freedom']
        
        no_agency = sum(1 for w in no_agency_words if w in desc_lower)
        high_agency = sum(1 for w in high_agency_words if w in desc_lower)
        
        if no_agency + high_agency > 0:
            actor_agency = high_agency / (no_agency + high_agency)
        else:
            actor_agency = 0.5
        
        return np.clip(actor_agency, 0.0, 1.0)
    
    def _measure_observer_interpretation(self, description: str) -> float:
        """
        How much can interpretation vary?
        
        0.0 = Objective (everyone sees same thing)
        1.0 = Subjective (each observer constructs different story)
        """
        desc_lower = description.lower()
        
        # Objectivity indicators
        objective_words = ['objective', 'measurable', 'data', 'fact', 'number', 'physics']
        subjective_words = ['perception', 'interpretation', 'meaning', 'story', 'narrative', 'experience']
        
        objective = sum(1 for w in objective_words if w in desc_lower)
        subjective = sum(1 for w in subjective_words if w in desc_lower)
        
        if objective + subjective > 0:
            interpretation = subjective / (objective + subjective)
        else:
            interpretation = 0.5
        
        return np.clip(interpretation, 0.0, 1.0)
    
    def _measure_format_flexibility(self, description: str) -> float:
        """
        How flexible is the narrative format/medium?
        
        0.0 = Rigid format (6-sided die always 6 outcomes)
        1.0 = Flexible format (diary can be any length, style, content)
        """
        desc_lower = description.lower()
        
        # Format constraint indicators
        rigid_words = ['format', 'structure', 'standard', 'protocol', 'template', 'fixed']
        flexible_words = ['flexible', 'varied', 'creative', 'open', 'freeform']
        
        rigid = sum(1 for w in rigid_words if w in desc_lower)
        flexible = sum(1 for w in flexible_words if w in desc_lower)
        
        if rigid + flexible > 0:
            flexibility = flexible / (rigid + flexible)
        else:
            flexibility = 0.5
        
        return np.clip(flexibility, 0.0, 1.0)
    
    def _identify_narrator(self, description: str) -> NarratorType:
        """Identify who narrates."""
        desc_lower = description.lower()
        
        if any(w in desc_lower for w in ['physics', 'law', 'determined', 'automatic']):
            return NarratorType.OBJECTIVE_EXTERNAL
        elif any(w in desc_lower for w in ['participant', 'actor', 'player']):
            return NarratorType.PARTICIPANT
        elif any(w in desc_lower for w in ['observer', 'witness', 'evaluator']):
            return NarratorType.OBSERVER
        elif any(w in desc_lower for w in ['god', 'omniscient', 'complete']):
            return NarratorType.OMNISCIENT
        elif any(w in desc_lower for w in ['multiple', 'collective', 'team', 'ensemble']):
            return NarratorType.COLLECTIVE
        elif any(w in desc_lower for w in ['system', 'algorithm', 'data']):
            return NarratorType.ALGORITHMIC
        else:
            return NarratorType.PARTICIPANT  # Default
    
    def _identify_narrated(self, description: str) -> NarratedType:
        """Identify what is being narrated."""
        desc_lower = description.lower()
        
        if any(w in desc_lower for w in ['physics', 'chemical', 'physical', 'mechanical']):
            return NarratedType.PHYSICAL_EVENT
        elif any(w in desc_lower for w in ['game', 'competition', 'performance', 'execute']):
            return NarratedType.PERFORMANCE
        elif any(w in desc_lower for w in ['relationship', 'conversation', 'interaction', 'connect']):
            return NarratedType.INTERACTION
        elif any(w in desc_lower for w in ['thought', 'feeling', 'perception', 'experience', 'identity']):
            return NarratedType.INTERNAL_STATE
        elif any(w in desc_lower for w in ['create', 'build', 'design', 'produce']):
            return NarratedType.CREATION
        elif any(w in desc_lower for w in ['growth', 'journey', 'development', 'evolution']):
            return NarratedType.TRAJECTORY
        else:
            return NarratedType.PERFORMANCE  # Default
    
    def _measure_narrator_narrated_coupling(
        self, narrator: NarratorType, narrated: NarratedType, description: str
    ) -> float:
        """
        How tightly coupled are narrator and narrated?
        
        0.0 = Completely separate (objective narrator of physical events)
        1.0 = Identical (participant narrating their own internal state)
        """
        # Coupling matrix
        coupling_matrix = {
            (NarratorType.OBJECTIVE_EXTERNAL, NarratedType.PHYSICAL_EVENT): 0.0,
            (NarratorType.OBJECTIVE_EXTERNAL, NarratedType.PERFORMANCE): 0.2,
            (NarratorType.PARTICIPANT, NarratedType.INTERNAL_STATE): 1.0,
            (NarratorType.PARTICIPANT, NarratedType.PERFORMANCE): 0.7,
            (NarratorType.PARTICIPANT, NarratedType.INTERACTION): 0.8,
            (NarratorType.OBSERVER, NarratedType.INTERACTION): 0.4,
            (NarratorType.COLLECTIVE, NarratedType.INTERACTION): 0.9,
        }
        
        coupling = coupling_matrix.get((narrator, narrated), 0.5)
        return coupling
    
    def _extract_constraints(self, description: str) -> List[NarrativeConstraint]:
        """Extract structural constraints from description."""
        constraints = []
        desc_lower = description.lower()
        
        # Physical constraints
        if any(w in desc_lower for w in ['physics', 'law', 'determined']):
            constraints.append(NarrativeConstraint(
                constraint_type='physical',
                strength=0.9,
                description='Physical laws determine outcomes',
                examples=['gravity', 'thermodynamics', 'probability']
            ))
        
        # Format constraints
        if any(w in desc_lower for w in ['format', 'structure', 'standard']):
            constraints.append(NarrativeConstraint(
                constraint_type='format',
                strength=0.6,
                description='Format limits expression',
                examples=['pitch deck', 'resume', 'form']
            ))
        
        # Temporal constraints
        if any(w in desc_lower for w in ['time limit', 'duration', 'deadline']):
            constraints.append(NarrativeConstraint(
                constraint_type='temporal',
                strength=0.5,
                description='Time limits narrative development',
                examples=['2-minute pitch', '48-minute game']
            ))
        
        # Social constraints
        if any(w in desc_lower for w in ['social', 'norm', 'expectation', 'convention']):
            constraints.append(NarrativeConstraint(
                constraint_type='social',
                strength=0.4,
                description='Social norms constrain narrative',
                examples=['professional behavior', 'genre conventions']
            ))
        
        # Resource constraints
        if any(w in desc_lower for w in ['resource', 'budget', 'capacity', 'limit']):
            constraints.append(NarrativeConstraint(
                constraint_type='resource',
                strength=0.5,
                description='Resources limit possibilities',
                examples=['budget', 'team size', 'time']
            ))
        
        return constraints
    
    def _count_degrees_of_freedom(
        self, constraints: List[NarrativeConstraint], openness: float
    ) -> int:
        """
        Count narrative dimensions that can vary.
        
        Rough estimate based on constraints and openness.
        """
        # Base degrees from openness
        base_degrees = int(openness * 100)
        
        # Reduce by constraints
        for constraint in constraints:
            reduction = int(constraint.strength * 20)
            base_degrees -= reduction
        
        return max(1, base_degrees)
    
    def _calculate_superficial_potential(
        self, structural_openness: float, format_flexibility: float
    ) -> float:
        """
        Apparent narrative freedom before constraints manifest.
        
        High when things SEEM open but are actually constrained.
        """
        return (structural_openness + format_flexibility) / 2
    
    def _calculate_actual_potential(
        self, constraints: List[NarrativeConstraint], dof: int
    ) -> float:
        """
        Real narrative freedom after all constraints applied.
        
        Lower when constraints are strong.
        """
        if not constraints:
            return dof / 100
        
        # Reduce by constraint strength
        constraint_factor = 1.0 - np.mean([c.strength for c in constraints])
        actual = (dof / 100) * constraint_factor
        
        return np.clip(actual, 0.0, 1.0)
    
    def _calculate_narrativity_score(
        self,
        structural_openness: float,
        temporal_freedom: float,
        actor_agency: float,
        observer_interpretation: float,
        format_flexibility: float
    ) -> float:
        """
        Overall narrativity score (weighted average).
        
        Weights reflect importance of each dimension.
        """
        weights = {
            'structural': 0.30,  # Most important
            'temporal': 0.20,
            'agency': 0.25,  # Very important
            'interpretation': 0.15,
            'format': 0.10
        }
        
        score = (
            structural_openness * weights['structural'] +
            temporal_freedom * weights['temporal'] +
            actor_agency * weights['agency'] +
            observer_interpretation * weights['interpretation'] +
            format_flexibility * weights['format']
        )
        
        return np.clip(score, 0.0, 1.0)
    
    def _predict_alpha_from_narrativity(
        self,
        narrativity: float,
        narrator: NarratorType,
        narrated: NarratedType
    ) -> float:
        """
        Predict α parameter from narrativity.
        
        Hypothesis: High narrativity → Low α (character-driven)
                   Low narrativity → High α (plot-driven)
        
        Reasoning: When there's little narrative freedom,
        content/facts matter most (plot). When there's high freedom,
        how things are described matters more (character).
        """
        # Base prediction (inverse relationship)
        base_alpha = 1.0 - narrativity
        
        # Adjust based on narrator-narrated pairing
        if narrated == NarratedType.PHYSICAL_EVENT:
            base_alpha = min(1.0, base_alpha + 0.2)  # Physical → more plot
        elif narrated == NarratedType.INTERNAL_STATE:
            base_alpha = max(0.0, base_alpha - 0.2)  # Internal → more character
        
        if narrator == NarratorType.OBJECTIVE_EXTERNAL:
            base_alpha = min(1.0, base_alpha + 0.15)  # Objective → more plot
        elif narrator == NarratorType.PARTICIPANT:
            base_alpha = max(0.0, base_alpha - 0.10)  # Participant → more character
        
        return np.clip(base_alpha, 0.0, 1.0)
    
    def _predict_narrative_features(
        self,
        narrativity: float,
        alpha: float,
        narrated_type: NarratedType
    ) -> List[str]:
        """Predict which narrative features should matter."""
        features = []
        
        # High narrativity (character-driven)
        if narrativity > 0.7:
            features.extend(['nominative', 'self_perception', 'narrative_potential'])
        
        # Moderate narrativity (hybrid)
        elif narrativity > 0.4:
            features.extend(['ensemble', 'relational', 'linguistic'])
        
        # Low narrativity (plot-driven)
        else:
            features.extend(['statistical', 'quantitative', 'performance'])
        
        # Refine by narrated type
        if narrated_type == NarratedType.INTERACTION:
            features.append('relational')
        elif narrated_type == NarratedType.CREATION:
            features.append('innovation')
        elif narrated_type == NarratedType.INTERNAL_STATE:
            features.append('self_perception')
        
        return list(set(features))  # Remove duplicates
    
    def place_on_spectrum(
        self,
        domains: Dict[str, NarrativityMeasure]
    ) -> Dict[str, any]:
        """
        Place all domains on the narrativity spectrum.
        
        Returns visualization data and insights.
        """
        if not domains:
            return {}
        
        # Sort by narrativity
        sorted_domains = sorted(
            domains.items(),
            key=lambda x: x[1].narrativity_score
        )
        
        spectrum = {
            'domains': [
                {
                    'name': name,
                    'narrativity': float(measure.narrativity_score),
                    'alpha': float(measure.alpha_prediction),
                    'narrator': measure.narrator_type.value,
                    'narrated': measure.narrated_type.value
                }
                for name, measure in sorted_domains
            ],
            'insights': self._generate_spectrum_insights(sorted_domains)
        }
        
        return spectrum
    
    def _generate_spectrum_insights(
        self, sorted_domains: List[Tuple[str, NarrativityMeasure]]
    ) -> List[str]:
        """Generate insights about the spectrum."""
        insights = []
        
        if len(sorted_domains) < 2:
            return insights
        
        # Identify clusters
        narrativity_values = [m.narrativity_score for _, m in sorted_domains]
        
        # Low narrativity cluster (< 0.3)
        low_narr = [name for name, m in sorted_domains if m.narrativity_score < 0.3]
        if low_narr:
            insights.append(
                f"Low narrativity (circumscribed): {', '.join(low_narr)} - "
                f"These domains are highly constrained, plot/content dominates."
            )
        
        # High narrativity cluster (> 0.7)
        high_narr = [name for name, m in sorted_domains if m.narrativity_score > 0.7]
        if high_narr:
            insights.append(
                f"High narrativity (open): {', '.join(high_narr)} - "
                f"These domains are open-ended, character/interpretation dominates."
            )
        
        # Correlation with α
        alphas = [m.alpha_prediction for _, m in sorted_domains]
        narr_values = [m.narrativity_score for _, m in sorted_domains]
        
        if len(alphas) >= 3:
            from scipy import stats
            corr, p = stats.pearsonr(narr_values, alphas)
            insights.append(
                f"Narrativity-α correlation: r={corr:.3f} - "
                f"{'Confirms inverse relationship' if corr < -0.5 else 'Relationship unclear'}"
            )
        
        return insights
    
    def generate_report(self, measure: NarrativityMeasure) -> str:
        """Generate narrativity analysis report."""
        report = []
        report.append("=" * 80)
        report.append(f"NARRATIVITY ANALYSIS: {measure.domain_name.upper()}")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"OVERALL NARRATIVITY: {measure.narrativity_score:.3f}")
        report.append(f"  (0.0 = circumscribed like die roll, 1.0 = open like diary)")
        report.append("")
        
        report.append("COMPONENT SCORES:")
        report.append(f"  Structural openness: {measure.structural_openness:.3f}")
        report.append(f"  Temporal freedom: {measure.temporal_freedom:.3f}")
        report.append(f"  Actor agency: {measure.actor_agency:.3f}")
        report.append(f"  Observer interpretation: {measure.observer_interpretation:.3f}")
        report.append(f"  Format flexibility: {measure.format_flexibility:.3f}")
        report.append("")
        
        report.append("NARRATIVE ELEMENTS:")
        report.append(f"  Narrator: {measure.narrator_type.value}")
        report.append(f"  Narrated: {measure.narrated_type.value}")
        report.append(f"  Coupling: {measure.narrator_narrated_coupling:.3f}")
        report.append("")
        
        report.append("CONSTRAINTS:")
        for constraint in measure.constraints:
            report.append(f"  - {constraint.constraint_type}: {constraint.strength:.2f} - {constraint.description}")
        report.append("")
        
        report.append("NARRATIVE POTENTIAL:")
        report.append(f"  Superficial (apparent): {measure.superficial_potential:.3f}")
        report.append(f"  Actual (real): {measure.actual_potential:.3f}")
        report.append(f"  Gap (illusion): {measure.potential_gap:.3f}")
        report.append(f"  Degrees of freedom: {measure.degrees_of_freedom}")
        report.append("")
        
        report.append("PREDICTIONS:")
        report.append(f"  α parameter: {measure.alpha_prediction:.3f}")
        report.append(f"  Key features: {', '.join(measure.narrative_features_prediction)}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def demo_narrativity_spectrum():
    """Demonstrate narrativity analysis across the spectrum."""
    analyzer = NarrivityAnalyzer()
    
    # Analyze domains across the spectrum
    domains = {
        'die_roll': """
            Rolling a standard 6-sided die. Outcome determined by physics (gravity, momentum, friction).
            Six possible outcomes, fully enumerable. No actor choice once released. Objective observation.
            Instant event. No interpretation - number that shows is the result.
        """,
        
        'basketball_game': """
            Team sport with 5 players per side competing in 48-minute game. Rules constrain play
            (dribbling, fouls, shot clock). Players have agency within rules. Performance measured
            objectively (score) but style and narrative emerge. Game unfolds over time with momentum shifts.
        """,
        
        'startup_pitch': """
            Company founders describe their product to investors. Format partially constrained
            (pitch deck structure, time limit). Creative freedom in framing problem-solution.
            Participant narration (founders tell their story). Outcomes measured (funding) but
            narrative matters. Must follow business norms but flexible within them.
        """,
        
        'therapy_session': """
            Patient discusses thoughts and feelings with therapist. Highly open format - patient
            can talk about anything. Internal states being narrated by participant. Therapist
            observes and interprets. No fixed structure beyond time limit. High interpretive freedom.
            Narrative can go anywhere based on patient's choices.
        """,
        
        'diary_entry': """
            Personal writing for oneself. No external constraints. Writer chooses topic, length,
            style, content freely. Internal states narrated by participant for self. Completely
            open format. Can be anything from single word to thousands. No rules, no observers,
            total freedom. Maximum narrativity - narrative goes wherever writer takes it.
        """
    }
    
    print("\n" + "=" * 80)
    print("NARRATIVITY SPECTRUM ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing domains from circumscribed to open...")
    print("")
    
    measures = {}
    
    for domain_name, description in domains.items():
        measure = analyzer.analyze_domain_narrativity(domain_name, description)
        measures[domain_name] = measure
        print(analyzer.generate_report(measure))
        print("")
    
    # Place on spectrum
    print("=" * 80)
    print("COMPLETE SPECTRUM")
    print("=" * 80)
    print("")
    
    spectrum = analyzer.place_on_spectrum(measures)
    
    for i, domain in enumerate(spectrum['domains'], 1):
        position = "█" * int(domain['narrativity'] * 20) + "░" * (20 - int(domain['narrativity'] * 20))
        print(f"{i}. {domain['name']:20s} [{position}] {domain['narrativity']:.3f}")
        print(f"   α={domain['alpha']:.2f}, {domain['narrator']}→{domain['narrated']}")
        print("")
    
    print("INSIGHTS:")
    for insight in spectrum['insights']:
        print(f"  • {insight}")
    
    print("\n" + "=" * 80)
    
    return measures, spectrum


if __name__ == "__main__":
    demo_narrativity_spectrum()

