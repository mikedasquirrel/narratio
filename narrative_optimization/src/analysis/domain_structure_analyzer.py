"""
Domain Structure Analyzer

Analyzes the STRUCTURAL properties of domains to predict which narrative
formulas will work.

Key insight: Domain structure (rules, mechanics, constraints) DETERMINES
which narrative patterns can exist.

Examples:
- Basketball: 5 players → ensemble effects matter
- Crypto: Network effects → innovation narratives matter
- Mental Health: Clinical framing → phonetic-severity matters
- Marriage: 2 people → complementarity matters
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ActorCountType(Enum):
    """Number of actors in domain."""
    INDIVIDUAL = 1  # Self-narratives
    DYADIC = 2  # Relationships
    SMALL_TEAM = (5, 11)  # Teams, small groups
    LARGE_TEAM = (12, 50)  # Organizations
    CROWD = 100  # Markets, masses


class TemporalStructureType(Enum):
    """How time operates in domain."""
    DISCRETE_EVENTS = "discrete"  # Games, transactions
    CONTINUOUS_TIME = "continuous"  # Markets, relationships
    FIXED_PHASES = "phased"  # Diagnosis → treatment → outcome
    CYCLICAL = "cyclical"  # Seasons, years
    ONE_SHOT = "oneshot"  # Single decision


class OutcomeType(Enum):
    """Nature of outcomes."""
    ZERO_SUM = "zerosum"  # Winner and loser
    ABSOLUTE = "absolute"  # Success/failure independent
    RELATIVE_RANKING = "ranking"  # Position in hierarchy
    SURVIVAL_DURATION = "survival"  # How long you last
    CONTINUOUS_PERFORMANCE = "continuous"  # Score/metric


@dataclass
class DomainStructure:
    """Structural properties of a domain."""
    domain_name: str
    actor_count: int
    actor_count_type: ActorCountType
    temporal_structure: TemporalStructureType
    outcome_type: OutcomeType
    information_asymmetry: float  # 0-1 (0=symmetric, 1=completely hidden)
    constraint_density: float  # 0-1 (0=unconstrained, 1=highly constrained)
    
    # Derived properties
    predicted_alpha: float  # Predicted α parameter
    predicted_transformer: str  # Predicted best transformer
    predicted_archetype: str  # Predicted narrative archetype
    reasoning: str  # Why these predictions
    
    def __repr__(self):
        return (
            f"{self.domain_name}: {self.actor_count} actors, {self.temporal_structure.value}, "
            f"predicted α={self.predicted_alpha:.2f}, {self.predicted_transformer}"
        )


class DomainStructureAnalyzer:
    """
    Analyzes domain structure to predict narrative formulas.
    
    Process:
    1. Extract structural properties from domain description
    2. Apply constraint propagation rules
    3. Predict α parameter and best transformer
    4. Provide reasoning for predictions
    """
    
    def __init__(self):
        # Mapping rules learned from validated domains
        self.structure_to_formula_rules = self._initialize_rules()
    
    def analyze_domain(
        self,
        domain_name: str,
        domain_description: str,
        examples: Optional[List[str]] = None
    ) -> DomainStructure:
        """
        Analyze domain structure and predict narrative formula.
        
        Parameters
        ----------
        domain_name : str
            Name of domain
        domain_description : str
            Description of domain mechanics, rules, constraints
        examples : List[str], optional
            Example instances from domain
            
        Returns
        -------
        DomainStructure with predictions
        """
        # Extract structural properties
        actor_count, actor_type = self._extract_actor_count(domain_description, examples)
        temporal_structure = self._extract_temporal_structure(domain_description)
        outcome_type = self._extract_outcome_type(domain_description)
        information_asymmetry = self._calculate_information_asymmetry(domain_description)
        constraint_density = self._calculate_constraint_density(domain_description)
        
        # Predict formula from structure
        predicted_alpha, predicted_transformer, archetype, reasoning = self._predict_formula(
            actor_type, temporal_structure, outcome_type,
            information_asymmetry, constraint_density
        )
        
        structure = DomainStructure(
            domain_name=domain_name,
            actor_count=actor_count,
            actor_count_type=actor_type,
            temporal_structure=temporal_structure,
            outcome_type=outcome_type,
            information_asymmetry=information_asymmetry,
            constraint_density=constraint_density,
            predicted_alpha=predicted_alpha,
            predicted_transformer=predicted_transformer,
            predicted_archetype=archetype,
            reasoning=reasoning
        )
        
        return structure
    
    def _extract_actor_count(
        self,
        description: str,
        examples: Optional[List[str]]
    ) -> Tuple[int, ActorCountType]:
        """Extract number of actors from description."""
        desc_lower = description.lower()
        
        # Check for explicit mentions
        if any(word in desc_lower for word in ['individual', 'person', 'one', 'solo', 'self']):
            return 1, ActorCountType.INDIVIDUAL
        
        if any(word in desc_lower for word in ['couple', 'pair', 'two people', 'dyad', 'relationship']):
            return 2, ActorCountType.DYADIC
        
        if any(word in desc_lower for word in ['team', '5 players', 'squad', 'group']):
            # Extract number if mentioned
            numbers = re.findall(r'\b(\d+)\s*(?:player|person|member)', desc_lower)
            if numbers:
                count = int(numbers[0])
                if count <= 11:
                    return count, ActorCountType.SMALL_TEAM
                elif count <= 50:
                    return count, ActorCountType.LARGE_TEAM
                else:
                    return count, ActorCountType.CROWD
            return 7, ActorCountType.SMALL_TEAM  # Default team size
        
        if any(word in desc_lower for word in ['market', 'crowd', 'population', 'masses']):
            return 1000, ActorCountType.CROWD
        
        # Default: analyze examples if provided
        if examples:
            # Count mentioned entities
            avg_count = np.mean([len(re.findall(r'\b[A-Z][a-z]+\b', ex)) for ex in examples])
            if avg_count <= 2:
                return 1, ActorCountType.INDIVIDUAL
            elif avg_count <= 4:
                return 2, ActorCountType.DYADIC
            elif avg_count <= 15:
                return int(avg_count), ActorCountType.SMALL_TEAM
            else:
                return int(avg_count), ActorCountType.CROWD
        
        return 1, ActorCountType.INDIVIDUAL  # Default
    
    def _extract_temporal_structure(self, description: str) -> TemporalStructureType:
        """Extract temporal structure from description."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['game', 'match', 'event', 'transaction', 'discrete']):
            return TemporalStructureType.DISCRETE_EVENTS
        
        if any(word in desc_lower for word in ['continuous', 'ongoing', 'always', 'constant']):
            return TemporalStructureType.CONTINUOUS_TIME
        
        if any(word in desc_lower for word in ['phase', 'stage', 'step', 'diagnosis', 'treatment']):
            return TemporalStructureType.FIXED_PHASES
        
        if any(word in desc_lower for word in ['season', 'year', 'cycle', 'periodic']):
            return TemporalStructureType.CYCLICAL
        
        if any(word in desc_lower for word in ['single', 'one-time', 'decision']):
            return TemporalStructureType.ONE_SHOT
        
        return TemporalStructureType.DISCRETE_EVENTS  # Default
    
    def _extract_outcome_type(self, description: str) -> OutcomeType:
        """Extract outcome type from description."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['win', 'lose', 'versus', 'compete', 'vs']):
            return OutcomeType.ZERO_SUM
        
        if any(word in desc_lower for word in ['rank', 'position', 'top', 'leaderboard']):
            return OutcomeType.RELATIVE_RANKING
        
        if any(word in desc_lower for word in ['survive', 'duration', 'how long', 'lifespan']):
            return OutcomeType.SURVIVAL_DURATION
        
        if any(word in desc_lower for word in ['score', 'metric', 'performance', 'rating']):
            return OutcomeType.CONTINUOUS_PERFORMANCE
        
        if any(word in desc_lower for word in ['success', 'failure', 'achieve']):
            return OutcomeType.ABSOLUTE
        
        return OutcomeType.ABSOLUTE  # Default
    
    def _calculate_information_asymmetry(self, description: str) -> float:
        """Calculate degree of information asymmetry (0-1)."""
        desc_lower = description.lower()
        
        # Symmetric indicators
        symmetric_words = ['public', 'transparent', 'open', 'visible', 'known']
        symmetric_count = sum(1 for word in symmetric_words if word in desc_lower)
        
        # Asymmetric indicators
        asymmetric_words = ['hidden', 'secret', 'private', 'unknown', 'blind', 'asymmetric']
        asymmetric_count = sum(1 for word in asymmetric_words if word in desc_lower)
        
        # Calculate score (0 = symmetric, 1 = asymmetric)
        if symmetric_count + asymmetric_count == 0:
            return 0.5  # Unknown
        
        asymmetry = asymmetric_count / (symmetric_count + asymmetric_count)
        return asymmetry
    
    def _calculate_constraint_density(self, description: str) -> float:
        """Calculate how constrained the domain is (0-1)."""
        desc_lower = description.lower()
        
        # Constraint indicators
        constraint_words = ['rule', 'regulation', 'law', 'must', 'require', 'constraint', 'limit']
        constraint_count = sum(1 for word in constraint_words if word in desc_lower)
        
        # Freedom indicators
        freedom_words = ['free', 'open', 'flexible', 'choice', 'option', 'varied']
        freedom_count = sum(1 for word in freedom_words if word in desc_lower)
        
        # Calculate density (0 = unconstrained, 1 = highly constrained)
        total = constraint_count + freedom_count
        if total == 0:
            return 0.5  # Unknown
        
        density = constraint_count / total
        return density
    
    def _predict_formula(
        self,
        actor_type: ActorCountType,
        temporal: TemporalStructureType,
        outcome: OutcomeType,
        asymmetry: float,
        density: float
    ) -> Tuple[float, str, str, str]:
        """
        Predict narrative formula from structural properties.
        
        Returns
        -------
        Tuple of (alpha, transformer, archetype, reasoning)
        """
        # Base prediction on actor count (primary determinant)
        if actor_type == ActorCountType.INDIVIDUAL:
            base_alpha = 0.2
            base_transformer = "self_perception"
            base_archetype = "Character-Driven (identity matters)"
            reasoning = "Individual actor → self-narratives dominate"
            
        elif actor_type == ActorCountType.DYADIC:
            base_alpha = 0.3
            base_transformer = "relational"
            base_archetype = "Relational (complementarity matters)"
            reasoning = "Two actors → relational dynamics dominate"
            
        elif actor_type == ActorCountType.SMALL_TEAM:
            base_alpha = 0.4
            base_transformer = "ensemble"
            base_archetype = "Ensemble-Driven (group chemistry matters)"
            reasoning = "Small team → ensemble effects dominate"
            
        elif actor_type == ActorCountType.LARGE_TEAM:
            base_alpha = 0.6
            base_transformer = "statistical"
            base_archetype = "Hybrid (structure + individuals)"
            reasoning = "Large team → statistical patterns emerge"
            
        else:  # CROWD
            base_alpha = 0.8
            base_transformer = "statistical"
            base_archetype = "Plot-Driven (content/events matter)"
            reasoning = "Crowd/market → statistical aggregates dominate"
        
        # Adjust based on other factors
        alpha_adjustments = []
        
        # High constraint density → increase α (more rule-driven, less character-driven)
        if density > 0.7:
            alpha_adjustments.append(+0.15)
            reasoning += "; High constraints → more rule-based"
        elif density < 0.3:
            alpha_adjustments.append(-0.15)
            reasoning += "; Low constraints → more character-based"
        
        # High information asymmetry → decrease α (character knowledge matters)
        if asymmetry > 0.7:
            alpha_adjustments.append(-0.1)
            reasoning += "; Hidden information → character perception matters"
        
        # Outcome type effects
        if outcome == OutcomeType.ZERO_SUM:
            alpha_adjustments.append(+0.05)
            reasoning += "; Zero-sum → competitive dynamics"
        
        # Calculate final α
        final_alpha = base_alpha + sum(alpha_adjustments)
        final_alpha = np.clip(final_alpha, 0.0, 1.0)
        
        # Adjust transformer if α changed significantly
        if final_alpha > 0.7 and base_transformer != "statistical":
            final_transformer = "statistical"
            reasoning += f" → Adjusted to {final_transformer} due to high α"
        elif final_alpha < 0.3 and base_transformer not in ["nominative", "self_perception"]:
            final_transformer = "nominative"
            reasoning += f" → Adjusted to {final_transformer} due to low α"
        else:
            final_transformer = base_transformer
        
        return final_alpha, final_transformer, base_archetype, reasoning
    
    def _initialize_rules(self) -> Dict:
        """Initialize structure-to-formula mapping rules."""
        return {
            'actor_count_primary': {
                1: {'alpha': 0.2, 'transformer': 'self_perception'},
                2: {'alpha': 0.3, 'transformer': 'relational'},
                (5, 11): {'alpha': 0.4, 'transformer': 'ensemble'},
                (12, 50): {'alpha': 0.6, 'transformer': 'statistical'},
                100: {'alpha': 0.8, 'transformer': 'statistical'}
            },
            'adjustments': {
                'high_constraints': +0.15,
                'low_constraints': -0.15,
                'high_asymmetry': -0.1,
                'zero_sum': +0.05
            }
        }
    
    def compare_structures(
        self,
        structure1: DomainStructure,
        structure2: DomainStructure
    ) -> Dict[str, Any]:
        """
        Compare two domain structures.
        
        Returns similarity score and identifies similar properties.
        """
        similarities = []
        
        # Actor count similarity
        actor_sim = 1.0 / (1 + abs(structure1.actor_count - structure2.actor_count) / 10)
        similarities.append(actor_sim)
        
        # Temporal structure (binary)
        temporal_sim = 1.0 if structure1.temporal_structure == structure2.temporal_structure else 0.3
        similarities.append(temporal_sim)
        
        # Outcome type (binary)
        outcome_sim = 1.0 if structure1.outcome_type == structure2.outcome_type else 0.3
        similarities.append(outcome_sim)
        
        # Information asymmetry
        asymmetry_sim = 1.0 - abs(structure1.information_asymmetry - structure2.information_asymmetry)
        similarities.append(asymmetry_sim)
        
        # Constraint density
        constraint_sim = 1.0 - abs(structure1.constraint_density - structure2.constraint_density)
        similarities.append(constraint_sim)
        
        # Overall similarity (weighted average)
        weights = [0.35, 0.2, 0.15, 0.15, 0.15]  # Actor count is most important
        overall_similarity = np.average(similarities, weights=weights)
        
        # Predict if formulas should be similar
        alpha_difference = abs(structure1.predicted_alpha - structure2.predicted_alpha)
        formula_similarity_predicted = 1.0 - alpha_difference
        
        return {
            'structural_similarity': overall_similarity,
            'predicted_formula_similarity': formula_similarity_predicted,
            'expect_similar_formulas': overall_similarity > 0.7,
            'alpha_difference': alpha_difference,
            'similar_properties': self._identify_similar_properties(structure1, structure2)
        }
    
    def _identify_similar_properties(
        self,
        s1: DomainStructure,
        s2: DomainStructure
    ) -> List[str]:
        """Identify which structural properties are similar."""
        similar = []
        
        if s1.actor_count_type == s2.actor_count_type:
            similar.append("actor_count_type")
        
        if s1.temporal_structure == s2.temporal_structure:
            similar.append("temporal_structure")
        
        if s1.outcome_type == s2.outcome_type:
            similar.append("outcome_type")
        
        if abs(s1.information_asymmetry - s2.information_asymmetry) < 0.2:
            similar.append("information_asymmetry")
        
        if abs(s1.constraint_density - s2.constraint_density) < 0.2:
            similar.append("constraint_density")
        
        return similar
    
    def validate_prediction(
        self,
        structure: DomainStructure,
        actual_alpha: float,
        actual_best_transformer: str
    ) -> Dict[str, Any]:
        """
        Validate structural predictions against empirical results.
        
        Parameters
        ----------
        structure : DomainStructure
            Structural analysis with predictions
        actual_alpha : float
            Empirically discovered α
        actual_best_transformer : str
            Empirically discovered best transformer
            
        Returns
        -------
        Dict with validation results
        """
        alpha_error = abs(structure.predicted_alpha - actual_alpha)
        alpha_accurate = alpha_error < 0.2  # Within ±0.2
        
        transformer_correct = structure.predicted_transformer == actual_best_transformer
        
        # Overall prediction success
        if alpha_accurate and transformer_correct:
            accuracy = "EXCELLENT"
            score = 1.0
        elif alpha_accurate or transformer_correct:
            accuracy = "PARTIAL"
            score = 0.7
        else:
            accuracy = "POOR"
            score = 0.3
        
        return {
            'accuracy': accuracy,
            'score': score,
            'alpha_error': alpha_error,
            'alpha_accurate': alpha_accurate,
            'transformer_correct': transformer_correct,
            'predicted_alpha': structure.predicted_alpha,
            'actual_alpha': actual_alpha,
            'predicted_transformer': structure.predicted_transformer,
            'actual_transformer': actual_best_transformer
        }
    
    def generate_report(self, structure: DomainStructure) -> str:
        """Generate structural analysis report."""
        report = []
        report.append("=" * 80)
        report.append(f"DOMAIN STRUCTURE ANALYSIS: {structure.domain_name.upper()}")
        report.append("=" * 80)
        report.append("")
        
        report.append("STRUCTURAL PROPERTIES:")
        report.append(f"  Actor Count: {structure.actor_count} ({structure.actor_count_type.name})")
        report.append(f"  Temporal Structure: {structure.temporal_structure.value}")
        report.append(f"  Outcome Type: {structure.outcome_type.value}")
        report.append(f"  Information Asymmetry: {structure.information_asymmetry:.2f}")
        report.append(f"  Constraint Density: {structure.constraint_density:.2f}")
        report.append("")
        
        report.append("PREDICTED NARRATIVE FORMULA:")
        report.append(f"  α Parameter: {structure.predicted_alpha:.3f}")
        report.append(f"  Best Transformer: {structure.predicted_transformer}")
        report.append(f"  Archetype: {structure.predicted_archetype}")
        report.append("")
        
        report.append("REASONING:")
        report.append(f"  {structure.reasoning}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def demo_domain_structure_analysis():
    """Demonstrate domain structure analysis on known domains."""
    analyzer = DomainStructureAnalyzer()
    
    # Analyze known domains
    domains = {
        'NBA': """
            Basketball games with 5 players per team competing in 48-minute games.
            Zero-sum outcomes (one team wins, one loses). Highly constrained by rules.
            Team chemistry and ensemble effects are critical. Discrete events (games)
            within cyclical structure (seasons).
        """,
        'Crypto': """
            Cryptocurrency market with thousands of coins competing for market cap rankings.
            Continuous price movements. Network effects and innovation narratives matter.
            Relative ranking outcomes. Public information but technical complexity.
        """,
        'Mental Health': """
            Clinical diagnostic system with individual patients seeking treatment.
            Phased structure (diagnosis → treatment → outcome). Medical terminology
            and phonetic properties of disorder names affect stigma and treatment seeking.
            High constraint density (medical system).
        """,
        'Marriage': """
            Two-person long-term relationships. Continuous time structure with lifecycle phases.
            Complementarity and compatibility are structural requirements. Information
            asymmetry (partners gradually reveal themselves). Success measured by duration.
        """
    }
    
    print("\n" + "=" * 80)
    print("DOMAIN STRUCTURE ANALYSIS - PREDICTIONS")
    print("=" * 80)
    
    structures = {}
    
    for domain_name, description in domains.items():
        structure = analyzer.analyze_domain(domain_name, description)
        structures[domain_name] = structure
        print("\n" + analyzer.generate_report(structure))
    
    # Compare structures
    print("\n" + "=" * 80)
    print("STRUCTURAL COMPARISONS")
    print("=" * 80)
    print("")
    
    domain_names = list(structures.keys())
    for i, name1 in enumerate(domain_names):
        for name2 in domain_names[i+1:]:
            comparison = analyzer.compare_structures(structures[name1], structures[name2])
            print(f"{name1} vs {name2}:")
            print(f"  Structural similarity: {comparison['structural_similarity']:.3f}")
            print(f"  Expect similar formulas: {comparison['expect_similar_formulas']}")
            print(f"  Similar properties: {', '.join(comparison['similar_properties'])}")
            print("")
    
    return structures


if __name__ == "__main__":
    demo_domain_structure_analysis()

