"""
Cross-Domain Validator

Enforces that individual domain validation must complete BEFORE
attempting cross-domain pattern learning or generalization.

Implements the gate: Only learn across domains after each validates independently.
"""

from typing import List, Dict, Optional, Set
from pathlib import Path
import json
from dataclasses import dataclass, asdict


@dataclass
class DomainValidationStatus:
    """Status of validation for a single domain"""
    domain_name: str
    validated: bool
    narrativity: float
    efficiency: float
    passes: bool
    date_validated: str
    transformer_count: int
    sample_size: int
    
    def to_dict(self):
        return asdict(self)


class CrossDomainValidator:
    """
    Ensures individual domain validation before cross-domain learning.
    
    Key principle: Theory-first OR data-first, each domain must independently
    validate narrative laws before we claim cross-domain patterns.
    
    Prevents:
    - Overfitting across domains
    - False generalizations
    - Assuming framework works everywhere
    
    Enables:
    - Honest science (report failures)
    - Domain-specific customization
    - Rigorous cross-domain learning after validation
    """
    
    def __init__(self, validation_registry_path: Optional[str] = None):
        """
        Parameters
        ----------
        validation_registry_path : str, optional
            Path to JSON file tracking validated domains.
            Defaults to ~/.narrative_cache/validation_registry.json
        """
        if validation_registry_path is None:
            cache_dir = Path.home() / '.narrative_cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            validation_registry_path = cache_dir / 'validation_registry.json'
        
        self.registry_path = Path(validation_registry_path)
        self.validated_domains: Dict[str, DomainValidationStatus] = {}
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self):
        """Load validation registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                
                for domain_name, domain_data in data.items():
                    self.validated_domains[domain_name] = DomainValidationStatus(**domain_data)
            except Exception as e:
                print(f"Warning: Could not load validation registry: {e}")
                self.validated_domains = {}
    
    def _save_registry(self):
        """Save validation registry to disk"""
        try:
            data = {
                name: status.to_dict() 
                for name, status in self.validated_domains.items()
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save validation registry: {e}")
    
    def register_validation(
        self,
        domain_name: str,
        narrativity: float,
        efficiency: float,
        passes: bool,
        transformer_count: int,
        sample_size: int,
        date_validated: str = None
    ):
        """
        Register that a domain has completed validation.
        
        Parameters
        ----------
        domain_name : str
            Name of the validated domain
        narrativity : float
            Calculated п
        efficiency : float
            Д/п value
        passes : bool
            Whether domain passed efficiency threshold
        transformer_count : int
            Number of transformers used
        sample_size : int
            Number of organisms analyzed
        date_validated : str, optional
            Date of validation (defaults to today)
        """
        from datetime import datetime
        
        if date_validated is None:
            date_validated = datetime.now().strftime('%Y-%m-%d')
        
        status = DomainValidationStatus(
            domain_name=domain_name,
            validated=True,
            narrativity=narrativity,
            efficiency=efficiency,
            passes=passes,
            date_validated=date_validated,
            transformer_count=transformer_count,
            sample_size=sample_size
        )
        
        self.validated_domains[domain_name] = status
        self._save_registry()
        
        print(f"✓ Registered validation for {domain_name}")
    
    def is_validated(self, domain_name: str) -> bool:
        """Check if domain has been validated"""
        return domain_name in self.validated_domains
    
    def get_validation_status(self, domain_name: str) -> Optional[DomainValidationStatus]:
        """Get validation status for a domain"""
        return self.validated_domains.get(domain_name)
    
    def validate_prerequisites(
        self,
        required_domains: List[str],
        require_all: bool = True
    ) -> bool:
        """
        Check if required domains have been validated.
        
        Parameters
        ----------
        required_domains : list
            Domains that must be validated
        require_all : bool
            If True, all domains must be validated.
            If False, at least one must be validated.
        
        Returns
        -------
        prerequisites_met : bool
            True if prerequisites are satisfied
        """
        validated = [self.is_validated(d) for d in required_domains]
        
        if require_all:
            return all(validated)
        else:
            return any(validated)
    
    def check_prerequisites_or_raise(
        self,
        required_domains: List[str],
        operation_name: str = "cross-domain learning"
    ):
        """
        Check prerequisites and raise exception if not met.
        
        Use this as a gate before cross-domain operations.
        """
        missing = [d for d in required_domains if not self.is_validated(d)]
        
        if missing:
            raise ValueError(
                f"Cannot perform {operation_name}: "
                f"The following domains must be validated first: {missing}\n"
                f"Run individual domain analysis for each before cross-domain learning."
            )
    
    def get_validated_domains(self) -> List[str]:
        """Get list of all validated domain names"""
        return list(self.validated_domains.keys())
    
    def get_passing_domains(self) -> List[str]:
        """Get domains that passed efficiency threshold"""
        return [
            name for name, status in self.validated_domains.items()
            if status.passes
        ]
    
    def get_failing_domains(self) -> List[str]:
        """Get domains that failed efficiency threshold"""
        return [
            name for name, status in self.validated_domains.items()
            if not status.passes
        ]
    
    def get_validation_summary(self) -> Dict:
        """Get summary statistics of all validations"""
        if not self.validated_domains:
            return {
                'total': 0,
                'passes': 0,
                'fails': 0,
                'pass_rate': 0.0
            }
        
        total = len(self.validated_domains)
        passing = self.get_passing_domains()
        failing = self.get_failing_domains()
        
        return {
            'total': total,
            'passes': len(passing),
            'fails': len(failing),
            'pass_rate': len(passing) / total,
            'passing_domains': passing,
            'failing_domains': failing,
            'avg_efficiency': sum(s.efficiency for s in self.validated_domains.values()) / total,
            'avg_narrativity': sum(s.narrativity for s in self.validated_domains.values()) / total
        }
    
    def print_validation_report(self):
        """Print formatted validation report"""
        summary = self.get_validation_summary()
        
        print("\n" + "="*80)
        print("CROSS-DOMAIN VALIDATION REGISTRY")
        print("="*80)
        print(f"\nValidated Domains: {summary['total']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%} ({summary['passes']}/{summary['total']})")
        
        if summary['total'] > 0:
            print(f"Average Efficiency: {summary['avg_efficiency']:.3f}")
            print(f"Average Narrativity: {summary['avg_narrativity']:.3f}")
        
        if summary['passing_domains']:
            print("\n" + "-"*80)
            print("✓ PASSING DOMAINS (Narrative Laws Apply):")
            print("-"*80)
            for domain_name in summary['passing_domains']:
                status = self.validated_domains[domain_name]
                print(f"  • {domain_name:30s} - Efficiency: {status.efficiency:.3f} (п={status.narrativity:.2f})")
        
        if summary['failing_domains']:
            print("\n" + "-"*80)
            print("❌ FAILING DOMAINS (Reality Constrains):")
            print("-"*80)
            for domain_name in summary['failing_domains']:
                status = self.validated_domains[domain_name]
                print(f"  • {domain_name:30s} - Efficiency: {status.efficiency:.3f} (п={status.narrativity:.2f})")
        
        print("="*80)
        
        # Interpretation
        if summary['pass_rate'] < 0.3:
            print("\n📊 INTERPRETATION: Narrative laws apply to MINORITY of domains (honest science)")
        elif summary['pass_rate'] < 0.6:
            print("\n📊 INTERPRETATION: Narrative laws apply to SOME domains (domain-specific)")
        else:
            print("\n📊 INTERPRETATION: Narrative laws apply to MOST domains (broad applicability)")
    
    def can_learn_cross_domain(
        self,
        min_domains: int = 3,
        min_passing: int = 2
    ) -> bool:
        """
        Check if enough domains are validated for cross-domain learning.
        
        Parameters
        ----------
        min_domains : int
            Minimum total validated domains required
        min_passing : int
            Minimum passing domains required
        
        Returns
        -------
        ready : bool
            True if ready for cross-domain learning
        """
        summary = self.get_validation_summary()
        
        return (
            summary['total'] >= min_domains and
            summary['passes'] >= min_passing
        )
    
    def learn_cross_domain_patterns(
        self,
        min_domains: int = 3,
        min_passing: int = 2
    ):
        """
        Learn patterns across validated domains.
        
        GATE: Only proceeds if prerequisites met.
        
        Parameters
        ----------
        min_domains : int
            Minimum validated domains required
        min_passing : int
            Minimum passing domains required
        
        Raises
        ------
        ValueError
            If prerequisites not met
        """
        if not self.can_learn_cross_domain(min_domains, min_passing):
            summary = self.get_validation_summary()
            raise ValueError(
                f"Prerequisites not met for cross-domain learning:\n"
                f"  Required: {min_domains} domains ({min_passing} passing)\n"
                f"  Current: {summary['total']} domains ({summary['passes']} passing)\n"
                f"  Status: Validate more domains before cross-domain analysis"
            )
        
        print("✓ Prerequisites met - ready for cross-domain pattern learning")
        print("\nValidated domains:")
        for domain_name in self.get_validated_domains():
            status = self.validated_domains[domain_name]
            mark = "✓" if status.passes else "❌"
            print(f"  {mark} {domain_name} (eff={status.efficiency:.3f})")
        
        # TODO: Actual cross-domain learning implementation
        # This would include:
        # - Pattern extraction across domains
        # - Meta-analysis of what predicts п
        # - Universal vs domain-specific feature identification
        # - Transfer learning experiments
        
        return {
            'ready': True,
            'validated_domains': self.get_validated_domains(),
            'passing_domains': self.get_passing_domains(),
            'summary': self.get_validation_summary()
        }
    
    def clear_domain(self, domain_name: str):
        """Remove a domain from validation registry (for re-validation)"""
        if domain_name in self.validated_domains:
            del self.validated_domains[domain_name]
            self._save_registry()
            print(f"✓ Cleared validation for {domain_name} - ready for re-validation")
        else:
            print(f"Domain {domain_name} not in registry")
    
    def clear_all(self, confirm: bool = False):
        """Clear all validations (requires confirmation)"""
        if not confirm:
            print("⚠️  WARNING: This will clear all domain validations!")
            print("   Call with confirm=True to proceed")
            return
        
        self.validated_domains = {}
        self._save_registry()
        print("✓ Cleared all domain validations")


# Global validator instance
_global_validator = None


def get_cross_domain_validator() -> CrossDomainValidator:
    """Get or create global cross-domain validator"""
    global _global_validator
    if _global_validator is None:
        _global_validator = CrossDomainValidator()
    return _global_validator

