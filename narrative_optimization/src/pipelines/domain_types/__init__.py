"""
Domain Type Abstractions

Domain-specific templates that customize transformer selection, validation metrics,
and reporting for different domain types (sports, entertainment, nominative, etc.).
"""

from .base import BaseDomainType
from .sports import SportsDomain, SportsIndividualDomain, SportsTeamDomain
from .entertainment import EntertainmentDomain
from .nominative import NominativeDomain
from .business import BusinessDomain, MedicalDomain

from pipelines.domain_config import DomainType


def get_domain_type_class(domain_type: DomainType) -> type:
    """
    Get domain type class for a DomainType enum.
    
    Parameters
    ----------
    domain_type : DomainType
        Domain type enum
        
    Returns
    -------
    domain_type_class : type
        Domain type class (or None if not found)
    """
    type_map = {
        DomainType.SPORTS: SportsDomain,
        DomainType.SPORTS_INDIVIDUAL: SportsIndividualDomain,
        DomainType.SPORTS_TEAM: SportsTeamDomain,
        DomainType.ENTERTAINMENT: EntertainmentDomain,
        DomainType.NOMINATIVE: NominativeDomain,
        DomainType.BUSINESS: BusinessDomain,
        DomainType.MEDICAL: MedicalDomain,
    }
    
    return type_map.get(domain_type)


__all__ = [
    'BaseDomainType',
    'SportsDomain',
    'SportsIndividualDomain',
    'SportsTeamDomain',
    'EntertainmentDomain',
    'NominativeDomain',
    'BusinessDomain',
    'MedicalDomain',
    'get_domain_type_class',
]
