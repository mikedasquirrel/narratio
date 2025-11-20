"""
Startup Data Collector

Collects startup descriptions and outcomes from public sources:
- YCombinator batch data
- TechCrunch profiles
- Crunchbase (if accessible)

Focus: Company description, founding team narrative, market positioning, outcomes
"""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Startup:
    """Startup with narrative and outcomes."""
    company_id: str
    name: str
    description: str
    founding_team_description: str
    founder_count: int
    founder_names: List[str]
    market_category: str
    yc_batch: Optional[str]
    
    # Outcomes
    total_funding: float  # In millions
    valuation: float  # In millions, -1 if unknown
    exit_type: str  # 'acquired', 'ipo', 'unicorn', 'operating', 'failed'
    years_active: float
    
    # Narrative elements (to be filled)
    innovation_score: float = 0.0
    team_coherence_score: float = 0.0
    market_positioning_score: float = 0.0
    
    # Binary outcome (for classification)
    successful: bool = False  # Top 25% (unicorn/acquired for >100M/IPO)
    
    def to_dict(self):
        return asdict(self)


class StartupDataCollector:
    """
    Collects startup data from public sources.
    
    Strategy:
    1. Start with YCombinator companies (well-documented)
    2. Extract descriptions from YC website
    3. Look up outcomes from public databases
    4. Enrich with founding team narratives
    """
    
    def __init__(self):
        self.startups: List[Startup] = []
        
    def add_manual_startup(
        self,
        name: str,
        description: str,
        founders: List[str],
        team_description: str,
        category: str,
        funding: float,
        valuation: float,
        exit_type: str,
        years: float,
        batch: Optional[str] = None
    ) -> Startup:
        """Add a manually entered startup."""
        
        company_id = name.lower().replace(' ', '_').replace('.', '')
        
        # Determine success (top 25% = unicorn/big exit)
        successful = (
            valuation >= 1000 or  # Unicorn
            (exit_type in ['acquired', 'ipo'] and valuation >= 100) or
            exit_type == 'unicorn'
        )
        
        startup = Startup(
            company_id=company_id,
            name=name,
            description=description,
            founding_team_description=team_description,
            founder_count=len(founders),
            founder_names=founders,
            market_category=category,
            yc_batch=batch,
            total_funding=funding,
            valuation=valuation,
            exit_type=exit_type,
            years_active=years,
            successful=successful
        )
        
        self.startups.append(startup)
        return startup
    
    def create_yc_collection_template(self) -> List[Dict]:
        """
        Create template for collecting YC startup data.
        
        Returns list of well-known YC companies to start with.
        """
        
        # Seed data: Famous YC companies
        template = [
            {
                'name': 'Airbnb',
                'description': 'Platform for people to rent out their homes to travelers, creating a marketplace for short-term lodging. Revolutionary idea that turned everyone into a potential hotel owner.',
                'founders': ['Brian Chesky', 'Joe Gebbia', 'Nathan Blecharczyk'],
                'team_description': 'Two designers and an engineer who couldn\'t pay rent. Strong ensemble with complementary skills - design vision plus technical execution.',
                'category': 'marketplace',
                'funding': 6000,
                'valuation': 75000,
                'exit_type': 'ipo',
                'years': 14,
                'batch': 'W09'
            },
            {
                'name': 'Stripe',
                'description': 'Developer-first payment processing platform that makes it easy to accept payments online. Seven lines of code to start accepting payments.',
                'founders': ['Patrick Collison', 'John Collison'],
                'team_description': 'Two Irish brothers, both technical prodigies. Strong sibling partnership with shared vision.',
                'category': 'fintech',
                'funding': 2200,
                'valuation': 95000,
                'exit_type': 'operating',
                'years': 13,
                'batch': 'S09'
            },
            {
                'name': 'Dropbox',
                'description': 'Cloud storage service that syncs files across devices. Simple concept executed perfectly - your files, everywhere.',
                'founders': ['Drew Houston', 'Arash Ferdowsi'],
                'team_description': 'Technical founder with clear vision plus strong technical co-founder. Product-focused duo.',
                'category': 'saas',
                'funding': 1700,
                'valuation': 10000,
                'exit_type': 'ipo',
                'years': 15,
                'batch': 'S07'
            },
            {
                'name': 'Instacart',
                'description': 'Grocery delivery service connecting customers with personal shoppers. Order groceries online, delivered in an hour.',
                'founders': ['Apoorva Mehta'],
                'team_description': 'Solo technical founder with supply chain background. Strong individual execution.',
                'category': 'marketplace',
                'funding': 2900,
                'valuation': 39000,
                'exit_type': 'operating',
                'years': 11,
                'batch': 'S12'
            },
            {
                'name': 'Coinbase',
                'description': 'Cryptocurrency exchange making it easy to buy, sell, and store digital currencies. The most trusted platform for crypto trading.',
                'founders': ['Brian Armstrong', 'Fred Ehrsam'],
                'team_description': 'Technical founder (ex-Airbnb engineer) plus finance expert. Perfect complementarity for crypto exchange.',
                'category': 'fintech',
                'funding': 547,
                'valuation': 85000,
                'exit_type': 'ipo',
                'years': 12,
                'batch': 'S12'
            },
            # Add examples of failures and moderate successes
            {
                'name': 'Homejoy',
                'description': 'Home cleaning marketplace connecting customers with cleaning professionals. Book a cleaning in 60 seconds.',
                'founders': ['Adora Cheung', 'Aaron Cheung'],
                'team_description': 'Sibling duo with strong work ethic. Technical backgrounds.',
                'category': 'marketplace',
                'funding': 40,
                'valuation': -1,
                'exit_type': 'failed',
                'years': 4,
                'batch': 'S10'
            },
            {
                'name': 'Parse',
                'description': 'Backend-as-a-service for mobile developers. Build mobile apps without managing servers.',
                'founders': ['Ilya Sukhar', 'James Yu', 'Kevin Lacker', 'Tikhon Bernstam'],
                'team_description': 'Four strong technical co-founders from Stanford. Engineering-heavy team.',
                'category': 'developer_tools',
                'funding': 7.5,
                'valuation': 85,
                'exit_type': 'acquired',
                'years': 4,
                'batch': 'S11'
            }
        ]
        
        return template
    
    def save_template(self, filepath: str):
        """Save collection template for manual data entry."""
        template = self.create_yc_collection_template()
        
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"✓ Template saved to: {filepath}")
        print(f"  Contains {len(template)} seed examples")
        print(f"\nNext steps:")
        print(f"  1. Review template format")
        print(f"  2. Add 50-100 more YC companies")
        print(f"  3. Focus on diverse outcomes (not just unicorns)")
        print(f"  4. Include founding team narratives")
        print(f"  5. Run analysis once we have 100+ companies")
    
    def load_from_template(self, filepath: str) -> List[Startup]:
        """Load startups from filled template."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.startups = []
        
        for item in data:
            startup = self.add_manual_startup(
                name=item['name'],
                description=item['description'],
                founders=item['founders'],
                team_description=item['team_description'],
                category=item['category'],
                funding=item['funding'],
                valuation=item.get('valuation', -1),
                exit_type=item['exit_type'],
                years=item['years'],
                batch=item.get('batch')
            )
        
        return self.startups
    
    def save_dataset(self, filepath: str):
        """Save collected startups as dataset."""
        data = [startup.to_dict() for startup in self.startups]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(self.startups)} startups to: {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.startups:
            return {'error': 'No startups loaded'}
        
        total = len(self.startups)
        successful = sum(1 for s in self.startups if s.successful)
        
        by_category = {}
        for s in self.startups:
            by_category[s.market_category] = by_category.get(s.market_category, 0) + 1
        
        by_exit = {}
        for s in self.startups:
            by_exit[s.exit_type] = by_exit.get(s.exit_type, 0) + 1
        
        avg_funding = np.mean([s.total_funding for s in self.startups])
        avg_team_size = np.mean([s.founder_count for s in self.startups])
        
        return {
            'total_startups': total,
            'successful': successful,
            'success_rate': successful / total,
            'by_category': by_category,
            'by_exit_type': by_exit,
            'avg_funding': avg_funding,
            'avg_founder_count': avg_team_size
        }


def main():
    """Initialize startup data collection."""
    print("=" * 80)
    print("STARTUP DATA COLLECTION - INITIALIZATION")
    print("=" * 80)
    print("\nStarting with YCombinator companies...")
    print("")
    
    collector = StartupDataCollector()
    
    # Create and save template
    template_path = '/Users/michaelsmerconish/Desktop/RandomCode/novelization/data/domains/startups_template.json'
    collector.save_template(template_path)
    
    # Load seed data
    collector.load_from_template(template_path)
    
    # Save initial dataset
    dataset_path = '/Users/michaelsmerconish/Desktop/RandomCode/novelization/data/domains/startups_initial.json'
    collector.save_dataset(dataset_path)
    
    # Show statistics
    stats = collector.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total companies: {stats['total_startups']}")
    print(f"  Successful: {stats['successful']} ({stats['success_rate']:.1%})")
    print(f"  Avg founding team size: {stats['avg_founder_count']:.1f}")
    print(f"  Avg funding: ${stats['avg_funding']:.1f}M")
    print(f"\nBy exit type:")
    for exit_type, count in stats['by_exit_type'].items():
        print(f"    {exit_type}: {count}")
    
    print("\n" + "=" * 80)
    print("READY TO ANALYZE")
    print("=" * 80)
    print("\nCurrent: 7 seed examples")
    print("Target: 100+ companies for robust analysis")
    print("\nNext: Expand dataset with more YC companies, then run analysis")


if __name__ == "__main__":
    import numpy as np
    main()

