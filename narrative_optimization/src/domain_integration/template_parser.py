"""
Domain Integration Template Parser

Parses user-provided domain specifications and generates integration plan.
"""

from typing import Dict, Any, List
import re


class DomainIntegrationTemplate:
    """
    Parse and process domain integration templates.
    
    Extracts domain specifications, validates data, and generates
    automated integration and experimentation plans.
    """
    
    def __init__(self):
        self.parsed_domain = None
        self.existing_domains = self._load_existing_domains()
    
    def _load_existing_domains(self) -> Dict[str, Any]:
        """Load information about existing tested domains."""
        return {
            'news_classification': {
                'type': 'content',
                'outcome': 'category',
                'results': {
                    'statistical': 0.69,
                    'ensemble': 0.28,
                    'linguistic': 0.37,
                    'self_perception': 0.34,
                    'potential': 0.29
                },
                'insight': 'Statistical wins - content IS the signal',
                'dimensions_matter': ['statistical', 'linguistic']
            },
            'relationships_synthetic': {
                'type': 'relationships',
                'outcome': 'compatibility',
                'results': {},  # Not yet tested
                'hypothesis': 'Ensemble + Relational should win',
                'dimensions_matter': ['ensemble', 'relational', 'potential']
            },
            'wellness_journals': {
                'type': 'wellness',
                'outcome': 'improvement',
                'results': {},  # Not yet tested
                'hypothesis': 'Self-Perception + Potential should win',
                'dimensions_matter': ['self_perception', 'potential', 'linguistic']
            }
        }
    
    def parse_template(self, template_text: str) -> Dict[str, Any]:
        """
        Parse filled-out template into structured format.
        
        Parameters
        ----------
        template_text : str
            Complete filled template
        
        Returns
        -------
        domain_spec : dict
            Structured domain specification
        """
        domain_spec = {
            'name': '',
            'type': '',
            'description': '',
            'data': {},
            'dimensions': [],
            'related_domains': [],
            'hypotheses': {},
            'success_metrics': {}
        }
        
        # Extract domain name (simple regex)
        name_match = re.search(r'Domain Name:\s*(.+)', template_text)
        if name_match:
            domain_spec['name'] = name_match.group(1).strip()
        
        # Extract domain type
        if 'Relationships' in template_text and '☑' in template_text:
            domain_spec['type'] = 'relationships'
        elif 'Communication' in template_text:
            domain_spec['type'] = 'communication'
        elif 'Wellness' in template_text:
            domain_spec['type'] = 'wellness'
        elif 'Content' in template_text:
            domain_spec['type'] = 'content'
        
        # Extract which dimensions selected (look for ☑)
        dimension_map = {
            'Ensemble': 'ensemble',
            'Linguistic': 'linguistic',
            'Self-Perception': 'self_perception',
            'Narrative Potential': 'potential',
            'Relational': 'relational',
            'Nominative': 'nominative'
        }
        
        for dim_name, dim_key in dimension_map.items():
            if f'☑ {dim_name}' in template_text or f'☑ **{dim_name}**' in template_text:
                domain_spec['dimensions'].append(dim_key)
        
        # Extract sample count
        count_match = re.search(r'Sample Count:\s*(\d+)', template_text)
        if count_match:
            domain_spec['data']['sample_count'] = int(count_match.group(1))
        
        self.parsed_domain = domain_spec
        return domain_spec
    
    def map_to_existing_domains(self, new_domain: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze relationships to existing domains.
        
        Returns similarity scores and transfer predictions.
        """
        relativity_map = {
            'similarities': {},
            'transfer_predictions': {},
            'novel_aspects': []
        }
        
        new_type = new_domain.get('type', '')
        new_dims = set(new_domain.get('dimensions', []))
        
        for existing_name, existing_info in self.existing_domains.items():
            existing_type = existing_info['type']
            existing_dims = set(existing_info.get('dimensions_matter', []))
            
            # Type similarity
            type_match = 1.0 if new_type == existing_type else 0.5 if new_type else 0.0
            
            # Dimension overlap
            dim_overlap = len(new_dims & existing_dims) / max(len(new_dims | existing_dims), 1)
            
            # Overall similarity
            similarity = 0.6 * type_match + 0.4 * dim_overlap
            
            relativity_map['similarities'][existing_name] = {
                'score': similarity,
                'type_match': type_match,
                'dimension_overlap': dim_overlap,
                'shared_dimensions': list(new_dims & existing_dims)
            }
            
            # Transfer predictions
            if similarity > 0.6:
                # High similarity - predict similar results
                if 'results' in existing_info and existing_info['results']:
                    relativity_map['transfer_predictions'][existing_name] = {
                        'confidence': 'high',
                        'prediction': f"Should behave similarly to {existing_name}",
                        'expected_winners': existing_info.get('dimensions_matter', [])
                    }
            elif similarity > 0.3:
                # Moderate similarity - some transfer
                relativity_map['transfer_predictions'][existing_name] = {
                    'confidence': 'moderate',
                    'prediction': f"Some patterns from {existing_name} may transfer",
                    'shared_aspects': list(new_dims & existing_dims)
                }
        
        # Identify novel aspects
        all_existing_dims = set()
        for existing in self.existing_domains.values():
            all_existing_dims.update(existing.get('dimensions_matter', []))
        
        novel_dims = new_dims - all_existing_dims
        if novel_dims:
            relativity_map['novel_aspects'] = [
                f"First domain emphasizing {dim}" for dim in novel_dims
            ]
        
        return relativity_map
    
    def generate_experiment_plan(self, domain_spec: Dict[str, Any], relativity_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate automated experiment plan.
        
        Based on domain characteristics and relativity to existing domains.
        """
        plan = {
            'experiments_to_run': [],
            'transformers_to_test': [],
            'combinations_to_test': [],
            'comparisons': []
        }
        
        # Test all selected dimensions
        selected_dims = domain_spec.get('dimensions', [])
        
        # Always test statistical baseline
        plan['transformers_to_test'].append('statistical')
        
        # Test each selected dimension
        for dim in selected_dims:
            plan['transformers_to_test'].append(dim)
        
        # Based on relativity, suggest combinations
        most_similar_domain = max(
            relativity_map['similarities'].items(),
            key=lambda x: x[1]['score']
        )[0] if relativity_map['similarities'] else None
        
        if most_similar_domain:
            similar_info = self.existing_domains[most_similar_domain]
            suggested_combos = similar_info.get('dimensions_matter', [])
            
            # Test top combinations from similar domain
            for i, dim1 in enumerate(suggested_combos[:2]):
                for dim2 in suggested_combos[i+1:3]:
                    if dim1 in selected_dims and dim2 in selected_dims:
                        plan['combinations_to_test'].append((dim1, dim2))
        
        # Generate experiment list
        plan['experiments_to_run'] = [
            {
                'name': f"{domain_spec['name']}_baseline",
                'description': 'Test statistical baseline',
                'transformers': ['statistical']
            },
            {
                'name': f"{domain_spec['name']}_individual",
                'description': 'Test each dimension individually',
                'transformers': selected_dims
            },
            {
                'name': f"{domain_spec['name']}_combinations",
                'description': 'Test dimension combinations',
                'transformers': plan['combinations_to_test']
            }
        ]
        
        # Add comparisons to similar domains
        if most_similar_domain:
            plan['comparisons'].append({
                'compare_to': most_similar_domain,
                'reason': f"Most similar domain (similarity: {relativity_map['similarities'][most_similar_domain]['score']:.2f})",
                'expectations': 'Should show similar dimension importance patterns'
            })
        
        return plan


if __name__ == '__main__':
    # Demo
    parser = DomainIntegrationTemplate()
    
    print("Domain Integration Template Parser")
    print("=" * 60)
    print(f"Existing domains loaded: {list(parser.existing_domains.keys())}")
    print("\nReady to parse new domain templates!")

