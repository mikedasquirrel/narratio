"""
Multi-Modal Pattern Analyzer

Analyzes patterns across multiple modalities:
- Text patterns (current)
- Numeric patterns (from features)
- Temporal patterns (from sequences)
- Network patterns (from relationships)

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from scipy import stats


class MultiModalPatternAnalyzer:
    """
    Analyze patterns across multiple modalities.
    
    Combines:
    - Text analysis (keyword patterns)
    - Numeric analysis (feature distributions)
    - Temporal analysis (time series patterns)
    - Network analysis (entity relationships)
    """
    
    def __init__(self):
        self.text_patterns = {}
        self.numeric_patterns = {}
        self.temporal_patterns = {}
        self.network_patterns = {}
        
    def analyze_text_patterns(
        self,
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """Analyze text-based patterns."""
        from ..learning import UniversalArchetypeLearner
        
        learner = UniversalArchetypeLearner()
        patterns = learner.discover_patterns(texts, outcomes)
        
        self.text_patterns = patterns
        return patterns
    
    def analyze_numeric_patterns(
        self,
        features: np.ndarray,
        feature_names: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Analyze numeric feature patterns.
        
        Parameters
        ----------
        features : ndarray
            Feature matrix
        feature_names : list
            Feature names
        outcomes : ndarray
            Outcomes
        
        Returns
        -------
        dict
            Numeric patterns
        """
        numeric_patterns = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_values = features[:, i]
            
            # Skip constant features
            if np.std(feature_values) < 1e-6:
                continue
            
            # Correlation with outcome
            if len(np.unique(outcomes)) > 1:
                corr, p_value = stats.pearsonr(feature_values, outcomes)
                
                if p_value < 0.05:
                    # Significant pattern
                    numeric_patterns[feature_name] = {
                        'type': 'numeric',
                        'correlation': corr,
                        'p_value': p_value,
                        'mean': np.mean(feature_values),
                        'std': np.std(feature_values),
                        'range': (np.min(feature_values), np.max(feature_values))
                    }
        
        self.numeric_patterns = numeric_patterns
        return numeric_patterns
    
    def analyze_temporal_patterns(
        self,
        sequences: List[List[str]],
        timestamps: np.ndarray,
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Analyze temporal/sequential patterns.
        
        Parameters
        ----------
        sequences : list of lists
            Sequences of events/tokens
        timestamps : ndarray
            Timestamps
        outcomes : ndarray
            Outcomes
        
        Returns
        -------
        dict
            Temporal patterns
        """
        temporal_patterns = {}
        
        # Find common sequences
        sequence_counts = defaultdict(int)
        for seq in sequences:
            # Look at bigrams and trigrams
            for i in range(len(seq) - 1):
                bigram = tuple(seq[i:i+2])
                sequence_counts[bigram] += 1
            
            if len(seq) >= 3:
                for i in range(len(seq) - 2):
                    trigram = tuple(seq[i:i+3])
                    sequence_counts[trigram] += 1
        
        # Check which sequences predict outcomes
        for sequence, count in sequence_counts.items():
            if count < 5:  # Too rare
                continue
            
            # Find instances with this sequence
            has_sequence = np.array([
                sequence in [tuple(seq[i:i+len(sequence)]) for i in range(len(seq) - len(sequence) + 1)]
                for seq in sequences
            ])
            
            if np.sum(has_sequence) > 0 and len(np.unique(outcomes)) > 1:
                # Calculate correlation
                corr = abs(np.corrcoef(has_sequence.astype(float), outcomes)[0, 1])
                
                if corr > 0.1:
                    temporal_patterns[str(sequence)] = {
                        'type': 'temporal',
                        'sequence': list(sequence),
                        'frequency': count / len(sequences),
                        'correlation': corr
                    }
        
        self.temporal_patterns = temporal_patterns
        return temporal_patterns
    
    def analyze_network_patterns(
        self,
        entity_pairs: List[Tuple[str, str]],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Analyze network/relationship patterns.
        
        Parameters
        ----------
        entity_pairs : list of tuples
            Entity relationships
        outcomes : ndarray
            Outcomes
        
        Returns
        -------
        dict
            Network patterns
        """
        import networkx as nx
        
        # Build network
        G = nx.Graph()
        for i, (entity1, entity2) in enumerate(entity_pairs):
            G.add_edge(entity1, entity2, outcome=outcomes[i])
        
        network_patterns = {}
        
        # Degree centrality patterns
        centrality = nx.degree_centrality(G)
        
        # Find high-centrality nodes
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        for node, cent in sorted_nodes[:10]:  # Top 10
            # Get outcomes for edges involving this node
            node_outcomes = [
                G[node][neighbor]['outcome']
                for neighbor in G.neighbors(node)
                if 'outcome' in G[node][neighbor]
            ]
            
            if len(node_outcomes) > 0:
                network_patterns[f"centrality_{node}"] = {
                    'type': 'network',
                    'node': node,
                    'centrality': cent,
                    'avg_outcome': np.mean(node_outcomes),
                    'n_connections': len(node_outcomes)
                }
        
        self.network_patterns = network_patterns
        return network_patterns
    
    def combine_modalities(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict]:
        """
        Combine patterns from all modalities.
        
        Parameters
        ----------
        weights : dict, optional
            Modality weights (modality_name -> weight)
        
        Returns
        -------
        dict
            Combined patterns
        """
        if weights is None:
            weights = {
                'text': 0.4,
                'numeric': 0.3,
                'temporal': 0.2,
                'network': 0.1
            }
        
        combined = {}
        
        # Add text patterns
        for name, data in self.text_patterns.items():
            data_copy = data.copy()
            data_copy['modality'] = 'text'
            data_copy['combined_score'] = data.get('correlation', 0.0) * weights['text']
            combined[f"text_{name}"] = data_copy
        
        # Add numeric patterns
        for name, data in self.numeric_patterns.items():
            data_copy = data.copy()
            data_copy['modality'] = 'numeric'
            data_copy['combined_score'] = abs(data['correlation']) * weights['numeric']
            combined[f"numeric_{name}"] = data_copy
        
        # Add temporal patterns
        for name, data in self.temporal_patterns.items():
            data_copy = data.copy()
            data_copy['modality'] = 'temporal'
            data_copy['combined_score'] = data['correlation'] * weights['temporal']
            combined[f"temporal_{name}"] = data_copy
        
        # Add network patterns
        for name, data in self.network_patterns.items():
            data_copy = data.copy()
            data_copy['modality'] = 'network'
            data_copy['combined_score'] = data['centrality'] * weights['network']
            combined[f"network_{name}"] = data_copy
        
        return combined
    
    def get_top_patterns_by_modality(
        self,
        n_per_modality: int = 5
    ) -> Dict[str, List[Tuple[str, Dict]]]:
        """
        Get top patterns from each modality.
        
        Parameters
        ----------
        n_per_modality : int
            Number per modality
        
        Returns
        -------
        dict
            Modality -> top patterns
        """
        top_patterns = {}
        
        # Text
        sorted_text = sorted(
            self.text_patterns.items(),
            key=lambda x: abs(x[1].get('correlation', 0)),
            reverse=True
        )
        top_patterns['text'] = sorted_text[:n_per_modality]
        
        # Numeric
        sorted_numeric = sorted(
            self.numeric_patterns.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )
        top_patterns['numeric'] = sorted_numeric[:n_per_modality]
        
        # Temporal
        sorted_temporal = sorted(
            self.temporal_patterns.items(),
            key=lambda x: x[1]['correlation'],
            reverse=True
        )
        top_patterns['temporal'] = sorted_temporal[:n_per_modality]
        
        # Network
        sorted_network = sorted(
            self.network_patterns.items(),
            key=lambda x: x[1]['centrality'],
            reverse=True
        )
        top_patterns['network'] = sorted_network[:n_per_modality]
        
        return top_patterns

