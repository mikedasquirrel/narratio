"""
Narrative Free Will Analyzer

Complete pipeline for analyzing narratives for free will vs determinism.
Integrates all transformers and analysis methods.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..transformers.free_will_analysis import FreeWillAnalysisTransformer


class NarrativeFreeWillAnalyzer:
    """
    Complete analysis pipeline for free will vs determinism in narratives.
    
    Performs comprehensive multi-dimensional analysis:
    1. Structural analysis (Sentence Transformers)
    2. Causal structure (spaCy dependency parsing)
    3. Semantic field analysis (fate vs choice language)
    4. Temporal dynamics (future vs past orientation)
    5. Information theory (predictability/entropy)
    6. Network structure (causal graphs)
    7. Observability analysis (visible vs hidden causality)
    """
    
    def __init__(
        self,
        use_sentence_transformers: bool = True,
        use_spacy: bool = True,
        model_name: str = 'all-MiniLM-L6-v2',
        spacy_model: str = 'en_core_web_sm',
        extract_causal_graphs: bool = True,
        track_observability: bool = True,
        temporal_weight: float = 0.30,
        semantic_weight: float = 0.40,
        predictability_weight: float = 0.30,
        custom_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            use_sentence_transformers: Use SentenceTransformers for embeddings
            use_spacy: Use spaCy for dependency parsing
            model_name: SentenceTransformer model name
            spacy_model: spaCy model name
            extract_causal_graphs: Build causal network graphs
            track_observability: Track visible vs hidden causality
            temporal_weight: Weight for temporal component (default 0.30)
            semantic_weight: Weight for semantic component (default 0.40)
            predictability_weight: Weight for predictability component (default 0.30)
            custom_weights: Optional dict to override any weights
        """
        self.transformer = FreeWillAnalysisTransformer(
            use_sentence_transformers=use_sentence_transformers,
            use_spacy=use_spacy,
            model_name=model_name,
            spacy_model=spacy_model,
            extract_causal_graphs=extract_causal_graphs,
            track_observability=track_observability,
            temporal_weight=temporal_weight,
            semantic_weight=semantic_weight,
            predictability_weight=predictability_weight,
            custom_weights=custom_weights
        )
        
        self.fitted = False
        self.corpus_stats = {}
    
    def fit(self, stories: List[str]):
        """
        Fit analyzer on corpus of narratives.
        
        Args:
            stories: List of narrative texts
        
        Returns:
            self
        """
        self.transformer.fit(stories)
        self.fitted = True
        
        # Calculate corpus-level statistics
        self._calculate_corpus_stats(stories)
        
        return self
    
    def analyze(self, story_text: str) -> Dict[str, Any]:
        """
        Complete analysis of single narrative.
        
        Args:
            story_text: Narrative text to analyze
        
        Returns:
            Dict containing all analysis results
        """
        if not self.fitted:
            raise ValueError("Analyzer must be fitted before analysis. Call fit() first.")
        
        # Extract features
        features = self.transformer.transform([story_text])[0]
        feature_names = self.transformer.get_feature_names()
        
        # Create feature dictionary
        feature_dict = dict(zip(feature_names, features))
        
        # Calculate scores
        determinism_score = self.transformer.calculate_narrative_determinism_score(story_text)
        agency_score = 1.0 - determinism_score
        
        # Extract individual components
        temporal_features = {
            'future_orientation': feature_dict.get('temporal_future_orientation', 0.0),
            'past_orientation': feature_dict.get('temporal_past_orientation', 0.0),
            'present_orientation': feature_dict.get('temporal_present_orientation', 0.0)
        }
        
        semantic_features = {
            'fate_density': feature_dict.get('semantic_density_fate', 0.0),
            'choice_density': feature_dict.get('semantic_density_choice', 0.0),
            'agency_density': feature_dict.get('semantic_density_agency', 0.0),
            'determinism_balance': feature_dict.get('determinism_balance', 0.0)
        }
        
        info_features = {
            'entropy': feature_dict.get('info_word_entropy', 0.0),
            'predictability': feature_dict.get('info_predictability', 0.0),
            'redundancy': feature_dict.get('info_redundancy', 0.0)
        }
        
        agency_features = {
            'n_agents': feature_dict.get('agency_n_agents', 0.0),
            'n_patients': feature_dict.get('agency_n_patients', 0.0),
            'free_will_score': feature_dict.get('agency_free_will_score', 0.0)
        }
        
        graph_features = {
            'path_dependency': feature_dict.get('graph_path_dependency', 0.0),
            'branching_factor': feature_dict.get('graph_branching_factor', 0.0),
            'deterministic_ratio': feature_dict.get('graph_deterministic_ratio', 0.0)
        }
        
        observability_features = {
            'explicit_ratio': feature_dict.get('observability_explicit_ratio', 0.0),
            'hidden_ratio': feature_dict.get('observability_hidden_ratio', 0.0),
            'omniscient_ratio': feature_dict.get('observability_omniscient_ratio', 0.0)
        }
        
        return {
            'determinism_score': float(determinism_score),
            'agency_score': float(agency_score),
            'free_will_ratio': feature_dict.get('agency_ratio', 0.0),
            'inevitability_score': feature_dict.get('composite_inevitability_score', 0.0),
            'temporal_features': temporal_features,
            'semantic_features': semantic_features,
            'information_theory': info_features,
            'agency_analysis': agency_features,
            'causal_structure': graph_features,
            'observability': observability_features,
            'all_features': feature_dict
        }
    
    def analyze_corpus(self, stories: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze collection of narratives.
        
        Args:
            stories: List of narrative texts
        
        Returns:
            List of analysis results for each story
        """
        if not self.fitted:
            self.fit(stories)
        
        results = []
        for story in stories:
            analysis = self.analyze(story)
            results.append(analysis)
        
        return results
    
    def compare_narrative_to_reality(
        self, 
        fictional_story: str, 
        real_world_events: str
    ) -> Dict[str, float]:
        """
        Measure structural similarity between fiction and reality.
        
        Args:
            fictional_story: Fictional narrative text
            real_world_events: Real-world event description
        
        Returns:
            Dict with similarity metrics
        """
        if not self.fitted:
            raise ValueError("Analyzer must be fitted first")
        
        # Analyze both
        fiction_analysis = self.analyze(fictional_story)
        reality_analysis = self.analyze(real_world_events)
        
        # Extract embeddings if available
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.transformer.use_sentence_transformers:
            fiction_embedding = self.transformer.semantic_model.encode(fictional_story)
            reality_embedding = self.transformer.semantic_model.encode(real_world_events)
            
            # Cosine similarity
            from scipy.spatial.distance import cosine
            semantic_similarity = 1.0 - cosine(fiction_embedding, reality_embedding)
        else:
            semantic_similarity = 0.0
        
        # Structural similarity (compare feature vectors)
        fiction_features = self.transformer.transform([fictional_story])[0]
        reality_features = self.transformer.transform([real_world_events])[0]
        
        from scipy.spatial.distance import cosine
        structural_similarity = 1.0 - cosine(fiction_features, reality_features)
        
        # Determinism score similarity
        determinism_similarity = 1.0 - abs(
            fiction_analysis['determinism_score'] - 
            reality_analysis['determinism_score']
        )
        
        return {
            'semantic_similarity': float(semantic_similarity),
            'structural_similarity': float(structural_similarity),
            'determinism_similarity': float(determinism_similarity),
            'maps_to_reality': semantic_similarity > 0.7 and structural_similarity > 0.6,
            'fiction_determinism': fiction_analysis['determinism_score'],
            'reality_determinism': reality_analysis['determinism_score']
        }
    
    def predict_outcome_from_structure(
        self, 
        beginning: str, 
        ending: str
    ) -> Dict[str, float]:
        """
        Test if narrative structure predicts outcomes.
        
        High similarity between beginning and ending = deterministic narrative.
        Low similarity = free will/surprise narrative.
        
        Args:
            beginning: Beginning of narrative
            ending: Ending of narrative
        
        Returns:
            Dict with prediction metrics
        """
        if not self.fitted:
            raise ValueError("Analyzer must be fitted first")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.transformer.use_sentence_transformers:
            beginning_embedding = self.transformer.semantic_model.encode(beginning)
            ending_embedding = self.transformer.semantic_model.encode(ending)
            
            from scipy.spatial.distance import cosine
            similarity = 1.0 - cosine(beginning_embedding, ending_embedding)
        else:
            similarity = 0.0
        
        # Analyze both parts
        beginning_analysis = self.analyze(beginning)
        ending_analysis = self.analyze(ending)
        
        # High similarity = deterministic (structure predicts outcome)
        # Low similarity = free will (surprise ending)
        determinism_from_structure = similarity
        
        return {
            'beginning_ending_similarity': float(similarity),
            'determinism_from_structure': float(determinism_from_structure),
            'beginning_determinism': beginning_analysis['determinism_score'],
            'ending_determinism': ending_analysis['determinism_score'],
            'is_deterministic': similarity > 0.7,
            'is_free_will': similarity < 0.3
        }
    
    def cluster_by_determinism(self, stories: List[str]) -> Dict[str, Any]:
        """
        Cluster narratives by determinism score.
        
        Args:
            stories: List of narrative texts
        
        Returns:
            Dict with clustering results
        """
        if not self.fitted:
            self.fit(stories)
        
        # Analyze all stories
        results = self.analyze_corpus(stories)
        
        # Extract determinism scores
        determinism_scores = [r['determinism_score'] for r in results]
        
        # Simple clustering: high, medium, low determinism
        high_determinism = [i for i, score in enumerate(determinism_scores) if score > 0.7]
        medium_determinism = [i for i, score in enumerate(determinism_scores) if 0.3 <= score <= 0.7]
        low_determinism = [i for i, score in enumerate(determinism_scores) if score < 0.3]
        
        return {
            'high_determinism': {
                'indices': high_determinism,
                'count': len(high_determinism),
                'avg_score': np.mean([determinism_scores[i] for i in high_determinism]) if high_determinism else 0.0
            },
            'medium_determinism': {
                'indices': medium_determinism,
                'count': len(medium_determinism),
                'avg_score': np.mean([determinism_scores[i] for i in medium_determinism]) if medium_determinism else 0.0
            },
            'low_determinism': {
                'indices': low_determinism,
                'count': len(low_determinism),
                'avg_score': np.mean([determinism_scores[i] for i in low_determinism]) if low_determinism else 0.0
            },
            'all_scores': determinism_scores
        }
    
    def analyze_nominative_determinism(self, story_text: str) -> Dict[str, Any]:
        """
        Analyze how naming patterns encode agency vs determinism.
        
        Args:
            story_text: Narrative text to analyze
        
        Returns:
            Dict containing:
            - character_agency_scores: Agency implied by character names
            - naming_pattern_analysis: How naming evolves
            - nominative_determinism_score: Overall score
            - naming_features: Detailed nominative features
        """
        if not self.fitted:
            raise ValueError("Analyzer must be fitted first")
        
        # Extract features
        features = self.transformer.transform([story_text])[0]
        feature_names = self.transformer.get_feature_names()
        
        # Create feature dictionary
        feature_dict = dict(zip(feature_names, features))
        
        # Extract nominative-specific features
        nominative_features = {
            'proper_name_density': feature_dict.get('nominative_proper_name_density', 0.0),
            'generic_label_ratio': feature_dict.get('nominative_generic_label_ratio', 0.0),
            'title_frequency': feature_dict.get('nominative_title_frequency', 0.0),
            'name_consistency': feature_dict.get('nominative_name_consistency', 0.0),
            'identity_assertions': feature_dict.get('nominative_identity_assertions', 0.0),
            'categorical_density': feature_dict.get('nominative_categorical_density', 0.0),
            'agency_naming_density': feature_dict.get('nominative_agency_naming_density', 0.0),
            'naming_balance': feature_dict.get('nominative_naming_balance', 0.0)
        }
        
        # Extract naming evolution features
        naming_evolution = {
            'names_gained': feature_dict.get('naming_evolution_names_gained', 0.0),
            'names_lost': feature_dict.get('naming_evolution_names_lost', 0.0),
            'title_accumulation': feature_dict.get('naming_evolution_title_accumulation', 0.0),
            'identity_shift': feature_dict.get('naming_evolution_identity_shift', 0.0),
            'agency_evolution': feature_dict.get('naming_evolution_agency_evolution', 0.0),
            'naming_stability': feature_dict.get('naming_evolution_naming_stability', 0.0)
        }
        
        # Calculate character agency scores
        character_agency_scores = {
            'proper_name_score': nominative_features['proper_name_density'] * nominative_features['name_consistency'],
            'generic_label_score': nominative_features['generic_label_ratio'] * (1 - nominative_features['name_consistency']),
            'title_determinism': nominative_features['title_frequency'],
            'identity_strength': nominative_features['identity_assertions']
        }
        
        # Calculate overall nominative determinism score
        # High = deterministic (generic labels, titles, low consistency)
        # Low = agentic (proper names, high consistency, agency language)
        deterministic_signals = (
            nominative_features['generic_label_ratio'] + 
            nominative_features['title_frequency'] + 
            nominative_features['categorical_density']
        ) / 3.0
        
        agentic_signals = (
            nominative_features['proper_name_density'] + 
            nominative_features['name_consistency'] + 
            nominative_features['agency_naming_density']
        ) / 3.0
        
        nominative_determinism_score = deterministic_signals / (deterministic_signals + agentic_signals + 1e-6)
        
        # Analyze naming patterns
        naming_pattern_analysis = {
            'pattern': 'stable' if naming_evolution['naming_stability'] > 0.7 else 
                      'evolving' if naming_evolution['naming_stability'] < 0.3 else 'mixed',
            'direction': 'gaining_agency' if naming_evolution['agency_evolution'] > 0.1 else
                        'losing_agency' if naming_evolution['agency_evolution'] < -0.1 else 'stable',
            'title_trend': 'accumulating' if naming_evolution['title_accumulation'] > 0.1 else 'stable',
            'identity_stability': 'shifting' if naming_evolution['identity_shift'] > 0.3 else 'stable'
        }
        
        return {
            'character_agency_scores': character_agency_scores,
            'naming_pattern_analysis': naming_pattern_analysis,
            'nominative_determinism_score': float(nominative_determinism_score),
            'naming_features': nominative_features,
            'naming_evolution': naming_evolution,
            'interpretation': self._generate_nominative_interpretation(
                nominative_determinism_score,
                nominative_features,
                naming_evolution,
                naming_pattern_analysis
            )
        }
    
    def _generate_nominative_interpretation(
        self,
        score: float,
        features: Dict[str, float],
        evolution: Dict[str, float],
        patterns: Dict[str, str]
    ) -> str:
        """Generate interpretation of nominative analysis."""
        interpretation = f"Nominative Determinism Score: {score:.3f}\n\n"
        
        if score > 0.7:
            interpretation += "Strong deterministic naming patterns detected:\n"
            if features['generic_label_ratio'] > 0.01:
                interpretation += "- Characters often referred to by generic labels rather than names\n"
            if features['title_frequency'] > 0.005:
                interpretation += "- Deterministic titles ('The Chosen', 'The Fated') present\n"
            if features['name_consistency'] < 0.5:
                interpretation += "- Low name consistency suggests reduced individual agency\n"
        elif score < 0.3:
            interpretation += "Strong agentic naming patterns detected:\n"
            if features['proper_name_density'] > 0.02:
                interpretation += "- High density of proper names suggests individual agency\n"
            if features['agency_naming_density'] > 0.01:
                interpretation += "- Agency-affirming language in naming\n"
            if features['name_consistency'] > 0.7:
                interpretation += "- Consistent naming suggests stable identity and agency\n"
        else:
            interpretation += "Mixed naming patterns suggest narrative tension between fate and choice.\n"
        
        interpretation += f"\nNaming Evolution: {patterns['pattern']}"
        if patterns['direction'] == 'gaining_agency':
            interpretation += " - Characters gain names and agency through narrative"
        elif patterns['direction'] == 'losing_agency':
            interpretation += " - Characters lose individual identity, becoming more generic"
        
        return interpretation
    
    def _calculate_corpus_stats(self, stories: List[str]):
        """Calculate corpus-level statistics."""
        if not stories:
            return
        
        # Analyze sample for statistics
        sample_size = min(100, len(stories))
        sample_stories = stories[:sample_size]
        
        determinism_scores = []
        agency_scores = []
        
        for story in sample_stories:
            analysis = self.analyze(story)
            determinism_scores.append(analysis['determinism_score'])
            agency_scores.append(analysis['agency_score'])
        
        self.corpus_stats = {
            'mean_determinism': float(np.mean(determinism_scores)),
            'std_determinism': float(np.std(determinism_scores)),
            'mean_agency': float(np.mean(agency_scores)),
            'std_agency': float(np.std(agency_scores)),
            'sample_size': sample_size,
            'total_stories': len(stories)
        }
    
    def get_interpretation(self) -> str:
        """Get interpretation of learned patterns."""
        return self.transformer.get_interpretation()
    
    def export_results(self, results: List[Dict[str, Any]], filepath: str):
        """
        Export analysis results to JSON file.
        
        Args:
            results: List of analysis results
            filepath: Path to output file
        """
        export_data = {
            'corpus_stats': self.corpus_stats,
            'results': results,
            'metadata': {
                'n_stories': len(results),
                'transformer_config': {
                    'use_sentence_transformers': self.transformer.use_sentence_transformers,
                    'use_spacy': self.transformer.use_spacy,
                    'extract_causal_graphs': self.transformer.extract_causal_graphs,
                    'track_observability': self.transformer.track_observability
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {filepath}")

