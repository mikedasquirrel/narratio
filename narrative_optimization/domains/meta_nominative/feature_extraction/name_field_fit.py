"""
Name-Field Fit Calculator for Meta-Nominative Analysis

Core innovation: Calculate how well researcher names match their research topics.
Uses four methods: phonetic, semantic, exact, and initial matching.
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import jellyfish  # For phonetic algorithms
from difflib import SequenceMatcher
import numpy as np


class NameFieldFitCalculator:
    """Calculate name-field fit scores for researchers."""
    
    def __init__(self):
        """Initialize calculator with topic keywords."""
        # Keywords for each research topic
        self.topic_keywords = {
            'dentists': ['dentist', 'dental', 'teeth', 'orthodont', 'dent'],
            'hurricanes': ['hurricane', 'storm', 'weather', 'cyclone', 'wind', 'hurr'],
            'lawyers': ['lawyer', 'attorney', 'legal', 'law', 'counsel', 'court'],
            'doctors': ['doctor', 'physician', 'medical', 'medicine', 'doc', 'medic'],
            'occupations': ['occupation', 'career', 'profession', 'job', 'work', 'occup'],
            'geography': ['geography', 'geographic', 'location', 'place', 'city', 'state', 'geo'],
            'marriage': ['marriage', 'marry', 'mate', 'spouse', 'partner', 'marri'],
            'brands': ['brand', 'product', 'company', 'business', 'market'],
            'academic': ['academic', 'researcher', 'scientist', 'scholar', 'professor', 'research', 'science'],
            'names_general': ['name', 'naming', 'nomenclature', 'nominative', 'identity']
        }
        
        # Semantic clusters (related concepts)
        self.semantic_clusters = {
            'medical': ['doctor', 'physician', 'nurse', 'surgeon', 'dentist', 'medic', 'clinic', 'hospital'],
            'legal': ['lawyer', 'attorney', 'judge', 'court', 'law', 'legal', 'justice'],
            'scientific': ['scientist', 'researcher', 'professor', 'scholar', 'academic', 'research', 'science'],
            'business': ['business', 'company', 'corporate', 'brand', 'market', 'commerce'],
            'geography': ['place', 'location', 'city', 'state', 'country', 'region', 'area', 'geo'],
            'professional': ['professional', 'career', 'occupation', 'job', 'work', 'employ']
        }
    
    def calculate_fit(self, researcher_name: str, research_topic: str) -> Dict[str, float]:
        """
        Calculate comprehensive name-field fit score.
        
        Args:
            researcher_name: Full name of researcher (e.g., "Dennis Smith")
            research_topic: Research topic (e.g., "dentists, occupations")
            
        Returns:
            Dictionary with fit scores and overall score (0-100)
        """
        # Parse name
        first_name, last_name = self._parse_name(researcher_name)
        
        # Extract keywords from topic
        topic_keywords = self._extract_topic_keywords(research_topic)
        
        # Calculate four types of fit
        phonetic_score = self._calculate_phonetic_fit(first_name, last_name, topic_keywords)
        semantic_score = self._calculate_semantic_fit(first_name, last_name, topic_keywords)
        exact_score = self._calculate_exact_fit(first_name, last_name, topic_keywords)
        initial_score = self._calculate_initial_fit(first_name, last_name, topic_keywords)
        
        # Weighted combination
        weights = {
            'phonetic': 0.35,  # Most important - sound similarity
            'semantic': 0.30,  # Meaning similarity
            'exact': 0.25,     # Perfect matches (rare but strong)
            'initial': 0.10    # Weakest signal
        }
        
        overall_score = (
            weights['phonetic'] * phonetic_score +
            weights['semantic'] * semantic_score +
            weights['exact'] * exact_score +
            weights['initial'] * initial_score
        )
        
        return {
            'overall_fit': overall_score,
            'phonetic_fit': phonetic_score,
            'semantic_fit': semantic_score,
            'exact_fit': exact_score,
            'initial_fit': initial_score,
            'researcher_name': researcher_name,
            'research_topic': research_topic,
            'first_name': first_name,
            'last_name': last_name,
            'topic_keywords': topic_keywords
        }
    
    def _parse_name(self, full_name: str) -> Tuple[str, str]:
        """Parse full name into first and last names."""
        parts = full_name.strip().split()
        if len(parts) == 0:
            return "", ""
        elif len(parts) == 1:
            return parts[0], ""
        else:
            # First word is first name, last word is last name
            return parts[0], parts[-1]
    
    def _extract_topic_keywords(self, topic: str) -> List[str]:
        """Extract keywords from research topic."""
        # Split comma-separated topics
        topics = [t.strip().lower() for t in topic.split(',')]
        
        # Get all related keywords
        keywords = set()
        for topic in topics:
            # Direct topic keywords
            if topic in self.topic_keywords:
                keywords.update(self.topic_keywords[topic])
            
            # Also add the topic itself
            keywords.add(topic)
        
        return list(keywords)
    
    def _calculate_phonetic_fit(self, first_name: str, last_name: str, keywords: List[str]) -> float:
        """
        Calculate phonetic similarity using multiple algorithms.
        
        Uses Soundex, Metaphone, and edit distance to find sound-alike matches.
        """
        if not first_name and not last_name:
            return 0.0
        
        max_score = 0.0
        
        # Check both first and last name
        for name in [first_name.lower(), last_name.lower()]:
            if not name:
                continue
            
            for keyword in keywords:
                # Soundex similarity
                try:
                    soundex_match = jellyfish.soundex(name) == jellyfish.soundex(keyword)
                    if soundex_match:
                        max_score = max(max_score, 100.0)
                        continue
                except:
                    pass
                
                # Metaphone similarity
                try:
                    metaphone_match = jellyfish.metaphone(name) == jellyfish.metaphone(keyword)
                    if metaphone_match:
                        max_score = max(max_score, 95.0)
                        continue
                except:
                    pass
                
                # Levenshtein distance (edit distance)
                distance = jellyfish.levenshtein_distance(name, keyword)
                max_len = max(len(name), len(keyword))
                if max_len > 0:
                    similarity = (1 - distance / max_len) * 100
                    # Only consider if relatively similar
                    if similarity > 60:
                        max_score = max(max_score, similarity * 0.8)
                
                # Jaro-Winkler similarity (good for typos)
                jaro_sim = jellyfish.jaro_winkler_similarity(name, keyword)
                if jaro_sim > 0.85:
                    max_score = max(max_score, jaro_sim * 85)
        
        return min(100.0, max_score)
    
    def _calculate_semantic_fit(self, first_name: str, last_name: str, keywords: List[str]) -> float:
        """
        Calculate semantic similarity using word meaning.
        
        Checks if name parts share semantic clusters with keywords.
        """
        if not first_name and not last_name:
            return 0.0
        
        max_score = 0.0
        
        # Check both names
        for name in [first_name.lower(), last_name.lower()]:
            if not name:
                continue
            
            # Check against semantic clusters
            for cluster_name, cluster_words in self.semantic_clusters.items():
                name_in_cluster = any(name in word or word in name for word in cluster_words)
                
                if name_in_cluster:
                    # Check if any keyword is also in this cluster
                    for keyword in keywords:
                        keyword_in_cluster = any(keyword in word or word in keyword for word in cluster_words)
                        if keyword_in_cluster:
                            max_score = max(max_score, 80.0)
            
            # Direct substring matching (partial semantic match)
            for keyword in keywords:
                # Check if name contains keyword or vice versa
                if len(name) >= 4 and len(keyword) >= 4:
                    if name in keyword or keyword in name:
                        overlap = min(len(name), len(keyword)) / max(len(name), len(keyword))
                        max_score = max(max_score, overlap * 70)
        
        return min(100.0, max_score)
    
    def _calculate_exact_fit(self, first_name: str, last_name: str, keywords: List[str]) -> float:
        """
        Calculate exact match score.
        
        Checks if name exactly matches a keyword (e.g., "Dr. Lawyer" studying lawyers).
        """
        if not first_name and not last_name:
            return 0.0
        
        first_lower = first_name.lower()
        last_lower = last_name.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact match
            if first_lower == keyword_lower or last_lower == keyword_lower:
                return 100.0
            
            # Very close match (differ by 1 character)
            for name in [first_lower, last_lower]:
                if abs(len(name) - len(keyword_lower)) <= 1:
                    matcher = SequenceMatcher(None, name, keyword_lower)
                    if matcher.ratio() > 0.9:
                        return 90.0
        
        return 0.0
    
    def _calculate_initial_fit(self, first_name: str, last_name: str, keywords: List[str]) -> float:
        """
        Calculate initial letter match.
        
        Checks if first letter of name matches first letter of keyword.
        """
        if not first_name and not last_name:
            return 0.0
        
        max_score = 0.0
        
        # Check both names
        for name in [first_name, last_name]:
            if not name:
                continue
            
            first_letter = name[0].lower()
            
            for keyword in keywords:
                if keyword and keyword[0].lower() == first_letter:
                    # Initial match found
                    # Score higher if names are longer (less likely by chance)
                    length_bonus = min(len(name), len(keyword)) / 10
                    max_score = max(max_score, 40 + length_bonus * 20)
        
        return min(100.0, max_score)
    
    def calculate_all_fits(self, researchers: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calculate fit scores for all researchers.
        
        Args:
            researchers: Dictionary of researcher data
            
        Returns:
            Dictionary with fit scores added
        """
        print(f"\n{'='*80}")
        print("CALCULATING NAME-FIELD FIT SCORES")
        print(f"{'='*80}\n")
        
        fit_scores = {}
        
        for name, data in researchers.items():
            topics = data.get('topics_studied', [])
            
            if not topics:
                # Use 'general' if no specific topics
                topic_str = 'names_general'
            else:
                topic_str = ', '.join(topics)
            
            fit = self.calculate_fit(name, topic_str)
            fit_scores[name] = fit
            
            # Add to researcher data
            data['name_field_fit'] = fit
            
            # Print interesting cases
            if fit['overall_fit'] > 30:
                print(f"  {name}: {fit['overall_fit']:.1f} fit")
                print(f"    Topic: {topic_str}")
                print(f"    Scores: Phonetic={fit['phonetic_fit']:.1f}, Semantic={fit['semantic_fit']:.1f}, Exact={fit['exact_fit']:.1f}, Initial={fit['initial_fit']:.1f}")
        
        print(f"\n✓ Calculated fit scores for {len(fit_scores)} researchers")
        
        # Summary statistics
        overall_scores = [f['overall_fit'] for f in fit_scores.values()]
        print(f"\nFit score distribution:")
        print(f"  Mean: {np.mean(overall_scores):.1f}")
        print(f"  Median: {np.median(overall_scores):.1f}")
        print(f"  Range: {np.min(overall_scores):.1f} - {np.max(overall_scores):.1f}")
        print(f"  High fit (>50): {sum(1 for s in overall_scores if s > 50)}")
        print(f"  Medium fit (20-50): {sum(1 for s in overall_scores if 20 <= s <= 50)}")
        print(f"  Low fit (<20): {sum(1 for s in overall_scores if s < 20)}")
        
        return fit_scores
    
    def save_fit_scores(self, fit_scores: Dict[str, Dict], output_path: Optional[Path] = None) -> Path:
        """Save fit scores to JSON."""
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative' / 'name_field_fit_scores.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable = {
            name: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in score.items()}
            for name, score in fit_scores.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump({
                'fit_scores': serializable,
                'total_researchers': len(fit_scores),
                'statistics': {
                    'mean_fit': float(np.mean([s['overall_fit'] for s in fit_scores.values()])),
                    'median_fit': float(np.median([s['overall_fit'] for s in fit_scores.values()])),
                    'max_fit': float(np.max([s['overall_fit'] for s in fit_scores.values()])),
                    'min_fit': float(np.min([s['overall_fit'] for s in fit_scores.values()]))
                }
            }, f, indent=2)
        
        print(f"\n✓ Saved fit scores to: {output_path}")
        return output_path


def main():
    """Calculate name-field fit for all researchers."""
    from pathlib import Path
    import json
    
    # Load researcher metadata
    # Get to novelization root, then to data
    data_dir = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative'
    metadata_path = data_dir / 'researchers_metadata.json'
    
    if not metadata_path.exists():
        print(f"Error: Researcher metadata not found at {metadata_path}")
        print("Run data collection first!")
        return
    
    with open(metadata_path) as f:
        data = json.load(f)
        researchers = data['researchers']
    
    print(f"Loaded {len(researchers)} researchers")
    
    # Calculate fits
    calculator = NameFieldFitCalculator()
    fit_scores = calculator.calculate_all_fits(researchers)
    calculator.save_fit_scores(fit_scores)
    
    # Update researcher metadata with fit scores
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✓ Name-field fit calculation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

