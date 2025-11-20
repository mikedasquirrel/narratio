"""
Novels Data Loader

Loads novels dataset and extracts:
- Narrative text (plot summaries, excerpts)
- Nominatives (character names, author names, book titles)
- Character roles and relationships
- Plot relative to characters
- Multi-task outcomes (ratings, awards, bestseller, acclaim, sales)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re


class NovelsDataLoader:
    """
    Loads and processes novels dataset with comprehensive nominative extraction.
    
    Extracts:
    - All character names (from plot summaries and metadata)
    - Author names
    - Book titles
    - Character roles and relationships
    - Plot summaries mentioning characters
    - Full narrative text combining all sources
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize novels data loader."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent / 'data'
        else:
            self.data_dir = Path(data_dir)
        
        self.dataset_path = self.data_dir / 'novels_dataset.json'
    
    def load_full_dataset(self, use_cache: bool = True, filter_data: bool = False) -> List[Dict[str, Any]]:
        """
        Load full novels dataset with all nominative information.
        
        Parameters
        ----------
        use_cache : bool
            Use cached processed data if available
        filter_data : bool
            Filter out incomplete entries
            
        Returns
        -------
        novels : list of dict
            Complete novels with all extracted information
        """
        print(f"Loading novels dataset from {self.dataset_path}...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"✓ Loaded {len(raw_data)} raw novels")
        
        # Process each novel to extract nominatives and create full narratives
        processed_novels = []
        
        for novel in raw_data:
            processed = self._process_novel(novel)
            if processed:
                processed_novels.append(processed)
        
        print(f"✓ Processed {len(processed_novels)} novels")
        
        return processed_novels
    
    def _process_novel(self, novel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single novel to extract all nominative information."""
        try:
            # Extract nominatives
            nominatives = self._extract_nominatives(novel)
            
            # Extract character information
            characters = self._extract_characters(novel, nominatives)
            
            # Create full narrative text
            full_narrative = self._create_full_narrative(novel, characters)
            
            # Extract plot relative to characters
            plot_by_character = self._extract_plot_by_character(novel, characters)
            
            # Create outcome variables
            outcomes = self._create_outcomes(novel)
            
            # Combine all information
            processed = {
                # Core identification
                'title': novel.get('title', ''),
                'author': novel.get('author', ''),
                'publication_year': novel.get('publication_year'),
                
                # Narrative text
                'plot_summary': novel.get('plot_summary', ''),
                'full_narrative': full_narrative,
                
                # Nominatives (CRITICAL for nominative transformers)
                'author_name': novel.get('author', ''),
                'book_title': novel.get('title', ''),
                'character_names': characters.get('names', []),
                'character_roles': characters.get('roles', {}),
                'character_relationships': characters.get('relationships', []),
                'all_nominatives': nominatives.get('all_names', []),
                
                # Plot information
                'plot_summary': novel.get('plot_summary', ''),
                'plot_by_character': plot_by_character,
                'main_characters': characters.get('main_characters', []),
                'supporting_characters': characters.get('supporting_characters', []),
                
                # Ensemble information
                'ensemble_size': len(characters.get('names', [])),
                'character_diversity': self._calculate_character_diversity(characters),
                
                # Metadata
                'genres': novel.get('genres', []),
                'awards': novel.get('awards', []),
                'won_major_award': novel.get('won_major_award', False),
                'is_bestseller': novel.get('is_bestseller', False),
                'on_best_list': novel.get('on_best_list', False),
                
                # Outcomes (multi-task)
                'ratings': outcomes['ratings'],
                'awards_binary': outcomes['awards_binary'],
                'bestseller_binary': outcomes['bestseller_binary'],
                'critical_acclaim': outcomes['critical_acclaim'],
                'sales': outcomes['sales'],
                
                # For framework analysis
                'success_score': outcomes['composite_score'],
            }
            
            return processed
            
        except Exception as e:
            print(f"  ⚠️  Error processing novel {novel.get('title', 'Unknown')}: {e}")
            return None
    
    def _extract_nominatives(self, novel: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all nominative elements (names) from novel."""
        nominatives = {
            'author_name': novel.get('author', ''),
            'book_title': novel.get('title', ''),
            'character_names': novel.get('character_names', []),
            'all_names': []
        }
        
        # Add author name
        if nominatives['author_name']:
            nominatives['all_names'].append(nominatives['author_name'])
        
        # Add book title (extract words that might be names)
        title_words = re.findall(r'\b[A-Z][a-z]+\b', nominatives['book_title'])
        nominatives['all_names'].extend(title_words)
        
        # Add character names from metadata
        if nominatives['character_names']:
            nominatives['all_names'].extend(nominatives['character_names'])
        
        # Extract character names from plot summary using NER-like patterns
        plot_text = novel.get('plot_summary', '') + ' ' + novel.get('full_narrative', '')
        extracted_names = self._extract_names_from_text(plot_text)
        nominatives['all_names'].extend(extracted_names)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in nominatives['all_names']:
            name_lower = name.lower()
            if name_lower not in seen and len(name) > 1:
                seen.add(name_lower)
                unique_names.append(name)
        
        nominatives['all_names'] = unique_names
        nominatives['character_names'] = list(set(nominatives['character_names'] + extracted_names))
        
        return nominatives
    
    def _extract_names_from_text(self, text: str) -> List[str]:
        """
        Extract character names from text using pattern matching.
        
        Looks for:
        - Capitalized words that appear multiple times (likely character names)
        - Patterns like "Character Name said" or "Character Name's"
        - Proper nouns in dialogue or narrative
        """
        if not text:
            return []
        
        # Pattern 1: Capitalized words (potential names)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Pattern 2: "Name said" or "Name's" patterns
        name_patterns = re.findall(r'\b([A-Z][a-z]+)\s+(?:said|thought|wondered|asked|replied|exclaimed|whispered|shouted)', text)
        name_patterns.extend(re.findall(r'\b([A-Z][a-z]+)\'s\b', text))
        
        # Pattern 3: Common name patterns in literature
        # "Mr./Mrs./Ms./Dr. Name"
        title_patterns = re.findall(r'\b(?:Mr|Mrs|Ms|Dr|Professor|Captain|General)\.?\s+([A-Z][a-z]+)', text)
        
        # Combine all patterns
        all_candidates = capitalized_words + name_patterns + title_patterns
        
        # Filter: keep words that appear multiple times (more likely to be character names)
        word_counts = {}
        for word in all_candidates:
            # Skip common words that aren't names
            if word.lower() in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                continue
            if len(word) < 2:
                continue
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return names that appear at least 2 times (likely characters)
        character_names = [name for name, count in word_counts.items() if count >= 2]
        
        # Limit to top 20 most frequent (to avoid noise)
        character_names = sorted(character_names, key=lambda x: word_counts[x], reverse=True)[:20]
        
        return character_names
    
    def _extract_characters(self, novel: Dict[str, Any], nominatives: Dict[str, Any]) -> Dict[str, Any]:
        """Extract character information including roles and relationships."""
        characters = {
            'names': nominatives.get('character_names', []),
            'roles': {},
            'relationships': [],
            'main_characters': [],
            'supporting_characters': []
        }
        
        plot_text = novel.get('plot_summary', '') + ' ' + novel.get('full_narrative', '')
        
        # Determine main vs supporting characters based on frequency
        name_counts = {}
        for name in characters['names']:
            count = plot_text.lower().count(name.lower())
            name_counts[name] = count
        
        # Main characters: top 3-5 most mentioned
        sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
        main_count = min(5, len(sorted_names))
        characters['main_characters'] = [name for name, _ in sorted_names[:main_count]]
        characters['supporting_characters'] = [name for name, _ in sorted_names[main_count:]]
        
        # Assign roles based on patterns in text
        for name in characters['names']:
            role = self._infer_character_role(name, plot_text)
            characters['roles'][name] = role
        
        # Extract relationships
        characters['relationships'] = self._extract_relationships(characters['names'], plot_text)
        
        return characters
    
    def _infer_character_role(self, name: str, text: str) -> str:
        """Infer character role from text patterns."""
        name_lower = name.lower()
        text_lower = text.lower()
        
        # Check for role indicators
        if any(word in text_lower for word in [f'{name_lower} protagonist', f'{name_lower} hero', f'{name_lower} main character']):
            return 'protagonist'
        elif any(word in text_lower for word in [f'{name_lower} antagonist', f'{name_lower} villain', f'{name_lower} enemy']):
            return 'antagonist'
        elif any(word in text_lower for word in [f'{name_lower} narrator', f'{name_lower} tells']):
            return 'narrator'
        elif any(word in text_lower for word in [f'{name_lower} friend', f'{name_lower} ally', f'{name_lower} companion']):
            return 'ally'
        elif any(word in text_lower for word in [f'{name_lower} mentor', f'{name_lower} teacher', f'{name_lower} guide']):
            return 'mentor'
        else:
            return 'character'  # Default role
    
    def _extract_relationships(self, character_names: List[str], text: str) -> List[Dict[str, str]]:
        """Extract relationships between characters."""
        relationships = []
        text_lower = text.lower()
        
        # Look for relationship patterns
        relationship_patterns = [
            ('loves', 'romantic'),
            ('hates', 'antagonistic'),
            ('friends with', 'friendship'),
            ('enemy of', 'enemy'),
            ('brother', 'family'),
            ('sister', 'family'),
            ('father', 'family'),
            ('mother', 'family'),
            ('mentor', 'mentorship'),
            ('student', 'mentorship'),
        ]
        
        for i, char1 in enumerate(character_names):
            for char2 in character_names[i+1:]:
                char1_lower = char1.lower()
                char2_lower = char2.lower()
                
                # Check for relationship indicators
                for pattern, rel_type in relationship_patterns:
                    if pattern in text_lower and (char1_lower in text_lower or char2_lower in text_lower):
                        relationships.append({
                            'character1': char1,
                            'character2': char2,
                            'relationship_type': rel_type
                        })
                        break
        
        return relationships
    
    def _extract_plot_by_character(self, novel: Dict[str, Any], characters: Dict[str, Any]) -> Dict[str, str]:
        """Extract plot information relative to each character."""
        plot_by_character = {}
        plot_text = novel.get('plot_summary', '') + ' ' + novel.get('full_narrative', '')
        
        for name in characters.get('names', []):
            # Extract sentences mentioning this character
            sentences = re.split(r'[.!?]+', plot_text)
            character_sentences = [s.strip() for s in sentences if name.lower() in s.lower()]
            plot_by_character[name] = ' '.join(character_sentences[:5])  # Limit to 5 sentences
        
        return plot_by_character
    
    def _calculate_character_diversity(self, characters: Dict[str, Any]) -> float:
        """Calculate character diversity metric."""
        names = characters.get('names', [])
        if not names:
            return 0.0
        
        # Simple diversity: number of unique first letters / total characters
        first_letters = set(name[0].lower() for name in names if name)
        diversity = len(first_letters) / max(len(names), 1)
        
        return diversity
    
    def _create_full_narrative(self, novel: Dict[str, Any], characters: Dict[str, Any]) -> str:
        """Create comprehensive narrative text including all character information."""
        parts = []
        
        # Plot summary
        if novel.get('plot_summary'):
            parts.append(novel['plot_summary'])
        
        # Character information
        if characters.get('names'):
            char_info = f"Main characters include: {', '.join(characters['main_characters'][:5])}."
            parts.append(char_info)
        
        # Full narrative if available
        if novel.get('full_narrative'):
            parts.append(novel['full_narrative'])
        
        return ' '.join(parts)
    
    def _create_outcomes(self, novel: Dict[str, Any]) -> Dict[str, Any]:
        """Create multi-task outcome variables."""
        outcomes = {
            'ratings': novel.get('average_rating'),
            'awards_binary': 1 if novel.get('won_major_award') else 0,
            'bestseller_binary': 1 if novel.get('is_bestseller') else 0,
            'critical_acclaim': novel.get('critical_acclaim_score', 0.0),
            'sales': novel.get('estimated_sales'),
            'composite_score': 0.0
        }
        
        # Create composite score from available outcomes
        scores = []
        if outcomes['ratings']:
            scores.append((outcomes['ratings'] - 2.0) / 3.0)  # Normalize 2-5 to 0-1
        if outcomes['awards_binary']:
            scores.append(1.0)
        if outcomes['bestseller_binary']:
            scores.append(0.8)
        if outcomes['critical_acclaim']:
            scores.append(outcomes['critical_acclaim'])
        
        outcomes['composite_score'] = np.mean(scores) if scores else 0.5
        
        return outcomes


def main():
    """Test data loader."""
    loader = NovelsDataLoader()
    novels = loader.load_full_dataset()
    
    print(f"\n✓ Loaded {len(novels)} novels")
    if novels:
        sample = novels[0]
        print(f"\nSample novel: {sample['title']}")
        print(f"  Author: {sample['author_name']}")
        print(f"  Characters: {len(sample['character_names'])}")
        print(f"  Main characters: {', '.join(sample['main_characters'][:5])}")
        print(f"  Ensemble size: {sample['ensemble_size']}")


if __name__ == '__main__':
    main()

