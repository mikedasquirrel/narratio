"""
Cultural/Contextual Transformer

Extracts genre conventions, cultural references, zeitgeist alignment, and tribal signals.
Cross-cutting enhancement that improves all domains via better context awareness.

Core insight: Contextual appropriateness matters - same narrative performs differently
in different genres, cultures, and time periods.
"""

import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from .utils.input_validation import ensure_string_list


class CulturalContextTransformer(BaseEstimator, TransformerMixin):
    """
    Extract cultural and contextual features from narrative text.
    
    Captures:
    1. Genre Convention Awareness - tropes, format, vocabulary
    2. Cultural References - allusions, shared knowledge
    3. Zeitgeist Alignment - trends, timeliness, contemporary issues
    4. Tribal Signals - in-group language, identity, values
    
    ~35 features total
    """
    
    def __init__(self):
        """Initialize cultural markers"""
        
        # Genre-specific tropes and conventions
        self.genre_markers = {
            'romance': ['love', 'heart', 'feel', 'relationship', 'together', 'kiss', 'forever'],
            'thriller': ['danger', 'threat', 'chase', 'escape', 'secret', 'mystery', 'suspect'],
            'comedy': ['funny', 'laugh', 'hilarious', 'joke', 'humor', 'comedy', 'ridiculous'],
            'drama': ['emotional', 'struggle', 'conflict', 'tragic', 'difficult', 'complex'],
            'scifi': ['future', 'technology', 'space', 'alien', 'robot', 'advanced', 'discover'],
            'horror': ['fear', 'terror', 'dark', 'death', 'haunted', 'nightmare', 'scream'],
            'action': ['fight', 'battle', 'weapon', 'explosion', 'chase', 'fast', 'intense'],
            'fantasy': ['magic', 'quest', 'ancient', 'power', 'destiny', 'realm', 'legend']
        }
        
        # Cultural reference types
        self.pop_culture_markers = [
            'movie', 'film', 'show', 'series', 'song', 'music', 'celebrity',
            'viral', 'meme', 'trending', 'famous', 'popular', 'mainstream'
        ]
        
        self.high_culture_markers = [
            'art', 'literature', 'classical', 'philosophy', 'theory', 'tradition',
            'masterpiece', 'canonical', 'seminal', 'influential'
        ]
        
        # Temporal/zeitgeist markers
        self.contemporary_markers = [
            'modern', 'current', 'today', 'now', 'recent', 'latest', 'new',
            'contemporary', 'digital', 'online', 'social media', 'app', 'tech'
        ]
        
        self.dated_markers = [
            'traditional', 'classic', 'old', 'vintage', 'retro', 'nostalgic',
            'back then', 'used to', 'remember when', 'in those days'
        ]
        
        # Contemporary issues/trends
        self.current_issues = [
            'climate', 'sustainability', 'diversity', 'inclusion', 'equity',
            'mental health', 'wellness', 'pandemic', 'remote', 'virtual',
            'ai', 'automation', 'privacy', 'data', 'social justice'
        ]
        
        # Generational markers
        self.generational_markers = {
            'boomer': ['traditional', 'established', 'experience', 'wisdom', 'stability'],
            'genx': ['independent', 'pragmatic', 'skeptical', 'balance', 'authentic'],
            'millennial': ['purpose', 'meaningful', 'experience', 'share', 'connect'],
            'genz': ['authentic', 'diverse', 'mental health', 'sustainable', 'digital']
        }
        
        # Tribal/in-group language
        self.community_markers = [
            'we', 'us', 'our', 'community', 'together', 'shared', 'collective',
            'belong', 'member', 'insider', 'fellow', 'comrade', 'ally'
        ]
        
        # Identity markers
        self.identity_markers = [
            'identity', 'who i am', 'who we are', 'define', 'represent',
            'stand for', 'believe in', 'value', 'principle', 'core'
        ]
        
        # Value alignment signals
        self.values_progressive = [
            'change', 'progress', 'innovation', 'reform', 'transform',
            'evolve', 'inclusive', 'diverse', 'equity', 'justice'
        ]
        
        self.values_conservative = [
            'tradition', 'stability', 'heritage', 'preserve', 'maintain',
            'protect', 'family', 'security', 'order', 'proven'
        ]
        
        # Format conventions
        self.narrative_formats = {
            'first_person': ['i', 'me', 'my', 'we', 'us', 'our'],
            'third_person': ['he', 'she', 'they', 'him', 'her', 'their'],
            'epistolary': ['dear', 'sincerely', 'letter', 'wrote', 'correspondence'],
            'dialogue_heavy': ['"', "'", 'said', 'asked', 'replied', 'told']
        }
        
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """Transform texts into cultural/contextual features"""
        features = []
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        for text in X:
            # Ensure text is string
            text = str(text) if not isinstance(text, str) else text
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            feat_dict = {}
            
            # === 1. GENRE CONVENTION AWARENESS (10 features) ===
            
            # Detect genre alignment (which genre markers are most present)
            genre_scores = {}
            for genre, markers in self.genre_markers.items():
                score = sum(1 for w in words if w in markers) / (len(words) + 1)
                genre_scores[genre] = score
            
            # Top genre affinity
            max_genre = max(genre_scores, key=genre_scores.get)
            feat_dict['genre_affinity_strength'] = genre_scores[max_genre]
            
            # Genre diversity (mixed genres)
            active_genres = sum(1 for score in genre_scores.values() if score > 0.01)
            feat_dict['genre_diversity'] = active_genres / len(self.genre_markers)
            
            # Genre-specific features (top 3 genres)
            sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (genre, score) in enumerate(sorted_genres[:3]):
                feat_dict[f'genre_{genre}_score'] = score
            
            # Trope usage (repeated genre patterns)
            if genre_scores[max_genre] > 0.02:
                feat_dict['trope_density'] = genre_scores[max_genre]
            else:
                feat_dict['trope_density'] = 0.0
            
            # Format convention (POV consistency)
            first_person = sum(1 for w in words if w in self.narrative_formats['first_person'])
            third_person = sum(1 for w in words if w in self.narrative_formats['third_person'])
            
            if first_person + third_person > 0:
                feat_dict['format_consistency'] = max(first_person, third_person) / (first_person + third_person)
            else:
                feat_dict['format_consistency'] = 1.0
            
            # Dialogue presence
            quote_count = text.count('"') + text.count("'")
            feat_dict['dialogue_density'] = quote_count / (len(sentences) + 1)
            
            # Convention adherence vs subversion
            # High genre score + unexpected elements = subversion
            all_genre_markers = [m for markers in self.genre_markers.values() for m in markers]
            genre_marker_count = sum(1 for w in words if w in all_genre_markers)
            non_genre_content = 1.0 - (genre_marker_count / (len(words) + 1))
            feat_dict['convention_subversion'] = non_genre_content * feat_dict['genre_affinity_strength']
            
            # === 2. CULTURAL REFERENCES (9 features) ===
            
            # Pop culture references
            pop_culture_count = sum(1 for w in words if w in self.pop_culture_markers)
            feat_dict['pop_culture_density'] = pop_culture_count / (len(words) + 1)
            
            # High culture references
            high_culture_count = sum(1 for w in words if w in self.high_culture_markers)
            feat_dict['high_culture_density'] = high_culture_count / (len(words) + 1)
            
            # Cultural reference balance
            feat_dict['culture_type_ratio'] = pop_culture_count / (high_culture_count + 1)
            
            # Allusion density (proper nouns as proxy)
            proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', text))
            feat_dict['allusion_density'] = proper_nouns / (len(words) + 1)
            
            # Shared knowledge assumptions (use of "we all know", "everyone", etc.)
            shared_knowledge_markers = ['we all', 'everyone knows', 'obviously', 'of course',
                                       'as we know', 'clearly', 'naturally']
            shared_knowledge = sum(1 for marker in shared_knowledge_markers if marker in text_lower)
            feat_dict['shared_knowledge_assumption'] = shared_knowledge / (len(sentences) + 1)
            
            # Reference recency (contemporary vs classic)
            contemporary_refs = sum(1 for w in words if w in self.contemporary_markers)
            classic_refs = sum(1 for w in words if w in self.dated_markers)
            
            feat_dict['reference_recency'] = (contemporary_refs - classic_refs) / (contemporary_refs + classic_refs + 1)
            
            # Intertextuality (references to other texts/media)
            intertextual_markers = ['like', 'similar to', 'reminds', 'recalls', 'echoes', 'inspired by']
            intertextual = sum(1 for marker in intertextual_markers if marker in text_lower)
            feat_dict['intertextuality'] = intertextual / (len(sentences) + 1)
            
            # Universal vs specific references
            universal_markers = ['human', 'life', 'love', 'death', 'time', 'truth', 'beauty']
            universal_count = sum(1 for w in words if w in universal_markers)
            feat_dict['universal_themes'] = universal_count / (len(words) + 1)
            
            # Cultural specificity
            feat_dict['cultural_specificity'] = feat_dict['allusion_density'] / (feat_dict['universal_themes'] + 0.01)
            
            # === 3. ZEITGEIST ALIGNMENT (8 features) ===
            
            # Contemporary issue engagement
            current_issue_count = sum(1 for issue in self.current_issues if issue in text_lower)
            feat_dict['contemporary_issues'] = current_issue_count / (len(sentences) + 1)
            
            # Timeliness (modern language)
            modern_count = sum(1 for w in words if w in self.contemporary_markers)
            feat_dict['modern_language'] = modern_count / (len(words) + 1)
            
            # Dated language
            dated_count = sum(1 for w in words if w in self.dated_markers)
            feat_dict['dated_language'] = dated_count / (len(words) + 1)
            
            # Temporal orientation (modern vs classic)
            feat_dict['temporal_orientation'] = (modern_count - dated_count) / (modern_count + dated_count + 1)
            
            # Trend awareness (using contemporary tech/social terms)
            tech_social_terms = ['app', 'online', 'digital', 'social', 'viral', 'platform',
                                'algorithm', 'data', 'streaming', 'cloud', 'ai']
            trend_count = sum(1 for w in words if w in tech_social_terms)
            feat_dict['trend_awareness'] = trend_count / (len(words) + 1)
            
            # Generational alignment (which generation's language)
            gen_scores = {}
            for gen, markers in self.generational_markers.items():
                score = sum(1 for w in words if w in markers)
                gen_scores[gen] = score
            
            total_gen_markers = sum(gen_scores.values())
            feat_dict['generational_specificity'] = total_gen_markers / (len(words) + 1)
            
            # Dominant generation
            if total_gen_markers > 0:
                dominant_gen = max(gen_scores, key=gen_scores.get)
                feat_dict['generation_alignment'] = gen_scores[dominant_gen] / total_gen_markers
            else:
                feat_dict['generation_alignment'] = 0.25  # Neutral
            
            # Timelessness (universal themes + low dated language)
            feat_dict['timelessness'] = feat_dict['universal_themes'] * (1.0 - feat_dict['dated_language'])
            
            # === 4. TRIBAL SIGNALS (8 features) ===
            
            # In-group language
            community_count = sum(1 for w in words if w in self.community_markers)
            feat_dict['in_group_language'] = community_count / (len(words) + 1)
            
            # First person plural (we/us/our)
            we_count = sum(1 for w in words if w in ['we', 'us', 'our', 'ours'])
            feat_dict['collective_identity'] = we_count / (len(words) + 1)
            
            # Identity markers
            identity_count = sum(1 for marker in self.identity_markers if marker in text_lower)
            feat_dict['identity_expression'] = identity_count / (len(sentences) + 1)
            
            # Value alignment (progressive vs conservative)
            progressive_count = sum(1 for w in words if w in self.values_progressive)
            conservative_count = sum(1 for w in words if w in self.values_conservative)
            
            feat_dict['value_progressive_density'] = progressive_count / (len(words) + 1)
            feat_dict['value_conservative_density'] = conservative_count / (len(words) + 1)
            
            if progressive_count + conservative_count > 0:
                feat_dict['value_orientation'] = (progressive_count - conservative_count) / (progressive_count + conservative_count)
            else:
                feat_dict['value_orientation'] = 0.0  # Neutral
            
            # Community norms (adherence to expectations)
            norm_markers = ['should', 'must', 'ought', 'supposed to', 'expected', 'appropriate']
            norm_count = sum(1 for w in words if w in norm_markers)
            feat_dict['norm_adherence'] = norm_count / (len(words) + 1)
            
            # Insider language density (jargon, abbreviations)
            abbrev_count = len(re.findall(r'\b[A-Z]{2,}\b', text))
            jargon_estimate = abbrev_count / (len(words) + 1)
            feat_dict['insider_jargon'] = jargon_estimate
            
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = [
            # Genre conventions (10)
            'genre_affinity_strength', 'genre_diversity', 'genre_romance_score',
            'genre_thriller_score', 'genre_comedy_score', 'trope_density',
            'format_consistency', 'dialogue_density', 'convention_subversion',
            
            # Cultural references (9)
            'pop_culture_density', 'high_culture_density', 'culture_type_ratio',
            'allusion_density', 'shared_knowledge_assumption', 'reference_recency',
            'intertextuality', 'universal_themes', 'cultural_specificity',
            
            # Zeitgeist alignment (8)
            'contemporary_issues', 'modern_language', 'dated_language',
            'temporal_orientation', 'trend_awareness', 'generational_specificity',
            'generation_alignment', 'timelessness',
            
            # Tribal signals (8)
            'in_group_language', 'collective_identity', 'identity_expression',
            'value_progressive_density', 'value_conservative_density', 'value_orientation',
            'norm_adherence', 'insider_jargon'
        ]
        
        return np.array([f'cultural_context_{n}' for n in names])

