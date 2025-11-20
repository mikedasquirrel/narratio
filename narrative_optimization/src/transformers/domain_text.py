"""
Domain narrative transformer for text classification.

Captures domain-specific patterns: writing style, document structure, and
topical coherence. Tests whether domain expertise in feature engineering
outperforms generic approaches.
"""

from typing import Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from .base import NarrativeTransformer


class DomainTextNarrativeTransformer(NarrativeTransformer):
    """
    Domain narrative for text: style + structure + topics.
    
    Narrative Hypothesis: Expert-crafted domain features that capture writing
    style, document structure, and topical coherence outperform generic approaches.
    
    This transformer encodes domain knowledge about text through:
    - Writing style features (sentence patterns, lexical diversity, readability)
    - Document structure (intro/body/conclusion patterns, paragraph organization)
    - Topical coherence (LDA topics + consistency measures)
    
    The narrative: "Good text has consistent style, clear structure, and coherent topics."
    
    Parameters
    ----------
    n_topics : int
        Number of topics to extract (LDA)
    style_features : bool
        Whether to include writing style features
    structure_features : bool
        Whether to include document structure features
    
    Attributes
    ----------
    lda_ : LatentDirichletAllocation
        Topic model
    vectorizer_ : CountVectorizer
        Vectorizer for topic modeling
    """
    
    def __init__(
        self,
        n_topics: int = 20,
        style_features: bool = True,
        structure_features: bool = True
    ):
        super().__init__(
            narrative_id="domain_text_narrative",
            description="Domain narrative: style + structure + topics capture text quality"
        )
        
        self.n_topics = n_topics
        self.style_features = style_features
        self.structure_features = structure_features
        
        self.lda_ = None
        self.vectorizer_ = None
    
    def fit(self, X, y=None):
        """
        Fit domain models to text data.
        
        Parameters
        ----------
        X : list or array of str
            Text documents
        y : ignored
        
        Returns
        -------
        self : DomainTextNarrativeTransformer
        """
        # Fit topic model
        self.vectorizer_ = CountVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        X_counts = self.vectorizer_.fit_transform(X)
        
        self.lda_ = LatentDirichletAllocation(
            n_components=min(self.n_topics, len(X)),
            random_state=42,
            max_iter=20,
            learning_method='online',
            batch_size=128
        )
        self.lda_.fit(X_counts)
        
        # Store metadata
        self.metadata['n_topics'] = self.lda_.n_components
        self.metadata['perplexity'] = float(self.lda_.perplexity(X_counts))
        self.metadata['approach'] = 'domain_specific'
        self.metadata['features_enabled'] = {
            'topics': True,
            'style': self.style_features,
            'structure': self.structure_features
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform text to domain-specific features.
        
        Parameters
        ----------
        X : list or array of str
            Text documents
        
        Returns
        -------
        X_transformed : array
            Domain feature matrix
        """
        self._validate_fitted()
        
        features_list = []
        
        # 1. Topic features
        X_counts = self.vectorizer_.transform(X)
        topic_distributions = self.lda_.transform(X_counts)
        features_list.append(topic_distributions)
        
        # Topic coherence: entropy of topic distribution (lower = more focused)
        topic_entropy = -np.sum(
            topic_distributions * np.log(topic_distributions + 1e-10),
            axis=1
        ).reshape(-1, 1)
        features_list.append(topic_entropy)
        
        # Dominant topic strength
        dominant_topic_strength = np.max(topic_distributions, axis=1).reshape(-1, 1)
        features_list.append(dominant_topic_strength)
        
        # 2. Style features
        if self.style_features:
            style_feats = self._extract_style_features(X)
            features_list.append(style_feats)
        
        # 3. Structure features
        if self.structure_features:
            structure_feats = self._extract_structure_features(X)
            features_list.append(structure_feats)
        
        # Combine all features
        features = np.hstack(features_list)
        
        return features
    
    def _extract_style_features(self, X: List[str]) -> np.ndarray:
        """
        Extract writing style features.
        
        Captures:
        - Lexical diversity (unique words / total words)
        - Average sentence length
        - Sentence length variance (consistency)
        - Average word length
        - Punctuation density
        """
        features = []
        
        for text in X:
            # Tokenize
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Lexical diversity
            lexical_diversity = len(set(words)) / (len(words) + 1)
            
            # Sentence lengths
            sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
            avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
            sentence_length_variance = np.var(sentence_lengths) if sentence_lengths else 0
            
            # Word length
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            
            # Punctuation density
            punctuation_count = len(re.findall(r'[,.!?;:]', text))
            punctuation_density = punctuation_count / (len(words) + 1)
            
            features.append([
                lexical_diversity,
                avg_sentence_length,
                sentence_length_variance,
                avg_word_length,
                punctuation_density
            ])
        
        return np.array(features)
    
    def _extract_structure_features(self, X: List[str]) -> np.ndarray:
        """
        Extract document structure features.
        
        Captures:
        - Number of paragraphs
        - Paragraph length consistency
        - Position of longest paragraph (intro/body/conclusion pattern)
        - Document length
        - Paragraph count / document length ratio
        """
        features = []
        
        for text in X:
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
            if not paragraphs:
                paragraphs = [text]  # Treat whole text as one paragraph
            
            # Paragraph metrics
            n_paragraphs = len(paragraphs)
            paragraph_lengths = [len(re.findall(r'\b\w+\b', p)) for p in paragraphs]
            
            avg_paragraph_length = np.mean(paragraph_lengths) if paragraph_lengths else 0
            paragraph_length_variance = np.var(paragraph_lengths) if len(paragraph_lengths) > 1 else 0
            
            # Position of longest paragraph (normalized 0-1)
            if paragraph_lengths:
                longest_para_position = np.argmax(paragraph_lengths) / len(paragraph_lengths)
            else:
                longest_para_position = 0.5
            
            # Document length
            document_length = sum(paragraph_lengths)
            
            # Structure density
            structure_density = n_paragraphs / (document_length + 1)
            
            features.append([
                n_paragraphs,
                avg_paragraph_length,
                paragraph_length_variance,
                longest_para_position,
                document_length,
                structure_density
            ])
        
        return np.array(features)
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of the domain narrative."""
        n_topics = self.metadata.get('n_topics', 0)
        perplexity = self.metadata.get('perplexity', 0)
        enabled = self.metadata.get('features_enabled', {})
        
        interpretation = (
            f"Domain Text Narrative: Combines expert-crafted features capturing what makes "
            f"text 'good' in this domain. Uses {n_topics} interpretable topics (perplexity: {perplexity:.1f}) "
            f"plus domain features. "
        )
        
        if enabled.get('style'):
            interpretation += (
                "Style features capture writing quality (lexical diversity, sentence patterns, "
                "consistency). "
            )
        
        if enabled.get('structure'):
            interpretation += (
                "Structure features capture organization (paragraph patterns, document flow). "
            )
        
        interpretation += (
            "This narrative embeds domain expertise: the hypothesis that good writing has "
            "coherent topics, consistent style, and clear structure. If this outperforms "
            "statistical and semantic approaches, it validates domain-specific feature engineering."
        )
        
        return interpretation

