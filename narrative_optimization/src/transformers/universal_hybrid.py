"""
Universal Hybrid Transformer

Handles ANY type of data: text, numbers, mixed dictionaries, DataFrames.
Automatically detects data types and extracts appropriate narrative features.

This is THE foundational transformer for domain-agnostic analysis.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from .utils.shared_models import SharedModelRegistry
from .utils.input_validation import ensure_string_list

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class UniversalHybridTransformer(BaseEstimator, TransformerMixin):
    """
    Universal transformer that handles ANY data type.
    
    Accepts:
    - Pure text (str)
    - Pure numbers (dict of metrics, DataFrame)
    - Mixed (dict with text + numbers)
    - List of any of the above
    - Nested structures (JSON-like)
    
    Automatically:
    - Detects data types
    - Extracts text features from text fields
    - Extracts statistical features from numeric fields
    - Extracts categorical features from categorical fields
    - Combines into unified feature vector
    
    Features: Variable (50-200 depending on input)
    - Text features: 20-100 (if text present)
    - Numeric features: 10-50 (if numbers present)
    - Categorical features: 5-20 (if categories present)
    - Interaction features: 5-30 (text-number relationships)
    
    Usage:
    ------
    # Works with pure text
    transformer.transform(["This is a story..."])
    
    # Works with pure numbers
    transformer.transform([{"metric1": 50, "metric2": 0.75}])
    
    # Works with mixed
    transformer.transform([{
        "text": "Story...",
        "score": 85,
        "category": "sports"
    }])
    
    # Works with DataFrame
    transformer.transform(df)
    """
    
    def __init__(
        self,
        extract_text_features: bool = True,
        extract_numeric_features: bool = True,
        extract_categorical_features: bool = True,
        use_advanced_nlp: bool = True,
        max_text_features: int = 100,
        max_numeric_features: int = 50
    ):
        """
        Initialize universal hybrid transformer.
        
        Parameters
        ----------
        extract_text_features : bool
            Extract features from text fields
        extract_numeric_features : bool
            Extract features from numeric fields
        extract_categorical_features : bool
            Extract features from categorical fields
        use_advanced_nlp : bool
            Use spaCy for advanced text analysis
        max_text_features : int
            Maximum text features to extract
        max_numeric_features : int
            Maximum numeric features to extract
        """
        self.extract_text_features = extract_text_features
        self.extract_numeric_features = extract_numeric_features
        self.extract_categorical_features = extract_categorical_features
        self.use_advanced_nlp = use_advanced_nlp
        self.max_text_features = max_text_features
        self.max_numeric_features = max_numeric_features
        
        self.nlp = None
        self.embedder = None
        self.scaler = StandardScaler()
        
        # Learned from data during fit
        self.text_fields_ = []
        self.numeric_fields_ = []
        self.categorical_fields_ = []
        self.field_types_ = {}
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """
        Fit transformer (detect data types, load models).
        
        Parameters
        ----------
        X : various types
            Can be list of str, list of dict, DataFrame, mixed
        y : ignored
        
        Returns
        -------
        self
        """
        # Analyze data structure
        self._analyze_data_structure(X)
        
        # Load models if needed
        if self.extract_text_features and self.text_fields_:
            if self.use_advanced_nlp and SPACY_AVAILABLE:
                self.nlp = SharedModelRegistry.get_spacy()
            self.embedder = SharedModelRegistry.get_sentence_transformer()
        
        # Learn numeric field statistics for normalization
        if self.extract_numeric_features and self.numeric_fields_:
            numeric_data = self._extract_numeric_data(X)
            if len(numeric_data) > 0:
                self.scaler.fit(numeric_data)
        
        return self
    
    def transform(self, X):
        """
        Transform ANY data type to features.
        
        Parameters
        ----------
        X : various types
            Data to transform
            
        Returns
        -------
        features : ndarray
            Unified feature matrix
        """
        features_list = []
        
        # Convert to standard format
        X_standard = self._standardize_input(X)
        
        for item in X_standard:
            feat = self._extract_all_features(item)
            features_list.append(feat)
        
        return np.array(features_list, dtype=np.float32)
    
    def _analyze_data_structure(self, X):
        """
        Analyze input data structure to determine what types of features to extract.
        """
        # Convert to standard format for analysis
        X_standard = self._standardize_input(X)
        
        if not X_standard:
            return
        
        # Analyze first few items to determine structure
        sample_size = min(10, len(X_standard))
        sample_items = X_standard[:sample_size]
        
        # Track field types across sample
        field_type_votes = {}
        
        for item in sample_items:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key not in field_type_votes:
                        field_type_votes[key] = {'text': 0, 'numeric': 0, 'categorical': 0}
                    
                    # Vote on type
                    if isinstance(value, str) and len(value) > 50:
                        field_type_votes[key]['text'] += 1
                    elif isinstance(value, (int, float)):
                        field_type_votes[key]['numeric'] += 1
                    elif isinstance(value, str):
                        field_type_votes[key]['categorical'] += 1
            elif isinstance(item, str):
                # Pure text input
                if 'text' not in field_type_votes:
                    field_type_votes['text'] = {'text': 0, 'numeric': 0, 'categorical': 0}
                field_type_votes['text']['text'] += 1
        
        # Determine field types by majority vote
        for field, votes in field_type_votes.items():
            dominant_type = max(votes, key=votes.get)
            self.field_types_[field] = dominant_type
            
            if dominant_type == 'text':
                self.text_fields_.append(field)
            elif dominant_type == 'numeric':
                self.numeric_fields_.append(field)
            elif dominant_type == 'categorical':
                self.categorical_fields_.append(field)
    
    def _standardize_input(self, X) -> List[Union[str, Dict]]:
        """Convert any input format to list of standardized items."""
        # Already a list
        if isinstance(X, list):
            return X
        
        # DataFrame
        elif isinstance(X, pd.DataFrame):
            return X.to_dict('records')
        
        # Single string
        elif isinstance(X, str):
            return [X]
        
        # Single dict
        elif isinstance(X, dict):
            return [X]
        
        # numpy array of strings
        elif isinstance(X, np.ndarray):
            if X.dtype.kind in ['U', 'S', 'O']:  # String types
                return X.tolist()
            else:
                return [X.tolist()]
        
        # Unknown, try to convert
        else:
            return [str(X)]
    
    def _extract_all_features(self, item: Union[str, Dict]) -> List[float]:
        """Extract all features from a single item."""
        features = []
        
        # Text features
        if self.extract_text_features and self.text_fields_:
            text_features = self._extract_text_features_from_item(item)
            features.extend(text_features)
        
        # Numeric features
        if self.extract_numeric_features and self.numeric_fields_:
            numeric_features = self._extract_numeric_features_from_item(item)
            features.extend(numeric_features)
        
        # Categorical features
        if self.extract_categorical_features and self.categorical_fields_:
            categorical_features = self._extract_categorical_features_from_item(item)
            features.extend(categorical_features)
        
        # Interaction features (text-numeric relationships)
        if self.text_fields_ and self.numeric_fields_:
            interaction_features = self._extract_interaction_features(item)
            features.extend(interaction_features)
        
        # If no features extracted, return minimal features
        if not features:
            features = [0.5] * 10  # Minimal fallback
        
        return features
    
    def _extract_text_features_from_item(self, item: Union[str, Dict]) -> List[float]:
        """Extract text narrative features."""
        # Get text content
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            # Concatenate all text fields
            text_parts = []
            for field in self.text_fields_:
                if field in item and item[field]:
                    text_parts.append(str(item[field]))
            text = " ".join(text_parts)
        else:
            text = str(item)
        
        if not text or len(text) < 10:
            return [0.0] * min(20, self.max_text_features)
        
        features = []
        
        # Basic text statistics
        words = text.split()
        n_words = len(words)
        
        # 1. Length features
        features.append(min(1.0, n_words / 500))  # Normalized length
        features.append(min(1.0, len(text) / 3000))  # Character length
        
        # 2. Lexical diversity
        unique_words = len(set(words))
        features.append(unique_words / (n_words + 1))  # TTR
        
        # 3-5. Basic linguistic features
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features.append(min(1.0, len(sentences) / 50))  # Sentence count
        features.append(n_words / (len(sentences) + 1) / 20)  # Avg sentence length (normalized)
        
        # Capitalization (proper nouns proxy)
        capitals = len(re.findall(r'\b[A-Z][a-z]+\b', text))
        features.append(capitals / (n_words + 1))
        
        # 6-10. Advanced NLP features (if available)
        if self.nlp:
            doc = self.nlp(text[:5000])  # Limit for speed
            
            # Verb density (action)
            verbs = sum(1 for token in doc if token.pos_ == 'VERB')
            features.append(verbs / (len(doc) + 1))
            
            # Adjective density (description)
            adjectives = sum(1 for token in doc if token.pos_ == 'ADJ')
            features.append(adjectives / (len(doc) + 1))
            
            # Entity density
            entities = len(list(doc.ents))
            features.append(entities / (len(doc) + 1) * 10)
            
            # First person (agency)
            first_person = sum(1 for token in doc if token.lemma_ in {'i', 'we', 'me', 'us'})
            features.append(first_person / (len(doc) + 1))
        else:
            # Fallback: regex-based
            verbs_approx = len(re.findall(r'\b\w+ed\b|\b\w+ing\b', text.lower()))
            features.append(verbs_approx / (n_words + 1))
            features.extend([0.3, 0.2, 0.2])  # Placeholders
        
        # 11-15. Sentiment proxies
        positive_words = sum(1 for w in words if w.lower() in {'good', 'great', 'excellent', 'wonderful', 'amazing'})
        negative_words = sum(1 for w in words if w.lower() in {'bad', 'terrible', 'awful', 'poor', 'horrible'})
        
        features.append(positive_words / (n_words + 1))
        features.append(negative_words / (n_words + 1))
        features.append((positive_words - negative_words) / (n_words + 1))  # Sentiment
        
        # Emotional intensity
        emotion_words = positive_words + negative_words
        features.append(emotion_words / (n_words + 1))
        
        # Future orientation
        future_words = len(re.findall(r'\bwill\b|\bshall\b|\bgoing to\b', text.lower()))
        features.append(future_words / (n_words + 1))
        
        # 16-20. Semantic embeddings (if available)
        if self.embedder:
            try:
                # Get embedding
                text_truncated = text[:1000] if len(text) > 1000 else text
                embedding = self.embedder.encode([text_truncated])[0]
                
                # Use summary statistics of embedding
                features.extend([
                    float(np.mean(embedding)),
                    float(np.std(embedding)),
                    float(np.max(embedding)),
                    float(np.min(embedding)),
                    float(np.median(embedding))
                ])
            except:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)
        
        # Pad or trim to max_text_features
        while len(features) < min(20, self.max_text_features):
            features.append(0.0)
        
        return features[:self.max_text_features]
    
    def _extract_numeric_features_from_item(self, item: Union[str, Dict]) -> List[float]:
        """Extract features from numeric fields."""
        if isinstance(item, str):
            # No numeric data in pure string
            return []
        
        if not isinstance(item, dict):
            return []
        
        features = []
        
        # Extract all numeric values
        numeric_values = []
        for field in self.numeric_fields_:
            if field in item:
                value = item[field]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    numeric_values.append(value)
                    
                    # Basic transformations
                    features.append(float(value))  # Raw value
                    features.append(np.log1p(abs(value)))  # Log transform
                    features.append(1.0 if value > 0 else 0.0)  # Sign
        
        # Aggregate statistics if multiple numeric fields
        if len(numeric_values) > 1:
            features.append(np.mean(numeric_values))
            features.append(np.std(numeric_values))
            features.append(np.max(numeric_values))
            features.append(np.min(numeric_values))
        
        # Pad or trim
        while len(features) < min(10, self.max_numeric_features):
            features.append(0.0)
        
        return features[:self.max_numeric_features]
    
    def _extract_categorical_features_from_item(self, item: Union[str, Dict]) -> List[float]:
        """Extract features from categorical fields."""
        if not isinstance(item, dict):
            return []
        
        features = []
        
        # One-hot encode categorical fields (simplified)
        for field in self.categorical_fields_:
            if field in item:
                value = str(item[field]).lower()
                
                # Hash to 0-1 range for feature
                hash_value = hash(value) % 1000 / 1000.0
                features.append(hash_value)
                
                # Length of category name (complexity proxy)
                features.append(min(1.0, len(value) / 50))
        
        # Category diversity
        if len(self.categorical_fields_) > 1:
            unique_values = len(set(str(item.get(f, '')) for f in self.categorical_fields_))
            features.append(unique_values / len(self.categorical_fields_))
        
        # Pad or trim
        while len(features) < 5:
            features.append(0.0)
        
        return features[:20]
    
    def _extract_interaction_features(self, item: Union[str, Dict]) -> List[float]:
        """Extract interaction features between text and numbers."""
        if not isinstance(item, dict):
            return []
        
        features = []
        
        # Get text content
        text_parts = []
        for field in self.text_fields_:
            if field in item and item[field]:
                text_parts.append(str(item[field]))
        text = " ".join(text_parts)
        
        if not text:
            return []
        
        # Get numeric values
        numeric_values = []
        for field in self.numeric_fields_:
            if field in item and isinstance(item[field], (int, float)):
                numeric_values.append(item[field])
        
        if not numeric_values:
            return []
        
        # Interaction features
        text_length = len(text.split())
        
        # 1. Text length vs numeric magnitude
        avg_numeric = np.mean(numeric_values)
        features.append(text_length / (avg_numeric + 1))
        
        # 2. Text complexity vs numeric complexity
        unique_words = len(set(text.lower().split()))
        numeric_variance = np.var(numeric_values) if len(numeric_values) > 1 else 0
        features.append((unique_words / (text_length + 1)) * numeric_variance)
        
        # 3. Sentiment vs numeric sign
        positive_words = sum(1 for w in text.lower().split() if w in {'good', 'great', 'high', 'strong', 'win'})
        negative_words = sum(1 for w in text.lower().split() if w in {'bad', 'poor', 'low', 'weak', 'lose'})
        sentiment = (positive_words - negative_words) / (text_length + 1)
        numeric_sign = 1.0 if avg_numeric > 0 else -1.0
        
        features.append(sentiment * numeric_sign)  # Alignment
        
        # 4. Numeric count in text (precision marker)
        numbers_in_text = len(re.findall(r'\d+', text))
        features.append(numbers_in_text / (text_length + 1))
        
        # 5. Text/numeric ratio (qualitative vs quantitative emphasis)
        features.append(text_length / (len(numeric_values) + 1))
        
        return features
    
    def _extract_numeric_data(self, X) -> np.ndarray:
        """Extract numeric data matrix for scaler fitting."""
        X_standard = self._standardize_input(X)
        
        numeric_data = []
        for item in X_standard:
            if isinstance(item, dict):
                values = []
                for field in self.numeric_fields_:
                    if field in item and isinstance(item[field], (int, float)):
                        values.append(item[field])
                if values:
                    numeric_data.append(values)
        
        if numeric_data:
            # Pad to same length
            max_len = max(len(row) for row in numeric_data)
            padded = [row + [0.0] * (max_len - len(row)) for row in numeric_data]
            return np.array(padded)
        
        return np.array([])
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names."""
        if self.feature_names_:
            return np.array(self.feature_names_)
        
        # Generate generic names
        names = []
        
        if self.text_fields_:
            for i in range(self.max_text_features):
                names.append(f'text_feature_{i}')
        
        if self.numeric_fields_:
            for i in range(self.max_numeric_features):
                names.append(f'numeric_feature_{i}')
        
        if self.categorical_fields_:
            for i in range(20):
                names.append(f'categorical_feature_{i}')
        
        if self.text_fields_ and self.numeric_fields_:
            for i in range(5):
                names.append(f'interaction_feature_{i}')
        
        return np.array(names)
    
    def get_data_structure_summary(self) -> Dict[str, Any]:
        """Get summary of detected data structure."""
        return {
            'text_fields': self.text_fields_,
            'numeric_fields': self.numeric_fields_,
            'categorical_fields': self.categorical_fields_,
            'field_types': self.field_types_,
            'models_loaded': {
                'spacy': self.nlp is not None,
                'embedder': self.embedder is not None
            }
        }

