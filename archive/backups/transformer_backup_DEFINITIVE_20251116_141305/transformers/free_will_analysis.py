"""
Free Will vs Determinism Narrative Analysis Transformer

Analyzes narratives for free will vs determinism signals through:
1. Semantic field analysis (fate vs choice language)
2. Temporal dynamics (future vs past orientation)
3. Information theory (predictability/entropy)
4. Causal structure (agency vs patient roles)
5. Network structure (causal graphs)

Hypothesis: Narratives with high determinism scores show predictable structure,
fate language, and inevitable outcomes. Free will narratives show choice points,
agency, and contingent outcomes.

Author: Narrative Integration System
Date: November 2025
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

from .base import NarrativeTransformer


class FreeWillAnalysisTransformer(NarrativeTransformer):
    """
    Comprehensive transformer for analyzing free will vs determinism in narratives.
    
    Extracts features across multiple dimensions:
    - Semantic fields (fate, choice, causality, contingency)
    - Temporal orientation (future vs past)
    - Information theory (predictability)
    - Causal structure (agency extraction)
    - Network metrics (causal graphs)
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
        Initialize free will analyzer.
        
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
        super().__init__(
            narrative_id="free_will_analysis",
            description="Analyzes narratives for free will vs determinism signals"
        )
        
        self.use_sentence_transformers = use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.model_name = model_name
        self.spacy_model = spacy_model
        self.extract_causal_graphs = extract_causal_graphs
        self.track_observability = track_observability
        
        # Configure weights
        self.temporal_weight = temporal_weight
        self.semantic_weight = semantic_weight
        self.predictability_weight = predictability_weight
        
        # Apply custom weights if provided
        if custom_weights:
            self.temporal_weight = custom_weights.get('temporal', temporal_weight)
            self.semantic_weight = custom_weights.get('semantic', semantic_weight)
            self.predictability_weight = custom_weights.get('predictability', predictability_weight)
        
        # Normalize weights to sum to 1.0
        total_weight = self.temporal_weight + self.semantic_weight + self.predictability_weight
        if total_weight > 0:
            self.temporal_weight /= total_weight
            self.semantic_weight /= total_weight
            self.predictability_weight /= total_weight
        
        # Initialize models lazily
        self.semantic_model = None
        self.nlp = None
        
        # Define semantic fields for free will/determinism analysis
        self._init_semantic_fields()
    
    def _init_semantic_fields(self):
        """Initialize semantic field dictionaries."""
        self.semantic_fields = {
            'fate': ['fate', 'destiny', 'prophesy', 'prophecy', 'doom', 'inevitable', 
                    'predetermined', 'ordained', 'meant to', 'foretold', 'predestined'],
            'choice': ['choose', 'chose', 'decision', 'decide', 'option', 'freedom', 
                      'free will', 'agency', 'deliberate', 'volition', 'autonomy'],
            'causality': ['because', 'caused', 'therefore', 'thus', 'resulted', 
                         'led to', 'due to', 'consequence', 'effect', 'outcome'],
            'contingency': ['might', 'could', 'perhaps', 'maybe', 'possibly', 
                           'uncertain', 'random', 'chance', 'accident', 'unexpected'],
            'inevitability': ['must', 'had to', 'forced', 'compelled', 'no choice', 
                             'inevitable', 'unavoidable', 'inescapable', 'bound to'],
            'agency': ['chose', 'decided', 'willed', 'intended', 'purposely', 
                      'deliberately', 'acted', 'initiated', 'started', 'began'],
            'structure': ['pattern', 'cycle', 'repetition', 'echo', 'parallel', 
                         'recurrence', 'rhythm', 'order', 'sequence'],
            'chaos': ['random', 'chance', 'accident', 'unexpected', 'surprise', 
                     'unpredictable', 'chaotic', 'disorder', 'entropy']
        }
        
        # Nominative-focused semantic fields for agency/determinism
        self.nominative_fields = {
            'deterministic_titles': [
                'the chosen', 'the destined', 'the fated', 'the prophesied',
                'the one', 'the savior', 'the heir', 'the cursed', 'the blessed',
                'the doomed', 'the marked', 'the selected', 'the anointed'
            ],
            'agency_names': [
                'decides', 'chooses', 'acts', 'initiates', 'creates',
                'determines', 'controls', 'directs', 'leads', 'commands',
                'shapes', 'forms', 'builds', 'destroys', 'changes'
            ],
            'role_labels': [
                'victim', 'hero', 'savior', 'witness', 'prophet',
                'messenger', 'warrior', 'leader', 'follower', 'servant',
                'master', 'apprentice', 'guardian', 'destroyer', 'creator'
            ],
            'identity_markers': [
                'i am', 'they call me', 'known as', 'named', 'called',
                'i was', 'became', 'turned into', 'transformed into',
                'born as', 'destined to be', 'meant to be', 'chosen to be'
            ],
            'generic_labels': [
                'the man', 'the woman', 'the child', 'the boy', 'the girl',
                'someone', 'anyone', 'no one', 'everyone', 'somebody',
                'a person', 'the figure', 'the stranger', 'the visitor'
            ]
        }
    
    def _lazy_load_models(self):
        """Load models on first use."""
        if self.use_sentence_transformers and self.semantic_model is None:
            try:
                self.semantic_model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer: {e}")
                self.use_sentence_transformers = False
        
        if self.use_spacy and self.nlp is None:
            try:
                self.nlp = spacy.load(self.spacy_model)
            except OSError:
                print(f"Warning: spaCy model '{self.spacy_model}' not found. "
                      "Install with: python -m spacy download {self.spacy_model}")
                self.use_spacy = False
    
    def fit(self, X, y=None):
        """
        Learn patterns from training data.
        
        Args:
            X: List of narrative texts
            y: Labels (optional)
        
        Returns:
            self
        """
        # Validate input
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty")
        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("Input X must be list or array of texts")
        if len(X) > 0 and not isinstance(X[0], str):
            raise ValueError("Input X must contain text strings")
        
        self._lazy_load_models()
        
        # Extract corpus-level statistics
        field_counts = Counter()
        total_docs = len(X)
        
        for text in X:
            text_lower = text.lower()
            for field_name, field_words in self.semantic_fields.items():
                if any(word in text_lower for word in field_words):
                    field_counts[field_name] += 1
        
        # Store metadata
        self.metadata['field_frequencies'] = {
            field: count / total_docs 
            for field, count in field_counts.items()
        }
        self.metadata['n_features'] = self._calculate_n_features()
        self.metadata['total_docs'] = total_docs
        self.metadata['feature_names'] = self._generate_feature_names()
        
        # If using sentence transformers, encode sample for dimension detection
        if self.use_sentence_transformers and len(X) > 0:
            sample_embedding = self.semantic_model.encode(X[0])
            self.metadata['embedding_dim'] = len(sample_embedding)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform texts to free will/determinism features.
        
        Args:
            X: List of narrative texts
        
        Returns:
            numpy.ndarray: Feature matrix (n_samples, n_features)
        """
        if not self.is_fitted_:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before transform. "
                "Call fit() or fit_transform() first."
            )
        
        # Validate input
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty")
        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("Input X must be list or array of texts")
        if len(X) > 0 and not isinstance(X[0], str):
            raise ValueError("Input X must contain text strings")
        
        features = []
        for text in X:
            doc_features = self._extract_document_features(text)
            features.append(doc_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_document_features(self, text: str) -> List[float]:
        """Extract comprehensive features from single document."""
        features = []
        
        # 1. Semantic field features
        semantic_features = self._extract_semantic_field_features(text)
        features.extend(semantic_features)
        
        # 2. Temporal features
        temporal_features = self._extract_temporal_features(text)
        features.extend(temporal_features)
        
        # 3. Information theory features
        info_features = self._extract_information_theory_features(text)
        features.extend(info_features)
        
        # 4. Agency/patient extraction (spaCy)
        if self.use_spacy:
            agency_features = self._extract_agency_features(text)
            features.extend(agency_features)
        else:
            # Placeholder zeros
            features.extend([0.0] * 5)
        
        # 5. Causal graph features
        if self.extract_causal_graphs:
            graph_features = self._extract_causal_graph_features(text)
            features.extend(graph_features)
        else:
            features.extend([0.0] * 4)
        
        # 6. Sentence transformer embeddings (if available)
        if self.use_sentence_transformers:
            embedding = self.semantic_model.encode(text)
            # Use PCA-like reduction or take first N dimensions
            # For now, take mean, std, min, max of embedding
            features.extend([
                float(np.mean(embedding)),
                float(np.std(embedding)),
                float(np.min(embedding)),
                float(np.max(embedding))
            ])
        else:
            features.extend([0.0] * 4)
        
        # 7. Observability features (visible vs hidden causality)
        if self.track_observability:
            observability_features = self._extract_observability_features(text)
            features.extend(observability_features)
        else:
            features.extend([0.0] * 3)
        
        # 8. Nominative agency features
        nominative_features = self._extract_nominative_agency_features(text)
        features.extend(nominative_features)
        
        # 9. Character naming evolution
        naming_evolution = self._extract_character_naming_evolution(text)
        features.extend(naming_evolution)
        
        # 10. Composite scores
        composite_scores = self._calculate_composite_scores(
            semantic_features, temporal_features, info_features
        )
        features.extend(composite_scores)
        
        return features
    
    def _extract_semantic_field_features(self, text: str) -> List[float]:
        """Extract semantic field density features."""
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) if words else 1
        
        features = []
        
        # Density for each semantic field
        for field_name in sorted(self.semantic_fields.keys()):
            field_words = self.semantic_fields[field_name]
            count = sum(1 for word in field_words if word in text_lower)
            density = count / n_words
            features.append(density)
        
        # Determinism balance (fate + inevitability) - (choice + agency)
        fate_density = sum(1 for w in self.semantic_fields['fate'] if w in text_lower) / n_words
        inevitability_density = sum(1 for w in self.semantic_fields['inevitability'] if w in text_lower) / n_words
        choice_density = sum(1 for w in self.semantic_fields['choice'] if w in text_lower) / n_words
        agency_density = sum(1 for w in self.semantic_fields['agency'] if w in text_lower) / n_words
        
        determinism_balance = (fate_density + inevitability_density) - (choice_density + agency_density)
        features.append(determinism_balance)
        
        # Agency ratio
        agency_total = choice_density + agency_density
        fate_total = fate_density + inevitability_density
        agency_ratio = agency_total / (agency_total + fate_total + 1e-6)
        features.append(agency_ratio)
        
        return features
    
    def _extract_temporal_features(self, text: str) -> List[float]:
        """Extract temporal orientation features."""
        text_lower = text.lower()
        
        # Future-oriented language
        future_patterns = r'\b(will|shall|going to|might|could|may|would|future|tomorrow|ahead)\b'
        future_count = len(re.findall(future_patterns, text_lower, re.I))
        
        # Past-oriented language
        past_patterns = r'\b(was|were|had|did|before|ago|past|yesterday|once)\b'
        past_count = len(re.findall(past_patterns, text_lower, re.I))
        
        # Present-oriented language
        present_patterns = r'\b(is|are|am|now|currently|today|present)\b'
        present_count = len(re.findall(present_patterns, text_lower, re.I))
        
        total_temporal = future_count + past_count + present_count + 1e-6
        
        features = [
            future_count / total_temporal,  # Future orientation
            past_count / total_temporal,    # Past orientation
            present_count / total_temporal,  # Present orientation
            future_count / (past_count + 1e-6)  # Future/past ratio
        ]
        
        return features
    
    def _extract_information_theory_features(self, text: str) -> List[float]:
        """Extract information theory features (predictability)."""
        words = text.lower().split()
        
        if len(words) < 2:
            return [0.0] * 5
        
        # Word-level entropy
        word_freq = Counter(words)
        probs = np.array(list(word_freq.values())) / len(words)
        word_entropy = entropy(probs, base=2)
        
        # Character-level entropy
        chars = list(text.lower().replace(' ', ''))
        if len(chars) > 0:
            char_freq = Counter(chars)
            char_probs = np.array(list(char_freq.values())) / len(chars)
            char_entropy = entropy(char_probs, base=2)
        else:
            char_entropy = 0.0
        
        # Bigram entropy
        bigrams = [words[i] + ' ' + words[i+1] for i in range(len(words)-1)]
        if len(bigrams) > 0:
            bigram_freq = Counter(bigrams)
            bigram_probs = np.array(list(bigram_freq.values())) / len(bigrams)
            bigram_entropy = entropy(bigram_probs, base=2)
        else:
            bigram_entropy = 0.0
        
        # Predictability score (inverse of entropy, normalized)
        max_entropy = math.log2(len(word_freq) + 1e-6)
        predictability = 1.0 - (word_entropy / (max_entropy + 1e-6))
        
        # Redundancy (1 - normalized entropy)
        redundancy = 1.0 - (word_entropy / (math.log2(len(words)) + 1e-6))
        
        features = [
            float(word_entropy),
            float(char_entropy),
            float(bigram_entropy),
            float(predictability),
            float(redundancy)
        ]
        
        return features
    
    def _extract_agency_features(self, text: str) -> List[float]:
        """Extract agency vs patient features using spaCy."""
        if not self.use_spacy or self.nlp is None:
            return [0.0] * 5
        
        doc = self.nlp(text)
        
        agents = []  # Characters who ACT (free will signals)
        patients = []  # Characters acted UPON (determinism signals)
        actions = []  # Action verbs
        
        for token in doc:
            # Subject = agent
            if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PROPN", "PRON"]:
                agents.append(token.text.lower())
            
            # Passive subject = patient
            if token.dep_ == "nsubjpass" and token.pos_ in ["NOUN", "PROPN", "PRON"]:
                patients.append(token.text.lower())
            
            # Action verbs
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp"]:
                actions.append(token.text.lower())
        
        n_agents = len(set(agents))
        n_patients = len(set(patients))
        n_actions = len(set(actions))
        
        # Free will score = agents / (agents + patients)
        total_roles = n_agents + n_patients + 1e-6
        free_will_score = n_agents / total_roles
        
        # Agency ratio
        agency_ratio = n_agents / (n_patients + 1e-6)
        
        # Action density
        action_density = n_actions / (len(doc) + 1e-6)
        
        features = [
            float(n_agents),
            float(n_patients),
            float(free_will_score),
            float(agency_ratio),
            float(action_density)
        ]
        
        return features
    
    def _extract_causal_graph_features(self, text: str) -> List[float]:
        """Extract causal graph structure features."""
        # Simple causal relationship extraction
        # Look for causal connectors and build a simple graph
        
        causal_connectors = [
            'because', 'therefore', 'thus', 'so', 'as a result',
            'led to', 'caused', 'resulted in', 'due to', 'consequence'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        G = nx.DiGraph()
        
        # Extract events (simplified: sentences as events)
        events = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower().strip()
            if len(sent_lower) > 10:  # Filter very short sentences
                events.append(f"event_{i}")
                G.add_node(f"event_{i}")
        
        # Add edges based on causal connectors
        for i in range(len(sentences) - 1):
            sent_lower = sentences[i].lower()
            next_sent_lower = sentences[i+1].lower()
            
            # Check if current sentence has causal connector pointing forward
            if any(connector in sent_lower for connector in causal_connectors):
                if i < len(events) and i+1 < len(events):
                    G.add_edge(events[i], events[i+1], type='causal')
        
        # Calculate graph metrics
        if G.number_of_nodes() == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        # Path dependency (average shortest path length)
        try:
            if nx.is_strongly_connected(G):
                path_length = nx.average_shortest_path_length(G)
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.strongly_connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                if subgraph.number_of_nodes() > 1:
                    path_length = nx.average_shortest_path_length(subgraph)
                else:
                    path_length = 0.0
        except:
            path_length = 0.0
        
        # Branching factor (average out-degree)
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        branching_factor = np.mean(out_degrees) if out_degrees else 0.0
        
        # Critical nodes (articulation points in undirected version)
        try:
            G_undirected = G.to_undirected()
            critical_nodes = len(list(nx.articulation_points(G_undirected)))
        except:
            critical_nodes = 0.0
        
        # Deterministic ratio (causal edges / total edges)
        causal_edges = sum(1 for u, v, d in G.edges(data=True) if d.get('type') == 'causal')
        total_edges = G.number_of_edges()
        deterministic_ratio = causal_edges / (total_edges + 1e-6)
        
        features = [
            float(path_length),
            float(branching_factor),
            float(critical_nodes),
            float(deterministic_ratio)
        ]
        
        return features
    
    def _extract_observability_features(self, text: str) -> List[float]:
        """Extract observability features (visible vs hidden causality)."""
        text_lower = text.lower()
        
        # Explicit causality markers (visible)
        explicit_markers = [
            'because', 'therefore', 'thus', 'as a result', 'due to',
            'caused', 'led to', 'resulted in', 'consequence'
        ]
        explicit_count = sum(1 for marker in explicit_markers if marker in text_lower)
        
        # Hidden causality markers (implicit, mysterious)
        hidden_markers = [
            'somehow', 'mysteriously', 'unexplained', 'unknown reason',
            'suddenly', 'without warning', 'inexplicably', 'strangely'
        ]
        hidden_count = sum(1 for marker in hidden_markers if marker in text_lower)
        
        # Omniscient narrator markers (visible causality)
        omniscient_markers = [
            'knew', 'understood', 'realized', 'saw that', 'observed',
            'noticed', 'perceived', 'comprehended'
        ]
        omniscient_count = sum(1 for marker in omniscient_markers if marker in text_lower)
        
        total_observability = explicit_count + hidden_count + omniscient_count + 1e-6
        
        features = [
            explicit_count / total_observability,  # Visible causality ratio
            hidden_count / total_observability,    # Hidden causality ratio
            omniscient_count / total_observability  # Omniscient perspective ratio
        ]
        
        return features
    
    def _extract_nominative_agency_features(self, text: str) -> List[float]:
        """Extract nominative features related to agency/determinism."""
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) if words else 1
        
        features = []
        
        # 1. Proper name density (capitalized words that aren't sentence starts)
        # Split into sentences and check for proper names
        sentences = re.split(r'[.!?]+', text)
        proper_names = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                words_in_sent = sent.split()
                # Skip first word (likely capitalized for sentence start)
                for word in words_in_sent[1:]:
                    if word and word[0].isupper() and word.lower() not in ['i']:
                        proper_names.append(word)
        
        proper_name_density = len(proper_names) / n_words
        features.append(proper_name_density)
        
        # 2. Generic label ratio
        generic_count = sum(1 for label in self.nominative_fields['generic_labels'] 
                          if label in text_lower)
        generic_ratio = generic_count / n_words
        features.append(generic_ratio)
        
        # 3. Title pattern frequency (deterministic titles)
        title_count = sum(1 for title in self.nominative_fields['deterministic_titles'] 
                         if title in text_lower)
        title_frequency = title_count / n_words
        features.append(title_frequency)
        
        # 4. Name consistency score (proper names vs generic labels)
        name_consistency = proper_name_density / (generic_ratio + proper_name_density + 1e-6)
        features.append(name_consistency)
        
        # 5. Identity assertion patterns
        identity_count = sum(1 for marker in self.nominative_fields['identity_markers'] 
                           if marker in text_lower)
        identity_density = identity_count / n_words
        features.append(identity_density)
        
        # 6. Categorical language density (role labels)
        role_count = sum(1 for role in self.nominative_fields['role_labels'] 
                        if role in text_lower)
        categorical_density = role_count / n_words
        features.append(categorical_density)
        
        # 7. Agency naming patterns
        agency_name_count = sum(1 for name in self.nominative_fields['agency_names'] 
                               if name in text_lower)
        agency_naming_density = agency_name_count / n_words
        features.append(agency_naming_density)
        
        # 8. Deterministic vs agentic naming balance
        deterministic_naming = title_frequency + generic_ratio
        agentic_naming = proper_name_density + agency_naming_density
        naming_balance = (deterministic_naming - agentic_naming) / (deterministic_naming + agentic_naming + 1e-6)
        features.append(naming_balance)
        
        return features
    
    def _extract_character_naming_evolution(self, text: str) -> List[float]:
        """Track how character naming evolves through narrative."""
        # Split text into thirds to track evolution
        text_length = len(text)
        if text_length < 30:  # Too short to analyze evolution
            return [0.0] * 6
        
        third_length = text_length // 3
        beginning = text[:third_length].lower()
        middle = text[third_length:2*third_length].lower()
        end = text[2*third_length:].lower()
        
        features = []
        
        # 1. Names gained (generic → proper) - increase in proper names
        # Count proper names in each section
        def count_proper_names(text_section):
            sentences = re.split(r'[.!?]+', text_section)
            count = 0
            for sent in sentences:
                words = sent.strip().split()
                if len(words) > 1:
                    # Check non-first words for capitalization
                    for word in words[1:]:
                        if word and len(word) > 0 and word[0].isupper() and word.lower() not in ['i']:
                            count += 1
            return count
        
        beginning_proper = count_proper_names(text[:third_length])
        end_proper = count_proper_names(text[2*third_length:])
        names_gained_score = (end_proper - beginning_proper) / (beginning_proper + end_proper + 1)
        features.append(max(0.0, names_gained_score))  # Only positive gains
        
        # 2. Names lost (proper → generic) - increase in generic labels
        beginning_generic = sum(1 for label in self.nominative_fields['generic_labels'] 
                              if label in beginning)
        end_generic = sum(1 for label in self.nominative_fields['generic_labels'] 
                         if label in end)
        names_lost_score = (end_generic - beginning_generic) / (beginning_generic + end_generic + 1)
        features.append(max(0.0, names_lost_score))  # Only positive losses
        
        # 3. Title accumulation
        beginning_titles = sum(1 for title in self.nominative_fields['deterministic_titles'] 
                             if title in beginning)
        end_titles = sum(1 for title in self.nominative_fields['deterministic_titles'] 
                        if title in end)
        title_accumulation = (end_titles - beginning_titles) / (beginning_titles + end_titles + 1)
        features.append(max(0.0, title_accumulation))
        
        # 4. Identity shift score
        beginning_identity = sum(1 for marker in self.nominative_fields['identity_markers'] 
                               if marker in beginning)
        end_identity = sum(1 for marker in self.nominative_fields['identity_markers'] 
                          if marker in end)
        identity_shift = abs(end_identity - beginning_identity) / (beginning_identity + end_identity + 1)
        features.append(identity_shift)
        
        # 5. Agency evolution
        beginning_agency = sum(1 for name in self.nominative_fields['agency_names'] 
                             if name in beginning)
        end_agency = sum(1 for name in self.nominative_fields['agency_names'] 
                        if name in end)
        agency_evolution = (end_agency - beginning_agency) / (beginning_agency + end_agency + 1)
        features.append(agency_evolution)
        
        # 6. Overall naming stability (low change = stable = deterministic)
        total_change = abs(names_gained_score) + abs(names_lost_score) + abs(title_accumulation) + identity_shift
        naming_stability = 1.0 / (1.0 + total_change)
        features.append(naming_stability)
        
        return features
    
    def _calculate_composite_scores(
        self, 
        semantic_features: List[float],
        temporal_features: List[float],
        info_features: List[float]
    ) -> List[float]:
        """Calculate composite determinism and free will scores."""
        # Extract key components
        # semantic_features: [8 field densities, determinism_balance, agency_ratio]
        determinism_balance = semantic_features[-2] if len(semantic_features) >= 2 else 0.0
        agency_ratio = semantic_features[-1] if len(semantic_features) >= 1 else 0.0
        
        # temporal_features: [future, past, present, future/past]
        future_orientation = temporal_features[0] if len(temporal_features) > 0 else 0.0
        
        # info_features: [word_entropy, char_entropy, bigram_entropy, predictability, redundancy]
        predictability = info_features[3] if len(info_features) > 3 else 0.0
        
        # Composite determinism score (weighted)
        # Component 1: Temporal - past orientation = deterministic
        temporal_score = 1.0 - future_orientation  # Past = deterministic
        
        # Component 2: Semantic - fate language = deterministic
        semantic_score = (determinism_balance + 1.0) / 2.0  # Normalize to [0, 1]
        
        # Component 3: Predictability - predictable = deterministic
        predictability_score = predictability
        
        determinism_score = (
            self.temporal_weight * temporal_score +
            self.semantic_weight * semantic_score +
            self.predictability_weight * predictability_score
        )
        
        # Free will score (inverse of determinism, adjusted by agency)
        free_will_score = (1.0 - determinism_score) * agency_ratio
        
        # Narrative inevitability (how much does structure predict outcome?)
        inevitability_score = (semantic_score + predictability_score) / 2.0
        
        return [
            float(determinism_score),
            float(free_will_score),
            float(inevitability_score)
        ]
    
    def _calculate_n_features(self) -> int:
        """Calculate total number of output features."""
        n_semantic = len(self.semantic_fields) + 2  # fields + balance + ratio
        n_temporal = 4
        n_info = 5
        n_agency = 5 if self.use_spacy else 0
        n_graph = 4 if self.extract_causal_graphs else 0
        n_embedding = 4 if self.use_sentence_transformers else 0
        n_observability = 3 if self.track_observability else 0
        n_nominative_agency = 8  # From _extract_nominative_agency_features
        n_naming_evolution = 6   # From _extract_character_naming_evolution
        n_composite = 3
        
        return (n_semantic + n_temporal + n_info + n_agency + 
                n_graph + n_embedding + n_observability + 
                n_nominative_agency + n_naming_evolution + n_composite)
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names."""
        names = []
        
        # Semantic field densities
        for field in sorted(self.semantic_fields.keys()):
            names.append(f"semantic_density_{field}")
        names.extend(['determinism_balance', 'agency_ratio'])
        
        # Temporal features
        names.extend([
            'temporal_future_orientation',
            'temporal_past_orientation',
            'temporal_present_orientation',
            'temporal_future_past_ratio'
        ])
        
        # Information theory
        names.extend([
            'info_word_entropy',
            'info_char_entropy',
            'info_bigram_entropy',
            'info_predictability',
            'info_redundancy'
        ])
        
        # Agency features
        if self.use_spacy:
            names.extend([
                'agency_n_agents',
                'agency_n_patients',
                'agency_free_will_score',
                'agency_ratio',
                'agency_action_density'
            ])
        
        # Causal graph features
        if self.extract_causal_graphs:
            names.extend([
                'graph_path_dependency',
                'graph_branching_factor',
                'graph_critical_nodes',
                'graph_deterministic_ratio'
            ])
        
        # Embedding features
        if self.use_sentence_transformers:
            names.extend([
                'embedding_mean',
                'embedding_std',
                'embedding_min',
                'embedding_max'
            ])
        
        # Observability features
        if self.track_observability:
            names.extend([
                'observability_explicit_ratio',
                'observability_hidden_ratio',
                'observability_omniscient_ratio'
            ])
        
        # Nominative agency features
        names.extend([
            'nominative_proper_name_density',
            'nominative_generic_label_ratio',
            'nominative_title_frequency',
            'nominative_name_consistency',
            'nominative_identity_assertions',
            'nominative_categorical_density',
            'nominative_agency_naming_density',
            'nominative_naming_balance'
        ])
        
        # Character naming evolution features
        names.extend([
            'naming_evolution_names_gained',
            'naming_evolution_names_lost',
            'naming_evolution_title_accumulation',
            'naming_evolution_identity_shift',
            'naming_evolution_agency_evolution',
            'naming_evolution_naming_stability'
        ])
        
        # Composite scores
        names.extend([
            'composite_determinism_score',
            'composite_free_will_score',
            'composite_inevitability_score'
        ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of learned patterns."""
        field_freqs = self.metadata.get('field_frequencies', {})
        total_docs = self.metadata.get('total_docs', 0)
        
        interpretation = f"Free Will vs Determinism Analysis of {total_docs} narratives:\n\n"
        
        # Most common semantic fields
        top_fields = sorted(field_freqs.items(), key=lambda x: x[1], reverse=True)[:5]
        interpretation += "Most common semantic fields:\n"
        for field, freq in top_fields:
            interpretation += f"  - {field}: {freq*100:.1f}% of narratives\n"
        
        # Determinism indicators
        fate_freq = field_freqs.get('fate', 0)
        inevitability_freq = field_freqs.get('inevitability', 0)
        choice_freq = field_freqs.get('choice', 0)
        agency_freq = field_freqs.get('agency', 0)
        
        interpretation += "\nInterpretation:\n"
        if fate_freq + inevitability_freq > choice_freq + agency_freq:
            interpretation += "  - Deterministic patterns dominate (fate/inevitability > choice/agency)\n"
        else:
            interpretation += "  - Free will patterns dominate (choice/agency > fate/inevitability)\n"
        
        if self.use_sentence_transformers:
            interpretation += "  - Using SentenceTransformers for structural analysis\n"
        if self.use_spacy:
            interpretation += "  - Using spaCy for agency/patient extraction\n"
        if self.extract_causal_graphs:
            interpretation += "  - Extracting causal graph structures\n"
        
        return interpretation
    
    def calculate_narrative_determinism_score(self, text: str) -> float:
        """
        Calculate determinism score for a single narrative.
        
        Returns:
            float: Score from 0.0 (pure free will) to 1.0 (pure determinism)
        """
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted first")
        
        features = self._extract_document_features(text)
        
        # Find determinism score index
        feature_names = self.get_feature_names()
        try:
            determinism_idx = feature_names.index('composite_determinism_score')
            return float(features[determinism_idx])
        except ValueError:
            # Fallback: calculate from components
            semantic_features = self._extract_semantic_field_features(text)
            temporal_features = self._extract_temporal_features(text)
            info_features = self._extract_information_theory_features(text)
            composite = self._calculate_composite_scores(
                semantic_features, temporal_features, info_features
            )
            return composite[0]  # determinism_score

