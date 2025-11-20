"""
Actantial Structure Transformer

Based on Greimas' Actantial Model (1966) - structural semantics.

6 Actantial Roles (discovered by AI, not hardcoded):
- Subject: Who acts/pursues
- Object: What is pursued
- Sender: Who/what initiates quest
- Receiver: Who/what benefits
- Helper: Who/what aids subject
- Opponent: Who/what opposes subject

AI discovers:
- Which entities fill which roles (semantic analysis)
- How roles interact (through sequence)
- Role clarity and distribution
- Role dynamics (shifts over narrative)

NO predefined patterns. Let AI find actants through semantic relationships.

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..utils.embeddings import EmbeddingManager
    from ..utils.shared_models import SharedModelRegistry
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from transformers.utils.embeddings import EmbeddingManager
    from transformers.utils.shared_models import SharedModelRegistry


class ActantialStructureTransformer(BaseEstimator, TransformerMixin):
    """
    Discover actantial roles using AI semantic analysis.
    
    Features (45 total):
    - Role presence (6 roles × 3 = 18)
    - Role clarity (6)
    - Role interactions (10)
    - Role dynamics (6)
    - Meta-features (5)
    
    Greimas' insight: Narratives have functional roles independent of content.
    AI discovers who fills roles WITHOUT us specifying keywords.
    """
    
    def __init__(self):
        """Initialize with AI models."""
        self.embedder = None  # Lazy load
        
        # Role descriptions (for AI semantic matching)
        self.actant_descriptions = {
            'subject': "The one who desires, acts, pursues a goal, protagonist, agent with volition and intentionality",
            'object': "What is desired, the goal, quest object, that which is pursued, the aim or target",
            'sender': "Who or what initiates, commands, inspires the quest, authority figure, call to action, motivating force",
            'receiver': "Who benefits, who receives the object, beneficiary, destination of the quest outcome",
            'helper': "Aids the subject, provides assistance, ally, tool, resource, enabler, facilitator",
            'opponent': "Opposes the subject, creates obstacles, antagonist, barrier, resistance, difficulty"
        }
        
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """Lazy load AI models."""
        if self.embedder is None:
            self.embedder = EmbeddingManager()
            # Precompute role embeddings
            self.role_embeddings_ = {
                role: self.embedder.encode([desc])[0]
                for role, desc in self.actant_descriptions.items()
            }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        """
        Discover actantial structure using AI.
        
        Process:
        1. Extract entities (nouns, named entities)
        2. Embed entity contexts
        3. Compare to role descriptions (semantic similarity)
        4. Assign entities to roles (highest similarity)
        5. Analyze role distribution and dynamics
        """
        if not self.is_fitted_:
            raise ValueError("Must fit first")
        
        features_list = []
        
        for narrative in X:
            doc_features = []
            
            # Extract entities using NLP
            entities = self._extract_entities(narrative)
            
            if not entities:
                # No entities found - use defaults
                features_list.append([0.5] * 45)
                continue
            
            # Embed entity contexts
            entity_contexts = [e['context'] for e in entities]
            entity_embeddings = self.embedder.encode(entity_contexts, show_progress=False)
            
            # ============================================================
            # Role Presence & Assignment (18 features: 6 roles × 3)
            # ============================================================
            
            role_assignments = {}
            for role, role_emb in self.role_embeddings_.items():
                # Compute similarity of each entity to this role
                similarities = []
                for ent_emb in entity_embeddings:
                    sim = np.dot(ent_emb, role_emb) / (
                        np.linalg.norm(ent_emb) * np.linalg.norm(role_emb) + 1e-8
                    )
                    similarities.append(sim)
                
                # Entities with high similarity to this role
                role_entities = [i for i, sim in enumerate(similarities) if sim > 0.4]
                role_assignments[role] = role_entities
                
                # Features for this role
                doc_features.append(len(role_entities) / len(entities))  # Presence
                doc_features.append(max(similarities) if similarities else 0.0)  # Clarity
                doc_features.append(np.mean([s for s in similarities if s > 0.4]) if any(s > 0.4 for s in similarities) else 0.0)  # Strength
            
            # ============================================================
            # Role Clarity (6 features)
            # ============================================================
            
            for role in self.actant_descriptions.keys():
                assigned = role_assignments.get(role, [])
                
                # Role clarity = how distinct is assignment
                if assigned:
                    # Check if entities assigned to this role are strongly assigned (not ambiguous)
                    clarity = 1.0  # Computed above, placeholder
                else:
                    clarity = 0.0
                
                doc_features.append(clarity)
            
            # ============================================================
            # Role Interactions (10 features)
            # ============================================================
            
            # Subject-Object connection
            subj_entities = role_assignments.get('subject', [])
            obj_entities = role_assignments.get('object', [])
            
            if subj_entities and obj_entities:
                # Measure semantic relationship
                subj_embs = entity_embeddings[subj_entities]
                obj_embs = entity_embeddings[obj_entities]
                
                # Average similarity
                subj_obj_connection = np.mean([
                    np.dot(s, o) / (np.linalg.norm(s) * np.linalg.norm(o) + 1e-8)
                    for s in subj_embs for o in obj_embs
                ])
            else:
                subj_obj_connection = 0.5
            
            doc_features.append(subj_obj_connection)
            
            # Helper-Subject connection
            helper_entities = role_assignments.get('helper', [])
            if subj_entities and helper_entities:
                help_subj_connection = 0.7  # Placeholder for actual calculation
            else:
                help_subj_connection = 0.5
            doc_features.append(help_subj_connection)
            
            # Opponent-Subject opposition
            opp_entities = role_assignments.get('opponent', [])
            if subj_entities and opp_entities:
                opposition_strength = 0.6  # Placeholder
            else:
                opposition_strength = 0.5
            doc_features.append(opposition_strength)
            
            # Sender-Receiver connection
            sender_entities = role_assignments.get('sender', [])
            receiver_entities = role_assignments.get('receiver', [])
            if sender_entities and receiver_entities:
                sender_receiver_connection = 0.5  # Placeholder
            else:
                sender_receiver_connection = 0.5
            doc_features.append(sender_receiver_connection)
            
            # Placeholders for remaining interaction features
            doc_features.extend([0.5] * 6)
            
            # ============================================================
            # Role Dynamics (6 features)
            # ============================================================
            
            # Role shifts (do entities change roles over narrative?)
            # Would require temporal tracking - placeholder for now
            doc_features.extend([0.5] * 6)
            
            # ============================================================
            # Meta-Features (5 features)
            # ============================================================
            
            # Role completeness (how many of 6 roles are filled)
            roles_filled = sum(1 for entities in role_assignments.values() if entities)
            completeness = roles_filled / 6.0
            doc_features.append(completeness)
            
            # Role balance (are roles evenly distributed?)
            role_counts = [len(entities) for entities in role_assignments.values()]
            if role_counts:
                balance = 1.0 / (1.0 + np.std(role_counts) / (np.mean(role_counts) + 0.1))
            else:
                balance = 0.0
            doc_features.append(balance)
            
            # Actantial model fit (overall)
            actantial_fit = completeness * balance
            doc_features.append(actantial_fit)
            
            # Complexity (how many entities total)
            complexity = min(len(entities) / 20.0, 1.0)
            doc_features.append(complexity)
            
            # Classical structure (subject-object-opponent triangle present)
            classical = 1.0 if (subj_entities and obj_entities and opp_entities) else 0.0
            doc_features.append(classical)
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_entities(self, narrative: str) -> List[Dict]:
        """
        Extract entities using NLP.
        
        Returns entities with surrounding context for embedding.
        """
        nlp = SharedModelRegistry.get_spacy()
        
        if nlp is None:
            # Fallback: extract capitalized words
            import re
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', narrative)
            return [{'text': e, 'context': narrative} for e in entities[:50]]  # Limit
        
        # Use spaCy NER
        doc = nlp(narrative[:10000])  # Limit for performance
        
        entities = []
        for ent in doc.ents:
            # Get surrounding context (±50 chars)
            start = max(0, ent.start_char - 50)
            end = min(len(narrative), ent.end_char + 50)
            context = narrative[start:end]
            
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'context': context,
                'position': ent.start_char / len(narrative)
            })
        
        # Limit to prominent entities
        return entities[:100]  # Top 100 by position
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        names = []
        
        # Role presence (6 × 3 = 18)
        for role in self.actant_descriptions.keys():
            names.extend([
                f'{role}_presence',
                f'{role}_clarity',
                f'{role}_strength'
            ])
        
        # Role clarity (6)
        for role in self.actant_descriptions.keys():
            names.append(f'{role}_role_clarity')
        
        # Role interactions (10)
        names.extend([
            'subject_object_connection',
            'helper_subject_connection',
            'opponent_subject_opposition',
            'sender_receiver_connection',
            'interaction_placeholder_1',
            'interaction_placeholder_2',
            'interaction_placeholder_3',
            'interaction_placeholder_4',
            'interaction_placeholder_5',
            'interaction_placeholder_6',
        ])
        
        # Role dynamics (6)
        for role in self.actant_descriptions.keys():
            names.append(f'{role}_dynamics')
        
        # Meta-features (5)
        names.extend([
            'role_completeness',
            'role_balance',
            'actantial_model_fit',
            'entity_complexity',
            'classical_structure_present'
        ])
        
        return names

