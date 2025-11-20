"""
Namespace Ecology Transformer

Darwinian evolution applied to names - treats namespace as ecosystem with:
- Niche specialization (empty phonetic niches)
- Ecosystem positioning (relationship to category leaders)
- Resource competition (cognitive slot competition)
- Adaptive radiation (names evolve to avoid each other)
- Selection pressure (namespace crowding)
- Fitness landscape (occupied phonetic spaces)

Research Foundation:
- Bitcoin occupies "digital gold" niche, others must differentiate
- Band names show adaptive radiation from Led Zeppelin lineage
- Crypto namespace: "coin" suffix saturated → "protocol" niche opened
- Evolutionary theory (Darwin) + cultural evolution (Dawkins memes)

Core Insight:
Names exist in an ECOLOGY with:
- Limited resources (attention, cognitive slots, phonetic space)
- Competition for niches
- Selection pressure (only fit names survive)
- Speciation events (new patterns emerge when niches saturate)
- Fitness landscape topology (some spaces more fertile)

This is the SELECTION mechanism missing from Formula = heredity model.
"""

from typing import List, Dict, Any, Set
import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string


class NamespaceEcologyTransformer(NarrativeTransformer):
    """
    Analyzes namespace as Darwinian ecosystem.
    
    Tests hypothesis that naming follows evolutionary dynamics with selection
    pressure, resource competition, niche specialization, and adaptive radiation.
    
    Features extracted (35):
    - Niche specialization (phonetic uniqueness in category)
    - Ecosystem position (distance to category leaders)
    - Resource competition (cognitive slot overlap)
    - Adaptive radiation indicators (pattern divergence)
    - Selection pressure (namespace crowding metrics)
    - Fitness landscape position (occupied vs empty space)
    - Competitive exclusion (similar names fighting for niche)
    - Speciation events (new pattern emergence)
    - Ecological distance (phonetic-semantic separation)
    - Carrying capacity (market saturation proximity)
    
    Parameters
    ----------
    category_leaders : list of str, optional
        Known category leaders for ecosystem positioning
    """
    
    def __init__(self, category_leaders: List[str] = None):
        super().__init__(
            narrative_id="namespace_ecology",
            description="Namespace ecology: Darwinian evolution and selection in naming space"
        )
        
        self.category_leaders = category_leaders or []
        self.corpus_names = set()
        self.phonetic_space = {}  # Occupied phonetic regions
        self.semantic_space = {}  # Occupied semantic regions
        
    def fit(self, X, y=None):
        """
        Learn namespace ecology from corpus.
        
        Builds map of occupied phonetic-semantic space to compute
        niche availability, competition intensity, and selection pressure.
        
        Parameters
        ----------
        X : list of str
            Text documents (names/narratives)
        y : ignored
        
        Returns
        -------
        self
        """
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        # Extract all primary names/words
        for text in X:
            words = re.findall(r'\b\w+\b', text.lower())
            if words:
                primary_word = words[0]
                self.corpus_names.add(primary_word)
                
                # Map phonetic space (syllables × first letter)
                syllables = self._count_syllables(primary_word)
                first_letter = primary_word[0] if primary_word else 'x'
                phonetic_key = (syllables, first_letter)
                
                if phonetic_key not in self.phonetic_space:
                    self.phonetic_space[phonetic_key] = []
                self.phonetic_space[phonetic_key].append(primary_word)
                
                # Map semantic space (length × ending)
                length_category = len(primary_word) // 3  # 0-2, 3-5, 6-8, etc.
                ending = primary_word[-2:] if len(primary_word) >= 2 else primary_word
                semantic_key = (length_category, ending)
                
                if semantic_key not in self.semantic_space:
                    self.semantic_space[semantic_key] = []
                self.semantic_space[semantic_key].append(primary_word)
        
        # Compute ecosystem statistics
        self.metadata['corpus_size'] = len(self.corpus_names)
        self.metadata['phonetic_regions_occupied'] = len(self.phonetic_space)
        self.metadata['semantic_regions_occupied'] = len(self.semantic_space)
        self.metadata['avg_names_per_phonetic_region'] = np.mean([len(v) for v in self.phonetic_space.values()])
        self.metadata['max_crowded_region'] = max(len(v) for v in self.phonetic_space.values()) if self.phonetic_space else 0
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to namespace ecology features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array, shape (n_samples, 35)
            Namespace ecology feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_ecology_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_ecology_features(self, text: str) -> np.ndarray:
        """Extract all 35 namespace ecology features."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return np.zeros(35)
        
        primary_word = words[0] if words else ""
        if not primary_word:
            return np.zeros(35)
            
        features = []
        
        # === NICHE SPECIALIZATION (7 features) ===
        
        # 1. Phonetic niche crowding (how many in same phonetic region)
        syllables = self._count_syllables(primary_word)
        first_letter = primary_word[0] if primary_word else 'x'
        phonetic_key = (syllables, first_letter)
        niche_occupancy = len(self.phonetic_space.get(phonetic_key, []))
        features.append(niche_occupancy)
        
        # 2. Semantic niche crowding
        length_category = len(primary_word) // 3
        ending = primary_word[-2:] if len(primary_word) >= 2 else primary_word
        semantic_key = (length_category, ending)
        semantic_occupancy = len(self.semantic_space.get(semantic_key, []))
        features.append(semantic_occupancy)
        
        # 3. Total niche crowding (phonetic + semantic)
        total_crowding = niche_occupancy + semantic_occupancy
        features.append(total_crowding)
        
        # 4. Niche uniqueness (inverse crowding)
        niche_uniqueness = 1.0 / (1.0 + total_crowding)
        features.append(niche_uniqueness)
        
        # 5. Empty niche proximity (distance to empty phonetic region)
        # Check adjacent syllable counts
        empty_nearby = 0
        for adj_syl in [syllables-1, syllables+1]:
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                adj_key = (adj_syl, letter)
                if adj_key not in self.phonetic_space or len(self.phonetic_space[adj_key]) == 0:
                    empty_nearby += 1
        features.append(empty_nearby)
        
        # 6. Niche specialization score (how unique is this combination)
        # Rare phonetic + rare semantic = high specialization
        total_regions = max(1, len(self.phonetic_space))
        phonetic_rarity = 1.0 - (niche_occupancy / total_regions)
        semantic_rarity = 1.0 - (semantic_occupancy / total_regions)
        specialization = (phonetic_rarity + semantic_rarity) / 2.0
        features.append(specialization)
        
        # 7. Niche saturation (occupancy / carrying capacity estimate)
        carrying_capacity = 10  # Assume max ~10 names per niche
        saturation = niche_occupancy / carrying_capacity
        features.append(min(1.0, saturation))
        
        # === ECOSYSTEM POSITION (7 features) ===
        
        # 8. Distance to category leaders (if provided)
        if self.category_leaders:
            min_distance = min(
                self._levenshtein_distance(primary_word, leader.lower())
                for leader in self.category_leaders
            )
            features.append(min_distance)
        else:
            features.append(10)  # Maximum distance if no leaders
        
        # 9. Similarity to most common name
        if self.corpus_names:
            # Use first few chars as proxy
            prefix = primary_word[:3]
            prefix_matches = sum(1 for name in self.corpus_names if name.startswith(prefix))
            features.append(prefix_matches / len(self.corpus_names))
        else:
            features.append(0)
        
        # 10. Centrality in namespace (average distance to all names)
        # Sample 100 names for efficiency
        sample_names = list(self.corpus_names)[:100]
        if sample_names:
            avg_distance = np.mean([
                self._levenshtein_distance(primary_word, name)
                for name in sample_names
            ])
            features.append(avg_distance)
        else:
            avg_distance = 5
            features.append(avg_distance)
        
        # 11. Peripherality score (inverse of centrality)
        peripherality = avg_distance / 10.0  # Normalize (use the value just computed)
        features.append(peripherality)
        
        # 12. Ecosystem connectivity (how many "nearby" names)
        nearby_count = sum(
            1 for name in sample_names
            if self._levenshtein_distance(primary_word, name) <= 2
        ) if sample_names else 0
        features.append(nearby_count)
        
        # 13. Leader-distance ratio (distance to leader / average distance)
        # Use actual values instead of feature indices to avoid index errors
        leader_distance = features[7]  # Distance to leaders (feature index 7)
        avg_dist = features[9]  # Centrality/avg distance (feature index 9)
        leader_ratio = leader_distance / max(1, avg_dist)
        features.append(leader_ratio)
        
        # 14. Network density (connections / possible connections)
        max_connections = min(len(self.corpus_names), 100)
        network_density = nearby_count / max(1, max_connections)
        features.append(network_density)
        
        # === RESOURCE COMPETITION (7 features) ===
        
        # 15. Cognitive slot competition (similar phonetics competing for attention)
        # High niche crowding = high competition
        cognitive_competition = min(10.0, niche_occupancy)
        features.append(cognitive_competition)
        
        # 16. Phonetic overlap with competitors
        # Count names with shared phonetic features
        if sample_names:
            first_two = primary_word[:2]
            last_two = primary_word[-2:]
            overlap_count = sum(
                1 for name in sample_names
                if name.startswith(first_two) or name.endswith(last_two)
            )
            phonetic_overlap = overlap_count / len(sample_names)
            features.append(phonetic_overlap)
        else:
            features.append(0)
        
        # 17. Semantic overlap (similar meanings competing)
        # Proxy: similar word embeddings (simplified as length similarity)
        length_bucket = len(primary_word) // 2
        length_similar = sum(
            1 for name in sample_names
            if abs(len(name) - len(primary_word)) <= 1
        ) if sample_names else 0
        features.append(length_similar)
        
        # 18. Resource scarcity (inverse of available niches)
        occupied_ratio = len(self.phonetic_space) / 260  # ~26 letters × 10 syllable ranges
        resource_scarcity = occupied_ratio
        features.append(resource_scarcity)
        
        # 19. Competitive exclusion risk (Gause's principle)
        # If very similar name exists, one will be excluded
        if sample_names:
            closest_competitor_distance = min(
                (self._levenshtein_distance(primary_word, name), name)
                for name in sample_names
                if name != primary_word
            )[0] if len(sample_names) > 1 else 10
            exclusion_risk = 1.0 / (1.0 + closest_competitor_distance)
            features.append(exclusion_risk)
        else:
            features.append(0)
        
        # 20. Selection pressure intensity (crowding × competition)
        selection_pressure = cognitive_competition * phonetic_overlap
        features.append(selection_pressure)
        
        # 21. Fitness landscape position (in valley vs on peak)
        # Peak = unique + memorable, Valley = crowded + forgettable
        # Proxy: uniqueness × (1 - crowding)
        fitness_position = features[3] * (1.0 - features[6])
        features.append(fitness_position)
        
        # === ADAPTIVE RADIATION (7 features) ===
        
        # 22. Pattern divergence (how different from corpus average)
        corpus_avg_length = np.mean([len(name) for name in sample_names]) if sample_names else 5
        length_divergence = abs(len(primary_word) - corpus_avg_length) / corpus_avg_length
        features.append(length_divergence)
        
        # 23. Phonetic divergence (syllable difference from mode)
        syllable_distribution = Counter(
            self._count_syllables(name) for name in sample_names
        ) if sample_names else Counter()
        modal_syllables = syllable_distribution.most_common(1)[0][0] if syllable_distribution else 2
        syllable_divergence = abs(syllables - modal_syllables)
        features.append(syllable_divergence)
        
        # 24. Evolutionary distance from ancestor pattern
        # If category leaders exist, measure divergence
        if self.category_leaders:
            min_evo_distance = min(
                self._evolutionary_distance(primary_word, leader.lower())
                for leader in self.category_leaders
            )
            features.append(min_evo_distance)
        else:
            features.append(5)  # Unknown
        
        # 25. Adaptive radiation index (new pattern vs imitation)
        # High divergence + low similarity = radiation
        # Low divergence + high similarity = imitation
        radiation_index = (features[22] + features[23]) / 2.0
        features.append(radiation_index)
        
        # 26. Speciation potential (can this spawn new lineage?)
        # Novel + memorable + fit niche = speciation potential
        speciation_potential = features[3] * features[6] * features[20]
        features.append(speciation_potential)
        
        # 27. Lineage strength (how many descendants in sample)
        # Proxy: how many names share first 2 chars
        if sample_names and len(primary_word) >= 2:
            lineage_count = sum(1 for name in sample_names if name.startswith(primary_word[:2]))
            lineage_strength = lineage_count / len(sample_names)
            features.append(lineage_strength)
        else:
            features.append(0)
        
        # 28. Extinction risk (low fitness + high competition)
        extinction_risk = (1.0 - features[20]) * features[19]
        features.append(extinction_risk)
        
        # === SELECTION & FITNESS (7 features) ===
        
        # 29. Selection coefficient (fitness advantage vs. average)
        # Proxy: uniqueness + memorability - complexity
        # Would need actual outcomes for true selection coefficient
        selection_coef = features[3] + 0.5 - features[18]  # Rough proxy
        features.append(selection_coef)
        
        # 30. Fitness landscape gradient (uphill or downhill)
        # Peak-seeking: unique + empty niche nearby
        gradient = features[3] * features[4]
        features.append(gradient)
        
        # 31. Carrying capacity distance (how close to saturation)
        # Uses niche saturation from feature 7
        capacity_distance = 1.0 - features[6]
        features.append(capacity_distance)
        
        # 32. Competitive advantage (relative fitness)
        # Better position + less crowding = advantage
        competitive_advantage = features[20] * (1.0 - features[6])
        features.append(competitive_advantage)
        
        # 33. Ecological niche width (specialist vs generalist)
        # Narrow niche = specialist (high specificity)
        # Wide niche = generalist (low specificity)
        niche_width = 1.0 - features[5]  # Inverse of specialization
        features.append(niche_width)
        
        # 34. Symbiotic potential (could coexist with others)
        # High if different enough to avoid competition
        symbiotic = features[10] / 10.0  # Peripherality as proxy
        features.append(symbiotic)
        
        # 35. Overall ecological fitness (composite)
        # Combines: niche uniqueness + fitness position + carrying capacity + competitive advantage
        ecological_fitness = (
            features[3] +   # niche uniqueness
            features[20] +  # fitness position
            features[30] +  # capacity distance
            features[31]    # competitive advantage
        ) / 4.0
        features.append(ecological_fitness)
        
        return np.array(features)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables (simple algorithm)."""
        word = word.lower()
        vowels = 'aeiou'
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            count = max(1, count - 1)
        
        return max(1, count)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _evolutionary_distance(self, name1: str, name2: str) -> float:
        """
        Evolutionary distance combining phonetic + semantic divergence.
        
        Like phylogenetic distance in biology.
        """
        # Edit distance (genetic drift)
        edit_dist = self._levenshtein_distance(name1, name2)
        
        # Length difference (morphological change)
        length_diff = abs(len(name1) - len(name2))
        
        # Syllable difference (prosodic evolution)
        syl_diff = abs(self._count_syllables(name1) - self._count_syllables(name2))
        
        # Composite evolutionary distance
        evo_distance = (edit_dist * 0.5 + length_diff * 0.3 + syl_diff * 0.2)
        
        return evo_distance
    
    def get_feature_names(self) -> List[str]:
        """Return names of all 35 features."""
        return [
            # Niche specialization (7)
            'phonetic_niche_crowding', 'semantic_niche_crowding', 'total_niche_crowding',
            'niche_uniqueness', 'empty_niche_proximity', 'niche_specialization_score',
            'niche_saturation',
            
            # Ecosystem position (7)
            'distance_to_leaders', 'similarity_to_common', 'namespace_centrality',
            'namespace_peripherality', 'ecosystem_connectivity', 'leader_distance_ratio',
            'network_density',
            
            # Resource competition (7)
            'cognitive_slot_competition', 'phonetic_overlap', 'semantic_overlap',
            'resource_scarcity', 'competitive_exclusion_risk', 'selection_pressure',
            'fitness_landscape_position',
            
            # Adaptive radiation (7)
            'pattern_divergence', 'phonetic_divergence', 'evolutionary_distance',
            'radiation_index', 'speciation_potential', 'lineage_strength', 'extinction_risk',
            
            # Selection & fitness (7)
            'selection_coefficient', 'fitness_gradient', 'carrying_capacity_distance',
            'competitive_advantage', 'ecological_niche_width', 'symbiotic_potential',
            'overall_ecological_fitness'
        ]
    
    def interpret_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Interpret namespace ecology features in plain English."""
        interpretation = {
            'summary': self._generate_summary(features),
            'features': {},
            'insights': []
        }
        
        # Niche analysis
        niche_crowding = features[2]
        if niche_crowding > 20:
            interpretation['insights'].append(f"HIGHLY crowded niche ({niche_crowding:.0f} competitors) - intense selection pressure")
        elif niche_crowding < 3:
            interpretation['insights'].append(f"Empty niche ({niche_crowding:.0f} competitors) - speciation opportunity")
        
        # Ecosystem position
        leader_distance = features[7]
        if leader_distance <= 2:
            interpretation['insights'].append("VERY close to category leader - substitution risk or coattail benefit")
        elif leader_distance >= 8:
            interpretation['insights'].append("Far from leaders - independent niche or peripheral position")
        
        # Fitness
        ecological_fitness = features[34]
        if ecological_fitness > 0.7:
            interpretation['insights'].append("HIGH ecological fitness - strong niche position")
        elif ecological_fitness < 0.3:
            interpretation['insights'].append("LOW ecological fitness - extinction risk")
        
        return interpretation
    
    def _generate_summary(self, features: np.ndarray) -> str:
        """Generate plain English summary of namespace ecology."""
        niche_uniqueness = features[3]
        ecological_fitness = features[34]
        niche_crowding = features[2]
        
        if niche_uniqueness > 0.7 and ecological_fitness > 0.6:
            return f"Optimal ecological position: Unique niche ({niche_uniqueness:.2f}), high fitness ({ecological_fitness:.2f}). Low competition."
        elif niche_crowding > 15:
            return f"Crowded niche ({niche_crowding:.0f} competitors): High selection pressure, competitive exclusion risk."
        elif niche_uniqueness < 0.3:
            return f"Poor niche positioning: High overlap, low uniqueness ({niche_uniqueness:.2f}), extinction risk."
        else:
            return f"Moderate ecological position: {niche_crowding:.0f} competitors, fitness {ecological_fitness:.2f}."

