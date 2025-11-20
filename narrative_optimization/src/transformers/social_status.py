"""
Social Status Transformer

Captures class, education, and prestige markers that reveal how narrative force
operates domain-relatively through social stratification. Same phonetics carry
different meanings in different social contexts.

Research Foundation:
- Baby names: "Bentley" signals aspirational working-class vs "Henry" (old money)
- Academic: "J. Smith" (initial) signals prestige vs "John Smith"
- Crypto: "Capital" morphemes signal finance-class legitimacy
- Bands: Working-class punk names vs elite art-rock names
- Geographic prestige: Urban/coastal vs. rural associations

Core Insight:
Narrative force is RELATIVE to social context. What predicts success in one
class/education/geographic context fails in another. This is the "domain-gravitational"
aspect of the theory - gravity varies by social position.

Critical confound: Currently unmeasured but affects ALL domains.

Universal across domains:
- Class associations affect perception and access
- Education signals credibility differently by audience
- Prestige markers open/close doors
- Geographic associations activate local networks
- Aspiration vs. inheritance signals predict trajectories
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string


class SocialStatusTransformer(NarrativeTransformer):
    """
    Analyzes social class, status, and prestige markers in narratives.
    
    Tests hypothesis that narrative force operates domain-relatively through
    social stratification - same features predict differently by social context.
    
    Features extracted (20):
    - Class associations (high SES vs low SES language)
    - Education level markers (technical, academic, vernacular)
    - Prestige signals (elite, exclusive, prestigious)
    - Geographic prestige (urban/coastal vs rural)
    - Aspiration level (upward mobility signals)
    - Status consistency (coherent class signals)
    - Credential references (degrees, titles, awards)
    - Institutional affiliations (elite institutions)
    - Sophistication markers (cultural capital)
    - Accessibility (inclusive vs exclusive language)
    
    Parameters
    ----------
    detect_geography : bool
        Whether to extract geographic prestige markers
    """
    
    def __init__(self, detect_geography: bool = True):
        super().__init__(
            narrative_id="social_status",
            description="Social status: class, education, prestige markers and domain-relative operation"
        )
        
        self.detect_geography = detect_geography
        
        # High SES language
        self.high_ses_markers = [
            'sophisticated', 'refined', 'elegant', 'exclusive', 'premium', 'luxury',
            'elite', 'prestigious', 'distinguished', 'upscale', 'high-end', 'finest',
            'curated', 'artisan', 'bespoke', 'boutique', 'heritage', 'legacy',
            'investment', 'portfolio', 'capital', 'wealth', 'affluent', 'prosperity'
        ]
        
        # Low SES language (not pejorative - working class authentic)
        self.working_class_markers = [
            'affordable', 'budget', 'value', 'practical', 'simple', 'basic',
            'everyday', 'regular', 'normal', 'common', 'standard', 'ordinary',
            'folks', 'guys', 'stuff', 'things', 'deal', 'cheap', 'bargain'
        ]
        
        # Aspirational language (upward mobility)
        self.aspirational_markers = [
            'aspire', 'dream', 'achieve', 'succeed', 'rise', 'climb', 'reach',
            'breakthrough', 'elevate', 'upgrade', 'better', 'improve', 'advance',
            'opportunity', 'potential', 'become', 'transform', 'ambition', 'goal'
        ]
        
        # Education level markers
        self.high_education_markers = [
            'research', 'study', 'analysis', 'theory', 'hypothesis', 'methodology',
            'empirical', 'systematic', 'comprehensive', 'rigorous', 'scholarly',
            'academic', 'intellectual', 'scientific', 'technical', 'specialized'
        ]
        
        self.vernacular_markers = [
            'like', 'just', 'really', 'pretty', 'kinda', 'sorta', 'lots', 'tons',
            'super', 'totally', 'basically', 'literally', 'actually', 'honestly'
        ]
        
        # Credential references
        self.credentials = [
            r'\bdr\.?\b', r'\bphd\b', r'\bmd\b', r'\bmba\b', r'\bma\b', r'\bms\b',
            r'\bprof\b', r'\bprofessor\b', r'\bresearcher\b', r'\bscholar\b',
            r'\bexpert\b', r'\bspecialist\b', r'\bcertified\b', r'\blicensed\b'
        ]
        
        # Prestige institutions
        self.elite_institutions = {
            'harvard', 'yale', 'princeton', 'stanford', 'mit', 'oxford', 'cambridge',
            'ivy league', 'prestigious', 'top-tier', 'leading', 'premier', 'renowned'
        }
        
        # Sophistication/cultural capital markers
        self.sophistication_markers = [
            'nuanced', 'subtle', 'complex', 'intricate', 'layered', 'multifaceted',
            'profound', 'insightful', 'thoughtful', 'considered', 'cultivated',
            'cosmopolitan', 'worldly', 'cultured', 'discerning', 'connoisseur'
        ]
        
        # Exclusivity vs. inclusivity
        self.exclusive_markers = [
            'exclusive', 'select', 'limited', 'private', 'members only', 'invitation',
            'elite', 'restricted', 'privilege', 'access', 'insider', 'vip'
        ]
        
        self.inclusive_markers = [
            'everyone', 'all', 'open', 'accessible', 'public', 'free', 'welcome',
            'inclusive', 'community', 'together', 'shared', 'common', 'universal'
        ]
        
        # Geographic prestige
        self.high_prestige_locations = {
            'manhattan', 'silicon valley', 'boston', 'san francisco', 'los angeles',
            'london', 'paris', 'tokyo', 'new york', 'california', 'switzerland',
            'monaco', 'beverly hills', 'upper east side', 'soho', 'tribeca'
        }
        
        self.working_class_locations = {
            'rust belt', 'midwest', 'rural', 'small town', 'heartland', 'main street',
            'suburban', 'provincial', 'local', 'regional', 'hometown'
        }
        
        # Occupational prestige
        self.high_prestige_occupations = {
            'ceo', 'executive', 'director', 'partner', 'founder', 'investor',
            'surgeon', 'professor', 'attorney', 'architect', 'engineer', 'scientist'
        }
    
    def fit(self, X, y=None):
        """
        Learn status patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        # Analyze corpus class distribution
        class_distribution = {'high_ses': 0, 'working_class': 0, 'aspirational': 0}
        education_distribution = {'high_education': 0, 'vernacular': 0}
        
        for text in X:
            text_lower = text.lower()
            
            # Class markers
            class_distribution['high_ses'] += sum(1 for m in self.high_ses_markers if m in text_lower)
            class_distribution['working_class'] += sum(1 for m in self.working_class_markers if m in text_lower)
            class_distribution['aspirational'] += sum(1 for m in self.aspirational_markers if m in text_lower)
            
            # Education markers
            education_distribution['high_education'] += sum(1 for m in self.high_education_markers if m in text_lower)
            education_distribution['vernacular'] += sum(1 for m in self.vernacular_markers if m in text_lower)
        
        # Metadata
        self.metadata['class_distribution'] = class_distribution
        self.metadata['dominant_class'] = max(class_distribution.items(), key=lambda x: x[1])[0]
        self.metadata['education_distribution'] = education_distribution
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to social status features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array, shape (n_samples, 20)
            Social status feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_status_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_status_features(self, text: str) -> np.ndarray:
        """Extract all 20 social status features."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = max(1, len(words))
        
        features = []
        
        # === CLASS ASSOCIATIONS (7 features) ===
        
        # 1. High SES marker density
        high_ses_count = sum(1 for m in self.high_ses_markers if m in text_lower)
        features.append(high_ses_count / word_count * 100)
        
        # 2. Working class marker density
        working_class_count = sum(1 for m in self.working_class_markers if m in text_lower)
        features.append(working_class_count / word_count * 100)
        
        # 3. Class orientation (high SES vs working class)
        total_class = high_ses_count + working_class_count
        class_orientation = high_ses_count / total_class if total_class > 0 else 0.5
        features.append(class_orientation)  # 0 = working class, 1 = high SES, 0.5 = neutral
        
        # 4. Aspirational language density
        aspirational_count = sum(1 for m in self.aspirational_markers if m in text_lower)
        features.append(aspirational_count / word_count * 100)
        
        # 5. Status consistency (clear vs. mixed signals)
        if total_class > 0:
            dominant_class = max(high_ses_count, working_class_count)
            consistency = dominant_class / total_class
        else:
            consistency = 0.5
        features.append(consistency)
        
        # 6. Aspiration-achievement ratio
        # High aspiration + low achievement markers = aspirational class
        achievement_markers = ['achieved', 'accomplished', 'succeeded', 'reached', 'attained']
        achievement_count = sum(1 for m in achievement_markers if m in text_lower)
        aspiration_ratio = aspirational_count / max(1, achievement_count + aspirational_count)
        features.append(aspiration_ratio)
        
        # 7. Mobility language (upward trajectory)
        mobility_words = ['rise', 'climb', 'ascend', 'elevate', 'upgrade', 'improve', 'better', 'advance']
        mobility_count = sum(1 for m in mobility_words if m in text_lower)
        features.append(mobility_count / word_count * 100)
        
        # === EDUCATION MARKERS (5 features) ===
        
        # 8. High education marker density
        high_ed_count = sum(1 for m in self.high_education_markers if m in text_lower)
        features.append(high_ed_count / word_count * 100)
        
        # 9. Vernacular marker density
        vernacular_count = sum(1 for m in self.vernacular_markers if m in text_lower)
        features.append(vernacular_count / word_count * 100)
        
        # 10. Education orientation (high vs vernacular)
        total_education = high_ed_count + vernacular_count
        education_orientation = high_ed_count / total_education if total_education > 0 else 0.5
        features.append(education_orientation)
        
        # 11. Credential references count
        credential_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.credentials)
        features.append(credential_count)
        
        # 12. Technical jargon density (proxy for specialized education)
        # Words > 10 chars, Latinate roots
        long_words = [w for w in words if len(w) > 10]
        jargon_density = len(long_words) / word_count * 100
        features.append(jargon_density)
        
        # === PRESTIGE SIGNALS (5 features) ===
        
        # 13. Elite institution mentions
        elite_inst_count = sum(1 for inst in self.elite_institutions if inst in text_lower)
        features.append(elite_inst_count)
        
        # 14. Sophistication marker density
        sophistication_count = sum(1 for m in self.sophistication_markers if m in text_lower)
        features.append(sophistication_count / word_count * 100)
        
        # 15. Exclusive vs. inclusive orientation
        exclusive_count = sum(1 for m in self.exclusive_markers if m in text_lower)
        inclusive_count = sum(1 for m in self.inclusive_markers if m in text_lower)
        total_exclusivity = exclusive_count + inclusive_count
        exclusive_orientation = exclusive_count / total_exclusivity if total_exclusivity > 0 else 0.5
        features.append(exclusive_orientation)  # 0 = inclusive, 1 = exclusive
        
        # 16. High-prestige occupation mentions
        prestige_occ_count = sum(1 for occ in self.high_prestige_occupations if occ in text_lower)
        features.append(prestige_occ_count)
        
        # 17. Overall prestige score (composite)
        prestige_score = (
            features[0] / 10 +      # high SES density
            features[7] / 10 +      # high education density
            elite_inst_count * 0.2 + # institutions
            features[13] / 10 +     # sophistication
            features[14]            # exclusivity
        ) / 5.0
        features.append(min(1.0, prestige_score))
        
        # === GEOGRAPHIC PRESTIGE (3 features) ===
        
        if self.detect_geography:
            # 18. High-prestige location mentions
            high_prestige_geo = sum(1 for loc in self.high_prestige_locations if loc in text_lower)
            features.append(high_prestige_geo)
            
            # 19. Working-class location mentions
            working_class_geo = sum(1 for loc in self.working_class_locations if loc in text_lower)
            features.append(working_class_geo)
            
            # 20. Geographic prestige orientation
            total_geo = high_prestige_geo + working_class_geo
            geo_orientation = high_prestige_geo / total_geo if total_geo > 0 else 0.5
            features.append(geo_orientation)
        else:
            features.extend([0.0, 0.0, 0.5])
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return names of all 20 features."""
        return [
            # Class associations (7)
            'high_ses_density', 'working_class_density', 'class_orientation',
            'aspirational_density', 'status_consistency', 'aspiration_achievement_ratio',
            'mobility_language',
            
            # Education markers (5)
            'high_education_density', 'vernacular_density', 'education_orientation',
            'credential_count', 'technical_jargon_density',
            
            # Prestige signals (5)
            'elite_institution_count', 'sophistication_density', 'exclusive_vs_inclusive',
            'prestige_occupation_count', 'overall_prestige_score',
            
            # Geographic prestige (3)
            'high_prestige_locations', 'working_class_locations', 'geographic_orientation'
        ]
    
    def interpret_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Interpret social status features in plain English.
        
        Parameters
        ----------
        features : array, shape (20,)
            Feature vector for one document
        
        Returns
        -------
        interpretation : dict
            Plain English interpretation
        """
        names = self.get_feature_names()
        
        interpretation = {
            'summary': self._generate_summary(features),
            'features': {},
            'insights': []
        }
        
        # Class orientation
        class_orientation = features[2]
        if class_orientation > 0.65:
            interpretation['insights'].append("High SES framing - appeals to affluent audience")
        elif class_orientation < 0.35:
            interpretation['insights'].append("Working-class authentic framing - accessible positioning")
        else:
            interpretation['insights'].append("Class-neutral framing")
        
        # Aspirational
        aspirational = features[3]
        if aspirational > 2.0:
            interpretation['insights'].append("Strong aspirational language - upward mobility narrative")
        
        # Education
        education = features[9]
        if education > 0.65:
            interpretation['insights'].append("High education framing - technical/academic audience")
        elif education < 0.35:
            interpretation['insights'].append("Vernacular style - accessible to broad audience")
        
        # Prestige
        prestige = features[16]
        if prestige > 0.6:
            interpretation['insights'].append("HIGH prestige positioning - elite/exclusive framing")
        elif prestige < 0.3:
            interpretation['insights'].append("LOW prestige positioning - accessible/democratic framing")
        
        # Exclusivity
        exclusivity = features[14]
        if exclusivity > 0.65:
            interpretation['insights'].append("Exclusive positioning - limited access narrative")
        elif exclusivity < 0.35:
            interpretation['insights'].append("Inclusive positioning - open access narrative")
        
        return interpretation
    
    def _generate_summary(self, features: np.ndarray) -> str:
        """Generate plain English summary."""
        class_orientation = features[2]
        education = features[9]
        prestige = features[16]
        exclusivity = features[14]
        
        summary_parts = []
        
        # Class
        if class_orientation > 0.65:
            summary_parts.append("High SES positioning")
        elif class_orientation < 0.35:
            summary_parts.append("Working-class authentic positioning")
        else:
            summary_parts.append("Class-neutral positioning")
        
        # Education
        if education > 0.65:
            summary_parts.append("academic/technical framing")
        elif education < 0.35:
            summary_parts.append("vernacular/accessible style")
        
        # Prestige
        if prestige > 0.6:
            summary_parts.append("elite/prestigious")
        elif prestige < 0.3:
            summary_parts.append("democratic/accessible")
        
        # Exclusivity
        if exclusivity > 0.65:
            summary_parts.append("exclusive access")
        elif exclusivity < 0.35:
            summary_parts.append("inclusive/open")
        
        return ", ".join(summary_parts) + "."

