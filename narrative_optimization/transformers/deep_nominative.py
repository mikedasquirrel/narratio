"""
Deep Nominative Transformer
===========================

Expands Golf's 30+ proper-noun feature stack into a universal transformer that
operates on ANY list of names (teams, players, founders, cases).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


VOWELS = set("aeiouy")
REGAL_TITLES = {"king", "queen", "prince", "princess", "duke", "sir", "lord"}
PRESTIGE_TOKENS = {"academy", "royal", "olympic", "federal", "supreme", "national"}
CULTURE_SUFFIXES = {
    "latin": ("ez", "es", "az", "ado", "ado", "ria"),
    "slavic": ("ov", "ova", "ski", "ska", "vic", "vich"),
    "french": ("eau", "ette", "ieux", "reaux"),
    "arabic": ("al", "ibn"),
    "nordic": ("sen", "sson", "dotter"),
}


class DeepNominativeTransformer(BaseEstimator, TransformerMixin):
    """Extract 40+ nominative richness features."""

    def __init__(self, max_names: int = 40):
        self.max_names = max_names
        self._feature_names = [
            "name_count",
            "avg_length",
            "std_length",
            "avg_syllables",
            "share_multisyllable",
            "vowel_ratio",
            "consonant_ratio",
            "avg_entropy",
            "alliteration_score",
            "unique_initials",
            "regal_density",
            "numeric_suffix_ratio",
            "prefix_mc_ratio",
            "prefix_von_ratio",
            "suffix_son_ratio",
            "suffix_ski_ratio",
            "latin_ratio",
            "slavic_ratio",
            "french_ratio",
            "arabic_ratio",
            "nordic_ratio",
            "cultural_span",
            "vowel_start_ratio",
            "vowel_end_ratio",
            "palindrome_ratio",
            "memorability_score",
            "prestige_token_density",
            "max_name_length",
            "short_name_ratio",
            "long_name_ratio",
            "letter_diversity",
            "bigram_uniqueness",
            "shared_bigram_ratio",
            "soft_consonant_ratio",
            "hard_consonant_ratio",
            "diaspora_marker_ratio",
            "assimilation_marker_ratio",
            "syllable_entropy",
            "uppercase_ratio",
            "hyphenated_ratio",
            "diacritic_ratio",
        ]

    def fit(self, X: Sequence[Iterable[str]], y=None):
        return self

    def transform(self, X: Sequence[Iterable[str]]):
        features = []
        for row in X:
            names = self._normalize(row)
            feat = self._featurize(names)
            features.append(feat)
        return np.array(features, dtype=float)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _normalize(self, row: Iterable[str]) -> List[str]:
        if row is None:
            return []
        if isinstance(row, str):
            row = re.split(r"[|,;/]+", row)

        normalized = []
        for value in row:
            if not value:
                continue
            if isinstance(value, dict):
                value = value.get("name") or value.get("full_name")
            if not value:
                continue
            text = str(value).strip()
            if text:
                normalized.append(text[:60])
            if len(normalized) >= self.max_names:
                break
        return normalized

    def _featurize(self, names: List[str]) -> List[float]:
        if not names:
            return [0.0] * len(self._feature_names)

        lengths = [len(name.replace(" ", "")) for name in names]
        syllables = [self._approx_syllables(name) for name in names]
        vowels = sum(self._count_vowels(name) for name in names)
        consonants = sum(self._count_consonants(name) for name in names)
        initials = [name[0].lower() for name in names if name]
        regal_hits = sum(
            1 for name in names if any(title in name.lower() for title in REGAL_TITLES)
        )
        numeric_suffix = sum(bool(re.search(r"(I|V|X|Jr|Sr)\b", name)) for name in names)
        prefix_mc = sum(name.lower().startswith(("mc", "mac")) for name in names)
        prefix_von = sum(" von " in name.lower() for name in names)
        suffix_son = sum(name.lower().endswith("son") for name in names)
        suffix_ski = sum(name.lower().endswith("ski") for name in names)
        vowel_start = sum(name[0].lower() in VOWELS for name in names if name)
        vowel_end = sum(name[-1].lower() in VOWELS for name in names if name)
        palindromes = sum(1 for name in names if self._is_palindrome(name))
        prestige_hits = sum(
            1 for name in names if any(tok in name.lower() for tok in PRESTIGE_TOKENS)
        )
        short_names = sum(len(name) <= 4 for name in names)
        long_names = sum(len(name) >= 10 for name in names)
        uppercase_ratio = sum(name.isupper() for name in names)
        hyphenated = sum("-" in name for name in names)
        diacritic = sum(self._has_diacritic(name) for name in names)

        cultures = Counter(self._infer_culture(name) for name in names)
        cultural_span = sum(1 for culture, count in cultures.items() if count > 0 and culture != "other")

        bigrams = [self._bigrams(name.lower()) for name in names]
        flat = [bg for sub in bigrams for bg in sub]
        unique_bigrams = len(set(flat))
        shared_bigrams = sum(1 for bg, count in Counter(flat).items() if count > 1)

        soft_consonants = sum(
            len(re.findall(r"[fshjlmnrv]", name.lower())) for name in names
        )
        hard_consonants = sum(
            len(re.findall(r"[bcdgkptqxz]", name.lower())) for name in names
        )
        diaspora_markers = sum(name.lower().startswith(("al ", "al-")) for name in names)
        assimilation_markers = sum(name.lower().endswith(("sen", "sson")) for name in names)

        entropy_values = [self._entropy(name) for name in names]
        syllable_entropy = self._entropy(syllables)

        n = len(names)
        vowel_ratio = vowels / max(vowels + consonants, 1)
        consonant_ratio = consonants / max(vowels + consonants, 1)

        return [
            float(n),
            float(np.mean(lengths)),
            float(np.std(lengths)),
            float(np.mean(syllables)),
            float(sum(s >= 3 for s in syllables) / n),
            float(vowel_ratio),
            float(consonant_ratio),
            float(np.mean(entropy_values)),
            float(self._alliteration_score(initials)),
            float(len(set(initials))),
            float(regal_hits / n),
            float(numeric_suffix / n),
            float(prefix_mc / n),
            float(prefix_von / n),
            float(suffix_son / n),
            float(suffix_ski / n),
            float(cultures.get("latin", 0) / n),
            float(cultures.get("slavic", 0) / n),
            float(cultures.get("french", 0) / n),
            float(cultures.get("arabic", 0) / n),
            float(cultures.get("nordic", 0) / n),
            float(cultural_span),
            float(vowel_start / n),
            float(vowel_end / n),
            float(palindromes / n),
            float(unique_bigrams / max(len(flat), 1)),
            float(prestige_hits / n),
            float(max(lengths)),
            float(short_names / n),
            float(long_names / n),
            float(self._letter_diversity(names)),
            float(unique_bigrams),
            float(shared_bigrams / max(len(flat), 1)),
            float(soft_consonants / max(len(flat), 1)),
            float(hard_consonants / max(len(flat), 1)),
            float(diaspora_markers / n),
            float(assimilation_markers / n),
            float(syllable_entropy),
            float(uppercase_ratio / n),
            float(hyphenated / n),
            float(diacritic / n),
        ]

    # ------------------------------------------------------------------ #
    # Primitive feature helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_vowels(name: str) -> int:
        return sum(1 for char in name.lower() if char in VOWELS)

    @staticmethod
    def _count_consonants(name: str) -> int:
        return sum(1 for char in name.lower() if char.isalpha() and char not in VOWELS)

    @staticmethod
    def _approx_syllables(name: str) -> int:
        cleaned = re.sub(r"[^a-z]", " ", name.lower())
        groups = re.findall(r"[aeiouy]{1,}", cleaned)
        return max(len(groups), 1)

    @staticmethod
    def _alliteration_score(initials: List[str]) -> float:
        if not initials:
            return 0.0
        counts = Counter(initials)
        dominant = counts.most_common(1)[0][1]
        return dominant / len(initials)

    @staticmethod
    def _entropy(values: Iterable[float]) -> float:
        if not values:
            return 0.0
        counts = Counter(values)
        total = sum(counts.values())
        probs = [count / total for count in counts.values()]
        return -sum(p * math.log(p + 1e-9) for p in probs)

    @staticmethod
    def _bigrams(text: str) -> List[str]:
        return [text[i : i + 2] for i in range(len(text) - 1) if text[i : i + 2].isalpha()]

    def _infer_culture(self, name: str) -> str:
        lower = name.lower()
        for culture, suffixes in CULTURE_SUFFIXES.items():
            if any(lower.endswith(suffix) for suffix in suffixes):
                return culture
        if lower.startswith("al "):
            return "arabic"
        return "other"

    @staticmethod
    def _letter_diversity(names: List[str]) -> float:
        joined = "".join(name.lower() for name in names if name)
        if not joined:
            return 0.0
        return len(set(joined)) / len(joined)

    @staticmethod
    def _is_palindrome(name: str) -> bool:
        stripped = re.sub(r"[^a-z]", "", name.lower())
        return bool(stripped) and stripped == stripped[::-1]

    @staticmethod
    def _has_diacritic(name: str) -> bool:
        return any(
            "WITH" in unicodedata.name(char, "")
            for char in name
            if char.isalpha()
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names)


__all__ = ["DeepNominativeTransformer"]


