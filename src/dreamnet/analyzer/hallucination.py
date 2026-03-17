"""Hallucination detection and classification for LLM outputs."""

from __future__ import annotations

import re
from collections import Counter

from dreamnet.models import HallucinationType, LLMOutput


# Indicators for different hallucination types
_CONFABULATION_MARKERS = [
    r"\b(studies show|research indicates|according to|it is well known)\b",
    r"\b(in \d{4},?\s+\w+\s+(discovered|found|proved))\b",
    r"\b(the .+ institute|university of .+ found)\b",
    r"\b(statistics? (show|indicate|prove))\b",
]

_CREATIVE_MARKERS = [
    r"\b(imagine|once upon|in a world where)\b",
    r"\b(metaphorically|poetically|symbolically)\b",
    r"\b(let me paint a picture|picture this)\b",
    r"[!]{2,}|[.]{3,}",
]

_HEDGING_MARKERS = [
    r"\b(I think|I believe|it seems|perhaps|possibly|might be)\b",
    r"\b(I'm not sure|I don't have|I cannot verify)\b",
]


class HallucinationDetector:
    """Classifies LLM outputs as factual, confabulated, or creative.

    Uses heuristic signals including hedging language, citation patterns,
    creative markers, and optional ground-truth comparison to classify outputs.
    """

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold
        self._confab_patterns = [re.compile(p, re.IGNORECASE) for p in _CONFABULATION_MARKERS]
        self._creative_patterns = [re.compile(p, re.IGNORECASE) for p in _CREATIVE_MARKERS]
        self._hedging_patterns = [re.compile(p, re.IGNORECASE) for p in _HEDGING_MARKERS]

    def classify(self, output: LLMOutput) -> tuple[HallucinationType, float]:
        """Classify an LLM output into a hallucination type with confidence."""
        scores: Counter[HallucinationType] = Counter()

        confab_hits = sum(1 for p in self._confab_patterns if p.search(output.text))
        creative_hits = sum(1 for p in self._creative_patterns if p.search(output.text))
        hedging_hits = sum(1 for p in self._hedging_patterns if p.search(output.text))

        scores[HallucinationType.CONFABULATED] = confab_hits * 2
        scores[HallucinationType.CREATIVE] = creative_hits * 2
        scores[HallucinationType.FACTUAL] = hedging_hits  # hedging suggests awareness

        # Ground truth comparison
        if output.ground_truth is not None:
            overlap = self._text_overlap(output.text, output.ground_truth)
            if overlap > 0.5:
                scores[HallucinationType.FACTUAL] += 5
            else:
                scores[HallucinationType.CONFABULATED] += 3

        total = sum(scores.values()) or 1
        best_type = max(scores, key=lambda k: scores[k]) if total > 0 else HallucinationType.FACTUAL
        confidence = min(scores[best_type] / total, 1.0) if total > 0 else 0.5

        # Default to factual with low confidence if no signals
        if total == 0:
            return HallucinationType.FACTUAL, 0.5

        return best_type, round(confidence, 3)

    def batch_classify(
        self, outputs: list[LLMOutput]
    ) -> list[tuple[HallucinationType, float]]:
        """Classify multiple outputs."""
        return [self.classify(o) for o in outputs]

    @staticmethod
    def _text_overlap(text_a: str, text_b: str) -> float:
        """Compute simple word-overlap ratio between two texts."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / max(len(words_a), len(words_b))
