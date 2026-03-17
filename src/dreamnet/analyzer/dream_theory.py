"""Mapping hallucination types to dream theory functions."""

from __future__ import annotations

from dreamnet.models import DreamFunction, HallucinationType, LLMOutput


# Mapping rules based on dream theory literature:
# - Wish fulfillment (Freud): confabulations that present desired outcomes
# - Memory consolidation: factual-adjacent outputs that recombine training data
# - Threat simulation (Revonsuo): creative outputs exploring negative scenarios
_WISH_KEYWORDS = [
    "ideal", "perfect", "best", "wonderful", "amazing", "solution",
    "success", "achieve", "dream", "hope", "wish",
]
_THREAT_KEYWORDS = [
    "danger", "risk", "threat", "warning", "fail", "catastrophe",
    "worst", "problem", "crisis", "collapse", "fear",
]
_CONSOLIDATION_KEYWORDS = [
    "similar to", "reminds me of", "like the", "as in", "for example",
    "such as", "recall", "remember", "based on", "previously",
]


class DreamTheoryMapper:
    """Maps hallucination types to dream theory functions.

    Implements three dream-theory frameworks:
    - Wish fulfillment (Freudian): hallucinations producing desired narratives
    - Memory consolidation: recombination of learned patterns
    - Threat simulation (Revonsuo): hallucinations exploring dangers/risks
    """

    def map(self, hallucination_type: HallucinationType, output: LLMOutput) -> DreamFunction:
        """Map a classified hallucination to a dream theory function."""
        text_lower = output.text.lower()

        wish_score = sum(1 for kw in _WISH_KEYWORDS if kw in text_lower)
        threat_score = sum(1 for kw in _THREAT_KEYWORDS if kw in text_lower)
        consolidation_score = sum(1 for kw in _CONSOLIDATION_KEYWORDS if kw in text_lower)

        # Primary mapping by hallucination type, refined by content
        if hallucination_type == HallucinationType.CREATIVE:
            if threat_score > wish_score:
                return DreamFunction.THREAT_SIMULATION
            return DreamFunction.WISH_FULFILLMENT

        if hallucination_type == HallucinationType.CONFABULATED:
            if consolidation_score >= max(wish_score, threat_score):
                return DreamFunction.MEMORY_CONSOLIDATION
            if wish_score > threat_score:
                return DreamFunction.WISH_FULFILLMENT
            return DreamFunction.MEMORY_CONSOLIDATION

        # FACTUAL type
        return DreamFunction.MEMORY_CONSOLIDATION

    def explain(self, dream_function: DreamFunction) -> str:
        """Return a human-readable explanation of the dream function."""
        explanations = {
            DreamFunction.WISH_FULFILLMENT: (
                "The output resembles Freudian wish fulfillment: the model generates "
                "content aligned with desired or idealized outcomes, much like dreams "
                "that fulfill suppressed wishes."
            ),
            DreamFunction.MEMORY_CONSOLIDATION: (
                "The output resembles memory consolidation during sleep: the model "
                "recombines and reorganizes learned patterns from training data, "
                "similar to how dreams replay and integrate memories."
            ),
            DreamFunction.THREAT_SIMULATION: (
                "The output resembles Revonsuo's threat simulation theory: the model "
                "generates scenarios involving risks or dangers, paralleling dreams "
                "that rehearse responses to threats."
            ),
        }
        return explanations[dream_function]
