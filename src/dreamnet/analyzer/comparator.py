"""Comparator finding parallels between human dreams and AI hallucinations."""

from __future__ import annotations

from dreamnet.analyzer.dream_theory import DreamTheoryMapper
from dreamnet.analyzer.hallucination import HallucinationDetector
from dreamnet.models import AnalysisResult, DreamParallel, LLMOutput


# Structural parallels between dream features and hallucination features
_PARALLEL_TEMPLATES = {
    "narrative_coherence": (
        "Both dreams and this hallucination exhibit loose narrative coherence -- "
        "plausible on the surface but with logical gaps upon scrutiny."
    ),
    "emotional_salience": (
        "The output is emotionally charged, mirroring how dreams prioritize "
        "emotional significance over factual accuracy."
    ),
    "source_amnesia": (
        "The model presents information without traceable sources, paralleling "
        "source amnesia in dreams where experiences feel real but lack origin."
    ),
    "recombination": (
        "Elements from disparate contexts are blended together, similar to how "
        "dreams combine fragments of different memories."
    ),
    "confabulation": (
        "Gaps in knowledge are filled with plausible but fabricated details, "
        "akin to how dreamers confabulate explanations for bizarre dream events."
    ),
}


class DreamHallucinationComparator:
    """Finds parallels between human dreams and AI hallucinations.

    Combines hallucination detection and dream theory mapping to produce
    a comprehensive analysis of how AI outputs mirror dreaming.
    """

    def __init__(self) -> None:
        self.detector = HallucinationDetector()
        self.mapper = DreamTheoryMapper()

    def analyze(self, output: LLMOutput) -> AnalysisResult:
        """Perform full analysis on a single LLM output."""
        h_type, confidence = self.detector.classify(output)
        dream_fn = self.mapper.map(h_type, output)
        parallels = self._find_parallels(output, h_type)

        return AnalysisResult(
            output=output,
            classification=h_type,
            classification_confidence=confidence,
            dream_mapping=dream_fn,
            parallels=parallels,
            reasoning=self.mapper.explain(dream_fn),
        )

    def analyze_batch(self, outputs: list[LLMOutput]) -> list[AnalysisResult]:
        """Analyze a batch of LLM outputs."""
        return [self.analyze(o) for o in outputs]

    def _find_parallels(
        self, output: LLMOutput, h_type: str
    ) -> list[DreamParallel]:
        """Identify dream-hallucination parallels in the output."""
        parallels: list[DreamParallel] = []
        text_lower = output.text.lower()
        dream_fn = self.mapper.map(h_type, output)

        # Source amnesia: claims without citations
        if any(
            phrase in text_lower
            for phrase in ["studies show", "research indicates", "it is known"]
        ):
            parallels.append(
                DreamParallel(
                    hallucination_type=h_type,
                    dream_function=dream_fn,
                    description=_PARALLEL_TEMPLATES["source_amnesia"],
                    confidence=0.8,
                    evidence=["Unsourced claims detected"],
                    llm_output=output,
                )
            )

        # Recombination: mixing of contexts
        if any(
            phrase in text_lower
            for phrase in ["similar to", "like the", "reminds me of", "combines"]
        ):
            parallels.append(
                DreamParallel(
                    hallucination_type=h_type,
                    dream_function=dream_fn,
                    description=_PARALLEL_TEMPLATES["recombination"],
                    confidence=0.7,
                    evidence=["Cross-context blending detected"],
                    llm_output=output,
                )
            )

        # Confabulation: gap-filling
        if h_type == "confabulated":
            parallels.append(
                DreamParallel(
                    hallucination_type=h_type,
                    dream_function=dream_fn,
                    description=_PARALLEL_TEMPLATES["confabulation"],
                    confidence=0.75,
                    evidence=["Output classified as confabulated"],
                    llm_output=output,
                )
            )

        # Emotional salience
        emotion_words = ["amazing", "terrible", "love", "hate", "fear", "joy", "anger"]
        if any(w in text_lower for w in emotion_words):
            parallels.append(
                DreamParallel(
                    hallucination_type=h_type,
                    dream_function=dream_fn,
                    description=_PARALLEL_TEMPLATES["emotional_salience"],
                    confidence=0.65,
                    evidence=["Emotionally charged language detected"],
                    llm_output=output,
                )
            )

        # Default: narrative coherence parallel
        if not parallels:
            parallels.append(
                DreamParallel(
                    hallucination_type=h_type,
                    dream_function=dream_fn,
                    description=_PARALLEL_TEMPLATES["narrative_coherence"],
                    confidence=0.5,
                    evidence=["General structural parallel"],
                    llm_output=output,
                )
            )

        return parallels
