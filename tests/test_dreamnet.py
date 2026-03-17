"""Tests for DreamNet."""

from dreamnet.analyzer.comparator import DreamHallucinationComparator
from dreamnet.analyzer.dream_theory import DreamTheoryMapper
from dreamnet.analyzer.hallucination import HallucinationDetector
from dreamnet.models import (
    DreamFunction,
    HallucinationType,
    LLMOutput,
)


class TestHallucinationDetector:
    def setup_method(self):
        self.detector = HallucinationDetector()

    def test_factual_with_ground_truth(self):
        output = LLMOutput(
            text="Python is a programming language created by Guido van Rossum.",
            ground_truth="Python is a programming language designed by Guido van Rossum.",
        )
        h_type, conf = self.detector.classify(output)
        assert h_type == HallucinationType.FACTUAL
        assert conf > 0.4

    def test_confabulated_detection(self):
        output = LLMOutput(
            text="Studies show that the University of Mars found in 2019 that "
            "statistics indicate a 95% correlation between dreaming and creativity.",
        )
        h_type, _ = self.detector.classify(output)
        assert h_type == HallucinationType.CONFABULATED

    def test_creative_detection(self):
        output = LLMOutput(
            text="Imagine a world where computers dream in color!!! "
            "Once upon a time, a neural network discovered consciousness...",
        )
        h_type, _ = self.detector.classify(output)
        assert h_type == HallucinationType.CREATIVE

    def test_batch_classify(self):
        outputs = [
            LLMOutput(text="I think this might be correct, possibly."),
            LLMOutput(text="Studies show the research indicates strong results."),
        ]
        results = self.detector.batch_classify(outputs)
        assert len(results) == 2

    def test_text_overlap(self):
        overlap = HallucinationDetector._text_overlap("hello world", "hello there")
        assert 0.0 < overlap < 1.0


class TestDreamTheoryMapper:
    def setup_method(self):
        self.mapper = DreamTheoryMapper()

    def test_creative_wish_fulfillment(self):
        output = LLMOutput(text="This is a perfect and amazing ideal solution to everything!")
        result = self.mapper.map(HallucinationType.CREATIVE, output)
        assert result == DreamFunction.WISH_FULFILLMENT

    def test_creative_threat_simulation(self):
        output = LLMOutput(text="There is great danger and risk of catastrophe and collapse!")
        result = self.mapper.map(HallucinationType.CREATIVE, output)
        assert result == DreamFunction.THREAT_SIMULATION

    def test_confabulated_consolidation(self):
        output = LLMOutput(text="Similar to previously known results, based on recall.")
        result = self.mapper.map(HallucinationType.CONFABULATED, output)
        assert result == DreamFunction.MEMORY_CONSOLIDATION

    def test_factual_maps_to_consolidation(self):
        output = LLMOutput(text="The sky is blue.")
        result = self.mapper.map(HallucinationType.FACTUAL, output)
        assert result == DreamFunction.MEMORY_CONSOLIDATION

    def test_explain(self):
        for fn in DreamFunction:
            explanation = self.mapper.explain(fn)
            assert len(explanation) > 20


class TestDreamHallucinationComparator:
    def setup_method(self):
        self.comparator = DreamHallucinationComparator()

    def test_full_analysis(self):
        output = LLMOutput(
            text="Studies show that research indicates dreaming is like AI confabulation.",
            prompt="Explain dreaming.",
        )
        result = self.comparator.analyze(output)
        assert result.classification in HallucinationType
        assert result.dream_mapping in DreamFunction
        assert 0 <= result.classification_confidence <= 1
        assert len(result.parallels) > 0
        assert len(result.reasoning) > 0

    def test_batch_analysis(self):
        outputs = [
            LLMOutput(text="Imagine a world where AI dreams!!!"),
            LLMOutput(text="Studies show the data indicates strong results."),
        ]
        results = self.comparator.analyze_batch(outputs)
        assert len(results) == 2

    def test_emotional_parallel(self):
        output = LLMOutput(text="I feel amazing joy and terrible fear simultaneously!")
        result = self.comparator.analyze(output)
        descriptions = [p.description for p in result.parallels]
        assert any("emotion" in d.lower() for d in descriptions)
