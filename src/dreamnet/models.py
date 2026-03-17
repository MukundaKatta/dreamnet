"""Data models for DreamNet."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class HallucinationType(str, Enum):
    """Classification of LLM hallucination types."""

    FACTUAL = "factual"
    CONFABULATED = "confabulated"
    CREATIVE = "creative"


class DreamFunction(str, Enum):
    """Dream theory functions mapped to hallucination types."""

    WISH_FULFILLMENT = "wish_fulfillment"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    THREAT_SIMULATION = "threat_simulation"


class LLMOutput(BaseModel):
    """Represents a single LLM output to be analyzed."""

    text: str
    prompt: str = ""
    source_model: str = "unknown"
    ground_truth: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class DreamParallel(BaseModel):
    """A parallel found between an AI hallucination and human dream theory."""

    hallucination_type: HallucinationType
    dream_function: DreamFunction
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    llm_output: Optional[LLMOutput] = None


class AnalysisResult(BaseModel):
    """Full analysis result for a single LLM output."""

    output: LLMOutput
    classification: HallucinationType
    classification_confidence: float = Field(ge=0.0, le=1.0)
    dream_mapping: Optional[DreamFunction] = None
    parallels: list[DreamParallel] = Field(default_factory=list)
    reasoning: str = ""
