"""
Pydantic models for the generation module.

EvaluationResult and EvaluationBatch were removed when the LLM-as-judge
evaluator was dropped from the pipeline. They had no callers.

Remaining models:
    LengthRange / LengthDistribution  — ScenarioConfig length specs
    GenerationCell                     — one (language × industry × scenario) task
    GeneratedPrompt / GenerationBatch  — structured output from the LLM
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey

FRACTION_TOLERANCE = 0.005


class LengthRange(BaseModel):
    """One word-count bucket with a sampling fraction and concrete examples."""

    min_words: Annotated[int, Field(ge=1)]
    max_words: Annotated[int, Field(ge=1)]
    fraction: Annotated[float, Field(gt=0.0, le=1.0)]
    examples: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def max_exceeds_min(self) -> LengthRange:
        if self.max_words <= self.min_words:
            raise ValueError(
                f"max_words ({self.max_words}) must exceed min_words ({self.min_words})"
            )
        return self

    def to_prompt_line(self) -> str:
        pct = round(self.fraction * 100)
        line = f"  {pct}%: {self.min_words}–{self.max_words} words"
        if self.examples:
            line += f'  (e.g. "{self.examples[0]}")'
        return line


class LengthDistribution(BaseModel):
    """Word-count distribution for a scenario. All range fractions must sum to 1.0."""

    ranges: list[LengthRange]

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> LengthDistribution:
        total = sum(r.fraction for r in self.ranges)
        if abs(total - 1.0) > FRACTION_TOLERANCE:
            raise ValueError(f"LengthDistribution fractions sum to {total:.4f}, expected 1.0")
        return self

    def to_prompt_str(self) -> str:
        return "\n".join(r.to_prompt_line() for r in self.ranges)


class GenerationCell(BaseModel):
    """One (language, industry, scenario) triple with a target row count.

    target_count=0 is valid for UI display purposes (all_cells_including_zero).
    Active generation cells always have target_count >= 1.
    """

    language: LanguageKey
    industry: IndustryKey
    scenario: ScenarioKey
    target_count: Annotated[int, Field(ge=0)]

    @property
    def cell_id(self) -> str:
        return f"{self.language}__{self.industry}__{self.scenario}"

    @property
    def is_active(self) -> bool:
        return self.target_count >= 1


class GeneratedPrompt(BaseModel):
    """A single generated training example validated on receipt from the LLM."""

    text: str
    word_count: int = Field(ge=1)

    @model_validator(mode="after")
    def text_not_empty(self) -> GeneratedPrompt:
        self.text = self.text.strip()
        if len(self.text) < 3:
            raise ValueError(f"Generated text too short: {self.text!r}")
        return self


class GenerationBatch(BaseModel):
    """Structured output from one LLM generation API call."""

    prompts: list[GeneratedPrompt]
