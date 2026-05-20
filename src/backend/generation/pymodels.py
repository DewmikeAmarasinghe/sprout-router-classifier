"""
Pydantic models for the data generation pipeline.

WHY word_count is NOT in the LLM response schema:
    The LLM occasionally returned word_count=0, failing Pydantic validation
    (ge=1) and wasting 3 retries. word_count is computed locally as
    len(text.split()). Faster, accurate, and removes a class of failures.

WHY GenerationCell uses enum types (not str):
    Functions like resolve_label(), ExampleStore.get(), PromptFactory.build()
    all expect LanguageKey / IndustryKey / ScenarioKey. Using the enum types
    here means cell.language etc. are already the correct type — no casting,
    no ty errors. Pydantic v2 coerces plain strings automatically, so building
    cells from CSV rows still works.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey

# ── LLM response models ───────────────────────────────────────────────────────


class GeneratedPrompt(BaseModel):
    """A single generated customer message returned by the LLM.

    Only the text field is in the response schema.
    word_count is a computed property — not asked from the LLM.
    """

    text: str = Field(min_length=1)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class GenerationBatch(BaseModel):
    """Batch of prompts returned by the LLM in a single call."""

    prompts: list[GeneratedPrompt]


# ── Generation task model ─────────────────────────────────────────────────────


class GenerationCell(BaseModel):
    """A single (language × industry × scenario) generation task.

    Uses enum types so cell.language / cell.industry / cell.scenario pass
    directly to resolve_label(), ExampleStore.get(), PromptFactory.build()
    without casting. Pydantic v2 coerces plain strings automatically.
    """

    language: LanguageKey
    industry: IndustryKey
    scenario: ScenarioKey
    target_count: int = Field(ge=1)

    @property
    def cell_id(self) -> str:
        return f"{self.language}__{self.industry}__{self.scenario}"


# ── Word count shape models ───────────────────────────────────────────────────
# Used by scenario_configs.py to define the expected message length distribution.
# Used by example_store.py and prompt_factory.py to build length guidance in prompts.


class LengthRange(BaseModel):
    """One word count bucket in a scenario's length distribution.

    Attributes:
        min_words: Inclusive lower bound.
        max_words: Inclusive upper bound.
        fraction:  Target proportion of generated messages in this range (0.0–1.0).
                   Fractions across all ranges in a LengthDistribution should sum to ~1.0.
        examples:  Representative example messages for this bucket.
                   Used in prompts to show the LLM what the right length looks like.
    """

    min_words: int
    max_words: int
    fraction: float = Field(ge=0.0, le=1.0)
    examples: list[str] = Field(default_factory=list)


class LengthDistribution(BaseModel):
    """Word count distribution target for a scenario type.

    Each scenario has a characteristic message length profile — e.g. simple
    transactional messages are mostly short (1-8 words), complex tasks tend
    toward longer messages (20-80 words).

    Attributes:
        ranges: Ordered list of LengthRange buckets (short → long).
    """

    ranges: list[LengthRange]

    def to_prompt_str(self) -> str:
        """Compact human-readable description for injection into generation prompts.

        Example output:
            '~45% (1-8 words), ~35% (9-20 words), ~20% (21-40 words)'
        """
        parts = [
            f"~{int(round(r.fraction * 100))}% ({r.min_words}–{r.max_words} words)"
            for r in self.ranges
        ]
        return ", ".join(parts)
