"""
Generation distribution — three-level fraction hierarchy.

All 9 scenarios defined for all 8 industries in all 5 languages.
fraction=0.0 means 0 rows generated but the cell is visible and editable in the UI.

FRACTION RULES:
  All scenario fractions within one IndustryBucket must sum to 1.0.
  All industry fractions within one LanguageBucket must sum to 1.0.
  All language fractions must sum to 1.0.
  Pydantic validates all these at import time — misconfiguration = instant error.

ADDING A NEW SCENARIO / INDUSTRY / LANGUAGE:
  1. Add the key in config/keys.py
  2. Add the config in the relevant config file
  3. Add bucket entries here
  → Appears automatically in the Gradio UI
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.generation.pymodels import GenerationCell

TOLERANCE = 0.005
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class ScenarioBucket(BaseModel):
    scenario: ScenarioKey
    fraction: Fraction
    computed_count: int = Field(default=0, exclude=True)


class IndustryBucket(BaseModel):
    industry: IndustryKey
    fraction: Fraction
    scenarios: list[ScenarioBucket]
    computed_count: int = Field(default=0, exclude=True)

    @model_validator(mode="after")
    def scenario_fractions_sum_to_one(self) -> IndustryBucket:
        total = sum(s.fraction for s in self.scenarios)
        if abs(total - 1.0) > TOLERANCE:
            raise ValueError(
                f"[industry={self.industry}] scenario fractions sum to {total:.4f}, "
                f"expected 1.0  (diff={abs(total - 1.0):.5f})"
            )
        return self

    def get_scenario(self, key: ScenarioKey) -> ScenarioBucket | None:
        return next((s for s in self.scenarios if s.scenario == key), None)


class LanguageBucket(BaseModel):
    language: LanguageKey
    fraction: Fraction
    industries: list[IndustryBucket]
    computed_count: int = Field(default=0, exclude=True)

    @model_validator(mode="after")
    def industry_fractions_sum_to_one(self) -> LanguageBucket:
        total = sum(i.fraction for i in self.industries)
        if abs(total - 1.0) > TOLERANCE:
            raise ValueError(
                f"[language={self.language}] industry fractions sum to {total:.4f}, "
                f"expected 1.0  (diff={abs(total - 1.0):.5f})"
            )
        return self

    def get_industry(self, key: IndustryKey) -> IndustryBucket | None:
        return next((i for i in self.industries if i.industry == key), None)


class GenerationDistribution(BaseModel):
    global_total: Annotated[int, Field(gt=0)]
    languages: list[LanguageBucket]

    @model_validator(mode="after")
    def language_fractions_sum_to_one(self) -> GenerationDistribution:
        total = sum(lang.fraction for lang in self.languages)
        if abs(total - 1.0) > TOLERANCE:
            raise ValueError(f"Language fractions sum to {total:.4f}, expected 1.0")
        return self

    def resolve(self) -> GenerationDistribution:
        """Compute all computed_count values. Called once at import time."""
        for lang in self.languages:
            lang.computed_count = round(self.global_total * lang.fraction)
            for ind in lang.industries:
                ind.computed_count = round(lang.computed_count * ind.fraction)
                for sc in ind.scenarios:
                    sc.computed_count = round(ind.computed_count * sc.fraction)
        return self

    def to_cells(self) -> list[GenerationCell]:
        """Active generation tasks only (computed_count > 0)."""
        return [
            GenerationCell(
                language=lang.language,
                industry=ind.industry,
                scenario=sc.scenario,
                target_count=sc.computed_count,
            )
            for lang in self.languages
            for ind in lang.industries
            for sc in ind.scenarios
            if sc.computed_count >= 1
        ]

    def all_cells_including_zero(self) -> list[GenerationCell]:
        """All cells including zero-count ones — for UI display."""
        return [
            GenerationCell(
                language=lang.language,
                industry=ind.industry,
                scenario=sc.scenario,
                target_count=sc.computed_count,
            )
            for lang in self.languages
            for ind in lang.industries
            for sc in ind.scenarios
        ]

    def summary(self) -> dict[str, int]:
        return {lang.language: lang.computed_count for lang in self.languages}

    def find_cell(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
    ) -> GenerationCell | None:
        lang_b = next((lb for lb in self.languages if lb.language == language), None)
        if not lang_b:
            return None
        ind_b = lang_b.get_industry(industry)
        if not ind_b:
            return None
        sc_b = ind_b.get_scenario(scenario)
        if not sc_b:
            return None
        return GenerationCell(
            language=language,
            industry=industry,
            scenario=scenario,
            target_count=sc_b.computed_count,
        )

    def get_language(self, key: LanguageKey) -> LanguageBucket | None:
        return next((lb for lb in self.languages if lb.language == key), None)


# ── Shorthand constructors ────────────────────────────────────────────────────


def sc(scenario: ScenarioKey, fraction: float) -> ScenarioBucket:
    """Create a ScenarioBucket. IDE shows all valid ScenarioKey members."""
    return ScenarioBucket(scenario=scenario, fraction=fraction)


def industry(
    key: IndustryKey,
    fraction: float,
    *,
    simple: float,
    named_loc: float,
    proximity: float,
    relative: float,
    complex_task: float,
    sensitive: float,
    escalation: float,
    response_lang: float,
    continuation: float,
) -> IndustryBucket:
    """Create an IndustryBucket with all 9 scenarios explicitly named.

    All 9 keyword arguments required — prevents accidental omissions.
    Set non-applicable ones to 0.0.
    Raises ValidationError immediately if fractions don't sum to 1.0.
    """
    return IndustryBucket(
        industry=key,
        fraction=fraction,
        scenarios=[
            sc(ScenarioKey.SIMPLE_TRANSACTIONAL, simple),
            sc(ScenarioKey.NAMED_LOCATION, named_loc),
            sc(ScenarioKey.LOCATION_PROXIMITY, proximity),
            sc(ScenarioKey.LOCATION_RELATIVE, relative),
            sc(ScenarioKey.COMPLEX_TASK, complex_task),
            sc(ScenarioKey.SENSITIVE_CONTEXT, sensitive),
            sc(ScenarioKey.ESCALATION, escalation),
            sc(ScenarioKey.RESPONSE_LANGUAGE, response_lang),
            sc(ScenarioKey.CONTINUATION, continuation),
        ],
    )


SK = ScenarioKey
IK = IndustryKey
LK = LanguageKey

DISTRIBUTION: GenerationDistribution = GenerationDistribution(
    global_total=60_000,
    languages=[
        # ── PURE ENGLISH (28% = ~16,800) ─────────────────────────────────────
        LanguageBucket(
            language=LK.PURE_ENGLISH,
            fraction=0.28,
            industries=[
                industry(
                    IK.ECOMMERCE,
                    0.14,
                    simple=0.45,
                    named_loc=0.08,
                    proximity=0.05,
                    relative=0.03,
                    complex_task=0.06,
                    sensitive=0.05,
                    escalation=0.12,
                    response_lang=0.00,
                    continuation=0.16,
                ),
                industry(
                    IK.HEALTHCARE,
                    0.13,
                    simple=0.40,
                    named_loc=0.10,
                    proximity=0.08,
                    relative=0.03,
                    complex_task=0.12,
                    sensitive=0.14,
                    escalation=0.07,
                    response_lang=0.00,
                    continuation=0.06,
                ),
                industry(
                    IK.BANKING,
                    0.20,
                    simple=0.28,
                    named_loc=0.07,
                    proximity=0.04,
                    relative=0.02,
                    complex_task=0.22,
                    sensitive=0.20,
                    escalation=0.09,
                    response_lang=0.00,
                    continuation=0.08,
                ),
                industry(
                    IK.INSURANCE,
                    0.13,
                    simple=0.30,
                    named_loc=0.06,
                    proximity=0.02,
                    relative=0.01,
                    complex_task=0.30,
                    sensitive=0.10,
                    escalation=0.10,
                    response_lang=0.00,
                    continuation=0.11,
                ),
                industry(
                    IK.TELECOM,
                    0.13,
                    simple=0.35,
                    named_loc=0.06,
                    proximity=0.03,
                    relative=0.01,
                    complex_task=0.16,
                    sensitive=0.03,
                    escalation=0.17,
                    response_lang=0.00,
                    continuation=0.19,
                ),
                industry(
                    IK.LOGISTICS,
                    0.10,
                    simple=0.40,
                    named_loc=0.10,
                    proximity=0.10,
                    relative=0.05,
                    complex_task=0.04,
                    sensitive=0.03,
                    escalation=0.10,
                    response_lang=0.00,
                    continuation=0.18,
                ),
                industry(
                    IK.HOSPITALITY,
                    0.09,
                    simple=0.48,
                    named_loc=0.14,
                    proximity=0.10,
                    relative=0.10,
                    complex_task=0.05,
                    sensitive=0.02,
                    escalation=0.07,
                    response_lang=0.00,
                    continuation=0.04,
                ),
                industry(
                    IK.EDUCATION,
                    0.08,
                    simple=0.50,
                    named_loc=0.14,
                    proximity=0.06,
                    relative=0.03,
                    complex_task=0.16,
                    sensitive=0.01,
                    escalation=0.05,
                    response_lang=0.00,
                    continuation=0.05,
                ),
            ],
        ),
        # ── SINGLISH LIGHT (21% = ~12,600, all label=1) ──────────────────────
        LanguageBucket(
            language=LK.SINGLISH_LIGHT,
            fraction=0.21,
            industries=[
                industry(
                    IK.ECOMMERCE,
                    0.17,
                    simple=0.33,
                    named_loc=0.08,
                    proximity=0.10,
                    relative=0.05,
                    complex_task=0.05,
                    sensitive=0.05,
                    escalation=0.13,
                    response_lang=0.07,
                    continuation=0.14,
                ),
                industry(
                    IK.HEALTHCARE,
                    0.14,
                    simple=0.28,
                    named_loc=0.08,
                    proximity=0.14,
                    relative=0.04,
                    complex_task=0.10,
                    sensitive=0.14,
                    escalation=0.08,
                    response_lang=0.08,
                    continuation=0.06,
                ),
                industry(
                    IK.BANKING,
                    0.18,
                    simple=0.24,
                    named_loc=0.07,
                    proximity=0.09,
                    relative=0.04,
                    complex_task=0.18,
                    sensitive=0.14,
                    escalation=0.10,
                    response_lang=0.06,
                    continuation=0.08,
                ),
                industry(
                    IK.INSURANCE,
                    0.11,
                    simple=0.26,
                    named_loc=0.05,
                    proximity=0.03,
                    relative=0.02,
                    complex_task=0.24,
                    sensitive=0.08,
                    escalation=0.12,
                    response_lang=0.10,
                    continuation=0.10,
                ),
                industry(
                    IK.TELECOM,
                    0.13,
                    simple=0.28,
                    named_loc=0.05,
                    proximity=0.05,
                    relative=0.02,
                    complex_task=0.12,
                    sensitive=0.03,
                    escalation=0.20,
                    response_lang=0.08,
                    continuation=0.17,
                ),
                industry(
                    IK.LOGISTICS,
                    0.11,
                    simple=0.30,
                    named_loc=0.08,
                    proximity=0.18,
                    relative=0.08,
                    complex_task=0.03,
                    sensitive=0.03,
                    escalation=0.12,
                    response_lang=0.04,
                    continuation=0.14,
                ),
                industry(
                    IK.HOSPITALITY,
                    0.08,
                    simple=0.36,
                    named_loc=0.10,
                    proximity=0.16,
                    relative=0.12,
                    complex_task=0.04,
                    sensitive=0.02,
                    escalation=0.08,
                    response_lang=0.08,
                    continuation=0.04,
                ),
                industry(
                    IK.EDUCATION,
                    0.08,
                    simple=0.35,
                    named_loc=0.08,
                    proximity=0.07,
                    relative=0.03,
                    complex_task=0.18,
                    sensitive=0.01,
                    escalation=0.06,
                    response_lang=0.14,
                    continuation=0.08,
                ),
            ],
        ),
        # ── SINGLISH HEAVY (13% = ~7,800, all label=1) ───────────────────────
        LanguageBucket(
            language=LK.SINGLISH_HEAVY,
            fraction=0.13,
            industries=[
                industry(
                    IK.ECOMMERCE,
                    0.19,
                    simple=0.35,
                    named_loc=0.06,
                    proximity=0.12,
                    relative=0.05,
                    complex_task=0.04,
                    sensitive=0.05,
                    escalation=0.14,
                    response_lang=0.09,
                    continuation=0.10,
                ),
                industry(
                    IK.HEALTHCARE,
                    0.16,
                    simple=0.30,
                    named_loc=0.07,
                    proximity=0.14,
                    relative=0.04,
                    complex_task=0.08,
                    sensitive=0.17,
                    escalation=0.07,
                    response_lang=0.09,
                    continuation=0.04,
                ),
                industry(
                    IK.BANKING,
                    0.18,
                    simple=0.26,
                    named_loc=0.06,
                    proximity=0.10,
                    relative=0.03,
                    complex_task=0.18,
                    sensitive=0.16,
                    escalation=0.10,
                    response_lang=0.06,
                    continuation=0.05,
                ),
                industry(
                    IK.INSURANCE,
                    0.10,
                    simple=0.28,
                    named_loc=0.04,
                    proximity=0.03,
                    relative=0.01,
                    complex_task=0.22,
                    sensitive=0.08,
                    escalation=0.14,
                    response_lang=0.10,
                    continuation=0.10,
                ),
                industry(
                    IK.TELECOM,
                    0.13,
                    simple=0.30,
                    named_loc=0.04,
                    proximity=0.05,
                    relative=0.01,
                    complex_task=0.10,
                    sensitive=0.03,
                    escalation=0.23,
                    response_lang=0.10,
                    continuation=0.14,
                ),
                industry(
                    IK.LOGISTICS,
                    0.10,
                    simple=0.32,
                    named_loc=0.07,
                    proximity=0.20,
                    relative=0.08,
                    complex_task=0.03,
                    sensitive=0.02,
                    escalation=0.12,
                    response_lang=0.04,
                    continuation=0.12,
                ),
                industry(
                    IK.HOSPITALITY,
                    0.07,
                    simple=0.36,
                    named_loc=0.08,
                    proximity=0.18,
                    relative=0.12,
                    complex_task=0.03,
                    sensitive=0.02,
                    escalation=0.08,
                    response_lang=0.10,
                    continuation=0.03,
                ),
                industry(
                    IK.EDUCATION,
                    0.07,
                    simple=0.34,
                    named_loc=0.07,
                    proximity=0.06,
                    relative=0.02,
                    complex_task=0.15,
                    sensitive=0.01,
                    escalation=0.06,
                    response_lang=0.20,
                    continuation=0.09,
                ),
            ],
        ),
        # ── TANGLISH LIGHT (19% = ~11,400, all label=1) ──────────────────────
        LanguageBucket(
            language=LK.TANGLISH_LIGHT,
            fraction=0.19,
            industries=[
                industry(
                    IK.ECOMMERCE,
                    0.16,
                    simple=0.33,
                    named_loc=0.08,
                    proximity=0.11,
                    relative=0.05,
                    complex_task=0.05,
                    sensitive=0.05,
                    escalation=0.13,
                    response_lang=0.07,
                    continuation=0.13,
                ),
                industry(
                    IK.HEALTHCARE,
                    0.15,
                    simple=0.27,
                    named_loc=0.08,
                    proximity=0.13,
                    relative=0.04,
                    complex_task=0.10,
                    sensitive=0.16,
                    escalation=0.08,
                    response_lang=0.08,
                    continuation=0.06,
                ),
                industry(
                    IK.BANKING,
                    0.18,
                    simple=0.24,
                    named_loc=0.07,
                    proximity=0.09,
                    relative=0.03,
                    complex_task=0.18,
                    sensitive=0.16,
                    escalation=0.10,
                    response_lang=0.06,
                    continuation=0.07,
                ),
                industry(
                    IK.INSURANCE,
                    0.11,
                    simple=0.26,
                    named_loc=0.05,
                    proximity=0.02,
                    relative=0.01,
                    complex_task=0.25,
                    sensitive=0.09,
                    escalation=0.12,
                    response_lang=0.10,
                    continuation=0.10,
                ),
                industry(
                    IK.TELECOM,
                    0.14,
                    simple=0.30,
                    named_loc=0.05,
                    proximity=0.04,
                    relative=0.01,
                    complex_task=0.14,
                    sensitive=0.03,
                    escalation=0.20,
                    response_lang=0.09,
                    continuation=0.14,
                ),
                industry(
                    IK.LOGISTICS,
                    0.11,
                    simple=0.31,
                    named_loc=0.08,
                    proximity=0.18,
                    relative=0.08,
                    complex_task=0.03,
                    sensitive=0.03,
                    escalation=0.12,
                    response_lang=0.04,
                    continuation=0.13,
                ),
                industry(
                    IK.HOSPITALITY,
                    0.07,
                    simple=0.36,
                    named_loc=0.10,
                    proximity=0.16,
                    relative=0.12,
                    complex_task=0.04,
                    sensitive=0.02,
                    escalation=0.08,
                    response_lang=0.08,
                    continuation=0.04,
                ),
                industry(
                    IK.EDUCATION,
                    0.08,
                    simple=0.34,
                    named_loc=0.08,
                    proximity=0.06,
                    relative=0.02,
                    complex_task=0.18,
                    sensitive=0.01,
                    escalation=0.06,
                    response_lang=0.16,
                    continuation=0.09,
                ),
            ],
        ),
        # ── TANGLISH HEAVY (19% = ~11,400, all label=1) ──────────────────────
        LanguageBucket(
            language=LK.TANGLISH_HEAVY,
            fraction=0.19,
            industries=[
                industry(
                    IK.ECOMMERCE,
                    0.17,
                    simple=0.35,
                    named_loc=0.06,
                    proximity=0.12,
                    relative=0.05,
                    complex_task=0.04,
                    sensitive=0.05,
                    escalation=0.14,
                    response_lang=0.09,
                    continuation=0.10,
                ),
                industry(
                    IK.HEALTHCARE,
                    0.17,
                    simple=0.28,
                    named_loc=0.07,
                    proximity=0.13,
                    relative=0.04,
                    complex_task=0.08,
                    sensitive=0.20,
                    escalation=0.07,
                    response_lang=0.10,
                    continuation=0.03,
                ),
                industry(
                    IK.BANKING,
                    0.18,
                    simple=0.25,
                    named_loc=0.06,
                    proximity=0.10,
                    relative=0.03,
                    complex_task=0.18,
                    sensitive=0.18,
                    escalation=0.10,
                    response_lang=0.06,
                    continuation=0.04,
                ),
                industry(
                    IK.INSURANCE,
                    0.11,
                    simple=0.27,
                    named_loc=0.04,
                    proximity=0.02,
                    relative=0.01,
                    complex_task=0.26,
                    sensitive=0.09,
                    escalation=0.14,
                    response_lang=0.10,
                    continuation=0.07,
                ),
                industry(
                    IK.TELECOM,
                    0.13,
                    simple=0.31,
                    named_loc=0.04,
                    proximity=0.04,
                    relative=0.01,
                    complex_task=0.12,
                    sensitive=0.03,
                    escalation=0.24,
                    response_lang=0.10,
                    continuation=0.11,
                ),
                industry(
                    IK.LOGISTICS,
                    0.10,
                    simple=0.32,
                    named_loc=0.07,
                    proximity=0.20,
                    relative=0.08,
                    complex_task=0.03,
                    sensitive=0.02,
                    escalation=0.12,
                    response_lang=0.04,
                    continuation=0.12,
                ),
                industry(
                    IK.HOSPITALITY,
                    0.06,
                    simple=0.37,
                    named_loc=0.08,
                    proximity=0.18,
                    relative=0.12,
                    complex_task=0.03,
                    sensitive=0.02,
                    escalation=0.08,
                    response_lang=0.10,
                    continuation=0.02,
                ),
                industry(
                    IK.EDUCATION,
                    0.08,
                    simple=0.34,
                    named_loc=0.07,
                    proximity=0.06,
                    relative=0.02,
                    complex_task=0.16,
                    sensitive=0.01,
                    escalation=0.06,
                    response_lang=0.20,
                    continuation=0.08,
                ),
            ],
        ),
    ],
).resolve()
