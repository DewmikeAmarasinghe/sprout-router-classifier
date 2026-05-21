"""
Generation distribution — three-level fraction hierarchy.

DISTRIBUTION v3 — 30% label=0 (gpt-4o-mini), 70% label=1 (gpt-4o):
  pure_english raised to 47% (was 43%).
  singlish_light: 23%, singlish_heavy: 12%, tanglish_light: 11%, tanglish_heavy: 7%.
  Within pure_english, simple_transactional and named_location raised further
  to achieve label=0 ≈ 64% within that bucket.

  Verified: 0.47 × 0.639 = 30.0% label=0 overall.
  Language fractions: 0.47+0.23+0.12+0.11+0.07 = 1.00.
  All scenario fractions within every industry verified to sum to 1.00.

FRACTION RULES (all validated by Pydantic at import time):
  Scenario fractions within each IndustryBucket must sum to 1.0.
  Industry fractions within each LanguageBucket must sum to 1.0.
  Language fractions must sum to 1.0.
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
        for lang in self.languages:
            lang.computed_count = round(self.global_total * lang.fraction)
            for ind in lang.industries:
                ind.computed_count = round(lang.computed_count * ind.fraction)
                for sc in ind.scenarios:
                    sc.computed_count = round(ind.computed_count * sc.fraction)
        return self

    def to_cells(self) -> list[GenerationCell]:
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
        """All cells including zero-count ones for UI display.

        Uses model_construct to bypass target_count ge=1 validation.
        Zero-count cells are display-only — never submitted to run_cells().
        """
        return [
            GenerationCell.model_construct(
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

    def planned_label_split(self) -> tuple[int, int]:
        """Return (label_0_count, label_1_count) from distribution fractions.

        Used by planned EDA (phase_4_eda.py --planned).
        """
        from backend.config.keys import LABEL_0_SCENARIOS

        label_0 = label_1 = 0
        for cell in self.to_cells():
            if cell.language == LanguageKey.PURE_ENGLISH and cell.scenario in LABEL_0_SCENARIOS:
                label_0 += cell.target_count
            else:
                label_1 += cell.target_count
        return label_0, label_1


def sc(scenario: ScenarioKey, fraction: float) -> ScenarioBucket:
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
    """Create an IndustryBucket. All 9 scenario keyword args required."""
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
        # ── PURE ENGLISH (47% = ~28,200) ──────────────────────────────────────
        # Raised to 47% to achieve 30% label=0 overall.
        # label=0 within pure_english ≈ 63.9%  (simple + named_loc per industry).
        # 0.47 × 0.639 = 30.0% label=0 overall.
        # Banking/insurance retain higher complex/sensitive due to their query nature.
        LanguageBucket(
            language=LK.PURE_ENGLISH,
            fraction=0.47,
            industries=[
                # ecommerce: simple=0.56 named=0.12 → label=0=0.68 | sum=1.00
                industry(
                    IK.ECOMMERCE,
                    0.12,
                    simple=0.56,
                    named_loc=0.12,
                    proximity=0.04,
                    relative=0.02,
                    complex_task=0.04,
                    sensitive=0.04,
                    escalation=0.08,
                    response_lang=0.00,
                    continuation=0.10,
                ),
                # healthcare: simple=0.52 named=0.13 → label=0=0.65 | sum=1.00
                industry(
                    IK.HEALTHCARE,
                    0.12,
                    simple=0.52,
                    named_loc=0.13,
                    proximity=0.05,
                    relative=0.02,
                    complex_task=0.09,
                    sensitive=0.09,
                    escalation=0.05,
                    response_lang=0.00,
                    continuation=0.05,
                ),
                # banking: simple=0.43 named=0.10 → label=0=0.53 | sum=1.00
                industry(
                    IK.BANKING,
                    0.20,
                    simple=0.43,
                    named_loc=0.10,
                    proximity=0.03,
                    relative=0.01,
                    complex_task=0.15,
                    sensitive=0.14,
                    escalation=0.08,
                    response_lang=0.00,
                    continuation=0.06,
                ),
                # insurance: simple=0.46 named=0.10 → label=0=0.56 | sum=1.00
                industry(
                    IK.INSURANCE,
                    0.12,
                    simple=0.46,
                    named_loc=0.10,
                    proximity=0.02,
                    relative=0.01,
                    complex_task=0.22,
                    sensitive=0.07,
                    escalation=0.07,
                    response_lang=0.00,
                    continuation=0.05,
                ),
                # telecom: simple=0.46 named=0.09 → label=0=0.55 | sum=1.00
                industry(
                    IK.TELECOM,
                    0.13,
                    simple=0.46,
                    named_loc=0.09,
                    proximity=0.03,
                    relative=0.01,
                    complex_task=0.12,
                    sensitive=0.03,
                    escalation=0.13,
                    response_lang=0.00,
                    continuation=0.13,
                ),
                # logistics: simple=0.54 named=0.13 → label=0=0.67 | sum=1.00
                industry(
                    IK.LOGISTICS,
                    0.06,
                    simple=0.54,
                    named_loc=0.13,
                    proximity=0.06,
                    relative=0.03,
                    complex_task=0.03,
                    sensitive=0.02,
                    escalation=0.07,
                    response_lang=0.00,
                    continuation=0.12,
                ),
                # hospitality: simple=0.60 named=0.17 → label=0=0.77 | sum=1.00
                industry(
                    IK.HOSPITALITY,
                    0.15,
                    simple=0.60,
                    named_loc=0.17,
                    proximity=0.06,
                    relative=0.07,
                    complex_task=0.03,
                    sensitive=0.02,
                    escalation=0.03,
                    response_lang=0.00,
                    continuation=0.02,
                ),
                # education: simple=0.62 named=0.17 → label=0=0.79 | sum=1.00
                industry(
                    IK.EDUCATION,
                    0.10,
                    simple=0.62,
                    named_loc=0.17,
                    proximity=0.04,
                    relative=0.02,
                    complex_task=0.09,
                    sensitive=0.01,
                    escalation=0.03,
                    response_lang=0.00,
                    continuation=0.02,
                ),
            ],
        ),
        # ── SINGLISH LIGHT (23% = ~13,800, all label=1) ───────────────────────
        LanguageBucket(
            language=LK.SINGLISH_LIGHT,
            fraction=0.23,
            industries=[
                industry(
                    IK.ECOMMERCE,
                    0.22,
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
                    0.17,
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
                    0.17,
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
                    0.04,
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
                    0.12,
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
                    0.10,
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
                    0.07,
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
        # ── SINGLISH HEAVY (12% = ~7,200, all label=1) ────────────────────────
        LanguageBucket(
            language=LK.SINGLISH_HEAVY,
            fraction=0.12,
            industries=[
                industry(
                    IK.ECOMMERCE,
                    0.25,
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
                    0.18,
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
                    0.17,
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
                    0.04,
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
                    0.08,
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
                    0.05,
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
        # ── TANGLISH LIGHT (11% = ~6,600, all label=1) ────────────────────────
        LanguageBucket(
            language=LK.TANGLISH_LIGHT,
            fraction=0.11,
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
                    0.18,
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
                    0.09,
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
                    0.05,
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
                    0.09,
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
        # ── TANGLISH HEAVY (7% = ~4,200, all label=1) ─────────────────────────
        LanguageBucket(
            language=LK.TANGLISH_HEAVY,
            fraction=0.07,
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
                    0.20,
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
                    0.09,
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
                    0.05,
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
