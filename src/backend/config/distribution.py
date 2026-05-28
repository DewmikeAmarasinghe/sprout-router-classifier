"""
Generation distribution — three-level fraction hierarchy.

DISTRIBUTION v4 — 50% label=0 (gpt-4o-mini), 50% label=1 (gpt-4o):

  TWO SCENARIOS ONLY:
    simple_transactional — ALL routine queries for the industry, including simple
                           named-place lookups ("Is Kandy branch open Sunday?").
                           The boundary: NO spatial proximity reasoning needed.
    location_proximity   — User needs nearest/closest place. Spatial distance
                           reasoning about Sri Lankan geography required.
                           A place name alone does NOT trigger this — only
                           "nearest / closest to me / which branch near X?" does.

  ROUTING LOGIC:
    label=0: pure_english + simple_transactional
    label=1: location_proximity (ALL languages — spatial reasoning always needed)
             OR any code-mixed language + simple_transactional

  MATH (verified):
    pure_english = 60%,  non-english = 40%
    Within pure_english: simple = 83.33%, named_loc = 16.67%
    → label=0 = 0.60 × 0.8333 = 50.0%
    → label=1 = 0.60 × 0.1667 + 0.40 × 1.0 = 10.0% + 40.0% = 50.0%

  Language fractions: 0.60+0.20+0.10+0.07+0.03 = 1.00
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
    proximity: float,
) -> IndustryBucket:
    """Create an IndustryBucket. simple + proximity must sum to 1.0."""
    return IndustryBucket(
        industry=key,
        fraction=fraction,
        scenarios=[
            sc(ScenarioKey.SIMPLE_TRANSACTIONAL, simple),
            sc(ScenarioKey.LOCATION_PROXIMITY, proximity),
        ],
    )


SK = ScenarioKey
IK = IndustryKey
LK = LanguageKey

DISTRIBUTION: GenerationDistribution = GenerationDistribution(
    global_total=60_000,
    languages=[
        # ── PURE ENGLISH (60% = 36,000) ────────────────────────────────────────
        # simple_transactional = 83.33% → label=0 = 0.60 × 0.8333 = 50.0%
        # named_location        = 16.67% → label=1 (spatial reasoning required)
        LanguageBucket(
            language=LK.PURE_ENGLISH,
            fraction=0.60,
            industries=[
                industry(IK.ECOMMERCE, 0.12, simple=0.84, proximity=0.16),
                industry(IK.HEALTHCARE, 0.12, simple=0.82, proximity=0.18),
                industry(IK.BANKING, 0.20, simple=0.85, proximity=0.15),
                industry(IK.INSURANCE, 0.12, simple=0.85, proximity=0.15),
                industry(IK.TELECOM, 0.13, simple=0.83, proximity=0.17),
                industry(IK.LOGISTICS, 0.06, simple=0.82, proximity=0.18),
                industry(IK.HOSPITALITY, 0.15, simple=0.82, proximity=0.18),
                industry(IK.EDUCATION, 0.10, simple=0.83, proximity=0.17),
            ],
        ),
        # ── SINGLISH LIGHT (20% = 12,000, all label=1) ────────────────────────
        # simple_transactional still label=1 for code-mixed; location_proximity always 1
        LanguageBucket(
            language=LK.SINGLISH_LIGHT,
            fraction=0.20,
            industries=[
                industry(IK.ECOMMERCE, 0.22, simple=0.60, proximity=0.40),
                industry(IK.HEALTHCARE, 0.17, simple=0.58, proximity=0.42),
                industry(IK.BANKING, 0.17, simple=0.62, proximity=0.38),
                industry(IK.INSURANCE, 0.04, simple=0.65, proximity=0.35),
                industry(IK.TELECOM, 0.12, simple=0.62, proximity=0.38),
                industry(IK.LOGISTICS, 0.11, simple=0.55, proximity=0.45),
                industry(IK.HOSPITALITY, 0.10, simple=0.58, proximity=0.42),
                industry(IK.EDUCATION, 0.07, simple=0.60, proximity=0.40),
            ],
        ),
        # ── SINGLISH HEAVY (10% = 6,000, all label=1) ─────────────────────────
        LanguageBucket(
            language=LK.SINGLISH_HEAVY,
            fraction=0.10,
            industries=[
                industry(IK.ECOMMERCE, 0.25, simple=0.60, proximity=0.40),
                industry(IK.HEALTHCARE, 0.18, simple=0.58, proximity=0.42),
                industry(IK.BANKING, 0.17, simple=0.62, proximity=0.38),
                industry(IK.INSURANCE, 0.04, simple=0.65, proximity=0.35),
                industry(IK.TELECOM, 0.13, simple=0.62, proximity=0.38),
                industry(IK.LOGISTICS, 0.10, simple=0.55, proximity=0.45),
                industry(IK.HOSPITALITY, 0.08, simple=0.58, proximity=0.42),
                industry(IK.EDUCATION, 0.05, simple=0.60, proximity=0.40),
            ],
        ),
        # ── TANGLISH LIGHT (7% = 4,200, all label=1) ──────────────────────────
        LanguageBucket(
            language=LK.TANGLISH_LIGHT,
            fraction=0.07,
            industries=[
                industry(IK.ECOMMERCE, 0.16, simple=0.60, proximity=0.40),
                industry(IK.HEALTHCARE, 0.18, simple=0.58, proximity=0.42),
                industry(IK.BANKING, 0.18, simple=0.62, proximity=0.38),
                industry(IK.INSURANCE, 0.09, simple=0.65, proximity=0.35),
                industry(IK.TELECOM, 0.14, simple=0.62, proximity=0.38),
                industry(IK.LOGISTICS, 0.11, simple=0.55, proximity=0.45),
                industry(IK.HOSPITALITY, 0.05, simple=0.58, proximity=0.42),
                industry(IK.EDUCATION, 0.09, simple=0.60, proximity=0.40),
            ],
        ),
        # ── TANGLISH HEAVY (3% = 1,800, all label=1) ──────────────────────────
        LanguageBucket(
            language=LK.TANGLISH_HEAVY,
            fraction=0.03,
            industries=[
                industry(IK.ECOMMERCE, 0.17, simple=0.60, proximity=0.40),
                industry(IK.HEALTHCARE, 0.20, simple=0.58, proximity=0.42),
                industry(IK.BANKING, 0.18, simple=0.62, proximity=0.38),
                industry(IK.INSURANCE, 0.09, simple=0.65, proximity=0.35),
                industry(IK.TELECOM, 0.13, simple=0.62, proximity=0.38),
                industry(IK.LOGISTICS, 0.10, simple=0.55, proximity=0.45),
                industry(IK.HOSPITALITY, 0.05, simple=0.58, proximity=0.42),
                industry(IK.EDUCATION, 0.08, simple=0.60, proximity=0.40),
            ],
        ),
    ],
).resolve()
