"""
Domain keys — shared vocabulary for the entire generation system.

Why StrEnum:
    Members ARE strings (no .value needed), serialize correctly to JSON/CSV,
    compare directly with string literals, and give full IDE autocomplete.
    Typos become red squiggles immediately.

Why separate from config files:
    LanguageKey, IndustryKey, ScenarioKey are imported by language_configs.py,
    industry_configs.py, scenario_configs.py, AND distribution.py.
    Keeping them here prevents circular imports.
"""

from __future__ import annotations

from enum import StrEnum, auto


class LanguageKey(StrEnum):
    """Training data language format.
    All formats use English alphabet letters only (romanized).
    is_pure_script() handles actual Sinhala/Tamil script — never trained on.
    """

    PURE_ENGLISH = auto()
    SINGLISH_LIGHT = auto()
    SINGLISH_HEAVY = auto()
    TANGLISH_LIGHT = auto()
    TANGLISH_HEAVY = auto()


class IndustryKey(StrEnum):
    """Sprout client industry verticals."""

    ECOMMERCE = auto()
    HEALTHCARE = auto()
    BANKING = auto()
    INSURANCE = auto()
    TELECOM = auto()
    LOGISTICS = auto()
    HOSPITALITY = auto()
    EDUCATION = auto()


class ScenarioKey(StrEnum):
    """
    Routing signal scenarios (2 total).

    label=0 eligible (only if language is PURE_ENGLISH):
        SIMPLE_TRANSACTIONAL — ALL routine queries for the industry cell.
            This includes simple named-place lookups like "Is the Kandy branch open Sunday?"
            because gpt-4o-mini handles direct lookups fine. The model learns that a
            Sri Lankan place name alone does NOT trigger routing to gpt-4o.

    always label=1 (even in pure English — requires spatial proximity reasoning):
        LOCATION_PROXIMITY — User needs to find the nearest/closest place,
            or needs spatial distance awareness between two Sri Lankan locations.
            e.g. "What is the nearest branch to me?"
                 "Which outlet closest to Colombo 3 has this dress in stock?"
                 "I am in Galle, where is the closest Vision Care?"
            gpt-4o-mini cannot reason about Sri Lankan geography / proximity.
    """

    SIMPLE_TRANSACTIONAL = auto()
    LOCATION_PROXIMITY = auto()


LABEL_0_SCENARIOS: frozenset[ScenarioKey] = frozenset({ScenarioKey.SIMPLE_TRANSACTIONAL})

ALWAYS_LABEL_1_SCENARIOS: frozenset[ScenarioKey] = frozenset({ScenarioKey.LOCATION_PROXIMITY})


def resolve_label(language: LanguageKey, scenario: ScenarioKey) -> int:
    """Return the correct training label for a (language, scenario) pair.

    Returns 1 if spatial proximity reasoning is needed (location_proximity), or if
    the language is code-mixed (gpt-4o-mini degrades on Singlish/Tanglish).
    Returns 0 only for pure English + simple transactional (any routine query).
    """
    if scenario in ALWAYS_LABEL_1_SCENARIOS:
        return 1
    return 0 if language == LanguageKey.PURE_ENGLISH else 1
