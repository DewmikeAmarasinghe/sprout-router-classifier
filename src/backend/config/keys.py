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
    Routing signal scenarios (9 total).

    label=0 eligible (only if language is PURE_ENGLISH):
        SIMPLE_TRANSACTIONAL, NAMED_LOCATION

    always label=1:
        LOCATION_PROXIMITY, LOCATION_RELATIVE, COMPLEX_TASK,
        SENSITIVE_CONTEXT, ESCALATION, RESPONSE_LANGUAGE, CONTINUATION
    """

    SIMPLE_TRANSACTIONAL = auto()
    NAMED_LOCATION = auto()
    LOCATION_PROXIMITY = auto()
    LOCATION_RELATIVE = auto()
    COMPLEX_TASK = auto()
    SENSITIVE_CONTEXT = auto()
    ESCALATION = auto()
    RESPONSE_LANGUAGE = auto()
    CONTINUATION = auto()


LABEL_0_SCENARIOS: frozenset[ScenarioKey] = frozenset(
    {
        ScenarioKey.SIMPLE_TRANSACTIONAL,
        ScenarioKey.NAMED_LOCATION,
    }
)

ALWAYS_LABEL_1_SCENARIOS: frozenset[ScenarioKey] = frozenset(set(ScenarioKey) - LABEL_0_SCENARIOS)


def resolve_label(language: LanguageKey, scenario: ScenarioKey) -> int:
    """Return the correct training label for a (language, scenario) pair.

    Returns 1 if the scenario always needs gpt-4o, or if the language is
    code-mixed (Singlish/Tanglish). Returns 0 only for pure English + simple intent.
    """
    if scenario in ALWAYS_LABEL_1_SCENARIOS:
        return 1
    return 0 if language == LanguageKey.PURE_ENGLISH else 1
