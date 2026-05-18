"""
PromptFactory — builds system prompts for any (language × industry × scenario) cell.

Anti-combinations section lists ALL other 8 scenarios (not just anti_scenario_keys),
so the LLM has a complete picture of what NOT to generate.
"""

from __future__ import annotations

import abc

from backend.config.industry_configs import INDUSTRY_CONFIGS, IndustryConfig
from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.config.language_configs import LANGUAGE_CONFIGS, LanguageConfig
from backend.config.scenario_configs import SCENARIO_CONFIGS, ScenarioConfig

PLATFORM_STYLES: tuple[str, ...] = (
    "WhatsApp Business — casual, short, emoji sometimes, typos common",
    "Instagram DM — casual, younger audience, abbreviations",
    "Facebook Messenger — casual to semi-formal",
    "Viber Business — casual, similar to WhatsApp",
    "SMS — very short, no formatting, direct",
    "website chatbot widget — slightly more formal, complete sentences",
    "mobile app embedded chat — medium formality, action-oriented",
    "Shopify / WooCommerce storefront chat — shopping context, product-focused",
)

SPROUT_CONTEXT = (
    "You are generating synthetic training data for Sprout — an AI-powered "
    "customer service chatbot platform built by hSenid Mobile (Sri Lanka).\n\n"
    "Sprout serves: ecommerce/fashion, healthcare (clinics, optical, dental), "
    "banking, insurance, telecom, logistics, hospitality, and education.\n\n"
    "Deployed on: WhatsApp Business, Instagram DMs, Facebook Messenger, Viber, "
    "website chat widgets, and mobile app embedded chats.\n\n"
    "The data trains a binary router:\n"
    "  label=0 → gpt-4o-mini  (pure English, simple, no complexity)\n"
    "  label=1 → gpt-4o       (code-mixed, complex, sensitive, location reasoning)"
)


def build_anti_combinations_section(
    language: LanguageKey,
    industry: IndustryKey,
    scenario: ScenarioKey,
) -> str:
    """List ALL other 8 scenarios as anti-combinations.

    Shows the LLM what it must NOT generate — a concrete example from each
    other scenario so the distinction is clear. All 8 are listed (not just
    the 2 from anti_scenario_keys) because any scenario confusion hurts
    classifier training quality.
    """
    lines = [
        f"DO NOT generate messages that fit any other combination — "
        f"ONLY generate {language}×{industry}×{scenario}:"
    ]
    for other in ScenarioKey:
        if other == scenario:
            continue
        other_sc = SCENARIO_CONFIGS[other]
        example = ""
        for r in other_sc.length_dist.ranges:
            if r.examples:
                example = f'e.g. "{r.examples[0]}"'
                break
        lines.append(f"  ✗ {language}×{industry}×{other}  [{other_sc.display_name}]  {example}")
    return "\n".join(lines) + "\n\n"


class SectionBuilder(abc.ABC):
    """Abstract base — builds the scenario-specific section of a prompt."""

    @abc.abstractmethod
    def build(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        lang: LanguageConfig,
        ind: IndustryConfig,
        sc: ScenarioConfig,
        examples: list[str],
        n: int,
    ) -> str:
        """Return the scenario-specific prompt section."""


class StandardSectionBuilder(SectionBuilder):
    def build(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        lang: LanguageConfig,
        ind: IndustryConfig,
        sc: ScenarioConfig,
        examples: list[str],
        n: int,
    ) -> str:
        label_note = (
            "1 (always — regardless of language)"
            if sc.always_label_1
            else "0 for pure English  /  1 for Singlish or Tanglish"
        )
        products_str = ", ".join(ind.product_examples[:8])
        locations_str = ", ".join(ind.location_types)
        examples_str = "\n".join(f'  {i + 1}. "{ex}"' for i, ex in enumerate(examples[:8]))

        return (
            f"── INDUSTRY ──\n"
            f"  {ind.display_name}: {ind.description}\n"
            f"  Location terms: {locations_str}\n"
            f"  Domain terms:   {products_str}\n"
            f"  Platform:       {ind.typical_platform}\n\n"
            f"── LANGUAGE ──\n"
            f"  {lang.display_name}: {lang.instruction}\n\n"
            f"── TASK: {sc.display_name} ──\n"
            f"  {sc.description}\n\n"
            f"  LABEL: {label_note}\n"
            f"  Reason: {sc.routing_reason}\n\n"
            f"── LENGTH ──\n"
            f"{sc.length_dist.to_prompt_str()}\n\n"
            f"── EXAMPLES FOR {language}×{industry}×{sc.key} ──\n"
            f"  (vary phrasing significantly — do NOT copy verbatim)\n"
            f"{examples_str}\n\n"
            f"{build_anti_combinations_section(language, industry, sc.key)}"
        )


class LocationRelativeSectionBuilder(SectionBuilder):
    def build(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        lang: LanguageConfig,
        ind: IndustryConfig,
        sc: ScenarioConfig,
        examples: list[str],
        n: int,
    ) -> str:
        base = StandardSectionBuilder().build(language, industry, lang, ind, sc, examples, n)
        return base + (
            "── IMPORTANT ──\n"
            "  Reference points are NOT limited to business locations.\n"
            "  Users refer to: schools, junctions, landmarks, shopping centers,\n"
            "  hospitals, buildings, and neighborhoods.\n"
            "  e.g. 'near Dharmapala Vidyalaya', '100m from the Cargills',\n"
            "  'past the Galle road junction', 'in the Pettah market area'\n\n"
        )


class ContinuationSectionBuilder(SectionBuilder):
    def build(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        lang: LanguageConfig,
        ind: IndustryConfig,
        sc: ScenarioConfig,
        examples: list[str],
        n: int,
    ) -> str:
        base = StandardSectionBuilder().build(language, industry, lang, ind, sc, examples, n)
        return base + (
            "── SUB-TYPE MIX ──\n"
            "  ~55% type A — PREVIOUS ACTION FAILED:\n"
            "    User reports failure, asks for help again.\n"
            '    e.g. "it still shows error", "tried again same problem"\n'
            "  ~45% type B — UNCLEAR INTENT / CLARIFICATION:\n"
            "    Very short unclear message or user asking to re-explain.\n"
            '    e.g. "help", "ada", "puriyala", "I didn\'t get that"\n\n'
        )


class PromptFactory:
    """Builds complete system prompts. Anti-combinations show all 8 other scenarios."""

    SECTION_BUILDERS: dict[ScenarioKey, SectionBuilder] = {
        ScenarioKey.SIMPLE_TRANSACTIONAL: StandardSectionBuilder(),
        ScenarioKey.NAMED_LOCATION: StandardSectionBuilder(),
        ScenarioKey.LOCATION_PROXIMITY: StandardSectionBuilder(),
        ScenarioKey.LOCATION_RELATIVE: LocationRelativeSectionBuilder(),
        ScenarioKey.COMPLEX_TASK: StandardSectionBuilder(),
        ScenarioKey.SENSITIVE_CONTEXT: StandardSectionBuilder(),
        ScenarioKey.ESCALATION: StandardSectionBuilder(),
        ScenarioKey.RESPONSE_LANGUAGE: StandardSectionBuilder(),
        ScenarioKey.CONTINUATION: ContinuationSectionBuilder(),
    }

    def build(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
        examples: list[str],
        n: int,
        platform_style: str = "WhatsApp Business — casual",
    ) -> str:
        """Build the complete first-turn prompt for a cell.

        Anti-examples are derived internally from all other ScenarioKey values.
        No anti_examples parameter needed.
        """
        lang_cfg = LANGUAGE_CONFIGS[language]
        ind_cfg = INDUSTRY_CONFIGS[industry]
        sc_cfg = SCENARIO_CONFIGS[scenario]
        builder = self.SECTION_BUILDERS[scenario]
        section = builder.build(language, industry, lang_cfg, ind_cfg, sc_cfg, examples, n)

        cell_id_box = (
            f"╔══════════════════════════════════════════════════════════╗\n"
            f"  GENERATING FOR:  {language}  ×  {industry}  ×  {scenario}\n"
            f"  Label:  {'1 (always)' if sc_cfg.always_label_1 else '0 (pure English) / 1 (code-mixed)'}\n"
            f"╚══════════════════════════════════════════════════════════╝\n\n"
        )

        return (
            f"{SPROUT_CONTEXT}\n\n"
            f"PLATFORM: {platform_style}\n\n"
            f"{cell_id_box}"
            f"{section}"
            f'Return ONLY valid JSON: {{"prompts": [{{"text": "...", "word_count": N}}, ...]}}\n'
            f"Generate exactly {n} messages. Each must be unique and realistic."
        )

    def build_preview(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
    ) -> str:
        """Build preview using cached or LengthRange fallback examples."""
        from backend.generation.example_store import example_store

        examples = example_store.get(language, industry, scenario)
        return self.build(
            language=language, industry=industry, scenario=scenario, examples=examples, n=50
        )


PROMPT_FACTORY = PromptFactory()
