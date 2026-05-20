"""
PromptFactory — assembles training data generation system prompts from config data only.

All content injected into prompts comes from:
  LANGUAGE_CONFIGS  — instruction, vocab_examples, clarification_examples
  INDUSTRY_CONFIGS  — display_name, description, product_examples, location_types
  SCENARIO_CONFIGS  — display_name, description, routing_reason, length_dist

Nothing is hardcoded in this file beyond structural constants (PLATFORM_STYLES,
CONTEXT_HEADER, SCENARIOS_WITH_SUBTYPES) which are prompt scaffolding, not
cell-specific data.

build() returns the SYSTEM PROMPT only.
The user message ("Generate exactly N messages.") is added by generator.py.
"""

from __future__ import annotations

from backend.config.industry_configs import INDUSTRY_CONFIGS
from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.config.language_configs import LANGUAGE_CONFIGS
from backend.config.scenario_configs import SCENARIO_CONFIGS, ScenarioConfig

PLATFORM_STYLES = [
    "WhatsApp Business",
    "Instagram DMs",
    "Facebook Messenger",
    "Viber",
    "website chat widget",
    "mobile app embedded chat",
]

CONTEXT_HEADER = """\
You are generating synthetic training data for Sprout — an AI-powered customer service chatbot platform built by hSenid Mobile (Sri Lanka).

Sprout serves: ecommerce/fashion, healthcare (clinics, optical, dental), banking, insurance, telecom, logistics, hospitality, and education.
Deployed on: WhatsApp Business, Instagram DMs, Facebook Messenger, Viber, website chat widgets, and mobile app chats.

Binary router labels:
  label=0 → gpt-4o-mini  (pure English, simple, no complexity signals)
  label=1 → gpt-4o       (Unicode mixed, complex, sensitive, needs spatial/emotional reasoning)\
"""

SCENARIOS_WITH_SUBTYPES: frozenset[ScenarioKey] = frozenset({ScenarioKey.CONTINUATION})


class PromptFactory:
    """Singleton — all generation system prompts are built here.

    build() returns the SYSTEM PROMPT only.
    The user message ("Generate exactly N messages.") is added by generator.py.
    """

    def build(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
        examples: list[str],
        platform_style: str,
    ) -> str:
        """Build the system prompt for a generation cell. No 'Generate N' instruction."""
        return assemble_prompt(language, industry, scenario, examples, platform_style)

    def build_preview(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
    ) -> str:
        """Build system prompt preview for Gradio UI. No API call."""
        from backend.generation.example_store import example_store

        examples = example_store.get(language, industry, scenario)
        return assemble_prompt(
            language, industry, scenario, examples, platform_style=PLATFORM_STYLES[0]
        )


PROMPT_FACTORY = PromptFactory()


def assemble_prompt(
    language: LanguageKey,
    industry: IndustryKey,
    scenario: ScenarioKey,
    examples: list[str],
    platform_style: str,
) -> str:
    sc_cfg = SCENARIO_CONFIGS[scenario]
    is_always_1 = sc_cfg.always_label_1 or language != LanguageKey.PURE_ENGLISH
    label_str = "1 (always)" if is_always_1 else "0"

    sections = [
        CONTEXT_HEADER,
        f"PLATFORM: {platform_style}",
        cell_box(language, industry, scenario, label_str),
        build_industry_section(industry),
        build_language_section(language),
        build_scenario_section(sc_cfg, label_str),
        build_length_section(sc_cfg),
    ]

    if scenario in SCENARIOS_WITH_SUBTYPES:
        sections.append(build_subtype_section(scenario, language))

    if examples:
        sections.append(build_examples_section(language, industry, scenario, examples))

    sections.append(build_anti_scenarios_section(sc_cfg))
    sections.append(build_output_rules(language, industry, scenario, platform_style))

    return "\n\n".join(sections)


def cell_box(
    language: LanguageKey,
    industry: IndustryKey,
    scenario: ScenarioKey,
    label_str: str,
) -> str:
    line = f"  GENERATING FOR:  {language}  ×  {industry}  ×  {scenario}"
    width = max(len(line) + 2, 60)
    bar = "═" * width
    return f"╔{bar}╗\n{line}\n  Label:  {label_str}\n╚{bar}╝"


def build_industry_section(industry: IndustryKey) -> str:
    """Inject display name, description, domain terms, and location types from IndustryConfig."""
    ind_cfg = INDUSTRY_CONFIGS[industry]
    return (
        "── INDUSTRY ──\n"
        f"  {ind_cfg.display_name}: {ind_cfg.description}\n"
        f"  Domain terms:   {', '.join(ind_cfg.product_examples[:8])}\n"
        f"  Location types: {', '.join(ind_cfg.location_types)}"
    )


def build_language_section(language: LanguageKey) -> str:
    """Inject the full language instruction from LanguageConfig.instruction.

    Previously broken: used getattr(lang_cfg, 'description', '') which returned ''
    since LanguageConfig has no description field — the correct field is instruction.
    This meant singlish_light's '1-3 romanized words' and singlish_heavy's
    '60-80% Sinhala' guidance were completely absent from all generated prompts.
    """
    lang_cfg = LANGUAGE_CONFIGS[language]
    lines = [
        "── LANGUAGE MANDATE ──",
        f"  {lang_cfg.display_name}: ",
        f"  {lang_cfg.instruction}",
    ]

    if language == LanguageKey.PURE_ENGLISH:
        lines.append(
            "  ⚠ STRICT: Write ONLY English. "
            "Do NOT include any Sinhala, Tamil, or other language words in any message."
        )
    elif language in (LanguageKey.SINGLISH_LIGHT, LanguageKey.SINGLISH_HEAVY):
        lines.append(
            "  ⚠ Mix romanized Sinhala words naturally with English. Do NOT include Tamil words."
        )
    elif language in (LanguageKey.TANGLISH_LIGHT, LanguageKey.TANGLISH_HEAVY):
        lines.append(
            "  ⚠ Mix romanized Tamil words naturally with English. Do NOT include Sinhala words."
        )

    return "\n".join(lines)


def build_scenario_section(sc_cfg: ScenarioConfig, label_str: str) -> str:
    """Inject scenario display name, description, label, and routing_reason from ScenarioConfig.

    routing_reason is used verbatim from the config — no language-specific overrides.
    The routing_reason for SIMPLE_TRANSACTIONAL already covers both cases:
    pure English (first sentence) and Unicode mixed (second sentence).
    """
    return (
        f"── TASK: {sc_cfg.display_name} ──\n"
        f"  {sc_cfg.description}\n"
        f"  LABEL: {label_str}\n"
        f"  Routing reason: {sc_cfg.routing_reason}"
    )


def build_length_section(sc_cfg: ScenarioConfig) -> str:
    lines = ["── LENGTH ──"]
    for r in sc_cfg.length_dist.ranges:
        pct = round(r.fraction * 100)
        ex = f'  (e.g. "{r.examples[0]}")' if r.examples else ""
        lines.append(f"  {pct}%: {r.min_words}–{r.max_words} words{ex}")
    return "\n".join(lines)


def build_subtype_section(scenario: ScenarioKey, language: LanguageKey) -> str:
    if scenario == ScenarioKey.CONTINUATION:
        return build_continuation_subtype(language)
    return ""


def build_continuation_subtype(language: LanguageKey) -> str:
    """Build continuation sub-type section using clarification_examples from LanguageConfig.

    Reads from LANGUAGE_CONFIGS[language].clarification_examples — no hardcoded dict.
    Values are plain strings; formatted with quotes here for the prompt.
    """
    lang_cfg = LANGUAGE_CONFIGS[language]
    clarif = [f'"{ex}"' for ex in lang_cfg.clarification_examples[:4]]
    clarif_str = ", ".join(clarif)
    lang_label = language.replace("_", " ")
    return (
        "── SUB-TYPE MIX ──\n"
        "  ~55% type A — PREVIOUS ACTION FAILED:\n"
        "    User reports that a previous chatbot action failed or keeps failing.\n"
        '    e.g. "still shows error", "tried again same problem", "keeps failing"\n\n'
        f"  ~45% type B — UNCLEAR INTENT / CLARIFICATION ({lang_label}):\n"
        f"    Very short unclear message or asking to re-explain — in {lang_label}.\n"
        f'    e.g. {clarif_str}, "I didn\'t get that", "can you explain again?"'
    )


def build_examples_section(
    language: LanguageKey,
    industry: IndustryKey,
    scenario: ScenarioKey,
    examples: list[str],
) -> str:
    numbered = "\n".join(f'  {i + 1}. "{ex}"' for i, ex in enumerate(examples))
    return (
        f"── EXAMPLES FOR {language}×{industry}×{scenario} ──\n"
        f"  (vary phrasing significantly — do NOT copy verbatim)\n"
        f"{numbered}"
    )


def build_anti_scenarios_section(sc_cfg: ScenarioConfig) -> str:
    """Show ALL other scenarios so the model knows exactly what NOT to generate."""
    lines = [
        "── DO NOT GENERATE ──",
        "  ONLY generate the target scenario above. Every other scenario is off-limits:",
    ]
    for key, cfg in SCENARIO_CONFIGS.items():
        if key == sc_cfg.key:
            continue
        ex = ""
        if cfg.length_dist.ranges and cfg.length_dist.ranges[0].examples:
            ex = f'  e.g. "{cfg.length_dist.ranges[0].examples[0]}"'
        lines.append(f"  ✗ {key:<35} [{cfg.display_name}]{ex}")
    return "\n".join(lines)


def build_output_rules(
    language: LanguageKey,
    industry: IndustryKey,
    scenario: ScenarioKey,
    platform: str,
) -> str:
    """Build output rules section. All label text comes from config lookups."""
    lang_display = LANGUAGE_CONFIGS[language].display_name
    ind_display = INDUSTRY_CONFIGS[industry].display_name
    sc_display = SCENARIO_CONFIGS[scenario].display_name

    return (
        "── OUTPUT RULES ──\n"
        f"  Write ACTUAL customer messages as typed in {platform}.\n\n"
        f"  1. Each 'text' must be a probable message a real customer would send to a\n"
        f"     Sprout chatbot ({lang_display} × {ind_display} × {sc_display}).\n"
        f"     NOT a meta-instruction asking to generate something. NOT a description.\n"
        f'     WRONG: "Generate a short message asking about order status."\n'
        f'     WRONG: "A customer asking about their delivery."\n'
        f"     CORRECT: \"My order hasn't arrived, it's been 3 days.\"\n\n"
        "  2. Use everyday vocabulary real customers use in chat. Avoid formal language.\n"
        '     WRONG: "The checkout flow exhibits persistent failures requiring escalation."\n'
        '     CORRECT: "It keeps failing when I try to pay."\n\n'
        "  3. Vary specifics — mention products, actions, places, times.\n"
        '     e.g. "blue dress", "promo code SAVE20", "saree in XL", "Colombo"\n\n'
        "  4. Occasional realistic typos (1–2 per batch).\n"
        '     e.g. "stil shows error", "chekcout not wrking"\n\n'
        f'Return ONLY this JSON: {{"prompts": [{{"text": "..."}}]}}'
    )
