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
You are generating synthetic training data for Sprout — an AI-powered customer service chatbot \
platform built by hSenid Mobile (Sri Lanka).

Sprout serves: ecommerce/fashion, healthcare (clinics, optical, dental), banking, insurance, \
telecom, logistics, hospitality, and education.
Deployed on: WhatsApp Business, Instagram DMs, Facebook Messenger, Viber, website chat widgets, \
and mobile app chats.

YOUR TASK:
  Generate realistic messages that Sri Lankan customers would ACTUALLY type to a business chatbot.
  Focus on the most probable, high-frequency queries for this specific language × industry × scenario.
  Each batch must cover DIFFERENT situations — avoid repeating similar queries.

Binary router labels:
  label=0 → gpt-4o-mini  (pure English, simple, no complexity signals)
  label=1 → gpt-4o       (Unicode mixed, complex, sensitive, needs spatial/emotional reasoning)\
"""

# Real Sri Lankan locations injected for location-heavy scenarios.
SRI_LANKAN_LOCATIONS = """\
── SRI LANKAN LOCATIONS ──
  Always use real Sri Lankan place names — never invent fictional ones.
  Here are examples to get you started; you may use any real location you know, not only these:
  Cities:        Colombo, Kandy, Galle, Negombo, Matara, Jaffna, Kurunegala,
                 Nuwara Eliya, Batticaloa, Trincomalee, Anuradhapura, Ratnapura
  Colombo areas: Fort, Pettah, Kollupitiya, Bambalapitiya, Wellawatte, Dehiwala,
                 Mount Lavinia, Borella, Nugegoda, Maharagama, Rajagiriya, Battaramulla
  Malls/areas:   Majestic City, Liberty Plaza, One Galle Face Mall, Odel, Cargills Square,
                 Unity Plaza, House of Fashion, Colombo City Centre
  Hospitals:     Asiri Central, Nawaloka, Lanka, Durdans, National Hospital
  ❌ Never invent fictional places: "City Park", "Cross Street", "River Bridge", "Grand Mosque junction"\
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

    # Add location guidance for location-specific scenarios
    if scenario in (
        ScenarioKey.NAMED_LOCATION,
        ScenarioKey.LOCATION_PROXIMITY,
        ScenarioKey.LOCATION_RELATIVE,
    ):
        sections.append(SRI_LANKAN_LOCATIONS)

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
        f"  Location types: {', '.join(ind_cfg.location_types)}\n"
        f"  Platform:       {ind_cfg.typical_platform}"
    )


def build_language_section(language: LanguageKey) -> str:
    """Inject the full language instruction from LanguageConfig.instruction."""
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

    routing_reason is used verbatim — no language-specific overrides.
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

    CRITICAL: Sub-type A = PREVIOUS SERVICE ACTION failed (payment, order, upgrade, booking).
    Never "The bot failed" or "Your chat froze" — the SERVICE failed, not the chatbot.
    """
    lang_cfg = LANGUAGE_CONFIGS[language]
    clarif = [f'"{ex}"' for ex in lang_cfg.clarification_examples[:4]]
    clarif_str = ", ".join(clarif)
    lang_label = language.replace("_", " ")
    return (
        "── SUB-TYPE MIX ──\n"
        "  ~55% type A — PREVIOUS SERVICE ACTION FAILED:\n"
        "    A service action (payment, booking, order, upgrade) failed — user follows up.\n"
        '    e.g. "still shows error", "tried again same problem", "payment keeps failing"\n'
        "    ❌ NEVER: 'The bot failed', 'Your chat froze', 'The AI is broken'\n\n"
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
    """Build output rules. All display text comes from config lookups.

    RULE 2 deliberately does NOT ban instruction words — customers DO instruct
    chatbots ("cancel my order", "send me the details", "check my balance").
    The issue is REPETITION: every message starting with the same word is unnatural.
    Rule 2 targets monotony, not instruction vocabulary.
    """
    lang_display = LANGUAGE_CONFIGS[language].display_name
    ind_display = INDUSTRY_CONFIGS[industry].display_name
    sc_display = SCENARIO_CONFIGS[scenario].display_name

    return (
        "── OUTPUT RULES (read carefully before generating) ──\n\n"
        "Before generating each message, think:\n"
        "  - Would a real Sri Lankan customer actually type this to this business?\n"
        "  - Is this different from what I already generated this session?\n"
        "  - Does this follow every rule below?\n\n"
        f"Each 'text' must be a probable message a real customer sends to a Sprout\n"
        f"chatbot ({lang_display} × {ind_display} × {sc_display}).\n\n"
        "ABSOLUTE RULES:\n\n"
        "  RULE 1: Output the raw message text ONLY. No prefix, label, number, or annotation.\n"
        '    ❌ "label=1 Nearest campus?"        ← label prefix\n'
        '    ❌ "Seed 3: 1-6 words:"             ← seed prefix\n'
        '    ❌ "Escalation 14: Not happy"        ← escalation prefix\n'
        '    ❌ "Generate a message about..."     ← meta-instruction\n'
        '    ❌ "A customer asking about..."      ← description, not message\n'
        "    ✅ \"My order hasn't arrived, it's been 3 days.\"\n\n"
        "  RULE 2: Vary how messages begin — do NOT start every message with the same word.\n"
        "    Real customers open messages many different ways. Mix:\n"
        '    • Questions:   "Can I track my parcel?", "Is COD available?"\n'
        '    • Commands:    "Cancel my order", "Send me the receipt"\n'
        '    • Statements:  "My payment failed again", "Still no update"\n'
        '    • Greetings:   "Hi, I need help with...", "Good morning"\n'
        '    • Short bursts: "Status?", "Price?", "hi", "ok"\n'
        '    ❌ All 50 messages starting with "Enna," or "I want" or "Hi,"\n\n'
        "  RULE 3: Continuation = mid-conversation follow-up. NOT a bot/AI complaint.\n"
        '    ❌ "The bot failed again."  ❌ "Your chat froze."  ❌ "AI broke."\n'
        '    ✅ "Payment keeps failing."  ✅ "Tried again, same error."\n\n'
        "  RULE 4: Use real Sri Lankan place names only — never invent fictional locations.\n"
        "    You may use any real place in Sri Lanka, not just the examples listed above.\n"
        '    ❌ "City Park", "Cross Street", "River Bridge"  (fictional — never use)\n'
        '    ✅ "Majestic City branch?", "Near Kandy City Centre?", or any real place\n\n'
        "  RULE 5: Vary reference IDs — never repeat the same format pattern:\n"
        "    ❌ SH7777, SH8888, SH9999\n"
        "    ✅ SH-1234, TRK-001, #45678, P-209, REF/2024/001, ORD-5K\n\n"
        "  RULE 6: Write casual chat language, not formal prose.\n"
        '    ❌ "The checkout process exhibits persistent failure requiring escalation."\n'
        '    ✅ "It keeps failing when I try to pay."\n\n'
        '  Occasional typos are fine (1–2 per batch): e.g. "stil shows error"\n\n'
        f'Return ONLY this JSON: {{"prompts": [{{"text": "..."}}]}}'
    )
