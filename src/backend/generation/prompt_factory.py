"""
PromptFactory — builds training data generation prompts from config components.

IMPROVEMENTS:
- DO NOT GENERATE shows ALL other scenarios (not just anti_scenario_keys)
- No duplicate TASK + SUB-TYPE sections
- Language-aware sub-type examples — "ada"/"puriyala" never in pure_english prompts
- Explicit OUTPUT RULES preventing "Generate a..." meta-outputs
- Vocabulary guidance + typo allowance
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
  label=1 → gpt-4o       (code-mixed, complex, sensitive, needs spatial/emotional reasoning)\
"""

# Language-appropriate short clarification words for CONTINUATION sub-type B.
# Prevents Sinhala/Tamil words appearing in pure_english prompts.
CLARIFICATION_WORDS_BY_LANGUAGE: dict[LanguageKey, list[str]] = {
    LanguageKey.PURE_ENGLISH: ['"help"', '"huh?"', '"what?"', '"I don\'t get it"', '"ok?"'],
    LanguageKey.SINGLISH_LIGHT: ['"ada"', '"help"', '"what?"', '"oya kiwweth?"'],
    LanguageKey.SINGLISH_HEAVY: ['"ada"', '"neeya kiwwe?"', '"mokak karanne?"'],
    LanguageKey.TANGLISH_LIGHT: ['"puriyala"', '"help"', '"sollu"', '"theriyala"'],
    LanguageKey.TANGLISH_HEAVY: ['"puriyala"', '"enna ithu?"', '"theriyala"'],
}

SCENARIOS_WITH_SUBTYPES: frozenset[ScenarioKey] = frozenset({ScenarioKey.CONTINUATION})


class PromptFactory:
    """Singleton builder — all generation prompts are built here."""

    def build(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
        examples: list[str],
        n: int,
        platform_style: str,
    ) -> str:
        return assemble_prompt(language, industry, scenario, examples, n, platform_style)

    def build_preview(
        self,
        language: LanguageKey,
        industry: IndustryKey,
        scenario: ScenarioKey,
    ) -> str:
        from backend.generation.example_store import example_store

        examples = example_store.get(language, industry, scenario)
        return assemble_prompt(
            language, industry, scenario, examples, n=50, platform_style=PLATFORM_STYLES[0]
        )


PROMPT_FACTORY = PromptFactory()


def assemble_prompt(
    language: LanguageKey,
    industry: IndustryKey,
    scenario: ScenarioKey,
    examples: list[str],
    n: int,
    platform_style: str,
) -> str:
    lang_cfg = LANGUAGE_CONFIGS[language]
    ind_cfg = INDUSTRY_CONFIGS[industry]
    sc_cfg = SCENARIO_CONFIGS[scenario]

    is_always_1 = sc_cfg.always_label_1 or language != LanguageKey.PURE_ENGLISH
    label_str = "1 (always)" if is_always_1 else "0"

    sections = [
        CONTEXT_HEADER,
        f"PLATFORM: {platform_style}",
        cell_box(language, industry, scenario, label_str),
        build_industry_section(ind_cfg),
        build_language_section(language, lang_cfg),
        build_scenario_section(sc_cfg, label_str),
        build_length_section(sc_cfg),
    ]

    if scenario in SCENARIOS_WITH_SUBTYPES:
        sections.append(build_subtype_section(scenario, language))

    if examples:
        sections.append(build_examples_section(language, industry, scenario, examples))

    sections.append(build_anti_scenarios_section(sc_cfg))
    sections.append(build_output_rules(n, platform_style))

    return "\n\n".join(sections)


def cell_box(
    language: LanguageKey, industry: IndustryKey, scenario: ScenarioKey, label_str: str
) -> str:
    line = f"  GENERATING FOR:  {language}  ×  {industry}  ×  {scenario}"
    width = max(len(line) + 2, 60)
    bar = "═" * width
    return f"╔{bar}╗\n{line}\n  Label:  {label_str}\n╚{bar}╝"


def build_industry_section(ind_cfg: object) -> str:
    lines = ["── INDUSTRY ──"]
    lines.append(f"  {getattr(ind_cfg, 'display_name', '')}: {getattr(ind_cfg, 'description', '')}")
    if loc := getattr(ind_cfg, "location_terms", None):
        lines.append(f"  Location terms: {', '.join(loc)}")
    if dom := getattr(ind_cfg, "domain_terms", None):
        lines.append(f"  Domain terms:   {', '.join(dom)}")
    if plat := getattr(ind_cfg, "platform", None):
        lines.append(f"  Platform:       {plat}")
    return "\n".join(lines)


def build_language_section(language: LanguageKey, lang_cfg: object) -> str:
    desc = getattr(lang_cfg, "description", "")
    display = getattr(lang_cfg, "display_name", language)
    lines = ["── LANGUAGE MANDATE ──", f"  {display}: {desc}"]

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
    clarif = CLARIFICATION_WORDS_BY_LANGUAGE.get(
        language, CLARIFICATION_WORDS_BY_LANGUAGE[LanguageKey.PURE_ENGLISH]
    )
    clarif_str = ", ".join(clarif[:4])
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
    """Show ALL other scenarios — not just anti_scenario_keys — so the model
    clearly understands what it must NOT generate."""
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


def build_output_rules(n: int, platform: str) -> str:
    return (
        "── OUTPUT RULES ──\n"
        f"  Write ACTUAL customer messages as typed in {platform}.\n\n"
        "  1. Each 'text' must be the raw customer message — NOT an instruction or description.\n"
        '     WRONG: "Generate a 1-6 word message about checkout error."\n'
        '     CORRECT: "Still shows error."\n\n'
        "  2. Use everyday vocabulary real customers use in chat. Avoid formal business terms.\n"
        '     WRONG: "The checkout flow exhibits persistent failures requiring escalation."\n'
        '     CORRECT: "It keeps failing when I try to pay."\n\n'
        "  3. Vary specifics — mention products, actions, places, times.\n"
        '     e.g. "blue dress", "promo code SAVE20", "saree in XL", "Colombo"\n\n'
        "  4. Occasional realistic typos (1–2 per 50 messages).\n"
        '     e.g. "stil shows error", "chekcout not wrking"\n\n'
        f'Return ONLY valid JSON: {{"prompts": [{{"text": "...", "word_count": N}}, ...]}}\n'
        f"Generate exactly {n} messages."
    )
