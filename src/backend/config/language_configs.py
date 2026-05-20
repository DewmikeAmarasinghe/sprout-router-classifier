"""
Language format configuration.

LanguageConfig uses frozen dataclass (not Pydantic) because these are
trusted internal definitions written by the developer, not user input.
All training data uses English alphabet letters — romanized Sinhala/Tamil.
is_pure_script() handles actual unicode script. Never trained on script.

clarification_examples:
    Short words/phrases used in CONTINUATION sub-type B prompts.
    Stored here so prompt_factory.py assembles from config, not hardcoded dicts.
    Values are plain strings (no outer quotes) — prompt_factory adds formatting.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.config.keys import LanguageKey


@dataclass(frozen=True)
class LanguageConfig:
    """Configuration for one language format used in generation."""

    key: LanguageKey
    display_name: str
    label: int
    instruction: str
    vocab_examples: tuple[str, ...]
    spelling_variations: bool
    clarification_examples: tuple[str, ...]


LANGUAGE_CONFIGS: dict[LanguageKey, LanguageConfig] = {
    LanguageKey.PURE_ENGLISH: LanguageConfig(
        key=LanguageKey.PURE_ENGLISH,
        display_name="Pure English",
        label=0,
        instruction=(
            "Write in pure English only. No Sinhala or Tamil words whatsoever. "
            "Formal or casual depending on platform context. "
            "Standard grammar, but conversational tone is fine."
        ),
        vocab_examples=(),
        spelling_variations=False,
        clarification_examples=("help", "huh?", "what?", "I don't get it", "ok?"),
    ),
    LanguageKey.SINGLISH_LIGHT: LanguageConfig(
        key=LanguageKey.SINGLISH_LIGHT,
        display_name="Singlish Light",
        label=1,
        instruction=(
            "Write mostly in English, naturally inserting 1–3 ROMANIZED Sinhala words. "
            "The message structure should feel English but with Sinhala words appearing organically. "
            "Draw from: kohomada, mama, eka, meka, dannawada, thiyanawada, karanna, "
            "api, mamath, wage, inne, ekata, thiyeno, hadanne, kiyanna, wela, awith, "
            "denna, puluwanda, koheda, oya, ape, wenna, ganna, honda, nehe, hari, "
            "oyage, mata, dunnoth, awith, oone, enawa."
        ),
        vocab_examples=(
            "kohomada",
            "mama",
            "eka",
            "meka",
            "dannawada",
            "thiyanawada",
            "karanna",
            "api",
            "mamath",
            "wage",
            "inne",
            "ekata",
            "thiyeno",
            "hadanne",
            "kiyanna",
            "wela",
            "awith",
            "denna",
            "puluwanda",
            "koheda",
            "oya",
            "ape",
            "nehe",
            "mata",
            "dunnoth",
        ),
        spelling_variations=True,
        clarification_examples=("ada", "help", "what?", "oya kiwweth?"),
    ),
    LanguageKey.SINGLISH_HEAVY: LanguageConfig(
        key=LanguageKey.SINGLISH_HEAVY,
        display_name="Singlish Heavy",
        label=1,
        instruction=(
            "Write predominantly in ROMANIZED Sinhala words (English alphabet), "
            "with some English mixed in. The message is majority Sinhala vocabulary "
            "written in English letters. Aim for 60–80% Sinhala words. "
            "Example style: 'kohomada mata denna, api order eka hadanne kohomada, "
            "mama eka size medium eka oone, delivery eka koheda enawa, "
            "meka cancel karanna puluwanda, me service eka thiyanawada.'"
        ),
        vocab_examples=(
            "kohomada",
            "api",
            "eka",
            "mama",
            "mata",
            "hadanne",
            "oone",
            "enawa",
            "karanna",
            "puluwanda",
            "meka",
            "thiyanawada",
            "koheda",
            "dunnoth",
            "kiyanna",
            "wage",
            "nehe",
            "hari",
            "oya",
            "ape",
        ),
        spelling_variations=True,
        clarification_examples=("ada", "neeya kiwwe?", "mokak karanne?"),
    ),
    LanguageKey.TANGLISH_LIGHT: LanguageConfig(
        key=LanguageKey.TANGLISH_LIGHT,
        display_name="Tanglish Light",
        label=1,
        instruction=(
            "Write mostly in English, naturally inserting 1–3 ROMANIZED Tamil words. "
            "The message structure is English but Tamil words appear organically. "
            "Draw from: enna, sollu, irukka, enakku, kiyanna, vendam, sari, romba, "
            "epdi, theriyum, illa, correct-a, podhuva, nalla, inge, ange, konjam, "
            "tharuma, yenna, paaru, solla, thevaiya, eppo, avan, aval."
        ),
        vocab_examples=(
            "enna",
            "sollu",
            "irukka",
            "enakku",
            "kiyanna",
            "vendam",
            "sari",
            "romba",
            "epdi",
            "theriyum",
            "illa",
            "nalla",
            "inge",
            "ange",
            "konjam",
            "tharuma",
            "yenna",
            "paaru",
            "thevaiya",
            "eppo",
        ),
        spelling_variations=True,
        clarification_examples=("puriyala", "help", "sollu", "theriyala"),
    ),
    LanguageKey.TANGLISH_HEAVY: LanguageConfig(
        key=LanguageKey.TANGLISH_HEAVY,
        display_name="Tanglish Heavy",
        label=1,
        instruction=(
            "Write predominantly in ROMANIZED Tamil words (English alphabet), "
            "with some English mixed in. Aim for 60–80% Tamil words. "
            "Example style: 'enna price inge, enakku free delivery irukka, "
            "return policy enna sollu, romba time aagudhu, epdi apply pannrathu, "
            "yenna nadakuthu theriyala, konjam help pannu.'"
        ),
        vocab_examples=(
            "enna",
            "sollu",
            "irukka",
            "enakku",
            "romba",
            "epdi",
            "theriyum",
            "illa",
            "inge",
            "ange",
            "konjam",
            "tharuma",
            "yenna",
            "paaru",
            "thevaiya",
            "aagudhu",
            "pannrathu",
            "nadakuthu",
            "theriyala",
        ),
        spelling_variations=True,
        clarification_examples=("puriyala", "enna ithu?", "theriyala"),
    ),
}
