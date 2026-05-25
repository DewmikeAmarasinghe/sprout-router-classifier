"""
Unicode script detector — THE ONLY ROUTING RULE.

is_pure_script(text) returns True if the text contains ANY Sinhala or Tamil
unicode characters. The name is historical — the function correctly catches
both pure-script messages and mixed English+script messages, because it returns
True on the first Sinhala or Tamil character found anywhere in the text.

The ML classifier never sees these messages because this rule intercepts them
first and routes directly to label=1 (gpt-4o).

ALL training data uses English alphabet letters (romanized Sinhala/Tamil).
This rule handles the native-script case that the ML model never trained on.

Unicode ranges:
    Sinhala: U+0D80 – U+0DFF
    Tamil:   U+0B80 – U+0BFF

Verified by phase_1_grounding.py (2 000 samples total):
    250 pure Sinhala messages              → all caught ✅
    250 pure Tamil messages                → all caught ✅
    250 mixed English + Sinhala messages   → all caught ✅
    250 mixed English + Tamil messages     → all caught ✅
    Romanized messages (Singlish/Tanglish) → none caught ✅  (pass to ML layer)
"""

from __future__ import annotations

SINHALA_START = 0x0D80
SINHALA_END = 0x0DFF
TAMIL_START = 0x0B80
TAMIL_END = 0x0BFF


def is_pure_script(text: str) -> bool:
    """Return True if text contains any Sinhala or Tamil unicode characters.

    Runs before every ML inference call. Latency: < 0.1ms.

    Args:
        text: The raw customer message to check.

    Returns:
        True  → label=1 immediately, skip ML model.
        False → pass to ML model for classification.

    Examples:
        >>> is_pure_script("ශ්‍රී ලංකාව")
        True
        >>> is_pure_script("வணக்கம்")
        True
        >>> is_pure_script("kohomada")   # romanized Sinhala
        False
        >>> is_pure_script("enna sollu")  # romanized Tamil
        False
        >>> is_pure_script("")
        False
    """
    for char in text:
        cp = ord(char)
        if SINHALA_START <= cp <= SINHALA_END:
            return True
        if TAMIL_START <= cp <= TAMIL_END:
            return True
    return False


def script_language(text: str) -> str | None:
    """Return the script language name if pure script is detected, else None.

    More informative than is_pure_script() — useful for logging.

    Returns:
        "sinhala" | "tamil" | None
    """
    has_sinhala = False
    has_tamil = False

    for char in text:
        cp = ord(char)
        if SINHALA_START <= cp <= SINHALA_END:
            has_sinhala = True
        if TAMIL_START <= cp <= TAMIL_END:
            has_tamil = True
        if has_sinhala and has_tamil:
            break  # mixed script — both detected

    if has_sinhala and has_tamil:
        return "mixed_script"
    if has_sinhala:
        return "sinhala"
    if has_tamil:
        return "tamil"
    return None
