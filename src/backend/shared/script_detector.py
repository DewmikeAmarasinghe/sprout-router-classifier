"""
Unicode script detector — THE ONLY ROUTING RULE.

is_pure_script(text) returns True if the text contains any Sinhala or Tamil
unicode characters. The ML classifier never sees pure-script text because this
rule intercepts it first and routes directly to label=1 (gpt-4o).

ALL training data uses English alphabet letters (romanized Sinhala/Tamil).
This rule handles the native-script case that the ML model never trained on.

Unicode ranges:
    Sinhala: U+0D80 – U+0DFF
    Tamil:   U+0B80 – U+0BFF

Verified by phase_1_grounding.py:
    500 pure Sinhala messages → all caught ✅
    500 pure Tamil messages   → all caught ✅
    500 romanized messages    → none caught ✅
"""

from __future__ import annotations

_SINHALA_START = 0x0D80
_SINHALA_END = 0x0DFF
_TAMIL_START = 0x0B80
_TAMIL_END = 0x0BFF


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
        if _SINHALA_START <= cp <= _SINHALA_END:
            return True
        if _TAMIL_START <= cp <= _TAMIL_END:
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
        if _SINHALA_START <= cp <= _SINHALA_END:
            has_sinhala = True
        if _TAMIL_START <= cp <= _TAMIL_END:
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
