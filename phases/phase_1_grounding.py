"""
Phase 1 — Grounding Dataset Generator.

Generates 500 pure Sinhala + 500 pure Tamil customer service messages,
then runs is_pure_script() on all 1,000 to verify the unicode rule catches 100%.

Run once:
    python phases/phase_1_grounding.py

Output:
    data/grounding/unicode_verification.csv

This proves the routing rule is correct BEFORE training:
    - Pure Sinhala messages → all caught → label=1 without ML
    - Pure Tamil messages   → all caught → label=1 without ML
    - These are NOT used for training — the ML model never sees pure-script text.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.shared.path_resolver import DATA_DIR
from backend.shared.script_detector import is_pure_script, script_language
from backend.shared.settings_manager import settings_manager

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / "grounding" / "unicode_verification.csv"
TARGET_EACH = 500

_SINHALA_PROMPT = (
    "Generate {n} realistic customer service messages written in pure Sinhala script. "
    "These MUST use actual Sinhala unicode characters (ශ, ල, ා, ් etc.) — NOT romanized. "
    "They should look like real WhatsApp or website chatbot messages. "
    "Topics: ordering, booking appointments, checking delivery, prices, complaints. "
    "Vary length: some short (5–10 words), some medium (10–25 words). "
    'Return ONLY valid JSON: {{"messages": ["message1", ..., "message{n}"]}}'
)

_TAMIL_PROMPT = (
    "Generate {n} realistic customer service messages written in pure Tamil script. "
    "These MUST use actual Tamil unicode characters (த, ம, ி, ழ etc.) — NOT romanized. "
    "They should look like real WhatsApp or website chatbot messages. "
    "Topics: ordering, booking appointments, checking delivery, prices, complaints. "
    "Vary length: some short (5–10 words), some medium (10–25 words). "
    'Return ONLY valid JSON: {{"messages": ["message1", ..., "message{n}"]}}'
)


def generate_script_messages(script: str, n: int) -> list[str]:
    """Generate n pure-script messages via OpenAI API."""
    from openai import OpenAI

    client = OpenAI()
    template = _SINHALA_PROMPT if script == "sinhala" else _TAMIL_PROMPT
    batch = 50
    messages: list[str] = []

    while len(messages) < n:
        remaining = min(batch, n - len(messages))
        response = client.chat.completions.create(
            model=settings_manager.get("GENERATION_LLM"),
            messages=[{"role": "user", "content": template.format(n=remaining)}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            log.warning("Empty response — retrying")
            continue

        data = json.loads(content)
        batch_msgs = data.get("messages", [])
        messages.extend(batch_msgs[:remaining])
        log.info(f"  {script}: {len(messages)}/{n}")

    return messages[:n]


def verify_and_save(
    sinhala_msgs: list[str],
    tamil_msgs: list[str],
) -> dict[str, int]:
    """Run is_pure_script() on all messages. Save results to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    si_caught = ta_caught = 0

    for msg in sinhala_msgs:
        caught = is_pure_script(msg)
        lang = script_language(msg) or "none"
        if caught:
            si_caught += 1
        rows.append({"text": msg, "expected_script": "sinhala", "caught": caught, "detected": lang})

    for msg in tamil_msgs:
        caught = is_pure_script(msg)
        lang = script_language(msg) or "none"
        if caught:
            ta_caught += 1
        rows.append({"text": msg, "expected_script": "tamil", "caught": caught, "detected": lang})

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "expected_script", "caught", "detected"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "sinhala_total": len(sinhala_msgs),
        "sinhala_caught": si_caught,
        "tamil_total": len(tamil_msgs),
        "tamil_caught": ta_caught,
    }


def main() -> None:
    print("=" * 60)
    print("Phase 1 — Grounding: Unicode Rule Verification")
    print("=" * 60)

    log.info(f"Generating {TARGET_EACH} pure Sinhala messages...")
    sinhala = generate_script_messages("sinhala", TARGET_EACH)

    log.info(f"Generating {TARGET_EACH} pure Tamil messages...")
    tamil = generate_script_messages("tamil", TARGET_EACH)

    log.info("Running is_pure_script() verification...")
    stats = verify_and_save(sinhala, tamil)

    si_pass = stats["sinhala_caught"] == stats["sinhala_total"]
    ta_pass = stats["tamil_caught"] == stats["tamil_total"]

    print("\n" + "=" * 60)
    print("GROUNDING VERIFICATION RESULTS")
    print("=" * 60)
    print(
        f"  Sinhala: {stats['sinhala_caught']}/{stats['sinhala_total']} caught  "
        f"{'✅ PASS' if si_pass else '❌ FAIL — fix unicode ranges in script_detector.py'}"
    )
    print(
        f"  Tamil:   {stats['tamil_caught']}/{stats['tamil_total']} caught  "
        f"{'✅ PASS' if ta_pass else '❌ FAIL — fix unicode ranges in script_detector.py'}"
    )
    print(f"\n  Output: {OUTPUT_PATH}")
    print("=" * 60)

    if not si_pass or not ta_pass:
        missed_si = stats["sinhala_total"] - stats["sinhala_caught"]
        missed_ta = stats["tamil_total"] - stats["tamil_caught"]
        log.warning(
            f"MISSED: {missed_si} Sinhala, {missed_ta} Tamil. "
            "Check unicode_verification.csv for rows where caught=False."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
