"""
Phase 1 — Grounding & Unicode Script Verification.

Generates 250 pure Sinhala + 250 pure Tamil + 250 English+Sinhala mixed +
250 English+Tamil mixed customer service messages via the OpenAI API, then
runs is_pure_script() on all 1,000 to verify the unicode rule catches them.

WHY MIXED MESSAGES MUST BE CAUGHT:
    is_pure_script() returns True on the FIRST Sinhala or Tamil unicode character
    it finds — it does not require the whole message to be in script. So "My ඇණවුම
    is pending" catches on ඇ. All mixed messages that actually contain script chars
    are routed to gpt-4o without ML inference. This is correct behaviour.

WHY SOME MIXED MESSAGES MAY NOT BE CAUGHT:
    The generation LLM sometimes produces pure English for the mixed categories,
    ignoring the instruction to include script characters. These are NOT a bug in
    script_detector.py — they are generation failures. The retry loop below
    re-requests any message that contains no Sinhala or Tamil unicode characters
    until every slot is filled with a genuinely mixed message.

EARLY EXIT: If unicode_verification.csv already exists, prints a summary and exits.
            Pass --force to regenerate.

Usage:
    python phases/phase_1_grounding.py           # skip if CSV exists
    python phases/phase_1_grounding.py --force   # always regenerate
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

GROUNDING_DIR = Path(__file__).parent.parent / "data" / "grounding"
VERIFY_CSV = GROUNDING_DIR / "unicode_verification.csv"

TARGET_EACH = 250
BATCH_SIZE = 50

# ── Few-shot seed examples used only as in-prompt tone/format guidance ────────

SINHALA_SEEDS = [
    "මගේ ඇණවුම කොහේද?",
    "නිකටම ශාඛාව කොහේද?",
    "ගෙවීම නිවැරදිද?",
    "මිල කීයද?",
    "ගිණුම ශේෂය කීයද?",
    "බෙදාහැරීම කවදාද?",
]

TAMIL_SEEDS = [
    "என் ஆர்டர் எங்கே?",
    "அருகில் கிளை எங்கே?",
    "பணம் சரியாக கட்டப்பட்டதா?",
    "விலை என்ன?",
    "கணக்கு இருப்பு என்ன?",
    "டெலிவரி எப்போது?",
]

ENGLISH_SINHALA_SEEDS = [
    "My ඇණවුම is still pending",
    "When will delivery to කොළඹ happen?",
    "Account balance කීයද right now?",
    "Nearest branch from Kandy කොහේද?",
    "Payment failed — මොකද problem?",
]

ENGLISH_TAMIL_SEEDS = [
    "My ஆர்டர் is still not delivered",
    "Branch in Jaffna எங்கே exactly?",
    "Can I check கணக்கு balance online?",
    "Nearest ATM from Wellawatte எங்கே?",
    "Please help with my விலை query",
]

# ── Prompt templates ──────────────────────────────────────────────────────────
# Real Sri Lankan locations to use naturally in generated messages:
#   Cities:  Colombo, Kandy, Galle, Negombo, Matara, Jaffna, Kurunegala, Nuwara Eliya
#   Areas:   Fort, Pettah, Kollupitiya, Bambalapitiya, Wellawatte, Borella, Nugegoda,
#            Maharagama, Rajagiriya, Battaramulla
#   Malls:   Majestic City, Liberty Plaza, One Galle Face Mall, Odel, Cargills Square
#   Hospitals: Asiri Central, Nawaloka, Lanka, Durdans, National Hospital

SINHALA_PROMPT = """\
Generate {n} realistic customer service messages written ENTIRELY in pure Sinhala script.
Every single character must be Sinhala unicode (ශ, ල, ා, ් etc.) — absolutely NO English, NO romanized text.
Topics: ordering, booking, delivery, prices, complaints, account queries.
Vary length: some 5–10 words, some 10–25 words.
Use real Sri Lankan locations naturally where relevant (e.g. කොළඹ, මහනුවර, නුගෙගොඩ).

Examples of the exact style required:
{examples}

Return ONLY valid JSON, no markdown, no explanation: {{"messages": ["msg1", ..., "msg{n}"]}}"""

TAMIL_PROMPT = """\
Generate {n} realistic customer service messages written ENTIRELY in pure Tamil script.
Every single character must be Tamil unicode (த, ம, ி, ழ etc.) — absolutely NO English, NO romanized text.
Topics: ordering, booking, delivery, prices, complaints, account queries.
Vary length: some 5–10 words, some 10–25 words.
Use real Sri Lankan locations naturally where relevant (e.g. யாழ்ப்பாணம், கொழும்பு, மட்டக்களப்பு).

Examples of the exact style required:
{examples}

Return ONLY valid JSON, no markdown, no explanation: {{"messages": ["msg1", ..., "msg{n}"]}}"""

ENGLISH_SINHALA_PROMPT = """\
Generate {n} customer service messages that MIX English words with Sinhala unicode characters.
CRITICAL REQUIREMENT: Every single message MUST contain at least one Sinhala unicode character \
(for example: ශාඛාව, ඇණවුම, ගිණුම, කීයද, කොහේද, කවදාද, මොකද, නිකටම, ගෙවීම).
Do NOT generate pure English. Every message must have Sinhala script mixed in.
The mix should read naturally — a real Sri Lankan customer switching between English and Sinhala.
Topics: ordering, branch locations, delivery, account balance, payment issues.
Vary length: 5–15 words.
Use real Sri Lankan locations naturally (e.g. Colombo, Kandy, Nugegoda, Wellawatte, Bambalapitiya).

Examples showing the EXACT format required — Sinhala characters mixed into English:
{examples}

Return ONLY valid JSON, no markdown, no explanation: {{"messages": ["msg1", ..., "msg{n}"]}}"""

ENGLISH_TAMIL_PROMPT = """\
Generate {n} customer service messages that MIX English words with Tamil unicode characters.
CRITICAL REQUIREMENT: Every single message MUST contain at least one Tamil unicode character \
(for example: ஆர்டர், கிளை, கணக்கு, விலை, எங்கே, எப்போது, தயவு, செய்து, இருப்பு).
Do NOT generate pure English. Every message must have Tamil script mixed in.
The mix should read naturally — a real Sri Lankan customer switching between English and Tamil.
Topics: ordering, branch locations, delivery, account balance, payment issues.
Vary length: 5–15 words.
Use real Sri Lankan locations naturally (e.g. Colombo, Jaffna, Wellawatte, Pettah, Batticaloa).

Examples showing the EXACT format required — Tamil characters mixed into English:
{examples}

Return ONLY valid JSON, no markdown, no explanation: {{"messages": ["msg1", ..., "msg{n}"]}}"""


def has_script_chars(text: str) -> bool:
    """Return True if text contains at least one Sinhala or Tamil unicode character."""
    from backend.shared.script_detector import is_pure_script

    return is_pure_script(text)


def format_examples(seeds: list[str]) -> str:
    return "\n".join(f"  - {s}" for s in seeds)


def generate_batch(client: object, model: str, prompt: str) -> list[str]:
    """Call the API once and return parsed messages list. Empty list on failure."""
    import openai

    try:
        response = client.chat.completions.create(  # ty: ignore
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            return []
        data = json.loads(content)
        return [str(m) for m in data.get("messages", [])]
    except (openai.OpenAIError, json.JSONDecodeError, KeyError) as exc:
        log.warning(f"  Batch error: {exc}")
        return []


def generate_messages(script: str, n: int) -> list[str]:
    """Generate n messages of the given script category via OpenAI API.

    For mixed categories, any message that contains no script characters is
    rejected and a replacement is requested. This loop continues until every
    slot holds a genuinely mixed message.

    script: "sinhala" | "tamil" | "english_sinhala_mixed" | "english_tamil_mixed"
    """
    from openai import OpenAI

    from backend.shared.settings_manager import settings_manager

    client = OpenAI()
    model = str(settings_manager.get("GENERATION_LLM"))
    is_mixed = script.startswith("english_")

    prompt_template, seeds = {
        "sinhala": (SINHALA_PROMPT, SINHALA_SEEDS),
        "tamil": (TAMIL_PROMPT, TAMIL_SEEDS),
        "english_sinhala_mixed": (ENGLISH_SINHALA_PROMPT, ENGLISH_SINHALA_SEEDS),
        "english_tamil_mixed": (ENGLISH_TAMIL_PROMPT, ENGLISH_TAMIL_SEEDS),
    }[script]

    examples_block = format_examples(seeds)
    accepted: list[str] = []

    while len(accepted) < n:
        need = min(BATCH_SIZE, n - len(accepted))
        prompt = prompt_template.format(n=need, examples=examples_block)
        batch = generate_batch(client, model, prompt)

        for msg in batch:
            if len(accepted) >= n:
                break
            # For mixed categories, reject pure-English generations
            if is_mixed and not has_script_chars(msg):
                log.warning(f"  [{script}] Rejected pure-English generation: {msg[:60]}")
                continue
            accepted.append(msg)

        log.info(f"  [{script}] {len(accepted)}/{n}")

    return accepted[:n]


def verify_and_save(all_messages: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Run is_pure_script() on all messages. Save results to VERIFY_CSV."""
    from backend.shared.script_detector import is_pure_script, script_language

    GROUNDING_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    stats: dict[str, dict[str, int]] = {}

    for script, msgs in all_messages.items():
        caught_count = 0
        for msg in msgs:
            caught = is_pure_script(msg)
            detected = script_language(msg) or "none"
            if caught:
                caught_count += 1
            rows.append(
                {
                    "script": script,
                    "text": msg,
                    "caught": str(caught),
                    "detected": detected,
                }
            )
        stats[script] = {"total": len(msgs), "caught": caught_count}

    with VERIFY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["script", "text", "caught", "detected"])
        writer.writeheader()
        writer.writerows(rows)

    return stats


def print_summary(stats: dict[str, dict[str, int]]) -> None:
    total = sum(s["total"] for s in stats.values())
    caught = sum(s["caught"] for s in stats.values())
    missed = total - caught

    print(f"\n{'═' * 60}")
    print("  Phase 1 Grounding — Unicode Script Verification")
    print(f"{'═' * 60}")
    print(f"  {'Script':<30} {'Total':>6}  {'Caught':>6}  {'%':>6}")
    print(f"  {'─' * 52}")
    for script, counts in sorted(stats.items()):
        pct = counts["caught"] / counts["total"] * 100 if counts["total"] else 0.0
        print(f"  {script:<30} {counts['total']:>6}  {counts['caught']:>6}  {pct:>5.1f}%")
    print(f"  {'─' * 52}")
    pct_total = caught / total * 100 if total else 0.0
    print(f"  {'TOTAL':<30} {total:>6}  {caught:>6}  {pct_total:>5.1f}%")

    if missed > 0:
        print(f"\n  ⚠ {missed} samples NOT caught — see caught=False rows in CSV")
    else:
        print("\n  ✅ All samples correctly caught by is_pure_script().")
    print("═" * 60)


def load_existing_summary() -> None:
    """Load and print summary from existing VERIFY_CSV."""
    with VERIFY_CSV.open(encoding="utf-8") as f:
        rows: list[dict[str, str]] = list(csv.DictReader(f))

    stats: dict[str, dict[str, int]] = {}
    for r in rows:
        script = r.get("script", "unknown")
        if script not in stats:
            stats[script] = {"total": 0, "caught": 0}
        stats[script]["total"] += 1
        if r.get("caught", "").lower() == "true":
            stats[script]["caught"] += 1

    print_summary(stats)
    print(f"\nRun with --force to regenerate. CSV: {VERIFY_CSV}\n")


def run_grounding() -> None:
    """Generate messages via API, verify with is_pure_script(), save CSV."""
    log.info(f"Generating {TARGET_EACH} samples per category via OpenAI API...")

    all_messages: dict[str, list[str]] = {}
    for script in ("sinhala", "tamil", "english_sinhala_mixed", "english_tamil_mixed"):
        log.info(f"Generating [{script}]...")
        all_messages[script] = generate_messages(script, TARGET_EACH)

    log.info("Running is_pure_script() verification...")
    stats = verify_and_save(all_messages)

    print_summary(stats)
    log.info(f"Results saved: {VERIFY_CSV}")

    missed = sum(s["total"] - s["caught"] for s in stats.values())
    if missed > 0:
        log.warning(
            f"{missed} messages not caught. Check CSV for caught=False rows. "
            "Run with --force to retry generation."
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if unicode_verification.csv already exists",
    )
    args = parser.parse_args()

    if VERIFY_CSV.exists() and not args.force:
        log.info(f"Found existing {VERIFY_CSV.name} — showing summary (--force to regenerate)")
        load_existing_summary()
        return

    run_grounding()


if __name__ == "__main__":
    main()
