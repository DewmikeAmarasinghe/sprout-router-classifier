"""
Scenario configuration — one entry per routing signal.

CONTINUATION description fix:
    Removed "puriyala" and "ada" from the scenario description text.
    These are Sinhala/Tamil words and should never appear in pure_english prompts.
    The PromptFactory.build_continuation_subtype() handles language-aware examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.config.keys import ScenarioKey
from backend.generation.pymodels import LengthDistribution, LengthRange


@dataclass(frozen=True)
class ScenarioConfig:
    key: ScenarioKey
    display_name: str
    description: str
    always_label_1: bool
    routing_reason: str
    length_dist: LengthDistribution
    anti_scenario_keys: tuple[ScenarioKey, ...] = field(default_factory=tuple)


SCENARIO_CONFIGS: dict[ScenarioKey, ScenarioConfig] = {
    ScenarioKey.SIMPLE_TRANSACTIONAL: ScenarioConfig(
        key=ScenarioKey.SIMPLE_TRANSACTIONAL,
        display_name="Simple Transactional",
        description=(
            "Routine query with a single clear intent: pricing, availability, "
            "hours, order status, basic product info. No complexity signals. "
            "Includes ultra-short opening messages ('hi', 'price?', 'status?')."
        ),
        always_label_1=False,
        routing_reason=(
            "Pure English + simple intent → gpt-4o-mini handles perfectly. "
            "Code-mixed version → gpt-4o-mini degrades."
        ),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=1,
                    max_words=2,
                    fraction=0.10,
                    examples=["hi", "hello", "price?", "status?", "delivery?"],
                ),
                LengthRange(
                    min_words=3,
                    max_words=8,
                    fraction=0.36,
                    examples=["When does the branch open?", "Track my order."],
                ),
                LengthRange(
                    min_words=9,
                    max_words=20,
                    fraction=0.39,
                    examples=["Do you have this t-shirt in XL size in blue?"],
                ),
                LengthRange(
                    min_words=21,
                    max_words=40,
                    fraction=0.15,
                    examples=["Is cash on delivery available for orders above 2000 rupees?"],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.LOCATION_PROXIMITY, ScenarioKey.CONTINUATION),
    ),
    ScenarioKey.NAMED_LOCATION: ScenarioConfig(
        key=ScenarioKey.NAMED_LOCATION,
        display_name="Named Location Lookup",
        description=(
            "User mentions a specific named Sri Lankan location for a simple lookup: "
            "hours, address, availability, contact. No distance reasoning required."
        ),
        always_label_1=False,
        routing_reason=(
            "Named location + simple lookup → gpt-4o-mini handles fine. "
            "Contrast: 'From Kandy which branch?' → LOCATION_RELATIVE (spatial reasoning)."
        ),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=3,
                    max_words=10,
                    fraction=0.50,
                    examples=["Nugegoda branch hours?", "Fort outlet open Sunday?"],
                ),
                LengthRange(
                    min_words=11,
                    max_words=25,
                    fraction=0.40,
                    examples=["What time does the Kandy Vision Care close on weekdays?"],
                ),
                LengthRange(
                    min_words=26,
                    max_words=45,
                    fraction=0.10,
                    examples=["Does the Battaramulla branch handle insurance claim submissions?"],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.LOCATION_PROXIMITY, ScenarioKey.LOCATION_RELATIVE),
    ),
    ScenarioKey.LOCATION_PROXIMITY: ScenarioConfig(
        key=ScenarioKey.LOCATION_PROXIMITY,
        display_name="Location Proximity (nearest/closest)",
        description=(
            "User wants the nearest or closest location. "
            "The signal is proximity INTENT — always requires spatial reasoning."
        ),
        always_label_1=True,
        routing_reason="Spatial reasoning required. gpt-4o-mini cannot do this reliably.",
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=3,
                    max_words=12,
                    fraction=0.55,
                    examples=["Nearest branch to me?", "Closest ATM koheda?"],
                ),
                LengthRange(
                    min_words=13,
                    max_words=30,
                    fraction=0.35,
                    examples=["Which clinic is closest to my area? Need to visit today."],
                ),
                LengthRange(
                    min_words=31,
                    max_words=65,
                    fraction=0.10,
                    examples=["Near Maharagama and need a branch urgently — which one?"],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.NAMED_LOCATION,),
    ),
    ScenarioKey.LOCATION_RELATIVE: ScenarioConfig(
        key=ScenarioKey.LOCATION_RELATIVE,
        display_name="Location Relative (spatial relationship)",
        description=(
            "User needs to understand spatial relationships between places. "
            "Reference points can be schools, junctions, roads, or landmarks — "
            "not just business locations."
        ),
        always_label_1=True,
        routing_reason="Requires reasoning about spatial relationships between named places.",
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=5,
                    max_words=20,
                    fraction=0.30,
                    examples=["From Kandy which branch?", "Nugey or Fort — closer?"],
                ),
                LengthRange(
                    min_words=21,
                    max_words=55,
                    fraction=0.50,
                    examples=[
                        "Coming from Maharagama — Nugegoda or Borella branch, which is closer?"
                    ],
                ),
                LengthRange(
                    min_words=56,
                    max_words=120,
                    fraction=0.20,
                    examples=[
                        "Does mahanuwara branch have this service, or is kurunagala "
                        "branch the right one — which is closer from my location?"
                    ],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.NAMED_LOCATION,),
    ),
    ScenarioKey.COMPLEX_TASK: ScenarioConfig(
        key=ScenarioKey.COMPLEX_TASK,
        display_name="Complex Multi-Step Task",
        description=(
            "Multi-condition or multi-step query: policy comparisons, eligibility "
            "with several variables, upgrade + pending state interactions."
        ),
        always_label_1=True,
        routing_reason="gpt-4o-mini degrades on multi-branch reasoning and policy comparisons.",
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=10,
                    max_words=40,
                    fraction=0.25,
                    examples=["Upgrade mid-cycle with pending claim — what happens to billing?"],
                ),
                LengthRange(
                    min_words=41,
                    max_words=80,
                    fraction=0.50,
                    examples=[
                        "Compare silver and gold health plans for a family of 4 with one over 60."
                    ],
                ),
                LengthRange(
                    min_words=81,
                    max_words=150,
                    fraction=0.25,
                    examples=[
                        "I have plan X and my spouse is dependent — if she claims beyond "
                        "her sublimit does it fall under my primary automatically?"
                    ],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL,),
    ),
    ScenarioKey.SENSITIVE_CONTEXT: ScenarioConfig(
        key=ScenarioKey.SENSITIVE_CONTEXT,
        display_name="Sensitive Context (fraud, medical, distress)",
        description=(
            "Unauthorized transactions, card fraud, medical symptoms, financial distress, "
            "account compromise. Requires empathetic tone."
        ),
        always_label_1=True,
        routing_reason="Sensitive situations need careful empathetic handling — gpt-4o-mini tone is flat.",
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=3,
                    max_words=18,
                    fraction=0.35,
                    examples=["Card used without permission.", "Something wrong with my vision."],
                ),
                LengthRange(
                    min_words=19,
                    max_words=50,
                    fraction=0.45,
                    examples=["Transfer I didn't authorize — need to stop it immediately."],
                ),
                LengthRange(
                    min_words=51,
                    max_words=100,
                    fraction=0.20,
                    examples=[
                        "Got OTP I didn't request and there's a pending transfer "
                        "to a number I don't recognize — what do I do?"
                    ],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL, ScenarioKey.ESCALATION),
    ),
    ScenarioKey.ESCALATION: ScenarioConfig(
        key=ScenarioKey.ESCALATION,
        display_name="Escalation (complaint, frustration, manager request)",
        description=(
            "Frustration or demand for escalation. "
            "Use natural, realistic customer language — not formal business vocabulary. "
            "Real customers say 'this is ridiculous', not 'this process exhibits inconsistency'."
        ),
        always_label_1=True,
        routing_reason="De-escalation requires emotional intelligence beyond gpt-4o-mini.",
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=3,
                    max_words=14,
                    fraction=0.30,
                    examples=["This is unacceptable!", "I want to speak to a manager NOW."],
                ),
                LengthRange(
                    min_words=15,
                    max_words=45,
                    fraction=0.50,
                    examples=["Called three times, every agent gives a different answer."],
                ),
                LengthRange(
                    min_words=46,
                    max_words=90,
                    fraction=0.20,
                    examples=[
                        "Agent told me claim would be processed in 48 hours — "
                        "it has been 6 days and no one can give me a straight answer."
                    ],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL, ScenarioKey.SENSITIVE_CONTEXT),
    ),
    ScenarioKey.RESPONSE_LANGUAGE: ScenarioConfig(
        key=ScenarioKey.RESPONSE_LANGUAGE,
        display_name="Response Language Request",
        description=(
            "User asks the chatbot to reply in Sinhala, Tamil, or another non-English language. "
            "Can be standalone or appended to another query."
        ),
        always_label_1=True,
        routing_reason="gpt-4o generates significantly higher quality non-English responses.",
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=2,
                    max_words=10,
                    fraction=0.55,
                    examples=["Sinhala la kiyanna.", "Reply in tamil please."],
                ),
                LengthRange(
                    min_words=11,
                    max_words=30,
                    fraction=0.35,
                    examples=["Can you explain the return policy in sinhala please?"],
                ),
                LengthRange(
                    min_words=31,
                    max_words=65,
                    fraction=0.10,
                    examples=["Walk me through the exclusions in sinhala? English is confusing."],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL,),
    ),
    ScenarioKey.CONTINUATION: ScenarioConfig(
        key=ScenarioKey.CONTINUATION,
        display_name="Continuation (failed action OR unclear intent)",
        description=(
            "Two merged sub-types: "
            "(A) ~55% Previous chatbot action FAILED — user reports it keeps failing. "
            "(B) ~45% Intent unclear OR clarification request — very short or asking to re-explain."
        ),
        always_label_1=True,
        routing_reason=(
            "A: Cheaper model already failed — escalate. "
            "B: Unclear intent needs careful contextual reasoning."
        ),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=1,
                    max_words=6,
                    fraction=0.45,
                    examples=["still error", "not working", "yes", "ok", "what?"],
                ),
                LengthRange(
                    min_words=7,
                    max_words=20,
                    fraction=0.35,
                    examples=["Tried again, shows same error.", "Didn't get that, explain again?"],
                ),
                LengthRange(
                    min_words=21,
                    max_words=60,
                    fraction=0.20,
                    examples=["Tried your steps three times and it keeps failing — system issue?"],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL,),
    ),
}
