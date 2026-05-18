"""
Scenario configuration definitions.

ScenarioConfig is a frozen dataclass — it is a trusted internal definition, not
user-supplied data, so Pydantic validation overhead is unnecessary. LengthDistribution and
LengthRange ARE Pydantic (defined in generation/pymodels.py) because they contain fraction
validation logic.

Adding a new scenario:
    1. Add a member to ScenarioKey in config/keys.py
    2. Add a ScenarioConfig entry to SCENARIO_CONFIGS here
    3. Add a SectionBuilder subclass in generation/prompt_factory.py
    4. Register it in PromptFactory.SECTION_BUILDERS
    5. Add ScenarioBucket entries in config/distribution.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.config.keys import ScenarioKey
from backend.generation.pymodels import LengthDistribution, LengthRange


@dataclass(frozen=True)
class ScenarioConfig:
    """Domain knowledge for one scenario. No prompt templates here."""

    key: ScenarioKey
    display_name: str
    description: str
    always_label_1: bool
    routing_reason: str
    length_dist: LengthDistribution  # required — every scenario must define lengths
    anti_scenario_keys: tuple[ScenarioKey, ...] = field(default_factory=tuple)


SCENARIO_CONFIGS: dict[ScenarioKey, ScenarioConfig] = {
    ScenarioKey.SIMPLE_TRANSACTIONAL: ScenarioConfig(
        key=ScenarioKey.SIMPLE_TRANSACTIONAL,
        display_name="Simple Transactional",
        description=(
            "Routine query with a single clear intent: pricing, availability, hours, "
            "order status, basic product info. No complexity signals."
        ),
        always_label_1=False,
        routing_reason=(
            "Pure English + simple intent → gpt-4o-mini handles perfectly. "
            "Singlish/Tanglish version → gpt-4o-mini degrades on code-mixed text."
        ),
        anti_scenario_keys=(ScenarioKey.LOCATION_PROXIMITY, ScenarioKey.CONTINUATION),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=3,
                    max_words=8,
                    fraction=0.40,
                    examples=["When does the branch open?", "Price?"],
                ),
                LengthRange(
                    min_words=8,
                    max_words=20,
                    fraction=0.45,
                    examples=["Do you have this t-shirt in XL size in blue?"],
                ),
                LengthRange(
                    min_words=20,
                    max_words=40,
                    fraction=0.15,
                    examples=[
                        "I want to know if cash on delivery is available for orders above 2000 rupees."
                    ],
                ),
            ]
        ),
    ),
    ScenarioKey.NAMED_LOCATION: ScenarioConfig(
        key=ScenarioKey.NAMED_LOCATION,
        display_name="Named Location Lookup",
        description=(
            "User mentions a specific named Sri Lankan location for a simple factual lookup: "
            "hours, address, availability, contact. Does NOT require distance or routing reasoning."
        ),
        always_label_1=False,
        routing_reason=(
            "Named location + simple lookup → gpt-4o-mini handles fine. "
            "Contrast: 'From Kandy which branch is closer?' → LOCATION_RELATIVE."
        ),
        anti_scenario_keys=(
            ScenarioKey.LOCATION_PROXIMITY,
            ScenarioKey.LOCATION_RELATIVE,
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
                    min_words=10,
                    max_words=25,
                    fraction=0.40,
                    examples=["What time does the Kandy Vision Care close on weekdays?"],
                ),
                LengthRange(
                    min_words=25,
                    max_words=45,
                    fraction=0.10,
                    examples=[
                        "Does the Battaramulla branch handle insurance renewals or only the main office?"
                    ],
                ),
            ]
        ),
    ),
    ScenarioKey.LOCATION_PROXIMITY: ScenarioConfig(
        key=ScenarioKey.LOCATION_PROXIMITY,
        display_name="Location Proximity (nearest/closest)",
        description=(
            "User wants the nearest or closest location. Named locations MAY appear — "
            "the routing signal is proximity INTENT, not presence of a place name. "
            "Requires spatial reasoning."
        ),
        always_label_1=True,
        routing_reason="Needs spatial reasoning about distances. gpt-4o-mini cannot do this reliably.",
        anti_scenario_keys=(ScenarioKey.NAMED_LOCATION,),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=4,
                    max_words=12,
                    fraction=0.55,
                    examples=[
                        "Nearest branch to me?",
                        "mata lagin thiyen outlet eka koheda?",
                    ],
                ),
                LengthRange(
                    min_words=12,
                    max_words=30,
                    fraction=0.35,
                    examples=["Which clinic is closest to my area? I need to visit today."],
                ),
                LengthRange(
                    min_words=30,
                    max_words=65,
                    fraction=0.10,
                    examples=[
                        "I'm near Maharagama and need to visit a branch urgently — which one should I go to?"
                    ],
                ),
            ]
        ),
    ),
    ScenarioKey.LOCATION_RELATIVE: ScenarioConfig(
        key=ScenarioKey.LOCATION_RELATIVE,
        display_name="Location Relative (spatial relationship)",
        description=(
            "User needs to understand the spatial relationship between places. "
            "Reference points are NOT limited to business locations: "
            "schools, landmarks, junctions, roads ('near Dharmapala Vidyalaya', "
            "'100m from the Cargills', 'past the Galle road junction')."
        ),
        always_label_1=True,
        routing_reason="Needs to reason about spatial relationships between named places or landmarks.",
        anti_scenario_keys=(ScenarioKey.NAMED_LOCATION,),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=8,
                    max_words=20,
                    fraction=0.30,
                    examples=[
                        "From Kandy which branch is closer?",
                        "Nugey or Fort — which is easier?",
                    ],
                ),
                LengthRange(
                    min_words=20,
                    max_words=55,
                    fraction=0.50,
                    examples=[
                        "Coming from Maharagama — is Nugegoda branch or Borella branch closer?"
                    ],
                ),
                LengthRange(
                    min_words=55,
                    max_words=120,
                    fraction=0.20,
                    examples=[
                        "Does mahanuwara branch have this service, otherwise is the kurunagala one the right one — which is closer from my location?"
                    ],
                ),
            ]
        ),
    ),
    ScenarioKey.COMPLEX_TASK: ScenarioConfig(
        key=ScenarioKey.COMPLEX_TASK,
        display_name="Complex Multi-Step Task",
        description=(
            "Multi-condition or multi-step query: policy comparisons, eligibility with "
            "several variables, upgrade + pending state interactions."
        ),
        always_label_1=True,
        routing_reason="gpt-4o-mini degrades visibly on multi-branch reasoning and policy comparisons.",
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL,),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=15,
                    max_words=40,
                    fraction=0.25,
                    examples=[
                        "If I upgrade mid-cycle and have a pending claim, what happens to billing?"
                    ],
                ),
                LengthRange(
                    min_words=40,
                    max_words=80,
                    fraction=0.50,
                    examples=[
                        "I want to compare silver and gold health plans for a family of 4 with one member over 60."
                    ],
                ),
                LengthRange(
                    min_words=80,
                    max_words=150,
                    fraction=0.25,
                    examples=[
                        "I have plan X and my spouse is dependent — if she claims beyond her sublimit does it fall under my primary automatically?"
                    ],
                ),
            ]
        ),
    ),
    ScenarioKey.SENSITIVE_CONTEXT: ScenarioConfig(
        key=ScenarioKey.SENSITIVE_CONTEXT,
        display_name="Sensitive Context (fraud, medical, distress)",
        description=(
            "Any message involving: unauthorized transactions, card fraud, "
            "medical symptoms or health urgency, financial distress, account compromise. "
            "Even simple English needs gpt-4o for appropriate tone."
        ),
        always_label_1=True,
        routing_reason="Sensitive situations require careful empathetic handling. gpt-4o-mini tone is inadequate.",
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL, ScenarioKey.ESCALATION),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=5,
                    max_words=18,
                    fraction=0.35,
                    examples=[
                        "My card was used without permission.",
                        "I think something is wrong with my vision.",
                    ],
                ),
                LengthRange(
                    min_words=18,
                    max_words=50,
                    fraction=0.45,
                    examples=[
                        "Someone made a transfer I didn't authorize — I need to stop it immediately."
                    ],
                ),
                LengthRange(
                    min_words=50,
                    max_words=100,
                    fraction=0.20,
                    examples=[
                        "I received an OTP I didn't request and there's a pending transfer to a number I don't recognize — what do I do?"
                    ],
                ),
            ]
        ),
    ),
    ScenarioKey.ESCALATION: ScenarioConfig(
        key=ScenarioKey.ESCALATION,
        display_name="Escalation (complaint, frustration, manager request)",
        description=(
            "Customer expressing frustration or demanding escalation: "
            "unacceptable service, long waits, wrong information, "
            "threats to leave, requests for supervisor or manager."
        ),
        always_label_1=True,
        routing_reason="De-escalation requires emotional intelligence beyond gpt-4o-mini.",
        anti_scenario_keys=(
            ScenarioKey.SIMPLE_TRANSACTIONAL,
            ScenarioKey.SENSITIVE_CONTEXT,
        ),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=3,
                    max_words=14,
                    fraction=0.30,
                    examples=[
                        "This is unacceptable!",
                        "I want to speak to a manager NOW.",
                    ],
                ),
                LengthRange(
                    min_words=14,
                    max_words=45,
                    fraction=0.50,
                    examples=[
                        "I've called three times and every agent gives a different answer — wasting my time."
                    ],
                ),
                LengthRange(
                    min_words=45,
                    max_words=90,
                    fraction=0.20,
                    examples=[
                        "Your agent told me my claim would be processed in 48 hours — it has been 6 days and no one can give me a straight answer."
                    ],
                ),
            ]
        ),
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
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL,),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=3,
                    max_words=10,
                    fraction=0.55,
                    examples=["Sinhala la kiyanna.", "Reply in tamil please."],
                ),
                LengthRange(
                    min_words=10,
                    max_words=30,
                    fraction=0.35,
                    examples=["Can you explain the return policy in sinhala? It's easier for me."],
                ),
                LengthRange(
                    min_words=30,
                    max_words=65,
                    fraction=0.10,
                    examples=[
                        "Could you walk me through the exclusions in sinhala? The English is confusing me."
                    ],
                ),
            ]
        ),
    ),
    ScenarioKey.CONTINUATION: ScenarioConfig(
        key=ScenarioKey.CONTINUATION,
        display_name="Continuation (failed action OR unclear intent)",
        description=(
            "Two merged sub-types: "
            "(A) Previous chatbot action FAILED — user reports failure and asks again. "
            "'it still shows error', 'tried again same problem'. "
            "(B) Very short unclear message OR clarification request — "
            "'I didn't get that', 'puriyala', 'can you explain again in simpler terms'."
        ),
        always_label_1=True,
        routing_reason=(
            "A: Cheaper model already failed — escalate. "
            "B: Unclear intent or clarification needs careful contextual reasoning."
        ),
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL,),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=1,
                    max_words=6,
                    fraction=0.45,
                    examples=[
                        "still error",
                        "not working",
                        "ada",
                        "puriyala",
                        "again sollu",
                    ],
                ),
                LengthRange(
                    min_words=6,
                    max_words=20,
                    fraction=0.35,
                    examples=[
                        "I tried again and it shows the same error.",
                        "I didn't get that, can you explain again?",
                    ],
                ),
                LengthRange(
                    min_words=20,
                    max_words=60,
                    fraction=0.20,
                    examples=[
                        "I've tried the steps you gave me three times and it keeps failing — is there something wrong on your end?"
                    ],
                ),
            ]
        ),
    ),
}
