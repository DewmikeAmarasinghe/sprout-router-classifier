"""
Scenario configuration — two routing signal scenarios.

ROUTING LOGIC:
    label=0 (gpt-4o-mini): pure_english + simple_transactional
        → ALL routine queries for the industry cell, including simple named-place
          lookups like "Is the Kandy branch open Sunday?" A Sri Lankan location
          name alone does NOT trigger routing to gpt-4o.

    label=1 (gpt-4o):      location_proximity (ALL languages, including pure_english)
                           + any code-mixed language + simple_transactional.

WHY location_proximity is always label=1:
    The user needs to find the nearest/closest place, or reason about distances
    between two Sri Lankan locations. gpt-4o-mini cannot do this reliably.
    e.g. "nearest branch to me" — needs geo-spatial reasoning about Sri Lanka.

WHY simple named-place lookups stay label=0 (in pure English):
    "Is the Hikkaduwa branch open tomorrow?" — this is a direct lookup, no
    spatial reasoning needed. gpt-4o-mini handles it perfectly.
    The whole point of the dataset is to teach the model this boundary.
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
            "ALL routine queries a customer sends to a Sprout chatbot for this industry. "
            "This is the PRIMARY scenario — generate the most realistic, high-frequency "
            "messages for the language × industry combination. "
            "Examples span the full range of what that industry's customers ask: "
            "pricing, availability, hours, order/booking status, product info, "
            "appointment scheduling, account queries, policy details, delivery tracking — "
            "whatever is most natural for THIS industry. "
            "IMPORTANT: Simple named-place lookups are INCLUDED here (NOT label=1). "
            "e.g. 'Is the Hikkaduwa branch open tomorrow?' or 'What time does "
            "the Kandy Vision Care close?' — these are direct lookups, not spatial queries. "
            "A Sri Lankan place name alone does NOT make a query label=1. "
            "EXCLUDED: proximity/nearest-place queries (those go to location_proximity)."
        ),
        always_label_1=False,
        routing_reason=(
            "Pure English + any routine query (including named-place lookups) → "
            "gpt-4o-mini handles perfectly. "
            "Code-mixed version → gpt-4o-mini degrades on Singlish/Tanglish."
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
                    examples=[
                        "When does the branch open?",
                        "Track my order.",
                        "Is Kandy branch open Sunday?",
                    ],
                ),
                LengthRange(
                    min_words=9,
                    max_words=20,
                    fraction=0.39,
                    examples=[
                        "Do you have this t-shirt in XL size in blue?",
                        "What time does the Hikkaduwa outlet close on weekdays?",
                    ],
                ),
                LengthRange(
                    min_words=21,
                    max_words=40,
                    fraction=0.15,
                    examples=[
                        "Is cash on delivery available for orders above 2000 rupees?",
                        "Does the Nugegoda branch handle insurance claim submissions?",
                    ],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.LOCATION_PROXIMITY,),
    ),
    ScenarioKey.LOCATION_PROXIMITY: ScenarioConfig(
        key=ScenarioKey.LOCATION_PROXIMITY,
        display_name="Location Proximity / Spatial",
        description=(
            "User needs to find the nearest or closest place — spatial distance reasoning "
            "between the user's current location and Sri Lankan outlets/stores/clinics. "
            "This is the KEY signal that routes to gpt-4o. "
            "IMPORTANT: This is ONLY about PROXIMITY (nearest/closest). "
            "Simple named-place lookups ('Is the Kandy branch open?') are NOT this scenario. "
            "The trigger is spatial awareness: the user needs to know WHICH location is "
            "nearest/closest to them or to another Sri Lankan place. "
            "Examples: 'nearest branch to me', 'closest outlet with this product', "
            "'which branch is nearest to Colombo 3?', "
            "'I am in Galle, where is the closest Vision Care?', "
            "'this item is not in stock here — which nearby branch has it?'"
        ),
        always_label_1=True,
        routing_reason=(
            "Spatial proximity reasoning about Sri Lankan geography — "
            "gpt-4o-mini cannot reliably answer 'which branch is closest to me' "
            "or 'nearest outlet to Nugegoda'. Requires geo-spatial awareness. "
            "NOTE: A query with a Sri Lankan place name is NOT automatically label=1. "
            "Only queries requiring PROXIMITY/DISTANCE reasoning are label=1."
        ),
        length_dist=LengthDistribution(
            ranges=[
                LengthRange(
                    min_words=3,
                    max_words=10,
                    fraction=0.45,
                    examples=[
                        "Nearest outlet to me?",
                        "Closest branch?",
                        "Any store near me?",
                        "Nearest branch to Nugegoda?",
                    ],
                ),
                LengthRange(
                    min_words=11,
                    max_words=25,
                    fraction=0.40,
                    examples=[
                        "I am in Galle, what is the closest Vision Care to me?",
                        "Which branch near Colombo 3 is open on Sunday?",
                        "This item is out of stock here — nearest branch that has it?",
                    ],
                ),
                LengthRange(
                    min_words=26,
                    max_words=45,
                    fraction=0.15,
                    examples=[
                        "I am currently at One Galle Face Mall — which is the nearest branch?",
                        "The Nugegoda branch doesn't have my size — which nearby outlet does?",
                    ],
                ),
            ]
        ),
        anti_scenario_keys=(ScenarioKey.SIMPLE_TRANSACTIONAL,),
    ),
}
