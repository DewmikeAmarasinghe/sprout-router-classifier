"""
Industry configuration — one entry per Sprout client vertical.

Adding a new industry:
    1. Add a member to IndustryKey in config/keys.py
    2. Add an IndustryConfig entry here
    3. Add IndustryBucket entries in config/distribution.py
    → It will automatically appear in the Gradio Generation tab
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.config.keys import IndustryKey


@dataclass(frozen=True)
class IndustryConfig:
    """Configuration for one Sprout client industry vertical."""

    key: IndustryKey
    display_name: str
    description: str  # injected into system prompt
    location_types: tuple[str, ...]  # how customers refer to this location
    product_examples: tuple[str, ...]  # realistic domain terms for this sector
    typical_platform: str  # primary channel for this industry


INDUSTRY_CONFIGS: dict[IndustryKey, IndustryConfig] = {
    IndustryKey.ECOMMERCE: IndustryConfig(
        key=IndustryKey.ECOMMERCE,
        display_name="E-commerce / Fashion Retail",
        description=(
            "Online fashion and retail (e.g. Thambili Island-style stores). "
            "Customers ask about product availability, sizing, cart issues, "
            "order status, returns, promotions, and cash-on-delivery."
        ),
        location_types=(
            "store",
            "outlet",
            "pickup point",
            "collection center",
            "showroom",
        ),
        product_examples=(
            "dress",
            "t-shirt",
            "sneakers",
            "handbag",
            "saree",
            "kurta",
            "promo code",
            "flash sale",
            "COD",
            "exchange",
            "track order",
            "return policy",
            "size chart",
            "free delivery",
        ),
        typical_platform="WhatsApp Business / Instagram DMs",
    ),
    IndustryKey.HEALTHCARE: IndustryConfig(
        key=IndustryKey.HEALTHCARE,
        display_name="Healthcare (Optical, Dental, Clinics)",
        description=(
            "Clinics, hospitals, optical centers and dental practices "
            "(e.g. Vision Care, Havelock Dental, AOD Medical). "
            "Customers book appointments, ask about reports, prescriptions, "
            "eye tests, dental procedures, and medical symptoms."
        ),
        location_types=(
            "clinic",
            "branch",
            "center",
            "hospital",
            "outlet",
            "dispensary",
        ),
        product_examples=(
            "eye test",
            "dental checkup",
            "lab report",
            "prescription refill",
            "appointment booking",
            "scan",
            "specialist consultation",
            "contact lenses",
            "glasses",
            "teeth whitening",
            "x-ray",
        ),
        typical_platform="WhatsApp Business / website chat widget",
    ),
    IndustryKey.BANKING: IndustryConfig(
        key=IndustryKey.BANKING,
        display_name="Retail Banking",
        description=(
            "Retail banking services. Customers ask about account balances, "
            "fund transfers, lost/stolen cards, unauthorized transactions, "
            "loan eligibility, fixed deposits, and account queries. "
            "Compliance-sensitive — formal English is common but Singlish appears."
        ),
        location_types=("branch", "ATM", "service center", "kiosk"),
        product_examples=(
            "savings account",
            "credit card",
            "fund transfer",
            "statement",
            "personal loan",
            "fixed deposit",
            "chargeback",
            "OTP verification",
            "freeze account",
            "credit limit",
            "online banking",
        ),
        typical_platform="Mobile app chat / WhatsApp Business",
    ),
    IndustryKey.INSURANCE: IndustryConfig(
        key=IndustryKey.INSURANCE,
        display_name="Insurance",
        description=(
            "Life, health, and vehicle insurance. "
            "Customers request premium quotes, submit claims, renew policies, "
            "upload documents, and ask about coverage and exclusions. "
            "Complex policy terms often require explanation."
        ),
        location_types=("branch", "service center", "agent office"),
        product_examples=(
            "premium quote",
            "claim submission",
            "policy renewal",
            "document upload",
            "motor insurance",
            "health plan",
            "coverage limit",
            "exclusion clause",
            "beneficiary update",
            "no-claim bonus",
            "cashless hospitalization",
        ),
        typical_platform="WhatsApp Business / website chat widget",
    ),
    IndustryKey.TELECOM: IndustryConfig(
        key=IndustryKey.TELECOM,
        display_name="Telecom (Mobile & Broadband)",
        description=(
            "Mobile and broadband telecom. "
            "Customers ask about data packages, plan upgrades, billing disputes, "
            "network outages, SIM replacement, and roaming services."
        ),
        location_types=("service center", "outlet", "dealer", "kiosk"),
        product_examples=(
            "data package",
            "plan upgrade",
            "bill payment",
            "SIM replacement",
            "broadband",
            "roaming",
            "network outage",
            "unlimited plan",
            "top-up",
            "data rollover",
            "family plan",
        ),
        typical_platform="WhatsApp Business / mobile app chat",
    ),
    IndustryKey.LOGISTICS: IndustryConfig(
        key=IndustryKey.LOGISTICS,
        display_name="Logistics & Courier",
        description=(
            "Courier and delivery services. "
            "Customers track shipments, schedule pickups, report delivery issues, "
            "arrange returns, and find nearest pickup points."
        ),
        location_types=(
            "depot",
            "pickup point",
            "hub",
            "service center",
            "drop-off point",
        ),
        product_examples=(
            "shipment tracking",
            "delivery status",
            "pickup schedule",
            "return request",
            "COD collection",
            "express delivery",
            "waybill number",
            "customs clearance",
            "proof of delivery",
        ),
        typical_platform="WhatsApp Business / SMS",
    ),
    IndustryKey.HOSPITALITY: IndustryConfig(
        key=IndustryKey.HOSPITALITY,
        display_name="Hospitality (Hotels, Villas, Resorts)",
        description=(
            "Hotels, boutique villas, and resorts (e.g. Mount Havana-style properties). "
            "Mix of formal English (tourists) and Singlish (local guests). "
            "Customers ask about availability, reservations, amenities, and transport."
        ),
        location_types=("property", "villa", "resort", "hotel", "location"),
        product_examples=(
            "room availability",
            "reservation",
            "check-in time",
            "pool access",
            "breakfast included",
            "early check-in",
            "airport transfer",
            "suite upgrade",
            "spa booking",
            "group booking",
        ),
        typical_platform="WhatsApp Business / website chat widget",
    ),
    IndustryKey.EDUCATION: IndustryConfig(
        key=IndustryKey.EDUCATION,
        display_name="Education (Colleges, Vocational Training)",
        description=(
            "Private colleges and vocational training institutions (e.g. AOD-style). "
            "Students and parents ask about courses, admissions, fees, "
            "intake schedules, scholarships, and campus locations."
        ),
        location_types=("campus", "study center", "branch", "learning center"),
        product_examples=(
            "course details",
            "admission requirements",
            "tuition fees",
            "academic schedule",
            "diploma program",
            "certificate course",
            "next intake date",
            "scholarship",
            "part-time option",
            "brochure",
        ),
        typical_platform="WhatsApp Business / Facebook Messenger",
    ),
}
