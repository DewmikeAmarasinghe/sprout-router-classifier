"""
Pydantic models for the router module.

ThresholdConfig: routing decision threshold.
RoutingResult:   rich result returned by RouterPredictor.predict().

All callers (routes_router.py, panel_router.py, phase_8_router.py, ablation.py)
expect RouterPredictor.predict(text) -> RoutingResult, not int.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ThresholdConfig(BaseModel):
    """Routing decision threshold configuration."""

    threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class RoutingResult(BaseModel):
    """Rich result from RouterPredictor.predict(text).

    label:          0 = gpt-4o-mini, 1 = gpt-4o
    routed_to:      human-readable model name string
    confidence:     P(label=1) from the ML model (0.0–1.0)
    routing_reason: explanation of the routing decision
    """

    label: int
    routed_to: str
    confidence: float
    routing_reason: str
