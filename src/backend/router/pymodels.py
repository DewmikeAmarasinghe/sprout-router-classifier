"""Pydantic models for the router module."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ThresholdConfig(BaseModel):
    """Routing decision thresholds loaded from settings."""

    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    safe_default_label: int = Field(default=1)

    @property
    def routed_model(self) -> str:
        return "gpt-4o-mini" if self.safe_default_label == 0 else "gpt-4o"


class RouterPrediction(BaseModel):
    """Result of routing one message through the three-layer system."""

    label: int  # 0 = gpt-4o-mini, 1 = gpt-4o
    confidence: float  # P(label=1) from the ML model (0.0–1.0)
    routed_to: str  # "gpt-4o-mini" or "gpt-4o"
    routing_reason: str  # "script_detector" | "model" | "below_threshold_default"

    @property
    def is_confident(self) -> bool:
        """True if the model prediction was used (vs safe-default fallback)."""
        return self.routing_reason == "model"
