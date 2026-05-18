"""API routes for configuration — read-only view of current settings."""

from __future__ import annotations

from fastapi import APIRouter

from backend.shared.settings_manager import settings_manager

router = APIRouter(tags=["config"])


@router.get("/")
def get_all_settings() -> dict:
    """Return all current in-memory settings."""
    return settings_manager.get_all()
