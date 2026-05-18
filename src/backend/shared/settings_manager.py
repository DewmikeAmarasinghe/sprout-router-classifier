"""
SettingsManager — in-memory settings wrapper.

Loads all uppercase constants from config/settings.py at startup.
Provides get/set for runtime overrides.

DELIBERATELY HAS NO save() METHOD:
    Settings are in-memory only. Changes made via set() exist only for the
    current server session and are forgotten on restart.
    To change a permanent default: edit config/settings.py directly.

Usage:
    from backend.shared.settings_manager import settings_manager
    model = settings_manager.get("GENERATION_LLM")
    settings_manager.set("GENERATION_BATCH_SIZE", 30)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


def _load_defaults() -> dict[str, Any]:
    """Import all uppercase constants from config/settings.py."""
    from backend.config import settings as _s

    return {key: getattr(_s, key) for key in dir(_s) if key.isupper() and not key.startswith("__")}


class SettingsManager:
    """Singleton. In-memory settings store loaded from config/settings.py."""

    _instance: SettingsManager | None = None

    def __new__(cls) -> SettingsManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return
        self._store: dict[str, Any] = _load_defaults()
        self._listeners: list[Callable[[str, Any], None]] = []
        self._ready = True

    def get(self, key: str) -> Any:
        """Return current value. Raises KeyError if unknown."""
        if key not in self._store:
            raise KeyError(f"Unknown setting: {key!r}. Available: {sorted(self._store)}")
        return self._store[key]

    def set(self, key: str, value: Any) -> None:
        """Update key in memory for this session."""
        if key not in self._store:
            raise KeyError(f"Unknown setting: {key!r}")
        self._store[key] = value
        for cb in self._listeners:
            try:
                cb(key, value)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Settings subscriber error: {exc}")

    def get_all(self) -> dict[str, Any]:
        """Return a copy of all current settings."""
        return dict(self._store)

    def subscribe(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback that fires when any setting changes."""
        self._listeners.append(callback)


settings_manager = SettingsManager()
