"""
Data pipeline orchestrator.
Steps: generate → split  (evaluator removed — manual quality check instead).

Each step can be run independently or as a sequence.
"""

from __future__ import annotations

import logging

from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.generation.generator import GeneratorService
from backend.generation.splitter import DataSplitter

log = logging.getLogger(__name__)

STEPS: list[str] = ["generate", "split"]


class DataPipeline:
    """Orchestrates data preparation steps."""

    def __init__(self) -> None:
        self._generator = GeneratorService()
        self._splitter = DataSplitter()

    def run_step(self, step: str, dataset_name: str, **kwargs: object) -> dict:
        """Run one named pipeline step.

        Args:
            step: One of STEPS ("generate", "split").
            dataset_name: e.g. "v1".
            **kwargs: Forwarded to the underlying service.

        Returns:
            Dict with result stats from the step.
        """
        if step not in STEPS:
            raise ValueError(f"Unknown step {step!r}. Valid: {STEPS}")

        log.info(f"Running step '{step}' for dataset '{dataset_name}'")

        if step == "generate":
            language = kwargs.get("language")
            industry = kwargs.get("industry")
            scenario = kwargs.get("scenario")
            resume = bool(kwargs.get("resume", False))
            self._generator.run(
                dataset_name,
                language=LanguageKey(language) if language else None,
                industry=IndustryKey(industry) if industry else None,
                scenario=ScenarioKey(scenario) if scenario else None,
                resume=resume,
            )
            return {}

        if step == "split":
            return self._splitter.run(dataset_name)

        raise RuntimeError(f"Unhandled step: {step!r}")  # unreachable

    def run_all(self, dataset_name: str, resume: bool = False) -> None:
        """Run all steps in sequence."""
        log.info(f"Running full pipeline for '{dataset_name}'")
        self.run_step("generate", dataset_name, resume=resume)
        self.run_step("split", dataset_name)
        log.info(f"Pipeline complete for '{dataset_name}'")
