"""API routes for generation pipeline. Calls backend services only."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.config.distribution import DISTRIBUTION
from backend.config.keys import IndustryKey, LanguageKey, ScenarioKey
from backend.generation.prompt_factory import PROMPT_FACTORY

router = APIRouter(tags=["generation"])


@router.get("/distribution/summary")
def distribution_summary() -> dict[str, int]:
    """Return computed row counts per language."""
    return DISTRIBUTION.summary()


@router.get("/distribution/cells")
def distribution_cells() -> list[dict]:
    """Return all cells with computed counts (including zeros)."""
    return [
        {
            "language": c.language,
            "industry": c.industry,
            "scenario": c.scenario,
            "target_count": c.target_count,
            "cell_id": c.cell_id,
        }
        for c in DISTRIBUTION.all_cells_including_zero()
    ]


class PromptPreviewRequest(BaseModel):
    language: LanguageKey
    industry: IndustryKey
    scenario: ScenarioKey


@router.post("/prompt/preview")
def preview_prompt(req: PromptPreviewRequest) -> dict[str, str]:
    """Return the system prompt for a given (language, industry, scenario) cell."""
    prompt = PROMPT_FACTORY.build_preview(req.language, req.industry, req.scenario)
    cell_id = f"{req.language}__{req.industry}__{req.scenario}"
    return {"prompt": prompt, "cell_id": cell_id}
