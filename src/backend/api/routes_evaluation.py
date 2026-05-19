"""
API routes for the final model evaluation phase.

These routes are for FINAL EVALUATION of trained models — NOT for LLM-as-judge
(which was removed from the pipeline). They expose:

    GET  /api/evaluation/results      list all experiment results from disk
    POST /api/evaluation/compare      run ModelComparator → master_comparison.csv
    GET  /api/evaluation/cost-sim     load saved cost simulation results
    POST /api/evaluation/cost-sim     run CostSimulator against all experiments

All business logic lives in backend/evaluation/ — these routes are thin wrappers only.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.evaluation.comparator import ModelComparator
from backend.evaluation.cost_simulator import CostSimulator

router = APIRouter(tags=["evaluation"])


@router.get("/results/{dataset_name}")
def get_results(dataset_name: str) -> list[dict]:
    """Return all experiment results for a dataset, sorted by recall_1."""
    try:
        comparator = ModelComparator()
        rows = comparator.compare(dataset_name)
        return [r.model_dump() for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/results/{dataset_name}/best")
def get_best_model(dataset_name: str) -> dict:
    """Return the single best model meeting the production threshold."""
    try:
        best = ModelComparator().best_model(dataset_name)
        if not best:
            raise HTTPException(status_code=404, detail="No experiment results found")
        return best.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/cost-sim/{dataset_name}")
def get_cost_simulation(dataset_name: str, daily_messages: int = 10_000) -> list[dict]:
    """Run cost simulation across all trained models and return sorted results."""
    try:
        comparator = ModelComparator()
        rows = comparator.compare(dataset_name)
        results = CostSimulator().simulate_all(dataset_name, rows, daily_messages)
        return [r.model_dump() for r in results]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
