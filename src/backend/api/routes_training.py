"""
API routes for training.

Exposes the same ClassicalMLTrainer and TransformerTrainer that phases/phase_5
and phases/phase_6 call directly. Both share the same backend services —
this is the HTTP entry point, phases are the CLI entry point.

Classical training: synchronous (2–10 min per experiment, acceptable for HTTP).
Transformer training: long-running (30–100 min). The route starts training and
    returns immediately with a job_id. Use GET /status/{job_id} to poll.

Routes:
    GET  /api/training/classical/results/{dataset}
    POST /api/training/classical/run
    POST /api/training/classical/run-all

    GET  /api/training/transformers/results/{dataset}
    POST /api/training/transformers/run
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["training"])

# In-memory job registry for long-running transformer training.
# Keyed by job_id → {status, result, error}.
training_jobs: dict[str, dict[str, Any]] = {}


# ── Classical ML ─────────────────────────────────────────────────────────────


class ClassicalTrainRequest(BaseModel):
    dataset: str = "v1"
    vectorizer_key: str
    classifier_key: str


@router.get("/classical/results/{dataset}")
def get_classical_results(dataset: str) -> list[dict]:
    """Return all saved classical experiment results for a dataset."""
    try:
        from backend.evaluation.comparator import load_all_results

        results = load_all_results(dataset)
        return [r.to_comparison_row() for r in results if r.approach == "classical"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/classical/run")
def run_classical_experiment(req: ClassicalTrainRequest) -> dict:
    """Train one (vectorizer, classifier) experiment synchronously."""
    try:
        from backend.training.classical.trainer import ClassicalMLTrainer

        result = ClassicalMLTrainer().train_experiment(
            req.dataset, req.vectorizer_key, req.classifier_key
        )
        return result.to_comparison_row()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/classical/run-all")
def run_all_classical(dataset: str = "v1") -> list[dict]:
    """Train all ACTIVE_COMBOS synchronously. May take 30–60 minutes."""
    try:
        from backend.training.classical.trainer import ClassicalMLTrainer

        results = ClassicalMLTrainer().train_combos(dataset)
        return [r.to_comparison_row() for r in results]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Transformer ───────────────────────────────────────────────────────────────


class TransformerTrainRequest(BaseModel):
    dataset: str = "v1"
    model_key: str
    param_overrides: dict[str, Any] | None = None


@router.get("/transformers/results/{dataset}")
def get_transformer_results(dataset: str) -> list[dict]:
    """Return all saved transformer experiment results for a dataset."""
    try:
        from backend.evaluation.comparator import load_all_results

        results = load_all_results(dataset)
        return [r.to_comparison_row() for r in results if r.approach == "transformer"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/transformers/run")
def run_transformer_experiment(req: TransformerTrainRequest) -> dict:
    """Start transformer training in a background thread. Returns job_id immediately.

    Poll GET /api/training/transformers/status/{job_id} to check progress.
    Training takes 30–100 min on CPU; use Kaggle for GPU-accelerated training.
    """
    job_id = str(uuid.uuid4())[:8]
    training_jobs[job_id] = {
        "status": "running",
        "model_key": req.model_key,
        "result": None,
        "error": None,
    }

    def run() -> None:
        try:
            from backend.training.transformers.trainer import TransformerTrainer

            result = TransformerTrainer().train_experiment(
                req.dataset, req.model_key, req.param_overrides
            )
            training_jobs[job_id]["status"] = "done"
            training_jobs[job_id]["result"] = result.to_comparison_row()
        except Exception as exc:  # noqa: BLE001
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = str(exc)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "status": "running", "model_key": req.model_key}


@router.get("/transformers/status/{job_id}")
def get_transformer_job_status(job_id: str) -> dict:
    """Poll the status of a running transformer training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return training_jobs[job_id]
