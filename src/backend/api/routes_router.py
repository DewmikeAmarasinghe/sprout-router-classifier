"""
API routes for the router module.

Exposes RouterPredictor via HTTP for integration testing and Sprout backend calls.
In production, Sprout calls POST /api/router/predict before every LLM API call.

Routes:
    POST /api/router/predict          route one message
    POST /api/router/predict-batch    route a list of messages
    GET  /api/router/config           return current threshold config
    GET  /api/router/status           check if a trained model is configured
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["router"])


class PredictRequest(BaseModel):
    text: str
    dataset: str = "v1"
    model_key: str | None = None
    threshold: float | None = None


class PredictBatchRequest(BaseModel):
    texts: list[str]
    dataset: str = "v1"
    model_key: str | None = None
    threshold: float | None = None


def load_predictor(dataset: str, model_key: str | None):
    """Load the best available predictor for a dataset.

    Path is constructed as a directory (experiment_id is the folder name,
    model.pkl lives inside). RouterPredictor.from_pkl handles this via
    the candidate-chain: path/model.pkl is tried first.
    """
    from backend.evaluation.comparator import ModelComparator
    from backend.router.predictor import RouterPredictor
    from backend.shared.path_resolver import get_experiment_path

    comparator = ModelComparator()
    best = comparator.best_model(dataset)

    if not best:
        raise ValueError(f"No trained models found for dataset '{dataset}'")

    experiment_id = model_key or best.experiment_id

    if best.approach == "classical":
        path = get_experiment_path(dataset, "classical") / "models" / experiment_id
        predictor = RouterPredictor.from_pkl(path)
    else:
        path = get_experiment_path(dataset, "transformers") / "models" / experiment_id
        predictor = RouterPredictor.from_hf_checkpoint(path)

    return predictor


@router.post("/predict")
def predict(req: PredictRequest) -> dict:
    """Route one message through the ML model."""
    try:
        from backend.shared.settings_manager import settings_manager

        predictor = load_predictor(req.dataset, req.model_key)
        threshold = req.threshold or float(settings_manager.get("CONFIDENCE_THRESHOLD"))
        predictor.set_threshold(threshold)
        result = predictor.predict(req.text)
        return result.model_dump()

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predict-batch")
def predict_batch(req: PredictBatchRequest) -> list[dict]:
    """Route a list of messages."""
    try:
        from backend.shared.settings_manager import settings_manager

        predictor = load_predictor(req.dataset, req.model_key)
        threshold = req.threshold or float(settings_manager.get("CONFIDENCE_THRESHOLD"))
        predictor.set_threshold(threshold)
        return [predictor.predict(t).model_dump() for t in req.texts]

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/config")
def get_config() -> dict:
    from backend.shared.settings_manager import settings_manager

    return {
        "confidence_threshold": settings_manager.get("CONFIDENCE_THRESHOLD"),
        "safe_default_label": settings_manager.get("SAFE_DEFAULT_LABEL"),
        "routed_to_on_default": (
            "gpt-4o" if settings_manager.get("SAFE_DEFAULT_LABEL") == 1 else "gpt-4o-mini"
        ),
    }


@router.get("/status")
def get_status(dataset: str = "v1") -> dict:
    try:
        from backend.evaluation.comparator import ModelComparator

        best = ModelComparator().best_model(dataset)
        if best:
            return {
                "ready": True,
                "best_model": best.experiment_id,
                "recall_1": best.recall_1,
                "passes_threshold": best.passes_production_threshold,
            }
        return {"ready": False, "message": "No trained models found"}
    except Exception as exc:
        return {"ready": False, "message": str(exc)}
