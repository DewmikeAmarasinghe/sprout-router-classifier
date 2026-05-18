"""
FastAPI application entry point.

Run:
    uvicorn backend.api.main:app --host 0.0.0.0 --port 7860 --reload

    http://localhost:7860/ui    Gradio UI
    http://localhost:7860/docs  FastAPI auto-docs
"""

from __future__ import annotations

# Load .env FIRST — before any backend import that might touch OpenAI
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sprout Router Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


from backend.api import routes_config, routes_generation  # noqa: E402

app.include_router(routes_config.router, prefix="/api/config")
app.include_router(routes_generation.router, prefix="/api/generation")


def _build_gradio() -> gr.Blocks:
    from frontend.app import build_app

    return build_app()


gradio_blocks = _build_gradio()
app = gr.mount_gradio_app(app, gradio_blocks, path="/ui")
