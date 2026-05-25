"""
FastAPI application entry point.

Run via uvicorn CLI (production standard):
    uvicorn backend.api.main:app --host 0.0.0.0 --port 7860

Run directly (convenience):
    python src/backend/api/main.py

    http://localhost:7860/ui    Gradio UI
    http://localhost:7860/docs  FastAPI auto-docs

Note: Do NOT use --reload when running generation — the file watcher will
interrupt active generation threads.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from frontend.app import build_app

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


from backend.api import routes_config, routes_generation, routes_router  # noqa: E402

app.include_router(routes_config.router, prefix="/api/config")
app.include_router(routes_generation.router, prefix="/api/generation")
app.include_router(routes_router.router, prefix="/api/router")

gradio_blocks = build_app()
app = gr.mount_gradio_app(app, gradio_blocks, path="/ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,  # never use reload — breaks generation threads
    )
