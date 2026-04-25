"""
Condition Monitoring System — FastAPI application entry point.
Provides REST API for vibration signal analysis, fault classification,
and antigravity regime detection.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routers import analysis, models as models_router
from services.inference import InferenceService

load_dotenv()

# ── Startup / Shutdown lifecycle ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, release on shutdown."""
    logger.info("Starting Condition Monitoring System API …")
    antigravity_enabled = os.getenv("ANTIGRAVITY_ENABLED", "false").lower() == "true"
    app.state.inference = InferenceService(antigravity_enabled=antigravity_enabled)
    app.state.inference.load_models()
    logger.info(
        f"Models loaded | antigravity_enabled={antigravity_enabled}"
    )
    yield
    logger.info("Shutting down — releasing model resources.")


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Condition Monitoring System API",
        description=(
            "AI-powered vibration analysis and fault classification for "
            "rotating machinery using STWIN.box MEMS sensors. "
            "Supports a 7-class fault classifier including the experimental "
            "Antigravity regime detection layer."
        ),
        version="1.1.0-antigravity",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    cors_origins = [o.strip() for o in cors_origins_raw.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
    app.include_router(models_router.router, prefix="/api/models", tags=["Models"])

    @app.get("/", tags=["Health"])
    async def root() -> dict:
        """Health-check endpoint."""
        return {
            "status": "ok",
            "service": "Condition Monitoring System",
            "version": "1.1.0-antigravity",
        }

    @app.get("/health", tags=["Health"])
    async def health() -> dict:
        """Detailed health check including model load status."""
        inference: InferenceService = app.state.inference
        return {
            "status": "ok",
            "models_loaded": inference.models_loaded,
            "antigravity_enabled": inference.antigravity_enabled,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info"),
        reload=True,
    )
