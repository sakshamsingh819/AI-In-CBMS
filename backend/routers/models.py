"""
Models router — ML model metadata endpoint.

Endpoints:
  GET /api/models — List all available models with metadata and antigravity capabilities
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from schemas import ModelInfo, ModelsResponse
from services.inference import InferenceService

router = APIRouter()


@router.get(
    "/",
    response_model=ModelsResponse,
    summary="List all available inference models",
    description=(
        "Returns metadata for all three model variants (Random Forest, LSTM, cGAN-Hybrid) "
        "including their fault class lists, feature vector lengths, load status, and — when "
        "ANTIGRAVITY_ENABLED=true — the per-class F1-score for the antigravity class on the "
        "synthetic test set."
    ),
    responses={
        200: {
            "description": "Model metadata list",
            "content": {
                "application/json": {
                    "example": {
                        "models": [
                            {
                                "name": "Random Forest",
                                "version": "v2",
                                "fault_classes": [
                                    "normal", "bpfi", "bpfo", "bsf",
                                    "unbalance", "misalignment", "looseness", "antigravity"
                                ],
                                "feature_vector_length": 54,
                                "antigravity_capable": True,
                                "antigravity_f1": 0.83,
                                "loaded": True,
                            }
                        ],
                        "active_model": "Random Forest",
                        "antigravity_enabled": True,
                    }
                }
            },
        }
    },
)
async def list_models(request: Request) -> ModelsResponse:
    """Return metadata for all available ML models."""
    inference: InferenceService = request.app.state.inference
    metas = inference.get_models_metadata()

    return ModelsResponse(
        models=[
            ModelInfo(
                name=m.name,
                version=m.version,
                fault_classes=m.fault_classes,
                feature_vector_length=m.feature_vector_length,
                antigravity_capable=m.antigravity_capable,
                antigravity_f1=m.antigravity_f1,
                loaded=m.loaded,
            )
            for m in metas
        ],
        active_model="Random Forest",
        antigravity_enabled=inference.antigravity_enabled,
    )
