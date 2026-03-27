"""NASCAR health check routes."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import SERVICE_NAME, SERVICE_VERSION

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Basic health — always returns 200 if service is up."""
    return JSONResponse({
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
    })


@router.get("/health/ready")
async def health_ready(request: Request) -> JSONResponse:
    """Readiness probe — indicates whether the model is loaded."""
    predictor = getattr(request.app.state, "predictor", None)
    model_loaded = predictor is not None and predictor.is_loaded

    return JSONResponse(
        status_code=200 if model_loaded else 503,
        content={
            "ready": model_loaded,
            "model_loaded": model_loaded,
            "driver_count": predictor.driver_count if model_loaded else 0,
            "service": SERVICE_NAME,
        },
    )


@router.get("/health/live")
async def health_live() -> JSONResponse:
    """Liveness probe — always 200 if process is running."""
    return JSONResponse({"status": "ok", "service": SERVICE_NAME})
