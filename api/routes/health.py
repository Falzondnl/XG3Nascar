"""NASCAR health check routes."""
from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import SERVICE_NAME, SERVICE_VERSION, R0_DIR

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


@router.get("/health/debug")
async def health_debug(request: Request) -> JSONResponse:
    """Debug endpoint — reports model paths, file presence, and library versions."""
    import importlib
    r0_abs = os.path.abspath(R0_DIR)
    r0_files = os.listdir(r0_abs) if os.path.exists(r0_abs) else []

    versions: dict[str, str] = {}
    for lib in ("sklearn", "catboost", "lightgbm", "xgboost", "numpy", "pandas"):
        try:
            mod = importlib.import_module(lib if lib != "sklearn" else "sklearn")
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not_installed"

    predictor = getattr(request.app.state, "predictor", None)
    return JSONResponse({
        "cwd": os.getcwd(),
        "r0_dir": R0_DIR,
        "r0_abs": r0_abs,
        "r0_exists": os.path.exists(r0_abs),
        "r0_files": r0_files,
        "model_loaded": predictor.is_loaded if predictor else False,
        "driver_count": predictor.driver_count if predictor else 0,
        "library_versions": versions,
    })
