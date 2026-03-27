"""NASCAR admin routes — ELO ratings, model status, retraining."""
from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_training_lock = threading.Lock()
_training_in_progress = False


@router.get("/elo-ratings")
async def get_elo_ratings(request: Request, top: int = 50) -> JSONResponse:
    """Return top-N drivers ranked by overall ELO rating."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    top = min(max(top, 1), 200)
    try:
        ratings = predictor.get_top_elo_drivers(n=top)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({
        "top_n": top,
        "driver_count": predictor.driver_count,
        "ratings": ratings,
    })


@router.post("/train")
async def trigger_training(request: Request) -> JSONResponse:
    """
    Trigger model retraining asynchronously.
    Returns immediately with job status.
    Only one training job runs at a time.
    """
    global _training_in_progress

    if _training_in_progress:
        return JSONResponse(
            status_code=409,
            content={"status": "already_running", "message": "Training already in progress"},
        )

    def _run_training() -> None:
        global _training_in_progress
        _training_in_progress = True
        try:
            from ml.trainer import train as run_train
            from config import NASCAR_CSV, R0_DIR
            metrics = run_train(csv_path=NASCAR_CSV, out_dir=R0_DIR)
            logger.info("Background training complete: %s", metrics)
            # Reload predictor in app state after training
            predictor = getattr(request.app.state, "predictor", None)
            if predictor is not None:
                try:
                    predictor.load(R0_DIR)
                    logger.info("Predictor hot-reloaded after training")
                except Exception as reload_exc:
                    logger.error("Predictor reload failed: %s", reload_exc)
        except Exception as exc:
            logger.error("Background training failed: %s", exc, exc_info=True)
        finally:
            _training_in_progress = False

    t = threading.Thread(target=_run_training, daemon=True, name="nascar-trainer")
    t.start()

    return JSONResponse({
        "status": "started",
        "message": "Training started in background. Check /health/ready to see when models are reloaded.",
    })


@router.get("/status")
async def model_status(request: Request) -> JSONResponse:
    """Return current model and predictor status."""
    predictor = getattr(request.app.state, "predictor", None)
    model_loaded = predictor is not None and predictor.is_loaded
    return JSONResponse({
        "model_loaded": model_loaded,
        "driver_count": predictor.driver_count if model_loaded else 0,
        "training_in_progress": _training_in_progress,
    })
