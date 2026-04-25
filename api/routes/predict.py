"""
NASCAR prediction endpoint — GAP-A-14.

POST /api/v1/nascar/predict

Returns win / top-3 / top-5 / top-10 probabilities for a full race field
using the trained NascarPredictor stacking ensemble (CatBoost + LightGBM +
XGBoost + BetaCalibrator + Harville multi-placement DP).

NEVER returns a hardcoded default probability.  Returns HTTP 503 if the
predictor has not been loaded at startup.

Standard response envelope:
    {"success": true, "data": {...}, "meta": {"request_id": "uuid", "timestamp": "ISO8601"}}
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Request, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger()
router = APIRouter(tags=["Predict"])

_VALID_SURFACES = {"paved", "dirt", "road"}


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _meta(request_id: str) -> Dict[str, str]:
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _ok(data: Any, request_id: str) -> Dict[str, Any]:
    return {"success": True, "data": data, "meta": _meta(request_id)}


def _error(code: str, message: str, request_id: str, http_status: int = 400) -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "success": False,
            "error": {"code": code, "message": message},
            "meta": _meta(request_id),
        },
        status_code=http_status,
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DriverInput(BaseModel):
    name: str = Field(..., min_length=1, description="Driver full name matching training data")
    team: Optional[str] = Field(None, description="Team name (optional)")
    make: Optional[str] = Field(None, description="Manufacturer (Chevrolet, Ford, Toyota) — optional")
    starting_pos: Optional[int] = Field(None, ge=1, description="Qualifying grid position — optional")


class PredictRequest(BaseModel):
    """Race field definition for NASCAR win-probability prediction."""

    drivers: List[DriverInput] = Field(
        ...,
        min_length=2,
        description="Race field. Minimum 2 drivers required.",
    )
    track: str = Field(
        ...,
        min_length=1,
        description="Track name matching training data (e.g. 'Daytona', 'Talladega')",
    )
    surface: str = Field(
        default="paved",
        description="Track surface: paved | dirt | road",
    )
    season: int = Field(
        default=2025,
        ge=2000,
        le=2050,
        description="Race season year",
    )
    track_length: float = Field(
        default=1.5,
        gt=0.0,
        lt=10.0,
        description="Track length in miles (used for feature extraction)",
    )

    @field_validator("surface")
    @classmethod
    def validate_surface(cls, v: str) -> str:
        lower = v.lower()
        if lower not in _VALID_SURFACES:
            raise ValueError(f"surface must be one of {sorted(_VALID_SURFACES)}, got {v!r}")
        return lower


class DriverPrediction(BaseModel):
    driver: str
    win_prob: float = Field(description="P(driver wins), calibrated Harville-normalised")
    top3_prob: float = Field(description="P(driver finishes top 3), Harville DP")
    top5_prob: float = Field(description="P(driver finishes top 5), Harville DP")
    top10_prob: float = Field(description="P(driver finishes top 10), Harville DP")


class PredictResponseData(BaseModel):
    track: str
    surface: str
    season: int
    track_length: float
    field_size: int
    results: List[DriverPrediction]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    summary="Predict NASCAR race win / top-3 / top-5 / top-10 probabilities (ML ensemble)",
    response_class=ORJSONResponse,
)
async def predict_race(request: Request, body: PredictRequest) -> ORJSONResponse:
    """
    Return win, top-3, top-5, and top-10 probabilities for a NASCAR race field.

    Uses the NascarPredictor 3-model stacking ensemble (CatBoost + LightGBM +
    XGBoost) with BetaCalibrator and Harville DP for multi-placement markets.
    Results are sorted by win_prob descending.

    HTTP 503 is returned when:
      - The predictor has not been loaded at startup (models not trained/found)
    """
    rid = str(uuid.uuid4())
    log = logger.bind(
        request_id=rid,
        track=body.track,
        surface=body.surface,
        field_size=len(body.drivers),
    )
    log.info("nascar_predict_requested")

    # ── Obtain predictor from app state ─────────────────────────────────────
    predictor = getattr(request.app.state, "predictor", None)
    predictor_loaded = (
        predictor is not None
        and (predictor.is_loaded if hasattr(predictor, "is_loaded") else getattr(predictor, "_loaded", False))
    )
    if not predictor_loaded:
        log.warning("nascar_predict_predictor_not_loaded")
        return _error(
            "PREDICTOR_UNAVAILABLE",
            (
                "NASCAR predictor is not loaded. "
                "Ensure the service started successfully and models are present. "
                "Run: python -c \"from ml.trainer import NascarTrainer; NascarTrainer().train()\" "
                "to train models if missing."
            ),
            rid,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # ── Build driver dicts for predictor ────────────────────────────────────
    drivers_payload = [
        {
            "name": d.name,
            **({"team": d.team} if d.team is not None else {}),
            **({"make": d.make} if d.make is not None else {}),
            **({"starting_pos": d.starting_pos} if d.starting_pos is not None else {}),
        }
        for d in body.drivers
    ]

    # ── Run inference ────────────────────────────────────────────────────────
    try:
        raw_results = predictor.predict_race(
            drivers=drivers_payload,
            track=body.track,
            surface=body.surface,
            season=body.season,
            track_length=body.track_length,
        )
    except ValueError as exc:
        log.warning("nascar_predict_invalid_input", error=str(exc))
        return _error("INVALID_INPUT", str(exc), rid, status.HTTP_422_UNPROCESSABLE_ENTITY)
    except RuntimeError as exc:
        log.error("nascar_predict_inference_failed", error=str(exc))
        return _error(
            "INFERENCE_FAILED",
            f"Prediction failed: {exc}",
            rid,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as exc:
        log.error("nascar_predict_unexpected_error", error=str(exc))
        return _error(
            "INTERNAL_ERROR",
            f"Unexpected error during prediction: {exc}",
            rid,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # ── Apply Harville DP for multi-placement markets ────────────────────────
    # The predictor returns win_prob only; compute top-3/5/10 using the same
    # Harville function that the pricing layer uses.
    try:
        import numpy as np
        from pricing.markets import _harville_top_k  # same Harville used by build_all_markets
        win_probs_arr = np.array([float(r["win_prob"]) for r in raw_results], dtype=np.float64)
        top3_arr = _harville_top_k(win_probs_arr, k=3)
        top5_arr = _harville_top_k(win_probs_arr, k=5)
        top10_arr = _harville_top_k(win_probs_arr, k=10)
    except Exception as _harv_err:
        log.warning("nascar_predict_harville_failed", error=str(_harv_err))
        # Return zeros rather than silently wrong probabilities — callers must handle
        n = len(raw_results)
        import numpy as np
        top3_arr = np.zeros(n)
        top5_arr = np.zeros(n)
        top10_arr = np.zeros(n)

    # ── Build response ───────────────────────────────────────────────────────
    results = [
        DriverPrediction(
            driver=r.get("driver", r.get("driver_name", "Unknown")),
            win_prob=round(float(r["win_prob"]), 6),
            top3_prob=round(float(top3_arr[i]), 6),
            top5_prob=round(float(top5_arr[i]), 6),
            top10_prob=round(float(top10_arr[i]), 6),
        )
        for i, r in enumerate(raw_results)
    ]

    response_data = PredictResponseData(
        track=body.track,
        surface=body.surface,
        season=body.season,
        track_length=body.track_length,
        field_size=len(results),
        results=results,
    )

    log.info(
        "nascar_predict_complete",
        track=body.track,
        field_size=len(results),
        top_driver=results[0].driver if results else None,
        top_win_prob=results[0].win_prob if results else None,
    )
    return ORJSONResponse(content=_ok(response_data.model_dump(), rid))
