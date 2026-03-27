"""NASCAR race prediction and pricing routes."""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pricing.markets import NascarPricer

logger = logging.getLogger(__name__)
router = APIRouter()

pricer = NascarPricer()


class DriverEntry(BaseModel):
    name: str
    team: Optional[str] = None
    make: Optional[str] = "Chevrolet"
    starting_pos: Optional[int] = None


class RacePriceRequest(BaseModel):
    race_name: str
    track: str
    season: int
    surface: Optional[str] = "paved"
    track_length: Optional[float] = 1.5
    drivers: list[DriverEntry] = Field(..., min_length=2)


class H2HRequest(BaseModel):
    race_name: str
    track: str
    season: int
    surface: Optional[str] = "paved"
    track_length: Optional[float] = 1.5
    drivers: list[DriverEntry] = Field(..., min_length=2)
    driver_a: str
    driver_b: str


@router.post("/price")
async def price_race(body: RacePriceRequest, request: Request) -> JSONResponse:
    """
    Price all standard NASCAR markets for a race field.

    Returns race_winner, top_3, top_5, top_10 markets with Harville-derived
    probabilities and decimal odds.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="NASCAR predictor not loaded — models may still be training",
        )

    drivers_payload = [d.model_dump() for d in body.drivers]

    try:
        predictions = predictor.predict_race(
            drivers=drivers_payload,
            track=body.track,
            surface=body.surface or "paved",
            season=body.season,
            track_length=body.track_length or 1.5,
        )
    except Exception as exc:
        logger.error("predict_race failed for %s: %s", body.race_name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    try:
        markets = pricer.price_race(predictions)
    except Exception as exc:
        logger.error("price_race failed for %s: %s", body.race_name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pricing error: {exc}") from exc

    logger.info(
        "price_race OK race=%s track=%s drivers=%d top_driver=%s win_prob=%.4f",
        body.race_name,
        body.track,
        len(body.drivers),
        predictions[0]["driver"] if predictions else "n/a",
        predictions[0]["win_prob"] if predictions else 0.0,
    )

    return JSONResponse({
        "race_name": body.race_name,
        "track": body.track,
        "season": body.season,
        "surface": body.surface,
        "field_size": len(body.drivers),
        "predictions": predictions,
        "markets": markets,
    })


@router.post("/h2h")
async def price_h2h(body: H2HRequest, request: Request) -> JSONResponse:
    """
    Head-to-head market between two specific drivers in the race field.
    Renormalises their win probabilities to sum=1.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    drivers_payload = [d.model_dump() for d in body.drivers]

    try:
        predictions = predictor.predict_race(
            drivers=drivers_payload,
            track=body.track,
            surface=body.surface or "paved",
            season=body.season,
            track_length=body.track_length or 1.5,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    try:
        h2h = pricer.price_h2h(body.driver_a, body.driver_b, predictions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"H2H pricing error: {exc}") from exc

    return JSONResponse({
        "race_name": body.race_name,
        "track": body.track,
        "season": body.season,
        "h2h": h2h,
    })
