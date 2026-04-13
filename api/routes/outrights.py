"""
NASCAR Season Championship Outright Market
===========================================

Produces NASCAR Cup Series (and optionally Xfinity/Truck) championship winner
prices derived from the live ELO ratings in app.state.predictor.

Endpoint
--------
GET /api/v1/nascar/outrights/championship?season=2026&surface=overall&series=cup

Returns a ranked list of drivers with fair win probability and decimal odds
at the configured championship margin (10%).

Methodology
-----------
Uses Harville softmax on elo_overall (or surface-specific ELO) to produce
championship win probabilities.  Driver names come from the ELO store directly
(NASCAR predictor stores driver name alongside ELO).

This mirrors the F1 WDC endpoint pattern:
  GET /api/v1/formula1/outrights/wdc
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_CHAMPIONSHIP_MARGIN: float = float(os.getenv("CHAMPIONSHIP_MARGIN", "0.10"))
_TOP_N_CHAMPIONSHIP: int = int(os.getenv("CHAMPIONSHIP_TOP_N", "30"))

_VALID_SURFACES = {"overall", "paved", "dirt", "road"}


def _harville_softmax_probs(entries: list[tuple[str, float]], temperature: float = 400.0) -> list[tuple[str, float]]:
    """Convert ELO ratings to championship win probabilities via Harville softmax."""
    if not entries:
        return []
    max_elo = max(elo for _, elo in entries)
    exp_vals = [(name, math.exp((elo - max_elo) / temperature)) for name, elo in entries]
    total = sum(v for _, v in exp_vals)
    if total == 0:
        n = len(exp_vals)
        return [(name, 1.0 / n) for name, _ in exp_vals]
    probs = [(name, v / total) for name, v in exp_vals]
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs


def _probability_to_decimal(prob: float, margin: float) -> float:
    """Convert fair probability to decimal odds with margin."""
    if prob <= 0:
        return 999.99
    return max(1.01, round(1.0 / (prob * (1.0 + margin)), 2))


@router.get("/championship")
async def get_championship_outrights(
    season: int = 2026,
    surface: str = "overall",
    series: str = "cup",
    top_n: int = _TOP_N_CHAMPIONSHIP,
    request: Request = None,
) -> JSONResponse:
    """
    NASCAR series championship outright market.

    Win probabilities derived from driver ELO ratings via Harville softmax.

    Args:
        season:  Championship season year.  Default: 2026.
        surface: ELO variant — "overall", "paved", "dirt", "road".
                 Default: "overall" (best for season outrights).
        series:  "cup", "xfinity", "truck".  Default: "cup".
                 (All series use the same underlying ELO model.)
        top_n:   Number of drivers in the market.  Default: 30.

    Returns:
        {
          "market": "nascar_cup_championship",
          "season_year": 2026,
          "surface": "overall",
          "series": "cup",
          "entries": [
            {"rank": 1, "name": "Tyler Reddick", "probability": 0.09, "price": 10.12,
             "career_wins": 12, "career_races": 224},
            ...
          ],
          "margin": 0.10
        }
    """
    top_n = max(5, min(top_n, 60))
    if surface not in _VALID_SURFACES:
        raise HTTPException(
            status_code=400,
            detail=f"surface must be one of {sorted(_VALID_SURFACES)}",
        )
    _VALID_SERIES = {"cup", "xfinity", "truck"}
    if series not in _VALID_SERIES:
        raise HTTPException(
            status_code=400,
            detail=f"series must be one of {sorted(_VALID_SERIES)}",
        )

    predictor = getattr(request.app.state, "predictor", None) if request else None
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="NASCAR predictor not loaded — models may still be initialising",
        )

    # Fetch driver ELO from predictor via get_top_elo_drivers
    # Returns list[dict] with keys: driver, elo_overall, elo_paved, elo_dirt, elo_road,
    #                                career_wins, career_races
    try:
        raw_driver_list: list[dict] = predictor.get_top_elo_drivers(n=top_n * 2)
    except Exception as exc:
        logger.error("championship: get_top_elo_drivers failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"ELO data unavailable: {exc}"
        ) from exc

    if not raw_driver_list:
        raise HTTPException(
            status_code=503,
            detail="No ELO data available — predictor may need retraining",
        )

    elo_key_map = {
        "overall": "elo_overall",
        "paved": "elo_paved",
        "dirt": "elo_dirt",
        "road": "elo_road",
    }
    elo_key = elo_key_map[surface]

    raw_pairs: list[tuple[str, float]] = []
    driver_extra: dict[str, dict] = {}
    for item in raw_driver_list:
        name = str(item.get("driver") or item.get("name") or "Unknown")
        elo_val = float(item.get(elo_key) or item.get("elo_overall", 1500.0))
        raw_pairs.append((name, elo_val))
        driver_extra[name] = {
            "career_wins": item.get("career_wins", 0),
            "career_races": item.get("career_races", 0),
        }

    # Sort by the chosen surface ELO, take top_n
    raw_pairs.sort(key=lambda x: x[1], reverse=True)
    raw_pairs = raw_pairs[:top_n]

    probs = _harville_softmax_probs(raw_pairs)

    entries: list[dict[str, Any]] = []
    for rank, (name, prob) in enumerate(probs, start=1):
        extra = driver_extra.get(name, {})
        entry: dict[str, Any] = {
            "rank": rank,
            "name": name,
            "probability": round(prob, 6),
            "price": _probability_to_decimal(prob, _CHAMPIONSHIP_MARGIN),
        }
        if extra.get("career_wins") is not None:
            entry["career_wins"] = extra["career_wins"]
        if extra.get("career_races") is not None:
            entry["career_races"] = extra["career_races"]
        entries.append(entry)

    logger.info(
        "nascar_championship season=%d surface=%s series=%s entries=%d top=%s",
        season, surface, series, len(entries),
        entries[0]["name"] if entries else "none",
    )

    return JSONResponse({
        "market": f"nascar_{series}_championship",
        "season_year": season,
        "surface": surface,
        "series": series,
        "entries": entries,
        "margin": _CHAMPIONSHIP_MARGIN,
        "total_probability": round(sum(e["probability"] for e in entries), 6),
    })
