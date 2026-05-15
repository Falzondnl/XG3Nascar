"""
NASCAR Tier 2B Reverse Engineer (Family F: Plackett-Luce)
===========================================================

Closed-form Plackett-Luce inverse for NASCAR race winner / championship
outright markets (Cup, Xfinity, Truck series).

Closed-form algorithm:
  1. Devig Pinnacle outright market -> fair win probs p_1, ..., p_n (sum=1).
  2. Recover PL skill weights: w_i = p_i / p_1.
  3. Top-5/top-10 finish, "lead lap" approximations via Gumbel-max top-K.

Why Family F for NASCAR?
  - 36-40 driver race winner market.
  - Pinnacle race winner / championship odds are the canonical input.
  - Same PL skills price top-5 / top-10 finish derivative markets.

LOCK-NASCAR-TIER-2B-REVERSE-ENGINEER-001
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger: structlog.BoundLogger = structlog.get_logger(__name__)


_MIN_PROB: float = 1e-9
_MAX_PROB: float = 1.0 - 1e-9
_MIN_FIELD_SIZE: int = 4
_MAX_FIELD_SIZE: int = 50
_ROUND_TRIP_TOLERANCE: float = 1e-6


@dataclass
class NASCARTier2BResult:
    driver_ids: List[str]
    skills: np.ndarray
    log_skills: np.ndarray
    win_probs: np.ndarray
    market_id: str = ""
    market_type: str = "race_winner"
    series: str = "cup"  # cup, xfinity, truck
    n_drivers: int = 0
    overround: float = 0.0
    prediction_source: str = "market_scrape_reverse_engineered"
    model_available: bool = False
    solver_converged: bool = True
    round_trip_residual: float = 0.0
    confidence: float = 1.0
    tier_2b_restricted: bool = False
    notes: str = ""

    def top_k_probs(self, k: int) -> np.ndarray:
        return _gumbel_max_top_k(self.skills.astype(np.float64), k)

    def top_5_probs(self) -> np.ndarray:
        return self.top_k_probs(5)

    def top_10_probs(self) -> np.ndarray:
        return self.top_k_probs(10)

    def to_summary(self) -> Dict[str, float | int | str | bool]:
        return {
            "n_drivers": int(self.n_drivers),
            "series": self.series,
            "overround": round(float(self.overround), 6),
            "market_type": self.market_type,
            "prediction_source": self.prediction_source,
            "model_available": self.model_available,
            "solver_converged": self.solver_converged,
            "round_trip_residual": float(self.round_trip_residual),
            "confidence": float(self.confidence),
            "tier_2b_restricted": self.tier_2b_restricted,
            "market_id": self.market_id,
        }


def devig_outright_market(
    odds_by_entity: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    if len(odds_by_entity) < 2:
        raise ValueError(f"Need >= 2 entities; got {len(odds_by_entity)}")
    implied: Dict[str, float] = {}
    for eid, odds in odds_by_entity.items():
        if odds is None or float(odds) <= 1.0:
            raise ValueError(f"Invalid odds for {eid}: {odds}")
        implied[eid] = 1.0 / float(odds)
    overround = sum(implied.values())
    if overround <= 0.0:
        raise ValueError(f"Degenerate overround: {overround}")
    return {eid: imp / overround for eid, imp in implied.items()}, overround


def plackett_luce_inverse(
    fair_probs: Dict[str, float],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    pairs = sorted(fair_probs.items(), key=lambda kv: -kv[1])
    ids = [p[0] for p in pairs]
    probs = np.array([p[1] for p in pairs], dtype=np.float64)
    probs = np.clip(probs, _MIN_PROB, _MAX_PROB)
    probs = probs / probs.sum()
    p1 = probs[0]
    if p1 <= 0.0:
        raise ValueError("Favourite prob zero")
    skills = probs / p1
    reconstructed = skills / skills.sum()
    return ids, skills, reconstructed


def _gumbel_max_top_k(skills: np.ndarray, k: int) -> np.ndarray:
    n = len(skills)
    if k >= n:
        return np.ones(n)
    if k == 1:
        return skills / skills.sum()
    n_mc = max(20_000, 200 * k)
    rng = np.random.default_rng(seed=20260515)
    log_skills = np.log(np.clip(skills, 1e-300, None))
    gumbel = -np.log(-np.log(rng.random((n_mc, n))))
    scores = gumbel + log_skills[np.newaxis, :]
    top_k_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(counts, top_k_idx.ravel(), 1.0)
    return np.clip(counts / float(n_mc), 0.0, 1.0)


class NASCARTier2BReverseEngineer:
    """PL reverse-engineer for NASCAR race + championship outrights."""

    def __init__(self) -> None:
        self._log = logger.bind(service="NASCARTier2BReverseEngineer")

    def reverse_engineer(
        self,
        market_id: str,
        outright_odds: Dict[str, float],
        series: str = "cup",
        market_type: str = "race_winner",
    ) -> NASCARTier2BResult:
        n = len(outright_odds)
        if n < _MIN_FIELD_SIZE:
            raise ValueError(f"NASCAR Tier 2B requires >= {_MIN_FIELD_SIZE}; got {n}")
        if n > _MAX_FIELD_SIZE:
            raise ValueError(f"NASCAR Tier 2B cap is {_MAX_FIELD_SIZE}; got {n}")

        fair_probs, overround = devig_outright_market(outright_odds)
        ids, skills, reconstructed = plackett_luce_inverse(fair_probs)

        input_order = np.array([fair_probs[i] for i in ids], dtype=np.float64)
        input_order = input_order / input_order.sum()
        residual = float(np.max(np.abs(reconstructed - input_order)))

        log_skills = np.log(np.clip(skills, _MIN_PROB, None))

        if overround > 1.30:
            confidence, note = 0.6, f"high overround ({overround:.3f})"
        elif overround > 1.15:
            confidence, note = 0.85, f"moderate overround ({overround:.3f})"
        else:
            confidence, note = 1.0, "tight market"

        tier_2b_restricted = n < 8

        result = NASCARTier2BResult(
            driver_ids=ids,
            skills=skills,
            log_skills=log_skills,
            win_probs=reconstructed,
            market_id=market_id,
            market_type=market_type,
            series=series,
            n_drivers=n,
            overround=overround,
            round_trip_residual=residual,
            confidence=confidence,
            tier_2b_restricted=tier_2b_restricted,
            notes=note,
        )
        self._log.info(
            "nascar_tier2b_solved",
            market_id=market_id,
            series=series,
            n_drivers=n,
            overround=round(overround, 4),
            favourite=ids[0],
            favourite_prob=round(float(reconstructed[0]), 4),
        )
        return result


_DEFAULT_ENGINEER = NASCARTier2BReverseEngineer()


def get_tier2b_engineer() -> NASCARTier2BReverseEngineer:
    return _DEFAULT_ENGINEER
