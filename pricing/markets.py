"""
NASCAR Pricing Engine — Harville multi-outcome markets.

Markets:
  - Race Winner: Harville win probs + 12% margin
  - Podium Finish (Top 3): Harville cumulative + 10% margin
  - Top 5 / Top 10: Harville cumulative + 10%/8% margin
  - Head-to-Head: renormalised win probs + 5% margin

Harville formula (recursive):
  P(i finishes k-th | positions 1..k-1 taken by set S) = p_i / (1 - sum(p_j for j in S))

No scipy dependency — uses pure Python math.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from config import (
    HARVILLE_TOP_N,
    MARGIN_H2H,
    MARGIN_RACE_WINNER,
    MARGIN_TOP_3,
    MARGIN_TOP_5,
    MARGIN_TOP_10,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    """Standard normal CDF — pure Python (no scipy). Uses math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _apply_margin(fair_prob: float, margin: float) -> float:
    """
    Apply sportsbook overround (margin) to a fair probability.
    Result: boosted_prob = fair_prob * (1 + margin).
    Clipped to [0.001, 0.999].
    """
    return max(0.001, min(0.999, fair_prob * (1.0 + margin)))


def _prob_to_decimal_odds(prob: float) -> float:
    """Convert probability to decimal odds. Min odds 1.001."""
    if prob <= 0.0:
        return 1000.0
    return max(1.001, round(1.0 / prob, 3))


def _harville_top_k(win_probs: np.ndarray, k: int) -> np.ndarray:
    """
    Compute P(driver i finishes in top k) for all drivers using Harville DP.

    For the exact recursion P(i finishes r-th):
      P(i 1st) = p_i
      P(i r-th) = sum_{j != i} P(j wins remaining_after_removing_j) * P(i (r-1)-th in rest)

    We use the Harville sequential approximation which is exact for independent races:
      P(i wins | set S already placed) = p_i / (1 - sum_{j in S} p_j)

    For top-k, we marginalize over all orderings of top (k-1) finishers.

    For fields up to HARVILLE_TOP_N drivers, use Monte Carlo sampling for speed
    (50K samples). For k=1 (win), return exact probs.
    For k>=2, use Monte Carlo simulation.
    """
    n = len(win_probs)
    if n == 0:
        return np.array([])

    win_probs = np.clip(win_probs, 1e-9, 1.0)
    probs = win_probs / win_probs.sum()

    if k == 1:
        return probs.copy()

    k = min(k, n - 1)

    # For small fields with k=3, use exact Harville recursion
    if n <= 30 and k == 3:
        return _harville_exact_top3(probs)

    # For all other cases: Monte Carlo Harville simulation (fast)
    return _harville_monte_carlo(probs, k, n_samples=80_000)


def _harville_exact_top3(probs: np.ndarray) -> np.ndarray:
    """Exact Harville for P(i in top 3) — O(n^3) but n<=30."""
    n = len(probs)
    top3 = np.zeros(n)
    for i in range(n):
        # P(i wins)
        p_win = probs[i]
        # P(i 2nd) = sum_j P(j wins) * P(i | rest after j)
        p_2nd = 0.0
        for j in range(n):
            if j == i:
                continue
            rest = 1.0 - probs[j]
            if rest < 1e-9:
                continue
            p_2nd += probs[j] * (probs[i] / rest)
        # P(i 3rd) = sum_{j,k} P(j 1st) * P(k 2nd | j placed) * P(i | j,k placed)
        p_3rd = 0.0
        for j in range(n):
            if j == i:
                continue
            rest_j = 1.0 - probs[j]
            if rest_j < 1e-9:
                continue
            for kk in range(n):
                if kk == i or kk == j:
                    continue
                rest_jk = 1.0 - probs[j] - probs[kk]
                if rest_jk < 1e-9:
                    continue
                p_3rd += (
                    probs[j]
                    * (probs[kk] / rest_j)
                    * (probs[i] / rest_jk)
                )
        top3[i] = p_win + p_2nd + p_3rd
    return np.clip(top3, 0.0, 1.0)


def _harville_monte_carlo(probs: np.ndarray, k: int, n_samples: int = 80_000) -> np.ndarray:
    """
    Monte Carlo simulation of Harville sequential draws.
    For each simulated race, draw top-k positions without replacement
    using Harville probs (renormalised at each step).
    Count how often each driver lands in top k.
    """
    n = len(probs)
    counts = np.zeros(n, dtype=np.float64)

    for _ in range(n_samples):
        remaining = probs.copy()
        placed = set()
        for _slot in range(k):
            total = remaining.sum()
            if total < 1e-9:
                break
            norm = remaining / total
            chosen = int(np.random.choice(n, p=norm))
            placed.add(chosen)
            counts[chosen] += 1
            remaining[chosen] = 0.0

    return np.clip(counts / n_samples, 0.0, 1.0)


class NascarPricer:
    """
    Generates all NASCAR race markets from win probability estimates.
    All odds are decimal (European format).
    """

    def price_race(
        self,
        predictions: list[dict],
    ) -> dict[str, Any]:
        """
        Generate full market set from prediction results.

        predictions: list of {driver, win_prob, ...} sorted by win_prob desc.
        Returns: dict of market_type -> list of {driver, fair_prob, market_prob, decimal_odds}
        """
        if not predictions:
            raise ValueError("predictions list cannot be empty")

        drivers = [p["driver"] for p in predictions]
        win_probs = np.array([p["win_prob"] for p in predictions], dtype=float)

        # Re-normalise (should already sum to 1 but defensive)
        total = win_probs.sum()
        if total < 1e-9:
            raise ValueError("All win probs are zero — cannot price markets")
        win_probs = win_probs / total

        # Compute Harville top-k probabilities
        top3_probs = _harville_top_k(win_probs, k=3)
        top5_probs = _harville_top_k(win_probs, k=5)
        top10_probs = _harville_top_k(win_probs, k=10)

        def build_selections(
            fair_probs: np.ndarray,
            margin: float,
            market_name: str,
        ) -> list[dict]:
            out = []
            for i, driver in enumerate(drivers):
                fp = float(fair_probs[i])
                mp = _apply_margin(fp, margin)
                out.append({
                    "driver": driver,
                    "fair_prob": round(fp, 6),
                    "market_prob": round(mp, 6),
                    "decimal_odds": _prob_to_decimal_odds(mp),
                })
            return sorted(out, key=lambda x: x["decimal_odds"])

        markets: dict[str, Any] = {
            "race_winner": {
                "market_type": "race_winner",
                "margin": MARGIN_RACE_WINNER,
                "total_overround": round(1.0 + MARGIN_RACE_WINNER, 4),
                "selections": build_selections(win_probs, MARGIN_RACE_WINNER, "race_winner"),
            },
            "top_3": {
                "market_type": "top_3",
                "margin": MARGIN_TOP_3,
                "total_overround": round(3.0 * (1.0 + MARGIN_TOP_3), 4),
                "selections": build_selections(top3_probs, MARGIN_TOP_3, "top_3"),
            },
            "top_5": {
                "market_type": "top_5",
                "margin": MARGIN_TOP_5,
                "total_overround": round(5.0 * (1.0 + MARGIN_TOP_5), 4),
                "selections": build_selections(top5_probs, MARGIN_TOP_5, "top_5"),
            },
            "top_10": {
                "market_type": "top_10",
                "margin": MARGIN_TOP_10,
                "total_overround": round(10.0 * (1.0 + MARGIN_TOP_10), 4),
                "selections": build_selections(top10_probs, MARGIN_TOP_10, "top_10"),
            },
        }

        return markets

    def price_h2h(
        self,
        driver_a: str,
        driver_b: str,
        predictions: list[dict],
    ) -> dict[str, Any]:
        """
        Head-to-head market between exactly two drivers.
        Renormalise their win probs to sum=1, apply H2H margin.
        """
        prob_map = {p["driver"]: p["win_prob"] for p in predictions}

        p_a = prob_map.get(driver_a)
        p_b = prob_map.get(driver_b)

        if p_a is None:
            raise ValueError(f"Driver not found in predictions: {driver_a}")
        if p_b is None:
            raise ValueError(f"Driver not found in predictions: {driver_b}")

        total = p_a + p_b
        if total < 1e-9:
            raise ValueError(f"Both driver win probs are zero: {driver_a} vs {driver_b}")

        fair_a = p_a / total
        fair_b = p_b / total

        market_a = _apply_margin(fair_a, MARGIN_H2H)
        market_b = _apply_margin(fair_b, MARGIN_H2H)

        return {
            "market_type": "h2h",
            "margin": MARGIN_H2H,
            "driver_a": {
                "driver": driver_a,
                "fair_prob": round(fair_a, 6),
                "market_prob": round(market_a, 6),
                "decimal_odds": _prob_to_decimal_odds(market_a),
            },
            "driver_b": {
                "driver": driver_b,
                "fair_prob": round(fair_b, 6),
                "market_prob": round(market_b, 6),
                "decimal_odds": _prob_to_decimal_odds(market_b),
            },
        }
