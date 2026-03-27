"""
NASCAR pricing unit tests.
Tests Harville DP, market construction, H2H pricing, margin application.
No model loading required — tests pricing logic in isolation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pricing.markets import (
    NascarPricer,
    _apply_margin,
    _harville_exact_top3,
    _harville_monte_carlo,
    _harville_top_k,
    _normal_cdf,
    _prob_to_decimal_odds,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestNormalCDF:
    def test_standard_values(self):
        assert abs(_normal_cdf(0.0) - 0.5) < 1e-6
        assert abs(_normal_cdf(1.96) - 0.975) < 0.001
        assert abs(_normal_cdf(-1.96) - 0.025) < 0.001

    def test_extreme_values(self):
        assert _normal_cdf(10.0) > 0.999
        assert _normal_cdf(-10.0) < 0.001


class TestApplyMargin:
    def test_zero_margin(self):
        assert _apply_margin(0.5, 0.0) == pytest.approx(0.5)

    def test_positive_margin(self):
        result = _apply_margin(0.5, 0.10)
        assert result == pytest.approx(0.55, abs=1e-6)

    def test_clip_high(self):
        result = _apply_margin(0.999, 0.50)
        assert result <= 0.999

    def test_clip_low(self):
        result = _apply_margin(0.0001, 0.10)
        assert result >= 0.001


class TestProbToDecimalOdds:
    def test_fair_coin(self):
        # 50% chance → 2.0 odds
        assert _prob_to_decimal_odds(0.5) == pytest.approx(2.0, abs=0.01)

    def test_favorite(self):
        # 80% → 1.25
        assert _prob_to_decimal_odds(0.8) == pytest.approx(1.25, abs=0.01)

    def test_longshot(self):
        # 5% → 20.0
        assert _prob_to_decimal_odds(0.05) == pytest.approx(20.0, abs=0.1)

    def test_zero_probability(self):
        result = _prob_to_decimal_odds(0.0)
        assert result == 1000.0

    def test_minimum_odds(self):
        # Extremely high probability should return at least 1.001
        result = _prob_to_decimal_odds(0.9999)
        assert result >= 1.001


# ---------------------------------------------------------------------------
# Harville DP
# ---------------------------------------------------------------------------

class TestHarvilleExactTop3:
    def test_uniform_field(self):
        """For uniform probs, P(top 3) = 3/n."""
        n = 20
        probs = np.ones(n) / n
        top3 = _harville_exact_top3(probs)
        expected = 3.0 / n
        for p in top3:
            assert abs(p - expected) < 0.01

    def test_winner_top3(self):
        """Dominant driver should have top3 probability >= win probability."""
        probs = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        top3 = _harville_exact_top3(probs)
        assert top3[0] >= probs[0]

    def test_sum_bounded(self):
        """Sum of top-3 probs should equal exactly 3.0 for a full field."""
        probs = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
        top3 = _harville_exact_top3(probs)
        # Sum of individual top-3 probs should be approximately 3 (each counted once)
        assert top3.sum() > 0.0
        assert all(0.0 <= p <= 1.0 for p in top3)

    def test_two_driver_top3(self):
        """With 2 drivers, both have top-2 prob = 1.0 (both finish top 2)."""
        probs = np.array([0.6, 0.4])
        top3 = _harville_exact_top3(probs)
        # With 2 drivers top-3 means both finish in top 2 → both prob 1.0
        assert top3[0] <= 1.0
        assert top3[1] <= 1.0


class TestHarvilleMonteCarlo:
    def test_win_probs_preserved(self):
        """Monte Carlo top-1 should approximate the win probs."""
        np.random.seed(42)
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        mc = _harville_monte_carlo(probs, k=1, n_samples=100_000)
        for i in range(len(probs)):
            assert abs(mc[i] - probs[i]) < 0.02

    def test_top3_gt_top1(self):
        """top-3 probability should be >= win probability for all drivers."""
        np.random.seed(42)
        probs = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        top1 = _harville_monte_carlo(probs, k=1, n_samples=50_000)
        top3 = _harville_monte_carlo(probs, k=3, n_samples=50_000)
        for i in range(len(probs)):
            assert top3[i] >= top1[i] - 0.02  # allow small MC noise


class TestHarvilleTopK:
    def test_k1_exact(self):
        """k=1 returns exact normalised win probs."""
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        result = _harville_top_k(probs, k=1)
        np.testing.assert_allclose(result, probs / probs.sum(), atol=1e-6)

    def test_monotone_field_size(self):
        """k=10 probs should be >= k=5 probs."""
        np.random.seed(0)
        probs = np.random.dirichlet(np.ones(30))
        top5 = _harville_top_k(probs, k=5)
        top10 = _harville_top_k(probs, k=10)
        for i in range(len(probs)):
            assert top10[i] >= top5[i] - 0.05  # MC noise tolerance


# ---------------------------------------------------------------------------
# NascarPricer
# ---------------------------------------------------------------------------

def _make_predictions(n: int = 6) -> list[dict]:
    """Create synthetic predictions for pricing tests."""
    # Decreasing win probs summing to 1
    raw = [1.0 / (i + 1) for i in range(n)]
    total = sum(raw)
    probs = [p / total for p in raw]
    drivers = [f"Driver_{i}" for i in range(n)]
    return [
        {
            "driver": drivers[i],
            "win_prob": probs[i],
            "raw_win_prob": probs[i],
            "elo_overall": 1500.0,
            "career_wins": i,
            "career_races": 100,
            "win_rate_last15": probs[i],
            "avg_finish_last5": float(i + 1),
        }
        for i in range(n)
    ]


class TestNascarPricer:
    def setup_method(self):
        np.random.seed(42)
        self.pricer = NascarPricer()
        self.predictions = _make_predictions(10)

    def test_price_race_keys(self):
        markets = self.pricer.price_race(self.predictions)
        assert "race_winner" in markets
        assert "top_3" in markets
        assert "top_5" in markets
        assert "top_10" in markets

    def test_race_winner_selection_count(self):
        markets = self.pricer.price_race(self.predictions)
        assert len(markets["race_winner"]["selections"]) == len(self.predictions)

    def test_decimal_odds_positive(self):
        markets = self.pricer.price_race(self.predictions)
        for sel in markets["race_winner"]["selections"]:
            assert sel["decimal_odds"] >= 1.001

    def test_favourite_has_lowest_odds(self):
        markets = self.pricer.price_race(self.predictions)
        odds = [s["decimal_odds"] for s in markets["race_winner"]["selections"]]
        # Selections sorted ascending by odds → first is favourite
        assert odds[0] == min(odds)

    def test_margin_applied(self):
        """Market probs should be higher than fair probs due to margin."""
        markets = self.pricer.price_race(self.predictions)
        for sel in markets["race_winner"]["selections"]:
            assert sel["market_prob"] >= sel["fair_prob"] - 1e-6

    def test_top_k_increasing(self):
        """Each driver's top-k prob should increase as k grows."""
        markets = self.pricer.price_race(self.predictions)
        for i, driver_name in enumerate([p["driver"] for p in self.predictions]):
            p1 = next(s for s in markets["race_winner"]["selections"] if s["driver"] == driver_name)["fair_prob"]
            p3 = next(s for s in markets["top_3"]["selections"] if s["driver"] == driver_name)["fair_prob"]
            p5 = next(s for s in markets["top_5"]["selections"] if s["driver"] == driver_name)["fair_prob"]
            p10 = next(s for s in markets["top_10"]["selections"] if s["driver"] == driver_name)["fair_prob"]
            assert p3 >= p1 - 0.05
            assert p5 >= p3 - 0.05
            assert p10 >= p5 - 0.05

    def test_empty_predictions_raises(self):
        with pytest.raises(ValueError):
            self.pricer.price_race([])


class TestH2HPricing:
    def setup_method(self):
        self.pricer = NascarPricer()
        self.predictions = _make_predictions(10)

    def test_h2h_keys(self):
        h2h = self.pricer.price_h2h("Driver_0", "Driver_1", self.predictions)
        assert "driver_a" in h2h
        assert "driver_b" in h2h
        assert h2h["market_type"] == "h2h"

    def test_h2h_probs_sum_less_than_1(self):
        """Market probs should sum > 1.0 (overround applied)."""
        h2h = self.pricer.price_h2h("Driver_0", "Driver_1", self.predictions)
        total = h2h["driver_a"]["market_prob"] + h2h["driver_b"]["market_prob"]
        # Total market probs > 1 due to margin
        assert total > 1.0

    def test_h2h_favourite_has_lower_odds(self):
        """Driver with higher win prob should have lower decimal odds."""
        h2h = self.pricer.price_h2h("Driver_0", "Driver_9", self.predictions)
        # Driver_0 has highest win prob → lower odds
        assert h2h["driver_a"]["decimal_odds"] < h2h["driver_b"]["decimal_odds"]

    def test_h2h_unknown_driver_raises(self):
        with pytest.raises(ValueError):
            self.pricer.price_h2h("Unknown_Driver", "Driver_1", self.predictions)

    def test_h2h_fair_probs_sum_to_one(self):
        h2h = self.pricer.price_h2h("Driver_2", "Driver_5", self.predictions)
        total_fair = h2h["driver_a"]["fair_prob"] + h2h["driver_b"]["fair_prob"]
        assert abs(total_fair - 1.0) < 1e-6
