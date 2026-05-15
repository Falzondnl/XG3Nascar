"""
LOCK-NASCAR-TIER-2B-REVERSE-ENGINEER-001
==========================================

Regression tests for NASCAR Tier 2B PL reverse-engineer.
"""
from __future__ import annotations

import numpy as np
import pytest

from pricing.tier2b_reverse_engineer import (
    NASCARTier2BReverseEngineer,
    devig_outright_market,
    plackett_luce_inverse,
    _gumbel_max_top_k,
    get_tier2b_engineer,
)


def test_pl_inverse_round_trip():
    fair = {"larson": 0.30, "byron": 0.25, "blaney": 0.20, "bell": 0.15, "elliot": 0.10}
    ids, skills, recon = plackett_luce_inverse(fair)
    for did, p in zip(ids, recon):
        assert abs(p - fair[did]) < 1e-12


def test_top_k_sums_to_k():
    skills = np.array([8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5], dtype=np.float64)
    for k in [1, 3, 5]:
        marg = _gumbel_max_top_k(skills, k)
        assert abs(marg.sum() - k) < 1e-9


def test_engineer_cup_race():
    odds = {
        "larson": 6.0, "byron": 7.0, "bell": 9.0, "blaney": 9.0,
        "elliot": 10.0, "truex": 12.0, "hamlin": 14.0, "busch": 17.0,
        "logano": 19.0, "buescher": 26.0, "reddick": 26.0, "ragan": 31.0,
    }
    eng = NASCARTier2BReverseEngineer()
    r = eng.reverse_engineer("daytona_2026", odds, series="cup")
    assert r.solver_converged
    assert r.prediction_source == "market_scrape_reverse_engineered"
    assert r.series == "cup"
    assert abs(r.top_5_probs().sum() - 5.0) < 1e-9
    assert abs(r.top_10_probs().sum() - 10.0) < 1e-9


def test_thin_field_raises():
    eng = NASCARTier2BReverseEngineer()
    with pytest.raises(ValueError):
        eng.reverse_engineer("thin", {"a": 2.0, "b": 3.0})


def test_singleton():
    assert get_tier2b_engineer() is get_tier2b_engineer()
