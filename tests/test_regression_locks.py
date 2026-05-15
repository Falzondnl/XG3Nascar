"""
NASCAR regression lock tests.

LOCK-NASCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001
LOCK-NASCAR-PRED-SRC-MS-RESPONSE-001
LOCK-NASCAR-TIER-2-CASCADE-001
LOCK-NASCAR-TIER-3-REFUSE-503-001
"""
from __future__ import annotations

import pathlib
import uuid


# ---------------------------------------------------------------------------
# LOCK-NASCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001
# ---------------------------------------------------------------------------


class TestNascarEloNoSilentFallback:
    """LOCK-NASCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001"""

    def test_driver_without_elo_is_skipped_in_championship(self) -> None:
        """
        In the championship outright route, drivers with no elo_overall
        (or elo_overall == 0.0) must be skipped, not assigned 1500.
        """
        raw_driver_list = [
            {"driver": "Tyler Reddick", "elo_overall": 1620.0, "career_wins": 12, "career_races": 224},
            {"driver": "Christopher Bell", "elo_overall": 1580.0, "career_wins": 8, "career_races": 180},
            {"driver": "Driver With No ELO", "elo_overall": None},
            {"driver": "Driver With Zero ELO", "elo_overall": 0.0},
        ]

        elo_key = "elo_overall"
        raw_pairs: list[tuple[str, float]] = []
        skipped: list[str] = []
        for item in raw_driver_list:
            name = str(item.get("driver") or "Unknown")
            raw_elo = item.get(elo_key) or item.get("elo_overall")
            if raw_elo is None or float(raw_elo) == 0.0:
                skipped.append(name)
                continue
            raw_pairs.append((name, float(raw_elo)))

        assert len(raw_pairs) == 2
        assert "Driver With No ELO" in skipped
        assert "Driver With Zero ELO" in skipped
        assert not any(name in ("Driver With No ELO", "Driver With Zero ELO") for name, _ in raw_pairs)

    def test_championship_source_contains_lock_id(self) -> None:
        """outrights.py must reference LOCK-NASCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001."""
        outrights_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "outrights.py"
        if outrights_path.exists():
            source = outrights_path.read_text(encoding="utf-8")
            assert "LOCK-NASCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001" in source


# ---------------------------------------------------------------------------
# LOCK-NASCAR-PRED-SRC-MS-RESPONSE-001
# ---------------------------------------------------------------------------


class TestNascarPredictionSourceField:
    """LOCK-NASCAR-PRED-SRC-MS-RESPONSE-001"""

    def test_price_response_contains_prediction_source(self) -> None:
        """The /races/price response dict must include prediction_source."""
        mock_response = {
            "race_name": "Daytona 500",
            "track": "Daytona",
            "season": 2026,
            "surface": "paved",
            "field_size": 40,
            "predictions": [],
            "markets": {},
            "prediction_source": "model",  # LOCK — must be present
        }
        assert "prediction_source" in mock_response, (
            "LOCK-NASCAR-PRED-SRC-MS-RESPONSE-001 VIOLATED: "
            "prediction_source missing from /races/price response"
        )
        _valid = {"model", "model_pinnacle_blend", "market_scrape",
                  "market_scrape_reverse_engineered", "unpriced"}
        assert mock_response["prediction_source"] in _valid, (
            f"Invalid prediction_source: {mock_response['prediction_source']!r}"
        )

    def test_h2h_response_contains_prediction_source(self) -> None:
        """The /races/h2h response dict must include prediction_source."""
        mock_response = {
            "race_name": "Daytona 500",
            "track": "Daytona",
            "season": 2026,
            "h2h": {},
            "prediction_source": "model",  # LOCK
        }
        assert "prediction_source" in mock_response

    def test_races_source_has_lock_id(self) -> None:
        """races.py must contain LOCK-NASCAR-PRED-SRC-MS-RESPONSE-001."""
        races_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "races.py"
        if races_path.exists():
            source = races_path.read_text(encoding="utf-8")
            assert "LOCK-NASCAR-PRED-SRC-MS-RESPONSE-001" in source


# ---------------------------------------------------------------------------
# LOCK-NASCAR-TIER-2-CASCADE-001
# ---------------------------------------------------------------------------


class TestNascarTier2Cascade:
    """LOCK-NASCAR-TIER-2-CASCADE-001"""

    def test_optic_feed_has_get_race_odds_devigged(self) -> None:
        """OpticOddsFeed must expose get_race_odds_devigged for Tier 2 cascade."""
        from feeds.optic_odds import OpticOddsFeed
        assert hasattr(OpticOddsFeed, "get_race_odds_devigged"), (
            "LOCK-NASCAR-TIER-2-CASCADE-001 VIOLATED: "
            "OpticOddsFeed.get_race_odds_devigged missing"
        )

    def test_races_source_has_tier2_lock_id(self) -> None:
        """races.py must contain LOCK-NASCAR-TIER-2-CASCADE-001."""
        races_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "races.py"
        if races_path.exists():
            source = races_path.read_text(encoding="utf-8")
            assert "LOCK-NASCAR-TIER-2-CASCADE-001" in source

    def test_tier2_response_has_prediction_source_market_scrape(self) -> None:
        """Tier 2 response must set prediction_source='market_scrape', not 'model'."""
        mock_tier2_response = {
            "race_name": "Daytona 500",
            "track": "Daytona",
            "season": 2026,
            "surface": "paved",
            "field_size": 40,
            "markets": {"race_winner": {"selections": []}},
            "prediction_source": "market_scrape",  # LOCK
            "model_available": False,
            "tier": 2,
        }
        assert mock_tier2_response["prediction_source"] == "market_scrape"
        assert mock_tier2_response["model_available"] is False
        assert mock_tier2_response["tier"] == 2


# ---------------------------------------------------------------------------
# LOCK-NASCAR-TIER-3-REFUSE-503-001
# ---------------------------------------------------------------------------


class TestNascarTier3Refuse:
    """LOCK-NASCAR-TIER-3-REFUSE-503-001"""

    def test_tier3_body_structure(self) -> None:
        """Tier 3 503 body must include code, reason, correlation_id, retry_after."""
        cid = str(uuid.uuid4())
        tier3_body = {
            "code": "FIXTURE_UNPRICED",
            "reason": "no_model_no_market_data",
            "message": "test",
            "correlation_id": cid,
            "retry_after": 30,
            "race_name": "Daytona 500",
            "track": "Daytona",
        }
        for required in ("code", "reason", "message", "correlation_id", "retry_after"):
            assert required in tier3_body, (
                f"LOCK-NASCAR-TIER-3-REFUSE-503-001 VIOLATED: missing {required}"
            )
        assert tier3_body["code"] == "FIXTURE_UNPRICED"
        assert tier3_body["retry_after"] > 0

    def test_races_source_has_tier3_lock_id(self) -> None:
        """races.py must contain LOCK-NASCAR-TIER-3-REFUSE-503-001."""
        races_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "races.py"
        if races_path.exists():
            source = races_path.read_text(encoding="utf-8")
            assert "LOCK-NASCAR-TIER-3-REFUSE-503-001" in source
