"""
Optic Odds feed adapter for NASCAR.
Fetches live odds from Optic Odds API (motorsports / nascar league).
Used for competitor price blending if available.

All public methods are async — uses httpx.AsyncClient to avoid blocking the
event loop.  The synchronous httpx.Client that previously blocked the async
context has been removed.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from config import OPTIC_ODDS_API_KEY, OPTIC_ODDS_BASE_URL

logger = logging.getLogger(__name__)

SPORT_ID = "motorsports"
# Optic Odds v3 confirmed slugs for NASCAR series (verified 2026-04-13)
LEAGUE_IDS = [
    "nascar_-_cup_series",
    "nascar_-_xfinity_series",
    "nascar_-_truck_series",
]
# Primary league for default upcoming query (Cup Series)
LEAGUE_ID = "nascar_-_cup_series"


class OpticOddsFeed:
    """
    Async wrapper around the Optic Odds v3 REST API for NASCAR.
    All methods handle errors gracefully — callers receive None on failure.
    """

    def __init__(self, api_key: str = OPTIC_ODDS_API_KEY) -> None:
        self._api_key = api_key
        self._base = OPTIC_ODDS_BASE_URL
        self._headers = {"X-Api-Key": api_key} if api_key else {}

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """Async HTTP GET helper — non-blocking, safe for FastAPI async handlers."""
        if not self._api_key:
            logger.debug("optic_odds_skipped no_api_key path=%s", path)
            return None
        url = f"{self._base}{path}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=self._headers, params=params or {})
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as exc:
            logger.warning("optic_odds_http_error path=%s error=%s", path, exc)
            return None
        except Exception as exc:
            logger.error("optic_odds_error path=%s error=%s", path, exc, exc_info=True)
            return None

    async def get_upcoming_races(self, series: str = "all") -> Optional[list[dict[str, Any]]]:
        """Fetch upcoming NASCAR races across Cup, Xfinity and Truck series.

        Args:
            series: "cup", "xfinity", "truck", or "all" (default).
        Returns:
            Combined list of fixture dicts from all requested series, or None on error.
        """
        if series == "all":
            leagues = LEAGUE_IDS
        elif series == "cup":
            leagues = ["nascar_-_cup_series"]
        elif series == "xfinity":
            leagues = ["nascar_-_xfinity_series"]
        elif series == "truck":
            leagues = ["nascar_-_truck_series"]
        else:
            leagues = [LEAGUE_ID]

        all_fixtures: list[dict[str, Any]] = []
        for league in leagues:
            data = await self._get(
                "/fixtures/active",
                params={"sport": SPORT_ID, "league": league},
            )
            if data is not None:
                all_fixtures.extend(data.get("data", []))
        return all_fixtures if all_fixtures else None

    async def get_race_odds(self, fixture_id: str) -> Optional[dict[str, Any]]:
        """Fetch race winner odds for a specific fixture."""
        data = await self._get(f"/fixtures/{fixture_id}/odds")
        if data is None:
            return None
        return data.get("data")

    async def get_race_odds_devigged(
        self,
        race_name: str,
        track: str,
        season: int,
        bookmaker: str = "pinnacle",
    ) -> Optional[dict[str, Any]]:
        """
        LOCK-NASCAR-TIER-2-CASCADE-001: Tier 2 fallback pricing.

        1. Discover the fixture ID for the named race via /fixtures/active search.
        2. Fetch Pinnacle outright winner odds from Optic Odds.
        3. Devig (remove overround) via ratio method → fair implied probabilities.
        4. Return structured market dict with prediction_source="market_scrape".

        Returns None when:
        - API key not configured
        - Fixture not found on Optic Odds
        - Pinnacle odds unavailable for this fixture
        - Fewer than 2 runners with odds (cannot form a market)

        Never raises — callers treat None as Tier 2 unavailable → Tier 3 refuse.
        """
        if not self._api_key:
            logger.debug(
                "nascar_tier2_skipped race=%s no_api_key "
                "LOCK-NASCAR-TIER-2-CASCADE-001",
                race_name,
            )
            return None

        # Step 1: Discover fixture by searching active fixtures for a name match.
        fixture_id: Optional[str] = None
        for league in LEAGUE_IDS:
            data = await self._get(
                "/fixtures/active",
                params={"sport": SPORT_ID, "league": league, "search": track},
            )
            if data is None:
                continue
            for fixture in data.get("data", []):
                fname = (fixture.get("name") or "").lower()
                if track.lower() in fname or race_name.lower() in fname:
                    fixture_id = fixture.get("id")
                    break
            if fixture_id:
                break

        if not fixture_id:
            logger.info(
                "nascar_tier2_fixture_not_found race=%s track=%s season=%d "
                "LOCK-NASCAR-TIER-2-CASCADE-001",
                race_name, track, season,
            )
            return None

        # Step 2: Fetch outright winner odds from Pinnacle via Optic Odds.
        odds_data = await self._get(
            "/fixtures/odds",
            params={
                "fixture_id": fixture_id,
                "market": "outright_winner",
                "sportsbook": bookmaker,
            },
        )
        if odds_data is None:
            logger.info(
                "nascar_tier2_odds_fetch_failed fixture_id=%s "
                "LOCK-NASCAR-TIER-2-CASCADE-001",
                fixture_id,
            )
            return None

        # Step 3: Parse runners and raw implied probabilities.
        raw_entries: list[tuple[str, float]] = []
        for entry in odds_data.get("data", []):
            for odd in entry.get("odds", []):
                market_id = odd.get("market_id", "")
                if market_id not in ("outright_winner", "winner", "race_winner"):
                    continue
                american = odd.get("price")
                name = odd.get("name") or odd.get("participant_name")
                if not name or american is None:
                    continue
                try:
                    a = float(american)
                    dec = (1 + a / 100) if a > 0 else (1 + 100 / abs(a))
                    if dec > 1.0:
                        raw_entries.append((str(name), 1.0 / dec))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue

        if len(raw_entries) < 2:
            logger.info(
                "nascar_tier2_insufficient_runners fixture_id=%s runners=%d "
                "LOCK-NASCAR-TIER-2-CASCADE-001",
                fixture_id, len(raw_entries),
            )
            return None

        # Step 4: Devig via ratio method (divide each implied prob by sum of all).
        total_implied = sum(p for _, p in raw_entries)
        if total_implied < 1e-9:
            return None

        fair_probs = [(name, p / total_implied) for name, p in raw_entries]
        fair_probs.sort(key=lambda x: x[1], reverse=True)

        selections = [
            {
                "driver": name,
                "fair_prob": round(fair_p, 6),
                "decimal_odds": round(1.0 / fair_p, 3) if fair_p > 0 else 999.99,
                "source": bookmaker,
            }
            for name, fair_p in fair_probs
        ]

        logger.info(
            "nascar_tier2_market_scrape_ok fixture_id=%s runners=%d "
            "bookmaker=%s overround=%.4f LOCK-NASCAR-TIER-2-CASCADE-001",
            fixture_id, len(selections), bookmaker, total_implied,
        )

        return {
            "race_winner": {
                "market_type": "race_winner",
                "source": bookmaker,
                "overround": round(total_implied, 4),
                "selections": selections,
            }
        }

    def is_available(self) -> bool:
        return bool(self._api_key)
