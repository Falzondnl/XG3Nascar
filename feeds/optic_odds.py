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

    def is_available(self) -> bool:
        return bool(self._api_key)
