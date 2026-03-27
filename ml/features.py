"""
NASCAR Feature Extractor — chronological, zero-leakage.

For each driver-race entry:
  - ELO per surface type (dirt/paved/road) + overall ELO
  - Rolling win rate (last 5/15/36 races)
  - Rolling avg finish (last 5/15 races)
  - Rolling laps-led pct (last 5 races)
  - Rolling rating (last 5 races)
  - Career stats: total wins, total races
  - Track-specific avg start/finish
  - Manufacturer encoding
  - Decade encoding
  - Track length, surface encoded

ELO is updated AFTER feature extraction for each race (no leakage).
All rolling windows computed from rows strictly BEFORE the current race.
"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from config import ELO_DEFAULT, ELO_K, MIN_RACES_FOR_PREDICTION

logger = logging.getLogger(__name__)

# Canonical feature list — must match exactly between training and inference
FEATURES: list[str] = [
    # ELO features
    "elo_overall",
    "elo_surface",
    "elo_field_avg",
    "elo_vs_field",
    # Rolling win rates
    "win_rate_last5",
    "win_rate_last15",
    "win_rate_last36",
    # Rolling finish stats
    "avg_finish_last5",
    "avg_finish_last15",
    # Rolling laps led
    "laps_led_pct_last5",
    # Rolling rating
    "avg_rating_last5",
    # Career stats
    "career_wins",
    "career_races",
    "career_win_rate",
    # Track-specific history
    "track_avg_start",
    "track_avg_finish",
    "track_races",
    # Starting position
    "start_pos",
    "start_pos_normalized",
    # Manufacturer encoding
    "make_chevrolet",
    "make_ford",
    "make_toyota",
    "make_dodge",
    "make_other",
    # Surface encoding
    "surface_paved",
    "surface_dirt",
    "surface_road",
    # Decade encoding (era)
    "decade_2000s",
    "decade_2010s",
    "decade_2020s",
    # Track metadata
    "track_length",
]

# Surface types used in ELO splits
SURFACE_TYPES = ["dirt", "paved", "road"]

# Normalize mixed 'asphalt'/'concrete' → 'paved'
_SURFACE_NORM: dict[str, str] = {
    "dirt": "dirt",
    "paved": "paved",
    "asphalt": "paved",
    "concrete": "paved",
    "road": "road",
    "road course": "road",
}


def _norm_surface(s: Any) -> str:
    if pd.isna(s):
        return "paved"
    return _SURFACE_NORM.get(str(s).lower().strip(), "paved")


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _elo_update_winner(winner_elo: float, loser_elo: float, k: float = ELO_K) -> tuple[float, float]:
    expected_w = _elo_expected(winner_elo, loser_elo)
    new_winner = winner_elo + k * (1.0 - expected_w)
    new_loser = loser_elo + k * (0.0 - (1.0 - expected_w))
    return new_winner, new_loser


class _DriverState:
    """Per-driver mutable state accumulated chronologically."""

    __slots__ = [
        "elo_overall",
        "elo_by_surface",
        "recent_wins",
        "recent_finishes",
        "recent_laps_led_pct",
        "recent_ratings",
        "career_wins",
        "career_races",
        "track_starts",
        "track_finishes",
    ]

    def __init__(self) -> None:
        self.elo_overall: float = ELO_DEFAULT
        self.elo_by_surface: dict[str, float] = {s: ELO_DEFAULT for s in SURFACE_TYPES}
        # Deques for rolling windows
        self.recent_wins: deque[int] = deque(maxlen=36)
        self.recent_finishes: deque[float] = deque(maxlen=15)
        self.recent_laps_led_pct: deque[float] = deque(maxlen=5)
        self.recent_ratings: deque[float] = deque(maxlen=5)
        # Career counters
        self.career_wins: int = 0
        self.career_races: int = 0
        # Per-track history: track_id -> list of (start, finish)
        self.track_starts: dict[str, list[float]] = defaultdict(list)
        self.track_finishes: dict[str, list[float]] = defaultdict(list)


def _rolling_mean(dq: deque, n: int, default: float) -> float:
    vals = list(dq)[-n:]
    if not vals:
        return default
    return float(np.mean(vals))


class NascarFeatureExtractor:
    """
    Builds a feature matrix from cup_results.csv.
    Processes rows in chronological order (Season, Race) to avoid leakage.
    Features are extracted BEFORE updating ELO/rolling state for each race.
    """

    def __init__(self) -> None:
        self._driver_state: dict[str, _DriverState] = {}
        self.driver_count: int = 0
        self._fitted: bool = False

    def _get_or_create(self, driver: str) -> _DriverState:
        if driver not in self._driver_state:
            self._driver_state[driver] = _DriverState()
        return self._driver_state[driver]

    def build_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Read cup_results.csv, process chronologically, return feature DataFrame.
        Only Cup series rows. Filter to rows where Win is defined.
        """
        logger.info("Loading NASCAR Cup data from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)

        # Filter to Cup series only
        df = df[df["series"] == "cup"].copy()
        logger.info("Cup rows: %d", len(df))

        # Normalise columns
        df["Surface"] = df["Surface"].apply(_norm_surface)
        df["Win"] = pd.to_numeric(df["Win"], errors="coerce").fillna(0.0)
        df["Finish"] = pd.to_numeric(df["Finish"], errors="coerce")
        df["Start"] = pd.to_numeric(df["Start"], errors="coerce").fillna(20.0)
        df["Led"] = pd.to_numeric(df["Led"], errors="coerce").fillna(0.0)
        df["Laps"] = pd.to_numeric(df["Laps"], errors="coerce").fillna(0.0)
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df["Length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(1.0)
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
        df["Race"] = pd.to_numeric(df["Race"], errors="coerce")
        df["Make"] = df["Make"].fillna("Other")
        df["Driver"] = df["Driver"].fillna("Unknown").str.strip()

        # Drop rows with missing essential fields
        df = df.dropna(subset=["Season", "Race", "Finish", "Driver"])
        df = df[df["Finish"] > 0].copy()

        # Sort chronologically
        df = df.sort_values(["Season", "Race", "Finish"]).reset_index(drop=True)

        logger.info("Processing %d rows chronologically ...", len(df))

        records: list[dict] = []

        # Group by (Season, Race) to process full race at a time
        for (season, race_num), race_df in df.groupby(["Season", "Race"], sort=True):
            race_id = f"{int(season)}_{int(race_num)}"
            surface = race_df["Surface"].iloc[0]
            track = str(race_df["Track"].iloc[0]) if pd.notna(race_df["Track"].iloc[0]) else "unknown"
            track_len = float(race_df["Length"].iloc[0]) if pd.notna(race_df["Length"].iloc[0]) else 1.0

            # Field ELO average BEFORE this race
            field_elos_overall = [
                self._get_or_create(d).elo_overall
                for d in race_df["Driver"].tolist()
            ]
            field_elo_avg = float(np.mean(field_elos_overall)) if field_elos_overall else ELO_DEFAULT

            # Field size for normalising start position
            field_size = len(race_df)

            # Extract features for each driver BEFORE updating state
            race_records: list[dict] = []
            for _, row in race_df.iterrows():
                driver = str(row["Driver"])
                st = self._get_or_create(driver)

                # Extract features
                surf_elo = st.elo_by_surface.get(surface, ELO_DEFAULT)
                cr = st.career_races

                feat: dict[str, Any] = {
                    "race_id": race_id,
                    "season": int(season),
                    "race_num": int(race_num),
                    "driver": driver,
                    "track": track,
                    "surface": surface,
                    # ELO
                    "elo_overall": st.elo_overall,
                    "elo_surface": surf_elo,
                    "elo_field_avg": field_elo_avg,
                    "elo_vs_field": st.elo_overall - field_elo_avg,
                    # Rolling win rates
                    "win_rate_last5": _rolling_mean(st.recent_wins, 5, 0.0),
                    "win_rate_last15": _rolling_mean(st.recent_wins, 15, 0.0),
                    "win_rate_last36": _rolling_mean(st.recent_wins, 36, 0.0),
                    # Rolling finish
                    "avg_finish_last5": _rolling_mean(st.recent_finishes, 5, 15.0),
                    "avg_finish_last15": _rolling_mean(st.recent_finishes, 15, 15.0),
                    # Rolling laps led
                    "laps_led_pct_last5": _rolling_mean(st.recent_laps_led_pct, 5, 0.0),
                    # Rolling rating
                    "avg_rating_last5": _rolling_mean(st.recent_ratings, 5, 75.0),
                    # Career
                    "career_wins": st.career_wins,
                    "career_races": cr,
                    "career_win_rate": (st.career_wins / cr) if cr >= 1 else 0.0,
                    # Track history
                    "track_avg_start": float(np.mean(st.track_starts[track])) if st.track_starts[track] else float(row["Start"]),
                    "track_avg_finish": float(np.mean(st.track_finishes[track])) if st.track_finishes[track] else 15.0,
                    "track_races": len(st.track_finishes[track]),
                    # Starting position
                    "start_pos": float(row["Start"]),
                    "start_pos_normalized": float(row["Start"]) / max(field_size, 1),
                    # Manufacturer
                    "make_chevrolet": 1.0 if str(row["Make"]).lower() == "chevrolet" else 0.0,
                    "make_ford": 1.0 if str(row["Make"]).lower() == "ford" else 0.0,
                    "make_toyota": 1.0 if str(row["Make"]).lower() == "toyota" else 0.0,
                    "make_dodge": 1.0 if str(row["Make"]).lower() == "dodge" else 0.0,
                    "make_other": 1.0 if str(row["Make"]).lower() not in ("chevrolet", "ford", "toyota", "dodge") else 0.0,
                    # Surface
                    "surface_paved": 1.0 if surface == "paved" else 0.0,
                    "surface_dirt": 1.0 if surface == "dirt" else 0.0,
                    "surface_road": 1.0 if surface == "road" else 0.0,
                    # Decade
                    "decade_2000s": 1.0 if 2000 <= int(season) <= 2009 else 0.0,
                    "decade_2010s": 1.0 if 2010 <= int(season) <= 2019 else 0.0,
                    "decade_2020s": 1.0 if int(season) >= 2020 else 0.0,
                    # Track metadata
                    "track_length": track_len,
                    # Target
                    "target": int(row["Win"]),
                }
                race_records.append((driver, row, feat))

            # NOW update ELO and rolling state for all drivers in this race
            # ELO update: winner vs field (simplified: winner beats all others; pairwise)
            winner_row = race_df[race_df["Win"] == 1.0]
            if not winner_row.empty:
                winner_driver = str(winner_row.iloc[0]["Driver"])
                winner_st = self._get_or_create(winner_driver)
                for _, row in race_df.iterrows():
                    if str(row["Driver"]) == winner_driver:
                        continue
                    loser_st = self._get_or_create(str(row["Driver"]))
                    # Overall ELO
                    new_w, new_l = _elo_update_winner(
                        winner_st.elo_overall, loser_st.elo_overall, k=ELO_K
                    )
                    winner_st.elo_overall = new_w
                    loser_st.elo_overall = new_l
                    # Surface ELO
                    surf_w = winner_st.elo_by_surface.get(surface, ELO_DEFAULT)
                    surf_l = loser_st.elo_by_surface.get(surface, ELO_DEFAULT)
                    new_sw, new_sl = _elo_update_winner(surf_w, surf_l, k=ELO_K)
                    winner_st.elo_by_surface[surface] = new_sw
                    loser_st.elo_by_surface[surface] = new_sl

            # Update rolling stats for all drivers
            for driver, row, feat in race_records:
                st = self._get_or_create(driver)
                finish = float(row["Finish"])
                laps = float(row["Laps"]) if pd.notna(row["Laps"]) else 0.0
                led = float(row["Led"]) if pd.notna(row["Led"]) else 0.0
                is_win = int(row["Win"])
                rating = float(row["Rating"]) if pd.notna(row["Rating"]) else 75.0

                st.recent_wins.append(is_win)
                st.recent_finishes.append(finish)
                st.recent_laps_led_pct.append(led / max(laps, 1))
                st.recent_ratings.append(rating)
                st.career_races += 1
                st.career_wins += is_win
                st.track_starts[feat["track"]].append(float(row["Start"]))
                st.track_finishes[feat["track"]].append(finish)

                records.append(feat)

        self.driver_count = len(self._driver_state)
        self._fitted = True
        logger.info(
            "Feature extraction complete: %d records, %d drivers tracked",
            len(records), self.driver_count,
        )
        result = pd.DataFrame(records)
        return result

    def get_features_for_race(
        self,
        drivers: list[dict],
        track: str,
        surface: str,
        season: int,
        track_length: float = 1.5,
    ) -> list[dict[str, Any]]:
        """
        Build feature dicts for inference (no state update).
        Each driver dict should have: name, team, make, starting_pos (optional).
        Uses the driver state accumulated during training.
        """
        if not self._fitted:
            raise RuntimeError("Extractor not fitted — build_dataset() must be called first")

        surface_norm = _norm_surface(surface)
        field_size = len(drivers)
        field_elos = [
            self._driver_state.get(d.get("name", ""), _DriverState()).elo_overall
            for d in drivers
        ]
        field_elo_avg = float(np.mean(field_elos)) if field_elos else ELO_DEFAULT

        results: list[dict] = []
        for i, driver_info in enumerate(drivers):
            driver_name = str(driver_info.get("name", "")).strip()
            st = self._driver_state.get(driver_name, _DriverState())
            surf_elo = st.elo_by_surface.get(surface_norm, ELO_DEFAULT)
            cr = st.career_races
            make = str(driver_info.get("make", "Other")).lower()
            start_pos = float(driver_info.get("starting_pos", i + 1))

            feat: dict[str, Any] = {
                "driver": driver_name,
                "elo_overall": st.elo_overall,
                "elo_surface": surf_elo,
                "elo_field_avg": field_elo_avg,
                "elo_vs_field": st.elo_overall - field_elo_avg,
                "win_rate_last5": _rolling_mean(st.recent_wins, 5, 0.0),
                "win_rate_last15": _rolling_mean(st.recent_wins, 15, 0.0),
                "win_rate_last36": _rolling_mean(st.recent_wins, 36, 0.0),
                "avg_finish_last5": _rolling_mean(st.recent_finishes, 5, 15.0),
                "avg_finish_last15": _rolling_mean(st.recent_finishes, 15, 15.0),
                "laps_led_pct_last5": _rolling_mean(st.recent_laps_led_pct, 5, 0.0),
                "avg_rating_last5": _rolling_mean(st.recent_ratings, 5, 75.0),
                "career_wins": st.career_wins,
                "career_races": cr,
                "career_win_rate": (st.career_wins / cr) if cr >= 1 else 0.0,
                "track_avg_start": float(np.mean(st.track_starts[track])) if st.track_starts[track] else start_pos,
                "track_avg_finish": float(np.mean(st.track_finishes[track])) if st.track_finishes[track] else 15.0,
                "track_races": len(st.track_finishes[track]),
                "start_pos": start_pos,
                "start_pos_normalized": start_pos / max(field_size, 1),
                "make_chevrolet": 1.0 if make == "chevrolet" else 0.0,
                "make_ford": 1.0 if make == "ford" else 0.0,
                "make_toyota": 1.0 if make == "toyota" else 0.0,
                "make_dodge": 1.0 if make == "dodge" else 0.0,
                "make_other": 1.0 if make not in ("chevrolet", "ford", "toyota", "dodge") else 0.0,
                "surface_paved": 1.0 if surface_norm == "paved" else 0.0,
                "surface_dirt": 1.0 if surface_norm == "dirt" else 0.0,
                "surface_road": 1.0 if surface_norm == "road" else 0.0,
                "decade_2000s": 1.0 if 2000 <= season <= 2009 else 0.0,
                "decade_2010s": 1.0 if 2010 <= season <= 2019 else 0.0,
                "decade_2020s": 1.0 if season >= 2020 else 0.0,
                "track_length": track_length,
            }
            results.append(feat)

        return results

    def get_top_elo_drivers(self, n: int = 50) -> list[dict]:
        """Return top-N drivers by overall ELO for admin endpoint."""
        ranked = sorted(
            [
                {
                    "driver": d,
                    "elo_overall": round(st.elo_overall, 1),
                    "elo_dirt": round(st.elo_by_surface.get("dirt", ELO_DEFAULT), 1),
                    "elo_paved": round(st.elo_by_surface.get("paved", ELO_DEFAULT), 1),
                    "elo_road": round(st.elo_by_surface.get("road", ELO_DEFAULT), 1),
                    "career_wins": st.career_wins,
                    "career_races": st.career_races,
                }
                for d, st in self._driver_state.items()
                if st.career_races >= MIN_RACES_FOR_PREDICTION
            ],
            key=lambda x: x["elo_overall"],
            reverse=True,
        )
        return ranked[:n]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("NascarFeatureExtractor saved to %s (drivers=%d)", path, self.driver_count)

    @classmethod
    def load(cls, path: str) -> "NascarFeatureExtractor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("NascarFeatureExtractor loaded from %s (drivers=%d)", path, obj.driver_count)
        return obj
