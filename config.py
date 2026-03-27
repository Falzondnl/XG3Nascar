"""NASCAR Microservice Configuration."""
from __future__ import annotations

import os

PORT = int(os.getenv("PORT", "8031"))
SERVICE_NAME = "nascar"
SERVICE_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

NASCAR_CSV = os.getenv(
    "NASCAR_CSV",
    "D:/codex/Data/motorsports/tier1/curated/nascar/cup_results.csv",
)
NASCAR_STANDINGS_CSV = os.getenv(
    "NASCAR_STANDINGS_CSV",
    "D:/codex/Data/motorsports/tier1/curated/nascar/nascar_cup_standings_1975_2025.csv",
)
NASCAR_DRIVER_SUMMARY_CSV = os.getenv(
    "NASCAR_DRIVER_SUMMARY_CSV",
    "D:/codex/Data/motorsports/tier1/curated/nascar/cup_driver_season_summary.csv",
)

R0_DIR = os.getenv("R0_DIR", "models/r0")
R1_DIR = os.getenv("R1_DIR", "models/r1")
R2_DIR = os.getenv("R2_DIR", "models/r2")

# Training temporal splits (Cup era 2000-2025)
TRAIN_SEASONS_MIN = 2000
TRAIN_SEASONS_MAX = 2018
VAL_SEASONS_MIN = 2019
VAL_SEASONS_MAX = 2021
TEST_SEASONS_MIN = 2022

# ELO system
ELO_K = 32
ELO_DEFAULT = 1500.0
MIN_RACES_FOR_PREDICTION = 5  # min career races before ELO is meaningful

# Harville multi-outcome DP
HARVILLE_TOP_N = 40  # max field size for Harville exact DP

# Optic Odds
OPTIC_ODDS_API_KEY = os.getenv("OPTIC_ODDS_API_KEY", "")
OPTIC_ODDS_BASE_URL = "https://api.opticodds.com/api/v3"

# Margin config (sportsbook margins per market type)
MARGIN_RACE_WINNER = 0.12   # 12% — wide field
MARGIN_TOP_3 = 0.10
MARGIN_TOP_5 = 0.10
MARGIN_TOP_10 = 0.08
MARGIN_H2H = 0.05
