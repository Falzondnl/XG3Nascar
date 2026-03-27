"""
NascarPredictor — Production inference engine.
Loads trained ensemble + calibrator + feature extractor.
Handles per-race win probability prediction with Harville market derivation.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from config import R0_DIR
from ml.features import FEATURES, NascarFeatureExtractor
from ml.ensemble import NascarEnsemble
from ml.calibrator import BetaCalibrator

logger = logging.getLogger(__name__)

ENSEMBLE_PKL = "ensemble.pkl"
CALIBRATOR_PKL = "calibrator.pkl"
EXTRACTOR_PKL = "extractor.pkl"


class NascarPredictor:
    """
    Production NASCAR race predictor.
    Loads from R0_DIR by default.
    """

    def __init__(self) -> None:
        self.ensemble: NascarEnsemble | None = None
        self.calibrator: BetaCalibrator | None = None
        self.extractor: NascarFeatureExtractor | None = None
        self._model_dir: str = R0_DIR
        self._loaded = False

    def load(self, model_dir: str = R0_DIR) -> "NascarPredictor":
        """Load all model artefacts from model_dir."""
        self._model_dir = model_dir

        ens_path = os.path.join(model_dir, ENSEMBLE_PKL)
        cal_path = os.path.join(model_dir, CALIBRATOR_PKL)
        ext_path = os.path.join(model_dir, EXTRACTOR_PKL)

        if not os.path.exists(ens_path):
            raise FileNotFoundError(f"Ensemble not found at: {ens_path}")
        if not os.path.exists(cal_path):
            raise FileNotFoundError(f"Calibrator not found at: {cal_path}")
        if not os.path.exists(ext_path):
            raise FileNotFoundError(f"Extractor not found at: {ext_path}")

        self.ensemble = NascarEnsemble.load(ens_path)
        self.calibrator = BetaCalibrator.load(cal_path)
        self.extractor = NascarFeatureExtractor.load(ext_path)
        self._loaded = True

        logger.info(
            "NascarPredictor loaded from %s (drivers=%d)",
            model_dir,
            self.extractor.driver_count,
        )
        return self

    def predict_race(
        self,
        drivers: list[dict],
        track: str,
        surface: str,
        season: int,
        track_length: float = 1.5,
    ) -> list[dict[str, Any]]:
        """
        Predict win probabilities for a race field.

        drivers: list of dicts with keys:
            - name (str): driver full name matching training data
            - team (str, optional): team name
            - make (str, optional): manufacturer
            - starting_pos (int, optional): qualifying grid position

        Returns list sorted by win_prob descending:
            [{driver, win_prob, top3_prob, top5_prob, top10_prob}]
        """
        if not self._loaded:
            raise RuntimeError("NascarPredictor not loaded — call load() first")
        if not drivers:
            raise ValueError("drivers list cannot be empty")

        # Build features for all drivers
        feature_dicts = self.extractor.get_features_for_race(
            drivers, track=track, surface=surface, season=season, track_length=track_length
        )
        if not feature_dicts:
            raise RuntimeError("Feature extractor returned empty result for provided drivers")

        X = pd.DataFrame([{k: v for k, v in fd.items() if k in FEATURES} for fd in feature_dicts])

        # Raw ensemble probabilities
        raw_probs = self.ensemble.predict_proba(X)

        # Calibrate
        cal_probs = self.calibrator.calibrate(raw_probs)

        # Normalise to sum=1 (softmax via normalisation)
        total = cal_probs.sum()
        if total < 1e-9:
            cal_probs = np.ones(len(cal_probs)) / len(cal_probs)
        else:
            cal_probs = cal_probs / total

        results = []
        for i, fd in enumerate(feature_dicts):
            results.append({
                "driver": fd.get("driver", f"Driver_{i}"),
                "win_prob": float(cal_probs[i]),
                "raw_win_prob": float(raw_probs[i]),
                "elo_overall": float(fd.get("elo_overall", 1500.0)),
                "career_wins": int(fd.get("career_wins", 0)),
                "career_races": int(fd.get("career_races", 0)),
                "win_rate_last15": float(fd.get("win_rate_last15", 0.0)),
                "avg_finish_last5": float(fd.get("avg_finish_last5", 15.0)),
            })

        results.sort(key=lambda x: x["win_prob"], reverse=True)
        return results

    def get_top_elo_drivers(self, n: int = 50) -> list[dict]:
        if not self._loaded:
            raise RuntimeError("Predictor not loaded")
        return self.extractor.get_top_elo_drivers(n)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def driver_count(self) -> int:
        if self.extractor:
            return self.extractor.driver_count
        return 0
