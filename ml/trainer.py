"""
NASCAR Model Trainer
Temporal split: train=2000-2018, val=2019-2021, test=2022-2025
Cup series only. 3-model stacking ensemble + BetaCalibrator.
Saves to models/r0/: ensemble.pkl, calibrator.pkl, extractor.pkl
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import (
    NASCAR_CSV,
    R0_DIR,
    TRAIN_SEASONS_MAX,
    TRAIN_SEASONS_MIN,
    VAL_SEASONS_MAX,
    VAL_SEASONS_MIN,
    TEST_SEASONS_MIN,
)
from ml.calibrator import BetaCalibrator
from ml.ensemble import NascarEnsemble
from ml.features import FEATURES, NascarFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train(csv_path: str = NASCAR_CSV, out_dir: str = R0_DIR) -> dict:
    """
    Full training pipeline for NASCAR Cup win prediction.
    Returns dict with AUC, Brier, split sizes.
    """
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Feature extraction (chronological, full dataset 1949-2025)
    # ------------------------------------------------------------------
    logger.info("=== STEP 1: Feature extraction (all years, no leakage) ===")
    extractor = NascarFeatureExtractor()
    dataset = extractor.build_dataset(csv_path)

    if dataset.empty:
        raise RuntimeError("Feature extraction returned empty dataset — check CSV path")

    logger.info(
        "Full dataset: %d rows, %d positive (win rate=%.4f), %d drivers",
        len(dataset),
        int(dataset["target"].sum()),
        dataset["target"].mean(),
        extractor.driver_count,
    )

    # ------------------------------------------------------------------
    # 2. Temporal split — train on 2000-2018 only (modern era)
    # ------------------------------------------------------------------
    logger.info(
        "=== STEP 2: Temporal split — train=%d-%d val=%d-%d test=%d+ ===",
        TRAIN_SEASONS_MIN, TRAIN_SEASONS_MAX,
        VAL_SEASONS_MIN, VAL_SEASONS_MAX,
        TEST_SEASONS_MIN,
    )
    train_df = dataset[
        (dataset["season"] >= TRAIN_SEASONS_MIN) & (dataset["season"] <= TRAIN_SEASONS_MAX)
    ].copy()
    val_df = dataset[
        (dataset["season"] >= VAL_SEASONS_MIN) & (dataset["season"] <= VAL_SEASONS_MAX)
    ].copy()
    test_df = dataset[dataset["season"] >= TEST_SEASONS_MIN].copy()

    logger.info(
        "Split sizes — train=%d (win_rate=%.4f) val=%d (win_rate=%.4f) test=%d (win_rate=%.4f)",
        len(train_df), train_df["target"].mean(),
        len(val_df), val_df["target"].mean(),
        len(test_df), test_df["target"].mean(),
    )

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("One or more splits empty — check season filtering")

    X_train = train_df[FEATURES].copy()
    y_train = train_df["target"].astype(int)
    groups_train = train_df["race_id"]

    X_val = val_df[FEATURES].copy()
    y_val = val_df["target"].astype(int)

    X_test = test_df[FEATURES].copy()
    y_test = test_df["target"].astype(int)

    # ------------------------------------------------------------------
    # 3. Ensemble training
    # ------------------------------------------------------------------
    logger.info("=== STEP 3: Training 3-model ensemble ===")
    ensemble = NascarEnsemble()
    ensemble.fit(X_train, y_train, groups_train, X_val, y_val)

    # ------------------------------------------------------------------
    # 4. Calibration on validation set
    # ------------------------------------------------------------------
    logger.info("=== STEP 4: BetaCalibrator on validation set ===")
    val_raw = ensemble.predict_proba(X_val)
    calibrator = BetaCalibrator()
    calibrator.fit(val_raw, y_val.values)

    # ------------------------------------------------------------------
    # 5. Evaluation on test set
    # ------------------------------------------------------------------
    logger.info("=== STEP 5: Test set evaluation ===")
    test_raw = ensemble.predict_proba(X_test)
    test_cal = calibrator.calibrate(test_raw)

    auc_raw = roc_auc_score(y_test, test_raw)
    auc_cal = roc_auc_score(y_test, test_cal)
    brier_raw = brier_score_loss(y_test, test_raw)
    brier_cal = brier_score_loss(y_test, test_cal)

    # Normalised within-race probabilities (sum to 1 per race)
    test_df = test_df.copy()
    test_df["raw_prob"] = test_raw
    test_df["cal_prob"] = test_cal
    race_totals = test_df.groupby("race_id")["cal_prob"].transform("sum")
    test_df["norm_prob"] = test_df["cal_prob"] / race_totals.clip(lower=1e-9)

    # Only evaluate on races where at least one winner is present
    test_df_with_winner = test_df[test_df.groupby("race_id")["target"].transform("sum") >= 1]
    if len(test_df_with_winner) > 0:
        auc_norm = roc_auc_score(
            test_df_with_winner["target"],
            test_df_with_winner["norm_prob"],
        )
        brier_norm = brier_score_loss(
            test_df_with_winner["target"],
            test_df_with_winner["norm_prob"],
        )
    else:
        auc_norm = auc_cal
        brier_norm = brier_cal

    logger.info(
        "=== RESULTS === "
        "Raw AUC=%.4f Brier=%.4f | "
        "Calibrated AUC=%.4f Brier=%.4f | "
        "Normalised AUC=%.4f Brier=%.4f",
        auc_raw, brier_raw,
        auc_cal, brier_cal,
        auc_norm, brier_norm,
    )

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
    logger.info("=== STEP 6: Saving artefacts to %s ===", out_dir)
    ensemble.save(os.path.join(out_dir, "ensemble.pkl"))
    calibrator.save(os.path.join(out_dir, "calibrator.pkl"))
    extractor.save(os.path.join(out_dir, "extractor.pkl"))

    elapsed = time.time() - t_start
    logger.info("Training complete in %.1fs", elapsed)

    metrics = {
        "auc_raw": round(float(auc_raw), 4),
        "brier_raw": round(float(brier_raw), 4),
        "auc_calibrated": round(float(auc_cal), 4),
        "brier_calibrated": round(float(brier_cal), 4),
        "auc_normalised": round(float(auc_norm), 4),
        "brier_normalised": round(float(brier_norm), 4),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_drivers": extractor.driver_count,
        "win_rate_train": round(float(y_train.mean()), 4),
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info("Metrics: %s", metrics)
    return metrics


class NascarTrainer:
    """Convenience class wrapping the train() function."""

    def __init__(self, csv_path: str = NASCAR_CSV, out_dir: str = R0_DIR) -> None:
        self.csv_path = csv_path
        self.out_dir = out_dir

    def train(self) -> dict:
        return train(csv_path=self.csv_path, out_dir=self.out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NASCAR Cup ML models")
    parser.add_argument("--csv", default=NASCAR_CSV, help="Path to cup_results.csv")
    parser.add_argument("--out", default=R0_DIR, help="Output directory for model pkl files")
    args = parser.parse_args()

    metrics = train(csv_path=args.csv, out_dir=args.out)
    print("\n=== FINAL METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
