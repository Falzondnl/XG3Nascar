"""
NASCAR 3-Model Stacking Ensemble
CatBoost + LightGBM + XGBoost → LogisticRegression meta-learner
GroupKFold on race_id to prevent within-race data leakage.
class_weight='balanced' on all base models due to ~2.8% win rate.
"""
from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ml.features import FEATURES

logger = logging.getLogger(__name__)

CB_PARAMS: dict[str, Any] = {
    "iterations": 400,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 50,
    "auto_class_weights": "Balanced",
}

LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "class_weight": "balanced",
}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
    "scale_pos_weight": 35,  # ~1/win_rate to balance classes
}

N_SPLITS = 5


class NascarEnsemble:
    """
    3-model stacking ensemble for win prediction in NASCAR Cup races.
    Base: CatBoost, LightGBM, XGBoost — all with class balancing.
    Meta: LogisticRegression on OOF predictions.
    """

    def __init__(self) -> None:
        self.cb_model: Any = None
        self.lgb_model: Any = None
        self.xgb_model: Any = None
        self.meta: LogisticRegression | None = None
        self._feature_names = FEATURES

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "NascarEnsemble":
        """
        Fit base models using val set for early stopping.
        Build OOF predictions via GroupKFold, then fit meta-learner.
        """
        from catboost import CatBoostClassifier, Pool
        import lightgbm as lgb
        import xgboost as xgb

        logger.info(
            "NascarEnsemble.fit: train=%d val=%d features=%d pos_rate_train=%.4f",
            len(X_train), len(X_val), len(FEATURES), y_train.mean(),
        )

        # ---- CatBoost ----
        logger.info("Training CatBoost ...")
        cb = CatBoostClassifier(**CB_PARAMS)
        train_pool = Pool(X_train[FEATURES], label=y_train)
        val_pool = Pool(X_val[FEATURES], label=y_val)
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        self.cb_model = cb
        logger.info("CatBoost done. Best iteration: %d", cb.get_best_iteration())

        # ---- LightGBM ----
        logger.info("Training LightGBM ...")
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(
            X_train[FEATURES], y_train,
            eval_set=[(X_val[FEATURES], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        self.lgb_model = lgb_model
        logger.info("LightGBM done.")

        # ---- XGBoost (early stopping via callback, NOT fit param) ----
        logger.info("Training XGBoost ...")
        xgb_model = xgb.XGBClassifier(
            **XGB_PARAMS,
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
        )
        xgb_model.fit(
            X_train[FEATURES], y_train,
            eval_set=[(X_val[FEATURES], y_val)],
            verbose=False,
        )
        self.xgb_model = xgb_model
        logger.info("XGBoost done.")

        # ---- OOF predictions for meta-learner ----
        logger.info("Building GroupKFold OOF predictions (k=%d) ...", N_SPLITS)
        gkf = GroupKFold(n_splits=N_SPLITS)
        oof_cb = np.zeros(len(X_train))
        oof_lgb = np.zeros(len(X_train))
        oof_xgb = np.zeros(len(X_train))

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            Xf_tr = X_train.iloc[tr_idx][FEATURES]
            yf_tr = y_train.iloc[tr_idx]
            Xf_va = X_train.iloc[va_idx][FEATURES]

            # CatBoost OOF
            cb_f = CatBoostClassifier(**CB_PARAMS)
            cb_f.fit(Pool(Xf_tr, label=yf_tr), verbose=0)
            oof_cb[va_idx] = cb_f.predict_proba(Xf_va)[:, 1]

            # LightGBM OOF
            lgb_f = lgb.LGBMClassifier(**LGB_PARAMS)
            lgb_f.fit(Xf_tr, yf_tr, callbacks=[lgb.log_evaluation(-1)])
            oof_lgb[va_idx] = lgb_f.predict_proba(Xf_va)[:, 1]

            # XGBoost OOF (no early stopping in fold — use fixed n_estimators)
            xgb_fold_params = {k: v for k, v in XGB_PARAMS.items()}
            xgb_f = xgb.XGBClassifier(**xgb_fold_params)
            xgb_f.fit(Xf_tr, yf_tr, verbose=False)
            oof_xgb[va_idx] = xgb_f.predict_proba(Xf_va)[:, 1]

            logger.info("  Fold %d/%d done", fold + 1, N_SPLITS)

        meta_X = np.column_stack([oof_cb, oof_lgb, oof_xgb])
        self.meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta.fit(meta_X, y_train.values)
        logger.info("Meta-learner fitted. Coefs: %s", self.meta.coef_)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(win) for each row as 1D array."""
        if self.meta is None:
            raise RuntimeError("NascarEnsemble not fitted — call fit() first")
        Xf = X[FEATURES] if isinstance(X, pd.DataFrame) else X

        cb_p = self.cb_model.predict_proba(Xf)[:, 1]
        lgb_p = self.lgb_model.predict_proba(Xf)[:, 1]
        xgb_p = self.xgb_model.predict_proba(Xf)[:, 1]

        meta_X = np.column_stack([cb_p, lgb_p, xgb_p])
        return self.meta.predict_proba(meta_X)[:, 1]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("NascarEnsemble saved to %s", path)

    @staticmethod
    def load(path: str) -> "NascarEnsemble":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("NascarEnsemble loaded from %s", path)
        return obj
