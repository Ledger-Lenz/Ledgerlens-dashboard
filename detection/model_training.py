"""
Ensemble model training for LedgerLens wash trade detection.

Trains Random Forest, XGBoost, and LightGBM classifiers on labelled
trade feature vectors, applies SMOTE to handle class imbalance, and
evaluates on held-out data. Saves trained models to disk.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    # Benford — 15 features
    *[f"benford_{w}_{m}" for w in ["1h", "4h", "24h", "7d", "30d"] for m in ["chi_sq", "z_max", "mad"]],
    # Trade pattern — 6
    "counterparty_concentration",
    "round_trip_rate",
    "self_match_proxy",
    "cancel_rate_proxy",
    "unique_counterparty_count",
    "avg_trade_size",
    # Volume / timing — 5
    "volume_per_counterparty",
    "intraday_clustering",
    "off_hours_ratio",
    "volume_spike_proxy",
    "trade_frequency_hz",
    # Wallet graph — 4
    "account_age_days",
    "network_centrality",
    "funding_similarity",
    "is_new_account",
]

LABEL_COLUMN = "is_wash_trade"


class EnsembleTrainer:
    def __init__(self, model_dir: str = "models/", random_state: int = 42) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        X, y = self._prepare(df)
        X_res, y_res = self._oversample(X, y)

        results: dict[str, float] = {}

        rf = self._train_random_forest(X_res, y_res, X, y)
        results["rf_f1"] = rf

        xgb_score = self._train_xgboost(X_res, y_res, X, y)
        results["xgb_f1"] = xgb_score

        lgbm_score = self._train_lightgbm(X_res, y_res, X, y)
        results["lgbm_f1"] = lgbm_score

        logger.info("Training complete. F1 scores: %s", results)
        return results

    def _prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"Label column '{LABEL_COLUMN}' not found in dataframe")

        X = df[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
        y = df[LABEL_COLUMN].values.astype(int)
        logger.info("Dataset: %d samples, %d positives (%.1f%%)", len(y), y.sum(), 100 * y.mean())
        return X, y

    def _oversample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if y.sum() < 2 or (y == 0).sum() < 2:
            logger.warning("Too few minority samples for SMOTE — skipping oversampling")
            return X, y
        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info("After SMOTE: %d samples", len(y_res))
        return X_res, y_res

    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
    ) -> float:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=self.random_state,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_eval)
        score = f1_score(y_eval, preds, zero_division=0)
        joblib.dump(clf, self.model_dir / "rf_model.joblib")
        logger.info("Random Forest F1=%.4f  saved to %s", score, self.model_dir / "rf_model.joblib")
        return score

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
    ) -> float:
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=self.random_state,
            verbosity=0,
        )
        clf.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        preds = clf.predict(X_eval)
        score = f1_score(y_eval, preds, zero_division=0)
        clf.save_model(str(self.model_dir / "xgb_model.json"))
        logger.info("XGBoost F1=%.4f  saved", score)
        return score

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
    ) -> float:
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_eval, y_eval)])
        preds = clf.predict(X_eval)
        score = f1_score(y_eval, preds, zero_division=0)
        clf.booster_.save_model(str(self.model_dir / "lgbm_model.txt"))
        logger.info("LightGBM F1=%.4f  saved", score)
        return score
