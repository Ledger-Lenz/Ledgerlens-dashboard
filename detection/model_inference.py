"""
Real-time risk scoring from loaded ensemble models.

Loads RF, XGBoost, and LightGBM models from disk and produces a combined
LedgerLens Risk Score (0–100) via soft voting on predicted probabilities.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from detection.model_training import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class RiskScore:
    wallet: str
    asset_pair: str
    score: int                          # 0–100
    benford_flag: bool
    ml_flag: bool
    confidence: int                     # 0–100
    benford_anomaly_score: float = 0.0
    top_features: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "wallet": self.wallet,
            "asset_pair": self.asset_pair,
            "score": self.score,
            "benford_flag": self.benford_flag,
            "ml_flag": self.ml_flag,
            "confidence": self.confidence,
            "benford_anomaly_score": self.benford_anomaly_score,
            "top_features": self.top_features,
        }


class ModelInference:
    """Load trained models and score wallet/pair feature vectors."""

    def __init__(self, model_dir: str = "models/") -> None:
        self.model_dir = Path(model_dir)
        self._rf = None
        self._xgb = None
        self._lgb = None
        self._loaded = False

    def load(self) -> None:
        rf_path = self.model_dir / "rf_model.joblib"
        xgb_path = self.model_dir / "xgb_model.json"
        lgb_path = self.model_dir / "lgbm_model.txt"

        if rf_path.exists():
            self._rf = joblib.load(rf_path)
            logger.info("Loaded RF model from %s", rf_path)
        if xgb_path.exists():
            self._xgb = xgb.XGBClassifier()
            self._xgb.load_model(str(xgb_path))
            logger.info("Loaded XGBoost model from %s", xgb_path)
        if lgb_path.exists():
            self._lgb = lgb.Booster(model_file=str(lgb_path))
            logger.info("Loaded LightGBM model from %s", lgb_path)

        if not any([self._rf, self._xgb, self._lgb]):
            logger.warning("No models found in %s — scoring will return 0", self.model_dir)
        self._loaded = True

    def score(
        self,
        wallet: str,
        asset_pair: str,
        features: dict[str, float],
        benford_anomaly_score: float = 0.0,
        benford_flag: bool = False,
    ) -> RiskScore:
        if not self._loaded:
            self.load()

        x = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]], dtype=np.float32)
        probs = self._ensemble_proba(x)

        ml_prob = float(np.mean(probs)) if probs else 0.0
        ml_flag = ml_prob >= 0.5

        # Combine Benford anomaly (0–100) with ML probability (0–1) into final score
        benford_weight = 0.35
        ml_weight = 0.65
        combined = benford_weight * (benford_anomaly_score / 100.0) + ml_weight * ml_prob
        score = min(100, int(combined * 100))

        confidence = int(100 * (1.0 - 2 * abs(ml_prob - 0.5))) if probs else 0

        return RiskScore(
            wallet=wallet,
            asset_pair=asset_pair,
            score=score,
            benford_flag=benford_flag,
            ml_flag=ml_flag,
            confidence=confidence,
            benford_anomaly_score=benford_anomaly_score,
        )

    def _ensemble_proba(self, x: np.ndarray) -> list[float]:
        probs: list[float] = []
        if self._rf is not None:
            try:
                probs.append(float(self._rf.predict_proba(x)[0, 1]))
            except Exception as exc:
                logger.error("RF inference error: %s", exc)
        if self._xgb is not None:
            try:
                probs.append(float(self._xgb.predict_proba(x)[0, 1]))
            except Exception as exc:
                logger.error("XGBoost inference error: %s", exc)
        if self._lgb is not None:
            try:
                probs.append(float(self._lgb.predict(x)[0]))
            except Exception as exc:
                logger.error("LightGBM inference error: %s", exc)
        return probs
