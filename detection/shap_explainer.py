"""
SHAP interpretability layer for LedgerLens risk scores.

Wraps the XGBoost model with a SHAP TreeExplainer and returns
human-readable top-feature contributions for each scored wallet.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from detection.model_training import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class ShapExplainer:
    def __init__(self, model_dir: str = "models/") -> None:
        self.model_dir = Path(model_dir)
        self._explainer = None

    def load(self, xgb_model) -> None:
        try:
            import shap
            self._explainer = shap.TreeExplainer(xgb_model)
            logger.info("SHAP TreeExplainer initialised")
        except ImportError:
            logger.warning("shap package not installed — explanations disabled")
        except Exception as exc:
            logger.error("Failed to initialise SHAP explainer: %s", exc)

    def explain(
        self,
        features: dict[str, float],
        top_n: int = 5,
    ) -> list[dict]:
        if self._explainer is None:
            return []

        try:
            import shap
            x = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]])
            shap_values = self._explainer.shap_values(x)

            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # class-1 SHAP values
            else:
                sv = shap_values[0]

            indexed = sorted(
                enumerate(sv), key=lambda kv: abs(kv[1]), reverse=True
            )[:top_n]

            return [
                {
                    "feature": FEATURE_COLUMNS[i],
                    "contribution": round(float(v), 4),
                    "direction": "increases_risk" if v > 0 else "decreases_risk",
                }
                for i, v in indexed
            ]
        except Exception as exc:
            logger.error("SHAP explanation failed: %s", exc)
            return []
