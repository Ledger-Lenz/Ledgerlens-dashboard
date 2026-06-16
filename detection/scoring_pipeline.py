"""
End-to-end scoring pipeline that wires together all detection layers.

ScoringPipeline.score_wallet() accepts raw trade records for a wallet,
runs the full Benford + feature engineering + ML inference chain, and
returns a final RiskScore ready for on-chain submission.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from detection.benford_engine import BenfordEngine
from detection.feature_engineering import FeatureEngineer
from detection.model_inference import ModelInference, RiskScore
from detection.shap_explainer import ShapExplainer
from ingestion.data_models import TradeRecord
from ingestion.historical_loader import HistoricalLoader

logger = logging.getLogger(__name__)


class ScoringPipeline:
    def __init__(
        self,
        horizon_url: str,
        model_dir: str = "models/",
        benford_min_samples: int = 30,
    ) -> None:
        self.loader = HistoricalLoader(horizon_url)
        self.benford = BenfordEngine(min_samples=benford_min_samples)
        self.engineer = FeatureEngineer(self.benford)
        self.inference = ModelInference(model_dir)
        self.explainer = ShapExplainer(model_dir)
        self._ready = False

    def load(self) -> None:
        self.inference.load()
        if self.inference._xgb is not None:
            self.explainer.load(self.inference._xgb)
        self._ready = True
        logger.info("ScoringPipeline ready (models_loaded=%s)", self.inference._loaded)

    def score_wallet(
        self,
        wallet: str,
        asset_pair: str,
        trades: list[TradeRecord],
        account_created_at: Optional[datetime] = None,
        explain: bool = False,
    ) -> RiskScore:
        if not self._ready:
            self.load()

        windows = self.loader.build_windows(wallet, asset_pair, trades)
        features = self.engineer.extract(wallet, asset_pair, windows, account_created_at)

        primary_window = next((w for w in windows if w.window_label == "24h"), None)
        benford_result = (
            self.benford.analyse(primary_window.amounts, "24h") if primary_window else None
        )
        benford_flag = benford_result.any_flag if benford_result else False
        benford_score = benford_result.anomaly_score if benford_result else 0.0

        risk = self.inference.score(
            wallet=wallet,
            asset_pair=asset_pair,
            features=features,
            benford_anomaly_score=benford_score,
            benford_flag=benford_flag,
        )

        if explain:
            risk.top_features = self.explainer.explain(features)

        logger.info(
            "Scored %s / %s → %d (benford=%s ml=%s)",
            wallet, asset_pair, risk.score, risk.benford_flag, risk.ml_flag,
        )
        return risk
