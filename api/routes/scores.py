from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from api.schemas import ScoreResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/score", tags=["Scores"])


def _get_inference(request: Request):
    inference = request.app.state.inference
    if inference is None:
        raise HTTPException(status_code=503, detail="Scoring engine not initialised")
    return inference


def _get_loader(request: Request):
    return request.app.state.loader


def _get_explainer(request: Request):
    return request.app.state.explainer


@router.get(
    "/{wallet}/{asset_pair}",
    response_model=ScoreResponse,
    summary="Get risk score for a wallet / asset pair",
    description=(
        "Returns the current LedgerLens Risk Score (0–100) for the given wallet "
        "and asset pair, along with Benford and ML flags, confidence, and SHAP explanation."
    ),
)
async def get_score(
    request: Request,
    wallet: Annotated[str, Path(description="Stellar wallet public key (G...)")],
    asset_pair: Annotated[str, Path(description="Asset pair e.g. XLM:native/USDC:GA5Z...")],
    refresh: Annotated[bool, Query(description="Force re-score from latest on-chain data")] = False,
):
    inference = _get_inference(request)
    loader = _get_loader(request)
    explainer = _get_explainer(request)

    cache = request.app.state.score_cache
    cache_key = f"{wallet}:{asset_pair}"

    if not refresh and cache_key in cache:
        return cache[cache_key]

    try:
        trades = await loader.fetch_trades_for_account(wallet, max_pages=10)
        windows = loader.build_windows(wallet, asset_pair, trades)
    except Exception as exc:
        logger.error("Failed to load trades for %s: %s", wallet, exc)
        raise HTTPException(status_code=502, detail="Failed to fetch trade data from Horizon") from exc

    from detection.feature_engineering import FeatureEngineer
    from detection.benford_engine import BenfordEngine

    benford_engine = BenfordEngine()
    window_map = {w.window_label: w for w in windows}
    primary_amounts = window_map.get("24h", windows[0] if windows else None)
    benford_result = None
    if primary_amounts:
        benford_result = benford_engine.analyse(primary_amounts.amounts, "24h")

    benford_flag = benford_result.any_flag if benford_result else False
    benford_score = benford_result.anomaly_score if benford_result else 0.0

    engineer = FeatureEngineer()
    features = engineer.extract(wallet, asset_pair, windows)

    risk = inference.score(
        wallet=wallet,
        asset_pair=asset_pair,
        features=features,
        benford_anomaly_score=benford_score,
        benford_flag=benford_flag,
    )

    top_features = explainer.explain(features) if explainer else []

    response = ScoreResponse(
        wallet=wallet,
        asset_pair=asset_pair,
        score=risk.score,
        benford_flag=risk.benford_flag,
        ml_flag=risk.ml_flag,
        confidence=risk.confidence,
        benford_anomaly_score=risk.benford_anomaly_score,
        explanation={"top_features": top_features},
    )
    cache[cache_key] = response
    return response
