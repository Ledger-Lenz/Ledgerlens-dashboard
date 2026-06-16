from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request

from api.schemas import AssetRankingResponse, AssetRiskEntry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assets", tags=["Assets"])

_VALID_WINDOWS = {"1h", "24h", "7d"}


@router.get(
    "/risk-ranking",
    response_model=AssetRankingResponse,
    summary="Rank monitored assets by aggregate risk score",
)
async def get_risk_ranking(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    window: Annotated[Literal["1h", "24h", "7d"], Query()] = "24h",
):
    store = request.app.state.asset_store

    # Aggregate by asset_code from cached score records
    agg: dict[str, dict] = {}
    for record in store.get(window, []):
        key = record["asset_code"]
        if key not in agg:
            agg[key] = {
                "asset_code": record["asset_code"],
                "asset_issuer": record.get("asset_issuer"),
                "scores": [],
                "flagged_wallets": set(),
                "last_updated": None,
            }
        agg[key]["scores"].append(record["score"])
        if record["score"] >= 75:
            agg[key]["flagged_wallets"].add(record["wallet"])
        ts = record.get("last_updated")
        if ts and (agg[key]["last_updated"] is None or ts > agg[key]["last_updated"]):
            agg[key]["last_updated"] = ts

    entries = []
    for data in agg.values():
        scores = data["scores"]
        entries.append(
            AssetRiskEntry(
                asset_code=data["asset_code"],
                asset_issuer=data["asset_issuer"],
                avg_score=round(sum(scores) / len(scores), 1),
                max_score=max(scores),
                flagged_wallet_count=len(data["flagged_wallets"]),
                last_updated=data["last_updated"],
            )
        )

    entries.sort(key=lambda e: e.avg_score, reverse=True)
    return AssetRankingResponse(assets=entries[:limit], total=len(entries), window=window)
