from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Query, Request

from api.schemas import AlertRecord, AlertsResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get(
    "/recent",
    response_model=AlertsResponse,
    summary="List wallets flagged in the last 24 hours",
)
async def get_recent_alerts(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    min_score: Annotated[int, Query(ge=0, le=100)] = 75,
    page: Annotated[int, Query(ge=1)] = 1,
):
    store = request.app.state.alert_store
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    filtered = [
        a for a in store
        if a["score"] >= min_score and a["flagged_at"] >= cutoff
    ]
    filtered.sort(key=lambda a: a["score"], reverse=True)

    offset = (page - 1) * limit
    page_items = filtered[offset : offset + limit]

    records = [
        AlertRecord(
            wallet=a["wallet"],
            asset_pair=a["asset_pair"],
            score=a["score"],
            flagged_at=a["flagged_at"],
            benford_flag=a.get("benford_flag", False),
            ml_flag=a.get("ml_flag", False),
        )
        for a in page_items
    ]
    return AlertsResponse(alerts=records, total=len(filtered), page=page, limit=limit)
