from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TopFeature(BaseModel):
    feature: str
    contribution: float
    direction: str


class Explanation(BaseModel):
    top_features: list[TopFeature] = Field(default_factory=list)


class ScoreResponse(BaseModel):
    wallet: str
    asset_pair: str
    score: int = Field(..., ge=0, le=100)
    benford_flag: bool
    ml_flag: bool
    confidence: int = Field(..., ge=0, le=100)
    benford_anomaly_score: float
    timestamp: Optional[datetime] = None
    explanation: Explanation = Field(default_factory=Explanation)


class AlertRecord(BaseModel):
    wallet: str
    asset_pair: str
    score: int
    flagged_at: datetime
    benford_flag: bool
    ml_flag: bool


class AlertsResponse(BaseModel):
    alerts: list[AlertRecord]
    total: int
    page: int
    limit: int


class AssetRiskEntry(BaseModel):
    asset_code: str
    asset_issuer: Optional[str]
    avg_score: float
    max_score: int
    flagged_wallet_count: int
    last_updated: Optional[datetime]


class AssetRankingResponse(BaseModel):
    assets: list[AssetRiskEntry]
    total: int
    window: str


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    horizon_url: str
    contract_id: Optional[str]
    models_loaded: bool
