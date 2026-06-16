from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AssetType(str, Enum):
    NATIVE = "native"
    CREDIT_ALPHANUM4 = "credit_alphanum4"
    CREDIT_ALPHANUM12 = "credit_alphanum12"


class Asset(BaseModel):
    asset_type: AssetType
    asset_code: Optional[str] = None
    asset_issuer: Optional[str] = None

    @property
    def symbol(self) -> str:
        if self.asset_type == AssetType.NATIVE:
            return "XLM:native"
        return f"{self.asset_code}:{self.asset_issuer}"

    def __str__(self) -> str:
        return self.symbol


class TradeRecord(BaseModel):
    id: str
    paging_token: str
    ledger_close_time: datetime
    trade_type: str = "orderbook"

    # Parties
    base_account: Optional[str] = None
    base_amount: float = Field(..., gt=0)
    base_asset: Asset

    counter_account: Optional[str] = None
    counter_amount: float = Field(..., gt=0)
    counter_asset: Asset

    # Order book context
    base_is_seller: bool = False
    price: Optional[float] = None

    @field_validator("base_amount", "counter_amount", mode="before")
    @classmethod
    def parse_amount(cls, v: object) -> float:
        return float(v)

    @property
    def pair(self) -> str:
        return f"{self.base_asset.symbol}/{self.counter_asset.symbol}"

    @property
    def leading_digit(self) -> int:
        s = f"{self.base_amount:.10f}".lstrip("0").replace(".", "")
        return int(s[0]) if s else 0


class AccountRecord(BaseModel):
    account_id: str
    sequence: int
    created_at: datetime
    num_sponsoring: int = 0
    num_sponsored: int = 0
    balances: list[dict] = Field(default_factory=list)


class TradeWindow(BaseModel):
    wallet: str
    asset_pair: str
    window_label: str
    trades: list[TradeRecord]
    start_time: datetime
    end_time: datetime

    @property
    def amounts(self) -> list[float]:
        return [t.base_amount for t in self.trades]

    @property
    def counterparties(self) -> list[str]:
        return [t.counter_account for t in self.trades if t.counter_account]
