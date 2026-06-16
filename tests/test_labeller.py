"""Tests for the heuristic wash-trade labeller."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from detection.wash_trade_labels import label_window
from ingestion.data_models import Asset, AssetType, TradeRecord, TradeWindow


def _asset() -> Asset:
    return Asset(asset_type=AssetType.NATIVE)


def _trade(amount: float, cp: str = "GB_CP", ts_offset_s: float = 0) -> TradeRecord:
    ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=ts_offset_s)
    return TradeRecord(
        id=f"t-{amount}-{ts_offset_s}",
        paging_token="tok",
        ledger_close_time=ts,
        base_account="GA_WALLET",
        base_amount=amount,
        base_asset=_asset(),
        counter_account=cp,
        counter_amount=amount * 0.5,
        counter_asset=_asset(),
    )


def _window(trades, label="24h") -> TradeWindow:
    now = datetime.now(timezone.utc)
    return TradeWindow(
        wallet="GA_WALLET",
        asset_pair="XLM/USDC",
        window_label=label,
        trades=trades,
        start_time=now - timedelta(hours=24),
        end_time=now,
    )


class TestLabeller:
    def test_insufficient_trades_returns_clean(self):
        result = label_window(_window([_trade(100)]))
        assert result.is_wash_trade == 0
        assert "insufficient_trades" in result.reasons

    def test_high_concentration_labelled_wash(self):
        # 90% of trades with same CP
        trades = [_trade(100 + i, cp="GB_FIXED", ts_offset_s=i * 60) for i in range(10)]
        result = label_window(_window(trades))
        assert result.is_wash_trade == 1
        assert "high_counterparty_concentration" in result.reasons

    def test_diverse_counterparties_clean(self):
        trades = [_trade(100 + i * 37, cp=f"GB_{i}", ts_offset_s=i * 120) for i in range(15)]
        result = label_window(_window(trades))
        assert "high_counterparty_concentration" not in result.reasons

    def test_fixed_lot_size_flagged(self):
        # Very uniform amounts → low CV
        trades = [_trade(1000.0, cp=f"GB_{i}", ts_offset_s=i * 300) for i in range(20)]
        result = label_window(_window(trades))
        assert "fixed_lot_size" in result.reasons

    def test_high_frequency_flagged(self):
        # Trades 1 second apart
        trades = [_trade(100 + i * 7, cp=f"GB_{i}", ts_offset_s=i * 1.0) for i in range(20)]
        result = label_window(_window(trades))
        assert "high_frequency" in result.reasons

    def test_clean_organic_trades(self):
        # Spread amounts, diverse counterparties, moderate frequency
        trades = [_trade(37 * (i + 1), cp=f"GB_{i}", ts_offset_s=i * 600) for i in range(15)]
        result = label_window(_window(trades))
        assert result.is_wash_trade == 0

    def test_confidence_bounded(self):
        for n in [4, 10, 50]:
            trades = [_trade(100, cp="GB_FIXED", ts_offset_s=i) for i in range(n)]
            result = label_window(_window(trades))
            assert 0.0 <= result.confidence <= 1.0
