"""Tests for the ML feature engineering module."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from detection.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from ingestion.data_models import Asset, AssetType, TradeRecord, TradeWindow


def _make_asset(code="XLM", native=True) -> Asset:
    if native:
        return Asset(asset_type=AssetType.NATIVE)
    return Asset(asset_type=AssetType.CREDIT_ALPHANUM4, asset_code=code, asset_issuer="GABC")


def _make_trade(
    amount: float = 100.0,
    base_account: str = "GA",
    counter_account: str = "GB",
    minutes_ago: float = 0,
) -> TradeRecord:
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return TradeRecord(
        id=f"id-{amount}-{minutes_ago}",
        paging_token="tok",
        ledger_close_time=ts,
        base_account=base_account,
        base_amount=amount,
        base_asset=_make_asset(),
        counter_account=counter_account,
        counter_amount=amount * 0.5,
        counter_asset=_make_asset("USDC", native=False),
    )


def _make_window(trades: List[TradeRecord], label="24h") -> TradeWindow:
    now = datetime.now(timezone.utc)
    return TradeWindow(
        wallet="GA_TEST",
        asset_pair="XLM/USDC",
        window_label=label,
        trades=trades,
        start_time=now - timedelta(hours=24),
        end_time=now,
    )


class TestFeatureEngineer:
    def setup_method(self):
        self.eng = FeatureEngineer()

    def test_returns_all_feature_columns(self):
        trades = [_make_trade(amount=i * 10.0, minutes_ago=i) for i in range(1, 35)]
        window = _make_window(trades)
        features = self.eng.extract("GA_TEST", "XLM/USDC", [window])
        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"

    def test_empty_trades_returns_zeros(self):
        window = _make_window([])
        features = self.eng.extract("GA_TEST", "XLM/USDC", [window])
        assert features["counterparty_concentration"] == 0.0
        assert features["round_trip_rate"] == 0.0

    def test_high_counterparty_concentration_single_cp(self):
        # All trades with same counterparty
        trades = [_make_trade(counter_account="GB_FIXED", minutes_ago=i) for i in range(50)]
        window = _make_window(trades)
        features = self.eng.extract("GA_TEST", "XLM/USDC", [window])
        assert features["counterparty_concentration"] == pytest.approx(1.0)

    def test_low_concentration_many_counterparties(self):
        trades = [_make_trade(counter_account=f"GB_{i}", minutes_ago=i) for i in range(50)]
        window = _make_window(trades)
        features = self.eng.extract("GA_TEST", "XLM/USDC", [window])
        assert features["counterparty_concentration"] < 0.1

    def test_new_account_flag(self):
        now = datetime.now(timezone.utc)
        features = self.eng.extract(
            "GA_TEST", "XLM/USDC", [],
            account_created_at=now - timedelta(days=3),
        )
        assert features["is_new_account"] == 1.0

    def test_old_account_not_flagged(self):
        now = datetime.now(timezone.utc)
        features = self.eng.extract(
            "GA_TEST", "XLM/USDC", [],
            account_created_at=now - timedelta(days=180),
        )
        assert features["is_new_account"] == 0.0

    def test_off_hours_ratio_all_midnight(self):
        trades = []
        base = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
        for i in range(40):
            t = _make_trade(minutes_ago=0)
            t = t.model_copy(update={"ledger_close_time": base + timedelta(minutes=i)})
            trades.append(t)
        window = _make_window(trades)
        features = self.eng.extract("GA_TEST", "XLM/USDC", [window])
        assert features["off_hours_ratio"] == pytest.approx(1.0)

    def test_round_trip_detection(self):
        ts = datetime.now(timezone.utc)
        # Trade A→B then B→A within 2 minutes
        t1 = _make_trade(base_account="GA", counter_account="GB", minutes_ago=5)
        t2 = _make_trade(base_account="GB", counter_account="GA", minutes_ago=4)
        window = _make_window([t1, t2])
        features = self.eng.extract("GA", "XLM/USDC", [window])
        assert features["round_trip_rate"] > 0.0

    def test_no_nan_or_inf_in_features(self):
        import math
        trades = [_make_trade(amount=i * 7.3, minutes_ago=i * 0.5) for i in range(1, 60)]
        window = _make_window(trades)
        features = self.eng.extract("GA_TEST", "XLM/USDC", [window])
        for k, v in features.items():
            assert not math.isnan(v), f"NaN in feature {k}"
            assert not math.isinf(v), f"Inf in feature {k}"

    def test_benford_features_in_output_with_sufficient_data(self):
        trades = [_make_trade(amount=i * 13.7, minutes_ago=i * 2) for i in range(1, 50)]
        window = _make_window(trades)
        features = self.eng.extract("GA_TEST", "XLM/USDC", [window])
        # At least some benford features should be non-zero when data is sufficient
        benford_vals = [v for k, v in features.items() if k.startswith("benford_24h")]
        assert any(v > 0 for v in benford_vals)
