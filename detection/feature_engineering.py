"""
On-chain ML feature extraction from SDEX trade windows.

Produces a flat feature vector (30+ features) for each wallet / asset pair.
Groups:
  - Benford features  (15)  — chi-square, Z-score max, MAD × 5 windows
  - Trade pattern     (6)   — counterparty concentration, round-trip rate, etc.
  - Volume / timing   (5)   — volume-to-counterparty ratio, spike freq, etc.
  - Wallet graph      (4)   — funding similarity, account age, centrality proxy
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from detection.benford_engine import BenfordEngine, BenfordResult
from ingestion.data_models import TradeRecord, TradeWindow

logger = logging.getLogger(__name__)

_WINDOWS = ["1h", "4h", "24h", "7d", "30d"]
_BENFORD_ENGINE = BenfordEngine()


def _safe(value: float) -> float:
    return 0.0 if (np.isnan(value) or np.isinf(value)) else float(value)


class FeatureEngineer:
    def __init__(self, benford_engine: Optional[BenfordEngine] = None) -> None:
        self.benford = benford_engine or _BENFORD_ENGINE

    def extract(
        self,
        wallet: str,
        asset_pair: str,
        windows: list[TradeWindow],
        account_created_at: Optional[datetime] = None,
    ) -> dict[str, float]:
        features: dict[str, float] = {}

        # Benford features across windows
        window_map = {w.window_label: w for w in windows}
        for label in _WINDOWS:
            w = window_map.get(label)
            result = self.benford.analyse(w.amounts, label) if w else None
            features.update(self._benford_features(label, result))

        # Use the 24h window as the primary trade pattern source
        primary = window_map.get("24h") or (windows[0] if windows else None)
        all_trades = primary.trades if primary else []

        features.update(self._trade_pattern_features(all_trades))
        features.update(self._volume_timing_features(all_trades))
        features.update(self._wallet_graph_features(wallet, all_trades, account_created_at))

        return features

    # ── Benford (15 features) ──────────────────────────────────────────────────

    @staticmethod
    def _benford_features(label: str, result: Optional[BenfordResult]) -> dict[str, float]:
        prefix = f"benford_{label}"
        if result is None:
            return {
                f"{prefix}_chi_sq": 0.0,
                f"{prefix}_z_max": 0.0,
                f"{prefix}_mad": 0.0,
            }
        return {
            f"{prefix}_chi_sq": _safe(result.chi_square),
            f"{prefix}_z_max": _safe(max(abs(z) for z in result.z_scores.values()) if result.z_scores else 0.0),
            f"{prefix}_mad": _safe(result.mad),
        }

    # ── Trade pattern (6 features) ────────────────────────────────────────────

    def _trade_pattern_features(self, trades: list[TradeRecord]) -> dict[str, float]:
        n = len(trades)
        if n == 0:
            return {
                "counterparty_concentration": 0.0,
                "round_trip_rate": 0.0,
                "self_match_proxy": 0.0,
                "cancel_rate_proxy": 0.0,
                "unique_counterparty_count": 0.0,
                "avg_trade_size": 0.0,
            }

        counterparties = [t.counter_account for t in trades if t.counter_account]
        cp_counts = Counter(counterparties)
        top_cp_fraction = (max(cp_counts.values()) / n) if cp_counts else 0.0

        unique_cps = len(cp_counts)
        round_trips = self._count_round_trips(trades)
        amounts = [t.base_amount for t in trades]

        return {
            "counterparty_concentration": _safe(top_cp_fraction),
            "round_trip_rate": _safe(round_trips / n),
            "self_match_proxy": _safe(1.0 - unique_cps / max(n, 1)),
            "cancel_rate_proxy": 0.0,  # requires order book data
            "unique_counterparty_count": float(unique_cps),
            "avg_trade_size": _safe(float(np.mean(amounts))),
        }

    @staticmethod
    def _count_round_trips(trades: list[TradeRecord], window_s: int = 300) -> int:
        """Count trades where the asset returns to the originating wallet within window_s seconds."""
        count = 0
        for i, t in enumerate(trades):
            for j in range(i + 1, len(trades)):
                other = trades[j]
                dt = abs((other.ledger_close_time - t.ledger_close_time).total_seconds())
                if dt > window_s:
                    break
                if (
                    t.base_account == other.counter_account
                    and t.counter_account == other.base_account
                ):
                    count += 1
                    break
        return count

    # ── Volume / timing (5 features) ──────────────────────────────────────────

    @staticmethod
    def _volume_timing_features(trades: list[TradeRecord]) -> dict[str, float]:
        n = len(trades)
        if n == 0:
            return {
                "volume_per_counterparty": 0.0,
                "intraday_clustering": 0.0,
                "off_hours_ratio": 0.0,
                "volume_spike_proxy": 0.0,
                "trade_frequency_hz": 0.0,
            }

        amounts = np.array([t.base_amount for t in trades])
        total_vol = float(np.sum(amounts))
        unique_cps = len({t.counter_account for t in trades if t.counter_account})
        vol_per_cp = total_vol / max(unique_cps, 1)

        # Intra-minute clustering: fraction of consecutive trades within 60s of each other
        times = sorted(t.ledger_close_time.timestamp() for t in trades)
        if len(times) > 1:
            gaps = np.diff(times)
            clustering = float(np.mean(gaps < 60))
        else:
            clustering = 0.0

        # Off-hours: trades between 00:00–06:00 UTC
        off_hours = sum(1 for t in trades if t.ledger_close_time.hour < 6) / n

        # Volume spike proxy: coefficient of variation of trade sizes
        cv = float(np.std(amounts) / np.mean(amounts)) if np.mean(amounts) > 0 else 0.0

        # Trade frequency in Hz over the observed window
        if len(times) > 1:
            span = times[-1] - times[0]
            freq = n / max(span, 1)
        else:
            freq = 0.0

        return {
            "volume_per_counterparty": _safe(vol_per_cp),
            "intraday_clustering": _safe(clustering),
            "off_hours_ratio": _safe(off_hours),
            "volume_spike_proxy": _safe(cv),
            "trade_frequency_hz": _safe(freq),
        }

    # ── Wallet graph (4 features) ─────────────────────────────────────────────

    @staticmethod
    def _wallet_graph_features(
        wallet: str,
        trades: list[TradeRecord],
        account_created_at: Optional[datetime],
    ) -> dict[str, float]:
        now = datetime.now(timezone.utc)
        age_days = (now - account_created_at).days if account_created_at else -1

        # Network centrality proxy: fraction of all seen accounts this wallet traded with
        all_accounts = {t.base_account for t in trades} | {t.counter_account for t in trades}
        all_accounts.discard(None)
        all_accounts.discard(wallet)
        centrality = len(all_accounts) / max(len(trades), 1)

        # Funding similarity placeholder (requires graph traversal across accounts)
        funding_similarity = 0.0

        return {
            "account_age_days": float(max(age_days, 0)),
            "network_centrality": _safe(centrality),
            "funding_similarity": _safe(funding_similarity),
            "is_new_account": float(0 <= age_days <= 7),
        }
