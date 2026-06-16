"""
Heuristic labeller for building initial wash-trade training data.

When no human-labelled dataset is available, this module applies a set
of conservative heuristics to assign weak labels (is_wash_trade=1/0)
to trade windows. These labels should be reviewed before model training.
"""
from __future__ import annotations

from dataclasses import dataclass

from ingestion.data_models import TradeWindow


@dataclass
class LabelResult:
    wallet: str
    asset_pair: str
    is_wash_trade: int       # 1 = likely wash, 0 = likely clean
    confidence: float        # 0–1 heuristic confidence
    reasons: list[str]


def label_window(window: TradeWindow) -> LabelResult:
    """Apply heuristics to a trade window and assign a weak wash-trade label."""
    reasons: list[str] = []
    score = 0.0

    trades = window.trades
    n = len(trades)
    if n < 5:
        return LabelResult(window.wallet, window.asset_pair, 0, 0.0, ["insufficient_trades"])

    amounts = [t.base_amount for t in trades]
    counterparties = [t.counter_account for t in trades if t.counter_account]

    # Heuristic 1: High counterparty concentration
    from collections import Counter
    cp_counts = Counter(counterparties)
    if cp_counts and max(cp_counts.values()) / n > 0.8:
        score += 0.4
        reasons.append("high_counterparty_concentration")

    # Heuristic 2: Very round amounts (suspiciously uniform sizes)
    round_count = sum(1 for a in amounts if a == round(a) and a % 100 == 0)
    if round_count / n > 0.7:
        score += 0.3
        reasons.append("round_amounts")

    # Heuristic 3: Tiny spread in amount sizes (bot with fixed lot size)
    if n >= 10:
        import statistics
        cv = statistics.stdev(amounts) / max(statistics.mean(amounts), 1e-9)
        if cv < 0.05:
            score += 0.3
            reasons.append("fixed_lot_size")

    # Heuristic 4: Rapid repeated trades (< 2s apart on average)
    if n >= 10:
        times = sorted(t.ledger_close_time.timestamp() for t in trades)
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        avg_gap = sum(diffs) / len(diffs)
        if avg_gap < 2.0:
            score += 0.2
            reasons.append("high_frequency")

    label = 1 if score >= 0.5 else 0
    confidence = min(1.0, score)
    return LabelResult(window.wallet, window.asset_pair, label, confidence, reasons)
