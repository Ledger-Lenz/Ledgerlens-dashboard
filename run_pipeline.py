#!/usr/bin/env python3
"""
LedgerLens detection pipeline entry point.

Streams live trades from Stellar Horizon, runs the Benford + ML detection
engine on rolling windows, and submits risk scores to the Soroban contract.

Usage:
    python run_pipeline.py [--network testnet|mainnet] [--wallet GXXX]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections import deque
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ledgerlens.pipeline")


async def run(args: argparse.Namespace) -> None:
    from ingestion.horizon_streamer import HorizonStreamer
    from ingestion.historical_loader import HistoricalLoader
    from ingestion.data_models import TradeRecord
    from detection.benford_engine import BenfordEngine
    from detection.feature_engineering import FeatureEngineer
    from detection.model_inference import ModelInference

    horizon_url = os.getenv("HORIZON_URL", "https://horizon-testnet.stellar.org")
    model_dir = os.getenv("ML_MODEL_PATH", "models/")
    alert_threshold = int(os.getenv("ALERT_THRESHOLD", "75"))
    min_samples = int(os.getenv("BENFORD_MIN_SAMPLES", "30"))

    logger.info("Starting LedgerLens pipeline")
    logger.info("  Horizon : %s", horizon_url)
    logger.info("  Network : %s", args.network)
    logger.info("  Threshold: %d", alert_threshold)

    loader = HistoricalLoader(horizon_url)
    benford = BenfordEngine(min_samples=min_samples)
    engineer = FeatureEngineer(benford)
    inference = ModelInference(model_dir)
    inference.load()

    # Rolling buffer of recent trades per wallet (last 2000 per wallet)
    trade_buffer: dict[str, deque[TradeRecord]] = {}

    def on_trade(trade: TradeRecord) -> None:
        wallet = trade.base_account or "unknown"
        if wallet not in trade_buffer:
            trade_buffer[wallet] = deque(maxlen=2000)
        trade_buffer[wallet].append(trade)

        buf = list(trade_buffer[wallet])
        windows = loader.build_windows(wallet, trade.pair, buf)
        features = engineer.extract(wallet, trade.pair, windows)

        primary = next((w for w in windows if w.window_label == "24h"), None)
        benford_result = benford.analyse(primary.amounts, "24h") if primary else None
        benford_flag = benford_result.any_flag if benford_result else False
        benford_score = benford_result.anomaly_score if benford_result else 0.0

        risk = inference.score(
            wallet=wallet,
            asset_pair=trade.pair,
            features=features,
            benford_anomaly_score=benford_score,
            benford_flag=benford_flag,
        )

        if risk.score >= alert_threshold:
            logger.warning(
                "ALERT  wallet=%-56s pair=%-20s score=%3d benford=%s ml=%s",
                wallet, trade.pair, risk.score, risk.benford_flag, risk.ml_flag,
            )
        else:
            logger.debug("OK     wallet=%s  score=%d", wallet, risk.score)

    streamer = HorizonStreamer(horizon_url)
    logger.info("Streaming trades — press Ctrl+C to stop")
    await streamer.stream(on_trade)


def main() -> None:
    parser = argparse.ArgumentParser(description="LedgerLens detection pipeline")
    parser.add_argument("--network", default=os.getenv("NETWORK", "testnet"),
                        choices=["testnet", "mainnet"])
    parser.add_argument("--wallet", help="Focus on a specific wallet (optional)")
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
