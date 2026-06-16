from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from ingestion.data_models import TradeRecord, TradeWindow
from ingestion.horizon_streamer import _raw_to_trade

logger = logging.getLogger(__name__)

_WINDOWS = {
    "1h": 3600,
    "4h": 14400,
    "24h": 86400,
    "7d": 604800,
    "30d": 2592000,
}


class HistoricalLoader:
    """Bulk-fetch historical trade pages from Horizon for a given wallet or pair."""

    def __init__(self, horizon_url: str, page_limit: int = 200) -> None:
        self.horizon_url = horizon_url.rstrip("/")
        self.page_limit = page_limit

    async def fetch_trades_for_account(
        self,
        account_id: str,
        since: Optional[datetime] = None,
        max_pages: int = 50,
    ) -> list[TradeRecord]:
        url = (
            f"{self.horizon_url}/accounts/{account_id}/trades"
            f"?order=desc&limit={self.page_limit}"
        )
        return await self._paginate(url, since=since, max_pages=max_pages)

    async def fetch_trades_for_pair(
        self,
        base_code: str,
        base_issuer: Optional[str],
        counter_code: str,
        counter_issuer: Optional[str],
        since: Optional[datetime] = None,
        max_pages: int = 50,
    ) -> list[TradeRecord]:
        if base_code.upper() == "XLM" and base_issuer is None:
            base_params = "base_asset_type=native"
        else:
            atype = "credit_alphanum4" if len(base_code) <= 4 else "credit_alphanum12"
            base_params = f"base_asset_type={atype}&base_asset_code={base_code}&base_asset_issuer={base_issuer}"

        if counter_code.upper() == "XLM" and counter_issuer is None:
            counter_params = "counter_asset_type=native"
        else:
            atype = "credit_alphanum4" if len(counter_code) <= 4 else "credit_alphanum12"
            counter_params = f"counter_asset_type={atype}&counter_asset_code={counter_code}&counter_asset_issuer={counter_issuer}"

        url = (
            f"{self.horizon_url}/trades?{base_params}&{counter_params}"
            f"&order=desc&limit={self.page_limit}"
        )
        return await self._paginate(url, since=since, max_pages=max_pages)

    def build_windows(
        self,
        wallet: str,
        asset_pair: str,
        trades: list[TradeRecord],
        reference_time: Optional[datetime] = None,
    ) -> list[TradeWindow]:
        now = reference_time or datetime.now(timezone.utc)
        windows: list[TradeWindow] = []

        for label, seconds in _WINDOWS.items():
            cutoff = now.timestamp() - seconds
            window_trades = [
                t for t in trades if t.ledger_close_time.timestamp() >= cutoff
            ]
            if not window_trades:
                continue
            start = min(t.ledger_close_time for t in window_trades)
            windows.append(
                TradeWindow(
                    wallet=wallet,
                    asset_pair=asset_pair,
                    window_label=label,
                    trades=window_trades,
                    start_time=start,
                    end_time=now,
                )
            )
        return windows

    async def _paginate(
        self,
        url: str,
        since: Optional[datetime],
        max_pages: int,
    ) -> list[TradeRecord]:
        all_trades: list[TradeRecord] = []
        next_url: Optional[str] = url

        async with httpx.AsyncClient(timeout=30) as client:
            for _ in range(max_pages):
                if next_url is None:
                    break
                try:
                    resp = await client.get(next_url)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.error("Horizon fetch error: %s", exc)
                    break

                records = data.get("_embedded", {}).get("records", [])
                if not records:
                    break

                for raw in records:
                    trade = _raw_to_trade(raw)
                    if trade is None:
                        continue
                    if since and trade.ledger_close_time < since:
                        return all_trades
                    all_trades.append(trade)

                next_link = data.get("_links", {}).get("next", {}).get("href")
                next_url = next_link if next_link else None
                await asyncio.sleep(0.1)

        logger.info("Loaded %d trades from %s", len(all_trades), url[:80])
        return all_trades
