from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Callable, Optional

import httpx
from stellar_sdk import Server, Asset as StellarAsset

from ingestion.data_models import Asset, AssetType, TradeRecord

logger = logging.getLogger(__name__)


def _parse_asset(asset_type: str, code: Optional[str], issuer: Optional[str]) -> Asset:
    if asset_type == "native":
        return Asset(asset_type=AssetType.NATIVE)
    atype = AssetType.CREDIT_ALPHANUM4 if len(code or "") <= 4 else AssetType.CREDIT_ALPHANUM12
    return Asset(asset_type=atype, asset_code=code, asset_issuer=issuer)


def _raw_to_trade(raw: dict) -> Optional[TradeRecord]:
    try:
        return TradeRecord(
            id=raw["id"],
            paging_token=raw["paging_token"],
            ledger_close_time=raw["ledger_close_time"],
            trade_type=raw.get("trade_type", "orderbook"),
            base_account=raw.get("base_account"),
            base_amount=raw["base_amount"],
            base_asset=_parse_asset(
                raw["base_asset_type"],
                raw.get("base_asset_code"),
                raw.get("base_asset_issuer"),
            ),
            counter_account=raw.get("counter_account"),
            counter_amount=raw["counter_amount"],
            counter_asset=_parse_asset(
                raw["counter_asset_type"],
                raw.get("counter_asset_code"),
                raw.get("counter_asset_issuer"),
            ),
            base_is_seller=raw.get("base_is_seller", False),
            price=float(raw["price"]["n"]) / float(raw["price"]["d"])
            if isinstance(raw.get("price"), dict)
            else None,
        )
    except Exception as exc:
        logger.warning("Failed to parse trade record %s: %s", raw.get("id"), exc)
        return None


class HorizonStreamer:
    """Stream live trade data from the Stellar Horizon SSE endpoint."""

    def __init__(
        self,
        horizon_url: str,
        base_asset: Optional[StellarAsset] = None,
        counter_asset: Optional[StellarAsset] = None,
    ) -> None:
        self.horizon_url = horizon_url.rstrip("/")
        self.base_asset = base_asset
        self.counter_asset = counter_asset
        self._server = Server(horizon_url)

    async def stream(
        self,
        on_trade: Callable[[TradeRecord], None],
        cursor: str = "now",
    ) -> None:
        """Stream trades indefinitely, calling on_trade for each valid record."""
        url = self._build_url(cursor)
        logger.info("Starting SSE stream: %s", url)

        while True:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("GET", url, headers={"Accept": "text/event-stream"}) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            import json
                            raw = json.loads(line[5:].strip())
                            if "id" not in raw:
                                continue
                            trade = _raw_to_trade(raw)
                            if trade:
                                cursor = trade.paging_token
                                on_trade(trade)
            except httpx.RemoteProtocolError:
                logger.info("SSE connection closed — reconnecting in 5s")
                await asyncio.sleep(5)
            except Exception as exc:
                logger.error("Stream error: %s — reconnecting in 10s", exc)
                await asyncio.sleep(10)

    async def stream_iter(self, cursor: str = "now") -> AsyncIterator[TradeRecord]:
        """Async generator variant for use in async-for loops."""
        queue: asyncio.Queue[TradeRecord] = asyncio.Queue()

        async def _enqueue(trade: TradeRecord) -> None:
            await queue.put(trade)

        task = asyncio.create_task(self.stream(lambda t: asyncio.ensure_future(_enqueue(t)), cursor))
        try:
            while True:
                yield await queue.get()
        finally:
            task.cancel()

    def _build_url(self, cursor: str) -> str:
        base = f"{self.horizon_url}/trades?order=asc&limit=200&cursor={cursor}"
        if self.base_asset and self.counter_asset:
            ba = self.base_asset
            ca = self.counter_asset
            if ba.is_native():
                base += "&base_asset_type=native"
            else:
                base += f"&base_asset_type=credit_alphanum{len(ba.code)}&base_asset_code={ba.code}&base_asset_issuer={ba.issuer}"
            if ca.is_native():
                base += "&counter_asset_type=native"
            else:
                base += f"&counter_asset_type=credit_alphanum{len(ca.code)}&counter_asset_code={ca.code}&counter_asset_issuer={ca.issuer}"
        return base
