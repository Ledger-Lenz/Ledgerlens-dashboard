"""
Resolves account metadata from Stellar Horizon.

Fetches account creation time and balances — used to populate
wallet-graph features (account age, funding source) in the ML pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import httpx

from ingestion.data_models import AccountRecord

logger = logging.getLogger(__name__)


class AccountResolver:
    def __init__(self, horizon_url: str) -> None:
        self.horizon_url = horizon_url.rstrip("/")

    async def get_account(self, account_id: str) -> Optional[AccountRecord]:
        url = f"{self.horizon_url}/accounts/{account_id}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()

            return AccountRecord(
                account_id=data["account_id"],
                sequence=int(data["sequence"]),
                created_at=self._parse_created_at(data),
                num_sponsoring=data.get("num_sponsoring", 0),
                num_sponsored=data.get("num_sponsored", 0),
                balances=data.get("balances", []),
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("Horizon account fetch error %s: %s", account_id, exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error fetching account %s: %s", account_id, exc)
            return None

    @staticmethod
    def _parse_created_at(data: dict) -> datetime:
        # Horizon does not provide account creation time directly.
        # We approximate it from the last_modified_ledger or fall back to epoch.
        ts_str = data.get("last_modified_time") or data.get("created_at")
        if ts_str:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return datetime(2015, 9, 30)  # Stellar network genesis

    async def get_funding_source(self, account_id: str) -> Optional[str]:
        """Return the account that funded this wallet (create_account operation source)."""
        url = f"{self.horizon_url}/accounts/{account_id}/operations?limit=1&order=asc"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                records = resp.json().get("_embedded", {}).get("records", [])
                if records and records[0].get("type") == "create_account":
                    return records[0].get("funder")
        except Exception as exc:
            logger.debug("Could not resolve funding source for %s: %s", account_id, exc)
        return None
