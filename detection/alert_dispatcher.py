"""
Dispatches flagged wallet events to registered webhooks.

Called by the scoring pipeline whenever a wallet's score crosses the
alert threshold. Delivers JSON payloads with HMAC-SHA256 signatures.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx

from detection.model_inference import RiskScore

logger = logging.getLogger(__name__)

_TIMEOUT = 5.0


def _sign(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _build_payload(risk: RiskScore, event: str = "alert") -> bytes:
    return json.dumps(
        {
            "event": event,
            "wallet": risk.wallet,
            "asset_pair": risk.asset_pair,
            "score": risk.score,
            "benford_flag": risk.benford_flag,
            "ml_flag": risk.ml_flag,
            "confidence": risk.confidence,
            "timestamp": int(time.time()),
        },
        separators=(",", ":"),
    ).encode()


async def dispatch_alert(
    risk: RiskScore,
    webhooks: list[dict[str, Any]],
    event: str = "alert",
) -> None:
    """Fire the alert to all registered webhook endpoints asynchronously."""
    if not webhooks:
        return

    payload = _build_payload(risk, event)
    headers = {
        "Content-Type": "application/json",
        "X-LedgerLens-Event": event,
        "X-LedgerLens-Score": str(risk.score),
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for wh in webhooks:
            url = wh.get("url", "")
            secret = wh.get("secret", "")
            wh_headers = dict(headers)
            if secret:
                wh_headers["X-LedgerLens-Signature"] = _sign(payload, secret)
            try:
                resp = await client.post(url, content=payload, headers=wh_headers)
                if resp.is_success:
                    logger.debug("Webhook delivered → %s  status=%d", url, resp.status_code)
                else:
                    logger.warning("Webhook non-2xx → %s  status=%d", url, resp.status_code)
            except Exception as exc:
                logger.error("Webhook delivery failed → %s: %s", url, exc)
