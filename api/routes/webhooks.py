"""
Webhook delivery route for protocol team integrations.

POST /webhooks/subscribe — register a URL to receive alert payloads
DELETE /webhooks/{id} — unsubscribe
POST /webhooks/test/{id} — fire a test payload to validate the endpoint
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, HttpUrl

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


class SubscribeRequest(BaseModel):
    url: HttpUrl
    secret: str = ""


class SubscribeResponse(BaseModel):
    id: str
    url: str
    registered_at: float


class WebhookRecord(BaseModel):
    id: str
    url: str
    secret: str
    registered_at: float


def _hmac_signature(payload: bytes, secret: str) -> str:
    import hmac as _hmac
    return _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


@router.post("/subscribe", response_model=SubscribeResponse, status_code=201)
async def subscribe(request: Request, body: SubscribeRequest):
    store = request.app.state.webhook_store
    record = WebhookRecord(
        id=str(uuid.uuid4()),
        url=str(body.url),
        secret=body.secret,
        registered_at=time.time(),
    )
    store[record.id] = record.model_dump()
    logger.info("Webhook subscribed: %s → %s", record.id, record.url)
    return SubscribeResponse(id=record.id, url=record.url, registered_at=record.registered_at)


@router.delete("/{webhook_id}", status_code=204)
async def unsubscribe(
    request: Request,
    webhook_id: Annotated[str, Path()],
):
    store = request.app.state.webhook_store
    if webhook_id not in store:
        raise HTTPException(status_code=404, detail="Webhook not found")
    del store[webhook_id]
    logger.info("Webhook unsubscribed: %s", webhook_id)


@router.post("/test/{webhook_id}", status_code=202)
async def test_webhook(
    request: Request,
    webhook_id: Annotated[str, Path()],
):
    store = request.app.state.webhook_store
    if webhook_id not in store:
        raise HTTPException(status_code=404, detail="Webhook not found")

    record = store[webhook_id]
    payload = b'{"event":"test","message":"LedgerLens webhook test"}'
    headers = {"Content-Type": "application/json", "X-LedgerLens-Event": "test"}
    if record["secret"]:
        headers["X-LedgerLens-Signature"] = _hmac_signature(payload, record["secret"])

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(record["url"], content=payload, headers=headers)
        return {"status": resp.status_code, "ok": resp.is_success}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Delivery failed: {exc}") from exc
