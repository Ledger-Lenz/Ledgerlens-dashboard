from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import alerts, assets, scores
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv()

    horizon_url = os.getenv("HORIZON_URL", "https://horizon-testnet.stellar.org")
    model_dir = os.getenv("ML_MODEL_PATH", "models/")
    contract_id = os.getenv("LEDGERLENS_CONTRACT_ID")

    # Ingestion
    from ingestion.historical_loader import HistoricalLoader
    app.state.loader = HistoricalLoader(horizon_url)

    # Inference
    from detection.model_inference import ModelInference
    inference = ModelInference(model_dir)
    inference.load()
    app.state.inference = inference

    # SHAP explainer (optional — requires trained XGBoost model)
    from detection.shap_explainer import ShapExplainer
    explainer = ShapExplainer(model_dir)
    if inference._xgb is not None:
        explainer.load(inference._xgb)
    app.state.explainer = explainer

    # In-memory stores (replace with DB in production)
    app.state.score_cache = {}
    app.state.alert_store = []
    app.state.asset_store = {}

    app.state.horizon_url = horizon_url
    app.state.contract_id = contract_id
    app.state.models_loaded = inference._loaded

    logger.info("LedgerLens API started — horizon=%s", horizon_url)
    yield
    logger.info("LedgerLens API shutting down")


app = FastAPI(
    title="LedgerLens API",
    description=(
        "Hybrid on-chain fraud detection for the Stellar DEX. "
        "Detects wash trading using Benford's Law + ensemble ML."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(scores.router)
app.include_router(alerts.router)
app.include_router(assets.router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="ok",
        horizon_url=app.state.horizon_url,
        contract_id=app.state.contract_id,
        models_loaded=app.state.models_loaded,
    )
