from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    horizon_url: str = "https://horizon-testnet.stellar.org"
    network: str = "testnet"

    target_asset_code: str = ""
    target_asset_issuer: str = ""

    ledgerlens_contract_id: str = ""
    service_account_secret: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    alert_threshold: int = 75
    benford_min_samples: int = 30
    benford_chi_square_p_value: float = 0.05
    benford_mad_threshold: float = 0.015

    ml_model_path: str = "models/"
    ml_retrain_interval_hours: int = 24

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
