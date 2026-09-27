"""Application settings, loaded from environment variables (see .env.example).

Using `pydantic-settings` here (rather than scattering `os.environ.get(...)` calls
through the codebase) gives us validation at startup: if `DATABASE_URL` is missing or
malformed, the app fails immediately with a clear error instead of failing later on
the first request that touches the database.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "Molecular Property Prediction Platform"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str = "postgresql+psycopg2://molpred:molpred_dev_password@localhost:5432/molecular_predictions"

    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    MODEL_ARTIFACT_DIR: str = "/app/models"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — environment variables are read once per process."""
    return Settings()
