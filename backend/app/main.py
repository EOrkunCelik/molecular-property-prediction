"""FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Or via Docker Compose (see the repository root docker-compose.yml), which is the
intended way to run the full stack (backend + Postgres + frontend) together.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.base import Base
from app.db.session import engine

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Predicts aqueous solubility (LogS) for small organic molecules from a SMILES "
        "string, using RDKit for cheminformatics and a scikit-learn regression model "
        "trained on the ESOL (Delaney) dataset. Predictions are experimental/educational "
        "and are not a substitute for laboratory measurement."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Starting %s (%s)", settings.PROJECT_NAME, settings.ENVIRONMENT)
    # Convenience for local/dev/demo use: ensure tables exist even if Alembic
    # migrations haven't been run yet. This is idempotent (CREATE TABLE IF NOT
    # EXISTS-equivalent) and safe to call every startup; production deployments
    # should still prefer running `alembic upgrade head` explicitly as part of the
    # deploy step (see backend/alembic/).
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not initialize database tables at startup: %s", exc)


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "name": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }


@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """The model hasn't been trained yet (`ml.models.registry.load_model_bundle` raises
    `FileNotFoundError` in that case). Surfaced as a clean 503 with the same actionable
    message the CLI scripts print, rather than an opaque 500 — this is the error someone
    will hit if they `docker compose up` before running `python scripts/train_model.py`.
    """
    logger.warning("Model not available: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so an unexpected error returns a clean JSON 500 instead of leaking a stack trace."""
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
