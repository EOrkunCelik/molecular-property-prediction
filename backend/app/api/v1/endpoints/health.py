from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.ml_runtime.model_loader import get_predictor, is_model_loaded
from app.schemas.prediction import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Liveness/readiness probe: reports whether the DB and the ML model are reachable.

    Returns HTTP 200 even when a dependency is down (so this endpoint itself never
    flaps due to transient DB issues) — callers should inspect the response body, not
    just the status code, to decide readiness.
    """
    model_loaded = is_model_loaded()
    model_name = None
    if model_loaded:
        model_name = get_predictor().bundle.metadata.get("model_name")

    database_connected = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database_connected = False

    status = "ok" if (model_loaded and database_connected) else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_name=model_name,
        database_connected=database_connected,
    )
