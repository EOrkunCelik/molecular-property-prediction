"""Loads the trained model once per process and exposes it as a FastAPI dependency.

Model loading (deserializing joblib + validating metadata) takes on the order of
milliseconds for this project's model, but doing it on every request would still be
wasteful and, for a larger model, would matter a lot. `get_predictor` is cached with
`lru_cache` so all requests share one `SolubilityPredictor` instance for the lifetime
of the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from ml.inference import SolubilityPredictor

logger = logging.getLogger(__name__)


@lru_cache
def get_predictor() -> SolubilityPredictor:
    settings = get_settings()
    models_dir = Path(settings.MODEL_ARTIFACT_DIR)
    logger.info("Loading model bundle from %s", models_dir)
    return SolubilityPredictor(models_dir=models_dir)


def is_model_loaded() -> bool:
    """Health-check helper: True if the model has been loaded (or loads successfully now)."""
    try:
        get_predictor()
        return True
    except Exception as exc:  # noqa: BLE001 - health check should never raise
        logger.warning("Model is not currently loadable: %s", exc)
        return False
