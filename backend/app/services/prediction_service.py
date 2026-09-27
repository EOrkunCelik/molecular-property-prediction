"""Prediction service: the business-logic layer between the API endpoints and both the
ML model and the database.

Keeping this logic out of `api/v1/endpoints/predictions.py` means the endpoint module
only has to deal with HTTP concerns (status codes, request/response schemas), while
this module can be unit-tested (or reused from a CLI, a batch job, etc.) without
spinning up FastAPI at all.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import PredictionRecord
from ml.inference import InvalidSMILESError, PredictionResult, SolubilityPredictor

__all__ = ["InvalidSMILESError", "create_prediction", "get_prediction", "list_predictions"]


def create_prediction(
    db: Session, smiles: str, predictor: SolubilityPredictor
) -> PredictionRecord:
    """Run inference on `smiles` and persist the result. Raises InvalidSMILESError for bad input."""
    result: PredictionResult = predictor.predict(smiles)

    record = PredictionRecord(
        original_smiles=result.original_smiles,
        canonical_smiles=result.canonical_smiles,
        molecular_formula=result.molecular_formula,
        predicted_log_solubility=result.predicted_log_solubility,
        descriptors=result.descriptors,
        model_name=result.model_name,
        model_metadata=result.model_metadata,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_prediction(db: Session, prediction_id: str) -> PredictionRecord | None:
    return db.get(PredictionRecord, prediction_id)


def list_predictions(
    db: Session, limit: int = 20, offset: int = 0
) -> tuple[list[PredictionRecord], int]:
    """Return a page of prediction history, newest first, plus the total row count."""
    total = db.scalar(select(func.count()).select_from(PredictionRecord)) or 0

    items = (
        db.execute(
            select(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return list(items), total
