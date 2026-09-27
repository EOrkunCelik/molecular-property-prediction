from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.db.models import PredictionRecord
from app.db.session import get_db
from app.ml_runtime.model_loader import get_predictor
from app.schemas.prediction import (
    PredictionListItem,
    PredictionListResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services import prediction_service
from app.services.chem import InvalidSMILESError
from app.services.visualization import render_structure_svg
from ml.inference import SolubilityPredictor

router = APIRouter()


def _to_response(record: PredictionRecord) -> PredictionResponse:
    """Adapt a SQLAlchemy row into the richer API response shape."""
    return PredictionResponse(
        id=record.id,
        original_smiles=record.original_smiles,
        canonical_smiles=record.canonical_smiles,
        molecular_formula=record.molecular_formula,
        descriptors=record.descriptors,
        predicted_log_solubility=record.predicted_log_solubility,
        predicted_solubility_mol_per_l=10.0**record.predicted_log_solubility,
        model_info={
            "model_name": record.model_name,
            "trained_at_utc": record.model_metadata.get("trained_at_utc"),
            "split_strategy": record.model_metadata.get("split_strategy"),
            "test_metrics": record.model_metadata.get("extra", {}).get("test_metrics"),
        },
        created_at=record.created_at,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["predictions"],
    summary="Predict aqueous solubility (LogS) for a molecule given as a SMILES string.",
)
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    predictor: SolubilityPredictor = Depends(get_predictor),
) -> PredictionResponse:
    try:
        record = prediction_service.create_prediction(db, request.smiles, predictor)
    except InvalidSMILESError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid SMILES string: {exc}",
        ) from exc
    return _to_response(record)


@router.get(
    "/predictions",
    response_model=PredictionListResponse,
    tags=["predictions"],
    summary="List past predictions, newest first.",
)
def list_predictions(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> PredictionListResponse:
    items, total = prediction_service.list_predictions(db, limit=limit, offset=offset)
    return PredictionListResponse(
        items=[PredictionListItem.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionResponse,
    tags=["predictions"],
    summary="Fetch a single prediction by ID.",
)
def get_prediction(prediction_id: str, db: Session = Depends(get_db)) -> PredictionResponse:
    record = prediction_service.get_prediction(db, prediction_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return _to_response(record)


@router.get(
    "/predictions/{prediction_id}/structure",
    tags=["predictions"],
    summary="2D SVG structure drawing for a previously made prediction.",
    response_class=Response,
)
def get_prediction_structure(prediction_id: str, db: Session = Depends(get_db)) -> Response:
    record = prediction_service.get_prediction(db, prediction_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    svg = render_structure_svg(record.canonical_smiles)
    return Response(content=svg, media_type="image/svg+xml")


@router.get(
    "/visualize",
    tags=["predictions"],
    summary="2D SVG structure drawing for an arbitrary SMILES string (no prediction/persistence).",
    response_class=Response,
)
def visualize(smiles: str = Query(..., min_length=1, max_length=2000)) -> Response:
    try:
        svg = render_structure_svg(smiles)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return Response(content=svg, media_type="image/svg+xml")
