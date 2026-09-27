from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.ml_runtime.model_loader import get_predictor
from ml.features.descriptors import describe_molecule
from ml.inference import InvalidSMILESError, PredictionResult


@dataclass
class _FakeModelBundle:
    metadata: dict


class FakePredictor:
    """Stands in for `ml.inference.SolubilityPredictor` in API tests.

    Uses real RDKit descriptor calculation (so the descriptors returned are genuine),
    but a trivial hard-coded "model" (predicted value derived deterministically from
    MolLogP) instead of loading an actual trained joblib artifact — API tests care
    about request/response plumbing and persistence, not model accuracy, which is
    covered separately by the ML test suite in the top-level `tests/` directory.
    """

    def __init__(self):
        self.bundle = _FakeModelBundle(
            metadata={
                "model_name": "fake_test_model",
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "split_strategy": "scaffold",
                "extra": {"test_metrics": {"mae": 0.5, "rmse": 0.7, "r2": 0.85}},
            }
        )

    def predict(self, smiles: str) -> PredictionResult:
        try:
            info = describe_molecule(smiles)
        except ValueError as exc:
            raise InvalidSMILESError(str(exc)) from exc

        fake_prediction = -0.5 * info["descriptors"]["MolLogP"]
        return PredictionResult(
            original_smiles=smiles,
            canonical_smiles=info["canonical_smiles"],
            molecular_formula=info["molecular_formula"],
            descriptors=info["descriptors"],
            predicted_log_solubility=fake_prediction,
            model_name=self.bundle.metadata["model_name"],
            model_metadata=self.bundle.metadata,
        )


@pytest.fixture
def db_session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def client(db_session_factory):
    def override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    def override_get_predictor():
        return FakePredictor()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_predictor] = override_get_predictor

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
