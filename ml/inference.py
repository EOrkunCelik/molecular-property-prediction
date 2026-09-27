"""Reproducible single-molecule inference.

This is the one function (`predict_solubility`) the FastAPI backend needs to know
about. It re-uses the exact same validation and descriptor-computation code the
training pipeline uses (see module docstrings in `ml.data.validation` and
`ml.pipeline`), so "the model behaves the same in the API as it did during training"
is true by construction rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml import config
from ml.data.validation import ValidationResult, validate_smiles
from ml.features.descriptors import describe_molecule
from ml.models.registry import ModelBundle, load_model_bundle


class InvalidSMILESError(ValueError):
    """Raised when the input SMILES string cannot be parsed by RDKit."""


@dataclass
class PredictionResult:
    original_smiles: str
    canonical_smiles: str
    molecular_formula: str
    descriptors: dict[str, float]
    predicted_log_solubility: float
    model_name: str
    model_metadata: dict[str, Any]


class SolubilityPredictor:
    """Thin wrapper around a loaded model bundle that exposes a single `predict` method.

    Instantiate once (e.g. as a FastAPI dependency / app-state singleton) and reuse
    across requests — loading the model from disk on every request would be wasteful
    and, for larger models than this one, slow.
    """

    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.bundle: ModelBundle = load_model_bundle(models_dir)
        self._feature_names: list[str] = self.bundle.metadata.get(
            "feature_names", config.DESCRIPTOR_NAMES
        )
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        """Guard against silent train/serve skew.

        If someone changes `config.DESCRIPTOR_NAMES` without retraining, the saved
        model's `feature_names` (frozen at training time) will no longer match the
        descriptors the *current* code would compute. Rather than silently feeding the
        model a differently-ordered or differently-sized vector, fail loudly at
        startup so the mismatch is caught in CI/local dev, not in production.
        """
        current = set(config.DESCRIPTOR_NAMES)
        trained_on = set(self._feature_names)
        if current != trained_on:
            raise RuntimeError(
                "Model/feature mismatch: the loaded model was trained on descriptors "
                f"{sorted(trained_on)} but the current code computes {sorted(current)}. "
                "Retrain the model with `python scripts/train_model.py`."
            )

    def predict(self, smiles: str) -> PredictionResult:
        validation: ValidationResult = validate_smiles(smiles)
        if not validation.is_valid:
            raise InvalidSMILESError(validation.error or "Invalid SMILES string")

        info = describe_molecule(validation.canonical_smiles)
        descriptors = info["descriptors"]

        # Build a single-row DataFrame with columns in the exact order the model
        # expects, so this works regardless of dict ordering guarantees.
        row = pd.DataFrame([[descriptors[name] for name in self._feature_names]], columns=self._feature_names)

        prediction = float(self.bundle.model.predict(row)[0])

        return PredictionResult(
            original_smiles=smiles,
            canonical_smiles=info["canonical_smiles"],
            molecular_formula=info["molecular_formula"],
            descriptors=descriptors,
            predicted_log_solubility=prediction,
            model_name=self.bundle.metadata.get("model_name", "unknown"),
            model_metadata=self.bundle.metadata,
        )
