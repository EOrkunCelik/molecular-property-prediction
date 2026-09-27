"""On-disk model registry: save/load the fitted model plus everything needed to
reproduce its exact input feature vector at inference time.

This is the single most important module for avoiding train/serve skew. Everything
`ml.inference` needs to turn a raw SMILES string into the same feature vector the
model was trained on (descriptor names, their order, the target column name, the
sklearn/rdkit versions used) is written here at training time and read back here at
serving time. If someone changes `config.DESCRIPTOR_NAMES` and forgets to retrain,
`load_model_bundle` will still work, but the metadata will visibly show a mismatch
between the descriptors the *current* code computes and the descriptors the *saved*
model expects — which is exactly the kind of silent bug this pattern is meant to catch
early (see the assertion in `ml.inference.predict`).
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn

from ml import config


@dataclass
class ModelMetadata:
    model_name: str
    feature_names: list[str]
    target_col: str
    split_strategy: str
    trained_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    sklearn_version: str = field(default_factory=lambda: sklearn.__version__)
    python_version: str = field(default_factory=platform.python_version)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def save_model_bundle(
    model: Any,
    metadata: ModelMetadata,
    metrics: dict[str, Any],
    models_dir: Path = config.MODELS_DIR,
) -> None:
    """Persist the fitted model, its metadata, and its evaluation metrics to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / config.MODEL_FILENAME)

    with open(models_dir / config.METADATA_FILENAME, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    with open(models_dir / config.METRICS_FILENAME, "w") as f:
        json.dump(metrics, f, indent=2)


@dataclass
class ModelBundle:
    model: Any
    metadata: dict
    metrics: dict


def load_model_bundle(models_dir: Path = config.MODELS_DIR) -> ModelBundle:
    """Load the fitted model, its metadata and its metrics from disk.

    Raises FileNotFoundError with a helpful message if the model hasn't been trained
    yet — this is the error the backend surfaces (as a 503) if it starts up before
    `scripts/train_model.py` has been run.
    """
    model_path = models_dir / config.MODEL_FILENAME
    metadata_path = models_dir / config.METADATA_FILENAME
    metrics_path = models_dir / config.METRICS_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python scripts/train_model.py` "
            "first (see the README's 'Reproducing the model' section)."
        )

    model = joblib.load(model_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return ModelBundle(model=model, metadata=metadata, metrics=metrics)
