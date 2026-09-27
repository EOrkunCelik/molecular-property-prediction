"""End-to-end dataset -> feature matrix pipeline.

`build_feature_matrix` is intentionally the *only* place that turns a DataFrame of
validated molecules into `(X, y)`. Both `scripts/train_model.py` (offline) and
`ml.inference.predict` (online, via a single-row DataFrame) call this same function, so
there is exactly one definition of "what a feature vector for this model looks like".
"""

from __future__ import annotations

import pandas as pd

from ml import config
from ml.features.descriptors import compute_descriptors_for_dataframe


def build_feature_matrix(
    df: pd.DataFrame,
    descriptor_names: list[str] | None = None,
    include_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Compute the descriptor feature matrix `X` (and target `y`, if present) for a
    DataFrame that already has a `canonical_smiles` column (i.e. has passed through
    `ml.data.validation.validate_dataframe`).
    """
    descriptor_names = descriptor_names or config.DESCRIPTOR_NAMES
    X = compute_descriptors_for_dataframe(df, names=descriptor_names)

    y = None
    if include_target and config.TARGET_COL in df.columns:
        y = df[config.TARGET_COL].reset_index(drop=True)
        X = X.reset_index(drop=True)

    return X, y
