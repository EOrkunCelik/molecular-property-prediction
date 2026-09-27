"""Dataset-level preprocessing after molecular validation.

This module handles repeated molecular structures after SMILES canonicalization.
Multiple source rows can represent the same canonical molecule, either because
they are duplicate measurements or because different input SMILES collapse to
the same molecular representation.

For supervised regression, keeping such rows separately would give identical
molecular features multiple target values and would also overweight repeated
structures. We therefore aggregate repeated canonical structures into one
observation using the median target value.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml import config

logger = logging.getLogger(__name__)


def aggregate_canonical_duplicates(
    df: pd.DataFrame,
    canonical_col: str = config.CANONICAL_SMILES_COL,
    target_col: str = config.TARGET_COL,
) -> pd.DataFrame:
    """Collapse repeated canonical molecules into one regression observation.

    Unique molecules are left unchanged. For molecules occurring multiple times,
    the target is replaced by the median of all measurements associated with that
    canonical structure. Other columns are taken from the first occurrence.

    This prevents repeated structures from receiving disproportionate weight and
    avoids presenting identical molecular representations with conflicting target
    values to the regression model.
    """
    if canonical_col not in df.columns:
        raise ValueError(f"Missing canonical molecule column: {canonical_col!r}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col!r}")

    before = len(df)

    aggregated = (
        df.groupby(canonical_col, sort=False, as_index=False)
        .agg(
            {
                **{
                    column: "first"
                    for column in df.columns
                    if column not in {canonical_col, target_col}
                },
                target_col: "median",
            }
        )
        .reset_index(drop=True)
    )

    removed = before - len(aggregated)

    logger.info(
        "Canonical duplicate aggregation: %d -> %d rows (%d repeated rows collapsed)",
        before,
        len(aggregated),
        removed,
    )

    return aggregated
