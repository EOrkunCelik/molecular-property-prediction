"""SMILES validation and canonicalization.

This module is used both by the offline training pipeline (to filter out unparsable
molecules before featurization) and by the FastAPI backend at request time (to reject
invalid user input with a clear error message). Keeping a single implementation avoids
train/serve skew: a molecule that RDKit rejects at inference time should be exactly the
same set of molecules RDKit rejected while building the training set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from rdkit import Chem, RDLogger

from ml import config

logger = logging.getLogger(__name__)

# RDKit logs a lot of "SMILES Parse Error" noise to stderr by default; we handle
# invalid-molecule reporting ourselves via `ValidationResult`, so silence it.
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a single SMILES string."""

    is_valid: bool
    original_smiles: str
    canonical_smiles: str | None = None
    error: str | None = None


def validate_smiles(smiles: str) -> ValidationResult:
    """Parse a single SMILES string with RDKit and return a canonical form if valid.

    A SMILES string is considered invalid if:
      * it is empty / not a string,
      * RDKit's parser (`Chem.MolFromSmiles`) cannot build a molecule from it, or
      * the resulting molecule has zero atoms.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return ValidationResult(is_valid=False, original_smiles=str(smiles), error="Empty SMILES string")

    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return ValidationResult(
            is_valid=False,
            original_smiles=smiles,
            error="RDKit could not parse this SMILES string into a valid molecule",
        )

    if mol.GetNumAtoms() == 0:
        return ValidationResult(
            is_valid=False, original_smiles=smiles, error="Parsed molecule has zero atoms"
        )

    canonical = Chem.MolToSmiles(mol, canonical=True)
    return ValidationResult(is_valid=True, original_smiles=smiles, canonical_smiles=canonical)


def validate_dataframe(
    df: pd.DataFrame, smiles_col: str = config.SMILES_COL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate every SMILES string in a DataFrame.

    Returns
    -------
    (valid_df, invalid_df)
        `valid_df` has an added `canonical_smiles` column and only contains rows whose
        SMILES parsed successfully. `invalid_df` contains the rejected rows plus the
        RDKit error message, for auditing / error analysis.
    """
    results = df[smiles_col].apply(validate_smiles)

    df = df.copy()
    df["_is_valid"] = [r.is_valid for r in results]
    df[config.CANONICAL_SMILES_COL] = [r.canonical_smiles for r in results]
    df["_validation_error"] = [r.error for r in results]

    valid_df = df[df["_is_valid"]].drop(columns=["_is_valid", "_validation_error"]).reset_index(
        drop=True
    )
    invalid_df = df[~df["_is_valid"]].drop(
        columns=["_is_valid", config.CANONICAL_SMILES_COL]
    ).reset_index(drop=True)

    logger.info(
        "SMILES validation: %d valid, %d invalid (of %d total)",
        len(valid_df),
        len(invalid_df),
        len(df),
    )
    if len(invalid_df) > 0:
        logger.warning(
            "Dropped %d rows with invalid SMILES: %s",
            len(invalid_df),
            invalid_df[smiles_col].tolist()[:10],
        )

    return valid_df, invalid_df
