"""Loading and cleaning the raw ESOL (Delaney) solubility dataset.

Design notes
------------
We deliberately keep this module dumb and boring: it only renames columns, drops
rows with missing critical fields, and removes exact duplicate rows. SMILES
*validity* is a separate concern (see `ml.data.validation`) because "the row is
present but incomplete" and "the row has an unparsable molecule" are different
failure modes that deserve separate handling and separate log lines when someone
is debugging a broken pipeline run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml import config

logger = logging.getLogger(__name__)


def load_raw_dataset(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw ESOL CSV and rename columns to the canonical internal schema.

    Raises
    ------
    FileNotFoundError
        If the dataset has not been downloaded yet (see scripts/download_data.py).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python scripts/download_data.py` first "
            "(or check the README for manual download instructions)."
        )

    df = pd.read_csv(path)

    missing_cols = {config.RAW_SMILES_COL, config.RAW_TARGET_COL} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Raw dataset is missing expected column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            config.RAW_ID_COL: config.ID_COL,
            config.RAW_SMILES_COL: config.SMILES_COL,
            config.RAW_TARGET_COL: config.TARGET_COL,
        }
    )

    keep_cols = [config.ID_COL, config.SMILES_COL, config.TARGET_COL]
    df = df[[c for c in keep_cols if c in df.columns]]

    logger.info("Loaded raw dataset: %d rows, %d columns", *df.shape)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and exact duplicates.

    Steps:
    1. Strip whitespace from SMILES strings (the raw file has some trailing spaces).
    2. Drop rows with a missing SMILES or missing target value.
    3. Drop exact duplicate rows (same SMILES *and* same target).
    4. Reset the index.
    """
    before = len(df)
    df = df.copy()

    df[config.SMILES_COL] = df[config.SMILES_COL].astype(str).str.strip()

    df = df.dropna(subset=[config.SMILES_COL, config.TARGET_COL])
    n_after_na = len(df)

    df = df.drop_duplicates(subset=[config.SMILES_COL, config.TARGET_COL], keep="first")
    n_after_dupes = len(df)

    df = df.reset_index(drop=True)

    logger.info(
        "Cleaned dataset: %d -> %d rows after dropping missing values "
        "(-%d) and exact duplicates (-%d)",
        before,
        n_after_dupes,
        before - n_after_na,
        n_after_na - n_after_dupes,
    )
    return df


def load_and_clean(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load the raw CSV and apply basic cleaning."""
    return clean_dataset(load_raw_dataset(path))
