from __future__ import annotations

import pandas as pd
import pytest

from ml import config


@pytest.fixture
def tiny_raw_dataframe() -> pd.DataFrame:
    """A tiny, hand-built dataset with the same schema as the raw ESOL CSV.

    Includes: valid distinct molecules, one duplicate row, one row with a missing
    target, and one row with an unparsable SMILES string — enough to exercise
    cleaning, validation, and (with a slightly larger set) splitting.
    """
    return pd.DataFrame(
        {
            config.ID_COL: ["ethanol", "benzene", "benzene_dup", "no_target", "bad_smiles", "methanol", "toluene", "phenol"],
            config.SMILES_COL: ["CCO", "c1ccccc1", "c1ccccc1", "CCC", "not_a_smiles!!", "CO", "Cc1ccccc1", "Oc1ccccc1"],
            config.TARGET_COL: [-0.14, -1.64, -1.64, None, -1.0, -0.5, -2.21, 0.0],
        }
    )


@pytest.fixture
def valid_smiles_dataframe() -> pd.DataFrame:
    """A small DataFrame of already-valid, already-canonicalized molecules with targets,
    large enough (8 distinct scaffolds) to exercise splitting without every split being empty.
    """
    data = {
        config.ID_COL: [
            "ethanol", "propanol", "butanol", "pentanol",
            "benzene", "toluene", "phenol", "aniline",
        ],
        config.SMILES_COL: [
            "CCO", "CCCO", "CCCCO", "CCCCCO",
            "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",
        ],
        config.TARGET_COL: [-0.14, 0.62, 0.0, -0.6, -1.64, -2.21, 0.0, -0.41],
    }
    df = pd.DataFrame(data)
    df[config.CANONICAL_SMILES_COL] = df[config.SMILES_COL]
    return df
