import pandas as pd
import pytest

from ml import config
from ml.data.preprocessing import aggregate_canonical_duplicates


def test_aggregate_canonical_duplicates():
    df = pd.DataFrame(
        {
            config.ID_COL: ["a", "b", "c"],
            config.SMILES_COL: ["CCO", "OCC", "CC"],
            config.CANONICAL_SMILES_COL: ["CCO", "CCO", "CC"],
            config.TARGET_COL: [-1.0, -3.0, -2.0],
        }
    )

    result = aggregate_canonical_duplicates(df)

    assert len(result) == 2

    ethanol = result[result[config.CANONICAL_SMILES_COL] == "CCO"].iloc[0]
    assert ethanol[config.TARGET_COL] == pytest.approx(-2.0)

    ethane = result[result[config.CANONICAL_SMILES_COL] == "CC"].iloc[0]
    assert ethane[config.TARGET_COL] == pytest.approx(-2.0)


def test_unique_molecules_are_preserved():
    df = pd.DataFrame(
        {
            config.ID_COL: ["a", "b"],
            config.SMILES_COL: ["CCO", "CC"],
            config.CANONICAL_SMILES_COL: ["CCO", "CC"],
            config.TARGET_COL: [-1.0, -2.0],
        }
    )

    result = aggregate_canonical_duplicates(df)

    assert len(result) == 2
    assert result[config.TARGET_COL].tolist() == [-1.0, -2.0]
