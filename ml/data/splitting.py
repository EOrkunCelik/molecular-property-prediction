"""Train/validation/test splitting strategies.

Why scaffold splitting matters here
------------------------------------
A purely random split of a molecular dataset tends to put near-identical analogues
(e.g. a series of chlorinated benzenes, or a homologous alcohol series like
1-butanol/1-pentanol/1-hexanol) on both sides of the train/test boundary. A model can
then do well on the test set simply by memorising "molecules that look like this have
solubility around X", which overstates how well it would generalise to a genuinely new
scaffold. A scaffold split (grouping molecules by their Bemis-Murcko core scaffold and
keeping each scaffold entirely within one split) is the standard, more realistic
evaluation protocol used in cheminformatics benchmarks such as MoleculeNet, and is used
here as the primary evaluation split. A random split is also produced for comparison,
so the README can honestly report both numbers and explain the gap.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

from ml import config

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def _generate_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko generic scaffold SMILES for a molecule.

    Falls back to the original SMILES (treated as its own singleton "scaffold") if the
    scaffold cannot be computed, which can happen for very small / unusual molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else smiles
    except Exception:  # pragma: no cover - defensive; RDKit scaffold edge cases
        return smiles


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = config.CANONICAL_SMILES_COL,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/val/test by Bemis-Murcko scaffold groups.

    Groups (not individual molecules) are shuffled and then greedily assigned to
    test -> val -> train until each split's target size is reached, so that no
    scaffold appears in more than one split.
    """
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for idx, smi in zip(df.index, df[smiles_col], strict=True):
        scaffolds[_generate_scaffold(smi)].append(idx)

    groups = list(scaffolds.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    groups.sort(key=len, reverse=True)

    n_total = len(df)
    targets = {
    "test": test_size * n_total,
    "val": val_size * n_total,
    "train": (1.0 - test_size - val_size) * n_total,
    }
    split_indices: dict[str, list[int]] = {
    "train": [],
    "val": [],
    "test": [],
    }
    for group in groups:
        # Assign the complete scaffold to the split that is furthest below
        # its requested size, relative to that split's target.
        group_size = len(group)

        eligible = [
            name
            for name in targets
            if len(split_indices[name]) + group_size <= targets[name]
        ]
        if eligible:
            split_name = max(
                eligible,
                key=lambda name: targets[name] - len(split_indices[name]),
            )
        else:
            split_name = min(
                targets,
                key=lambda name: (
                    len(split_indices[name]) + group_size - targets[name]
                )
                / targets[name],
            )

        split_indices[split_name].extend(group)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    logger.info(
        "Scaffold split: %d unique scaffolds -> train=%d, val=%d, test=%d",
        len(groups),
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def random_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simple i.i.d. random split, provided for comparison against the scaffold split."""
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=relative_val_size, random_state=seed)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Random split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df)
    )
    return train_df, val_df, test_df


def assert_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smiles_col: str = config.CANONICAL_SMILES_COL,
) -> None:
    """Raise an AssertionError if any canonical SMILES appears in more than one split.

    This is a cheap, high-value sanity check to run right after splitting and again
    right before training: it catches the most common and most damaging form of data
    leakage in molecular ML pipelines (the same molecule, or a duplicate entry of it,
    ending up in both train and test).
    """
    train_set = set(train_df[smiles_col])
    val_set = set(val_df[smiles_col])
    test_set = set(test_df[smiles_col])

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    assert not train_val_overlap, f"Leakage: {len(train_val_overlap)} molecules in both train and val"
    assert not train_test_overlap, f"Leakage: {len(train_test_overlap)} molecules in both train and test"
    assert not val_test_overlap, f"Leakage: {len(val_test_overlap)} molecules in both val and test"

    logger.info("No data leakage detected between train/val/test splits.")
