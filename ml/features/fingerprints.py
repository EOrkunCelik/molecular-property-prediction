"""Morgan (circular / ECFP-like) fingerprints.

These are evaluated in `notebooks/01_eda_and_experiments.ipynb` and
`scripts/train_model.py --compare-representations` as an *alternative* representation
to the interpretable 2D descriptor set in `ml.features.descriptors`. Fingerprints
often give a small accuracy edge (they encode substructure information the fixed
descriptor list cannot) at the cost of interpretability: a Ridge/Random Forest model
trained on 1024 anonymous bits can no longer tell you "solubility dropped because LogP
went up", which matters for a portfolio project whose stated goal is a scientifically
defensible, explainable pipeline. The final shipped model therefore uses the
descriptor-based representation unless the comparison shows fingerprints give a
material accuracy improvement (see `docs/architecture.md` / README model comparison
section for the actual numbers from a real training run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from ml import config

RDLogger.DisableLog("rdApp.*")


def compute_morgan_fingerprint(
    smiles: str,
    radius: int = config.FINGERPRINT_RADIUS,
    n_bits: int = config.FINGERPRINT_N_BITS,
) -> np.ndarray:
    """Compute a single Morgan fingerprint as a dense 0/1 numpy array of length n_bits."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


def compute_fingerprints_for_dataframe(
    df: pd.DataFrame,
    smiles_col: str = config.CANONICAL_SMILES_COL,
    radius: int = config.FINGERPRINT_RADIUS,
    n_bits: int = config.FINGERPRINT_N_BITS,
) -> pd.DataFrame:
    """Compute Morgan fingerprints for every row; returns one column per bit (fp_0..fp_{n-1})."""
    fps = [compute_morgan_fingerprint(smi, radius, n_bits) for smi in df[smiles_col]]
    columns = [f"fp_{i}" for i in range(n_bits)]
    return pd.DataFrame(fps, columns=columns, index=df.index)
