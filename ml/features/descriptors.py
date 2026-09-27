"""RDKit 2D physicochemical descriptor calculation.

The descriptor set (see `ml.config.DESCRIPTOR_NAMES`) is not an arbitrary "throw every
RDKit descriptor at the model" choice. Each descriptor has a known, literature-supported
relationship to aqueous solubility (see Delaney, 2004 and the general "size, polarity,
lipophilicity" framework used across QSAR solubility models):

* MolWt, HeavyAtomCount        -> larger molecules are generally less soluble
* MolLogP                      -> the single strongest predictor: more lipophilic
                                   (higher LogP) molecules are less water-soluble
* TPSA, NumHDonors/Acceptors   -> polar / H-bonding molecules are more water-soluble
* NumRotatableBonds            -> flexible molecules pack less well into a crystal
                                   lattice, which (all else equal) increases solubility
* RingCount, NumAromaticRings, NumAliphaticRings, FractionCSP3
                                -> rigidity/planarity affects crystal packing energy,
                                   which is one of the two thermodynamic terms (the
                                   other being solvation) that determine solubility
* NumHeteroatoms                -> correlates with polarity/H-bonding capacity
* BalabanJ                      -> topological complexity, included as a general
                                   structural descriptor used in several QSPR studies

This module computes exactly these, plus a couple of `describe_molecule` convenience
fields (canonical SMILES, molecular formula) used purely for display in the API/UI and
NOT fed into the model.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from ml import config

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def _num_heteroatoms(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumHeteroatoms(mol)


# Registry mapping descriptor name -> callable(mol) -> float
_DESCRIPTOR_FUNCS: dict[str, callable] = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "NumHeteroatoms": _num_heteroatoms,
    "BalabanJ": Descriptors.BalabanJ,
}


def compute_descriptors(mol: Chem.Mol, names: list[str] | None = None) -> dict[str, float]:
    """Compute the configured set of RDKit descriptors for a single (already-parsed) molecule.

    Any descriptor that raises (RDKit's BalabanJ can occasionally fail on disconnected
    or unusual graphs) is recorded as NaN rather than aborting the whole computation,
    so one pathological molecule cannot silently drop a whole batch during training.
    """
    names = names or config.DESCRIPTOR_NAMES
    values: dict[str, float] = {}
    for name in names:
        func = _DESCRIPTOR_FUNCS.get(name)
        if func is None:
            raise KeyError(f"Unknown descriptor '{name}'. Add it to _DESCRIPTOR_FUNCS.")
        try:
            values[name] = float(func(mol))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Descriptor '%s' failed for a molecule: %s", name, exc)
            values[name] = float("nan")
    return values


def describe_molecule(smiles: str) -> dict:
    """Human-readable molecular information for a single SMILES string.

    Returns canonical SMILES, molecular formula and the full descriptor dict. Assumes
    the SMILES has already been validated (see `ml.data.validation.validate_smiles`);
    raises ValueError if it cannot be parsed, so callers should validate first if they
    want a friendlier error message.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")

    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "descriptors": compute_descriptors(mol),
    }


def compute_descriptors_for_dataframe(
    df: pd.DataFrame,
    smiles_col: str = config.CANONICAL_SMILES_COL,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Vectorized (row-wise) descriptor computation for a DataFrame of valid SMILES.

    Assumes every SMILES in `smiles_col` is already valid (i.e. this DataFrame has
    already been through `ml.data.validation.validate_dataframe`). Returns a new
    DataFrame with one column per descriptor, aligned to the input's index.
    """
    names = names or config.DESCRIPTOR_NAMES
    records = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Should not happen if validation ran first; fail loudly rather than
            # silently inserting NaNs that would only surface much later.
            raise ValueError(f"Unexpected invalid SMILES reached descriptor stage: {smi!r}")
        records.append(compute_descriptors(mol, names))

    desc_df = pd.DataFrame(records, index=df.index)
    n_nan = int(desc_df.isna().any(axis=1).sum())
    if n_nan:
        logger.warning("%d rows have at least one NaN descriptor value", n_nan)
    return desc_df
