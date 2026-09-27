from __future__ import annotations

import math

import pytest
from rdkit import Chem

from ml import config
from ml.features.descriptors import compute_descriptors, describe_molecule


class TestComputeDescriptors:
    def test_ethanol_molecular_weight(self):
        mol = Chem.MolFromSmiles("CCO")
        descriptors = compute_descriptors(mol)
        # Ethanol MW is ~46.07 g/mol; allow a little tolerance for RDKit version drift.
        assert math.isclose(descriptors["MolWt"], 46.07, abs_tol=0.05)

    def test_returns_all_configured_descriptors(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        descriptors = compute_descriptors(mol)
        assert set(descriptors.keys()) == set(config.DESCRIPTOR_NAMES)

    def test_benzene_has_one_aromatic_ring_and_zero_hbond_donors(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        descriptors = compute_descriptors(mol)
        assert descriptors["NumAromaticRings"] == 1
        assert descriptors["NumHDonors"] == 0

    def test_phenol_has_one_hbond_donor(self):
        mol = Chem.MolFromSmiles("Oc1ccccc1")
        descriptors = compute_descriptors(mol)
        assert descriptors["NumHDonors"] == 1

    def test_unknown_descriptor_name_raises(self):
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(KeyError):
            compute_descriptors(mol, names=["NotARealDescriptor"])


class TestDescribeMolecule:
    def test_describe_ethanol(self):
        info = describe_molecule("CCO")
        assert info["molecular_formula"] == "C2H6O"
        assert "descriptors" in info
        assert info["canonical_smiles"]

    def test_invalid_smiles_raises_value_error(self):
        with pytest.raises(ValueError):
            describe_molecule("not a smiles!!")
