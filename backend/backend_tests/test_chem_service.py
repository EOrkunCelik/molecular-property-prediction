from __future__ import annotations

import pytest

from app.services.chem import InvalidSMILESError, validate_and_describe


class TestValidateAndDescribe:
    def test_valid_smiles_returns_descriptors(self):
        info = validate_and_describe("CCO")  # ethanol
        assert info["molecular_formula"] == "C2H6O"
        assert "MolWt" in info["descriptors"]

    def test_invalid_smiles_raises(self):
        with pytest.raises(InvalidSMILESError):
            validate_and_describe("definitely not a smiles!!")

    def test_empty_smiles_raises(self):
        with pytest.raises(InvalidSMILESError):
            validate_and_describe("")
