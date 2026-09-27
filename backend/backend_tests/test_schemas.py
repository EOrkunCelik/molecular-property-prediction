from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.prediction import PredictionRequest


class TestPredictionRequest:
    def test_valid_smiles_is_accepted(self):
        req = PredictionRequest(smiles="CCO")
        assert req.smiles == "CCO"

    def test_smiles_is_stripped(self):
        req = PredictionRequest(smiles="  CCO  ")
        assert req.smiles == "CCO"

    def test_blank_smiles_is_rejected(self):
        with pytest.raises(ValidationError):
            PredictionRequest(smiles="   ")

    def test_empty_smiles_is_rejected(self):
        with pytest.raises(ValidationError):
            PredictionRequest(smiles="")

    def test_overly_long_smiles_is_rejected(self):
        with pytest.raises(ValidationError):
            PredictionRequest(smiles="C" * 3000)
