from __future__ import annotations

from ml.data.validation import validate_dataframe, validate_smiles


class TestValidateSmiles:
    def test_valid_smiles_is_accepted(self):
        result = validate_smiles("CCO")  # ethanol
        assert result.is_valid
        assert result.error is None
        assert result.canonical_smiles is not None

    def test_canonicalization_is_idempotent(self):
        # Two different (but equivalent) SMILES for benzene should canonicalize identically.
        r1 = validate_smiles("c1ccccc1")
        r2 = validate_smiles("C1=CC=CC=C1")
        assert r1.is_valid and r2.is_valid
        assert r1.canonical_smiles == r2.canonical_smiles

    def test_garbage_string_is_rejected(self):
        result = validate_smiles("this is not a smiles string!!")
        assert not result.is_valid
        assert result.error is not None

    def test_empty_string_is_rejected(self):
        result = validate_smiles("")
        assert not result.is_valid

    def test_whitespace_only_is_rejected(self):
        result = validate_smiles("   ")
        assert not result.is_valid

    def test_non_string_input_is_rejected(self):
        result = validate_smiles(None)  # type: ignore[arg-type]
        assert not result.is_valid

    def test_unbalanced_ring_closure_is_rejected(self):
        # Missing ring-closure digit -> RDKit should fail to parse this.
        result = validate_smiles("C1CCCC")
        assert not result.is_valid


class TestValidateDataframe:
    def test_splits_valid_and_invalid_rows(self, tiny_raw_dataframe):
        # tiny_raw_dataframe has exactly one unparsable SMILES ("not_a_smiles!!")
        valid_df, invalid_df = validate_dataframe(tiny_raw_dataframe)
        assert len(invalid_df) == 1
        assert invalid_df.iloc[0]["compound_id"] == "bad_smiles"
        assert len(valid_df) == len(tiny_raw_dataframe) - 1

    def test_valid_dataframe_has_canonical_smiles_column(self, tiny_raw_dataframe):
        valid_df, _ = validate_dataframe(tiny_raw_dataframe)
        assert "canonical_smiles" in valid_df.columns
        assert valid_df["canonical_smiles"].notna().all()
