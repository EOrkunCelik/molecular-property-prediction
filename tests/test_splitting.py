from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ml.data.splitting import assert_no_leakage, random_split, scaffold_split


class TestScaffoldSplit:
    def test_split_sizes_cover_the_whole_dataset(self, valid_smiles_dataframe):
        train_df, val_df, test_df = scaffold_split(
            valid_smiles_dataframe, test_size=0.25, val_size=0.25, seed=0
        )
        assert len(train_df) + len(val_df) + len(test_df) == len(valid_smiles_dataframe)

    def test_no_leakage_between_splits(self, valid_smiles_dataframe):
        train_df, val_df, test_df = scaffold_split(
            valid_smiles_dataframe, test_size=0.25, val_size=0.25, seed=0
        )
        assert_no_leakage(train_df, val_df, test_df)  # should not raise

    def test_no_scaffold_overlap_between_splits(self, valid_smiles_dataframe):
        train_df, val_df, test_df = scaffold_split(
            valid_smiles_dataframe, test_size=0.25, val_size=0.25, seed=0
        )

        def scaffolds(df):
            return {
                MurckoScaffold.MurckoScaffoldSmiles(
                    mol=Chem.MolFromSmiles(smiles),
                    includeChirality=False,
                )
                or smiles
                for smiles in df["canonical_smiles"]
            }
        train_scaffolds = scaffolds(train_df)
        val_scaffolds = scaffolds(val_df)
        test_scaffolds = scaffolds(test_df)
        assert train_scaffolds.isdisjoint(val_scaffolds)
        assert train_scaffolds.isdisjoint(test_scaffolds)
        assert val_scaffolds.isdisjoint(test_scaffolds)

    def test_deterministic_given_same_seed(self, valid_smiles_dataframe):
        split_a = scaffold_split(valid_smiles_dataframe, test_size=0.25, val_size=0.25, seed=7)
        split_b = scaffold_split(valid_smiles_dataframe, test_size=0.25, val_size=0.25, seed=7)
        for df_a, df_b in zip(split_a, split_b, strict=True):
            pd.testing.assert_frame_equal(df_a, df_b)


class TestRandomSplit:
    def test_split_sizes_cover_the_whole_dataset(self, valid_smiles_dataframe):
        train_df, val_df, test_df = random_split(valid_smiles_dataframe, test_size=0.25, val_size=0.25, seed=0)
        assert len(train_df) + len(val_df) + len(test_df) == len(valid_smiles_dataframe)

    def test_no_leakage_between_splits(self, valid_smiles_dataframe):
        train_df, val_df, test_df = random_split(valid_smiles_dataframe, test_size=0.25, val_size=0.25, seed=0)
        assert_no_leakage(train_df, val_df, test_df)


class TestAssertNoLeakage:
    def test_raises_when_same_molecule_in_two_splits(self):
        train_df = pd.DataFrame({"canonical_smiles": ["CCO", "CCC"]})
        val_df = pd.DataFrame({"canonical_smiles": ["CCO"]})  # leaked from train
        test_df = pd.DataFrame({"canonical_smiles": ["c1ccccc1"]})
        with pytest.raises(AssertionError):
            assert_no_leakage(train_df, val_df, test_df)
