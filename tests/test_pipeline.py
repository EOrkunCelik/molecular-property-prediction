from __future__ import annotations

from ml import config
from ml.pipeline import build_feature_matrix


class TestBuildFeatureMatrix:
    def test_output_shape(self, valid_smiles_dataframe):
        X, y = build_feature_matrix(valid_smiles_dataframe)
        assert X.shape == (len(valid_smiles_dataframe), len(config.DESCRIPTOR_NAMES))
        assert y.shape[0] == len(valid_smiles_dataframe)

    def test_column_order_matches_config(self, valid_smiles_dataframe):
        X, _ = build_feature_matrix(valid_smiles_dataframe)
        assert list(X.columns) == config.DESCRIPTOR_NAMES

    def test_no_target_returns_none(self, valid_smiles_dataframe):
        df_no_target = valid_smiles_dataframe.drop(columns=[config.TARGET_COL])
        X, y = build_feature_matrix(df_no_target)
        assert y is None
        assert len(X) == len(df_no_target)
