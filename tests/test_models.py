import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.models.train import CandidateModel, fit_candidate


def _candidate() -> CandidateModel:
    return CandidateModel(
        name="ridge_test",
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        param_grid={"model__alpha": [0.1, 1.0]},
    )

def test_fit_candidate_with_groups():
    X = pd.DataFrame(
        {
            "feature_1": np.arange(20, dtype=float),
            "feature_2": np.arange(20, dtype=float) ** 2,
        }
    )
    y = pd.Series(np.arange(20, dtype=float))
    groups = pd.Series(np.repeat(np.arange(5), 4))
    model = fit_candidate(
        _candidate(),
        X,
        y,
        groups=groups,
        cv_folds=5,
    )
    predictions = model.predict(X)
    assert len(predictions) == len(X)

def test_fit_candidate_without_groups():
    X = pd.DataFrame(
        {
            "feature_1": np.arange(20, dtype=float),
            "feature_2": np.arange(20, dtype=float) ** 2,
        }
    )
    y = pd.Series(np.arange(20, dtype=float))
    model = fit_candidate(
        _candidate(),
        X,
        y,
        groups=None,
        cv_folds=5,
    )
    predictions = model.predict(X)
    assert len(predictions) == len(X)
def test_fit_candidate_reduces_cv_folds_for_few_groups():
    X = pd.DataFrame(
        {
            "feature_1": np.arange(12, dtype=float),
            "feature_2": np.arange(12, dtype=float) ** 2,
        }
    )
    y = pd.Series(np.arange(12, dtype=float))
    groups = pd.Series(np.repeat(np.arange(3), 4))
    model = fit_candidate(
        _candidate(),
        X,
        y,
        groups=groups,
        cv_folds=5,
    )
    predictions = model.predict(X)
    assert len(predictions) == len(X)
def test_fit_candidate_rejects_single_group():
    X = pd.DataFrame(
        {
            "feature_1": np.arange(10, dtype=float),
            "feature_2": np.arange(10, dtype=float) ** 2,
        }
    )
    y = pd.Series(np.arange(10, dtype=float))
    groups = pd.Series(np.zeros(10, dtype=int))
    with pytest.raises(
        ValueError,
        match="Grouped cross-validation requires at least 2 unique groups",
    ):
        fit_candidate(
            _candidate(),
            X,
            y,
            groups=groups,
            cv_folds=5,
        )
