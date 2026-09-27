"""Candidate models for the LogS regression task.

Model choice rationale (also summarised in the README):

* DummyRegressor (mean predictor) — mandatory sanity-check baseline. Any model that
  cannot beat "always predict the training mean" is not worth shipping.
* Ridge regression — a strong, well-understood linear baseline on standardized
  descriptors. Because the descriptor set is small (13 features) and some descriptors
  are correlated (e.g. MolWt and HeavyAtomCount), L2 regularisation is preferred over
  plain OLS.
* RandomForestRegressor — captures non-linear interactions between descriptors
  (e.g. the effect of LogP on solubility depends on TPSA) without much tuning effort,
  and gives free feature-importance estimates for error analysis.
* GradientBoostingRegressor — typically the strongest of the three on small-to-medium
  tabular QSAR datasets; included because ESOL-scale datasets (~1100 molecules) are
  exactly the regime where boosted trees tend to outperform both linear models and
  plain random forests.

We deliberately stop at three real candidates plus the baseline. A deep learning model
(e.g. a graph neural network) would be scientifically interesting but is not justified
on ~1000 rows of tabular descriptor data, and adding one "to look more advanced" is
explicitly the kind of overengineering this project is trying to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml import config


@dataclass
class CandidateModel:
    """A named, fittable model plus the hyperparameter grid to search (if any)."""

    name: str
    estimator: Any
    param_grid: dict[str, list] = field(default_factory=dict)


def get_candidate_models(seed: int = config.RANDOM_SEED) -> list[CandidateModel]:
    """Return the full list of candidate models to train and compare.

    Every model is wrapped in a `Pipeline` with a `StandardScaler` first. This matters
    even for the tree-based models (which don't strictly need scaling) because it
    keeps the interface identical across all candidates, and lets Ridge use the same
    pipeline machinery without a special case.
    """
    return [
        CandidateModel(
            name="baseline_mean",
            estimator=Pipeline(
                [("scaler", StandardScaler()), ("model", DummyRegressor(strategy="mean"))]
            ),
        ),
        CandidateModel(
            name="ridge",
            estimator=Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=seed))]),
            param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        CandidateModel(
            name="random_forest",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(random_state=seed, n_jobs=-1)),
                ]
            ),
            param_grid={
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        CandidateModel(
            name="gradient_boosting",
            estimator=Pipeline(
                [("scaler", StandardScaler()), ("model", GradientBoostingRegressor(random_state=seed))]
            ),
            param_grid={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.01, 0.05, 0.1],
            },
        ),
    ]


def fit_candidate(
    candidate: CandidateModel,
    X_train,
    y_train,
    groups=None,
    cv_folds: int = 5,
    seed: int = config.RANDOM_SEED,
    scoring: str = "neg_root_mean_squared_error",
) -> Any:
    """Fit a candidate model, using grouped CV when groups are provided.
    If ``groups`` is provided, GroupKFold keeps samples from the same group
    together during hyperparameter tuning. Otherwise, ordinary shuffled KFold
    is used.
    Returns the fitted estimator (either the plain pipeline, or the
    ``best_estimator_`` of the grid search) ready to call ``.predict()`` on.
    """
    if not candidate.param_grid:
        candidate.estimator.fit(X_train, y_train)
        return candidate.estimator
    if groups is not None:
        n_unique_groups = len(set(groups))
        if n_unique_groups < 2:
            raise ValueError(
                "Grouped cross-validation requires at least 2 unique groups."
            )
        effective_cv_folds = min(cv_folds, n_unique_groups)
        cv = GroupKFold(n_splits=effective_cv_folds)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        candidate.estimator,
        param_grid=candidate.param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    if groups is not None:
        search.fit(X_train, y_train, groups=groups)
    else:
        search.fit(X_train, y_train)
    return search.best_estimator_
