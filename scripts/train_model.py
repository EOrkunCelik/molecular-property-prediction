#!/usr/bin/env python3
"""Train and evaluate the solubility prediction model end-to-end.

This is the single command that reproduces the model shipped with the backend:

    python scripts/train_model.py

What it does, in order:
  1. Load the raw ESOL dataset and clean it (missing values, exact duplicates).
  2. Validate every SMILES string with RDKit; drop unparsable molecules.
  3. Split into train/val/test using a Bemis-Murcko scaffold split (default) or a
     random split (`--split random`), and assert there is no molecule leakage
     between splits.
  4. Compute the RDKit 2D descriptor feature matrix for each split.
  5. Train four candidate models (mean baseline, Ridge, Random Forest, Gradient
     Boosting), tuning hyperparameters via grid search + cross-validation where a
     grid is defined.
  6. Evaluate every candidate on the held-out validation set and pick the best one
     by RMSE.
  7. Refit the winning model on train+val, evaluate once on the untouched test set
     (the number reported in the README), and run basic error analysis.
  8. Save the fitted model, its metadata, its metrics, the processed data splits,
     and comparison/diagnostic plots to disk.

Run with `--quick` for a fast smoke-test pass with tiny hyperparameter grids (useful
for CI or for verifying the pipeline runs before committing to a full training run).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml import config  # noqa: E402
from ml.data.loader import load_and_clean  # noqa: E402
from ml.data.preprocessing import aggregate_canonical_duplicates
from ml.data.splitting import (
    _generate_scaffold,
    assert_no_leakage,
    random_split,
    scaffold_split,
)  # noqa: E402
from ml.data.validation import validate_dataframe  # noqa: E402
from ml.models.evaluate import (  # noqa: E402
    build_comparison_table,
    compute_metrics,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
    worst_predictions,
)
from ml.models.registry import ModelMetadata, save_model_bundle  # noqa: E402
from ml.models.train import fit_candidate, get_candidate_models  # noqa: E402
from ml.pipeline import build_feature_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_model")

PLOTS_DIR = config.PROJECT_ROOT / "docs" / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split",
        choices=["scaffold", "random"],
        default="scaffold" if config.SCAFFOLD_SPLIT else "random",
        help="Splitting strategy for train/val/test (default: scaffold).",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny hyperparameter grids for a fast smoke test instead of the full search.",
    )
    return parser.parse_args()


def _maybe_shrink_grids(models, quick: bool):
    if not quick:
        return models
    for candidate in models:
        for key, values in list(candidate.param_grid.items()):
            candidate.param_grid[key] = values[:1]
    return models


def main() -> None:
    args = parse_args()
    logger.info("=== Loading and cleaning raw dataset ===")
    df = load_and_clean()

    logger.info("=== Validating SMILES with RDKit ===")
    valid_df, invalid_df = validate_dataframe(df)
    valid_df = aggregate_canonical_duplicates(valid_df)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    invalid_df.to_csv(config.PROCESSED_DATA_DIR / "invalid_smiles.csv", index=False)

    logger.info("=== Splitting into train/val/test (%s split) ===", args.split)
    split_fn = scaffold_split if args.split == "scaffold" else random_split
    train_df, val_df, test_df = split_fn(valid_df, seed=args.seed)
    assert_no_leakage(train_df, val_df, test_df)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.to_csv(config.PROCESSED_DATA_DIR / f"{name}.csv", index=False)

    logger.info("=== Computing descriptor feature matrices ===")
    X_train, y_train = build_feature_matrix(train_df)
    X_val, y_val = build_feature_matrix(val_df)
    X_test, y_test = build_feature_matrix(test_df)

    train_groups = train_df[config.CANONICAL_SMILES_COL].map(_generate_scaffold)

    logger.info(
        "Feature matrix shapes: train=%s, val=%s, test=%s", X_train.shape, X_val.shape, X_test.shape
    )

    logger.info("=== Training candidate models ===")
    candidates = _maybe_shrink_grids(get_candidate_models(seed=args.seed), args.quick)
    fitted = {}
    val_results = {}
    for candidate in candidates:
        logger.info("Fitting candidate: %s", candidate.name)
        model = fit_candidate(candidate, X_train, y_train, groups=train_groups, seed=args.seed)
        fitted[candidate.name] = model
        val_pred = model.predict(X_val)
        val_results[candidate.name] = compute_metrics(y_val, val_pred)
        logger.info("  val metrics: %s", val_results[candidate.name])

    comparison_table = build_comparison_table(val_results)
    logger.info("=== Validation set model comparison ===\n%s", comparison_table)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_table.to_csv(config.PROCESSED_DATA_DIR / "model_comparison_val.csv")
    plot_model_comparison(comparison_table, PLOTS_DIR / "model_comparison.png")

    best_name = comparison_table.index[0]
    logger.info("=== Best model on validation set: %s ===", best_name)

    # Refit the winning model's *pipeline definition* on train+val combined, so the
    # final model benefits from every labelled example not held out for testing.
    import pandas as pd

    X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    best_candidate = next(c for c in candidates if c.name == best_name)
    final_model = fit_candidate(best_candidate, X_trainval, y_trainval, seed=args.seed)

    logger.info("=== Final evaluation on untouched test set ===")
    test_pred = final_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_pred)
    logger.info("Test metrics: %s", test_metrics)

    plot_predicted_vs_actual(
        y_test.values, test_pred, PLOTS_DIR / "predicted_vs_actual_test.png",
        title=f"Predicted vs. actual (test set) — {best_name}",
    )
    plot_residuals(
        y_test.values, test_pred, PLOTS_DIR / "residuals_test.png",
        title=f"Residuals (test set) — {best_name}",
    )
    worst = worst_predictions(
        test_df, y_test.values, test_pred, id_col=config.ID_COL, smiles_col=config.CANONICAL_SMILES_COL
    )
    worst.to_csv(config.PROCESSED_DATA_DIR / "worst_predictions_test.csv", index=False)
    logger.info("=== Worst 5 predictions on test set ===\n%s", worst.head(5))

    metadata = ModelMetadata(
        model_name=best_name,
        feature_names=list(X_train.columns),
        target_col=config.TARGET_COL,
        split_strategy=args.split,
        extra={
            "val_comparison": val_results,
            "test_metrics": test_metrics,
            "selected_hyperparameters": final_model.named_steps["model"].get_params(),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_invalid_smiles_dropped": len(invalid_df),
            "n_canonical_molecules": len(train_df) + len(val_df) + len(test_df),
            "seed": args.seed,
        },
    )
    save_model_bundle(final_model, metadata, test_metrics)
    logger.info("Saved model bundle to %s", config.MODELS_DIR)

    summary = {
        "best_model": best_name,
        "test_metrics": test_metrics,
        "val_comparison": val_results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
