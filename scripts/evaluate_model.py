#!/usr/bin/env python3
"""Re-evaluate the currently saved model against the persisted test split.

Useful for CI (verify the checked-in model still meets a minimum bar) or for a quick
sanity check without paying the cost of a full retrain:

    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --min-r2 0.6   # exit non-zero if R^2 falls below this
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ml import config  # noqa: E402
from ml.models.evaluate import compute_metrics  # noqa: E402
from ml.models.registry import load_model_bundle  # noqa: E402
from ml.pipeline import build_feature_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-r2", type=float, default=None, help="Fail if test R^2 is below this value.")
    args = parser.parse_args()

    test_path = config.PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        print(
            f"No persisted test split found at {test_path}. Run scripts/train_model.py first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    test_df = pd.read_csv(test_path)
    X_test, y_test = build_feature_matrix(test_df)

    bundle = load_model_bundle()
    predictions = bundle.model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)

    print(json.dumps({"model_name": bundle.metadata.get("model_name"), "metrics": metrics}, indent=2))

    if args.min_r2 is not None and metrics["r2"] < args.min_r2:
        print(f"FAIL: test R^2 {metrics['r2']:.3f} is below the required minimum {args.min_r2}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
