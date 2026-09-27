"""Evaluation metrics, model comparison, and error analysis.

Plots are saved to disk (PNG) rather than shown interactively, since this module is
meant to be called from `scripts/train_model.py` and `scripts/evaluate_model.py` in a
non-interactive environment (CI, Docker build, or a plain terminal).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display available in Docker/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard regression metrics for the LogS prediction task."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def build_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn `{model_name: metrics_dict}` into a tidy, sorted-by-RMSE comparison table."""
    table = pd.DataFrame(results).T
    table.index.name = "model"
    return table.sort_values("rmse")


def plot_model_comparison(comparison_table: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of validation RMSE and MAE per candidate model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison_table[["mae", "rmse"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Error (log10 mol/L)")
    ax.set_title("Model comparison (lower is better)")
    ax.legend(["MAE", "RMSE"])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "Predicted vs. actual"
) -> None:
    """Scatter plot of predicted vs. true LogS with a y=x reference line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18, edgecolor="none")
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Measured log(solubility) [mol/L]")
    ax.set_ylabel("Predicted log(solubility) [mol/L]")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "Residuals"
) -> None:
    """Residual plot (predicted value vs. error) to visually check for systematic bias."""
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, s=18, edgecolor="none")
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted log(solubility) [mol/L]")
    ax.set_ylabel("Residual (predicted - measured)")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def worst_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id_col: str,
    smiles_col: str,
    n: int = 15,
) -> pd.DataFrame:
    """Return the `n` rows with the largest absolute error, for qualitative error analysis."""
    out = df[[id_col, smiles_col]].copy()
    out["measured"] = y_true
    out["predicted"] = y_pred
    out["abs_error"] = np.abs(out["predicted"] - out["measured"])
    return out.sort_values("abs_error", ascending=False).head(n).reset_index(drop=True)
