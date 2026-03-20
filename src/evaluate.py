from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    residual_std = float(np.std(np.asarray(y_true) - np.asarray(y_pred)))
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'residual_std': residual_std,
    }


def save_metrics(metrics: dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def save_regression_plots(y_true, y_pred, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(y_true, y_pred, alpha=0.7)
    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax1.plot([min_v, max_v], [min_v, max_v], linestyle='--')
    ax1.set_title('Predicted vs Actual')
    ax1.set_xlabel('Actual Price')
    ax1.set_ylabel('Predicted Price')
    pred_plot = out / 'predicted_vs_actual.png'
    fig1.tight_layout()
    fig1.savefig(pred_plot, dpi=150)
    plt.close(fig1)

    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(residuals, bins=20)
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Count')
    resid_plot = out / 'residuals.png'
    fig2.tight_layout()
    fig2.savefig(resid_plot, dpi=150)
    plt.close(fig2)

    return {
        'predicted_vs_actual': str(pred_plot),
        'residuals': str(resid_plot),
    }
