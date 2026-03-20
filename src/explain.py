from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_feature_names(pipeline, fallback_columns: list[str]) -> list[str]:
    try:
        names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return fallback_columns


def save_global_explanations(pipeline, x_train: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    transformed = preprocessor.transform(x_train)
    feature_names = _safe_feature_names(pipeline, list(x_train.columns))

    # Feature-importance fallback that always works for tree models and linear coefficients.
    importance_path = out / 'feature_importance.png'
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef.ravel()) if coef.ndim > 1 else np.abs(coef)

    if importances is not None:
        top_n = min(12, len(importances))
        idx = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.asarray(feature_names)[idx], importances[idx])
        ax.set_title('Top Feature Importances')
        fig.tight_layout()
        fig.savefig(importance_path, dpi=150)
        plt.close(fig)

    shap_summary_path = out / 'shap_summary.png'
    try:
        import shap

        sample = transformed[: min(200, transformed.shape[0])]
        if hasattr(sample, 'toarray'):
            sample = sample.toarray()

        explainer = shap.Explainer(model.predict, sample)
        shap_values = explainer(sample)
        shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        shap_output = str(shap_summary_path)
    except Exception:
        shap_output = ''

    return {
        'feature_importance': str(importance_path) if importance_path.exists() else '',
        'shap_summary': shap_output,
    }


def explain_single_prediction(pipeline, input_df: pd.DataFrame) -> dict[str, float]:
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    features = _safe_feature_names(pipeline, list(input_df.columns))
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()

    values = transformed[0]
    if hasattr(model, 'feature_importances_'):
        weights = np.asarray(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        coef = np.asarray(model.coef_)
        weights = coef.ravel() if coef.ndim > 1 else coef
    else:
        return {}

    contributions = values * weights
    top_idx = np.argsort(np.abs(contributions))[-5:][::-1]
    return {str(features[i]): float(contributions[i]) for i in top_idx}
