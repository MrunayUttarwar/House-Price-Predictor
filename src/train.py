from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import load_config
from src.data import load_dataset, split_data
from src.evaluate import regression_metrics, save_metrics, save_regression_plots
from src.explain import save_global_explanations
from src.modeling import build_candidate_pipeline, candidate_models, fit_candidate


def _start_mlflow(uri: str, experiment: str):
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        return mlflow
    except Exception:
        return None


def write_model_card(
    path: Path,
    *,
    best_model_name: str,
    metrics: dict[str, float],
    feature_plots: dict[str, str],
) -> None:
    card = f"""# Model Card: House Price Prediction

## Model
- Best model: `{best_model_name}`
- Task: Regression (predict house price)

## Metrics (test split)
- RMSE: {metrics['rmse']:.2f}
- MAE: {metrics['mae']:.2f}
- R2: {metrics['r2']:.4f}
- Residual Std: {metrics['residual_std']:.2f}

## Artifacts
- Predicted vs Actual plot: `{feature_plots.get('predicted_vs_actual', '')}`
- Residual plot: `{feature_plots.get('residuals', '')}`
- Feature importance: `{feature_plots.get('feature_importance', '')}`
- SHAP summary: `{feature_plots.get('shap_summary', '')}`
"""
    path.write_text(card, encoding='utf-8')


def run_training(config_path: str) -> dict[str, object]:
    cfg = load_config(config_path)
    artifact_dir = Path(cfg.artifact_dir)
    plots_dir = artifact_dir / 'plots'
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.data_path)
    x_train, x_test, y_train, y_test = split_data(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    mlflow = _start_mlflow(cfg.mlflow_tracking_uri, cfg.mlflow_experiment)

    leaderboard: list[dict[str, object]] = []
    best_rmse = float('inf')
    best_name = ''
    best_pipeline = None

    for candidate in candidate_models(cfg.random_state):
        pipeline = build_candidate_pipeline(candidate.estimator)
        fitted, best_params = fit_candidate(
            model_name=candidate.name,
            pipeline=pipeline,
            param_distributions=candidate.param_distributions,
            x_train=x_train,
            y_train=y_train,
            cv_folds=cfg.cv_folds,
            n_iter_search=cfg.n_iter_search,
            random_state=cfg.random_state,
        )
        y_pred = fitted.predict(x_test)
        m = regression_metrics(y_test, y_pred)

        record = {
            'model': candidate.name,
            'params': best_params,
            'metrics': m,
        }
        leaderboard.append(record)

        if mlflow is not None:
            with mlflow.start_run(run_name=candidate.name):
                mlflow.log_params(best_params)
                mlflow.log_metric('rmse', m['rmse'])
                mlflow.log_metric('mae', m['mae'])
                mlflow.log_metric('r2', m['r2'])

        if m['rmse'] < best_rmse:
            best_rmse = m['rmse']
            best_name = candidate.name
            best_pipeline = fitted

    if best_pipeline is None:
        raise RuntimeError('No model trained successfully.')

    y_pred_best = best_pipeline.predict(x_test)
    final_metrics = regression_metrics(y_test, y_pred_best)
    plots = save_regression_plots(y_test, y_pred_best, plots_dir)
    explain_plots = save_global_explanations(best_pipeline, x_train, plots_dir)
    plots.update(explain_plots)

    model_path = artifact_dir / 'model.joblib'
    preprocessor_path = artifact_dir / 'preprocessor.joblib'
    metrics_path = artifact_dir / 'metrics.json'
    leaderboard_path = artifact_dir / 'leaderboard.json'
    card_path = artifact_dir / 'model_card.md'

    joblib.dump(best_pipeline, model_path)
    joblib.dump(best_pipeline.named_steps['preprocessor'], preprocessor_path)

    metrics_payload = {
        'best_model': best_name,
        'metrics': final_metrics,
        'prediction_interval_approx_95pct': {
            'plus_minus': float(1.96 * final_metrics['residual_std'])
        },
        'plots': plots,
    }
    save_metrics(metrics_payload, metrics_path)
    leaderboard_path.write_text(json.dumps(leaderboard, indent=2), encoding='utf-8')
    write_model_card(card_path, best_model_name=best_name, metrics=final_metrics, feature_plots=plots)

    return {
        'model_path': str(model_path),
        'preprocessor_path': str(preprocessor_path),
        'metrics_path': str(metrics_path),
        'leaderboard_path': str(leaderboard_path),
        'model_card_path': str(card_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train house price model.')
    parser.add_argument('--config', default='configs/train.yaml', help='Path to YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    outputs = run_training(args.config)
    print(json.dumps(outputs, indent=2))
