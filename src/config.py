from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    data_path: str
    target_column: str
    random_state: int
    test_size: float
    cv_folds: int
    n_iter_search: int
    artifact_dir: str
    mlflow_tracking_uri: str
    mlflow_experiment: str


DEFAULT_CONFIG_PATH = Path('configs/train.yaml')


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> TrainConfig:
    config_path = Path(path)
    with config_path.open('r', encoding='utf-8') as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return TrainConfig(
        data_path=raw['data_path'],
        target_column=raw['target_column'],
        random_state=int(raw['random_state']),
        test_size=float(raw['test_size']),
        cv_folds=int(raw['cv_folds']),
        n_iter_search=int(raw['n_iter_search']),
        artifact_dir=raw['artifact_dir'],
        mlflow_tracking_uri=raw['mlflow_tracking_uri'],
        mlflow_experiment=raw['mlflow_experiment'],
    )
