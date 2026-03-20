from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.modeling import build_candidate_pipeline
from src.inference import Predictor


def _train_sample_model(tmp_path: Path) -> Path:
    train = pd.DataFrame(
        [
            {
                'area': 7420,
                'bedrooms': 4,
                'bathrooms': 2,
                'stories': 3,
                'parking': 2,
                'mainroad': 'yes',
                'guestroom': 'no',
                'basement': 'no',
                'hotwaterheating': 'no',
                'airconditioning': 'yes',
                'prefarea': 'yes',
                'furnishingstatus': 'furnished',
                'price': 13300000,
            },
            {
                'area': 8960,
                'bedrooms': 4,
                'bathrooms': 4,
                'stories': 4,
                'parking': 3,
                'mainroad': 'yes',
                'guestroom': 'no',
                'basement': 'no',
                'hotwaterheating': 'no',
                'airconditioning': 'yes',
                'prefarea': 'no',
                'furnishingstatus': 'furnished',
                'price': 12250000,
            },
        ]
    )
    x = train.drop(columns=['price'])
    y = train['price']

    pipe = build_candidate_pipeline(LinearRegression())
    pipe.fit(x, y)

    model_path = tmp_path / 'model.joblib'
    joblib.dump(pipe, model_path)
    return model_path


def test_predictor_runs_end_to_end(tmp_path: Path):
    model_path = _train_sample_model(tmp_path)
    predictor = Predictor(model_path)

    sample = pd.DataFrame(
        [
            {
                'area': 7500,
                'bedrooms': 4,
                'bathrooms': 2,
                'stories': 2,
                'parking': 3,
                'mainroad': 'yes',
                'guestroom': 'no',
                'basement': 'yes',
                'hotwaterheating': 'no',
                'airconditioning': 'yes',
                'prefarea': 'yes',
                'furnishingstatus': 'furnished',
            }
        ]
    )

    preds = predictor.predict(sample)
    assert len(preds) == 1
    assert preds[0] > 0
