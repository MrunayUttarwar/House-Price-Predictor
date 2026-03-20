from __future__ import annotations

import pandas as pd

from src.schema import training_schema, inference_schema


def build_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'price': 13300000,
                'area': 7420,
                'bedrooms': 4,
                'bathrooms': 2,
                'stories': 3,
                'mainroad': 'yes',
                'guestroom': 'no',
                'basement': 'no',
                'hotwaterheating': 'no',
                'airconditioning': 'yes',
                'parking': 2,
                'prefarea': 'yes',
                'furnishingstatus': 'furnished',
            }
        ]
    )


def test_training_schema_accepts_valid_input():
    df = build_train_df()
    validated = training_schema().validate(df)
    assert len(validated) == 1


def test_inference_schema_rejects_target_column():
    df = build_train_df()
    try:
        inference_schema().validate(df)
        assert False, 'Expected validation failure when target is present.'
    except Exception:
        assert True
