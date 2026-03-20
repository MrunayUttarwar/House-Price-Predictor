from __future__ import annotations

from sklearn.linear_model import LinearRegression

from src.modeling import build_candidate_pipeline
from src.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def test_preprocessing_pipeline_builds():
    model = LinearRegression()
    pipeline = build_candidate_pipeline(model)
    preprocessor = pipeline.named_steps['preprocessor']
    names = [name for name, _, _ in preprocessor.transformers]

    assert 'num' in names
    assert 'cat' in names
    assert len(NUMERIC_FEATURES) == 5
    assert len(CATEGORICAL_FEATURES) == 7
