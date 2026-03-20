from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.schema import inference_schema


class Predictor:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)

    def predict(self, data: pd.DataFrame) -> list[float]:
        clean = inference_schema().validate(data)
        preds = self.pipeline.predict(clean)
        return [float(p) for p in preds]
