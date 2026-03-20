from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.inference import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict house price from JSON input')
    parser.add_argument('--input', required=True, help='Path to JSON file with one record or list of records')
    parser.add_argument('--model-path', default='artifacts/model.joblib', help='Path to model artifact')
    return parser.parse_args()


def load_input(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError('Input JSON must be an object or a list of objects.')
    return pd.DataFrame.from_records(records)


if __name__ == '__main__':
    args = parse_args()
    predictor = Predictor(args.model_path)
    frame = load_input(args.input)
    preds = predictor.predict(frame)
    print(json.dumps({'predictions': preds}, indent=2))
