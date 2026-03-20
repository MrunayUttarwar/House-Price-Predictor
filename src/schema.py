from __future__ import annotations

from typing import Final

try:
    import pandera.pandas as pa
except ModuleNotFoundError:
    import pandera as pa

NUMERIC_FEATURES: Final[list[str]] = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
CATEGORICAL_FEATURES: Final[list[str]] = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus',
]
TARGET_COLUMN: Final[str] = 'price'

ALLOWED_BINARY: Final[set[str]] = {'yes', 'no'}
ALLOWED_FURNISHING: Final[set[str]] = {'furnished', 'semi-furnished', 'unfurnished'}


def training_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            'price': pa.Column(float, coerce=True, nullable=False),
            'area': pa.Column(float, coerce=True, nullable=False, checks=pa.Check.ge(1)),
            'bedrooms': pa.Column(int, coerce=True, nullable=False, checks=pa.Check.ge(0)),
            'bathrooms': pa.Column(int, coerce=True, nullable=False, checks=pa.Check.ge(0)),
            'stories': pa.Column(int, coerce=True, nullable=False, checks=pa.Check.ge(0)),
            'mainroad': pa.Column(str, nullable=False, checks=pa.Check.isin(ALLOWED_BINARY)),
            'guestroom': pa.Column(str, nullable=False, checks=pa.Check.isin(ALLOWED_BINARY)),
            'basement': pa.Column(str, nullable=False, checks=pa.Check.isin(ALLOWED_BINARY)),
            'hotwaterheating': pa.Column(str, nullable=False, checks=pa.Check.isin(ALLOWED_BINARY)),
            'airconditioning': pa.Column(str, nullable=False, checks=pa.Check.isin(ALLOWED_BINARY)),
            'parking': pa.Column(int, coerce=True, nullable=False, checks=pa.Check.ge(0)),
            'prefarea': pa.Column(str, nullable=False, checks=pa.Check.isin(ALLOWED_BINARY)),
            'furnishingstatus': pa.Column(str, nullable=False, checks=pa.Check.isin(ALLOWED_FURNISHING)),
        },
        strict=True,
    )


def inference_schema() -> pa.DataFrameSchema:
    fields = {k: v for k, v in training_schema().columns.items() if k != TARGET_COLUMN}
    return pa.DataFrameSchema(fields, strict=True)
