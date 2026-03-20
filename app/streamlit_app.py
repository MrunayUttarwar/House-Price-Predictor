from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path('artifacts')
MODEL_PATH = ARTIFACT_DIR / 'model.joblib'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


def load_project_modules() -> tuple[list[str], Any, Any]:
    try:
        from src.explain import explain_single_prediction
        from src.schema import CATEGORICAL_FEATURES, inference_schema
    except ModuleNotFoundError:
        root_dir = Path(__file__).resolve().parents[1]
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        from src.explain import explain_single_prediction
        from src.schema import CATEGORICAL_FEATURES, inference_schema
    return CATEGORICAL_FEATURES, inference_schema, explain_single_prediction


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding='utf-8'))


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(11,114,133,0.14), transparent 35%),
                radial-gradient(circle at 100% 100%, rgba(255,146,43,0.14), transparent 35%),
                #f8f9fa;
        }
        .hero {
            border-radius: 16px;
            padding: 1rem 1.2rem;
            background: linear-gradient(100deg, #0b7285, #087f5b);
            color: #ffffff;
            margin-bottom: 0.7rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 1.7rem;
        }
        .hero p {
            margin: 0.35rem 0 0;
            opacity: 0.95;
        }
        .metric-card {
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            background: #ffffff;
            border: 1px solid #e9ecef;
            box-shadow: 0 6px 24px rgba(33, 37, 41, 0.06);
        }
        .metric-label {
            font-size: 0.82rem;
            color: #495057;
            margin: 0;
        }
        .metric-value {
            margin: 0.2rem 0 0;
            font-size: 1.18rem;
            font-weight: 700;
            color: #212529;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_input_form(categorical_features: list[str]) -> dict[str, object]:
    inputs: dict[str, object] = {}
    left, right = st.columns(2)

    with left:
        st.markdown("#### Property Size")
        inputs['area'] = st.number_input('Area (sq ft)', min_value=100, step=10, value=1500)
        inputs['stories'] = st.number_input('Stories', min_value=0, step=1, value=2)
        inputs['parking'] = st.number_input('Parking', min_value=0, step=1, value=1)
        st.markdown("#### Interior")
        inputs['bedrooms'] = st.number_input('Bedrooms', min_value=0, step=1, value=3)
        inputs['bathrooms'] = st.number_input('Bathrooms', min_value=0, step=1, value=2)

    with right:
        st.markdown("#### Amenities")
        for col in categorical_features:
            if col == 'furnishingstatus':
                inputs[col] = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])
            else:
                label = col.replace('hotwaterheating', 'hot water heating').replace('airconditioning', 'air conditioning')
                inputs[col] = st.selectbox(label.title(), ['yes', 'no'])

    return inputs


def main() -> None:
    st.set_page_config(page_title='House Price ML App', page_icon=':house:', layout='wide')
    inject_styles()
    st.markdown(
        """
        <div class="hero">
            <h1>House Price Prediction Studio</h1>
            <p>Production-style ML inference with schema validation, model diagnostics, and explainability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error('Model artifact not found. Run: python -m src.train --config configs/train.yaml')
        return

    model = load_model()
    metrics = load_metrics()
    categorical_features, inference_schema, explain_single_prediction = load_project_modules()
    metric_values = metrics.get('metrics', {}) if metrics else {}
    best_model = metrics.get('best_model', 'unknown') if metrics else 'unknown'

    st.sidebar.header('Model Diagnostics')
    st.sidebar.write(f"Best model: `{best_model}`")
    st.sidebar.write(f"RMSE: `{metric_values.get('rmse', 'n/a')}`")
    st.sidebar.write(f"MAE: `{metric_values.get('mae', 'n/a')}`")
    st.sidebar.write(f"R2: `{metric_values.get('r2', 'n/a')}`")
    if METRICS_PATH.exists():
        st.sidebar.download_button(
            label='Download metrics.json',
            data=METRICS_PATH.read_bytes(),
            file_name='metrics.json',
            mime='application/json',
        )
    for plot_name in ['predicted_vs_actual', 'residuals', 'feature_importance', 'shap_summary']:
        plot_path = metrics.get('plots', {}).get(plot_name, '') if metrics else ''
        if plot_path and Path(plot_path).exists():
            st.sidebar.image(plot_path, caption=plot_name.replace('_', ' ').title(), use_column_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card('Best Model', str(best_model))
    with c2:
        metric_card('RMSE', f"{metric_values.get('rmse', 'n/a')}")
    with c3:
        metric_card('R2', f"{metric_values.get('r2', 'n/a')}")

    st.markdown("### Enter House Features")
    user_inputs = build_input_form(categorical_features)

    if st.button('Predict Price', type='primary', use_container_width=True):
        try:
            input_df = pd.DataFrame([user_inputs])
            validated = inference_schema().validate(input_df)
            prediction = float(model.predict(validated)[0])
            st.success(f'Estimated Price: ${prediction:,.2f}')

            residual_std = metrics.get('metrics', {}).get('residual_std') if metrics else None
            if residual_std is not None:
                lower = prediction - 1.96 * residual_std
                upper = prediction + 1.96 * residual_std
                st.info(f'Approximate 95% expected range: ${lower:,.2f} to ${upper:,.2f}')

            contributions = explain_single_prediction(model, validated)
            if contributions:
                st.subheader('Top Drivers For This Prediction')
                contribution_df = pd.DataFrame(
                    {
                        'feature': list(contributions.keys()),
                        'contribution': list(contributions.values()),
                    }
                ).set_index('feature')
                st.bar_chart(contribution_df, horizontal=True)
                for feature, value in contributions.items():
                    direction = 'increased' if value >= 0 else 'decreased'
                    st.write(f'- `{feature}` {direction} the estimate by `{abs(value):,.2f}` model units.')
        except Exception as exc:
            st.error(f'Input validation/prediction failed: {exc}')


if __name__ == '__main__':
    main()
