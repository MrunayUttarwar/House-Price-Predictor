from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.features import build_pipeline


@dataclass
class CandidateModel:
    name: str
    estimator: object
    param_distributions: dict[str, list[object]]



def candidate_models(random_state: int) -> list[CandidateModel]:
    return [
        CandidateModel(
            name='linear_regression',
            estimator=LinearRegression(),
            param_distributions={},
        ),
        CandidateModel(
            name='random_forest',
            estimator=RandomForestRegressor(random_state=random_state),
            param_distributions={
                'model__n_estimators': [100, 200, 400],
                'model__max_depth': [None, 8, 12, 16],
                'model__min_samples_split': [2, 4, 8],
                'model__min_samples_leaf': [1, 2, 4],
            },
        ),
        CandidateModel(
            name='gradient_boosting',
            estimator=GradientBoostingRegressor(random_state=random_state),
            param_distributions={
                'model__n_estimators': [100, 200, 400],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [2, 3, 4],
                'model__subsample': [0.8, 1.0],
            },
        ),
    ]


def fit_candidate(
    *,
    model_name: str,
    pipeline: Pipeline,
    param_distributions: dict[str, list[object]],
    x_train,
    y_train,
    cv_folds: int,
    n_iter_search: int,
    random_state: int,
):
    if model_name == 'linear_regression' or not param_distributions:
        pipeline.fit(x_train, y_train)
        return pipeline, {}

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        scoring='neg_root_mean_squared_error',
        cv=cv_folds,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def build_candidate_pipeline(estimator: object) -> Pipeline:
    return build_pipeline(estimator)
