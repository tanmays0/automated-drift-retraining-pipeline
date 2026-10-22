"""Train demand model and register it in MLflow Model Registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_generator import FEATURE_COLS, TARGET_COL

MODEL_NAME = "demand-forecast-model"
EXPERIMENT_NAME = "demand-forecasting-drift"


@dataclass
class TrainResult:
    run_id: str
    model_uri: str
    registered_version: str
    metrics: dict[str, float]
    model_name: str = MODEL_NAME


def train_and_register(
    data: pd.DataFrame,
    tracking_uri: str | Path | None = None,
    model_name: str = MODEL_NAME,
) -> TrainResult:
    # SQLite backend (MLflow 3.x deprecated plain ./mlruns file store)
    if tracking_uri is None:
        db_path = Path("mlflow.db").resolve()
        tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(EXPERIMENT_NAME)

    x = data[FEATURE_COLS]
    y = data[TARGET_COL]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )

    with mlflow.start_run(run_name="drift-triggered-retrain") as run:
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
        }
        mlflow.log_params(
            {
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 100,
                "max_depth": 3,
                "trigger": "data_drift",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="pickle",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)

        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=mv.version,
        )

        return TrainResult(
            run_id=run.info.run_id,
            model_uri=model_uri,
            registered_version=str(mv.version),
            metrics=metrics,
            model_name=model_name,
        )


def train_result_to_dict(result: TrainResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "model_uri": result.model_uri,
        "registered_version": result.registered_version,
        "metrics": result.metrics,
        "model_name": result.model_name,
    }
