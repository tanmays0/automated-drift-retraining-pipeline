"""Prefect flow: monitor drift → retrain → register → email alert."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, task

from src.alerts import alert_from_pipeline
from src.data_generator import save_datasets
from src.drift_monitor import detect_drift, drift_result_to_dict
from src.train import train_and_register, train_result_to_dict


@task(name="generate-synthetic-data")
def generate_data_task(data_dir: str = "data") -> dict[str, str]:
    reference, current = save_datasets(data_dir=data_dir)
    return {
        "reference_path": str(Path(data_dir) / "reference.csv"),
        "current_path": str(Path(data_dir) / "current_drifted.csv"),
        "n_reference": str(len(reference)),
        "n_current": str(len(current)),
    }


@task(name="detect-data-drift")
def detect_drift_task(
    reference_path: str,
    current_path: str,
    threshold: float = 0.1,
) -> dict[str, Any]:
    import pandas as pd

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    result = detect_drift(reference, current, threshold=threshold)
    return drift_result_to_dict(result)


@task(name="retrain-and-register")
def retrain_task(current_path: str) -> dict[str, Any]:
    import pandas as pd

    current = pd.read_csv(current_path)
    result = train_and_register(current)
    return train_result_to_dict(result)


@task(name="send-email-alert")
def alert_task(
    drift: dict[str, Any],
    train: dict[str, Any] | None = None,
) -> str:
    path = alert_from_pipeline(drift=drift, train=train)
    return str(path)


@flow(name="drift-triggered-retraining", log_prints=True)
def drift_retrain_flow(threshold: float = 0.1) -> dict[str, Any]:
    """
    Automated re-training pipeline triggered by data drift.

    1. Generate reference + drifted demand data
    2. Monitor with Evidently; gate on Wasserstein > threshold
    3. If drifted: retrain sklearn model and register in MLflow
    4. Write email alert notification file
    """
    paths = generate_data_task()
    print(f"Generated reference={paths['n_reference']} current={paths['n_current']}")

    drift = detect_drift_task(
        reference_path=paths["reference_path"],
        current_path=paths["current_path"],
        threshold=threshold,
    )
    print(
        f"Drifted={drift['drifted']} max_wasserstein={drift['max_wasserstein']:.4f} "
        f"(threshold={threshold})"
    )
    print(f"Evidently report: {drift['report_path']}")

    train_info: dict[str, Any] | None = None
    if drift["drifted"]:
        print("Threshold exceeded — triggering retrain DAG tasks...")
        train_info = retrain_task(current_path=paths["current_path"])
        print(
            f"Registered {train_info['model_name']} "
            f"v{train_info['registered_version']} run={train_info['run_id']}"
        )
    else:
        print("No retrain needed.")

    alert_path = alert_task(drift=drift, train=train_info)
    print(f"Email alert written: {alert_path}")

    return {
        "drift": drift,
        "train": train_info,
        "alert_path": alert_path,
    }


if __name__ == "__main__":
    drift_retrain_flow()
