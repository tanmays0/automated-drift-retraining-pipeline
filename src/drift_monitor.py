"""Data/concept drift detection with Evidently AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import wasserstein_distance

from src.data_generator import FEATURE_COLS, TARGET_COL

DRIFT_THRESHOLD = 0.1


@dataclass
class DriftResult:
    drifted: bool
    max_wasserstein: float
    per_column: dict[str, float]
    report_path: str
    threshold: float = DRIFT_THRESHOLD


def _column_wasserstein(reference: pd.Series, current: pd.Series) -> float:
    return float(wasserstein_distance(reference.astype(float), current.astype(float)))


def compute_wasserstein_scores(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    cols = columns or FEATURE_COLS + [TARGET_COL]
    return {c: _column_wasserstein(reference[c], current[c]) for c in cols}


def build_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generate an Evidently HTML drift report for screenshots / audit."""
    # Evidently 0.7+ keeps the stable Report API under `evidently.legacy`
    from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    column_mapping = ColumnMapping(
        target=TARGET_COL,
        numerical_features=FEATURE_COLS,
    )
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    report.save_html(str(output_path))
    return output_path


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DRIFT_THRESHOLD,
    report_dir: Path | str = "outputs/drift_reports",
) -> DriftResult:
    """
    Compare reference vs current. Trigger retrain when any column's
    Wasserstein distance exceeds `threshold` (default 0.1).
    """
    scores = compute_wasserstein_scores(reference, current)
    max_score = max(scores.values()) if scores else 0.0
    drifted = max_score > threshold

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "drift_report.html"
    build_evidently_report(reference, current, report_path)

    return DriftResult(
        drifted=drifted,
        max_wasserstein=max_score,
        per_column=scores,
        report_path=str(report_path),
        threshold=threshold,
    )


def drift_result_to_dict(result: DriftResult) -> dict[str, Any]:
    return {
        "drifted": result.drifted,
        "max_wasserstein": result.max_wasserstein,
        "per_column": result.per_column,
        "report_path": result.report_path,
        "threshold": result.threshold,
    }
