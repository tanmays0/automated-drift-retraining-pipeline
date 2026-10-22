"""Email-style alert notifications written locally (no SMTP required)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def send_email_alert(
    subject: str,
    body: str,
    alert_dir: Path | str = "outputs/alerts",
    to_email: str = "ml-ops@example.com",
) -> Path:
    """
    Persist an email notification as a .txt file for demo/screenshots.
    Lightest path — no Discord/Slack/SMTP credentials needed.
    """
    alert_dir = Path(alert_dir)
    alert_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = alert_dir / f"alert_{stamp}.txt"

    message = (
        f"From: drift-monitor@pipeline.local\n"
        f"To: {to_email}\n"
        f"Date: {datetime.now(timezone.utc).isoformat()}\n"
        f"Subject: {subject}\n"
        f"\n"
        f"{body}\n"
    )
    path.write_text(message, encoding="utf-8")
    return path


def alert_from_pipeline(
    drift: dict[str, Any],
    train: dict[str, Any] | None = None,
    alert_dir: Path | str = "outputs/alerts",
) -> Path:
    if drift.get("drifted"):
        subject = "[ALERT] Data drift detected — retrain triggered"
        lines = [
            "Automated demand-forecasting pipeline alert.",
            "",
            f"Drift threshold (Wasserstein): {drift.get('threshold')}",
            f"Max Wasserstein distance: {drift.get('max_wasserstein'):.4f}",
            f"Evidently report: {drift.get('report_path')}",
            "",
            "Per-column Wasserstein scores:",
        ]
        for col, score in (drift.get("per_column") or {}).items():
            lines.append(f"  - {col}: {score:.4f}")
        if train:
            lines.extend(
                [
                    "",
                    "Retrain completed and model registered in MLflow:",
                    f"  model_name: {train.get('model_name')}",
                    f"  version: {train.get('registered_version')}",
                    f"  run_id: {train.get('run_id')}",
                    f"  metrics: {train.get('metrics')}",
                ]
            )
        body = "\n".join(lines)
    else:
        subject = "[OK] No significant data drift"
        body = (
            "Monitoring run completed. Wasserstein distances are within threshold.\n"
            f"Max Wasserstein: {drift.get('max_wasserstein'):.4f}\n"
            f"Report: {drift.get('report_path')}\n"
        )
    return send_email_alert(subject=subject, body=body, alert_dir=alert_dir)
