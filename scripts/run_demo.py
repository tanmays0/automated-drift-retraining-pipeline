#!/usr/bin/env python3
"""One-command demo for Assignment 8 — drift-triggered retrain pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import drift_retrain_flow


def main() -> None:
    print("=" * 60)
    print("Assignment 8: Automated Re-Training on Data Drift")
    print("Domain: Demand Forecasting | Stack: Prefect + Evidently + sklearn + MLflow")
    print("=" * 60)
    result = drift_retrain_flow(threshold=0.1)
    summary_path = ROOT / "outputs" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=" * 60)
    print(f"Summary saved: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
