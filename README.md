# Automated Re-Training Pipeline Triggered by Data Drift

**Assignment 8** — Demand Forecasting MLOps pipeline.

When incoming data drifts past a Wasserstein threshold (**> 0.1**), the Prefect flow automatically retrains a scikit-learn demand model, registers it in the **MLflow Model Registry**, and writes an **email alert** file.

## Tech stack

| Piece | Choice |
|-------|--------|
| Orchestration | Prefect (local flow) |
| Drift | Evidently AI + Wasserstein distance |
| Model | scikit-learn `GradientBoostingRegressor` |
| Registry | MLflow Model Registry (local `mlruns/`) |
| Alerts | Email-style `.txt` files under `outputs/alerts/` |

## Quick start

```bash
# Python 3.12 recommended (via uv)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

python scripts/run_demo.py
```

### Optional UIs (for screenshots)

```bash
# MLflow tracking UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# open http://127.0.0.1:5000 → Models → demand-forecast-model

# Open Evidently HTML report
open outputs/drift_reports/drift_report.html
```

## Pipeline flow

1. Generate synthetic **reference** and **drifted** demand datasets  
2. Run Evidently drift report; compute per-column Wasserstein distances  
3. If `max(Wasserstein) > 0.1` → retrain + register `demand-forecast-model`  
4. Write email notification to `outputs/alerts/alert_*.txt`

## Project layout

```
src/
  data_generator.py   # synthetic demand data
  drift_monitor.py    # Evidently + Wasserstein gate
  train.py            # sklearn + MLflow registry
  alerts.py           # local email alerts
  pipeline.py         # Prefect flow
scripts/run_demo.py   # one-command demo
outputs/              # reports, alerts, run_summary.json
submission/           # screenshots for LMS upload
```

## Submission

See [submission/SUBMISSION.md](submission/SUBMISSION.md) for the GitHub link and screenshot index.
