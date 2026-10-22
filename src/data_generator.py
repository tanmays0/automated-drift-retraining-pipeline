"""Synthetic demand-forecasting datasets (reference + drifted)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = ["price", "promo", "day_of_week", "temperature", "store_id"]
TARGET_COL = "units_sold"


def _base_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": rng.normal(12.0, 2.0, n).clip(4, 30),
            "promo": rng.binomial(1, 0.25, n),
            "day_of_week": rng.integers(0, 7, n),
            "temperature": rng.normal(22.0, 5.0, n),
            "store_id": rng.integers(1, 6, n),
        }
    )


def generate_reference(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Stable demand regime used as the monitoring baseline."""
    rng = np.random.default_rng(seed)
    df = _base_frame(n, rng)
    noise = rng.normal(0, 3, n)
    df[TARGET_COL] = (
        80
        - 2.5 * df["price"]
        + 18 * df["promo"]
        + 1.2 * df["day_of_week"]
        + 0.4 * df["temperature"]
        + 2.0 * df["store_id"]
        + noise
    ).clip(lower=1)
    return df


def generate_drifted(n: int = 2000, seed: int = 99) -> pd.DataFrame:
    """Shifted feature distributions + changed demand relationship (concept drift)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "price": rng.normal(18.0, 3.5, n).clip(4, 40),  # higher prices
            "promo": rng.binomial(1, 0.55, n),  # more promos
            "day_of_week": rng.integers(0, 7, n),
            "temperature": rng.normal(30.0, 6.0, n),  # hotter season
            "store_id": rng.integers(1, 6, n),
        }
    )
    noise = rng.normal(0, 4, n)
    # Concept drift: price elasticity and promo lift change
    df[TARGET_COL] = (
        95
        - 4.0 * df["price"]
        + 28 * df["promo"]
        + 0.5 * df["day_of_week"]
        + 0.9 * df["temperature"]
        + 1.5 * df["store_id"]
        + noise
    ).clip(lower=1)
    return df


def save_datasets(
    data_dir: Path | str = "data",
    n_ref: int = 2000,
    n_cur: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    reference = generate_reference(n_ref)
    current = generate_drifted(n_cur)
    reference.to_csv(data_dir / "reference.csv", index=False)
    current.to_csv(data_dir / "current_drifted.csv", index=False)
    return reference, current
