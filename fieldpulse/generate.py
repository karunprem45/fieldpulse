"""Synthetic field-test logs with injected, labeled faults.

Each run is one machine test session sampled at 1 Hz. The generator writes the raw log
and, separately, a label file saying exactly which rows were corrupted and how. The
pipeline never reads the labels; they are only used to measure how well it catches faults.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import SIGNALS


def _smooth(x, window=30):
    return pd.Series(x).rolling(window, min_periods=1).mean().to_numpy()


def make_clean_run(run_id: str, seconds: int, rng: np.random.Generator) -> pd.DataFrame:
    mode = np.empty(seconds, dtype=int)
    i = 0
    while i < seconds:
        length = int(rng.integers(120, 400))
        mode[i:i + length] = rng.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
        i += length

    load = _smooth(np.array([0.10, 0.40, 0.85])[mode])
    rpm = _smooth(np.array([800, 1800, 2100])[mode]) + rng.normal(0, 25, seconds)
    speed = np.clip(_smooth(np.array([0, 25, 9])[mode]) + rng.normal(0, 0.4, seconds), 0, None)
    warmup = 1 - np.exp(-np.arange(seconds) / 300)
    temp = 40 + (45 + 8 * load) * warmup + rng.normal(0, 0.3, seconds)
    fuel = 2 + 0.012 * rpm * load + rng.normal(0, 0.4, seconds)
    hyd = 60 + 120 * load + rng.normal(0, 2.0, seconds)

    start = pd.Timestamp("2026-06-01 08:00") + pd.Timedelta(hours=int(rng.integers(0, 2000)))
    return pd.DataFrame({
        "run_id": run_id,
        "machine_id": f"TRK-{int(rng.integers(100, 110))}",
        "timestamp": pd.date_range(start, periods=seconds, freq="s"),
        "engine_rpm": rpm,
        "engine_temp_c": temp,
        "ground_speed_kph": speed,
        "fuel_rate_lph": fuel,
        "hyd_pressure_bar": hyd,
        "_mode": mode,
    })


def inject_faults(df: pd.DataFrame, rng: np.random.Generator):
    """Corrupt a clean run. Returns (raw_df, labels_df)."""
    df = df.copy()
    n = len(df)
    labels = []

    def window(min_len, max_len, earliest=0):
        length = int(rng.integers(min_len, max_len))
        start = int(rng.integers(earliest, n - length))
        return start, start + length

    def label(rows, fault, col):
        for ts in df.loc[rows, "timestamp"]:
            labels.append((df["run_id"].iat[0], ts, fault, col))

    # Sensor dropout: readings go missing.
    col = rng.choice(SIGNALS)
    a, b = window(20, 60)
    df.loc[a:b - 1, col] = np.nan
    label(range(a, b), "dropout", col)

    # Stuck sensor: the value freezes.
    col = rng.choice(["engine_rpm", "fuel_rate_lph", "hyd_pressure_bar"])
    a, b = window(60, 120)
    df.loc[a:b - 1, col] = df.at[a, col]
    label(range(a, b), "stuck", col)

    # Spikes: isolated impossible values.
    col = rng.choice(SIGNALS)
    rows = rng.choice(n, size=int(rng.integers(3, 9)), replace=False)
    df.loc[rows, col] = rng.choice([9999.0, -1.0], size=len(rows))
    label(rows, "spike", col)

    # Unit flip: temperature logged in Fahrenheit after warm-up.
    a, b = window(60, 150, earliest=700)
    df.loc[a:b - 1, "engine_temp_c"] = df.loc[a:b - 1, "engine_temp_c"] * 9 / 5 + 32
    label(range(a, b), "unit_flip", "engine_temp_c")

    # Fuel leak: fuel rate too high for how hard the engine is working.
    # Every value stays inside physical limits, so range checks cannot see it.
    candidates = np.flatnonzero(df["_mode"].to_numpy()[:-150] < 2)
    if len(candidates) > 0:
        a = int(rng.choice(candidates))
        b = a + int(rng.integers(60, 150))
        df.loc[a:b - 1, "fuel_rate_lph"] += 12
        label(range(a, b), "fuel_leak", "fuel_rate_lph")

    # Structural faults: duplicated rows and a block of missing seconds.
    dup_rows = rng.choice(n, size=int(rng.integers(5, 20)), replace=False)
    df = pd.concat([df, df.loc[dup_rows]]).sort_values("timestamp", kind="stable")
    a, b = window(10, 40)
    gap_ts = df["timestamp"].drop_duplicates().iloc[a:b]
    df = df[~df["timestamp"].isin(gap_ts)]

    labels_df = pd.DataFrame(labels, columns=["run_id", "timestamp", "fault_type", "column"])
    return df.drop(columns="_mode").reset_index(drop=True), labels_df


def generate(out_dir: str, n_runs: int = 10, seconds: int = 1800, seed: int = 7) -> None:
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for k in range(n_runs):
        run_id = f"RUN-{k:03d}"
        raw, labels = inject_faults(make_clean_run(run_id, seconds, rng), rng)
        raw.to_csv(out / "raw" / f"{run_id}.csv", index=False)
        labels.to_csv(out / "labels" / f"{run_id}.csv", index=False)
