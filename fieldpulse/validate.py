"""Rule-based data quality checks. Every check reports what it found; nothing is dropped here."""
import pandas as pd

from . import LIMITS, NOISY_SIGNALS, SIGNALS

REQUIRED = ["run_id", "machine_id", "timestamp", *SIGNALS]


def check_schema(df: pd.DataFrame) -> list[str]:
    problems = [f"missing column: {c}" for c in REQUIRED if c not in df.columns]
    for c in SIGNALS:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"non-numeric column: {c}")
    return problems


def flag_rows(df: pd.DataFrame, stuck_window: int = 30) -> pd.DataFrame:
    """Return a boolean frame: one column per check:signal, True where the row fails."""
    flags = {}
    for col in SIGNALS:
        lo, hi = LIMITS[col]
        flags[f"missing:{col}"] = df[col].isna()
        flags[f"out_of_range:{col}"] = (df[col] < lo) | (df[col] > hi)
    for col in NOISY_SIGNALS:
        # Exact repeats, not a tiny std: rolling std on floats is rarely exactly 0.
        repeats = df[col].diff().eq(0).astype(float)
        flat_end = repeats.rolling(stuck_window - 1).sum().eq(stuck_window - 1).astype(float)
        # Mark every row of the flat window, not just the row where it was detected.
        flat_any = flat_end[::-1].rolling(stuck_window, min_periods=1).max()[::-1]
        flags[f"stuck:{col}"] = flat_any.fillna(0).astype(bool)
    return pd.DataFrame(flags, index=df.index)


def find_gaps(df: pd.DataFrame, period: str = "1s") -> pd.DataFrame:
    ts = df["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    step = ts.diff()
    expected = pd.Timedelta(period)
    is_gap = step > expected
    return pd.DataFrame({
        "gap_start": ts.shift(1)[is_gap],
        "missing_seconds": (step[is_gap] / expected - 1).astype(int),
    }).reset_index(drop=True)


def count_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=["run_id", "timestamp"]).sum())
