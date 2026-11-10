"""Turn a validated raw run into an analysis-ready table with a quality flag on every row."""
import numpy as np
import pandas as pd

from . import SIGNALS
from .validate import flag_rows


def clean_run(raw: pd.DataFrame, max_fill_seconds: int = 5):
    """Returns (clean_df, stats). Bad values become NaN and keep a flag saying why."""
    stats = {"rows_in": len(raw)}
    df = raw.drop_duplicates(subset=["run_id", "timestamp"], keep="first")
    stats["duplicates_removed"] = stats["rows_in"] - len(df)
    df = df.sort_values("timestamp").set_index("timestamp")

    flags = flag_rows(df)
    for col in SIGNALS:
        bad = flags[f"out_of_range:{col}"]
        if f"stuck:{col}" in flags:
            bad = bad | flags[f"stuck:{col}"]
        df.loc[bad, col] = np.nan

    # Reindex to a full 1 Hz grid so gaps become explicit rows instead of silent jumps.
    full = pd.date_range(df.index.min(), df.index.max(), freq="s", name="timestamp")
    df = df.reindex(full)
    flags = flags.reindex(full, fill_value=False)
    df["is_gap_row"] = df["run_id"].isna()
    stats["gap_rows"] = int(df["is_gap_row"].sum())
    df[["run_id", "machine_id"]] = df[["run_id", "machine_id"]].ffill()

    # Fill only short holes. Long holes stay empty so nobody mistakes guesses for data.
    # (pandas' interpolate(limit=n) would fill the first n seconds of a long hole too.)
    before = df[SIGNALS].isna()
    for col in SIGNALS:
        hole_id = before[col].ne(before[col].shift()).cumsum()
        hole_len = before[col].groupby(hole_id).transform("size")
        short = before[col] & (hole_len <= max_fill_seconds)
        filled = df[col].interpolate(limit_area="inside")
        df.loc[short, col] = filled[short]
    df["is_imputed"] = (before & df[SIGNALS].notna()).any(axis=1)

    names = np.array(flags.columns, dtype=object)
    df["rule_flags"] = [";".join(names[row]) for row in flags.to_numpy()]
    df.loc[df["is_gap_row"], "rule_flags"] = "gap"
    df = df.reset_index()

    stats["rows_out"] = len(df)
    stats["rows_flagged_by_rules"] = int((df["rule_flags"] != "").sum())
    return df, stats
