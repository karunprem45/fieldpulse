"""Score the pipeline against the injected fault labels. Only works on generated data."""
from pathlib import Path

import pandas as pd


def evaluate(scored: pd.DataFrame, labels_dir: str) -> pd.DataFrame:
    files = sorted(Path(labels_dir).glob("*.csv"))
    labels = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in files])
    truth = labels.groupby(["run_id", "timestamp"])["fault_type"].first().reset_index()

    df = scored[["run_id", "timestamp", "rule_flags", "model_flags"]].merge(
        truth, on=["run_id", "timestamp"], how="left")
    # Rows that fell inside an injected gap no longer hold any reading, so there is
    # nothing left to catch. They are reported as gap rows instead.
    df = df[df["rule_flags"].ne("gap")]
    df["by_rules"] = df["rule_flags"].ne("") & df["rule_flags"].ne("gap")
    df["by_model"] = df["model_flags"].ne("")
    df["by_either"] = df["by_rules"] | df["by_model"]

    rows = []
    for fault, grp in df.dropna(subset=["fault_type"]).groupby("fault_type"):
        rows.append({
            "fault_type": fault,
            "rows": len(grp),
            "recall_rules": grp["by_rules"].mean(),
            "recall_model": grp["by_model"].mean(),
            "recall_combined": grp["by_either"].mean(),
        })
    report = pd.DataFrame(rows)

    clean_rows = df[df["fault_type"].isna()]
    flagged = df[df["by_either"]]
    report.attrs["precision_combined"] = flagged["fault_type"].notna().mean()
    report.attrs["false_alarm_rate"] = clean_rows["by_either"].mean()
    return report
