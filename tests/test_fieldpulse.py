import numpy as np
import pandas as pd

from fieldpulse import anomaly, store
from fieldpulse.clean import clean_run
from fieldpulse.generate import inject_faults, make_clean_run
from fieldpulse.pipeline import run
from fieldpulse.generate import generate
from fieldpulse.validate import check_schema, count_duplicates, find_gaps, flag_rows


def tiny_run(n=120):
    rng = np.random.default_rng(0)
    return make_clean_run("RUN-T", n, rng).drop(columns="_mode")


def test_schema_catches_missing_and_text_columns():
    df = tiny_run().drop(columns="fuel_rate_lph")
    df["engine_rpm"] = "fast"
    problems = check_schema(df)
    assert "missing column: fuel_rate_lph" in problems
    assert "non-numeric column: engine_rpm" in problems


def test_out_of_range_is_flagged():
    df = tiny_run()
    df.loc[10, "engine_rpm"] = 9999
    assert flag_rows(df)["out_of_range:engine_rpm"].sum() == 1


def test_stuck_sensor_flags_whole_window():
    df = tiny_run()
    df.loc[20:79, "hyd_pressure_bar"] = 150.0
    stuck = flag_rows(df)["stuck:hyd_pressure_bar"]
    assert stuck[20:80].all()
    assert not stuck[:20].any()


def test_parked_machine_is_not_called_stuck():
    df = tiny_run()
    df["ground_speed_kph"] = 0.0
    assert not any(c.startswith("stuck:ground_speed") for c in flag_rows(df).columns)


def test_gaps_and_duplicates_are_counted():
    df = tiny_run()
    df = pd.concat([df, df.iloc[[5, 6]]])
    df = df[~df.index.isin(range(30, 40))]
    assert count_duplicates(df) == 2
    gaps = find_gaps(df)
    assert len(gaps) == 1 and gaps["missing_seconds"].iat[0] == 10


def test_clean_run_dedupes_and_makes_gaps_explicit():
    df = tiny_run()
    df = pd.concat([df, df.iloc[[5]]])
    df = df[~df.index.isin(range(30, 40))]
    clean, stats = clean_run(df)
    assert stats["duplicates_removed"] == 1
    assert stats["gap_rows"] == 10
    assert len(clean) == 120
    assert clean["timestamp"].is_unique


def test_long_holes_are_not_filled():
    df = tiny_run()
    df.loc[50:69, "engine_temp_c"] = np.nan
    clean, _ = clean_run(df, max_fill_seconds=5)
    assert clean.loc[50:69, "engine_temp_c"].isna().all()


def test_model_flags_fuel_leak_that_rules_miss():
    rng = np.random.default_rng(3)
    frames = [make_clean_run(f"RUN-{i}", 1200, rng) for i in range(4)]
    df = pd.concat(frames, ignore_index=True)
    idle = df.index[(df["_mode"] == 0) & (df["run_id"] == "RUN-0")][:100]
    df.loc[idle, "fuel_rate_lph"] += 12
    clean = pd.concat([clean_run(g.drop(columns="_mode"))[0] for _, g in df.groupby("run_id")],
                      ignore_index=True)
    leak_ts = set(df.loc[idle, "timestamp"])
    leak = clean["timestamp"].isin(leak_ts) & clean["run_id"].eq("RUN-0")
    assert (clean.loc[leak, "rule_flags"] == "").all()
    scored = anomaly.score(clean, anomaly.fit_models(clean))
    assert scored.loc[leak, "model_flags"].ne("").mean() > 0.9


def test_rerun_is_idempotent_and_skips_unchanged(tmp_path):
    generate(str(tmp_path), n_runs=2, seconds=900)
    db = str(tmp_path / "fp.db")
    first = run(str(tmp_path), db)
    second = run(str(tmp_path), db)
    third = run(str(tmp_path), db, force=True)
    con = store.connect(db)
    rows = con.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    expected = sum(r["rows_out"] for r in first["runs"])
    assert second["files_skipped_unchanged"] == 2
    assert len(third["runs"]) == 2
    assert rows == expected
