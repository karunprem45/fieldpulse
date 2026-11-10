"""Run the pipeline: validate, clean, score, load, report.

    python -m fieldpulse.pipeline --generate 10      # make synthetic runs first
    python -m fieldpulse.pipeline                    # process data/raw into data/fieldpulse.db
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import anomaly, store
from .clean import clean_run
from .evaluate import evaluate
from .generate import generate
from .validate import check_schema, count_duplicates, find_gaps


def run(data_dir: str = "data", db_path: str = "data/fieldpulse.db", force: bool = False) -> dict:
    raw_dir = Path(data_dir) / "raw"
    con = store.connect(db_path)
    cleaned, summaries, skipped, rejected = [], [], 0, []

    for path in sorted(raw_dir.glob("*.csv")):
        digest = store.file_hash(path)
        if not force and store.already_loaded(con, path, digest):
            skipped += 1
            continue
        raw = pd.read_csv(path, parse_dates=["timestamp"])
        problems = check_schema(raw)
        if problems:
            rejected.append({"file": path.name, "problems": problems})
            continue
        run_id = raw["run_id"].iat[0]
        gaps = find_gaps(raw)
        df, stats = clean_run(raw)
        stats.update(run_id=run_id, file=path.name, gap_count=len(gaps),
                     duplicate_rows=count_duplicates(raw))
        cleaned.append((path, digest, run_id, df))
        summaries.append(stats)

    result = {"files_skipped_unchanged": skipped, "files_rejected": rejected, "runs": []}
    if not cleaned:
        return result

    batch = pd.concat([df for *_, df in cleaned], ignore_index=True)
    scored = anomaly.score(batch, anomaly.fit_models(batch))

    for (path, digest, run_id, _), stats in zip(cleaned, summaries):
        part = scored[scored["run_id"] == run_id]
        stats["rows_flagged_by_model"] = int(part["model_flags"].ne("").sum())
        bad = part["rule_flags"].ne("") | part["model_flags"].ne("")
        stats["quality_score"] = round(100 * (1 - bad.mean()), 2)
        store.replace_run(con, run_id, {
            "readings": part.assign(timestamp=part["timestamp"].astype(str)),
            "run_summary": pd.DataFrame([stats]),
        })
        store.mark_loaded(con, path, digest, run_id)
        result["runs"].append(stats)

    labels_dir = Path(data_dir) / "labels"
    if labels_dir.exists():
        report = evaluate(scored, str(labels_dir))
        result["evaluation"] = {
            "by_fault": report.round(3).to_dict(orient="records"),
            "precision_combined": round(float(report.attrs["precision_combined"]), 3),
            "false_alarm_rate": round(float(report.attrs["false_alarm_rate"]), 4),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--db", default="data/fieldpulse.db")
    p.add_argument("--generate", type=int, default=0, help="create N synthetic runs first")
    p.add_argument("--force", action="store_true", help="reprocess files even if unchanged")
    args = p.parse_args()
    if args.generate:
        generate(args.data_dir, n_runs=args.generate)
    result = run(args.data_dir, args.db, args.force)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latest.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result.get("evaluation", {}), indent=2))
    print(f"runs loaded: {len(result['runs'])}, skipped unchanged: {result['files_skipped_unchanged']}, "
          f"rejected: {len(result['files_rejected'])}")


if __name__ == "__main__":
    main()
