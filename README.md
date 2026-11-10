# FieldPulse

A Python data pipeline for machine field-test telemetry. It takes raw 1 Hz sensor logs from
test runs (engine RPM, coolant temperature, ground speed, fuel rate, hydraulic pressure),
checks them for quality problems, cleans them, and loads them into SQLite with a quality
flag on every row, so analysts can see exactly what was fixed, what was filled in, and what
should not be trusted.

## What it catches

| Problem | How |
|---|---|
| Missing columns or text in numeric fields | Schema check; the file is rejected, not loaded |
| Duplicate rows | Deduplicated on `(run_id, timestamp)` |
| Missing seconds | Reindexed to a full 1 Hz grid so gaps become explicit `is_gap_row` rows |
| Sensor dropouts | Missing-value flag; holes up to 5 s are interpolated and marked `is_imputed`, longer holes stay empty |
| Impossible values / unit errors | Physical range limits per signal |
| Stuck sensors | 30 identical readings in a row on a signal that always has noise |
| Readings that are plausible alone but wrong together (e.g. a fuel leak) | Cross-signal consistency model, below |

## The model

Rule checks look at one signal at a time, so they cannot see a fuel rate that is normal for
field work but far too high for an idling engine. The pipeline trains a gradient-boosted
regressor to predict fuel rate and hydraulic pressure from the other signals, using only rows
the rules passed. It then flags stretches where the residual stays large.

Faults the rules miss still end up in that training data. An absolute-error loss and a second
fit after trimming poorly explained rows keep them from being learned as normal.

## Data integrity

- Loads are idempotent: each run is deleted and re-inserted in one transaction.
- Files are tracked by SHA-256, so unchanged files are skipped and edited files are reprocessed.
- Nothing is silently dropped. Every changed value keeps a flag saying why.

## Measuring it

`fieldpulse/generate.py` creates synthetic test runs and injects labeled faults (dropout,
stuck sensor, spikes, Fahrenheit unit flip, fuel leak, duplicates, gaps). The pipeline never
reads the labels; `evaluate.py` compares its flags against them and reports recall per fault
type, precision, and the false-alarm rate on clean rows.

Synthetic data is easier than real data, because the same person wrote the faults and the
checks. Treat these numbers as proof that the logic works, not as real-world accuracy.

## Run it

```bash
pip install -r requirements.txt
pytest -q
python -m fieldpulse.pipeline --generate 10   # creates data/raw + data/labels, then processes
python -m fieldpulse.pipeline                 # re-run: unchanged files are skipped
```

Results go to `data/fieldpulse.db` (tables `readings`, `run_summary`, `processed_files`) and
`reports/latest.json`. GitHub Actions runs the tests and the pipeline on every push and weekly.
