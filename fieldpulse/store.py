"""SQLite storage. Loads are idempotent: re-running a file replaces its rows, never doubles them."""
import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_files (
    file_name TEXT PRIMARY KEY, sha256 TEXT NOT NULL, run_id TEXT NOT NULL, loaded_at TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def already_loaded(con, path: Path, digest: str) -> bool:
    row = con.execute("SELECT sha256 FROM processed_files WHERE file_name = ?", (path.name,)).fetchone()
    return row is not None and row[0] == digest


def replace_run(con, run_id: str, tables: dict[str, pd.DataFrame]) -> None:
    """Delete then insert inside one transaction, so a crash never leaves half a run."""
    with con:
        for name, frame in tables.items():
            exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
            if exists:
                con.execute(f"DELETE FROM {name} WHERE run_id = ?", (run_id,))
            frame.to_sql(name, con, if_exists="append", index=False)


def mark_loaded(con, path: Path, digest: str, run_id: str) -> None:
    with con:
        con.execute(
            "INSERT OR REPLACE INTO processed_files VALUES (?, ?, ?, datetime('now'))",
            (path.name, digest, run_id))
