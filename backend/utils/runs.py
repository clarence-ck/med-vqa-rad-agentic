"""Evaluation runs: each run lives in its own folder and is never overwritten by a later run.

experiments/runs/<run_id>/
    run.json            status (running/completed), start/finish, code version, models, stage timings and costs
    cases/              one JSON per evaluated case (all four systems)
    judgments.json      LLM-judge verdicts
    variance.json       repeat runs of the baseline and the agentic loop
    final_metrics.json  every headline number
    cases.csv           per-case answers and correctness, for reading without the app

The "current" run is the one named by MEDVQA_RUN (scripts/run_all.py sets it for every stage), else the latest
completed run, else the latest run. Analyses started from the UI for cases outside the current run go to
backend/.live_cache/, so they never change a run's record.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "experiments" / "runs"
LIVE_CACHE_DIR = ROOT / "backend" / ".live_cache"


def new_run_id() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H%M")


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def cases_dir(run_id: str) -> Path:
    return run_dir(run_id) / "cases"


def read_run(run_id: str) -> dict:
    try:
        return json.loads((run_dir(run_id) / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_run(run_id: str, data: dict) -> None:
    path = run_dir(run_id) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def list_runs() -> list[dict]:
    """All runs, oldest first (run IDs sort by start time)."""
    if not RUNS_DIR.exists():
        return []
    return [{"run_id": d.name, **read_run(d.name)} for d in sorted(RUNS_DIR.iterdir())
            if d.is_dir() and (d / "run.json").exists()]


def current_run_id() -> str | None:
    if os.getenv("MEDVQA_RUN"):
        return os.environ["MEDVQA_RUN"]
    runs = list_runs()
    completed = [r for r in runs if r.get("status") == "completed"]
    return (completed or runs or [{}])[-1].get("run_id")


def latest_unfinished_run_id() -> str | None:
    unfinished = [r for r in list_runs() if r.get("status") != "completed"]
    return unfinished[-1]["run_id"] if unfinished else None
