"""Run the whole evaluation and rebuild every output with one command.

Every evaluation is a *run* with its own folder, experiments/runs/<run_id>/ (see backend/utils/runs.py). A new
run never changes an earlier one; an interrupted run is continued with --resume.

Stages, in order (each can be run on its own with --stages):
  data         download VQA-RAD if data/vqa-rad/ is missing                          free
  eval         every question in experiments/eval_set.json x 4 systems               paid
  judge        blinded LLM judge for strict-wrong open-ended answers                   paid
  variance     baseline + agentic loop again on 50 of the cases                        paid
  metrics      final_metrics.json + cases.csv in the run folder                        free
  docs         README number blocks, docs/report/results.tex, experiments/runs/README.md free
  diagram      system_diagram.png (+ the report's copy)                                free
  screenshots  start backend + frontend, capture the dashboard                         free (Node + Chrome/Edge)
  report       compile the LaTeX report with tectonic                                  free (tectonic)

Paid stages ask first (or --yes) and share one spend cap per run, --budget. Judge runs before variance, so if
money runs short only the (least important) variance repeats are cut. The free output stages only run once
the run is complete, so the README and report never show a half-finished run.

Usage (run from repo root):
    python scripts/run_all.py --dry-run                   # plan + cost estimate, spends nothing
    python scripts/run_all.py --budget 10                 # a new run, then all outputs
    python scripts/run_all.py --resume                    # continue the latest unfinished run
    python scripts/run_all.py --stages metrics,docs,diagram,screenshots,report   # rebuild outputs (free)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from utils import runs  # noqa: E402
from utils.data_loader import eval_set_ids  # noqa: E402

SCRIPTS = ROOT / "scripts"
REPORT_DIR = ROOT / "docs" / "report"
REPORT_PDF = REPORT_DIR / "MGAI_Final_Report_Group9_revised.pdf"

STAGES = ("data", "eval", "judge", "variance", "metrics", "docs", "diagram", "screenshots", "report")
PAID = ("eval", "judge", "variance")
OUTPUTS = ("metrics", "docs", "diagram", "screenshots", "report")
# USD for 300 questions with the blind check and zoom (per-call costs of the 14 Oct run, plus one blind reading
# per case and more re-looks); replaced by the last comparable run's actual cost once there is one.
DEFAULT_ESTIMATE = {"eval": 6.70, "judge": 0.18, "variance": 0.75}
EVAL_CASES = len(eval_set_ids() or []) or 200   # experiments/eval_set.json (scripts/make_eval_set.py)
VARIANCE_CASES, CONCURRENCY = 50, 4


def run(cmd: list, cwd: Path = ROOT) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def python(script: str, *args) -> None:
    run([sys.executable, SCRIPTS / script, *args])


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def estimates() -> dict:
    """Cost of each paid stage, scaled from the latest completed run of the same system (one with the blind check),
    or the default estimate."""
    for run_info in reversed([r for r in runs.list_runs() if r.get("status") == "completed"]):
        metrics = load_json(runs.run_dir(run_info["run_id"]) / "final_metrics.json")
        if metrics.get("blind_reading") and metrics.get("cases"):
            scale = EVAL_CASES / metrics["cases"]
            stages = run_info.get("stages", {})
            return {s: round(((stages.get(s) or {}).get("cost_usd") or DEFAULT_ESTIMATE[s]) * (scale if s != "variance" else 1), 2)
                    for s in PAID}
    return dict(DEFAULT_ESTIMATE)


# ---------------------------------------------------------------------------- what a run has spent / finished
def stage_spend(run_id: str, stage: str) -> float:
    folder = runs.run_dir(run_id)
    if stage == "eval":
        return sum(sum(v.get("cost_usd", 0) for v in (load_json(Path(f)).get("usage") or {}).values())
                   for f in glob.glob(str(runs.cases_dir(run_id) / "*.json")))
    if stage == "judge":
        return float(load_json(folder / "judgments.json").get("cost_usd", 0))
    return float(load_json(folder / "variance.json").get("total_cost_usd", 0))


def run_spend(run_id: str) -> float:
    return sum(stage_spend(run_id, s) for s in PAID)


def eval_complete(run_id: str) -> int:
    """Number of finished cases (all four systems)."""
    return sum(1 for f in glob.glob(str(runs.cases_dir(run_id) / "*.json")) if load_json(Path(f)).get("pipeline_result"))


def stage_note(run_id: str, stage: str) -> str:
    if stage == "eval":
        return f"{eval_complete(run_id)} cases x 4 systems, {CONCURRENCY} at a time"
    if stage == "judge":
        return f"{len(load_json(runs.run_dir(run_id) / 'judgments.json').get('judgments', {}))} open-ended cases"
    return f"{len(load_json(runs.run_dir(run_id) / 'variance.json').get('rows', []))} cases, one at a time"


# ---------------------------------------------------------------------------- stages
def stage_data(run_id, budget_left):
    if (ROOT / "data" / "vqa-rad").exists():
        print("  data/vqa-rad/ present")
        return
    python("download_dataset.py")


def stage_eval(run_id, budget_left):
    python("run_case_explorer_cache.py", "--run-id", run_id, "--limit", EVAL_CASES, "--ablation",
           "--concurrency", CONCURRENCY, "--retries", 2, "--max-cost", f"{budget_left:.2f}")


def stage_judge(run_id, budget_left):
    python("judge_open_ended.py", "--run-id", run_id, "--max-cost", f"{budget_left:.2f}")


def stage_variance(run_id, budget_left):
    python("pilot.py", "--systems", "baseline,agentic_loop", "--from-eval-sample", EVAL_CASES, "--n", VARIANCE_CASES,
           "--out", runs.run_dir(run_id) / "variance.json", "--max-cost", f"{budget_left:.2f}")


def stage_metrics(run_id, budget_left):
    python("final_metrics.py", "--run-id", run_id)


def stage_docs(run_id, budget_left):
    python("update_docs.py")


def stage_diagram(run_id, budget_left):
    python("system_diagram.py")
    shutil.copyfile(ROOT / "system_diagram.png", REPORT_DIR / "system_diagram.png")


def _wait_for(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"{url} did not come up within {timeout:.0f} s")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":  # npm/vite spawn children: kill the whole tree
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)


def stage_screenshots(run_id, budget_left):
    node, npm = shutil.which("node"), shutil.which("npm")
    if not (node and npm):
        raise RuntimeError("Node.js/npm not found; install Node 18+ to capture screenshots")
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").exists():
        run([npm, "install"], cwd=frontend)
    popen = {"start_new_session": True} if sys.platform != "win32" else {}
    log_dir = ROOT / "backend" / "logs"  # gitignored
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "run_all_backend.log", "w") as backend_log, open(log_dir / "run_all_frontend.log", "w") as frontend_log:
        backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"], cwd=ROOT / "backend",
                                   stdout=backend_log, stderr=subprocess.STDOUT, **popen)
        frontend_proc = subprocess.Popen([npm, "run", "dev", "--", "--port", "5173", "--strictPort"], cwd=frontend,
                                         stdout=frontend_log, stderr=subprocess.STDOUT, **popen)
        try:
            _wait_for("http://localhost:8000/", 180)
            _wait_for("http://localhost:5173/", 120)
            run([node, "scripts/capture-screenshots.mjs"], cwd=frontend)
        finally:
            _stop(frontend_proc)
            _stop(backend)


def stage_report(run_id, budget_left):
    tectonic = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("tectonic not found: conda install -c conda-forge tectonic (or set TECTONIC=<path>)")
    run([tectonic, "-X", "compile", "main.tex"], cwd=REPORT_DIR)
    shutil.move(REPORT_DIR / "main.pdf", REPORT_PDF)
    print(f"  -> {REPORT_PDF.relative_to(ROOT)}")


RUNNERS = {name: globals()[f"stage_{name}"] for name in STAGES}


# ---------------------------------------------------------------------------- run bookkeeping
def start_run(run_id: str, budget: float) -> None:
    from config import CODE_VERSION, model_manifest  # loads .env; only needed when a run starts
    runs.write_run(run_id, {"run_id": run_id, "status": "running", "started": now(), "finished": None,
                            "code_version": CODE_VERSION, "models": model_manifest(), "budget_usd": budget, "stages": {}})


def record_stage(run_id: str, stage: str, started: str, seconds: float) -> None:
    info = runs.read_run(run_id)
    entry = info.setdefault("stages", {}).get(stage, {})
    info["stages"][stage] = {
        "started": entry.get("started", started),  # a resumed stage keeps its first start time
        "finished": now(),
        "wall_clock_s": int(entry.get("wall_clock_s", 0) + seconds),
        "cost_usd": round(stage_spend(run_id, stage), 4),
        "note": stage_note(run_id, stage),
    }
    runs.write_run(run_id, info)


def finish_run(run_id: str) -> bool:
    info = runs.read_run(run_id)
    if eval_complete(run_id) < EVAL_CASES or not all(s in info.get("stages", {}) for s in PAID):
        return False
    info.update(status="completed", finished=now())
    runs.write_run(run_id, info)
    return True


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--stages", default=",".join(STAGES), help=f"Comma-separated subset of: {', '.join(STAGES)}")
    parser.add_argument("--budget", type=float, default=10.0, help="USD cap for the whole run, across sessions (default 10.0)")
    parser.add_argument("--resume", action="store_true", help="Continue the latest unfinished run (or --run-id)")
    parser.add_argument("--run-id", default=None, help="Run to resume or rebuild (default: new run / current run)")
    parser.add_argument("--yes", action="store_true", help="Do not ask before running paid stages")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and cost estimate, then exit")
    args = parser.parse_args()

    selected = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(selected) - set(STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    selected = [s for s in STAGES if s in selected]  # always run in dependency order
    paid = [s for s in selected if s in PAID]

    if args.resume:
        run_id = args.run_id or runs.latest_unfinished_run_id()
        if not run_id:
            sys.exit("Nothing to resume: every run is completed")
        new_run = False
    elif paid:
        run_id = args.run_id or runs.new_run_id()
        if runs.run_dir(run_id).exists():
            sys.exit(f"Run {run_id} already exists; use --resume to continue it")
        new_run = True
    else:
        run_id = args.run_id or runs.current_run_id()
        if not run_id:
            sys.exit("No run yet: run python scripts/run_all.py first")
        new_run = False

    est = estimates()
    spent = 0.0 if new_run else run_spend(run_id)
    if not new_run:  # only the remaining evaluation cases still cost money
        est["eval"] = round(est["eval"] * (1 - eval_complete(run_id) / EVAL_CASES), 2)
    print(f"Run {run_id} ({'new' if new_run else 'existing'}):", " -> ".join(selected))
    if paid:
        total = sum(est[s] for s in paid)
        print("Estimated OpenAI cost: " + ", ".join(f"{s} ${est[s]:.2f}" for s in paid)
              + f" = ${total:.2f}; budget ${args.budget:.2f}" + (f", already spent ${spent:.2f}" if spent else ""))
        if not new_run:
            print(f"Resuming: {eval_complete(run_id)}/{EVAL_CASES} evaluation cases done; finished cases are kept.")
    if args.dry_run:
        return
    if paid and not args.yes:
        try:
            answer = input("Proceed with paid stages? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            sys.exit("Cancelled (nothing was spent).")

    os.environ["MEDVQA_RUN"] = run_id  # every stage (and the servers for screenshots) works on this run
    if new_run:
        start_run(run_id, args.budget)

    for stage in selected:
        if stage in OUTPUTS and paid and runs.read_run(run_id).get("status") != "completed":
            print(f"\n== {stage}: skipped, run {run_id} is not complete (resume with: python scripts/run_all.py --resume)")
            continue
        left = args.budget - run_spend(run_id)
        if stage in PAID and left <= 0:
            print(f"\n== {stage}: skipped, budget of ${args.budget:.2f} used up")
            continue
        print(f"\n== {stage}", flush=True)
        started, t0 = now(), time.time()
        try:
            RUNNERS[stage](run_id, left)
        except (subprocess.CalledProcessError, RuntimeError) as error:
            if stage in PAID:
                record_stage(run_id, stage, started, time.time() - t0)
            sys.exit(f"\nStage '{stage}' failed: {error}\nFix it, then: python scripts/run_all.py --resume"
                     if stage in PAID else f"\nStage '{stage}' failed: {error}")
        if stage in PAID:
            record_stage(run_id, stage, started, time.time() - t0)
            print(f"  {stage}: ${stage_spend(run_id, stage):.2f}  (run total ${run_spend(run_id):.2f})")
            if stage == paid[-1]:
                print(f"  run {run_id}: " + ("completed" if finish_run(run_id) else
                      f"not complete ({eval_complete(run_id)}/{EVAL_CASES} cases); continue with --resume"))
        else:
            print(f"  {stage}: done in {dt.timedelta(seconds=int(time.time() - t0))}")

    if paid:
        print(f"\nOpenAI spend for run {run_id}: ${run_spend(run_id):.2f}")
        print("Review by hand after a new run: the judge spot check (judge_review_sample.md in the run folder) and the\n"
              "narrative in README 'Systematic failure patterns' and the report, which describe error patterns.")


if __name__ == "__main__":
    main()
