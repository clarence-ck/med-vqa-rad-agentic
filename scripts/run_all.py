"""Run the whole evaluation and rebuild every output with one command.

Stages, in order (each can be run on its own with --stages):
  data         download VQA-RAD if data/vqa-rad/ is missing                       free
  eval         200 cases x 4 systems (baseline, workflow, agentic loop, ablation)   paid
  variance     re-run baseline + agentic loop on 50 evaluated cases                 paid
  judge        blinded LLM judge for strict-wrong open-ended answers                paid
  metrics      experiments/final_metrics.json                                       free
  docs         README number blocks + docs/report/results.tex                       free
  diagram      system_diagram.png (+ the report's copy)                             free
  screenshots  start backend + frontend, capture the dashboard                      free (needs Node + Chrome/Edge)
  report       compile the LaTeX report with tectonic                               free (needs tectonic)

Paid stages are capped by --budget (USD, shared across stages) and need confirmation (or --yes).
Wall-clock times are written to experiments/run_log.json and end up in the README and report.

Usage (run from repo root):
    python scripts/run_all.py --dry-run                     # plan + cost estimate, spends nothing
    python scripts/run_all.py                               # everything (asks before spending)
    python scripts/run_all.py --stages metrics,docs,diagram,screenshots,report   # rebuild outputs only (free)
    python scripts/run_all.py --stages eval --resume        # continue an interrupted evaluation
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
SCRIPTS = ROOT / "scripts"
EXPERIMENTS = ROOT / "experiments"
RUN_LOG = EXPERIMENTS / "run_log.json"
METRICS = EXPERIMENTS / "final_metrics.json"
CACHE_DIR = ROOT / "backend" / ".case_explorer_cache"
REPORT_DIR = ROOT / "docs" / "report"
REPORT_PDF = REPORT_DIR / "MGAI_Final_Report_Group9_revised.pdf"

STAGES = ("data", "eval", "variance", "judge", "metrics", "docs", "diagram", "screenshots", "report")
PAID = ("eval", "variance", "judge")
DEFAULT_ESTIMATE = {"eval": 3.45, "variance": 0.55, "judge": 0.12}  # USD, used before any run has been logged
EVAL_CASES, VARIANCE_CASES, CONCURRENCY = 200, 50, 4


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env={**os.environ, **(env or {})}, check=True)


def python(script: str, *args) -> None:
    run([sys.executable, SCRIPTS / script, *args])


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def estimates() -> dict:
    """Cost of each paid stage in the last run, or a default."""
    costs = load_json(METRICS).get("costs_usd", {})
    last = {"eval": costs.get("evaluation_run"), "variance": costs.get("variance_repeats"), "judge": costs.get("judge")}
    return {stage: round(last[stage] or DEFAULT_ESTIMATE[stage], 2) for stage in PAID}


# ---------------------------------------------------------------------------- spend per paid stage
def eval_spend(since: float) -> float:
    total = 0.0
    for file in glob.glob(str(CACHE_DIR / "*.json")):
        payload = load_json(Path(file))
        if (payload.get("cached_at") or 0) >= since:
            total += sum(v.get("cost_usd", 0) for v in (payload.get("usage") or {}).values())
    return total


def variance_spend(_since: float) -> float:
    return float(load_json(EXPERIMENTS / "variance_repeat" / "results.json").get("total_cost_usd", 0))


def judge_spend(_since: float) -> float:
    return float(load_json(EXPERIMENTS / "judge" / "judgments.json").get("cost_usd", 0))


# ---------------------------------------------------------------------------- stages
def stage_data(args, budget_left):
    if (ROOT / "data" / "vqa-rad").exists():
        print("  data/vqa-rad/ present")
        return
    python("download_dataset.py")


def stage_eval(args, budget_left):
    flags = [] if args.resume else ["--refresh"]
    python("run_case_explorer_cache.py", "--limit", EVAL_CASES, "--ablation", "--concurrency", CONCURRENCY,
           "--retries", 2, "--max-cost", f"{budget_left:.2f}", *flags)


def stage_variance(args, budget_left):
    python("pilot.py", "--systems", "baseline,agentic_loop", "--from-eval-sample", EVAL_CASES,
           "--n", VARIANCE_CASES, "--name", "variance_repeat", "--max-cost", f"{budget_left:.2f}")


def stage_judge(args, budget_left):
    flags = [] if args.resume else ["--fresh"]
    python("judge_open_ended.py", "--max-cost", f"{budget_left:.2f}", *flags)


def stage_metrics(args, budget_left):
    python("final_metrics.py")


def stage_docs(args, budget_left):
    python("update_docs.py")


def stage_diagram(args, budget_left):
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


def stage_screenshots(args, budget_left):
    node, npm = shutil.which("node"), shutil.which("npm")
    if not (node and npm):
        raise RuntimeError("Node.js/npm not found; install Node 18+ to capture screenshots")
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").exists():
        run([npm, "install"], cwd=frontend)
    popen = {"start_new_session": True} if sys.platform != "win32" else {}
    log_dir = ROOT / "backend" / "logs"  # gitignored
    log_dir.mkdir(exist_ok=True)
    backend_log, frontend_log = open(log_dir / "run_all_backend.log", "w"), open(log_dir / "run_all_frontend.log", "w")
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
        backend_log.close()
        frontend_log.close()


def stage_report(args, budget_left):
    tectonic = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("tectonic not found: conda install -c conda-forge tectonic (or set TECTONIC=<path>)")
    run([tectonic, "-X", "compile", "main.tex"], cwd=REPORT_DIR)
    shutil.move(REPORT_DIR / "main.pdf", REPORT_PDF)
    print(f"  -> {REPORT_PDF.relative_to(ROOT)}")


RUNNERS = {name: globals()[f"stage_{name}"] for name in STAGES}
SPEND = {"eval": eval_spend, "variance": variance_spend, "judge": judge_spend}
NOTES = {
    "eval": lambda: f"{EVAL_CASES} cases x 4 systems, {CONCURRENCY} at a time",
    "variance": lambda: f"{VARIANCE_CASES} cases, one at a time",
    "judge": lambda: f"{len(load_json(EXPERIMENTS / 'judge' / 'judgments.json').get('judgments', {}))} open-ended cases",
}


def record(stage: str, started: dt.datetime, finished: dt.datetime) -> None:
    log = load_json(RUN_LOG) or {"stages": {}}
    entry = {"started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
             "wall_clock_s": int((finished - started).total_seconds())}
    if stage in NOTES:
        entry["note"] = NOTES[stage]()
    log.setdefault("stages", {})[stage] = entry
    RUN_LOG.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--stages", default=",".join(STAGES), help=f"Comma-separated subset of: {', '.join(STAGES)}")
    parser.add_argument("--budget", type=float, default=5.0, help="USD cap shared by the paid stages (default 5.0)")
    parser.add_argument("--yes", action="store_true", help="Do not ask before running paid stages")
    parser.add_argument("--resume", action="store_true", help="Keep existing results (eval: skip cached cases; judge: keep verdicts)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and cost estimate, then exit")
    args = parser.parse_args()

    selected = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(selected) - set(STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    selected = [s for s in STAGES if s in selected]  # always run in dependency order

    est = estimates()
    paid = [s for s in selected if s in PAID]
    print("Plan:", " -> ".join(selected))
    if paid:
        total = sum(est[s] for s in paid)
        print("Estimated OpenAI cost (last run): " + ", ".join(f"{s} ${est[s]:.2f}" for s in paid)
              + f"  = ${total:.2f}  (budget cap ${args.budget:.2f})")
        if total > args.budget:
            print(f"Warning: the estimate exceeds --budget; stages stop starting new work at the cap.")
    if args.dry_run:
        return
    if paid and not args.yes:
        if not sys.stdin.isatty():
            sys.exit("Paid stages need confirmation: re-run with --yes")
        try:
            answer = input("Proceed with paid stages? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            sys.exit("Cancelled (nothing was spent).")

    spent = 0.0
    for stage in selected:
        left = args.budget - spent
        if stage in PAID and left <= 0:
            print(f"\n== {stage}: skipped, budget used up (${spent:.2f})")
            continue
        print(f"\n== {stage}", flush=True)
        started = dt.datetime.now()
        try:
            RUNNERS[stage](args, left)
        except (subprocess.CalledProcessError, RuntimeError) as error:
            sys.exit(f"\nStage '{stage}' failed: {error}\n"
                     f"Fix it, then re-run with --stages {stage},... (--resume keeps paid results).")
        finished = dt.datetime.now()
        if stage in PAID:
            cost = SPEND[stage](started.timestamp())
            spent += cost
            record(stage, started, finished)
            print(f"  {stage}: ${cost:.2f} in {finished - started}  (total ${spent:.2f})")
        else:
            print(f"  {stage}: done in {finished - started}")

    if paid:
        print(f"\nOpenAI spend this run: ${spent:.2f}")
        print("Review by hand after a re-run: the judge spot check (experiments/judge/review_sample.md) and the\n"
              "narrative in README 'Systematic failure patterns' / report Sections 5-7, which describe error patterns.")


if __name__ == "__main__":
    main()
