"""Write every evaluation number in the README and the report from the current run's final_metrics.json.

- README.md: replaces the text between <!-- generated:NAME --> and <!-- /generated:NAME --> markers.
- docs/report/results.tex: LaTeX macros (\\AgStrict, \\PairBaseP, ...) that main.tex uses for every number.
- experiments/runs/README.md: an index of every evaluation run.

Nothing in these blocks should be edited by hand; change this script instead.

Usage (run from repo root, after scripts/final_metrics.py):
    python scripts/update_docs.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "backend"))

from utils import runs  # noqa: E402

README = ROOT_DIR / "README.md"
RUNS_INDEX = runs.RUNS_DIR / "README.md"
RESULTS_TEX = ROOT_DIR / "docs" / "report" / "results.tex"
SCREENSHOTS = "eval-dashboard-screenshots"

SYSTEM_LABELS = {
    "baseline": "Single-shot baseline",
    "workflow": "Workflow baseline (hand-designed)",
    "agentic": "**Agentic controller loop**",
    "pipeline": "Fixed pipeline (ablation)",
}


def p_value(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def minutes(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"


def rounded(value: float) -> str:
    """Whole percent, rounding halves up (Python's format rounds 0.5 to even)."""
    return str(int(value + 0.5))


def value_range(values: list[float]) -> str:
    low, high = rounded(min(values)), rounded(max(values))
    return low if low == high else f"{low}–{high}"


def short(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def derived(m: dict) -> dict:
    """Numbers computed from the metrics that several blocks share."""
    s = m["systems"]
    paths = m["controller_paths"]
    sc = m["self_consistency_signal"]
    stages = m.get("run_timings", {})
    costs = m["costs_usd"]
    return {
        "not_direct": sum(v["cases"] for k, v in paths.items() if k != "direct"),
        "sc_medlow_n": sc["medium"]["cases"] + sc["low"]["cases"],
        "sc_medlow_range": value_range([sc["medium"]["agentic"], sc["low"]["agentic"]]),
        "vs_pipeline_strict": round(s["agentic"]["strict"] - s["pipeline"]["strict"], 1),
        "vs_pipeline_judged": round(s["agentic"]["judged"] - s["pipeline"]["judged"], 1),
        "abstained_baseline_wrong": rounded(100 - paths["abstained"]["baseline"]) if "abstained" in paths else None,
        "total_cost": round(sum(v for v in costs.values() if v), 2),
        "new_question_cost": round(s["agentic"]["mean_cost_usd"] + s["baseline"]["mean_cost_usd"]
                                   + s["workflow"]["mean_cost_usd"], 2),
        "flip_pct": {k: round(100 * v["flips"] / v["cases"]) for k, v in m.get("variance", {}).items()},
        "stage_minutes": {k: round(v["wall_clock_s"] / 60) for k, v in stages.items()},
    }


# ---------------------------------------------------------------------------- README blocks
def block_glance(m: dict, d: dict) -> str:
    s, pair = m["systems"], m["paired"]
    rows = "\n".join(
        f"| {SYSTEM_LABELS[k]} | {'**' if k == 'agentic' else ''}{s[k]['strict']}%{'**' if k == 'agentic' else ''} | "
        f"{'**' if k == 'agentic' else ''}{s[k]['judged']}%{'**' if k == 'agentic' else ''} | "
        f"{s[k]['mean_time_s']} s · {s[k]['median_time_s']} s | ${s[k]['mean_cost_usd']:.4f} |"
        for k in ("baseline", "workflow", "agentic", "pipeline"))
    ab, ap = pair["agentic_vs_baseline_strict"], pair["agentic_vs_pipeline_strict"]
    abj, apj = pair["agentic_vs_baseline_judged"], pair["agentic_vs_pipeline_judged"]
    tie = ab["p"] >= 0.05 and abj["p"] >= 0.05
    flips = d["flip_pct"]
    return f"""Run `{m['run']['run_id']}`: {m['cases']} seeded-random VQA-RAD test questions ({m['close_ended_cases']} close-ended, {m['open_ended_cases']} open-ended), all systems on the same answering model
(`gpt-6-luna`; the agentic loop and the fixed pipeline also use `gpt-6.1-sol` as verifier). Per-case times were
measured with four cases running at once; the mean is pulled up by a few long re-look cases (actual run times are in
[Run timings](#run-timings)). "Strict" uses the rule-based scorer; "judged" also accepts open-ended paraphrases
confirmed by a blinded LLM judge (see [Evaluation design](#evaluation-design)).

| System | Strict accuracy | Judged accuracy | Time / case (mean · median) | Cost / case |
|---|---|---|---|---|
{rows}

- **Agentic vs single-shot: {'a tie' if tie else 'a significant difference'}** (exact McNemar p = {p_value(ab['p'])} strict, {p_value(abj['p'])} judged).
- **Agentic vs fixed pipeline:** {d['vs_pipeline_strict']:+} points strict (p = {p_value(ap['p'])}), {d['vs_pipeline_judged']:+} judged (p = {p_value(apj['p'])}).
- **Reliability:** verification and self-consistency signals predict errors, the loop can abstain, and
  {flips.get('agentic', '–')}% of its answers flip between runs vs {flips.get('baseline', '–')}% for the single prompt."""


def block_paths(m: dict, d: dict) -> str:
    sequences = {
        "direct": "analyze → look → answer → check → finish",
        "relook": "… → check → look(focus) → answer(guided) → check → finish",
        "reanswer": "… → check → answer(guided) → check → finish",
        "abstained": "… → check → finish(abstain)",
    }
    rows = "\n".join(f"| {k} | {v['cases']} | {sequences.get(k, '')} |" for k, v in m["controller_paths"].items())
    return f"""| Path | Cases | Tool sequence |
|---|---|---|
{rows}"""


def block_cost(m: dict, d: dict) -> str:
    c = m["costs_usd"]
    stages = m.get("run_timings", {})
    names = {"eval": "Main evaluation (4 systems)", "variance": "Variance repeats (baseline + agentic)",
             "judge": "LLM judge (`gpt-6.1-sol`)"}
    cost_of = {"eval": c.get("evaluation_run"), "variance": c.get("variance_repeats"), "judge": c.get("judge")}
    rows = "\n".join(
        f"| {names.get(k, k)} | {v['started'].replace('T', ' ')} | {v['finished'][11:]} | {minutes(v['wall_clock_s'])} | "
        f"{v.get('note', '')} | {'$' + format(cost_of[k], '.2f') if cost_of.get(k) is not None else '–'} |"
        for k, v in stages.items() if k in names)
    return f"""- **Cost:** every OpenAI call is recorded (`backend/utils/usage.py`): evaluation ${c['evaluation_run']:.2f}, repeats
  ${c['variance_repeats']:.2f}, judge ${c['judge']:.2f} (${d['total_cost']:.2f} in total).

### Run timings

Wall-clock time of each stage of run `{m['run']['run_id']}`, recorded by `scripts/run_all.py`
(`experiments/runs/{m['run']['run_id']}/run.json`):

| Stage | Started | Finished | Wall clock | Work | Cost |
|---|---|---|---|---|---|
{rows}

Per-case times in the results tables are each system's own latency inside a case; a full case (all four systems in
turn) took {m['case_wall_time_mean_s']} s on average."""


def block_results(m: dict, d: dict) -> str:
    s, pair, paths = m["systems"], m["paired"], m["controller_paths"]
    label = {"baseline": "Single-shot baseline", "workflow": "Workflow baseline", "agentic": "**Agentic loop**",
             "pipeline": "Fixed pipeline (ablation)"}
    def bold(k, text):
        return f"**{text}**" if k == "agentic" else text

    overall = "\n".join(
        f"| {label[k]} | {bold(k, str(s[k]['strict']) + '%')} | {bold(k, str(s[k]['judged']) + '%')} | {s[k]['close_ended']}% | "
        f"{s[k]['open_ended']}% → {s[k]['open_ended_judged']}% |" for k in label)

    def pair_cell(key):
        v = pair[key]
        return f"{v['only_first']} vs {v['only_second']}, p = {p_value(v['p'])}"

    path_rows = "\n".join(
        f"| {k} | {v['cases']} | {'— (abstained)' if k == 'abstained' else rounded(v['agentic']) + '%'} | "
        f"{rounded(v['baseline'])}% | {rounded(v['pipeline'])}% |" for k, v in paths.items())
    vs, sc = m["verification_signal"], m["self_consistency_signal"]
    var = m.get("variance", {})
    var_rows = "\n".join(
        f"| {'Single-shot baseline' if k == 'baseline' else 'Agentic loop'} | {v['run1']}/{v['cases']} | "
        f"{v['run2']}/{v['cases']} | {v['flips']} ({d['flip_pct'][k]}%) |" for k, v in var.items())
    ab = paths.get("abstained", {})
    return f"""### Overall

| System | Strict | Judged | Close-ended ({m['close_ended_cases']}) | Open-ended ({m['open_ended_cases']}), strict → judged |
|---|---|---|---|---|
{overall}

Paired comparisons (cases where exactly one of the two systems is right):

| Comparison | Strict | Judged |
|---|---|---|
| Agentic vs single-shot | {pair_cell('agentic_vs_baseline_strict')} | {pair_cell('agentic_vs_baseline_judged')} |
| Agentic vs fixed pipeline | {pair_cell('agentic_vs_pipeline_strict')} | {pair_cell('agentic_vs_pipeline_judged')} |

### By controller path

| Path | Cases | Agentic | Single-shot | Fixed pipeline |
|---|---|---|---|---|
{path_rows}

![Accuracy by controller path]({SCREENSHOTS}/03-accuracy-by-controller-path.png)

### Uncertainty signals predict errors

| Signal | Levels | Agentic accuracy |
|---|---|---|
| Verification | verified ({vs['verified']['cases']}) / partially verified ({vs['partially_verified']['cases']}) / unsupported or low confidence ({vs['unsupported_or_low']['cases']}) | {rounded(vs['verified']['agentic'])}% / {rounded(vs['partially_verified']['agentic'])}% / {rounded(vs['unsupported_or_low']['agentic'])}% |
| Self-consistency | high ({sc['high']['cases']}) / medium or low ({d['sc_medlow_n']}) | {rounded(sc['high']['agentic'])}% / {d['sc_medlow_range']}% |
| Abstention | {ab.get('cases', 0)} cases, on which the single-shot baseline is wrong {d['abstained_baseline_wrong']}% of the time | — |

On the {m['agentic_answered']['cases']} cases it answered, the agentic loop is {rounded(m['agentic_answered']['strict'])}% correct (strict).

### Stability across runs (same {next(iter(var.values()))['cases'] if var else 0} cases)

| System | Run 1 | Run 2 | Answers that flipped right ↔ wrong |
|---|---|---|---|
{var_rows}{block_blind_zoom(m)}"""


def block_blind_zoom(m: dict) -> str:
    """Only for runs with the blind check and zoom tool."""
    b, z = m.get("blind_reading"), m.get("zoom") or {}
    if not b:
        return ""
    ab, bb = b["agentic_vs_blind"], b["blind_vs_baseline"]
    zoom_line = (f"Zoomed re-looks happened on {z['cases']} cases (agentic {z['agentic']}% vs single-shot {z['baseline']}% there)."
                 if z.get("cases") else "No re-look used the zoom in this run.")
    return f"""

### Blind check and zoom

The check's blind reading is the verifier model (`gpt-6.1-sol`) answering from the image alone. Scored as a system
of its own it reaches {b['strict']}% strict: a reference for how much of the agentic result could come from the
larger verifier model (agentic vs blind reading {ab['only_first']} vs {ab['only_second']}, p = {p_value(ab['p'])}; blind reading vs
single-shot {bb['only_first']} vs {bb['only_second']}, p = {p_value(bb['p'])}). {zoom_line}"""


def block_failure(m: dict, d: dict) -> str:
    sc = m["self_consistency_signal"]
    wins = m["wins"]
    cases = "\n".join(
        f"| {c['sample_id']} \"{short(c['question'], 50)}\" | {c['story']} (agentic: \"{short(c['agentic_answer'], 40)}\"; "
        f"single prompt: \"{short(c['baseline_answer'], 40)}\"; reference: \"{short(c['ground_truth'], 30)}\") |"
        for c in m.get("showcase", []))
    return f"""- **Self-consistency checks:** 2 independent re-answers per case; low agreement flags the case for review. Low
  consistency marks uncertainty, not certain error: {rounded(sc['low']['agentic'])}% of low-consistency answers were still correct.
- **Cross-modal agreement:** the verifier sees the image, so "explanation grounded in evidence" and "no
  contradictions" are checked against the image itself, not only a text description.
- **Verifier model:** a different, larger model (`gpt-6.1-sol`) verifies `gpt-6-luna`'s answers and steers
  {d['not_direct']} of {m['cases']} episodes away from the direct path.
- **Disagreement analysis:** the three systems disagree on {rounded(m['disagreement_pct'])}% of cases under the same matching rules;
  wins (only one system right): agentic {wins['agentic']}, single-shot {wins['baseline']}, workflow {wins['workflow']}.

Representative cases, chosen automatically by `scripts/final_metrics.py` (screenshots in `{SCREENSHOTS}/`):

| Case | What happened |
|---|---|
{cases}"""


def block_limit_cost(m: dict, d: dict) -> str:
    r = m["agentic_vs_baseline"]
    return (f"- About {rounded(r['cost_ratio'])}× the cost and {rounded(r['median_latency_ratio'])}× the (median) latency "
            f"of a single prompt for similar accuracy.")


def block_question_cost(m: dict, d: dict) -> str:
    return (f"Open http://localhost:5173. Each new question runs all three systems and costs about "
            f"${d['new_question_cost']:.2f} with the default models.")


def block_reproduce(m: dict, d: dict) -> str:
    c, mins = m["costs_usd"], d["stage_minutes"]
    return f"""```bash
python scripts/run_all.py --dry-run         # show the stages and the estimated cost; spends nothing
python scripts/run_all.py                   # everything: evaluation, repeats, judge, metrics, docs, figures, report
python scripts/run_all.py --stages metrics,docs,diagram,report   # rebuild outputs from cached results (free)
```

| Stage | What it runs | Last cost | Last wall clock |
|---|---|---|---|
| `eval` | `run_case_explorer_cache.py --ablation --concurrency 4` ({m['cases']} questions × 4 systems) | ${c['evaluation_run']:.2f} | ~{mins.get('eval', '?')} min |
| `variance` | `repeat_cases.py --from-eval-sample {m['cases']} --n 50` (baseline + agentic loop again) | ${c['variance_repeats']:.2f} | ~{mins.get('variance', '?')} min |
| `judge` | `judge_open_ended.py` | ${c['judge']:.2f} | ~{mins.get('judge', '?')} min |
| `metrics` | `final_metrics.py` → `final_metrics.json` and `cases.csv` in the run folder | free | seconds |
| `docs` | `update_docs.py` → README blocks and `docs/report/results.tex` | free | seconds |
| `diagram` | `system_diagram.py` → `system_diagram.png` (+ copy for the report) | free | seconds |
| `screenshots` | starts backend + frontend, `capture-screenshots.mjs` for the showcase cases | free | ~1 min |
| `report` | `tectonic main.tex` → `docs/report/MGAI_Final_Report_Group9_revised.pdf` | free | ~1 min |"""


BLOCKS = {
    "glance": block_glance,
    "paths": block_paths,
    "cost": block_cost,
    "results": block_results,
    "failure": block_failure,
    "limit-cost": block_limit_cost,
    "question-cost": block_question_cost,
    "reproduce": block_reproduce,
}


def update_readme(m: dict, d: dict) -> list[str]:
    text = README.read_text(encoding="utf-8")
    missing = []
    for name, render in BLOCKS.items():
        pattern = re.compile(rf"(<!-- generated:{name} -->\n).*?(\n<!-- /generated:{name} -->)", re.S)
        if not pattern.search(text):
            missing.append(name)
            continue
        text = pattern.sub(lambda match: match.group(1) + render(m, d) + match.group(2), text)
    README.write_text(text, encoding="utf-8", newline="\n")
    return missing


# ---------------------------------------------------------------------------- LaTeX macros
def latex_macros(m: dict, d: dict) -> str:
    s, pair, paths = m["systems"], m["paired"], m["controller_paths"]
    vs, sc, c = m["verification_signal"], m["self_consistency_signal"], m["costs_usd"]
    r = m["agentic_vs_baseline"]
    short_name = {"baseline": "Base", "workflow": "Wf", "agentic": "Ag", "pipeline": "Pl"}
    macros = {"NCases": m["cases"], "NClose": m["close_ended_cases"], "NOpen": m["open_ended_cases"]}
    for k, n in short_name.items():
        v = s[k]
        macros.update({
            f"{n}Strict": v["strict"], f"{n}Judged": v["judged"], f"{n}Close": v["close_ended"],
            f"{n}Open": v["open_ended"], f"{n}OpenJ": v["open_ended_judged"], f"{n}Tmean": v["mean_time_s"],
            f"{n}Tmed": v["median_time_s"], f"{n}Cost": f"{v['mean_cost_usd']:.4f}",
        })
    for key, n in (("agentic_vs_baseline_strict", "PairBase"), ("agentic_vs_baseline_judged", "PairBaseJ"),
                   ("agentic_vs_pipeline_strict", "PairPl"), ("agentic_vs_pipeline_judged", "PairPlJ")):
        macros.update({f"{n}A": pair[key]["only_first"], f"{n}B": pair[key]["only_second"],
                       f"{n}P": p_value(pair[key]["p"]).replace("< ", "<")})
    for path, n in (("direct", "Direct"), ("relook", "Relook"), ("reanswer", "Reanswer"), ("abstained", "Abstain")):
        v = paths.get(path, {"cases": 0, "agentic": 0, "baseline": 0, "pipeline": 0})
        macros.update({f"Path{n}N": v["cases"], f"Path{n}Ag": rounded(v["agentic"]),
                       f"Path{n}Base": rounded(v["baseline"]), f"Path{n}Pl": rounded(v["pipeline"])})
    macros.update({
        "SigVerN": vs["verified"]["cases"], "SigVerAcc": rounded(vs["verified"]["agentic"]),
        "SigPartN": vs["partially_verified"]["cases"], "SigPartAcc": rounded(vs["partially_verified"]["agentic"]),
        "SigLowN": vs["unsupported_or_low"]["cases"], "SigLowAcc": rounded(vs["unsupported_or_low"]["agentic"]),
        "ScHighN": sc["high"]["cases"], "ScHighAcc": rounded(sc["high"]["agentic"]),
        "ScMedLowN": d["sc_medlow_n"], "ScMedLowRange": d["sc_medlow_range"].replace("–", "--"),
        "VarCases": next(iter(m["variance"].values()))["cases"] if m.get("variance") else 0,
        "VarBaseFlip": d["flip_pct"].get("baseline", 0), "VarAgFlip": d["flip_pct"].get("agentic", 0),
        "DisagreePct": rounded(m["disagreement_pct"]), "WinsAg": m["wins"]["agentic"],
        "WinsBase": m["wins"]["baseline"], "WinsWf": m["wins"]["workflow"], "NotDirect": d["not_direct"],
        "VsPlStrict": f"{d['vs_pipeline_strict']:+}", "VsPlJudged": f"{d['vs_pipeline_judged']:+}",
        "AnsweredN": m["agentic_answered"]["cases"], "AnsweredAcc": rounded(m["agentic_answered"]["strict"]),
        "CostRun": f"{c['evaluation_run']:.2f}", "CostRepeats": f"{c['variance_repeats']:.2f}",
        "CostJudge": f"{c['judge']:.2f}", "CostTotal": f"{d['total_cost']:.2f}",
        "CostRatio": rounded(r["cost_ratio"]), "LatRatio": rounded(r["median_latency_ratio"]),
        "MinEval": d["stage_minutes"].get("eval", 0), "MinRepeats": d["stage_minutes"].get("variance", 0),
        "MinJudge": d["stage_minutes"].get("judge", 0),
        "RunId": m["run"]["run_id"].replace("_", "\\_"),
        "BlindStrict": (m.get("blind_reading") or {}).get("strict", "--"),
        "BlindAgP": p_value((m.get("blind_reading") or {}).get("agentic_vs_blind", {}).get("p", 1.0)).replace("< ", "<"),
        "ZoomN": (m.get("zoom") or {}).get("cases") or 0,
    })
    lines = [f"% Generated by scripts/update_docs.py from run {m['run']['run_id']} -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    return "\n".join(lines) + "\n"


def runs_index() -> str:
    lines = ["# Evaluation runs", "",
             "One folder per run; a later run never changes an earlier one. Generated by `scripts/update_docs.py`.",
             "The README and the report show the current run (the latest completed one).", "",
             "| Run | Status | Code | Wall clock | Cost | Agentic strict / judged | Single-shot strict / judged |",
             "|---|---|---|---|---|---|---|"]
    for run in reversed(runs.list_runs()):
        metrics_file = runs.run_dir(run["run_id"]) / "final_metrics.json"
        m = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
        stages = run.get("stages", {})
        wall = sum(v.get("wall_clock_s", 0) for v in stages.values())
        cost = sum(v.get("cost_usd", 0) or 0 for v in stages.values())
        s = m.get("systems", {})
        acc = lambda k: f"{s[k]['strict']}% / {s[k]['judged']}%" if k in s else "–"  # noqa: E731
        lines.append(f"| [`{run['run_id']}`]({run['run_id']}/) | {run.get('status', '?')} | "
                     f"`{run.get('code_version', '?')}` | {minutes(wall)} | ${cost:.2f} | {acc('agentic')} | {acc('baseline')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    run_id = runs.current_run_id()
    metrics_file = runs.run_dir(run_id) / "final_metrics.json" if run_id else None
    if not metrics_file or not metrics_file.exists():
        sys.exit("No final_metrics.json for the current run: run scripts/final_metrics.py first")
    m = json.loads(metrics_file.read_text(encoding="utf-8"))
    d = derived(m)
    missing = update_readme(m, d)
    RESULTS_TEX.write_text(latex_macros(m, d), encoding="utf-8", newline="\n")
    RUNS_INDEX.write_text(runs_index(), encoding="utf-8", newline="\n")
    print(f"Updated README.md ({len(BLOCKS) - len(missing)} blocks), {RESULTS_TEX.relative_to(ROOT_DIR)} and "
          f"{RUNS_INDEX.relative_to(ROOT_DIR)} from run {run_id}")
    if missing:
        print(f"README has no markers for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
