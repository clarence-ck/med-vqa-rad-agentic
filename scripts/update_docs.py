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
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


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


def vs_unverified_n(m: dict) -> int:
    vs = m["verification_signal"]
    return vs["partially_verified"]["cases"] + vs["unsupported_or_low"]["cases"]


def vs_unverified_acc(m: dict) -> float:
    """Agentic accuracy on answered questions the check did not verify (abstentions excluded)."""
    vs = m["verification_signal"]
    n = vs_unverified_n(m)
    right = (vs["partially_verified"]["cases"] * (vs["partially_verified"]["agentic"] or 0)
             + vs["unsupported_or_low"]["cases"] * (vs["unsupported_or_low"]["agentic"] or 0)) / 100
    return round(100 * right / n, 1) if n else 0.0


def derived(m: dict) -> dict:
    """Numbers computed from the metrics that several blocks share."""
    s = m["systems"]
    paths = m["controller_paths"]
    sc = m["self_consistency_signal"]
    stages = m.get("run_timings", {})
    costs = m["costs_usd"]
    return {
        "not_direct": sum(v["cases"] for k, v in paths.items() if k != "direct"),
        "unverified_n": vs_unverified_n(m),
        "unverified_acc": vs_unverified_acc(m),
        "sc_medlow_n": sc["medium"]["cases"] + sc["low"]["cases"],
        "sc_medlow_range": value_range([sc["medium"]["agentic"], sc["low"]["agentic"]]),
        "vs_pipeline_strict": round(s["agentic"]["strict"] - s["pipeline"]["strict"], 1),
        "vs_pipeline_judged": round(s["agentic"]["judged"] - s["pipeline"]["judged"], 1),
        "abstained_baseline_wrong": rounded(100 - paths["abstained"]["baseline"]) if "abstained" in paths else None,
        "total_cost": round(sum(v for v in costs.values() if v), 2),
        "new_question_cost": round(s["agentic"]["mean_cost_usd"] + s["baseline"]["mean_cost_usd"]
                                   + s["workflow"]["mean_cost_usd"], 3),
        "flip_pct": {k: round(100 * v["flips"] / v["cases"]) for k, v in m.get("variance", {}).items()},
        "stage_minutes": {k: round(v["wall_clock_s"] / 60) for k, v in stages.items()},
    }


# ---------------------------------------------------------------------------- README blocks
def gap(diff: float) -> str:
    return f"{abs(diff):.1f} points {'higher' if diff > 0 else 'lower'}"


def sig(p: float) -> str:
    return "significant" if p < 0.05 else "could be chance"


def block_glance(m: dict, d: dict) -> str:
    s, pair = m["systems"], m["paired"]
    rows = "\n".join(
        f"| {SYSTEM_LABELS[k]} | {s[k]['strict']}% | {s[k]['judged']}% | "
        f"{s[k]['mean_time_s']} s · {s[k]['median_time_s']} s | ${s[k]['mean_cost_usd']:.4f} |"
        for k in ("baseline", "workflow", "agentic", "pipeline"))
    ab, ap = pair["agentic_vs_baseline_strict"], pair["agentic_vs_pipeline_strict"]
    abj, apj = pair["agentic_vs_baseline_judged"], pair["agentic_vs_pipeline_judged"]
    flips = d["flip_pct"]
    ahead = s["agentic"]["strict"] > s["baseline"]["strict"]
    if ab["p"] < 0.05 and abj["p"] < 0.05:
        verdict = "agentic ahead, significant on both scores" if ahead else "single-shot ahead, significant on both scores"
    elif abj["p"] < 0.05 or ab["p"] < 0.05:
        which = "judged" if abj["p"] < 0.05 else "strict"
        verdict = f"agentic ahead, significant on the {which} score only" if ahead else f"single-shot ahead, significant on the {which} score only"
    else:
        verdict = "no significant difference"
    if ap["p"] >= 0.05 and apj["p"] >= 0.05:
        pipe_p, pipe_pj, pipe_end = f"p = {p_value(ap['p'])}", f"p = {p_value(apj['p'])}", " Both gaps could be chance."
    else:
        pipe_p, pipe_pj, pipe_end = (f"p = {p_value(ap['p'])}, {sig(ap['p'])}", f"p = {p_value(apj['p'])}, {sig(apj['p'])}", "")
    blind = m.get("blind_reading")
    blind_line = (f"""
- **Agentic vs its verifier model alone:** inside `check`, the verifier model `gpt-6.1-sol` first answers the question
  by itself (the *blind reading*). Scored on its own, that answer reached {blind['strict']}% strict accuracy, against {s['agentic']['strict']}% for the
  full loop (p = {p_value(blind['agentic_vs_blind']['p'])}, {sig(blind['agentic_vs_blind']['p'])}).""" if blind else "")
    verified = m["verification_signal"]["verified"]
    var_cases = next(iter(m.get("variance", {}).values()))["cases"] if m.get("variance") else 0
    return f"""Run `{m['run']['run_id']}` tested {m['cases']} VQA-RAD questions, picked at random with a fixed seed so the same pick can be made
again. The pick keeps the dataset's mix of question kinds: {m['close_ended_cases']} close-ended (yes/no, or one of two terms such as
normal/abnormal) and {m['open_ended_cases']} open-ended. All four systems used the same answering model, `gpt-6-luna`. The agentic loop
and the fixed pipeline also used a second model, `gpt-6.1-sol`, to check answers.

How to read the table:

- **Strict accuracy:** answers are marked by fixed rules. For example, a yes/no answer must start with the right word.
- **Judged accuracy:** the same, except that open-ended answers are marked by a third model, `gpt-6-astra`, which
  decides whether an answer means the same as the reference answer. It is not told which system wrote which answer.
- **Time:** four questions ran at the same time. The {m['controller_paths'].get('relook', {}).get('cases', 0)} questions where the loop took a second, closer look were the
  slowest and pull the mean up; the median is the middle value. Full timings are in [Run timings](#run-timings).

| System | Strict accuracy | Judged accuracy | Time / case (mean · median) | Cost / case |
|---|---|---|---|---|
{rows}

**How to read the p-values.** Every system answered the same questions, so two systems are compared question by
question (exact McNemar test). The p-value is the chance of seeing a gap at least this large if the two systems were
really equally good. Below 0.05, the usual cut-off, the gap is called *significant*: it is unlikely to be luck. Above
0.05, the gap could be chance.

- **Agentic vs single-shot: {verdict}.** The agentic loop scored {abs(s['agentic']['strict'] - s['baseline']['strict']):.1f}
  points {'higher' if s['agentic']['strict'] > s['baseline']['strict'] else 'lower'} on strict accuracy (p = {p_value(ab['p'])}, {sig(ab['p'])}) and {gap(s['agentic']['judged'] - s['baseline']['judged'])} on judged accuracy (p = {p_value(abj['p'])},
  {sig(abj['p'])}).
- **Agentic vs fixed pipeline:** the agentic loop scored {gap(d['vs_pipeline_strict'])} on strict accuracy ({pipe_p}) and {abs(d['vs_pipeline_judged']):.1f}
  points {'higher' if d['vs_pipeline_judged'] > 0 else 'lower'} on judged accuracy ({pipe_pj}).{pipe_end}{blind_line}
- **Confidence and stability:** answers the loop marked as *verified* were right {rounded(verified['agentic'])}% of the time; its other answers
  (abstentions not counted) were right {rounded(d['unverified_acc'])}% of the time. When the same {var_cases} questions were run a second time, {flips.get('agentic', '–')}% of
  the loop's answers changed between right and wrong, against {flips.get('baseline', '–')}% for single-shot."""


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
             "judge": "LLM judge (`gpt-6-astra`)"}
    cost_of = {"eval": c.get("evaluation_run"), "variance": c.get("variance_repeats"), "judge": c.get("judge")}
    rows = "\n".join(
        f"| {names.get(k, k)} | {v['started'].replace('T', ' ')} | {v['finished'][11:]} | {minutes(v['wall_clock_s'])} | "
        f"{v.get('note', '').replace(' x ', ' × ')} | {'$' + format(cost_of[k], '.2f') if cost_of.get(k) is not None else '–'} |"
        for k, v in stages.items() if k in names)
    return f"""- **Cost:** every OpenAI call is recorded by `backend/utils/usage.py`. The evaluation cost ${c['evaluation_run']:.2f}, the repeats ${c['variance_repeats']:.2f}
  and the judging ${c['judge']:.2f}, a total of ${d['total_cost']:.2f}.

### Run timings

How long each stage of run `{m['run']['run_id']}` took, as recorded by `scripts/run_all.py` in
`experiments/runs/{m['run']['run_id']}/run.json`:

| Stage | Started | Finished | Wall clock | Work | Cost |
|---|---|---|---|---|---|
{rows}

The per-case times in the results tables are each system's own response time. The four systems ran side by side,
so one whole case took {m['case_wall_time_mean_s']} seconds on average."""


def block_results(m: dict, d: dict) -> str:
    s, pair, paths = m["systems"], m["paired"], m["controller_paths"]
    label = {"baseline": "Single-shot baseline", "workflow": "Workflow baseline", "agentic": "**Agentic loop**",
             "pipeline": "Fixed pipeline (ablation)"}
    overall = "\n".join(
        f"| {label[k]} | {s[k]['strict']}% | {s[k]['judged']}% | {s[k]['close_ended']}% | "
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
    first = pair["agentic_vs_baseline_strict"]
    return f"""### Overall

| System | Strict | Judged | Close-ended ({m['close_ended_cases']}) | Open-ended ({m['open_ended_cases']}), strict → judged |
|---|---|---|---|---|
{overall}

Head-to-head comparisons. "{first['only_first']} vs {first['only_second']}" means {first['only_first']} questions only the first system got right and {first['only_second']} only the second got
right; questions both got right, or both got wrong, are left out. A p-value below 0.05 means the gap is significant.

| Comparison | Strict | Judged |
|---|---|---|
| Agentic vs single-shot | {pair_cell('agentic_vs_baseline_strict')} | {pair_cell('agentic_vs_baseline_judged')} |
| Agentic vs fixed pipeline | {pair_cell('agentic_vs_pipeline_strict')} | {pair_cell('agentic_vs_pipeline_judged')} |

### By controller path

The path is the route the loop took on a question (see [Controller loop](#controller-loop)). The other columns show
how the other systems did on the same questions.

| Path | Cases | Agentic | Single-shot | Fixed pipeline |
|---|---|---|---|---|
{path_rows}

### Uncertainty signals predict errors

These signals tell you how far to trust an answer: answers with a good signal were right much more often.

| Signal | Levels | Agentic accuracy |
|---|---|---|
| Verification (answered questions) | verified ({vs['verified']['cases']}) / not verified ({d['unverified_n']}) | {rounded(vs['verified']['agentic'])}% / {rounded(d['unverified_acc'])}% |
| Self-consistency | high ({sc['high']['cases']}) / medium or low ({d['sc_medlow_n']}) | {rounded(sc['high']['agentic'])}% / {d['sc_medlow_range']}% |
| Abstention | {ab.get('cases', 0)} cases; the single-shot baseline was wrong on {d['abstained_baseline_wrong']}% of them | — |

### Stability across runs (same {next(iter(var.values()))['cases'] if var else 0} cases)

Strict scoring. "Original evaluation" is the answer from the main {m['cases']}-question evaluation. "Independent repeat" is a
fresh answer to the same question, made later in the same run.

| System | Original evaluation | Independent repeat | Answers that flipped right ↔ wrong |
|---|---|---|---|
{var_rows}{block_blind_zoom(m)}"""


def block_blind_zoom(m: dict) -> str:
    """Only for runs with the blind check and zoom tool."""
    b, z = m.get("blind_reading"), m.get("zoom") or {}
    if not b:
        return ""
    ab, bb = b["agentic_vs_blind"], b["blind_vs_baseline"]
    mech = m.get("mechanisms") or {}
    if not mech:
        return ""
    flag_pct = rounded(100 * mech["blind_flagged_and_wrong"] / mech["blind_flagged_first_answer"]) if mech["blind_flagged_first_answer"] else "–"
    return f"""

### What the loop's mechanisms did

| Mechanism | Result in this run |
|---|---|
| Revision (re-look or re-answer) | Used on {mech['revised']} questions. It turned {mech['revised_fixed']} wrong first answers into right ones and {mech['revised_broken']} right first answers into wrong ones (strict scoring). |
| Blind check | Disagreed with {mech['blind_flagged_first_answer']} first answers, and {mech['blind_flagged_and_wrong']} of those ({flag_pct}%) were indeed wrong. For {mech['blind_decided_by_meaning']} answers the word rules could not tell whether the two agreed, so the verifier compared their meanings. |
| Zoom | Used on {z.get('episodes', 0)} questions; {z.get('episodes', 0) - z.get('cases', 0)} of them later ended in abstention. On the other {z.get('cases', 0)}, the loop was right {z.get('agentic')}% of the time and single-shot {z.get('baseline')}%. |
| Abstention | Used on {mech['abstained']} questions. The loop's last answer before abstaining would have been right in {mech['abstained_own_answer_right']} of them; single-shot was right in {mech['abstained_baseline_right']}. |
| Answered questions only | On the {mech['answered']} questions the loop answered, it was right {mech['answered_agentic_strict']:.1f}% of the time and single-shot {mech['answered_baseline_strict']:.1f}%. |

The blind reading is the verifier model, `gpt-6.1-sol`, answering straight from the image. Scored on its own, it
reached {b['strict']}% strict accuracy. Against the loop: {ab['only_first']} vs {ab['only_second']} (p = {p_value(ab['p'])}, {sig(ab['p'])}). Against single-shot: {bb['only_first']} vs {bb['only_second']}
(p = {p_value(bb['p'])}, {sig(bb['p'])})."""


PLAIN_STORY = {
    "relook_win": "A closer second look got it right; single-shot got it wrong",
    "reanswer_win": "An answer rewritten on the verifier's advice got it right; single-shot got it wrong",
    "recovery_failure": "The loop looked again or rewrote its answer and still ended wrong, while single-shot was right",
    "flagged_correct": "The extra answers disagreed, so the case was marked for review, but the final answer was right",
}


def plain_story(c: dict) -> str:
    if c["kind"] == "recovery_failure" and c["story"].startswith("An intermediate"):
        return ("An earlier answer was right, but a later rewrite replaced it with a wrong one"
                + (", and the check approved the wrong one" if c.get("verification_status") == "verified" else "")
                + ". Single-shot was right")
    return PLAIN_STORY.get(c["kind"], c["story"])


def block_failure(m: dict, d: dict) -> str:
    sc = m["self_consistency_signal"]
    wins = m["wins"]
    cases = "\n".join(
        f"| {c['sample_id']} \"{short(c['question'], 50)}\" | {plain_story(c)}. Agentic: \"{short(c['agentic_answer'], 40)}\"; "
        f"single-shot: \"{short(c['baseline_answer'], 40)}\"; reference: \"{short(c['ground_truth'], 30)}\". |"
        for c in m.get("showcase", []))
    return f"""- **Self-consistency checks:** the answerer gives two extra answers of its own for each case. If they disagree, the
  case is marked for review. Disagreement means "uncertain", not "wrong": {rounded(sc['low']['agentic'])}% of low-consistency answers were still
  right.
- **Cross-modal agreement:** the verifier looks at the image itself. It checks whether the explanation matches what is
  visible and whether the answer contradicts the image, instead of only reading a text description.
- **Verifier model:** a different, stronger model, `gpt-6.1-sol`, checks the answers of `gpt-6-luna`. After its
  checks, {d['not_direct']} of the {m['cases']} cases left the direct path.
- **Disagreement analysis:** compared with the same rules used for scoring, the three systems' answers did not all
  match on {rounded(m['disagreement_pct'])}% of cases. Where only one system was right, it was the agentic loop in {wins['agentic']} cases, single-shot in {wins['baseline']} and
  the workflow in {wins['workflow']}.

These examples were picked automatically by `scripts/final_metrics.py`. Screenshots are in
`{SCREENSHOTS}/`.

| Case | What happened |
|---|---|
{cases}"""


def block_limit_cost(m: dict, d: dict) -> str:
    r = m["agentic_vs_baseline"]
    return (f"- The agentic loop costs about {rounded(r['cost_ratio'])} times more than a single prompt, and its median response time is about "
            f"{rounded(r['median_latency_ratio'])} times\n  longer.")


def block_question_cost(m: dict, d: dict) -> str:
    return (f"Open http://localhost:5173. Each new question runs all three systems and costs about "
            f"${d['new_question_cost']:.3f} with the default models.")


def block_reproduce(m: dict, d: dict) -> str:
    c, mins = m["costs_usd"], d["stage_minutes"]
    return f"""```bash
python scripts/run_all.py --dry-run         # lists the stages and estimated cost; spends nothing
python scripts/run_all.py                   # runs evaluation, repeats, judging, metrics, docs, figures and report
python scripts/run_all.py --stages metrics,docs,diagram,report   # rebuilds outputs from saved results, at no cost
```

| Stage | What it runs | Last cost | Last wall clock |
|---|---|---|---|
| `eval` | `run_case_explorer_cache.py --ablation --concurrency 4` ({m['cases']} questions × 4 systems) | ${c['evaluation_run']:.2f} | ~{mins.get('eval', '?')} min |
| `variance` | `repeat_cases.py --from-eval-sample {m['cases']} --n {next(iter(m.get('variance', {}).values()), {}).get('cases', 100)}` (baseline + agentic loop again) | ${c['variance_repeats']:.2f} | ~{mins.get('variance', '?')} min |
| `judge` | `judge_open_ended.py` | ${c['judge']:.2f} | ~{mins.get('judge', '?')} min |
| `metrics` | `final_metrics.py` → `final_metrics.json` and `cases.csv` in the run folder | free | seconds |
| `docs` | `update_docs.py` → README sections and `docs/report/results.tex` | free | seconds |
| `diagram` | `system_diagram.py` → `docs/report/system_diagram.png` for the README and report | free | seconds |
| `screenshots` | Starts the backend and frontend, then runs `capture-screenshots.mjs` for example cases | free | ~1 min |
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
        # 95% interval half-width for an accuracy near the agentic loop's, in points
        "CIHalf": f"{196 * (s['agentic']['strict'] / 100 * (1 - s['agentic']['strict'] / 100) / m['cases']) ** 0.5:.1f}",
        "BlindStrict": (m.get("blind_reading") or {}).get("strict", "--"),
        "BlindAgP": p_value((m.get("blind_reading") or {}).get("agentic_vs_blind", {}).get("p", 1.0)).replace("< ", "<"),
        "ZoomN": (m.get("zoom") or {}).get("cases") or 0,
        "ZoomEpisodes": (m.get("zoom") or {}).get("episodes") or 0,
        "PostHoc": (m.get("mechanisms") or {}).get("posthoc_defer_to_blind_strict", "--"),
        "ZoomAg": rounded((m.get("zoom") or {}).get("agentic") or 0),
        "ZoomBase": rounded((m.get("zoom") or {}).get("baseline") or 0),
        "RevN": (m.get("mechanisms") or {}).get("revised", 0),
        "RevFixed": (m.get("mechanisms") or {}).get("revised_fixed", 0),
        "RevBroken": (m.get("mechanisms") or {}).get("revised_broken", 0),
        "BlindFlagN": (m.get("mechanisms") or {}).get("blind_flagged_first_answer", 0),
        "BlindFlagWrong": (m.get("mechanisms") or {}).get("blind_flagged_and_wrong", 0),
        "MeaningN": (m.get("mechanisms") or {}).get("blind_decided_by_meaning", 0),
        "AbstainOwnRight": (m.get("mechanisms") or {}).get("abstained_own_answer_right", 0),
        "AbstainBaseRight": (m.get("mechanisms") or {}).get("abstained_baseline_right", 0),
        "AnsweredAgAcc": rounded((m.get("mechanisms") or {}).get("answered_agentic_strict") or 0),
        "AnsweredBaseAcc": rounded((m.get("mechanisms") or {}).get("answered_baseline_strict") or 0),
        "UnverN": d["unverified_n"],
        "UnverAcc": rounded(d["unverified_acc"]),
        "VsBaseStrict": f"{s['agentic']['strict'] - s['baseline']['strict']:+.1f}",
        "VsBaseJudged": f"{s['agentic']['judged'] - s['baseline']['judged']:+.1f}",
    })
    lines = [f"% Generated by scripts/update_docs.py from run {m['run']['run_id']} -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    return "\n".join(lines) + "\n"


def runs_index() -> str:
    lines = ["# Evaluation runs", "",
             "Each evaluation has its own folder. Starting a new evaluation does not change an older run. This file is generated",
             "by `scripts/update_docs.py`. The root README and report use the newest completed run.", "",
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
