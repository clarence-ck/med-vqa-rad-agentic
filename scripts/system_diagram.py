"""Regenerate docs/report/system_diagram.png (used by the README and the report): the agentic loop and evaluation.

Usage (from repo root): python scripts/system_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "report" / "system_diagram.png"

BG = "#F8FAFC"
TEXT = "#0F172A"
MUTED = "#64748B"
GREEN, GREEN_SOFT = "#3FAE5A", "#EAF7EE"
PURPLE, PURPLE_SOFT = "#6D5BD0", "#F3F0FF"
BLUE, BLUE_SOFT = "#4E7AB5", "#EEF5FF"
AMBER, AMBER_SOFT = "#D18B00", "#FFF6DF"
RED, RED_SOFT = "#D64545", "#FDECEC"
SLATE = "#94A3B8"


def box(ax, x, y, w, h, fc, ec, title, subtitle=None, fs1=12, fs2=8.5, z=3):
    ax.add_patch(FancyBboxPatch((x + 0.06, y - 0.06), w, h, boxstyle="round,pad=0.02,rounding_size=0.16",
                                linewidth=0, facecolor=TEXT, alpha=0.07, zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.16",
                                linewidth=1.7, edgecolor=ec, facecolor=fc, zorder=z))
    ty = y + h * 0.64 if subtitle else y + h / 2
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs1, color=TEXT, weight="bold", zorder=z + 1)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.3, subtitle, ha="center", va="center", fontsize=fs2, color=MUTED,
                zorder=z + 1, linespacing=1.25)


def frame(ax, x, y, w, h, label, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.22",
                                linewidth=2, edgecolor=edge, facecolor="none", linestyle=(0, (4, 3)), zorder=0))
    ax.text(x + 0.25, y + h - 0.05, f"  {label}  ", ha="left", va="center", fontsize=10, color=edge, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=edge, lw=1.3), zorder=2)


def arrow(ax, start, end, color=SLATE, lw=1.8, ls="-", rad=0.0, label=None, label_xy=None, lc=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=5, shrinkA=2, shrinkB=2))
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.5, color=lc, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none"), zorder=6)


def path(ax, points, color=SLATE, lw=1.8, ls="-", label=None, label_xy=None, lc=MUTED):
    """Orthogonal polyline ending in an arrowhead."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, linestyle=ls, zorder=5, solid_capstyle="round")
    arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls)
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=8.5, color=lc, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none"), zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(19, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.text(9.5, 10.6, "MedVisionQuery — agentic controller loop for medical VQA", ha="center", fontsize=17,
            weight="bold", color=TEXT)

    # ---- Input lane ----------------------------------------------------------------
    frame(ax, 0.3, 0.5, 3.1, 9.6, "Input & UI", GREEN)
    box(ax, 0.6, 8.2, 2.5, 1.0, GREEN_SOFT, GREEN, "User", "radiology image\n+ question")
    box(ax, 0.6, 6.5, 2.5, 1.2, GREEN_SOFT, GREEN, "React UI", "Dashboard · Case Explorer\n· Evaluation")
    box(ax, 0.6, 4.9, 2.5, 1.0, GREEN_SOFT, GREEN, "FastAPI", "/api/analyze\n/api/dataset/*")
    arrow(ax, (1.85, 8.2), (1.85, 7.7))
    arrow(ax, (1.85, 6.5), (1.85, 5.9))

    # ---- Agentic loop: one agent (the controller) + its tools ---------------------------
    frame(ax, 3.8, 3.6, 10.4, 6.5, "Agentic loop: ONE Haystack Agent per request", PURPLE)
    box(ax, 4.1, 6.3, 3.0, 1.75, PURPLE_SOFT, PURPLE, "Controller (the agent)",
        "gpt-6-luna\nreads every tool result and\nchooses the next tool call", fs1=11, fs2=8.2)
    path(ax, [(3.1, 5.6), (3.6, 5.6), (3.6, 7.15), (4.1, 7.15)], color=GREEN, label="question",
         label_xy=(3.6, 6.4), lc=GREEN)
    # the loop: call a tool, get the result back, decide again
    arrow(ax, (4.75, 8.05), (6.45, 8.05), color=PURPLE, lw=2, rad=-1.1)
    ax.text(5.6, 9.25, "repeat until finish (≤ 16 steps)", ha="center", va="center", fontsize=8.5,
            color=PURPLE, weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none"), zorder=6)
    arrow(ax, (7.1, 7.55), (8.55, 7.55), color=PURPLE, lw=2, label="call a tool", label_xy=(7.82, 7.85), lc=PURPLE)
    arrow(ax, (8.55, 6.8), (7.1, 6.8), color=PURPLE, lw=2, label="result", label_xy=(7.82, 6.5), lc=PURPLE)

    ax.add_patch(FancyBboxPatch((8.6, 4.95), 5.25, 4.75, boxstyle="round,pad=0.02,rounding_size=0.18",
                                linewidth=1.4, edgecolor=SLATE, facecolor="white", zorder=1))
    ax.text(11.22, 9.45, "Tools: functions the controller calls (not agents)", ha="center", va="center",
            fontsize=9.5, color=TEXT, weight="bold", zorder=4)
    tools = [
        ("analyze_question()", "question type + expected answer format · gpt-6-luna", BLUE_SOFT, BLUE),
        ("look(focus)", "examine the image, or zoom into one region · gpt-6-luna", BLUE_SOFT, BLUE),
        ("answer(guidance)", "answer from image + all observations · gpt-6-luna", BLUE_SOFT, BLUE),
        ("check() · gpt-6.1-sol", "blind reading first, then review; suggests the next step", AMBER_SOFT, AMBER),
        ("finish(abstain)", "end with the latest checked answer, or abstain", PURPLE_SOFT, PURPLE),
    ]
    for i, (title, sub, fc, ec) in enumerate(tools):
        y = 8.45 - i * 0.85
        h = 0.72
        ax.add_patch(FancyBboxPatch((8.8, y), 4.85, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(8.95, y + h * 0.68, title, ha="left", va="center", fontsize=9.2, color=TEXT, weight="bold", zorder=4)
        ax.text(8.95, y + h * 0.28, sub, ha="left", va="center", fontsize=7.5, color=MUTED, zorder=4)

    box(ax, 4.1, 4.15, 3.0, 1.0, "white", SLATE, "Shared image + working memory",
        "looks, answers, checks (per request)", fs1=9.2, fs2=7.8)
    ax.text(8.65, 4.58, "Example episode: analyze → look → answer → check (partial; relook 'left base')\n"
            "→ look(focus) → answer → check (verified) → finish",
            ha="left", va="center", fontsize=7.8, color=TEXT, zorder=4, linespacing=1.3)
    ax.text(8.65, 3.95, "budgets: ≤ 2 focused looks, ≤ 3 answers; a safety net checks the\nlatest answer if the controller stalls",
            ha="left", va="center", fontsize=7.2, color=MUTED, style="italic")

    # ---- Output / guardrails -----------------------------------------------------------
    frame(ax, 14.6, 3.6, 4.1, 6.5, "Decision & safety", RED)
    box(ax, 14.9, 8.4, 3.5, 1.1, RED_SOFT, RED, "Best-verified answer", "a later answer replaces an earlier\none only if its check is better")
    box(ax, 14.9, 6.9, 3.5, 1.1, RED_SOFT, RED, "Self-consistency", "2 independent re-answers;\nlow agreement → review flag")
    box(ax, 14.9, 5.4, 3.5, 1.1, RED_SOFT, RED, "Guardrail", "failure categories, disclaimer,\nflag for review")
    box(ax, 14.9, 3.9, 3.5, 1.1, "white", RED, "Final state",
        "verified · partially verified · low confidence\n· review · abstained · rejected (non-radiology)", fs2=7.6)
    path(ax, [(13.65, 5.5), (14.45, 5.5), (14.45, 8.95), (14.9, 8.95)], color=RED, label="final", label_xy=(14.45, 7.2), lc=RED)
    for y0, y1 in [(8.4, 8.0), (6.9, 6.5), (5.4, 5.0)]:
        arrow(ax, (16.65, y0), (16.65, y1), color=RED)
    path(ax, [(14.9, 4.2), (14.4, 4.2), (14.4, 3.3), (3.6, 3.3), (3.6, 5.2), (3.1, 5.2)], color=SLATE,
         ls=(0, (4, 3)), label="answer + explanation + trace + signals → UI", label_xy=(9.0, 3.3))

    # ---- Evaluation ----------------------------------------------------------------------
    frame(ax, 3.8, 0.5, 14.9, 2.45, "Evaluation (scripts/run_all.py: the VQA-RAD questions in experiments/eval_set.json)", BLUE)
    evals = [
        (4.1, "Single-shot baseline", "one vision prompt · gpt-6-luna"),
        (7.05, "Workflow baseline", "fixed rules over image +\nquestion analysis"),
        (10.0, "Fixed pipeline (ablation)", "six tools in a\nprescribed order"),
        (12.95, "Strict scorer + LLM judge", "content-word rules; blinded\ngpt-6-astra judge (open-ended)"),
        (15.9, "Dashboard", "accuracy, controller paths,\nsignals, disagreement"),
    ]
    for x, title, sub in evals:
        box(ax, x, 0.8, 2.55, 1.35, BLUE_SOFT, BLUE, title, sub, fs1=9.8, fs2=7.8)

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
