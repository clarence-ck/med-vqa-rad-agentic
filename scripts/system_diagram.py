"""Regenerate system_diagram.png: the agentic controller loop and the evaluation setup.

Usage (from repo root): python scripts/system_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "system_diagram.png"

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

    # ---- Agentic loop --------------------------------------------------------------
    frame(ax, 3.8, 3.6, 10.4, 6.5, "Agentic loop (Haystack Agent, one per request)", PURPLE)
    box(ax, 6.6, 8.55, 4.8, 1.05, PURPLE_SOFT, PURPLE, "Controller  ·  gpt-6-luna",
        "decides the next tool call from the tool results")
    path(ax, [(3.1, 5.6), (3.6, 5.6), (3.6, 9.05), (6.6, 9.05)], color=GREEN, label="question",
         label_xy=(5.0, 9.05), lc=GREEN)

    tools = [
        (4.1, "analyze_question()", "type + expected format\ngpt-6-luna", BLUE_SOFT, BLUE),
        (6.1, "look(focus)", "question-aware vision;\nre-examine a region\ngpt-6-luna", BLUE_SOFT, BLUE),
        (8.1, "answer(guidance)", "image + all observations\ngpt-6-luna", BLUE_SOFT, BLUE),
        (10.1, "check()", "verifies against the image;\nrecommends next action\ngpt-6.1-sol", AMBER_SOFT, AMBER),
        (12.1, "finish(abstain)", "ends the episode", PURPLE_SOFT, PURPLE),
    ]
    for x, title, sub, fc, ec in tools:
        box(ax, x, 6.25, 1.85, 1.45, fc, ec, title, sub, fs1=9.2, fs2=7.6)
        arrow(ax, (9.0, 8.55), (x + 0.92, 7.7), color=PURPLE, lw=1.4)

    # check -> next action
    path(ax, [(10.75, 6.25), (10.75, 5.75), (9.3, 5.75), (9.3, 6.25)], color=AMBER,
         label="reanswer(guidance)", label_xy=(10.0, 5.75), lc=AMBER)
    path(ax, [(11.35, 6.25), (11.35, 5.2), (7.0, 5.2), (7.0, 6.25)], color=AMBER,
         label="relook(focus)", label_xy=(8.2, 5.2), lc=AMBER)
    path(ax, [(11.65, 6.25), (11.65, 5.75), (13.0, 5.75), (13.0, 6.25)], color=GREEN,
         label="accept / abstain", label_xy=(12.55, 5.75), lc=GREEN)

    box(ax, 4.1, 3.95, 3.4, 0.95, "white", SLATE, "Shared image + working memory",
        "looks, answers, checks (per request)", fs1=9.5, fs2=7.8)
    ax.text(13.95, 4.1, "budgets: ≤ 2 focused looks, ≤ 3 answers,\n≤ 16 agent steps; safety net checks the\nlatest answer if the controller stalls",
            ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")

    # ---- Output / guardrails -----------------------------------------------------------
    frame(ax, 14.6, 3.6, 4.1, 6.5, "Decision & safety", RED)
    box(ax, 14.9, 8.4, 3.5, 1.1, RED_SOFT, RED, "Best-verified answer", "a later answer replaces an earlier\none only if its check is better")
    box(ax, 14.9, 6.9, 3.5, 1.1, RED_SOFT, RED, "Self-consistency", "2 independent re-answers;\nlow agreement → review flag")
    box(ax, 14.9, 5.4, 3.5, 1.1, RED_SOFT, RED, "Guardrail", "failure categories, disclaimer,\nflag for review")
    box(ax, 14.9, 3.9, 3.5, 1.1, "white", RED, "Final state",
        "verified · partially verified · low confidence\n· review · abstained · rejected (non-radiology)", fs2=7.6)
    path(ax, [(13.95, 6.95), (14.45, 6.95), (14.45, 8.95), (14.9, 8.95)], color=RED)
    for y0, y1 in [(8.4, 8.0), (6.9, 6.5), (5.4, 5.0)]:
        arrow(ax, (16.65, y0), (16.65, y1), color=RED)
    path(ax, [(14.9, 4.2), (14.4, 4.2), (14.4, 3.3), (3.6, 3.3), (3.6, 5.2), (3.1, 5.2)], color=SLATE,
         ls=(0, (4, 3)), label="answer + explanation + trace + signals → UI", label_xy=(9.0, 3.3))

    # ---- Evaluation ----------------------------------------------------------------------
    frame(ax, 3.8, 0.5, 14.9, 2.45, "Evaluation (scripts/run_case_explorer_cache.py, 200 random VQA-RAD test cases)", BLUE)
    evals = [
        (4.1, "Single-shot baseline", "one vision prompt · gpt-6-luna"),
        (7.05, "Workflow baseline", "fixed rules over image +\nquestion analysis"),
        (10.0, "Fixed pipeline (ablation)", "original six stages,\nreasoner sees image"),
        (12.95, "Strict scorer + LLM judge", "content-word rules;\nblinded gpt-6.1-sol judge"),
        (15.9, "Dashboard", "accuracy, controller paths,\nsignals, disagreement"),
    ]
    for x, title, sub in evals:
        box(ax, x, 0.8, 2.55, 1.35, BLUE_SOFT, BLUE, title, sub, fs1=9.8, fs2=7.8)

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
