import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

SCALE = 1.24
BG = "#F8FAFC"
TEXT = "#101828"
MUTED = "#5B6576"
PURPLE = "#6D5BD0"
PURPLE_DARK = "#4D3DB5"
PURPLE_SOFT = "#F1EEFF"
OPENAI = "#4E81C9"
HF = "#9A84D8"
GREEN = "#3FAE5A"
GREEN_SOFT = "#E8F8ED"
AMBER = "#E5A100"
AMBER_SOFT = "#FFF4CC"
RED = "#D64545"
RED_SOFT = "#FDECEC"
SLATE = "#5B7280"
SLATE_SOFT = "#EAF0F4"
ORANGE = "#F59E0B"
TEAL = "#4E7AB5"
TEAL_SOFT = "#EAF2FF"
SHADOW = "#0F172A"


def scaled(value):
    return value * SCALE


def shadow_round(ax, x, y, w, h, radius=0.18, z=1):
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.08, y - 0.08),
            w,
            h,
            boxstyle=f"round,pad=0.08,rounding_size={radius}",
            facecolor=SHADOW,
            edgecolor="none",
            alpha=0.08,
            zorder=z,
        )
    )


def shadow_circle(ax, cx, cy, r, z=1):
    ax.add_patch(
        Circle(
            (cx + 0.08, cy - 0.08),
            r,
            facecolor=SHADOW,
            edgecolor="none",
            alpha=0.08,
            zorder=z,
        )
    )


def pill(ax, cx, cy, text, fc, tc="white", ec=None, fs=7.6, z=6, w_override=None, h_override=None, pad=0.05):
    ec = ec or fc
    w = w_override if w_override is not None else max(1.22, 0.095 * len(text) + 0.55)
    h = h_override if h_override is not None else 0.40
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle=f"round,pad={pad},rounding_size=0.18",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
            zorder=z,
        )
    )
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=scaled(fs),
        color=tc,
        fontweight="bold",
        zorder=z + 1,
    )


def frame(ax, x, y, w, h, label_text, edge, fill="none", label_fc="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.2",
            facecolor=fill,
            edgecolor=edge,
            linewidth=2.1,
            linestyle="--",
            zorder=0,
        )
    )
    pill(ax, x + 1.85, y + h + 0.16, label_text, label_fc, tc=edge, ec=edge, fs=7.8, z=3)


def box(
    ax,
    cx,
    cy,
    w,
    h,
    fc,
    ec,
    title,
    subtitle=None,
    title_fs=10.5,
    subtitle_fs=8.0,
    tc=TEXT,
    sc=MUTED,
    lw=2.0,
    ls="-",
    align="center",
    shadow=True,
    z=3,
):
    x = cx - w / 2
    y = cy - h / 2
    if shadow:
        shadow_round(ax, x, y, w, h, z=z - 1)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            zorder=z,
        )
    )
    if align == "left":
        tx = x + 0.26
        ha = "left"
    else:
        tx = cx
        ha = "center"
    ty = cy + (0.15 if subtitle else 0)
    ax.text(
        tx,
        ty,
        title,
        ha=ha,
        va="center",
        fontsize=scaled(title_fs),
        color=tc,
        fontweight="bold",
        zorder=z + 1,
    )
    if subtitle:
        ax.text(
            tx,
            cy - 0.19,
            subtitle,
            ha=ha,
            va="center",
            fontsize=scaled(subtitle_fs),
            color=sc,
            zorder=z + 1,
        )
    return x, y, w, h


def circle_node(
    ax,
    cx,
    cy,
    r,
    fc,
    ec,
    title,
    subtitle=None,
    title_fs=10.6,
    subtitle_fs=7.8,
    tc=TEXT,
    sc=MUTED,
    z=3,
):
    shadow_circle(ax, cx, cy, r, z=z - 1)
    ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=z))
    ty = cy + (0.20 if subtitle else 0)
    ax.text(
        cx,
        ty,
        title,
        ha="center",
        va="center",
        fontsize=scaled(title_fs),
        color=tc,
        fontweight="bold",
        zorder=z + 1,
    )
    if subtitle:
        ax.text(
            cx,
            cy - 0.24,
            subtitle,
            ha="center",
            va="center",
            fontsize=scaled(subtitle_fs),
            color=sc,
            zorder=z + 1,
        )


def diamond(
    ax,
    cx,
    cy,
    w,
    h,
    fc,
    ec,
    title,
    subtitle=None,
    title_fs=10.0,
    subtitle_fs=7.8,
    tc=TEXT,
    sc=MUTED,
    z=4,
):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    shadow_pts = [(x + 0.08, y - 0.08) for x, y in pts]
    ax.add_patch(
        Polygon(shadow_pts, closed=True, facecolor=SHADOW, edgecolor="none", alpha=0.08, zorder=z - 1)
    )
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=z))
    ty = cy + (0.14 if subtitle else 0)
    ax.text(
        cx,
        ty,
        title,
        ha="center",
        va="center",
        fontsize=scaled(title_fs),
        color=tc,
        fontweight="bold",
        zorder=z + 1,
    )
    if subtitle:
        ax.text(
            cx,
            cy - 0.20,
            subtitle,
            ha="center",
            va="center",
            fontsize=scaled(subtitle_fs),
            color=sc,
            zorder=z + 1,
        )


def arrow(ax, start, end, color="#64748B", lw=2.4, rad=0.0, style="-|>", dashed=False, alpha=1.0, z=5):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=16,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)


def label(ax, x, y, text, color=MUTED, fs=7.8, ha="center", va="center", italic=False, z=6):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=scaled(fs),
        color=color,
        fontstyle="italic" if italic else "normal",
        zorder=z,
    )


fig, ax = plt.subplots(figsize=(16.6, 18.8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 18)
ax.axis("off")

ax.text(
    8,
    17.45,
    "MedVisionQuery — Agentic Multimodal Medical VQA",
    ha="center",
    va="center",
    fontsize=scaled(14.8),
    color=TEXT,
    fontweight="bold",
)
ax.text(
    8,
    17.00,
    "Controller-centric orchestration with shared state, guardrails, and baseline comparisons",
    ha="center",
    va="center",
    fontsize=scaled(8.6),
    color=MUTED,
)

box(ax, 8, 16.10, 4.1, 0.84, GREEN_SOFT, GREEN, "User", "medical image + question", tc="#14532D", sc="#2F6E43")
box(ax, 8, 14.92, 6.05, 0.98, GREEN, GREEN, "React UI", "Dashboard · Case Explorer · Evaluation", tc="white", sc="#E7F8EC")
arrow(ax, (8, 15.67), (8, 15.36), color=GREEN, lw=2.5)

frame(ax, 1.0, 5.22, 14.0, 9.03, "Agentic orchestration", PURPLE, fill="#FBF9FF", label_fc="#F5F2FF")
ax.add_patch(
    Ellipse(
        (8, 10.00),
        11.2,
        6.4,
        facecolor="none",
        edgecolor="#E6DEFF",
        linewidth=1.7,
        linestyle=(0, (7, 5)),
        zorder=1,
    )
)
label(
    ax,
    8,
    14.02,
    "Controller dynamically selects tools and reads from shared state",
    color=PURPLE_DARK,
    fs=8.1,
    italic=True,
)

box(ax, 8, 13.35, 5.45, 1.10, PURPLE, PURPLE, "Haystack Agent Controller", "OpenAI model · planner and router", tc="white", sc="#F1EDFF", lw=2.2)
arrow(ax, (8, 14.42), (8, 13.95), color=PURPLE, lw=2.45)

img_x, img_y, img_w, img_h = box(
    ax,
    8,
    11.90,
    5.0,
    1.14,
    TEAL_SOFT,
    OPENAI,
    "ImageInterpreter",
    "extracts modality, region, and findings",
    align="left",
    lw=2.1,
)
pill(ax, img_x + img_w - 1.02, img_y + img_h - 0.24, "OpenAI Vision", OPENAI)

qa_x, qa_y, qa_w, qa_h = box(
    ax,
    3.95,
    10.40,
    4.65,
    1.10,
    PURPLE_SOFT,
    OPENAI,
    "QuestionAnalyzer",
    "question type · target · answer format",
    align="left",
    lw=2.1,
)
pill(ax, qa_x + qa_w - 0.86, qa_y + qa_h - 0.24, "OpenAI", OPENAI)

kr_x, kr_y, kr_w, kr_h = box(
    ax,
    12.05,
    10.40,
    4.65,
    1.10,
    PURPLE_SOFT,
    OPENAI,
    "KnowledgeRetriever",
    "medical background context",
    align="left",
    lw=2.1,
    ls="--",
)
pill(ax, kr_x + kr_w - 0.86, kr_y + kr_h - 0.24, "OpenAI", OPENAI)
pill(ax, kr_x + kr_w - 0.50, kr_y + 0.20, "optional", AMBER_SOFT, tc="#8A5A00", ec=AMBER, fs=5.8, w_override=0.64, h_override=0.22, pad=0.005)

circle_node(ax, 8, 9.68, 1.18, "white", PURPLE, "Shared State", "image · question\nintermediate outputs", tc=PURPLE_DARK, sc="#6C61A8")

ra_x, ra_y, ra_w, ra_h = box(
    ax,
    4.55,
    7.78,
    4.9,
    1.14,
    TEAL_SOFT,
    OPENAI,
    "ReasoningAgent",
    "synthesizes answer + explanation",
    align="left",
    lw=2.1,
)
pill(ax, ra_x + ra_w - 0.80, ra_y + ra_h - 0.24, "OpenAI", OPENAI)

va_x, va_y, va_w, va_h = box(
    ax,
    11.45,
    7.78,
    5.05,
    1.14,
    TEAL_SOFT,
    OPENAI,
    "VerificationAgent",
    "checks grounding, confidence, and safety",
    align="left",
    lw=2.1,
)
pill(ax, va_x + va_w - 0.82, va_y + va_h - 0.24, "OpenAI", OPENAI)

arrow(ax, (8, 12.78), (8, 12.46), color=PURPLE, lw=2.2)
arrow(ax, (6.95, 12.90), (5.62, 10.92), color=PURPLE, lw=2.1, rad=0.12)
arrow(ax, (9.05, 12.90), (10.38, 10.92), color=PURPLE, lw=2.1, rad=-0.12)

arrow(ax, (8, 11.30), (8, 10.98), color=SLATE, lw=2.0)
arrow(ax, (6.35, 10.08), (7.02, 9.98), color=SLATE, lw=2.0, rad=-0.06)
arrow(ax, (9.65, 10.08), (8.98, 9.98), color=SLATE, lw=2.0, rad=0.06)
label(ax, 12.08, 11.38, "skip for simple questions", color=AMBER, fs=7.2, italic=True, ha="left")

arrow(ax, (7.02, 8.95), (5.82, 8.28), color=SLATE, lw=2.1, rad=0.1)
arrow(ax, (8.98, 8.95), (10.18, 8.28), color=SLATE, lw=2.1, rad=-0.1)
label(ax, 8, 8.78, "shared evidence and context", color=SLATE, fs=7.2, italic=True)

arrow(ax, (7.20, 7.78), (8.80, 7.78), color=SLATE, lw=2.15)
label(ax, 8, 8.16, "draft answer", color=SLATE, fs=7.0, italic=True)

diamond(ax, 8, 6.05, 2.5, 1.38, AMBER_SOFT, AMBER, "Guardrail", "select final response", tc="#8A5A00", sc="#9A6D00")
arrow(ax, (11.10, 7.22), (8.92, 6.30), color=SLATE, lw=2.25)

arrow(ax, (13.72, 8.08), (10.15, 13.10), color=RED, lw=2.05, rad=0.38, dashed=True, alpha=0.95)
label(ax, 13.1, 10.50, "retry / revise", color=RED, fs=7.1, italic=True, ha="left")

box(ax, 3.2, 4.62, 3.18, 0.94, GREEN_SOFT, GREEN, "Verified Answer", "high-confidence grounded response", tc="#14532D", sc="#2F6E43")
box(ax, 8.0, 4.62, 3.34, 0.94, AMBER_SOFT, AMBER, "Answer + Disclaimer", "guarded response under uncertainty", tc="#8A5A00", sc="#9A6D00")
box(ax, 12.8, 4.62, 3.28, 0.94, RED_SOFT, RED, "Flag for Review", "unsupported or risky response", tc="#9B1C1C", sc="#B63A3A")

arrow(ax, (7.15, 5.65), (4.10, 4.98), color=GREEN, lw=2.15, rad=0.05)
arrow(ax, (8, 5.40), (8, 5.10), color=AMBER, lw=2.15)
arrow(ax, (8.85, 5.65), (11.90, 4.98), color=RED, lw=2.15, rad=-0.05)

box(ax, 8, 2.90, 5.35, 0.98, TEAL, TEAL, "Case Explorer Cache + Evaluation", "metrics, disagreement, close/open-ended accuracy", tc="white", sc="#E9F2FF")
arrow(ax, (8, 5.35), (8, 3.47), color=TEAL, lw=2.25)

frame(ax, 1.0, 0.45, 14.0, 2.20, "Non-agentic baselines", SLATE, fill="#F7FAFC", label_fc="#F8FBFD")

b1_x, b1_y, b1_w, b1_h = box(
    ax,
    4.1,
    1.52,
    5.02,
    1.10,
    "white",
    OPENAI,
    "Baseline 1 — Single-shot",
    "one multimodal prompt · direct answer",
    align="left",
    lw=2.0,
)
pill(ax, b1_x + b1_w - 0.90, b1_y + b1_h - 0.24, "OpenAI", OPENAI)

b2_x, b2_y, b2_w, b2_h = box(
    ax,
    11.9,
    1.52,
    5.16,
    1.10,
    "white",
    SLATE,
    "Baseline 2 — Workflow",
    "deterministic rules · no synthesis LLM",
    align="left",
    lw=2.0,
)
pill(ax, b2_x + b2_w - 0.98, b2_y + b2_h - 0.24, "Rules", SLATE)

arrow(ax, (6.15, 14.48), (4.12, 2.05), color=SLATE, lw=1.65, rad=0.33, dashed=True, alpha=0.85)
arrow(ax, (9.85, 14.48), (11.88, 2.05), color=SLATE, lw=1.65, rad=-0.33, dashed=True, alpha=0.85)

arrow(ax, (4.82, 2.04), (6.75, 2.84), color=TEAL, lw=1.95, rad=-0.04)
arrow(ax, (11.18, 2.04), (9.25, 2.84), color=TEAL, lw=1.95, rad=0.04)

legend_items = [
    mpatches.Patch(color=OPENAI, label="OpenAI — vision, reasoning, verification, orchestration"),
    mpatches.Patch(color=GREEN, label="grounded / verified response"),
    mpatches.Patch(color=AMBER, label="guarded answer with disclaimer"),
    mpatches.Patch(color=RED, label="flagged for review"),
]
ax.legend(
    handles=legend_items,
    loc="lower right",
    fontsize=scaled(7.8),
    framealpha=0.95,
    bbox_to_anchor=(1.0, -0.01),
)

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

out_dir = os.path.join(base_dir, "docs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "system_diagram.png")

plt.tight_layout(pad=0.6)
plt.savefig(out_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved ->", os.path.abspath(out_path))
plt.show()