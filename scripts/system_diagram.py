import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

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
ORANGE_SOFT = "#FFF7E6"
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


fig, ax = plt.subplots(figsize=(15.0, 9.8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")

outer = FancyBboxPatch(
    (0.35, 0.35),
    14.3,
    9.25,
    boxstyle="round,pad=0.08,rounding_size=0.22",
    facecolor="none",
    edgecolor=ORANGE,
    linewidth=1.9,
    linestyle=(0, (4, 4)),
    zorder=0,
)
ax.add_patch(outer)

pill(ax, 2.2, 9.25, "Agentic Process Map", ORANGE_SOFT, tc="#B45309", ec=ORANGE, fs=7.2, w_override=2.35, h_override=0.38, pad=0.01)
ax.text(7.5, 9.45, "MedVisionQuery — Agentic Multimodal Medical VQA", ha="center", va="center", fontsize=scaled(15.2), color=TEXT, fontweight="bold")
ax.text(7.5, 9.1, "Reimagined layout: left-to-right orchestration with parallel agent workspace", ha="center", va="center", fontsize=scaled(8.7), color=MUTED)

frame(ax, 0.75, 2.2, 2.95, 6.35, "Input Lane", GREEN, fill="#F6FCF8", label_fc=GREEN_SOFT)
box(ax, 2.22, 7.5, 2.35, 0.95, GREEN_SOFT, GREEN, "User", "medical image + question", tc="#14532D", sc="#2F6E43", title_fs=9.6, subtitle_fs=7.0)
box(ax, 2.22, 6.0, 2.7, 1.0, GREEN, GREEN, "React UI", "upload + inspect", tc="white", sc="#E7F8EC", title_fs=9.8, subtitle_fs=6.9)
box(ax, 2.22, 4.55, 2.35, 0.95, PURPLE_SOFT, PURPLE, "FastAPI", "/api/analyze", tc=PURPLE_DARK, sc="#6C61A8", title_fs=9.4, subtitle_fs=6.9)
arrow(ax, (2.22, 7.0), (2.22, 6.55), color=GREEN, lw=2.0)
arrow(ax, (2.22, 5.48), (2.22, 5.03), color=GREEN, lw=2.0)

frame(ax, 4.2, 2.15, 6.75, 6.4, "Agent Workspace", PURPLE, fill="#F8F7FF", label_fc=PURPLE_SOFT)
box(ax, 7.58, 7.6, 3.7, 1.02, PURPLE, PURPLE, "Haystack Agent Controller", "routes tools and manages shared state", tc="white", sc="#F1EDFF", title_fs=10.0, subtitle_fs=7.0)
pill(ax, 9.98, 7.95, "controller", OPENAI, fs=6.2, w_override=1.32, h_override=0.27, pad=0.01)

box(ax, 5.65, 6.15, 2.9, 1.05, TEAL_SOFT, OPENAI, "ImageInterpreter", "modality + findings", title_fs=9.6, subtitle_fs=6.9, lw=2.0)
box(ax, 5.65, 4.45, 2.9, 1.05, PURPLE_SOFT, OPENAI, "QuestionAnalyzer", "intent + answer format", title_fs=9.6, subtitle_fs=6.9, lw=2.0)
box(ax, 8.95, 6.15, 2.9, 1.05, TEAL_SOFT, OPENAI, "ReasoningAgent", "synthesize answer", title_fs=9.6, subtitle_fs=6.9, lw=2.0)
box(ax, 8.95, 4.45, 2.9, 1.05, TEAL_SOFT, OPENAI, "VerificationAgent", "grounding + confidence", title_fs=9.6, subtitle_fs=6.9, lw=2.0)

pill(ax, 7.3, 3.45, "optional", AMBER_SOFT, tc="#8A5A00", ec=AMBER, fs=6.1, w_override=0.9, h_override=0.26, pad=0.01)
label(ax, 7.3, 3.15, "KnowledgeRetriever", color=PURPLE_DARK, fs=6.8, ha="center")

arrow(ax, (3.4, 4.55), (5.95, 7.25), color=SLATE, lw=2.0, rad=0.03)
arrow(ax, (7.0, 7.08), (5.65, 6.72), color=SLATE, lw=2.0)
arrow(ax, (5.65, 5.62), (5.65, 4.98), color=SLATE, lw=2.0)
arrow(ax, (5.65, 6.15), (8.95, 6.15), color=SLATE, lw=2.0)
arrow(ax, (5.9, 4.45), (8.7, 6.0), color=SLATE, lw=2.0, rad=0.04)
arrow(ax, (8.95, 5.62), (8.95, 4.98), color=SLATE, lw=2.0)
arrow(ax, (5.65, 5.0), (7.3, 3.72), color=HF, lw=1.6, dashed=True, rad=-0.05)
arrow(ax, (7.3, 3.72), (8.95, 5.58), color=HF, lw=1.6, dashed=True, rad=-0.03)

frame(ax, 11.35, 2.15, 2.9, 6.4, "Decision", AMBER, fill="#FFFCF2", label_fc=AMBER_SOFT)
box(ax, 12.8, 5.1, 2.25, 1.05, AMBER_SOFT, AMBER, "Guardrail", "verified / disclaimer / review", tc="#8A5A00", sc="#9A6D00", title_fs=9.6, subtitle_fs=6.8)
arrow(ax, (9.95, 4.45), (11.7, 5.1), color=AMBER, lw=2.1, rad=-0.04)

ax.text(8.0, 1.35, "OUTPUT", ha="left", va="center", fontsize=scaled(8.2), color=MUTED, fontweight="bold")
box(ax, 8.6, 1.0, 2.35, 0.95, GREEN_SOFT, GREEN, "Verified", "grounded response", tc="#14532D", sc="#2F6E43", title_fs=9.0, subtitle_fs=6.5)
box(ax, 11.2, 1.0, 2.35, 0.95, AMBER_SOFT, AMBER, "Disclaimer", "guarded response", tc="#8A5A00", sc="#9A6D00", title_fs=9.0, subtitle_fs=6.5)
box(ax, 13.8, 1.0, 2.35, 0.95, RED_SOFT, RED, "Review", "flagged response", tc="#9B1C1C", sc="#B63A3A", title_fs=9.0, subtitle_fs=6.5)
arrow(ax, (12.8, 4.56), (12.8, 3.05), color=AMBER, lw=2.0)
arrow(ax, (12.8, 2.75), (8.6, 1.52), color=GREEN, lw=1.9, rad=0.06)
arrow(ax, (12.8, 2.75), (11.2, 1.52), color=AMBER, lw=1.9, rad=0.04)
arrow(ax, (12.8, 2.75), (13.8, 1.52), color=RED, lw=1.9, rad=-0.04)

label(ax, 7.5, 0.46, "returned to the UI as structured JSON", color=MUTED, fs=6.8, italic=True)

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