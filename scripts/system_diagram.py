import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

BG = "#F8FAFC"
TEXT = "#0F172A"
MUTED = "#64748B"
GREEN = "#3FAE5A"
GREEN_SOFT = "#EAF7EE"
PURPLE = "#6D5BD0"
PURPLE_SOFT = "#F3F0FF"
BLUE = "#4E7AB5"
BLUE_SOFT = "#EEF5FF"
AMBER = "#D18B00"
AMBER_SOFT = "#FFF6DF"
RED = "#D64545"
RED_SOFT = "#FDECEC"
SLATE = "#94A3B8"
SHADOW = "#0F172A"


def shadow_box(ax, x, y, w, h, r=0.18, alpha=0.08, z=1):
    ax.add_patch(FancyBboxPatch(
        (x + 0.08, y - 0.08), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=0, facecolor=SHADOW, alpha=alpha, zorder=z,
    ))


def box(ax, x, y, w, h, fc, ec, title, subtitle=None,
        tc=TEXT, sc=MUTED, lw=1.8, fs1=13, fs2=9, rounded=0.18, z=3):
    shadow_box(ax, x, y, w, h, r=rounded, z=z - 1)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounded}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
    ))
    ty = y + h * 0.62 if subtitle else y + h / 2
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=fs1, color=tc, weight="bold", zorder=z + 1)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=fs2, color=sc, zorder=z + 1)


def pill(ax, x, y, w, h, text, fc, ec=None, tc=TEXT, fs=9, z=4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.3, edgecolor=ec or fc, facecolor=fc, zorder=z,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold", zorder=z + 1)


def frame(ax, x, y, w, h, label, edge, fill=None, lab_fc="white", lab_tc=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        linewidth=2, edgecolor=edge, facecolor=fill or "none",
        linestyle=(0, (4, 3)), zorder=0,
    ))
    pill(ax, x + 0.22, y + h - 0.06, 2.10, 0.40, label,
         lab_fc, edge, lab_tc or edge, fs=9, z=2)


def lbl(ax, x, y, s, fs=9, c=MUTED, weight=None, ha="center", va="center",
        style="normal", z=10):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va,
            weight=weight, style=style, zorder=z)


def arrow(ax, start, end, color=SLATE, lw=2.0, style="-|>", z=5, ls="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0,
    ))


def orth_arrow(ax, points, color=SLATE, lw=2.0, z=5, ls="-"):
    for i in range(len(points) - 2):
        (x1, y1), (x2, y2) = points[i], points[i + 1]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, zorder=z)
    arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls, z=z)


def diamond(ax, cx, cy, w, h, fc, ec, title, subtitle=None,
            tc=TEXT, sc=MUTED, lw=1.8, z=3):
    pts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
    shad = [(x + 0.08, y - 0.08) for x, y in pts]
    ax.add_patch(Polygon(shad, closed=True, facecolor=SHADOW, edgecolor="none",
                         alpha=0.08, zorder=z - 1))
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=z))
    ax.text(cx, cy + 0.15, title, ha="center", va="center",
            fontsize=11.5, color=tc, weight="bold", zorder=z + 1)
    if subtitle:
        ax.text(cx, cy - 0.16, subtitle, ha="center", va="center",
                fontsize=8, color=sc, zorder=z + 1)


def build_diagram(output_path="system_diagram.png"):
    # ── Canvas  (18 × 12 gives ample vertical breathing room) ──────────────
    fig, ax = plt.subplots(figsize=(18, 12), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # ── Outer border ────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.22, 0.06), 17.56, 11.52,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        linewidth=1.8, edgecolor=AMBER, facecolor="none",
        linestyle=(0, (4, 4)), zorder=0,
    ))
    pill(ax, 0.65, 11.15, 2.50, 0.44,
         "Agentic Process Map", AMBER_SOFT, AMBER, "#9A6700", fs=9)
    lbl(ax, 9, 11.18, "MedVisionQuery — Agentic Multimodal Medical VQA",
        fs=19, c=TEXT, weight="bold")
    lbl(ax, 9, 10.80, "interpret + classify → route → optional retrieval → reason → verify",
        fs=10.5, c=MUTED)

    # ── Section frames ──────────────────────────────────────────────────────
    # Input Lane  x 0.40 – 3.60
    frame(ax, 0.40, 1.60, 3.20, 9.00, "Input Lane", GREEN,
          fill="#F8FCF9", lab_fc=GREEN_SOFT)
    # Agent Workspace  x 3.80 – 14.00
    frame(ax, 3.80, 1.60, 10.20, 9.00, "Agent Workspace", PURPLE,
          fill="#FBFAFF", lab_fc=PURPLE_SOFT)
    # Decision  x 14.20 – 17.70
    frame(ax, 14.20, 1.60, 3.50, 9.00, "Decision", AMBER,
          fill="#FFFCF4", lab_fc=AMBER_SOFT)
    # Final Answer States strip — simple border + plain text label
    ax.add_patch(FancyBboxPatch(
        (6.80, 0.08), 10.90, 1.40,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.6, edgecolor=SLATE, facecolor="#FBFCFE",
        linestyle=(0, (4, 3)), zorder=0,
    ))
    lbl(ax, 9.55, 1.35, "Final Answer States",
        fs=9, c=SLATE, weight="bold")

    # ══════════════════════════════════════════════════════
    # INPUT LANE  (centred at x ≈ 2.0)
    # ══════════════════════════════════════════════════════
    box(ax, 0.75, 8.90, 2.50, 1.00, GREEN_SOFT, GREEN,
        "User", "medical image + question",
        tc="#14532D", sc="#2F6E43", fs1=13, fs2=8.7)
    box(ax, 0.65, 7.05, 2.70, 1.10, GREEN, GREEN,
        "React UI", "upload + inspect",
        tc="white", sc="#E8F8EC", fs1=14, fs2=9)
    box(ax, 0.75, 5.35, 2.50, 1.00, PURPLE_SOFT, PURPLE,
        "FastAPI", "/api/analyze",
        tc="#4D3DB5", sc="#6B65A8", fs1=13, fs2=8.7)

    orth_arrow(ax, [(2.00, 8.90), (2.00, 8.15)], color=GREEN)   # User → React UI
    orth_arrow(ax, [(2.00, 7.05), (2.00, 6.35)], color=GREEN)   # React UI → FastAPI

    # ══════════════════════════════════════════════════════
    # AGENT WORKSPACE
    # ══════════════════════════════════════════════════════

    # Controller  (y 8.90 – 10.00)
    box(ax, 5.00, 8.90, 6.80, 1.10, PURPLE, PURPLE,
        "Haystack Agentic Controller",
        "routing, state, tool orchestration, retries",
        tc="white", sc="#ECE7FF", fs1=15, fs2=9)
    pill(ax, 10.55, 9.62, 1.42, 0.28,
         "controller", BLUE, BLUE, "white", fs=7.4, z=6)

    # FastAPI right centre (3.25, 5.85) → up wall of workspace → Controller
    orth_arrow(ax,
               [(3.25, 5.85), (3.75, 5.85), (3.75, 9.45), (5.00, 9.45)],
               color=SLATE)

    # Stage-1 label
    lbl(ax, 8.50, 8.65, "stage 1: sequential understanding",
        fs=8.5, c=PURPLE, weight="bold")

    # Controller → Image Interpreter (straight arrow)
    arrow(ax, (6.10, 8.90), (6.10, 7.85), color=SLATE)

    # Stage-1 boxes  (y 6.75 – 7.85)
    box(ax, 4.30, 6.75, 3.60, 1.10, BLUE_SOFT, BLUE,
        "Image Interpreter", "visual evidence / findings", fs1=13, fs2=8.8)
    pill(ax, 7.28, 7.42, 0.62, 0.24, "auto-block", RED, RED, "white", fs=6.5, z=6)
    box(ax, 9.10, 6.75, 3.60, 1.10, PURPLE_SOFT, BLUE,
        "Question Analyzer", "intent / type / answer format", fs1=13, fs2=8.7)

    # Sequential flow: image interpretation feeds question analysis
    orth_arrow(ax, [(7.90, 7.30), (9.10, 7.30)], color=SLATE)

    # Stage-2 label
    lbl(ax, 8.50, 6.48, "stage 2: routing + synthesis + validation",
        fs=8.5, c=PURPLE, weight="bold")

    # Difficulty Router  x 8.10-13.70, y 4.90-5.75
    box(ax, 8.10, 4.90, 5.60, 0.85, PURPLE_SOFT, PURPLE,
        "Difficulty Router", "sets difficulty · retrieval need · verification policy · fallback",
        tc="#4D3DB5", sc="#6B65A8", fs1=13, fs2=8.3)
    pill(ax, 12.38, 5.47, 1.22, 0.25,
         "planning", PURPLE, PURPLE, "white", fs=7.2, z=6)

    # Q-Analyzer bottom (6.75) → Difficulty Router top (5.75)
    orth_arrow(ax, [(10.90, 6.75), (10.90, 5.75)], color=SLATE)

    # Image Interpreter output also feeds Difficulty Router (mandatory image interpretation input)
    orth_arrow(ax, [(8.45, 5.88), (9.20, 5.88), (9.20, 5.75)], color=SLATE, lw=1.4)
    lbl(ax, 8.82, 5.98, "image interpretation", fs=6.5, c=MUTED, ha="center")

    # Stage-2 boxes  top = 4.40  →  safely below diamond bottom 4.70
    # Knowledge Retriever  x 4.30-7.70, y 3.40-4.40
    box(ax, 4.30, 3.40, 3.40, 1.00, AMBER_SOFT, AMBER,
        "Knowledge Retriever", "conditional enrichment",
        tc="#8A5A00", sc="#9A6D00", fs1=12.3, fs2=8.1)
    # Reasoning Agent  x 9.20-12.60, y 3.40-4.40
    box(ax, 9.20, 3.40, 3.40, 1.00, BLUE_SOFT, BLUE,
        "Reasoning Agent", "uses visual + intent + optional context",
        fs1=12.5, fs2=8.1)
    # Verification Agent  x 9.20-12.60, y 2.15-3.05
    box(ax, 9.20, 2.15, 3.40, 0.90, BLUE_SOFT, BLUE,
        "Verification Agent", "policy-guided verification",
        fs1=11.3, fs2=7.8)
    pill(ax, 12.00, 2.65, 0.60, 0.22, "policy", PURPLE, PURPLE, "white", fs=6.5, z=6)

    # Image Interpreter → Reasoning Agent
    # Corridor at x=8.45 (gap between retriever right 7.70 & reasoning left 9.20)
    orth_arrow(ax,
               [(6.10, 6.75), (6.10, 6.10), (8.45, 6.10),
                (8.45, 3.90), (9.20, 3.90)],
               color=SLATE)

    # Router left (8.10, 5.32) → Knowledge Retriever top — "yes" path
    orth_arrow(ax,
               [(8.10, 5.32), (6.00, 5.32), (6.00, 4.40)],
               color=AMBER, lw=1.9, ls="--")
    lbl(ax, 7.00, 5.54, "yes → retrieve",
        fs=8.1, c="#9A6700", weight="bold")

    # Router bottom (10.90, 4.90) → Reasoning Agent top (10.90, 4.40) — "no" path
    orth_arrow(ax, [(10.90, 4.90), (10.90, 4.40)], color=SLATE, lw=1.9)
    lbl(ax, 11.55, 4.65, "no → skip",
        fs=8.1, c=MUTED, weight="bold", ha="left")

    # Knowledge Retriever right → Reasoning Agent left  (dashed amber)
    orth_arrow(ax, [(7.70, 3.90), (9.20, 3.90)], color=AMBER, lw=1.9, ls="--")

    # Reasoning Agent → Verification Agent
    orth_arrow(ax, [(10.90, 3.40), (10.90, 3.05)], color=SLATE)

    # Router → Verification Agent (policy guidance) - dotted purple line
    orth_arrow(ax, [(11.50, 4.90), (11.50, 3.75), (12.20, 3.75), (12.20, 3.05)], color=PURPLE, lw=1.5, ls=":")
    lbl(ax, 12.60, 3.88, "policy", fs=7, c=PURPLE, style="italic", ha="left")

    # ══════════════════════════════════════════════════════
    # DECISION
    # ══════════════════════════════════════════════════════
    # Guardrail  x 14.40-17.20, y 5.10-6.20
    box(ax, 14.40, 5.10, 2.80, 1.10, AMBER_SOFT, AMBER,
        "Guardrail", "verification + failures\n→ flag / guarded / disclaimer",
        tc="#8A5A00", sc="#9A6D00", fs1=13, fs2=8.0)
    lbl(ax, 15.80, 4.82, "verified / partially / low / unsupported + severity",
        fs=7.5, c=MUTED)

    # Verification Agent right (12.60, 2.60) → Guardrail left (14.40, 5.65)
    orth_arrow(ax,
               [(12.60, 2.60), (13.50, 2.60), (13.50, 5.65), (14.40, 5.65)],
               color=AMBER, lw=2.0)

    # ══════════════════════════════════════════════════════
    # FINAL ANSWER STATES (bottom strip, y 0.08 – 1.43)
    # ══════════════════════════════════════════════════════
    # Error (auto-block for non-radiology images)
    box(ax, 4.30, 0.22, 2.30, 0.70, RED_SOFT, RED,
        "Error", "invalid image",
        tc="#9B1C1C", sc="#B63A3A", fs1=11.2, fs2=7.5)
    # Verified
    box(ax, 6.90, 0.22, 2.30, 0.70, GREEN_SOFT, GREEN,
        "Verified", "answer shown directly",
        tc="#14532D", sc="#2F6E43", fs1=11.2, fs2=7.7)
    # Partially Verified
    box(ax, 9.50, 0.22, 2.30, 0.70, AMBER_SOFT, AMBER,
        "Partially Verified", "some uncertainty",
        tc="#8A5A00", sc="#9A6D00", fs1=11.2, fs2=7.7)
    # Low Confidence
    box(ax, 12.10, 0.22, 2.30, 0.70, RED_SOFT, RED,
        "Low Confidence", "guarded + disclaimer",
        tc="#9B1C1C", sc="#B63A3A", fs1=11.2, fs2=7.5)
    # Review
    box(ax, 14.70, 0.22, 2.30, 0.70, RED_SOFT, RED,
        "Review", "review recommended",
        tc="#9B1C1C", sc="#B63A3A", fs1=11.2, fs2=7.5)

    # Guardrail → horizontal routing rail → five coloured drops
    GX  = 15.80   # guardrail centre-x
    GY  = 5.10    # guardrail bottom-y
    RY  = 1.10    # routing rail y  (below Final Answer States label, above boxes)
    EX  = 5.45    # Error centre-x
    VX  = 8.05    # Verified centre-x
    PX  = 10.65   # Partially Verified centre-x
    DX  = 13.25   # Low Confidence centre-x
    RVX = 15.85   # Review centre-x

    ax.plot([GX,  GX],  [GY, RY], color=AMBER, linewidth=2.0, zorder=5)
    ax.plot([EX,  GX],  [RY, RY], color=SLATE, linewidth=1.8, zorder=5)
    orth_arrow(ax, [(EX,  RY), (EX,  0.92)],  color=RED,   lw=1.9)
    orth_arrow(ax, [(VX,  RY), (VX,  0.92)],  color=GREEN, lw=1.9)
    orth_arrow(ax, [(PX,  RY), (PX,  0.92)],  color=AMBER, lw=1.9)
    orth_arrow(ax, [(DX,  RY), (DX,  0.92)],  color=RED,   lw=1.9)
    orth_arrow(ax, [(RVX, RY), (RVX, 0.92)],  color=RED,   lw=1.9)
    # Image Interpreter auto-block → Error state (route above Knowledge Retriever)
    orth_arrow(ax, [(6.10, 6.75), (6.10, 5.60), (4.20, 5.60), (4.20, 1.10), (EX, 1.10), (EX, 0.92)], color=RED, lw=1.5, ls="--")

    # Footnote — in the left margin below the Input Lane boxes
    lbl(ax, 2.00, 1.20,
        "Router sets policy → Verification uses policy\n"
        "Guardrail = verification_status + failure_categories\n"
        "→ Verified / Partially Verified / Low Confidence / Review / Error",
        fs=7.5, c=MUTED, style="italic", ha="center")

    plt.tight_layout(pad=0.25)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    build_diagram()
