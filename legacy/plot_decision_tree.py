"""
plot_decision_tree.py
Draws the artefact-detection decision tree and saves it as an SVG vector image.

Metrics are computed lazily — each "Compute X" process box appears immediately
before the decision that needs it, rather than all upfront.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── colour palette ────────────────────────────────────────────────────────────
C_PROC  = "#4A90D9"   # process / compute box   (blue)
C_START = "#1A5276"   # start / end              (dark blue)
C_DECIS = "#F5A623"   # decision diamond         (amber)
C_CAL   = "#7B68EE"   # calibration              (slate blue)
C_FLUSH = "#E74C3C"   # flush                    (red)
C_SLING = "#E67E22"   # slinger                  (orange)
C_GAS   = "#27AE60"   # gasbubble                (green)
C_TRANS = "#8E44AD"   # transducer hoog          (purple)
C_INF   = "#16A085"   # infuus op CVD            (teal)
C_NONE  = "#95A5A6"   # no artefact / end        (grey)
C_NOTE  = "#ECF0F1"   # note background          (light grey)
WHITE   = "white"
DARK    = "#2C3E50"

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 28))
ax.set_xlim(0, 18)
ax.set_ylim(0, 28)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")


# ── drawing helpers ───────────────────────────────────────────────────────────

def box(cx, cy, w, h, text, color, fontsize=8.0, radius=0.2, text_color=WHITE):
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.0, edgecolor=WHITE, facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", multialignment="center", zorder=4)


def diamond(cx, cy, w, h, text, fontsize=7.8):
    xs = [cx, cx+w/2, cx, cx-w/2, cx]
    ys = [cy+h/2, cy, cy-h/2, cy, cy+h/2]
    ax.fill(xs, ys, color=C_DECIS, zorder=3)
    ax.plot(xs, ys, color=WHITE, lw=1.0, zorder=4)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=WHITE,
            fontweight="bold", multialignment="center", zorder=5)


def arrow_v(x, y1, y2, label="", side="right"):
    """Vertical arrow from y1 down to y2."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=DARK,
                                lw=1.3, mutation_scale=13), zorder=2)
    if label:
        ox = 0.22 if side == "right" else -0.22
        ax.text(x + ox, (y1+y2)/2, label,
                ha="center", va="center", fontsize=7.5,
                color=DARK, fontweight="bold", zorder=5)


def arrow_h(x1, x2, y, label="", side="above"):
    """Horizontal arrow from x1 to x2."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=DARK,
                                lw=1.3, mutation_scale=13), zorder=2)
    if label:
        oy = 0.18 if side == "above" else -0.18
        ax.text((x1+x2)/2, y+oy, label,
                ha="center", va="center", fontsize=7.5,
                color=DARK, fontweight="bold", zorder=5)


def elbow_left(x_from, y_from, x_to, y_to, label=""):
    """Go left then down: horizontal to x_to, then vertical to y_to."""
    ax.plot([x_from, x_to], [y_from, y_from],
            color=DARK, lw=1.3, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    ax.annotate("", xy=(x_to, y_to), xytext=(x_to, y_from),
                arrowprops=dict(arrowstyle="-|>", color=DARK,
                                lw=1.3, mutation_scale=13), zorder=2)
    if label:
        ax.text(x_from - 0.2, y_from + 0.18, label,
                ha="right", va="center", fontsize=7.5,
                color=DARK, fontweight="bold", zorder=5)


def elbow_right(x_from, y_from, x_to, y_to, label=""):
    """Go right then down."""
    ax.plot([x_from, x_to], [y_from, y_from],
            color=DARK, lw=1.3, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    ax.annotate("", xy=(x_to, y_to), xytext=(x_to, y_from),
                arrowprops=dict(arrowstyle="-|>", color=DARK,
                                lw=1.3, mutation_scale=13), zorder=2)
    if label:
        ax.text(x_from + 0.2, y_from + 0.18, label,
                ha="left", va="center", fontsize=7.5,
                color=DARK, fontweight="bold", zorder=5)


def line(xs, ys):
    ax.plot(xs, ys, color=DARK, lw=1.3, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")


# ── layout constants ──────────────────────────────────────────────────────────
CX   = 10.5   # main spine x
BW   = 4.4    # main box width
BH   = 0.62   # main box height
CBH  = 0.72   # compute box height  (slightly taller for two-line text)
DW   = 4.2    # diamond width
DH   = 0.88   # diamond height
ARTW = 3.0    # artefact box width
ARTH = 0.58

XL   = 3.8    # left artefact x  (D5 branch)
XR   = 16.2   # right artefact x (slinger, gasbubble, infuus)

# ── Y positions (top → bottom) ────────────────────────────────────────────────
Y_START  = 27.2

# ── step 1: P_total ──────────────────────────────────────────────────────────
Y_C1     = 26.1   # Compute P_total
Y_D1     = 24.8   # D1 diamond

# left sub-branch for D1 = Yes
X_SUB    = 3.4
Y_C1L    = 23.5   # Compute LP  (sub-tree)
Y_D1A    = 22.2   # D1a diamond
Y_CAL    = 20.8
Y_FLU    = 20.8
X_CAL    = 1.5
X_FLU    = 5.3

# ── step 2: ringing_ratio ─────────────────────────────────────────────────────
Y_C2     = 23.2   # Compute ringing_ratio
Y_D2     = 21.8   # D2 diamond

# ── step 3: HP_env + P_cardiac ───────────────────────────────────────────────
Y_C3     = 20.4   # Compute HP_env + P_cardiac
Y_D3     = 19.0   # D3 diamond

# ── step 4: LP ───────────────────────────────────────────────────────────────
Y_C4     = 17.6   # Compute LP (both channels)
Y_D4     = 16.2   # D4 diamond
Y_D5     = 14.3   # D5 diamond

# ── terminal nodes ────────────────────────────────────────────────────────────
Y_NONE   = 12.5   # No artefact → End
Y_NOTE   =  3.3   # iteration note box


# ── nodes ─────────────────────────────────────────────────────────────────────

# Start
box(CX, Y_START, BW, BH, "START  —  new pass over signal", C_START, fontsize=9)

# ── Step 1 ────────────────────────────────────────────────────────────────────
box(CX, Y_C1, BW, CBH,
    "Compute  P_total(t)  and its baseline\n"
    "[sum of spectrogram power in 0.5 – 15 Hz per window]",
    C_PROC, fontsize=7.8)

diamond(CX, Y_D1, DW, DH,
        "P_total(t)  <<  baseline_P_total ?\n"
        "(oscillations have collapsed)")

# Sub-tree: D1 = Yes
box(X_SUB, Y_C1L, 2.8, CBH,
    "Compute  LP(t)  and its baseline\n"
    "[low-pass ≤ 0.25 Hz]",
    C_PROC, fontsize=7.5)

diamond(X_SUB, Y_D1A, 2.8, 0.80,
        "LP(t)  <<  baseline_LP ?\n(signal near zero)",
        fontsize=7.5)

box(X_CAL, Y_CAL, 2.6, ARTH, "LP near zero\n→  CALIBRATION", C_CAL)
box(X_FLU, Y_FLU, 2.6, ARTH, "LP near maximum\n→  FLUSH",     C_FLUSH)

# ── Step 2 ────────────────────────────────────────────────────────────────────
box(CX, Y_C2, BW, CBH,
    "Compute  ringing_ratio(t)  and its baseline\n"
    "[P_high / P_total  where  P_high = power above 2 × f_dom]",
    C_PROC, fontsize=7.8)

diamond(CX, Y_D2, DW, DH,
        "ringing_ratio(t)  >>  baseline + k × IQR ?\n"
        "(high-frequency power dominates)")

box(XR, Y_D2, ARTW, ARTH,
    "Ringing above 2× cardiac\n→  SLINGER", C_SLING)

# ── Step 3 ────────────────────────────────────────────────────────────────────
box(CX, Y_C3, BW, CBH,
    "Compute  HP_env(t)  and  P_cardiac(t)  + baselines\n"
    "[HP envelope = rolling σ of high-pass; P_cardiac = power near f_dom]",
    C_PROC, fontsize=7.5)

diamond(CX, Y_D3, DW, DH,
        "HP_env(t)  <<  baseline_HP_env\n"
        "AND  P_cardiac(t)  <<  baseline_P_cardiac ?")

box(XR, Y_D3, ARTW, ARTH,
    "Beats damped, diastolic rises\n→  GASBUBBLE", C_GAS)

# ── Step 4 ────────────────────────────────────────────────────────────────────
box(CX, Y_C4, BW, CBH,
    "Compute  LP_ABP(t),  LP_CVP(t)  and their baselines\n"
    "[low-pass ≤ 0.25 Hz for each channel separately]",
    C_PROC, fontsize=7.8)

diamond(CX, Y_D4, DW, DH,
        "LP(t)  significantly  ≠  baseline_LP ?\n"
        "(slow baseline has shifted)")

diamond(CX, Y_D5, DW, DH,
        "ABP LP shifted  (or both channels) ?\n"
        "(vs. CVP LP shifted only)")

box(XL, Y_D5, ARTW, ARTH,
    "Baseline shift in ABP\n→  TRANSDUCER HOOG", C_TRANS)

box(XR, Y_D5, ARTW, ARTH,
    "CVP LP shifted — ABP unchanged\n→  INFUUS OP CVD", C_INF)

# ── Terminal: no artefact ─────────────────────────────────────────────────────
box(CX, Y_NONE, BW, BH,
    "No artefact detected in this pass\n→  END", C_NONE, fontsize=8.5)

# ── Iteration note ────────────────────────────────────────────────────────────
note_w, note_h = 14.0, 1.6
rect = mpatches.FancyBboxPatch(
    (CX - note_w/2, Y_NOTE - note_h/2), note_w, note_h,
    boxstyle="round,pad=0,rounding_size=0.25",
    linewidth=1.2, edgecolor="#BDC3C7", facecolor=C_NOTE, zorder=3)
ax.add_patch(rect)
ax.text(CX, Y_NOTE + 0.25,
        "Iteration rule",
        ha="center", va="center", fontsize=8.5,
        color=DARK, fontweight="bold", zorder=4)
ax.text(CX, Y_NOTE - 0.28,
        "Whenever an artefact outcome is reached, exclude that period from the signal "
        "and re-run from START.\n"
        "Repeat until a full pass produces no new artefact detection.",
        ha="center", va="center", fontsize=7.8,
        color=DARK, zorder=4, multialignment="center")


# ── arrows ─────────────────────────────────────────────────────────────────────

# Start → C1 → D1
arrow_v(CX, Y_START - BH/2,  Y_C1 + CBH/2)
arrow_v(CX, Y_C1   - CBH/2,  Y_D1 + DH/2)

# D1 → Yes → left sub-tree
line([CX - DW/2, X_SUB], [Y_D1, Y_D1])
arrow_v(X_SUB, Y_D1, Y_C1L + CBH/2)
ax.text(CX - DW/2 - 0.18, Y_D1 + 0.18, "Yes",
        ha="right", va="center", fontsize=7.5, color=DARK, fontweight="bold")

# C1L → D1A
arrow_v(X_SUB, Y_C1L - CBH/2, Y_D1A + 0.40)

# D1A → CALIBRATION (left) and FLUSH (right)
arrow_h(X_SUB - 1.4, X_CAL + 1.3, Y_D1A, label="Yes", side="above")
arrow_h(X_SUB + 1.4, X_FLU - 1.3, Y_D1A, label="No",  side="above")
arrow_v(X_CAL, Y_D1A, Y_CAL + ARTH/2)
arrow_v(X_FLU, Y_D1A, Y_FLU + ARTH/2)

# D1 → No → C2 → D2
arrow_v(CX, Y_D1 - DH/2, Y_C2 + CBH/2, label="No", side="right")
arrow_v(CX, Y_C2 - CBH/2, Y_D2 + DH/2)

# D2 → Yes → SLINGER
arrow_h(CX + DW/2, XR - ARTW/2, Y_D2, label="Yes", side="above")

# D2 → No → C3 → D3
arrow_v(CX, Y_D2 - DH/2, Y_C3 + CBH/2, label="No", side="right")
arrow_v(CX, Y_C3 - CBH/2, Y_D3 + DH/2)

# D3 → Yes → GASBUBBLE
arrow_h(CX + DW/2, XR - ARTW/2, Y_D3, label="Yes", side="above")

# D3 → No → C4 → D4
arrow_v(CX, Y_D3 - DH/2, Y_C4 + CBH/2, label="No", side="right")
arrow_v(CX, Y_C4 - CBH/2, Y_D4 + DH/2)

# D4 → Yes → D5
arrow_v(CX, Y_D4 - DH/2, Y_D5 + DH/2, label="Yes", side="right")

# D4 → No → No artefact
elbow_left(CX - DW/2, Y_D4, CX - DW/2 - 1.6, Y_NONE + BH/2, label="No")

# D5 → Yes (ABP) → TRANSDUCER HOOG
arrow_h(CX - DW/2, XL + ARTW/2, Y_D5, label="Yes", side="above")

# D5 → No (CVP only) → INFUUS OP CVD
arrow_h(CX + DW/2, XR - ARTW/2, Y_D5, label="No", side="above")


# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_CAL,   "Calibration"),
    (C_FLUSH, "Flush"),
    (C_SLING, "Slinger"),
    (C_GAS,   "Gasbubble"),
    (C_TRANS, "Transducer hoog"),
    (C_INF,   "Infuus op CVD"),
]
lx0, ly0 = 2.2, 2.2
for i, (color, label) in enumerate(legend_items):
    lx = lx0 + (i % 3) * 4.6
    ly = ly0 - (i // 3) * 0.62
    ax.add_patch(mpatches.FancyBboxPatch(
        (lx, ly - 0.18), 0.38, 0.36,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=color, edgecolor="none", zorder=5))
    ax.text(lx + 0.54, ly, label,
            va="center", fontsize=8, color=DARK, zorder=5)

ax.text(CX, 2.72, "Artefact colour key",
        ha="center", fontsize=8.5, color=DARK, fontweight="bold")

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(CX, 27.75,
        "Artefact detection decision tree  —  lazy metric computation",
        ha="center", va="center", fontsize=11,
        fontweight="bold", color=DARK)

# ── save ──────────────────────────────────────────────────────────────────────
out = "decision_tree.svg"
fig.savefig(out, format="svg", bbox_inches="tight")
print(f"Saved: {out}")
