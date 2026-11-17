"""Shared figure style for the README charts (clean light theme, 1600 px wide)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

INK = "#1f2328"
MUTED = "#656d76"
GRID = "#d8dee4"
BG = "#ffffff"
RED = "#cf3e4f"
BLUE = "#2f6fdf"
GREEN = "#2a9d70"
AMBER = "#e39b22"
PURPLE = "#8250df"
GRAY = "#8c959f"
PALETTE = [BLUE, RED, GREEN, AMBER, PURPLE, GRAY]

WIDTH_IN, DPI = 10.0, 160  # 1600 px wide


def setup():
    fams = [f.name for f in font_manager.fontManager.ttflist]
    base = "Lato" if "Lato" in fams else "DejaVu Sans"  # any sans with a bold weight works
    cjk = [n for n in ("Noto Sans CJK JP", "Noto Sans CJK SC") if n in fams]
    plt.rcParams.update({
        "font.family": [base, "DejaVu Sans"] + cjk,
        "font.size": 12,
        "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 12, "axes.labelcolor": INK,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "legend.frameon": False, "legend.fontsize": 11,
    })


def fig(h=5.6, **kw):
    setup()
    return plt.subplots(figsize=(WIDTH_IN, h), dpi=DPI, **kw)


def subtitle(ax, text):
    """Call after ax.set_title(); adds a muted line under the title."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=30)
    ax.text(0, 1.02, text, transform=ax.transAxes, color=MUTED, fontsize=11, va="bottom")


def note(f, text):
    f.text(0.01, 0.01, text, color=MUTED, fontsize=9.5, ha="left", va="bottom")


def save(f, name, out="assets"):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    f.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.25,
              metadata={"Software": None})  # no tool/version metadata in the PNG
    plt.close(f)
    print("wrote", p)


def text_card(title, sub, rows, name, label_w=0.17, wrap=92, foot=None, colors=None):
    """Render example outputs as a clean table-like card. rows = [(label, text), ...]"""
    import textwrap
    from matplotlib.patches import FancyBboxPatch
    setup()
    wrapped = [textwrap.wrap(t, wrap) or [""] for _, t in rows]
    line_h = 0.24  # inches per text line
    heights = [0.22 + line_h * len(w) for w in wrapped]
    H = 1.1 + sum(heights) + 0.12 * len(rows) + (0.35 if foot else 0)
    f = plt.figure(figsize=(WIDTH_IN, H), dpi=DPI)
    ax = f.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, WIDTH_IN); ax.set_ylim(0, H)
    y = H - 0.3
    ax.text(0.3, y, title, fontsize=16, fontweight="bold", va="top")
    y -= 0.38
    ax.text(0.3, y, sub, fontsize=11, color=MUTED, va="top")
    y -= 0.42
    for i, ((lab, _), w, h) in enumerate(zip(rows, wrapped, heights)):
        c = (colors[i] if colors else BLUE)
        ax.add_patch(FancyBboxPatch((0.3, y - h), WIDTH_IN - 0.6, h,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#f6f8fa", ec=GRID, lw=1))
        ax.add_patch(FancyBboxPatch((0.3, y - h), 0.07, h, boxstyle="square,pad=0", fc=c, ec="none"))
        ax.text(0.55, y - 0.13, lab, fontsize=11.5, fontweight="bold", color=c, va="top")
        x0 = 0.55 + label_w * WIDTH_IN
        for j, line in enumerate(w):
            ax.text(x0, y - 0.13 - j * line_h, line, fontsize=11, va="top", color=INK, usetex=False, parse_math=False)
        y -= h + 0.12
    if foot:
        ax.text(0.3, 0.18, foot, fontsize=9.5, color=MUTED, va="bottom")
    p = os.path.join("assets", name); os.makedirs("assets", exist_ok=True)
    f.savefig(p, dpi=DPI, metadata={"Software": None}); plt.close(f); print("wrote", p)
