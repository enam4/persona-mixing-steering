"""Shared figure style for the README charts.

Every chart is rendered twice, once per GitHub theme: `name.png` (light) and `name-dark.png` (dark).
The README picks the right one with a <picture> tag. Run `python make_figures.py`; it calls
`both_themes(__file__)`, which re-runs the script once per theme with FIG_THEME set.
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

THEME = os.environ.get("FIG_THEME", "light")
DARK = THEME == "dark"

# Colours follow GitHub's own light and dark palettes so the charts sit flush on the page.
if DARK:
    BG, PANEL, INK, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#9198a1", "#30363d"
    RED, BLUE, GREEN, AMBER, PURPLE, GRAY = "#f8515f", "#4c8ff8", "#3fb97f", "#e3a52c", "#a57ef8", "#7d8590"
    NEUTRAL = "#3d444d"   # "no result" fill in categorical heatmaps
    ON_BAR = "#0d1117"    # labels drawn inside coloured bars
else:
    BG, PANEL, INK, MUTED, GRID = "#ffffff", "#f6f8fa", "#1f2328", "#59636e", "#d8dee4"
    RED, BLUE, GREEN, AMBER, PURPLE, GRAY = "#cf3e4f", "#2f6fdf", "#1f8a5f", "#c98512", "#8250df", "#8c959f"
    NEUTRAL = "#d0d7de"
    ON_BAR = "#ffffff"
PALETTE = [BLUE, RED, GREEN, AMBER, PURPLE, GRAY]

WIDTH_IN, DPI = 10.0, 160  # 1600 px wide

# Any of these sans fonts works; the CJK, Hangul and Thai fallbacks keep model outputs readable.
SANS = ["Lato", "Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
SCRIPT_FALLBACKS = ["Noto Sans CJK JP", "Noto Sans CJK SC", "Microsoft YaHei", "Malgun Gothic",
                    "Yu Gothic", "Leelawadee UI", "Noto Sans Thai"]


def both_themes(script):
    """Re-run `script` once per theme, then stop the parent process."""
    if "FIG_THEME" in os.environ:
        return
    for t in ("light", "dark"):
        subprocess.run([sys.executable, script], env={**os.environ, "FIG_THEME": t}, check=True,
                       cwd=os.path.dirname(os.path.abspath(script)))
    sys.exit(0)


def setup():
    have = {f.name for f in font_manager.fontManager.ttflist}
    fams = [n for n in SANS if n in have] + [n for n in SCRIPT_FALLBACKS if n in have]
    plt.rcParams.update({
        "font.family": fams or ["DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 14, "axes.titlecolor": INK,
        "axes.labelsize": 12, "axes.labelcolor": INK,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "ytick.left": False,
        "text.color": INK,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "legend.frameon": False, "legend.fontsize": 11, "legend.labelcolor": INK,
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


def outname(name):
    base, ext = os.path.splitext(name)
    return f"{base}-dark{ext}" if DARK else name


def save(f, name, out="assets"):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, outname(name))
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
    ax.text(0.3, y, title, fontsize=16, fontweight="bold", va="top", color=INK)
    y -= 0.38
    ax.text(0.3, y, sub, fontsize=11, color=MUTED, va="top")
    y -= 0.42
    for i, ((lab, _), w, h) in enumerate(zip(rows, wrapped, heights)):
        c = (colors[i] if colors else BLUE)
        ax.add_patch(FancyBboxPatch((0.3, y - h), WIDTH_IN - 0.6, h,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=PANEL, ec=GRID, lw=1))
        ax.add_patch(FancyBboxPatch((0.3, y - h), 0.07, h, boxstyle="square,pad=0", fc=c, ec="none"))
        ax.text(0.55, y - 0.13, lab, fontsize=11.5, fontweight="bold", color=c, va="top")
        x0 = 0.55 + label_w * WIDTH_IN
        for j, line in enumerate(w):
            ax.text(x0, y - 0.13 - j * line_h, line, fontsize=11, va="top", color=INK, usetex=False, parse_math=False)
        y -= h + 0.12
    if foot:
        ax.text(0.3, 0.18, foot, fontsize=9.5, color=MUTED, va="bottom")
    p = os.path.join("assets", outname(name)); os.makedirs("assets", exist_ok=True)
    f.savefig(p, dpi=DPI, metadata={"Software": None}); plt.close(f); print("wrote", p)
