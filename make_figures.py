"""Builds the README figures in assets/ from raw-generations.json, analysis.json and vec-stats.json."""
import json, re
import numpy as np
from matplotlib.patches import Rectangle
from figstyle import *

P = theme("violet")
RAW = json.load(open("raw-generations.json", encoding="utf8"))["rows"]
S = json.load(open("analysis.json"))["summary"]
NAMES = ["pirate", "anxious", "shakespeare", "corporate", "conspiracy", "peasant", "commentator", "retriever"]
PRETTY = {"pirate": "pirate", "anxious": "anxious", "shakespeare": "shakespearean", "corporate": "linkedin",
          "conspiracy": "conspiracy", "peasant": "peasant", "commentator": "commentator", "retriever": "retriever"}


def out(label_, f, q):
    return next(r["out"] for r in RAW if r["label"] == label_ and r["f"] == f and r["q"] == q)


def clean(s, n=200):
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n].rstrip() + " …") if len(s) > n else s


four = lambda s: re.search(r"\b(4|four)\b", s, re.I) is not None

# 0. banner: two persona blobs that overlap into a third shape
yy, xx = np.mgrid[0:100, 0:400] / np.array([100, 400])[:, None, None]
blob = lambda cx, cy, r: np.exp(-(((xx - cx) * 4) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
a, b = blob(0.72, 0.42, 0.28), blob(0.86, 0.62, 0.26)
field = np.clip(0.08 + 0.55 * a + 0.55 * b + 0.6 * a * b * 3, 0, 1)
banner("banner.png", "persona-mixing-steering",
       "add two personality directions at once and you get a new character, and lose 2+2.", field)

# 1. hero: four mixes
picks = [("anxious+shakespeare", "What's 2+2?"), ("pirate+commentator", "What's 2+2?"),
         ("anxious+corporate", "How do I boil an egg?"), ("shakespeare+commentator", "How do I boil an egg?")]
rows = []
for lab, q in picks:
    a_, b_ = lab.split("+")
    rows.append((f"[{PRETTY[a_]}\n+{PRETTY[b_]}]".replace("\n", " "), f'> {q}   << "{clean(out(lab, 0.45, q), 170)}"'))
terminal("hero-mixes.png", "two personality directions at once, same boring questions",
         "qwen2.5-1.5b-instruct · two trait vectors summed at layer 16 (0.45× each) · greedy",
         rows, label_w=26, levels=[1.0] * 4,
         foot="verbatim from raw-generations.json · alone at 0.8×, each trait here answered 2+2 with four, except anxious")


# 2. 2+2 accuracy, with the anxious vector split out
def says4(kind, f, keep=lambda traits: True):
    rr = [r for r in RAW if r["kind"] == kind and r["f"] == f and r["q"] == "What's 2+2?" and keep(r["traits"])]
    return sum(four(r["out"]) for r in rr), len(rr)


no_anx = lambda traits: "anxious" not in traits
bars = [("normal", (S["baseline@0.0"]["math_4"], S["baseline@0.0"]["math_n"]), 1.0, "fg"),
        ("1 trait\n0.8×", says4("single", 0.8), 0.5, "fg"),
        ("1 trait\nno anx.", says4("single", 0.8, no_anx), 0.18, "fg"),
        ("2 traits\n0.45×", says4("combo", 0.45), 1.0, "acc"),
        ("2 traits\nno anx.", says4("combo", 0.45, no_anx), 0.4, "acc"),
        ("2 traits\n0.6×", says4("combo", 0.6), 1.0, "acc"),
        ("random\n0.45×", (S["random_combo@0.45"]["math_4"], S["random_combo@0.45"]["math_n"]), 0.3, "dim"),
        ("random\n0.6×", (S["random_combo@0.6"]["math_4"], S["random_combo@0.6"]["math_n"]), 0.3, "dim")]
f, ax = fig(6.2, bottom_in=1.2)
x = np.arange(len(bars))
for i, (_, (k, n), lv, ramp) in enumerate(bars):
    v = 100 * k / n
    dbar(ax, x[i], v, lv, ramp, width=0.64)
    label(ax, i, v + 3, f"{k}/{n}", color=getattr(P, ramp), ha="center", va="bottom")
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=12)
ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel('% answering 2+2 with "4"')
ax.grid(axis="x", visible=False)
title(ax, "pairs miss 2+2 more, and anxious drives much of it",
      "'no anx.' = drop every condition with anxious · 0.45× pair ≈ same total push as 0.8× single")
save(f, "two-plus-two.png")

# 3. collapse rate
conds = [("normal", "baseline@0.0", 1.0, "fg"), ("1 trait\n0.8×", "single@0.8", 0.5, "fg"), ("1 trait\n1.0×", "single@1.0", 0.5, "fg"),
         ("2 traits\n0.45×", "combo@0.45", 1.0, "acc"), ("2 traits\n0.6×", "combo@0.6", 1.0, "acc"),
         ("random\n0.45×", "random_combo@0.45", 0.3, "dim"), ("random\n0.6×", "random_combo@0.6", 0.3, "dim")]
f, ax = fig(5.4, bottom_in=1.2)
x = np.arange(len(conds))
for i, (_, k, lv, ramp) in enumerate(conds):
    v = 100 * S[k]["collapsed"] / S[k]["n"]
    dbar(ax, x[i], v, lv, ramp, width=0.64)
    label(ax, i, v + 1, f'{S[k]["collapsed"]}/{S[k]["n"]}', color=getattr(P, ramp), ha="center", va="bottom")
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in conds], fontsize=12)
ax.set_ylim(0, 40); ax.set_ylabel("% of answers collapsed")
ax.grid(axis="x", visible=False)
title(ax, "collapse (loops or gibberish), all four prompts", "collapsed = repeated 4-gram rate >= 0.3 or distinct-token ratio < 0.4")
save(f, "collapse-rate.png")

# 4. pair matrix for 2+2 at 0.45x (diagonal = single trait at 0.8x)
WRONG = {("pirate", "commentator"), ("peasant", "commentator"), ("shakespeare", "commentator")}
STY = {0: (1.0, "acc"), 1: (0.22, "dim"), 2: (1.0, "fg")}
f, ax = fig(9.0, left=0.2, right=0.95, bottom_in=2.6)
for i, a_ in enumerate(NAMES):
    for j, b_ in enumerate(NAMES):
        p = (a_, b_) if i < j else (b_, a_)
        s = out(a_, 0.8, "What's 2+2?") if i == j else out(f"{p[0]}+{p[1]}", 0.45, "What's 2+2?")
        code = 0 if four(s) else (2 if (i != j and p in WRONG) else 1)
        lv, ramp = STY[code]
        ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, facecolor=D(lv, ramp), edgecolor="none"))
        if i == j:
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, edgecolor=P.fg, lw=3))
ax.set_xlim(-0.55, 7.55); ax.set_ylim(7.55, -0.55); ax.set_aspect("equal")
ax.set_xticks(range(8)); ax.set_xticklabels([PRETTY[n] for n in NAMES], rotation=40, ha="right")
ax.set_yticks(range(8)); ax.set_yticklabels([PRETTY[n] for n in NAMES])
ax.grid(False); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.legend(handles=[swatch(1.0, "acc", "says 4"), swatch(0.22, "dim", "dodges / panics / loops"), swatch(1.0, "fg", "wrong number")],
          loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3)
title(ax, '"what\'s 2+2?" for every pair of traits', "off-diagonal: two traits at 0.45× each · outlined diagonal: one trait at 0.8×")
save(f, "pair-matrix-2plus2.png")

# 5. trait-vector overlap
V = json.load(open("vec-stats.json"))
C = np.array([[V["cos"][a_][b_] for b_ in NAMES] for a_ in NAMES])
f, ax = fig(8.6, left=0.2, right=0.95, bottom_in=1.9)
for i in range(8):
    for j in range(8):
        ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, facecolor=D((C[i, j] - 0.4) / 0.6), edgecolor="none"))
        if i != j:
            label(ax, j, i, f"{C[i, j]:.2f}"[1:], ha="center", va="center", boxed=True, fontsize=11.5,
                  color=P.acc if C[i, j] >= 0.85 else P.fg, fontweight="bold" if C[i, j] >= 0.85 else "normal")
ax.set_xlim(-0.55, 7.55); ax.set_ylim(7.55, -0.55); ax.set_aspect("equal")
ax.set_xticks(range(8)); ax.set_xticklabels([PRETTY[n] for n in NAMES], rotation=40, ha="right")
ax.set_yticks(range(8)); ax.set_yticklabels([PRETTY[n] for n in NAMES])
ax.grid(False); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
title(ax, f"the trait directions overlap a lot (cosine, layer {V['layer']})",
      "pairs share a 'be a character' component · steering is at layer 16 (cos 0.53-0.90 there)")
save(f, "trait-cosine.png")
