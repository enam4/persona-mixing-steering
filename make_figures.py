"""Builds the README figures in assets/ from raw-generations.json, analysis.json and vec-stats.json."""
import json, re
import numpy as np
from matplotlib.colors import ListedColormap
from figstyle import *

RAW = json.load(open("raw-generations.json"))["rows"]
S = json.load(open("analysis.json"))["summary"]
NAMES = ["pirate", "anxious", "shakespeare", "corporate", "conspiracy", "peasant", "commentator", "retriever"]
PRETTY = {"pirate": "pirate", "anxious": "anxious", "shakespeare": "Shakespearean", "corporate": "LinkedIn corporate",
          "conspiracy": "conspiracy", "peasant": "medieval peasant", "commentator": "sports commentator", "retriever": "golden retriever"}


def out(label, f, q):
    return next(r["out"] for r in RAW if r["label"] == label and r["f"] == f and r["q"] == q)


def clean(s, n=200):
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n].rstrip() + " …") if len(s) > n else s


four = lambda s: re.search(r"\b(4|four)\b", s, re.I) is not None

# 1. hero card: four mixes, each with what the traits did alone
picks = [("anxious+shakespeare", "What's 2+2?"), ("pirate+commentator", "What's 2+2?"),
         ("anxious+corporate", "How do I boil an egg?"), ("shakespeare+commentator", "How do I boil an egg?")]
rows = []
for lab, q in picks:
    a, b = lab.split("+")
    rows.append((f"{PRETTY[a]}\n+ {PRETTY[b]}".replace("\n", " "), f'{q}  →  "{clean(out(lab, 0.45, q), 170)}"'))
text_card("Two personality directions at once, same boring questions",
          "Qwen2.5-1.5B-Instruct · two trait vectors summed at layer 16 (0.45× each) · greedy decoding",
          rows, "hero-mixes.png", label_w=0.30, wrap=76, colors=[PURPLE] * 4,
          foot="Verbatim outputs from raw-generations.json (… marks a cut). Alone at 0.8×, each of these traits answered 2+2 with four, except anxious.")

# 2. capability: said 4 to "What's 2+2?"
conds = [("normal model", "baseline@0.0", GREEN), ("one trait\n0.8×", "single@0.8", BLUE), ("one trait\n1.0×", "single@1.0", BLUE),
         ("two traits\n0.45× each", "combo@0.45", PURPLE), ("two traits\n0.6× each", "combo@0.6", PURPLE),
         ("random pair\n0.45×", "random_combo@0.45", GRAY), ("random pair\n0.6×", "random_combo@0.6", GRAY)]
f, ax = fig(5.2)
x = np.arange(len(conds))
v = [100 * S[k]["math_4"] / S[k]["math_n"] for _, k, _ in conds]
ax.bar(x, v, color=[c for *_, c in conds], width=0.62)
for i, (_, k, c) in enumerate(conds):
    ax.text(i, v[i] + 2, f'{S[k]["math_4"]}/{S[k]["math_n"]}', ha="center", fontweight="bold", color=c)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in conds], fontsize=10.5)
ax.set_ylim(0, 112); ax.set_ylabel('% answering "What\'s 2+2?" with 4')
ax.set_title("At the same total push, two traits break basic arithmetic more than one")
subtitle(ax, "A 0.45×-each pair is about the same total push as one trait at 0.8×. Random pairs are rescaled to match the trait pairs.")
ax.grid(axis="x", visible=False)
save(f, "two-plus-two.png")

# 3. collapse rate
f, ax = fig(4.6)
v = [100 * S[k]["collapsed"] / S[k]["n"] for _, k, _ in conds]
ax.bar(x, v, color=[c for *_, c in conds], width=0.62)
for i, (_, k, c) in enumerate(conds):
    ax.text(i, v[i] + 1, f'{S[k]["collapsed"]}/{S[k]["n"]}', ha="center", fontweight="bold", color=c)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in conds], fontsize=10.5)
ax.set_ylim(0, 40); ax.set_ylabel("% of answers that collapsed")
ax.set_title("Collapse (loops or gibberish) across all four prompts")
subtitle(ax, "collapsed = repeated 4-gram rate ≥ 0.3 or distinct-token ratio < 0.4")
ax.grid(axis="x", visible=False)
save(f, "collapse-rate.png")

# 4. pair matrix for 2+2 at 0.45x (diagonal = single trait at 0.8x)
M = np.full((8, 8), np.nan); T = [["" for _ in range(8)] for _ in range(8)]
# Pairs whose 2+2 answer names a wrong number, checked by reading every answer (listed in results.md).
WRONG = {("pirate", "commentator"), ("peasant", "commentator"), ("shakespeare", "commentator")}
def code2(s, pair=None):
    if four(s): return 0
    return 2 if pair in WRONG else 1
for i, a in enumerate(NAMES):
    for j, b in enumerate(NAMES):
        p = (a, b) if i < j else (b, a)
        s = out(a, 0.8, "What's 2+2?") if i == j else out(f"{p[0]}+{p[1]}", 0.45, "What's 2+2?")
        M[i, j] = code2(s, None if i == j else p)
f, ax = fig(7.0)
cm = ListedColormap([GREEN, "#d0d7de", RED])
ax.imshow(M, cmap=cm, vmin=0, vmax=2)
ax.set_xticks(range(8)); ax.set_xticklabels([PRETTY[n] for n in NAMES], rotation=35, ha="right")
ax.set_yticks(range(8)); ax.set_yticklabels([PRETTY[n] for n in NAMES])
for i in range(9):
    ax.axhline(i - 0.5, color="white", lw=2); ax.axvline(i - 0.5, color="white", lw=2)
from matplotlib.patches import Rectangle
for i in range(8):
    ax.add_patch(Rectangle((i - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec=INK, lw=2.2, zorder=5))
ax.grid(False)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title('"What\'s 2+2?" for every pair of traits')
subtitle(ax, "Off-diagonal: two traits at 0.45× each. Outlined diagonal: one trait alone at 0.8×.")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREEN, label="says 4"), Patch(color="#d0d7de", label="dodges, panics or loops"), Patch(color=RED, label="gives a wrong number")],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
save(f, "pair-matrix-2plus2.png")
print("wrong-number pairs:", sorted({tuple(sorted((NAMES[i], NAMES[j]))) for i in range(8) for j in range(8) if i != j and M[i, j] == 2}))

# 5. trait-vector overlap
V = json.load(open("vec-stats.json"))
C = np.array([[V["cos"][a][b] for b in NAMES] for a in NAMES])
f, ax = fig(6.6)
im = ax.imshow(C, cmap="Purples", vmin=0.4, vmax=1)
for i in range(8):
    for j in range(8):
        ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if C[i, j] > 0.75 else INK)
ax.set_xticks(range(8)); ax.set_xticklabels([PRETTY[n] for n in NAMES], rotation=35, ha="right")
ax.set_yticks(range(8)); ax.set_yticklabels([PRETTY[n] for n in NAMES]); ax.grid(False)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title(f"The trait directions overlap a lot (cosine similarity, layer {V['layer']})")
subtitle(ax, "Pairs share a big \"be a character\" component, so a pair pushes that part harder than one trait does.")
save(f, "trait-cosine.png")
