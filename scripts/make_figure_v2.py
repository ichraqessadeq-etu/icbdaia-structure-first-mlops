# -*- coding: utf-8 -*-
"""Fig. 2 of the revised paper: execution outcome per model and condition (100% stacked bars).
Palette: categorical slots 1-4 of the reference palette (validated: CVD dE >= 9.1, normal dE >= 22.9;
two slots below 3:1 contrast -> every segment >= 7% carries a visible count label, plus a legend).
    py -3.13 make_figure_v2.py   -> ../figures/fig_outcomes.pdf and .png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
CATS = [("all_pass", "All tests pass", "#2a78d6"),
        ("some_fail", "Some tests fail", "#eb6834"),
        ("unresolved", "Tests do not load: first-party module missing", "#1baf7a"),
        ("package", "Tests do not load: package outside the environment / other", "#eda100")]
ROWS = [("gpt-4o", "one_shot", "GPT-4o  One-shot (OS)"), ("gpt-4o", "one_shot_seed", "GPT-4o  OS + seed instruction"),
        ("gpt-4o", "two_call", "GPT-4o  Two calls, no skeleton"), ("gpt-4o", "structure_first", "GPT-4o  Structure-first (SF)"),
        ("gpt-4o", "structure_first_plan", "GPT-4o  SF + skeleton to step 3"),
        ("claude-sonnet-4-5", "one_shot", "Claude  One-shot (OS)"), ("claude-sonnet-4-5", "structure_first", "Claude  Structure-first (SF)")]

ex = []
for f in ("execution_results.json", "execution_results_ablation.json"):
    if os.path.exists(f):
        ex += json.load(open(f, encoding="utf-8"))
e = pd.DataFrame(ex)
meta = pd.read_csv("artifacts_v2.csv")[["artifact", "model", "strategy"]]
d = meta.merge(e[["artifact", "category"]], on="artifact", how="inner")


def bucket(c):
    if c == "all_pass":
        return "all_pass"
    if c == "some_fail":
        return "some_fail"
    if c == "collection_unresolved_first_party":
        return "unresolved"
    return "package"


d["b"] = d.category.map(bucket)
rows = [(lab, d[(d.model == m) & (d.strategy == s)]) for m, s, lab in ROWS]
rows = [(lab, s) for lab, s in rows if len(s)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(rows) + 0.9), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for i, (lab, s) in enumerate(rows):
    y = len(rows) - 1 - i
    left = 0.0
    for key, _, col in CATS:
        k = int((s.b == key).sum()); w = k / len(s)
        if w > 0:
            ax.barh(y, w, left=left, height=0.62, color=col, edgecolor=SURFACE, linewidth=1.2)
            if w >= 0.07:
                ax.text(left + w / 2, y, str(k), ha="center", va="center", color=INK, fontsize=7)
        left += w
    ax.text(1.01, y, f"n = {len(s)}", va="center", ha="left", color=INK2, fontsize=7)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([lab for lab, _ in rows][::-1], color=INK)
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=INK2)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c3"); ax.tick_params(length=0)
ax.xaxis.grid(True, color="#e7e6e2", linewidth=0.6); ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CATS]
ax.legend(handles, [l for _, l, _ in CATS], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False,
          fontsize=7, handlelength=1.0, columnspacing=1.2)
fig.tight_layout()
os.makedirs("../figures", exist_ok=True)
fig.savefig("../figures/fig_outcomes.pdf", facecolor=SURFACE)
fig.savefig("../figures/fig_outcomes.png", facecolor=SURFACE)
print({lab: s.b.value_counts().to_dict() for lab, s in rows})
