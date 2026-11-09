# -*- coding: utf-8 -*-
"""Figure 1 of the ICBDAIA paper, from statistics.json and statistics_strict.json (no API call).

    cd results && python ../scripts/analyze_results.py && STRICT=1 python ../scripts/analyze_results.py
    python ../scripts/make_figure.py      (from results/) -> ../figures/fig1.pdf
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("statistics.json", encoding="utf-8"))
T = json.load(open("statistics_strict.json", encoding="utf-8"))
MODELS = [("gpt-4o", "GPT-4o (40 per cell)"), ("claude-sonnet-4-5", "Claude Sonnet 4.5 (10 per cell)")]
PROMPTS = [("specified", "Specified"), ("under_specified", "Under-specified")]
COL = {"one_shot": "#2a78d6", "structure_first": "#eb6834"}
LAB = {"one_shot": "One-shot", "structure_first": "Structure-first"}
INK, MUTED = "#222222", "#6b6b6b"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})
fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.0))
w = 0.36


def pct(frac):
    a, n = map(int, frac.split("/"))
    return 100 * a / n


for c, (m, mtitle) in enumerate(MODELS):
    for r, (ylabel, get) in enumerate([
            ("Test functions per artifact", lambda st, p, s: S[m][p]["tests_mean"][s]),
            ("RVH-positive artifacts (%)\nsensitivity analysis", lambda st, p, s: pct(T[m][p]["rvh_artifacts"][s]))]):
        ax = axes[r, c]
        for k, s in enumerate(("one_shot", "structure_first")):
            xs = [i + (k - 0.5) * (w + 0.02) for i in range(len(PROMPTS))]
            ys = [get(None, p, s) for p, _ in PROMPTS]
            ax.bar(xs, ys, w, color=COL[s], label=LAB[s], edgecolor="white", linewidth=1)
            for x, y in zip(xs, ys):
                ax.text(x, y, f"{y:.0f}" if r else f"{y:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(PROMPTS)), [t for _, t in PROMPTS])
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6); ax.set_axisbelow(True)
        ax.margins(y=0.18)
        if c == 0:
            ax.set_ylabel(ylabel)
        if r == 0:
            ax.set_title(mtitle, fontsize=8.5, color=INK)
axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig("../figures/fig1.pdf")
fig.savefig("../figures/fig1.png", dpi=200)
print("saved ../figures/fig1.pdf")
