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
fig, axes = plt.subplots(1, 4, figsize=(5.4, 1.75))
w = 0.36


def pct(frac):
    a, n = map(int, frac.split("/"))
    return 100 * a / n


PANELS = [("gpt-4o", "tests"), ("claude-sonnet-4-5", "tests"), ("gpt-4o", "rvh"), ("claude-sonnet-4-5", "rvh")]
SHORT = {"gpt-4o": "GPT-4o", "claude-sonnet-4-5": "Claude Sonnet 4.5"}
for ax, (m, what) in zip(axes, PANELS):
    for k, s in enumerate(("one_shot", "structure_first")):
        xs = [i + (k - 0.5) * (w + 0.02) for i in range(len(PROMPTS))]
        ys = [S[m][p]["tests_mean"][s] if what == "tests" else pct(T[m][p]["rvh_artifacts"][s]) for p, _ in PROMPTS]
        ax.bar(xs, ys, w, color=COL[s], label=LAB[s], edgecolor="white", linewidth=1)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.1f}" if what == "tests" else f"{y:.0f}", ha="center", va="bottom", fontsize=5.5, color=INK)
    ax.set_xticks(range(len(PROMPTS)), ["Specified", "Under-spec."])
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6); ax.set_axisbelow(True)
    ax.margins(y=0.2)
    sub = "test functions per artifact" if what == "tests" else "RVH-positive artifacts (%)"
    ax.set_title(("(a) " if what == "tests" else "(b) ") + SHORT[m] + chr(10) + sub, fontsize=6.5, color=INK)
    ax.tick_params(labelsize=6)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
fig.savefig("../figures/fig1.pdf")
fig.savefig("../figures/fig1.png", dpi=200)
print("saved ../figures/fig1.pdf")
