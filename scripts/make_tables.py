# -*- coding: utf-8 -*-
"""Generate the LaTeX rows of the paper's main table from the data (no number is typed by hand).

Inputs: artifacts_v2.csv (reanalyze_v2.py), execution_results.json and execution_results_ablation.json
(execute_artifacts.py). Output: ../figures/table_main.tex and a printed check.
    py -3.13 make_tables.py
"""
import json, os
import pandas as pd

d = pd.read_csv("artifacts_v2.csv")
ex = []
for f in ("execution_results.json", "execution_results_ablation.json"):
    if os.path.exists(f):
        ex += json.load(open(f, encoding="utf-8"))
e = pd.DataFrame(ex)
if len(e):
    e["ap"] = (e["category"] == "all_pass").astype(int)
    e["unres_exec"] = (e["category"] == "collection_unresolved_first_party").astype(int)
    e["rep"] = e.apply(lambda r: int(r["category"] == "all_pass" or (isinstance(r.get("names_repaired"), dict)
                                                                       and r["names_repaired"]["category"] == "all_pass")), axis=1)
    d = d.merge(e[["artifact", "ap", "rep", "unres_exec"]], on="artifact", how="left")
else:
    d["ap"] = d["rep"] = float("nan")

ROWS = [("gpt-4o", "one_shot", "One-shot (OS)", 1), ("gpt-4o", "one_shot_seed", "OS + seed instruction", 1),
        ("gpt-4o", "two_call", "Two calls, no skeleton", 2), ("gpt-4o", "structure_first", "Structure-first (SF)", 3),
        ("gpt-4o", "structure_first_plan", "SF + skeleton to step 3", 3),
        ("claude-sonnet-4-5", "one_shot", "One-shot (OS)", 1), ("claude-sonnet-4-5", "structure_first", "Structure-first (SF)", 3)]


def frac(k, n):
    return f"{int(k)} ({100 * k / n:.0f}\\%)"


lines = []
for model, st, label, calls in ROWS:
    s = d[(d.model == model) & (d.strategy == st)]
    n = len(s)
    if n == 0:
        continue
    ap = "--" if s.ap.isna().any() else frac(s.ap.sum(), n)
    rep = "--" if s.rep.isna().any() else frac(s.rep.sum(), n)
    mname = "GPT-4o" if model == "gpt-4o" else "Claude"
    lines.append(f"{mname} & {label} & {n} & {calls} & {s.tests.mean():.1f} & {frac(s.rvh_v2.sum(), n)} & "
                 f"{int(s.unres_exec.sum())} & {int(s.iih_X2_any.sum())} & {ap} & {rep} & {s.cost_usd.mean():.3f}\\\\")
os.makedirs("../figures", exist_ok=True)
open("../figures/table_main.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
cl = d[d.model == "claude-sonnet-4-5"]
print("Claude unpinned packages:", cl.groupby("strategy").iih_X2_any.sum().to_dict(),
      "| RVH counting undetermined:", cl.groupby("strategy").rvh_v2_undet_as_viol.sum().to_dict())
