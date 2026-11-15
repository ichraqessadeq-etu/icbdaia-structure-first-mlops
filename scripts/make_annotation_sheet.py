# -*- coding: utf-8 -*-
"""Build the blind annotation workbook for the manual validation of the RVH oracle (review point 1.3).

Items (shuffled, detector version hidden from the annotators):
  A. every RVH finding of the revised detector (v2);
  B. every RVH finding of the published detector (v1) that v2 no longer reports (to check that the
     removals are right);
  C. 15 artifacts without any v2 finding (stratified by model and strategy), for recall: the
     annotator lists missed unseeded sources of randomness.
Each annotator fills a copy independently: verdict TP (the call makes the run non-reproducible),
FP (it does not), or ? (cannot be decided from the artifact). The key (item -> source) is written to a
separate file that the annotators must not open.

    py -3.13 make_annotation_sheet.py   -> annotation/RVH_annotation_sheet.xlsx, annotation/KEY_do_not_open.csv
"""
import csv, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import extract_code_blocks, detect_rvh as v1
from detectors_v2 import detect_rvh_v2
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

random.seed(2026)
rows = json.load(open("artifacts_v2.json", encoding="utf-8"))
items = []


def text_of(art):
    f = f"{art}/final_artifact.txt" if os.path.exists(f"{art}/final_artifact.txt") else f"{art}/response.txt"
    return open(f, encoding="utf-8").read()


def context(block, line, k=6):
    L = block.splitlines()
    lo, hi = max(0, line - 1 - k), min(len(L), line + k)
    return "\n".join(f"{'>>' if i + 1 == line else '  '}{i + 1:4d} {L[i]}" for i in range(lo, hi))


for r in rows:
    art = r["artifact"]
    for bi, b in enumerate(extract_code_blocks(text_of(art), "python")):
        r1 = v1(b)
        if r1["parse_error"]:
            continue
        r2 = detect_rvh_v2(b)
        f2 = {(f["lineno"], f["rule"]) for f in r2["findings"] if f["kind"] == "RVH"}
        lines2 = {f["lineno"] for f in r2["findings"] if f["kind"] == "RVH"}
        kind = "test" if "def test" in b else "pipeline"
        for f in r2["findings"]:
            if f["kind"] == "RVH":
                items.append({"source": "A_v2", "artifact": art, "model": r["model"], "strategy": r["strategy"],
                              "block": bi, "block_kind": kind, "line": f["lineno"], "call": f["detail"],
                              "rule": f["rule"], "code": context(b, f["lineno"])})
        for v in r1["violations"]:
            if v["lineno"] not in lines2:
                items.append({"source": "B_v1_only", "artifact": art, "model": r["model"], "strategy": r["strategy"],
                              "block": bi, "block_kind": kind, "line": v["lineno"], "call": v["detail"],
                              "rule": v["rule"], "code": context(b, v["lineno"])})

neg = [r for r in rows if not r["rvh_v2"] and not r["v2_undetermined"]]
strata = {}
for r in neg:
    strata.setdefault((r["model"], r["strategy"]), []).append(r)
pick = []
for (m, s), lst in sorted(strata.items()):
    pick += random.sample(lst, min(5 if m == "gpt-4o" else 3, len(lst)))
for r in pick[:15]:
    items.append({"source": "C_negative", "artifact": r["artifact"], "model": r["model"], "strategy": r["strategy"],
                  "block": "", "block_kind": "all", "line": "", "call": "(whole artifact: list any unseeded randomness)",
                  "rule": "", "code": f"Open {r['artifact']}/" + ("final_artifact.txt" if os.path.exists(r['artifact'] + '/final_artifact.txt') else "response.txt")})

random.shuffle(items)
os.makedirs("annotation", exist_ok=True)
wb = Workbook(); ws = wb.active; ws.title = "Items"
hdr = ["item", "artifact", "block_kind", "line", "call flagged", "code (>> = flagged line)", "verdict (TP / FP / ?)",
       "missed violations (C items only)", "comment"]
ws.append(hdr)
for c in ws[1]:
    c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDDDDD")
for i, it in enumerate(items, 1):
    ws.append([i, it["artifact"], it["block_kind"], it["line"], it["call"], it["code"], "", "", ""])
for col, w in zip("ABCDEFGHI", (6, 44, 10, 6, 40, 100, 14, 22, 30)):
    ws.column_dimensions[col].width = w
for row in ws.iter_rows(min_row=2):
    for c in row:
        c.alignment = Alignment(wrap_text=True, vertical="top")
g = wb.create_sheet("Guide")
for line in [
    "Validation of the reproducibility-violation (RVH) oracle. Work alone; do not discuss items before both sheets are done.",
    "For each item, read the flagged call in context (open the artifact file if needed) and answer:",
    "TP = this call makes two runs of the pipeline or of the tests give different results (no seed reaches it).",
    "FP = the call is in fact reproducible (seed passed elsewhere, global np.random.seed set before, deterministic by default, e.g. KFold without shuffle).",
    "? = cannot be decided from the artifact (e.g. the seed comes from a file the model did not generate).",
    "C items (whole artifact): list every unseeded source of randomness the detector missed, with its line; write 0 if none.",
    "Python facts: scikit-learn estimators with random_state=None use NumPy's global generator; np.random.default_rng() ignores np.random.seed;",
    "KFold/StratifiedKFold are deterministic unless shuffle=True; XGBoost's default seed is fixed (0).",
]:
    g.append([line])
g.column_dimensions["A"].width = 160
wb.save("annotation/RVH_annotation_sheet.xlsx")
with open("annotation/KEY_do_not_open.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["item", "source", "model", "strategy", "rule"])
    for i, it in enumerate(items, 1):
        w.writerow([i, it["source"], it["model"], it["strategy"], it["rule"]])
from collections import Counter
print(len(items), "items", Counter(it["source"] for it in items), Counter((it["source"], it["model"]) for it in items))
