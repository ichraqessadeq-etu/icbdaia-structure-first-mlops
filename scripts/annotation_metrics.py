# -*- coding: utf-8 -*-
"""Validation of the RVH oracle from two independent annotations (no adjudication meeting).

Input: results/annotation/annotation_verdicts.csv (172 items, anonymised: verdicts only, no names, no comments).
Rules: Cohen's kappa on the raw labels (TP / FP / ?) of the 157 items with a verdict (A_v2: v2 findings,
B_v1_only: v1 findings removed by v2). An item counts as correct or wrong only when both annotators agree;
disagreements are reported apart and give a range. For the 15 artifacts without findings (C_negative), a missed
violation counts when at least one annotator reports one (conservative for the detector).
    cd results && python ../scripts/annotation_metrics.py   -> annotation_metrics.json
"""
import collections, csv, json

nm = lambda v: v.strip().upper()
rows = list(csv.DictReader(open("annotation/annotation_verdicts.csv", encoding="utf-8")))

pairs = [(r, nm(r["verdict_annotator_1"]), nm(r["verdict_annotator_2"])) for r in rows if r["source"] != "C_negative"]
n = len(pairs); po = sum(x == y for _, x, y in pairs) / n
ca = collections.Counter(x for _, x, _ in pairs); cb = collections.Counter(y for _, _, y in pairs)
pe = sum(ca[l] * cb[l] for l in ("TP", "FP", "?")) / n / n
out = {"n_items_with_verdict": n, "raw_agreement": round(po, 3), "cohen_kappa": round((po - pe) / (1 - pe), 3)}

for src, good in (("A_v2", "TP"), ("B_v1_only", "FP")):
    for model in ("gpt-4o", "claude-sonnet-4-5", "all"):
        sub = [(x, y) for k, x, y in pairs if k["source"] == src and (model == "all" or k["model"] == model)]
        agree_good = sum(x == y == good for x, y in sub)
        agree_bad = sum(x == y and x in ("TP", "FP") and x != good for x, y in sub)
        agree_undet = sum(x == y == "?" for x, y in sub)
        disagree = sum(x != y for x, y in sub)
        decided = agree_good + agree_bad
        out[f"{src}_{model}"] = {"n": len(sub), "agreed_correct": agree_good, "agreed_wrong": agree_bad,
                                 "agreed_undetermined": agree_undet, "disagreements": disagree,
                                 "share_correct_among_agreed_decided": round(agree_good / decided, 3) if decided else None,
                                 "range_if_disagreements_all_wrong_or_all_correct":
                                     [round(agree_good / len(sub), 3), round((agree_good + disagree + agree_undet) / len(sub), 3)] if sub else None}
c_items = [r for r in rows if r["source"] == "C_negative"]
missed = lambda v: v.strip() not in ("", "0", "?")
out["C_negative"] = {"n_artifacts": len(c_items),
                     "missed_by_A": sum(missed(r["missed_annotator_1"]) for r in c_items),
                     "missed_by_B": sum(missed(r["missed_annotator_2"]) for r in c_items),
                     "missed_by_at_least_one": sum(missed(r["missed_annotator_1"]) or missed(r["missed_annotator_2"]) for r in c_items),
                     "items_with_misses": [int(r["item"]) for r in c_items if missed(r["missed_annotator_1"]) or missed(r["missed_annotator_2"])]}
json.dump(out, open("annotation_metrics.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
