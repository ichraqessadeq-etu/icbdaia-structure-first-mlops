# -*- coding: utf-8 -*-
"""Statistics for the structure-first study, computed from saved runs (no API call).

Every folder named runs*/ that holds a results.json is read; folders whose name contains
"FAILED" or that are dry runs are ignored. Batches of the same model are pooled.

    py -3.13 analyze_results.py        -> prints the tables, writes statistics.json
"""
import ast, glob, json, os, sys
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import extract_code_blocks

# Sensitivity analysis: an estimator built from an unpacked dict, e.g. RandomForestClassifier(**params),
# cannot be judged statically (random_state may be a key of the dict). With STRICT=1 such
# "unseeded_estimator" violations are dropped; the detector itself is unchanged.
STRICT = os.environ.get("STRICT") == "1"


def _kwargs_unpack_lines(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    return {n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and any(k.arg is None for k in n.keywords)}

STRATS, PROMPTS = ("one_shot", "structure_first"), ("specified", "under_specified")
BOOT_SEED = 42   # each comparison draws from its own generator: results do not depend on folder order


def artifact(d):
    a = json.load(open(f"{d}/analysis.json", encoding="utf-8"))
    f = f"{d}/final_artifact.txt" if os.path.exists(f"{d}/final_artifact.txt") else f"{d}/response.txt"
    py = extract_code_blocks(open(f, encoding="utf-8").read(), "python")
    loc = sum(1 for b in py for l in b.splitlines() if l.strip() and not l.strip().startswith("#"))
    viol, rvh = 0, a["is_rvh"]
    if STRICT:
        for i, bl in enumerate(a["blocks"]):
            if not bl["parses"]:
                continue
            unpack = _kwargs_unpack_lines(py[i]) if i < len(py) else set()
            viol += sum(1 for v in bl["rvh"]["violations"]
                        if not (v["rule"] == "unseeded_estimator" and v["lineno"] in unpack))
        rvh = viol > 0
    else:
        viol = sum(len(bl["rvh"]["violations"]) for bl in a["blocks"] if bl["parses"])
    return {"invalid": a["is_invalid"], "rvh": rvh, "iih": a["is_iih"], "tests": a["n_tests"],
            "complete": a["complete"], "loc": loc, "viol": viol,
            "tokens": a["cost"]["prompt_tokens"] + a["cost"]["completion_tokens"],
            "errors": len(a["cost"]["errors"])}


def collect():
    groups = {}
    for root in sorted(glob.glob("runs*")):
        if "FAILED" in root or not os.path.exists(f"{root}/results.json"):
            continue
        meta = json.load(open(f"{root}/results.json", encoding="utf-8"))["meta"]
        if meta.get("dry_run"):
            continue
        g = groups.setdefault(meta["model"], {"batches": [], "cells": {}})
        g["batches"].append(root)
        n_done = meta.get("n_completed_rounds")   # round-based runs: ignore an incomplete last round
        for s in STRATS:
            for p in PROMPTS:
                dirs = sorted(glob.glob(f"{root}/{s}__{p}/run_*"))
                if n_done is not None:
                    dirs = [d for d in dirs if int(d.rsplit("_", 1)[1]) <= n_done]
                g["cells"].setdefault((s, p), []).extend(artifact(d) for d in dirs)
    return groups


def rate_ratio_ci(os_, sf, n_boot=5000):
    """Cluster (artifact-level) bootstrap 95% CI of the violations-per-LOC rate ratio SF/OS."""
    rng = np.random.default_rng(BOOT_SEED)
    def rr(a, b):
        va, la = sum(r["viol"] for r in a), sum(r["loc"] for r in a)
        vb, lb = sum(r["viol"] for r in b), sum(r["loc"] for r in b)
        return (vb / lb) / (va / la) if va and lb and la else np.nan
    boots = [rr([os_[i] for i in rng.integers(0, len(os_), len(os_))],
                [sf[i] for i in rng.integers(0, len(sf), len(sf))]) for _ in range(n_boot)]
    boots = np.array(boots)[~np.isnan(boots)]
    return rr(os_, sf), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) else (None, None)


def compare(os_, sf):
    n1, n2 = len(os_), len(sf)
    a, b = sum(r["rvh"] for r in os_), sum(r["rvh"] for r in sf)
    v1, l1 = sum(r["viol"] for r in os_), sum(r["loc"] for r in os_)
    v2, l2 = sum(r["viol"] for r in sf), sum(r["loc"] for r in sf)
    dens = lambda rows: [1000 * r["viol"] / r["loc"] if r["loc"] else 0.0 for r in rows]
    mw = lambda k: float(f"{mannwhitneyu([r[k] for r in os_], [r[k] for r in sf]).pvalue:.2g}")
    rr, ci = rate_ratio_ci(os_, sf)
    return {
        "n": {"one_shot": n1, "structure_first": n2},
        "invalid": {"one_shot": sum(r["invalid"] for r in os_), "structure_first": sum(r["invalid"] for r in sf)},
        "iih": {"one_shot": sum(r["iih"] for r in os_), "structure_first": sum(r["iih"] for r in sf)},
        "rvh_artifacts": {"one_shot": f"{a}/{n1}", "structure_first": f"{b}/{n2}",
                          "fisher_p": round(fisher_exact([[a, n1 - a], [b, n2 - b]])[1], 3)},
        "tests_mean": {"one_shot": round(np.mean([r['tests'] for r in os_]), 2),
                       "structure_first": round(np.mean([r['tests'] for r in sf]), 2), "mannwhitney_p": mw("tests")},
        "loc_mean": {"one_shot": round(np.mean([r['loc'] for r in os_]), 1),
                     "structure_first": round(np.mean([r['loc'] for r in sf]), 1), "mannwhitney_p": mw("loc")},
        "rvh_per_kloc": {
            "one_shot": round(1000 * v1 / l1, 2) if l1 else None,
            "structure_first": round(1000 * v2 / l2, 2) if l2 else None,
            "violations": {"one_shot": v1, "structure_first": v2},
            "rate_ratio": round(rr, 2) if rr == rr else None,
            "rate_ratio_ci95_cluster_bootstrap": [round(ci[0], 2), round(ci[1], 2)] if ci[0] is not None else None,
            "poisson_rate_test_p": round(binomtest(v2, v1 + v2, l2 / (l1 + l2)).pvalue, 4) if v1 + v2 else None,
            "per_artifact_density_mannwhitney_p": float(f"{mannwhitneyu(dens(os_), dens(sf)).pvalue:.2g}"),
        },
        "tokens_mean": {"one_shot": round(np.mean([r['tokens'] for r in os_])),
                        "structure_first": round(np.mean([r['tokens'] for r in sf]))},
        "api_errors": sum(r["errors"] for r in os_ + sf),
    }


def main():
    groups = collect()
    if not groups:
        sys.exit("no completed run folder found")
    out = {}
    for model, g in groups.items():
        out[model] = {"batches": g["batches"]}
        for p in PROMPTS:
            out[model][p] = compare(g["cells"][("one_shot", p)], g["cells"][("structure_first", p)])
        pooled = {s: g["cells"][(s, "specified")] + g["cells"][(s, "under_specified")] for s in STRATS}
        out[model]["pooled_prompts"] = compare(pooled["one_shot"], pooled["structure_first"])
    json.dump(out, open("statistics_strict.json" if STRICT else "statistics.json", "w", encoding="utf-8"), indent=2)
    for model, res in out.items():
        print(f"\n===== {model}  ({', '.join(res['batches'])})")
        for k in (*PROMPTS, "pooled_prompts"):
            r = res[k]; d = r["rvh_per_kloc"]
            print(f"{k:16s} n={r['n']['one_shot']}/{r['n']['structure_first']} | RVH artifacts "
                  f"{r['rvh_artifacts']['one_shot']} vs {r['rvh_artifacts']['structure_first']} (p={r['rvh_artifacts']['fisher_p']}) | "
                  f"tests {r['tests_mean']['one_shot']} vs {r['tests_mean']['structure_first']} (p={r['tests_mean']['mannwhitney_p']}) | "
                  f"RVH/kLOC {d['one_shot']} vs {d['structure_first']} RR={d['rate_ratio']} CI{d['rate_ratio_ci95_cluster_bootstrap']} "
                  f"p_art={d['per_artifact_density_mannwhitney_p']} | tokens {r['tokens_mean']['one_shot']} vs {r['tokens_mean']['structure_first']}")
    print("\nsaved -> statistics.json")


if __name__ == "__main__":
    main()
