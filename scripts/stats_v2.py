# -*- coding: utf-8 -*-
"""Statistics for the revised paper, from artifacts_v2.csv (run reanalyze_v2.py first). No API call.

    py -3.13 stats_v2.py   -> stats_v2.json and a printed summary

Reported, per model (GPT-4o, Claude), pooled over prompts and per prompt:
  - RVH prevalence (v1 = published detector, v2 = revised detector + manual traces) with the risk
    difference SF-OS (Newcombe hybrid score 95% CI), odds ratio (95% CI) and Fisher's exact p;
  - logistic regression RVH ~ strategy * prompt (+ batch for GPT-4o);
  - negative binomial regression of the number of v2 violations with offset log(LOC);
  - test functions: means, ratio SF/OS with bootstrap 95% CI, Cliff's delta;
  - location of v2 violations (pipeline vs tests) per strategy; IIH extension; plan adherence;
  - batch effect (GPT-4o), prompt effect (both analyses), cost in USD, workflow checks;
  - a-priori power for a one-sided reduction, from the 20% baseline of the earlier study.
"""
import json, math, warnings
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
RNG_SEED = 42
d = pd.read_csv("artifacts_v2.csv")
d = d[d.strategy.isin(["one_shot", "structure_first"])]   # main study only (ablation analysed separately)

# Manual traces of the unresolved unpacked-dictionary calls (kwargs_manual_check*.csv):
# GPT-4o: 2 seeded, 4 violations (already counted by v2 in the same artifacts); Claude: 2 undetermined.
d["rvh"] = d["rvh_v2"]
d["sf"] = (d.strategy == "structure_first").astype(int)
d["under"] = (d.prompt == "under_specified").astype(int)
d["viol_v2"] = d.v2_viol_pipeline + d.v2_viol_test


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1, n1, k0, n0):
    """95% CI of p1 - p0 (Newcombe hybrid score, method 10)."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1); l0, u0 = wilson(k0, n0)
    dlt = p1 - p0
    return dlt, dlt - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), dlt + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def odds_ratio(k1, n1, k0, n0):
    a, b, c, e = k1 + .5, n1 - k1 + .5, k0 + .5, n0 - k0 + .5      # Haldane correction
    lor = math.log(a * e / (b * c)); se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / e)
    return math.exp(lor), math.exp(lor - 1.96 * se), math.exp(lor + 1.96 * se)


def compare_prop(sub, col):
    sf, os_ = sub[sub.sf == 1][col], sub[sub.sf == 0][col]
    k1, n1, k0, n0 = int(sf.sum()), len(sf), int(os_.sum()), len(os_)
    rd, lo, hi = newcombe(k1, n1, k0, n0)
    orr, olo, ohi = odds_ratio(k1, n1, k0, n0)
    return {"sf": f"{k1}/{n1}", "os": f"{k0}/{n0}", "risk_diff": round(rd, 3), "rd_ci95": [round(lo, 3), round(hi, 3)],
            "odds_ratio": round(orr, 2), "or_ci95": [round(olo, 2), round(ohi, 2)],
            "fisher_p": round(fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1], 3)}


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x); lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def ratio_ci(x, y, n_boot=5000):
    rng = np.random.default_rng(RNG_SEED)
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = [rng.choice(x, len(x)).mean() / rng.choice(y, len(y)).mean() for _ in range(n_boot)]
    return x.mean() / y.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def numeric(sub, col):
    sf, os_ = sub[sub.sf == 1][col], sub[sub.sf == 0][col]
    r, lo, hi = ratio_ci(sf, os_)
    u = mannwhitneyu(sf, os_)
    return {"sf_mean": round(sf.mean(), 2), "os_mean": round(os_.mean(), 2), "ratio": round(r, 2),
            "ratio_ci95": [round(lo, 2), round(hi, 2)], "cliffs_delta": round(cliffs_delta(sf, os_), 2),
            "U": float(u.statistic), "mw_p": float(f"{u.pvalue:.2g}")}


def logit(sub, with_batch):
    f = "rvh ~ sf * under" + (" + C(batch)" if with_batch else "")
    try:
        m = smf.logit(f, data=sub).fit(disp=0)
        ci = m.conf_int()
        return {k: {"OR": round(math.exp(m.params[k]), 2), "ci95": [round(math.exp(ci.loc[k, 0]), 2), round(math.exp(ci.loc[k, 1]), 2)],
                    "p": round(m.pvalues[k], 3)} for k in m.params.index if k != "Intercept"}
    except Exception as e:
        return {"error": str(e)[:120]}


def negbin(sub):
    s = sub[sub["loc"] > 0].copy()
    try:
        m = smf.glm("viol_v2 ~ sf + under", data=s, family=sm.families.NegativeBinomial(alpha=1.0),
                    offset=np.log(s["loc"] / 1000)).fit()
        ci = m.conf_int()
        return {"rate_ratio_sf": round(math.exp(m.params["sf"]), 2),
                "ci95": [round(math.exp(ci.loc["sf", 0]), 2), round(math.exp(ci.loc["sf", 1]), 2)],
                "p": round(m.pvalues["sf"], 3),
                "rate_ratio_under": round(math.exp(m.params["under"]), 2), "p_under": round(m.pvalues["under"], 3)}
    except Exception as e:
        return {"error": str(e)[:120]}


def power_one_sided(p0, n, alpha=0.05, reps=4000):
    """Smallest p1 < p0 detectable with 80% power (one-sided Fisher), by simulation."""
    rng = np.random.default_rng(RNG_SEED)
    res = {}
    for p1 in np.arange(p0 - 0.02, -0.001, -0.02):
        a = rng.binomial(n, p1, reps); b = rng.binomial(n, p0, reps)
        pw = np.mean([fisher_exact([[x, n - x], [y, n - y]], alternative="less")[1] < alpha for x, y in zip(a, b)])
        res[round(float(p1), 2)] = round(float(pw), 3)
        if pw >= 0.8:
            return {"baseline": p0, "n_per_group": n, "min_detectable_p1": round(float(p1), 2), "power": round(float(pw), 3)}
    return {"baseline": p0, "n_per_group": n, "min_detectable_p1": None, "power_at_zero": res}


out = {}
for model, sub in d.groupby("model"):
    o = {"n": len(sub)}
    for name, s in [("pooled", sub)] + [(p, sub[sub.prompt == p]) for p in ("specified", "under_specified")]:
        o[name] = {"rvh_v1": compare_prop(s, "rvh_v1"), "rvh_v2": compare_prop(s, "rvh"),
                   "rvh_v2_undet_as_viol": compare_prop(s, "rvh_v2_undet_as_viol"),
                   "tests": numeric(s, "tests"), "tests_with_assert": numeric(s, "tests_with_assert"),
                   "asserts": numeric(s, "asserts"), "loc": numeric(s, "loc"),
                   "iih_ext": compare_prop(s, "iih_ext"), "iih_X1": compare_prop(s, "iih_X1_any"),
                   "iih_X2": compare_prop(s, "iih_X2_any")}
    o["logit"] = logit(sub, with_batch=(model == "gpt-4o"))
    o["negbin_v2"] = negbin(sub)
    # prompt effect, both analyses
    o["prompt_effect"] = {}
    for col in ("rvh_v1", "rvh"):
        u, sp = sub[sub.under == 1][col], sub[sub.under == 0][col]
        o["prompt_effect"][col] = {"under": f"{int(u.sum())}/{len(u)}", "specified": f"{int(sp.sum())}/{len(sp)}",
                                   "fisher_p": round(fisher_exact([[u.sum(), len(u) - u.sum()], [sp.sum(), len(sp) - sp.sum()]])[1], 3)}
    # location of v2 violations
    loc = {}
    for st, s in sub.groupby("strategy"):
        loc[st] = {"viol_pipeline": int(s.v2_viol_pipeline.sum()), "viol_test": int(s.v2_viol_test.sum()),
                   "artifacts_pipeline_viol": int((s.v2_viol_pipeline > 0).sum()),
                   "artifacts_test_viol": int((s.v2_viol_test > 0).sum()), "n": len(s),
                   "loc_test_share": round(s.loc_test.sum() / s["loc"].sum(), 2),
                   "viol_per_kloc_pipeline": round(1000 * s.v2_viol_pipeline.sum() / max(1, (s["loc"] - s.loc_test).sum()), 2),
                   "viol_per_kloc_test": round(1000 * s.v2_viol_test.sum() / max(1, s.loc_test.sum()), 2)}
    o["location_v2"] = loc
    # plan adherence (structure-first only)
    s = sub[sub.sf == 1]
    o["plan"] = {"planned_tests_mean": round(s.planned_tests.mean(), 2),
                 "planned_written_share": round(s.planned_written.sum() / max(1, s.planned_tests.sum()), 3),
                 "unplanned_written_mean": round(s.unplanned_written.mean(), 2),
                 "skeleton_mentions_seed": f"{int(s.skeleton_mentions_seed.sum())}/{len(s)}",
                 "rvh_if_skeleton_seed": f"{int(s[s.skeleton_mentions_seed == 1].rvh.sum())}/{int((s.skeleton_mentions_seed == 1).sum())}",
                 "rvh_if_no_skeleton_seed": f"{int(s[s.skeleton_mentions_seed == 0].rvh.sum())}/{int((s.skeleton_mentions_seed == 0).sum())}"}
    # cost
    o["cost"] = {st: {"usd_mean": round(s.cost_usd.mean(), 4), "tokens_in_mean": round(s.tokens_in.mean()),
                      "tokens_out_mean": round(s.tokens_out.mean()), "seconds_mean": round(s.seconds.mean(), 1),
                      "usd_per_rvh_free_artifact": round(s.cost_usd.sum() / max(1, (s.rvh == 0).sum()), 4)}
                 for st, s in sub.groupby("strategy")}
    # workflows
    o["yaml"] = {st: {"artifacts_with_yaml": int((s.yaml_n_yaml > 0).sum()), "all_parse": int((s.yaml_yaml_parses == s.yaml_n_yaml).sum()),
                      "runs_pytest": int((s.yaml_runs_pytest > 0).sum()), "installs_deps": int((s.yaml_installs_deps > 0).sum()),
                      "sha_pinned_actions": int(s.yaml_actions_sha_pinned.sum()),
                      "tag_pinned_actions": int(s.yaml_actions_tag_pinned.sum()),
                      "unpinned_actions": int(s.yaml_actions_unpinned.sum())}
                 for st, s in sub.groupby("strategy")}
    if model == "gpt-4o":
        o["batch_effect"] = {}
        for st, s in sub.groupby("strategy"):
            b1, b2 = s[s.batch == "b1"], s[s.batch == "b2"]
            o["batch_effect"][st] = {"rvh_b1": f"{int(b1.rvh.sum())}/{len(b1)}", "rvh_b2": f"{int(b2.rvh.sum())}/{len(b2)}",
                                     "rvh_p": round(fisher_exact([[b1.rvh.sum(), len(b1) - b1.rvh.sum()], [b2.rvh.sum(), len(b2) - b2.rvh.sum()]])[1], 3),
                                     "tests_b1": round(b1.tests.mean(), 2), "tests_b2": round(b2.tests.mean(), 2),
                                     "tests_p": float(f"{mannwhitneyu(b1.tests, b2.tests).pvalue:.2g}")}
    out[model] = o

out["power_a_priori"] = {"gpt4o_pooled": power_one_sided(0.20, 80), "gpt4o_per_prompt": power_one_sided(0.20, 40),
                         "claude_pooled": power_one_sided(0.20, 20),
                         "note": "baseline 20% = specified prompt, T=1.0, GPT-4o, in the earlier detection study (2 of 10)"}
json.dump(out, open("stats_v2.json", "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:12000])
