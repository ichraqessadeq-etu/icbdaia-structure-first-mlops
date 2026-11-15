# -*- coding: utf-8 -*-
"""Per-artifact re-analysis after the pre-submission review (no API call).

For every saved artifact (GPT-4o batches 1-2, Claude rounds 1-10) this script records:
  - RVH with the published detector (v1) and with the revised detector (v2), split by location
    (pipeline code vs test code), and the v2 UNDETERMINED cases;
  - IIH v1 and the IIH extension (unresolved first-party imports, unpinned packages);
  - test functions, test functions with at least one assertion, assertions;
  - for structure-first: plan adherence (share of the tests planned in the skeleton that are written
    in step 3) and whether the skeleton mentions seeding;
  - static checks of the GitHub Actions workflow(s);
  - input and output tokens and their cost in USD (list prices).
Output: artifacts_v2.csv and artifacts_v2.json (one row per artifact).

    py -3.13 reanalyze_v2.py
"""
import ast, csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import extract_code_blocks, detect_rvh as detect_rvh_v1
from detectors_v2 import detect_rvh_v2, detect_iih_v2

STRATS = ("one_shot", "structure_first", "two_call", "structure_first_plan", "one_shot_seed")
PROMPTS = ("specified", "under_specified")
PRICE = {"gpt-4o": (2.5, 10.0), "claude-sonnet-4-5": (3.0, 15.0)}       # USD per million tokens (in, out)
FILE_RE = re.compile(r"([A-Za-z_][\w./-]*?)\.py\b")


def is_test_block(code):
    return bool(re.search(r"^\s*def test_\w*\s*\(", code, re.M)) or bool(re.search(r"^\s*import pytest|^\s*from pytest", code, re.M))


def provided_modules(text):
    mods = set()
    for m in FILE_RE.finditer(text):
        path = m.group(1).strip("./")
        parts = [p for p in re.split(r"[/\\]", path) if p]
        if parts:
            mods.add(parts[-1]); mods.add(parts[0])
    mods |= set(re.findall(r"([A-Za-z_]\w*)/", text))      # directories in file-tree listings
    return mods | {"conftest", "src", "tests", "app"}


def count_asserts(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0, 0
    n_assert, tests_with = 0, 0
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test"):
            k = 0
            for n in ast.walk(fn):
                if isinstance(n, ast.Assert):
                    k += 1
                elif isinstance(n, ast.Call):
                    name = ""
                    f = n.func
                    if isinstance(f, ast.Attribute):
                        name = f.attr
                    elif isinstance(f, ast.Name):
                        name = f.id
                    if name in ("raises", "warns", "approx") or name.startswith("assert"):
                        k += 1
            n_assert += k
            tests_with += k > 0
    return n_assert, tests_with


def test_names(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(re.findall(r"def (test_\w+)", code))
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test")}


def yaml_checks(blocks):
    import yaml
    out = {"n_yaml": len(blocks), "yaml_parses": 0, "has_jobs": 0, "runs_pytest": 0, "installs_deps": 0,
           "actions_tag_pinned": 0, "actions_sha_pinned": 0, "actions_unpinned": 0, "python_versions": set()}
    for b in blocks:
        try:
            doc = yaml.safe_load(b)
            ok = isinstance(doc, dict)
        except Exception:
            ok = False
        out["yaml_parses"] += ok
        if ok and isinstance(doc.get("jobs"), dict):
            out["has_jobs"] += 1
        out["runs_pytest"] += bool(re.search(r"\bpytest\b", b))
        out["installs_deps"] += bool(re.search(r"pip install", b))
        for u in re.findall(r"uses:\s*([\w./-]+)(?:@([\w.\-]+))?", b):
            if not u[1]:
                out["actions_unpinned"] += 1
            elif re.fullmatch(r"[0-9a-f]{40}", u[1]):
                out["actions_sha_pinned"] += 1
            else:
                out["actions_tag_pinned"] += 1
        for v in re.findall(r"python-version:\s*\[?\s*['\"]?([\d.]+)", b):
            out["python_versions"].add(v)
    out["python_versions"] = ";".join(sorted(out["python_versions"]))
    return out


def analyse(d, model, batch):
    a = json.load(open(f"{d}/analysis.json", encoding="utf-8"))
    sf = os.path.exists(f"{d}/final_artifact.txt")
    text = open(f"{d}/final_artifact.txt" if sf else f"{d}/response.txt", encoding="utf-8").read()
    py = extract_code_blocks(text, "python")
    yml = extract_code_blocks(text, "yaml") + extract_code_blocks(text, "yml")
    names_src = text
    has_skel = os.path.exists(f"{d}/step1_skeleton.txt")
    if has_skel:   # the file layout is fixed in the skeleton (step 1), which is not part of the analysed artifact
        names_src += "\n" + open(f"{d}/step1_skeleton.txt", encoding="utf-8").read()
    mods = provided_modules(names_src)
    r = {"model": model, "batch": batch, "strategy": d.replace("\\", "/").split("/")[-2].split("__")[0],
         "prompt": d.replace("\\", "/").split("/")[-2].split("__")[1], "artifact": d.replace("\\", "/"),
         "loc": 0, "loc_test": 0, "tests": 0, "tests_with_assert": 0, "asserts": 0,
         "v1_viol_pipeline": 0, "v1_viol_test": 0, "v2_viol_pipeline": 0, "v2_viol_test": 0,
         "v2_undetermined": 0, "v2_rules": {}, "iih_v1": 0, "iih_X1": 0, "iih_X2": 0, "iih_detail": [],
         "parse_fail_blocks": 0}
    for b in py:
        loc = sum(1 for l in b.splitlines() if l.strip() and not l.strip().startswith("#"))
        t = is_test_block(b)
        r["loc"] += loc
        r["loc_test"] += loc if t else 0
        where = "test" if t else "pipeline"
        v1 = detect_rvh_v1(b)
        if v1["parse_error"]:
            r["parse_fail_blocks"] += 1
            continue
        r[f"v1_viol_{where}"] += len(v1["violations"])
        v2 = detect_rvh_v2(b)
        for f in v2["findings"]:
            if f["kind"] == "RVH":
                r[f"v2_viol_{where}"] += 1
                r["v2_rules"][f["rule"]] = r["v2_rules"].get(f["rule"], 0) + 1
            else:
                r["v2_undetermined"] += 1
        iih = detect_iih_v2(b, mods)
        for f in iih["findings"]:
            if f["rule"].startswith("v1_"):
                r["iih_v1"] += 1
            elif f["rule"].startswith("X1"):
                r["iih_X1"] += 1
            else:
                r["iih_X2"] += 1
            r["iih_detail"].append(f"{f['rule']}:{f['detail']}")
        if t:
            n_a, tw = count_asserts(b)
            r["asserts"] += n_a
            r["tests_with_assert"] += tw
    r["tests"] = a["n_tests"]
    r["rvh_v1"] = int(a["is_rvh"])
    r["rvh_v2"] = int(r["v2_viol_pipeline"] + r["v2_viol_test"] > 0)
    r["rvh_v2_undet_as_viol"] = int(r["rvh_v2"] or r["v2_undetermined"] > 0)
    r["iih_ext"] = int(r["iih_v1"] + r["iih_X1"] + r["iih_X2"] > 0)
    r["iih_X1_any"] = int(r["iih_X1"] > 0)
    r["iih_X2_any"] = int(r["iih_X2"] > 0)
    r.update({f"yaml_{k}": v for k, v in yaml_checks(yml).items()})
    pin, pout = a["cost"]["prompt_tokens"], a["cost"]["completion_tokens"]
    r["tokens_in"], r["tokens_out"] = pin, pout
    r["cost_usd"] = round((pin * PRICE[model][0] + pout * PRICE[model][1]) / 1e6, 5)
    r["seconds"] = a["cost"].get("seconds")
    if has_skel:
        sk = open(f"{d}/step1_skeleton.txt", encoding="utf-8").read()
        planned = set(re.findall(r"def (test_\w+)", sk))
        s3 = open(f"{d}/step3_tests_ci.txt", encoding="utf-8").read()
        written = set().union(*[test_names(b) for b in extract_code_blocks(s3, "python")]) if extract_code_blocks(s3, "python") else set()
        r["planned_tests"] = len(planned)
        r["planned_written"] = len(planned & written)
        r["unplanned_written"] = len(written - planned)
        r["skeleton_mentions_seed"] = int(bool(re.search(r"random_state|seed", sk, re.I)))
    r["v2_rules"] = ";".join(f"{k}={v}" for k, v in sorted(r["v2_rules"].items()))
    r["iih_detail"] = " | ".join(r["iih_detail"])
    return r


B1 = "runs" if os.path.isdir("runs") else "runs_gpt4o_batch1"   # batch-1 folder (renamed in the repository)


def main():
    rows = []
    sources = [(B1, "gpt-4o", "b1", None), ("runs_gpt4o_batch2", "gpt-4o", "b2", None),
               ("runs_claude", "claude-sonnet-4-5", "c", 10), ("runs_ablation", "gpt-4o", "abl", 30)]
    for root, model, batch, max_round in sources:
        for s in STRATS:
            for p in PROMPTS:
                for d in sorted(glob.glob(f"{root}/{s}__{p}/run_*")):
                    if max_round and int(d.rsplit("_", 1)[1]) > max_round:
                        continue
                    rows.append(analyse(d, model, batch))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open("artifacts_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump(rows, open("artifacts_v2.json", "w", encoding="utf-8"), indent=1)
    print(len(rows), "artifacts ->", "artifacts_v2.csv")


if __name__ == "__main__":
    main()
