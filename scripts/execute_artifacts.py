# -*- coding: utf-8 -*-
"""Execute every generated artifact in the pinned environment and measure test outcomes and flakiness.

For each artifact (no API call):
  1. its Python blocks are written to files: a block whose first lines or preceding text name a file
     (e.g. "# fraud_pipeline.py", "### `src/train.py`") is written under that name; an unnamed test block
     becomes test_pipeline.py; an unnamed pipeline block is written under the first-party module name the
     tests import (or pipeline.py when they import none);
  2. the credit-card data (OpenML 1597 sub-sample, same columns as the ULB dataset) is placed at the usual
     paths (creditcard.csv, data/creditcard.csv, data/raw/creditcard.csv);
  3. pytest runs K times (default 3) in the pinned environment (Python 3.7.7, scikit-learn 1.0.2,
     pandas 1.3.5, NumPy 1.21.6, MLflow 1.30.1, pytest 7.4.4), each run in a fresh copy of the folder, with
     MLflow writing to a local file store; per-test outcomes are compared across runs.
Outcome per artifact: collection error (with its cause), all tests pass, some tests fail, timeout;
flaky tests = tests whose outcome differs between runs.

    py -3.13 execute_artifacts.py --workers 2 --runs 3        -> execution_results.json
    py -3.13 execute_artifacts.py --only runs/one_shot__specified/run_01   (one artifact, for checks)
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import extract_code_blocks

# Interpreter of the pinned environment (Python 3.7.7, see README) and path of the credit-card CSV:
PINNED_PY = os.environ.get("PINNED_PYTHON", sys.executable)
DATA = Path(os.environ.get("FRAUD_CSV", "data/creditcard_sample.csv"))
FENCE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<body>.*?)```", re.DOTALL)
NAME_IN_CODE = re.compile(r"^\s*#\s*(?:file(?:name)?:\s*)?([\w./-]+\.py)\b", re.I)
NAME_BEFORE = re.compile(r"([\w./-]+\.py)")
KNOWN_TOP = None


def stdlib_and_third():
    global KNOWN_TOP
    if KNOWN_TOP is None:
        from detectors_v2 import KNOWN_THIRD_PARTY
        KNOWN_TOP = set(sys.stdlib_module_names) | KNOWN_THIRD_PARTY
    return KNOWN_TOP


def python_blocks_with_names(text):
    out = []
    for m in FENCE.finditer(text):
        lang = (m.group("lang") or "").lower()
        if lang not in ("python", "py", "python3"):
            continue
        body = m.group("body")
        name = None
        for line in body.splitlines()[:3]:
            mm = NAME_IN_CODE.match(line)
            if mm:
                name = mm.group(1); break
        if not name:
            before = text[max(0, m.start() - 250):m.start()]
            cands = NAME_BEFORE.findall(before.splitlines()[-1] if before.strip() else "") or NAME_BEFORE.findall(before[-120:])
            if cands:
                name = cands[-1]
        out.append((name, body))
    return out


def first_party_imports(code):
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    known, mods = stdlib_and_third(), []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            top = n.module.split(".")[0]
            if top not in known and top not in mods:
                mods.append(n.module)
        elif isinstance(n, ast.Import):
            for a in n.names:
                top = a.name.split(".")[0]
                if top not in known and top not in mods:
                    mods.append(a.name)
    return mods


def layout(text, folder: Path):
    blocks = python_blocks_with_names(text)
    is_test = lambda c: bool(re.search(r"^\s*def test_\w*\s*\(", c, re.M))
    tests = [(n, c) for n, c in blocks if is_test(c)]
    pipes = [(n, c) for n, c in blocks if not is_test(c)]
    imported = []
    for _, c in tests:
        imported += [m for m in first_party_imports(c) if m not in imported]
    written = {}
    for n, c in pipes:
        if not n:
            n = (imported[0].replace(".", "/") + ".py") if imported else "pipeline.py"
            k = 2
            while n in written:
                n = n[:-3] + f"_{k}.py"; k += 1
        written[n] = c
    unnamed = 0
    for n, c in tests:
        if n and not Path(n).name.startswith("test"):
            written[n] = c            # a named module that also embeds tests keeps its name (it is imported)
            continue
        if not n:
            unnamed += 1
            n = "test_pipeline.py" if unnamed == 1 else f"test_pipeline_{unnamed}.py"
        written[n] = c
    for n, c in written.items():
        p = folder / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(c, encoding="utf-8")
        for parent in p.parents:          # make package dirs importable
            if parent == folder:
                break
            (parent / "__init__.py").touch(exist_ok=True)
    for rel in ("creditcard.csv", "data/creditcard.csv", "data/raw/creditcard.csv"):
        q = folder / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA, q)
    return {"files": sorted(written), "n_test_files": len(tests), "imported_first_party": imported}


OUT_RE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.M)


def run_pytest(folder: Path, timeout):
    env = dict(os.environ, MLFLOW_TRACKING_URI=(folder / "mlruns").as_uri(), PYTHONHASHSEED="random",
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    env.pop("PYTHONPATH", None)
    t0 = time.time()
    try:
        p = subprocess.run([PINNED_PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "--timeout=0"]
                           if False else [PINNED_PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider"],
                           cwd=folder, env=env, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"timeout": True, "seconds": timeout, "outcomes": {}}
    out = p.stdout + p.stderr
    outcomes = {tid: st for st, tid in OUT_RE.findall(out)}
    collect_err = re.findall(r"^E\s+(ModuleNotFoundError|ImportError|SyntaxError|NameError|AttributeError)[^\n]*", out, re.M)
    return {"timeout": False, "returncode": p.returncode, "seconds": round(time.time() - t0, 1),
            "outcomes": outcomes, "collection_errors": collect_err[:5],
            "interrupted": "Interrupted" in out or "errors during collection" in out,
            "tail": "\n".join(out.splitlines()[-12:]), "tail_full": out[-6000:]}


PINNED_TOP = {"pandas", "numpy", "sklearn", "scipy", "mlflow", "pytest", "joblib", "yaml", "matplotlib", "xgboost"}


def classify_collection_error(out):
    """Cause of a collection failure: a package outside the pinned environment (missing, or installed but
    unusable under Python 3.7), an unresolved first-party module, a syntax error in generated code, other."""
    from detectors_v2 import KNOWN_THIRD_PARTY
    m = re.search(r"No module named '([\w.]+)'", out)
    if m:
        top = m.group(1).split(".")[0]
        return "collection_missing_package" if top in KNOWN_THIRD_PARTY else "collection_unresolved_first_party"
    for m in re.finditer(r"site-packages[\\/](\w+)[\\/]", out):
        if m.group(1) not in PINNED_TOP and m.group(1) not in ("_pytest", "pluggy", "importlib"):
            return "collection_missing_package"
    if "SyntaxError" in out:
        return "collection_syntax_error"
    if re.search(r"\b(ImportError|NameError|AttributeError)\b", out):
        return "collection_import_or_name_error"
    return "collection_other"


def repair_names_and_run(template: Path, text: str, timeout, max_passes=6):
    """Lenient check for artifacts whose tests import a first-party module that does not exist:
    every missing module is created with the whole non-test code of the artifact, then pytest runs once.
    It separates naming / interface defects from tests that actually fail."""
    folder = template.parent / "lenient"
    shutil.copytree(template, folder)
    pipeline_code = "\n\n".join(c for n, c in python_blocks_with_names(text)
                                if not re.search(r"^\s*def test_\w*\s*\(", c, re.M)) or "pass\n"
    created = []
    for _ in range(max_passes):
        r = run_pytest(folder, timeout)
        m = re.search(r"No module named '([\w.]+)'", r.get("tail_full", ""))
        if r["timeout"] or not m:
            break
        mod = m.group(1)
        path = folder / (mod.replace(".", "/") + ".py")
        if path.exists() or mod in created:
            break
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pipeline_code, encoding="utf-8")
        for parent in path.parents:
            if parent == folder:
                break
            (parent / "__init__.py").touch(exist_ok=True)
        created.append(mod)
    oc = r.get("outcomes", {})
    if r["timeout"]:
        cat = "timeout"
    elif oc and all(v in ("PASSED", "SKIPPED", "XFAIL") for v in oc.values()):
        cat = "all_pass"
    elif oc:
        cat = "some_fail"
    else:
        cat = classify_collection_error(r.get("tail_full", ""))
    shutil.rmtree(folder, ignore_errors=True)
    return {"modules_created": created, "category": cat, "n_tests_run": len(oc),
            "n_passed": sum(v == "PASSED" for v in oc.values())}


def execute(artifact, runs, timeout, workdir: Path):
    text_file = Path(artifact, "final_artifact.txt")
    if not text_file.exists():
        text_file = Path(artifact, "response.txt")
    text = text_file.read_text(encoding="utf-8")
    base = workdir / artifact.replace("/", "__")
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    lay = layout(text, base / "src_template")
    res = []
    for k in range(runs):
        f = base / f"run{k + 1}"
        shutil.copytree(base / "src_template", f)
        res.append(run_pytest(f, timeout))
        shutil.rmtree(f, ignore_errors=True)
    tests = sorted(set().union(*[set(r["outcomes"]) for r in res]))
    flaky = [t for t in tests if len({r["outcomes"].get(t, "MISSING") for r in res}) > 1]
    first = res[0]
    if first["timeout"]:
        cat = "timeout"
    elif first.get("interrupted") or (not first["outcomes"] and first.get("returncode") not in (0, 5)):
        cat = classify_collection_error(first.get("tail_full", ""))
    elif first.get("returncode") == 5 or not first["outcomes"]:
        cat = "no_tests_collected"
    elif all(v in ("PASSED", "SKIPPED", "XFAIL") for v in first["outcomes"].values()):
        cat = "all_pass"
    else:
        cat = "some_fail"
    n = len(first["outcomes"])
    passed = sum(v == "PASSED" for v in first["outcomes"].values())
    lenient = None
    if cat == "collection_unresolved_first_party":
        lenient = repair_names_and_run(base / "src_template", text, timeout)
    shutil.rmtree(base, ignore_errors=True)
    return {"artifact": artifact, "layout": lay, "category": cat, "n_tests_run": n, "n_passed_run1": passed,
            "pass_rate_run1": round(passed / n, 3) if n else None, "flaky_tests": flaky, "n_flaky": len(flaky),
            "artifact_flaky": int(bool(flaky)), "runs": [{k: v for k, v in r.items() if k not in ("outcomes", "tail_full")} for r in res],
            "outcomes_run1": first["outcomes"], "names_repaired": lenient}


B1 = "runs" if os.path.isdir("runs") else "runs_gpt4o_batch1"   # batch-1 folder (renamed in the repository)


def all_artifacts():
    import glob
    arts = []
    for root, max_round in ((B1, None), ("runs_gpt4o_batch2", None), ("runs_claude", 10)):
        for d in sorted(glob.glob(f"{root}/*__*/run_*")):
            if max_round and int(d.rsplit("_", 1)[1]) > max_round:
                continue
            arts.append(d.replace("\\", "/"))
    return arts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--only", default=None)
    ap.add_argument("--roots", default=None, help="comma list of run folders (default: the 200 analysed artifacts)")
    ap.add_argument("--out", default="execution_results.json")
    a = ap.parse_args()
    workdir = Path(tempfile.gettempdir()) / "icbdaia_exec"
    workdir.mkdir(exist_ok=True)
    if a.only:
        arts = [a.only]
    elif a.roots:
        import glob
        arts = [d.replace("\\", "/") for r in a.roots.split(",") for d in sorted(glob.glob(f"{r}/*__*/run_*"))]
    else:
        arts = all_artifacts()
    done = {}
    if Path(a.out).exists() and not a.only:
        done = {r["artifact"]: r for r in json.load(open(a.out, encoding="utf-8"))}
    todo = [x for x in arts if x not in done]
    print(f"{len(arts)} artifacts, {len(todo)} to run", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(execute, x, a.runs, a.timeout, workdir): x for x in todo}
        for i, f in enumerate(as_completed(futs), 1):
            x = futs[f]
            try:
                r = f.result()
            except Exception as e:
                r = {"artifact": x, "category": "harness_error", "error": f"{type(e).__name__}: {e}"}
            done[x] = r
            print(f"[{i}/{len(todo)}] {x}: {r['category']} tests={r.get('n_tests_run')} "
                  f"passed={r.get('n_passed_run1')} flaky={r.get('n_flaky')}", flush=True)
            if not a.only:
                json.dump(list(done.values()), open(a.out, "w", encoding="utf-8"), indent=1)
    if a.only:
        print(json.dumps(done[a.only], indent=1)[:4000])


if __name__ == "__main__":
    main()
