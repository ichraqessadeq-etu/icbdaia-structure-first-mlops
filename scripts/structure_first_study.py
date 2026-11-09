# -*- coding: utf-8 -*-
"""
Structure-first vs one-shot LLM generation of MLOps CI/CD artifacts.

Question: does generating a pipeline the way an engineer builds it -- skeleton first
(modules, function signatures, test names, workflow steps), then implementation, then
tests and workflow -- produce fewer operational hallucinations than one-shot generation?

Design (2 x 2, N runs per cell, one fixed temperature):
    strategy  : one_shot | structure_first (3 calls: skeleton -> implementation -> tests+CI)
    prompt    : specified | under_specified
Oracle: the RVH / IIH detectors of the MMC study (detectors.py, unchanged).
Extra static measures: completeness (pipeline + pytest + workflow present), number of
tests, MLflow use, error handling, seeding, cost (calls, tokens, seconds).

Everything is saved: every raw response, the assembled artifact, token usage.

    python structure_first_study.py --dry-run     # offline check, fake LLM, no API key
    python structure_first_study.py               # real run (OPENAI_API_KEY in env or .env)
    python structure_first_study.py --n 20 --temperature 0.7 --model gpt-4o
    python structure_first_study.py --reanalyze   # re-score saved responses, no API call
    python structure_first_study.py --provider anthropic --model claude-sonnet-4-5 --out runs_claude
    # a second batch of the same model goes to its own folder (e.g. --out runs_gpt4o_batch2);
    # analyze_results.py pools every runs*/ folder by model.
"""
from __future__ import annotations
import argparse, ast, json, os, sys, time, datetime, hashlib, platform
from pathlib import Path

from detectors import extract_code_blocks, detect_rvh, detect_iih

THIRD_PARTY = {"pandas", "numpy", "sklearn", "scipy", "mlflow", "pytest",
               "joblib", "xgboost", "matplotlib", "yaml"}           # same pinned set as MMC
PINNED_ENV = set(getattr(sys, "stdlib_module_names", set())) | THIRD_PARTY

# ---------------------------------------------------------------------------
# Prompts. The task text is the MMC Module-2 task, so prompt specification is
# held identical to the earlier study; only the generation strategy changes.
# ---------------------------------------------------------------------------
TASK_SPECIFIED = """Generate a complete CI/CD pipeline configuration for a fraud detection model.

Requirements:
- Model: scikit-learn RandomForestClassifier
- Dataset: Credit card fraud (imbalanced, 0.17% fraud rate)
- Tools: MLflow for experiment tracking, pytest for testing
- Triggers: automatic retraining when concept drift is detected

Deliverables:
1. A complete Python pipeline script (data loading -> training -> evaluation -> logging to MLflow)
2. A pytest test file with at least 3 tests
3. A GitHub Actions YAML workflow

Be complete and production-ready. Include error handling."""

TASK_UNDER = "Write me a CI/CD pipeline for my fraud detection model."

SYSTEM = "You are an expert MLOps engineer."
MAX_TOKENS = 2500           # per-call output cap; set by --max-tokens (GPT-4o runs used 2500, never reached)
API_VERSION = "2023-06-01"   # Anthropic API version pinned for the Claude arm (see AnthropicLLM)
PRICES_USD_PER_MTOK = {"claude-sonnet-4-5": (3.0, 15.0), "claude-haiku-4-5": (1.0, 5.0), "gpt-4o": (2.5, 10.0)}   # (input, output), for --budget-usd

STRUCT_STEP1 = """{task}

The deliverables are: the full Python pipeline module, the complete pytest test file and the
GitHub Actions workflow.
Work like an engineer: do NOT write the implementation yet.
Step 1 - produce ONLY the structure:
- the file layout,
- for the pipeline module: every function signature with a one-line docstring and `...` as body,
- for the test file: the test function names with a one-line docstring each,
- for the CI workflow: the ordered list of jobs and steps (as a short YAML outline).
Use fenced code blocks."""

STRUCT_STEP2 = """Here is the agreed structure of the pipeline:

{skeleton}

Step 2 - implement the pipeline module completely, following this structure exactly
(same functions, same signatures). Return the full Python module in one ```python block."""

STRUCT_STEP3 = """Here is the implemented pipeline module:

```python
{implementation}
```

Step 3 - write (a) the complete pytest test file implementing the planned tests and importing
from this module, in one ```python block, and (b) the complete GitHub Actions workflow in one
```yaml block, following the planned jobs and steps."""

# ---------------------------------------------------------------------------
# LLM client (real or fake)
# ---------------------------------------------------------------------------
class FakeLLM:
    """Offline stand-in: returns canned, deterministic responses to test the pipeline."""
    model = "fake"
    def __call__(self, prompt, temperature):
        if "Step 1" in prompt:
            text = ("```python\ndef load_data(path):\n    \"\"\"Load CSV.\"\"\"\n    ...\n"
                    "def train(df):\n    \"\"\"Train model.\"\"\"\n    ...\n```\n"
                    "```python\ndef test_load():\n    \"\"\"loads\"\"\"\n    ...\n```\n"
                    "```yaml\njobs: [test, train]\n```")
        elif "Step 2" in prompt:
            text = ("```python\nimport pandas as pd\nimport mlflow\n"
                    "from sklearn.ensemble import RandomForestClassifier\n"
                    "from sklearn.model_selection import train_test_split\n\n"
                    "def load_data(path):\n    try:\n        return pd.read_csv(path)\n"
                    "    except FileNotFoundError:\n        raise\n\n"
                    "def train(df):\n    X, y = df.drop(columns=['Class']), df['Class']\n"
                    "    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                    "    m = RandomForestClassifier(random_state=42).fit(Xtr, ytr)\n"
                    "    mlflow.log_metric('score', m.score(Xte, yte))\n    return m\n```")
        elif "Step 3" in prompt:
            text = ("```python\nimport pytest\nfrom pipeline import load_data\n\n"
                    "def test_a():\n    assert True\n\ndef test_b():\n    assert True\n\n"
                    "def test_c():\n    assert True\n```\n```yaml\nname: ci\non: [push]\n```")
        else:  # one-shot; under-specified prompt -> unseeded model to exercise RVH
            seed = "" if "Write me" in prompt else ", random_state=42"
            text = ("```python\nimport pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\n"
                    f"m = RandomForestClassifier(n_estimators=10{seed})\n```")
        return text, {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4}


class OpenAILLM:
    def __init__(self, model):
        try:
            from dotenv import load_dotenv; load_dotenv()
        except Exception:
            pass
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            from getpass import getpass
            key = getpass("OPENAI_API_KEY (invisible): ")
        self.client, self.model = OpenAI(api_key=key), model

    def check(self, temperature=0.7):
        """Fail fast on an invalid/expired key or unknown model, before any study call."""
        try:
            self.client.chat.completions.create(model=self.model, max_tokens=1,
                                                messages=[{"role": "user", "content": "ping"}])
        except Exception as e:
            sys.exit(f"API check failed, nothing was run: {type(e).__name__}: {e}")

    def __call__(self, prompt, temperature, retries=3):
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=temperature, max_tokens=MAX_TOKENS,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}])
                u = r.usage
                return r.choices[0].message.content, {"prompt_tokens": u.prompt_tokens,
                                                      "completion_tokens": u.completion_tokens,
                                                      "truncated": r.choices[0].finish_reason == "length"}
            except Exception as e:                     # keep the run alive, record the error
                err = f"{type(e).__name__}: {e}"
                if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError",
                                        "BadRequestError", "TypeError"):
                    sys.exit(f"Fatal API error, run stopped: {err}")
                time.sleep(2 * (attempt + 1))
        return f"ERROR: {err}", {"prompt_tokens": 0, "completion_tokens": 0, "error": err}


class AnthropicLLM:
    """Second generator family, same prompts, same temperature, same token budget."""
    def __init__(self, model):
        try:
            from dotenv import load_dotenv; load_dotenv()
        except Exception:
            pass
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            from getpass import getpass
            key = getpass("ANTHROPIC_API_KEY (invisible): ")
        # SDK >= 2.0 sends API version 2026-10-21, which rejects `temperature` on every model.
        # To keep T identical to the GPT-4o arm we pin the previous version (on which
        # claude-sonnet-4-5 accepts temperature) and pass temperature through extra_body.
        self.client = anthropic.Anthropic(api_key=key, default_headers={"anthropic-version": API_VERSION})
        self.model = model

    def _create(self, prompt, temperature, max_tokens, system=None):
        kw = {"system": system} if system else {}
        req = dict(model=self.model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
                   extra_body={"temperature": temperature}, **kw)
        if max_tokens <= 8192:
            return self.client.messages.create(**req)
        with self.client.messages.stream(**req) as stream:   # large caps require streaming in the SDK
            return stream.get_final_message()

    def check(self, temperature=0.7):
        try:   # same call shape as the real run, so a parameter error stops everything here
            self._create("ping", temperature, max_tokens=1)
        except Exception as e:
            sys.exit(f"API check failed, nothing was run: {type(e).__name__}: {e}")

    def __call__(self, prompt, temperature, retries=3):
        for attempt in range(retries):
            try:
                r = self._create(prompt, temperature, max_tokens=MAX_TOKENS, system=SYSTEM)
                text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
                return text, {"prompt_tokens": r.usage.input_tokens,
                              "completion_tokens": r.usage.output_tokens,
                              "truncated": r.stop_reason == "max_tokens"}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError",
                                        "BadRequestError", "TypeError"):
                    sys.exit(f"Fatal API error, run stopped: {err}")
                time.sleep(2 * (attempt + 1))
        return f"ERROR: {err}", {"prompt_tokens": 0, "completion_tokens": 0, "error": err}

# ---------------------------------------------------------------------------
# Analysis of one final artifact (concatenated deliverable text)
# ---------------------------------------------------------------------------
def analyze(final_text: str) -> dict:
    py = extract_code_blocks(final_text, "python")
    yml = extract_code_blocks(final_text, "yaml") + extract_code_blocks(final_text, "yml")
    blocks, n_ok = [], 0
    for b in py:
        try:
            ast.parse(b); ok = True; n_ok += 1
        except SyntaxError:
            ok = False
        blocks.append({"parses": ok, "rvh": detect_rvh(b), "iih": detect_iih(b, PINNED_ENV)})
    parsed = [b for b, bl in zip(py, blocks) if bl["parses"]]
    joined = "\n".join(parsed)
    funcs = [n.name for b in parsed for n in ast.walk(ast.parse(b)) if isinstance(n, ast.FunctionDef)]
    n_tests = sum(1 for f in funcs if f.startswith("test"))
    has_pipeline = any(not f.startswith("test") for f in funcs)
    return {
        "n_python_blocks": len(py), "n_parse_ok": n_ok, "n_yaml_blocks": len(yml),
        "is_invalid": len(py) == 0 or n_ok == 0,
        "is_rvh": any(bl["rvh"]["is_rvh"] for bl in blocks if bl["parses"]),
        "is_iih": any(bl["iih"]["is_iih"] for bl in blocks if bl["parses"]),
        "n_tests": n_tests,
        "complete": bool(has_pipeline and n_tests >= 1 and len(yml) >= 1),
        "uses_mlflow": "mlflow" in joined,
        "has_error_handling": "try:" in joined or "except" in joined,
        "blocks": blocks,
    }

# ---------------------------------------------------------------------------
# Generation strategies
# ---------------------------------------------------------------------------
# Deliverable-matched control: the one-shot prompt asks for exactly the same three
# deliverables the structure-first steps produce, so the two strategies differ only in
# HOW the artifact is generated (one call vs staged), not in WHAT is requested.
ONE_SHOT_DELIVERABLES = ("\n\nProvide the full Python pipeline module, the complete pytest test file "
                         "and the GitHub Actions workflow, each in its own fenced code block.")


def one_shot(llm, task, temperature, d: Path):
    t0 = time.time()
    text, use = llm(task + ONE_SHOT_DELIVERABLES, temperature)
    (d / "response.txt").write_text(text, encoding="utf-8")
    return text, {"calls": 1, **_sum([use]), "seconds": round(time.time() - t0, 2)}


def structure_first(llm, task, temperature, d: Path):
    t0, uses = time.time(), []
    s1, u = llm(STRUCT_STEP1.format(task=task), temperature); uses.append(u)
    (d / "step1_skeleton.txt").write_text(s1, encoding="utf-8")
    s2, u = llm(STRUCT_STEP2.format(skeleton=s1), temperature); uses.append(u)
    (d / "step2_implementation.txt").write_text(s2, encoding="utf-8")
    impl = "\n\n".join(extract_code_blocks(s2, "python")) or s2
    s3, u = llm(STRUCT_STEP3.format(implementation=impl), temperature); uses.append(u)
    (d / "step3_tests_ci.txt").write_text(s3, encoding="utf-8")
    final = s2 + "\n\n" + s3            # the skeleton is scaffolding, not a deliverable
    (d / "final_artifact.txt").write_text(final, encoding="utf-8")
    return final, {"calls": 3, **_sum(uses), "seconds": round(time.time() - t0, 2)}


def _sum(uses):
    return {"prompt_tokens": sum(u.get("prompt_tokens", 0) for u in uses),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in uses),
            "errors": [u["error"] for u in uses if "error" in u],
            "truncated_calls": sum(1 for u in uses if u.get("truncated"))}


STRATEGIES = {"one_shot": one_shot, "structure_first": structure_first}
PROMPTS = {"specified": TASK_SPECIFIED, "under_specified": TASK_UNDER}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def summarize(rows):
    n = len(rows); valid = [r for r in rows if not r["is_invalid"]]; nv = len(valid)
    rate = lambda k, pool: round(sum(r[k] for r in pool) / len(pool), 3) if pool else None
    return {"n": n, "n_valid": nv, "invalid_rate": rate("is_invalid", rows),
            "rvh_rate": rate("is_rvh", valid), "iih_rate": rate("is_iih", valid),
            "complete_rate": rate("complete", rows),
            "mean_tests": round(sum(r["n_tests"] for r in rows) / n, 2) if n else None,
            "mlflow_rate": rate("uses_mlflow", valid), "error_handling_rate": rate("has_error_handling", valid),
            "mean_calls": round(sum(r["cost"]["calls"] for r in rows) / n, 2) if n else None,
            "mean_tokens": round(sum(r["cost"]["prompt_tokens"] + r["cost"]["completion_tokens"] for r in rows) / n, 1) if n else None,
            "n_api_errors": sum(len(r["cost"]["errors"]) for r in rows)}


def run(args):
    global MAX_TOKENS
    MAX_TOKENS = args.max_tokens
    if args.dry_run:
        llm = FakeLLM()
    elif args.provider == "anthropic":
        llm = AnthropicLLM(args.model)
    else:
        llm = OpenAILLM(args.model)
    if not args.dry_run:
        llm.check(args.temperature)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"'{out}' already contains results; choose another --out or move it, nothing was overwritten.")
    out.mkdir(parents=True, exist_ok=True)
    report = {"meta": {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "provider": "fake" if args.dry_run else args.provider,
                       "model": llm.model, "temperature": args.temperature, "n_per_cell": args.n,
                       "max_tokens": args.max_tokens,
                       "anthropic_api_version": API_VERSION if args.provider == "anthropic" else None,
                       "python": platform.python_version(), "dry_run": args.dry_run,
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "cells": []}
    # Rounds: each round produces one artifact in every cell (cell order rotated), so a run
    # stopped by the budget guard or an API error still leaves balanced cells.
    cells = [(f"{sn}__{pn}", sn, pn, st, task) for sn, st in STRATEGIES.items() for pn, task in PROMPTS.items()]
    rows = {c[0]: [] for c in cells}
    spent, worst_round, stop_reason = 0.0, 0.0, None
    price = PRICES_USD_PER_MTOK.get(llm.model)
    if args.budget_usd and not price and not args.dry_run:
        sys.exit(f"no price known for {llm.model}; add it to PRICES_USD_PER_MTOK or drop --budget-usd")
    for i in range(1, args.n + 1):
        if args.budget_usd and i > 1 and spent + 1.25 * worst_round > args.budget_usd:
            stop_reason = (f"budget guard: spent {spent:.2f} USD, next round could cost up to "
                           f"{1.25 * worst_round:.2f} USD, budget {args.budget_usd:.2f} USD")
            break
        round_cost = 0.0
        for k in range(len(cells)):
            cell, sn, pn, strat, task = cells[(k + i - 1) % len(cells)]
            d = out / cell / f"run_{i:02d}"; d.mkdir(parents=True, exist_ok=True)
            print(f"[round {i}/{args.n}] {cell}   (spent so far ~{spent + round_cost:.2f} USD)", flush=True)
            final, cost = strat(llm, task, args.temperature, d)
            res = analyze(final); res.update({"run": i, "cost": cost})
            (d / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            rows[cell].append(res)
            if price:
                round_cost += (cost["prompt_tokens"] * price[0] + cost["completion_tokens"] * price[1]) / 1e6
            if cost.get("truncated_calls") and not args.dry_run:
                sys.exit(f"Run stopped: {cell} run {i} hit max_tokens={MAX_TOKENS} "
                         f"({cost['truncated_calls']} truncated call(s)). Raise --max-tokens and restart "
                         f"in a new --out folder; truncated artifacts are not analysable.")
            if not args.dry_run:
                time.sleep(0.5)
        spent += round_cost; worst_round = max(worst_round, round_cost)
        _write_report(out, report, cells, rows, i, spent, None)
    _write_report(out, report, cells, rows, len(next(iter(rows.values()))), spent, stop_reason)
    if stop_reason:
        print(stop_reason)
    print_table(report["cells"]); print(f"saved -> {out / 'results.json'}")


def _write_report(out, report, cells, rows, n_done, spent, stop_reason):
    """Written after every complete round: results.json only ever describes complete rounds."""
    report["cells"] = []
    for cell, sn, pn, _, _ in cells:
        s = summarize(rows[cell][:n_done]); s.update({"cell": cell, "strategy": sn, "prompt": pn})
        report["cells"].append(s)
    report["meta"].update({"n_completed_rounds": n_done, "estimated_cost_usd": round(spent, 2),
                           "stop_reason": stop_reason})
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def reanalyze(args):
    out = Path(args.out); cells = []
    for cell_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        rows = []
        for d in sorted(cell_dir.glob("run_*")):
            f = d / "final_artifact.txt" if (d / "final_artifact.txt").exists() else d / "response.txt"
            old = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
            res = analyze(f.read_text(encoding="utf-8")); res.update({"run": old["run"], "cost": old["cost"]})
            rows.append(res)
        s = summarize(rows); s["cell"] = cell_dir.name; cells.append(s)
    (out / "results_reanalyzed.json").write_text(json.dumps({"cells": cells}, indent=2), encoding="utf-8")
    print_table(cells)


def print_table(cells):
    cols = ["cell", "n", "n_valid", "invalid_rate", "rvh_rate", "iih_rate", "complete_rate",
            "mean_tests", "mean_calls", "mean_tokens", "n_api_errors"]
    print("\n" + " | ".join(cols))
    for c in cells:
        print(" | ".join(str(c.get(k)) for k in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--max-tokens", type=int, default=2500,
                    help="per-call output cap (Claude needs ~16000; GPT-4o never exceeded ~1600)")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop before a round that could exceed this estimated spend")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reanalyze", action="store_true")
    a = ap.parse_args()
    reanalyze(a) if a.reanalyze else run(a)
