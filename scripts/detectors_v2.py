# -*- coding: utf-8 -*-
"""Revised static detectors (v2), written after the pre-submission review of 13 Nov. 2026.

detectors.py (v1, copied verbatim from the MMC study) is kept unchanged so that the published
analysis can be reproduced; v2 fixes the false-positive and false-negative sources identified
in the review and in our own inspection.

RVH changes (each rule carries a code so that its effect can be reported separately):
  R1  splitters: KFold / StratifiedKFold / GroupKFold are deterministic unless shuffle=True, and
      train_test_split is deterministic with shuffle=False -> no violation in those cases.
  R2  XGBClassifier / LGBMClassifier use fixed default seeds -> no longer flagged.
  R3  an explicit random_state=None (or seed=None) is a violation (v1 accepted the keyword).
  R4  an estimator built from an unpacked dictionary (Est(**params)) is resolved inside the block:
      seeded if a dictionary literal, dict() call or subscript assignment in the block sets the
      key 'random_state' (or 'seed'); a dictionary literal assigned in the block to the unpacked name
      decides directly (seeded or violation); otherwise reported as UNDETERMINED, never as a violation.
  R5  coherence with the global seed: when np.random.seed(...) is called in the block, scikit-learn
      estimators and pandas .sample() without random_state draw from NumPy's seeded global
      generator and are reproducible within the process -> no violation (v1 rule iii already
      accepted the global seed for direct RNG calls, rule i did not).
  R6  seeded generator constructors (np.random.default_rng(42), np.random.RandomState(42),
      random.Random(42)) are not violations (v1 flagged them when no global seed was set);
      np.random.default_rng() / RandomState() WITHOUT a seed are violations even under a global
      seed, because they do not read the global state.
  R7  the global seed is matched to its generator: np.random.seed covers np.random.*,
      random.seed covers the standard random module (v1 let either cover both).

IIH extension (secondary measure, flags reported by kind):
  X1  unresolved first-party import: a module imported by the artifact that is neither the standard
      library, a known third-party package, nor any file the artifact itself provides.
  X2  third-party package outside the pinned environment (for instance evidently, seaborn).
The v1 IIH rules (env-sensitive frameworks, TF1 APIs, removed APIs) are kept.
"""
from __future__ import annotations
import ast, re, sys
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------------------------
@dataclass
class Finding:
    kind: str      # "RVH", "UNDETERMINED", "IIH"
    rule: str
    detail: str
    lineno: int


def _chain(node) -> tuple:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return ()


def _kw(node, name):
    for k in node.keywords:
        if k.arg == name:
            return k
    return None


def _is_none(expr):
    return isinstance(expr, ast.Constant) and expr.value is None


def _is_false(expr):
    return isinstance(expr, ast.Constant) and expr.value is False


def _is_true(expr):
    return isinstance(expr, ast.Constant) and expr.value is True


# Estimators and splitters that need a seed. Value = keywords that seed them.
ESTIMATORS = {
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor", "DecisionTreeClassifier",
    "DecisionTreeRegressor", "KMeans", "MLPClassifier", "MLPRegressor", "IsolationForest",
    "SGDClassifier", "HistGradientBoostingClassifier", "BaggingClassifier", "AdaBoostClassifier",
    "SMOTE", "RandomUnderSampler", "RandomOverSampler",
}
ALWAYS_SHUFFLING_SPLITTERS = {"ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit"}
OPTIONALLY_SHUFFLING_SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold"}
REPEATED_SPLITTERS = {"RepeatedKFold", "RepeatedStratifiedKFold"}   # always random
DEFAULT_SEEDED = {"XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor"}   # R2
SEED_KEYS = {"random_state", "seed"}

NP_PREFIXES = (("np", "random"), ("numpy", "random"))
SEEDED_CONSTRUCTORS = {"default_rng", "RandomState", "Generator", "SeedSequence"}


def _dict_keys_setting_seed(tree) -> bool:
    """R4: does any dictionary-like construct in the block set a seed key?"""
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and k.value in SEED_KEYS:
                    return True
        elif isinstance(n, ast.Call) and _chain(n.func)[-1:] == ("dict",):
            if any(k.arg in SEED_KEYS for k in n.keywords):
                return True
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
            s = n.slice
            if isinstance(s, ast.Constant) and s.value in SEED_KEYS:
                return True
        elif isinstance(n, ast.Call) and _chain(n.func)[-1:] in (("setdefault",), ("update",)):
            if n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value in SEED_KEYS:
                return True
            if any(k.arg in SEED_KEYS for k in n.keywords):
                return True
    return False


def detect_rvh_v2(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"parses": False, "parse_error": str(e), "is_rvh": False, "findings": []}

    np_seed = py_seed = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            c = _chain(n.func)
            if c in (("np", "random", "seed"), ("numpy", "random", "seed")):
                np_seed = True
            if c == ("random", "seed"):
                py_seed = True
    dict_seeded = _dict_keys_setting_seed(tree)
    literal_dicts = {}      # name -> True/False (literal dict assigned to that name sets a seed key?)
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict) and None not in n.value.keys:   # merges ({**a, **b}) are not literals
            has = any(isinstance(k, ast.Constant) and k.value in SEED_KEYS for k in n.value.keys)
            for t in n.targets:
                if isinstance(t, ast.Name):
                    literal_dicts[t.id] = literal_dicts.get(t.id, False) or has

    out = []
    add = lambda kind, rule, detail, node: out.append(Finding(kind, rule, detail, getattr(node, "lineno", 0)))
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        c = _chain(n.func)
        if not c:
            continue
        f = c[-1]
        unpack = any(k.arg is None for k in n.keywords)
        seed_kw = next((k for k in n.keywords if k.arg in SEED_KEYS), None)

        # --- estimators and splitters -------------------------------------------------------
        if f in DEFAULT_SEEDED:
            if seed_kw is not None and _is_none(seed_kw.value):
                add("RVH", "R3_explicit_none", f"{f}(random_state=None)", n)
            continue
        is_est = f in ESTIMATORS or f in ALWAYS_SHUFFLING_SPLITTERS or f in REPEATED_SPLITTERS
        if f in OPTIONALLY_SHUFFLING_SPLITTERS:
            sh = _kw(n, "shuffle")
            if sh is None or _is_false(sh.value):
                continue                                  # R1: deterministic folds
            is_est = True
        if f == "train_test_split":
            sh = _kw(n, "shuffle")
            if sh is not None and _is_false(sh.value):
                continue                                  # R1
            is_est = True
        if is_est:
            if seed_kw is not None:
                if _is_none(seed_kw.value):
                    add("RVH", "R3_explicit_none", f"{f}(random_state=None)", n)
                continue
            if unpack:
                names = [k.value.id for k in n.keywords if k.arg is None and isinstance(k.value, ast.Name)]
                if names and all(nm in literal_dicts for nm in names):
                    if not any(literal_dicts[nm] for nm in names):
                        add("RVH", "R4_literal_dict_without_seed", f"{f}(**{names[0]}) and {names[0]} has no seed key", n)
                    continue
                if dict_seeded:
                    continue                              # R4 resolved as seeded
                add("UNDETERMINED", "R4_unresolved_kwargs", f"{f}(**...) with no seed key found in block", n)
                continue
            if np_seed and f in ESTIMATORS | {"train_test_split"} | ALWAYS_SHUFFLING_SPLITTERS \
                    | REPEATED_SPLITTERS | OPTIONALLY_SHUFFLING_SPLITTERS:
                continue                                  # R5 global NumPy seed
            add("RVH", "unseeded_estimator", f"{f}(...) without random_state", n)
            continue

        # --- pandas .sample ------------------------------------------------------------------
        if f == "sample" and c[:1] != ("random",):
            if seed_kw is None and not np_seed:
                add("RVH", "unseeded_sample", ".sample(...) without random_state", n)
            elif seed_kw is not None and _is_none(seed_kw.value):
                add("RVH", "R3_explicit_none", ".sample(random_state=None)", n)
            continue

        # --- NumPy random ----------------------------------------------------------------------
        if any(c[:len(p)] == p for p in NP_PREFIXES) and len(c) > 2:
            if f == "seed":
                continue
            if f in SEEDED_CONSTRUCTORS:                  # R6
                if not n.args and seed_kw is None and _kw(n, "seed") is None:
                    if f != "SeedSequence":
                        add("RVH", "R6_unseeded_generator", f"np.random.{f}() without a seed", n)
                continue
            if not np_seed:
                add("RVH", "unseeded_rng", f"{'.'.join(c)}(...) with no np.random.seed", n)
            continue

        # --- standard library random ---------------------------------------------------------
        if c[:1] == ("random",) and len(c) == 2:
            if f in ("seed", "Random", "SystemRandom"):
                if f == "Random" and not n.args:
                    add("RVH", "R6_unseeded_generator", "random.Random() without a seed", n)
                continue
            if not py_seed:                               # R7
                add("RVH", "unseeded_rng", f"random.{f}(...) with no random.seed", n)
    return {"parses": True, "np_seed": np_seed, "py_seed": py_seed, "dict_seeded": dict_seeded,
            "is_rvh": any(x.kind == "RVH" for x in out),
            "findings": [asdict(x) for x in out]}


# ---------------------------------------------------------------------------------------------
# IIH, v1 rules + extension
# ---------------------------------------------------------------------------------------------
ENV_SENSITIVE = {"tensorflow", "tensorflow_addons", "keras", "torch", "torchvision", "jax", "flax",
                 "transformers", "lightgbm", "catboost", "xgboost", "cupy", "cudf", "cuml", "pyspark",
                 "dask", "h2o", "paddle", "mxnet", "theano", "caffe"}
REMOVED_PATTERNS = {r"\bsklearn\.externals\.joblib\b": "sklearn.externals.joblib (removed in 0.23)",
                    r"\.ix\[": "DataFrame.ix (removed in pandas 1.0)",
                    r"\bnp\.int\b": "np.int (removed in NumPy 1.24)",
                    r"\bnp\.float\b": "np.float (removed in NumPy 1.24)"}
TF1 = {("tf", "Session"), ("tf", "placeholder"), ("tf", "global_variables_initializer")}
# Known third-party distributions (import names), used to tell first-party modules apart.
KNOWN_THIRD_PARTY = {
    "pandas", "numpy", "sklearn", "scipy", "mlflow", "pytest", "joblib", "xgboost", "matplotlib", "yaml",
    "seaborn", "evidently", "great_expectations", "imblearn", "lightgbm", "catboost", "shap", "optuna",
    "requests", "boto3", "botocore", "fastapi", "flask", "pydantic", "dotenv", "tqdm", "click", "typer",
    "alibi_detect", "deepchecks", "pandera", "hypothesis", "prometheus_client", "dvc", "hydra", "omegaconf",
    "tensorflow", "keras", "torch", "statsmodels", "plotly", "uvicorn", "sqlalchemy", "psycopg2", "kafka",
    "airflow", "prefect", "kubernetes", "docker", "azure", "google", "jsonschema", "schedule", "slack_sdk",
    "category_encoders", "featuretools", "nannyml", "river", "alibi", "onnx", "onnxruntime", "skl2onnx",
    "cloudpickle", "pyarrow", "s3fs", "fsspec", "redis", "pymongo", "httpx", "rich", "loguru", "ruamel",
    "toml", "tomli", "pytest_mock", "mock", "freezegun", "responses", "moto", "faker", "coverage",
}
PINNED_THIRD_PARTY = {"pandas", "numpy", "sklearn", "scipy", "mlflow", "pytest", "joblib", "xgboost",
                      "matplotlib", "yaml"}


def _imports(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name.split(".")[0], n.lineno, a.name
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            yield n.module.split(".")[0], n.lineno, n.module


def detect_iih_v2(code: str, provided_modules: set) -> dict:
    """provided_modules: top-level module names the artifact itself defines (file names given in
    the response, e.g. 'fraud_pipeline' for fraud_pipeline.py, plus package directories)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"parses": False, "findings": []}
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    out = []
    for top, line, full in _imports(tree):
        if top in ENV_SENSITIVE and top not in PINNED_THIRD_PARTY:
            out.append(Finding("IIH", "v1_framework_not_in_env", full, line))
        elif top in stdlib or top in PINNED_THIRD_PARTY or top == "__future__":
            continue
        elif top in KNOWN_THIRD_PARTY:
            out.append(Finding("IIH", "X2_unpinned_package", full, line))
        elif top not in provided_modules:
            out.append(Finding("IIH", "X1_unresolved_first_party", full, line))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _chain(n.func)[-2:] in TF1:
            out.append(Finding("IIH", "v1_incompatible_api", ".".join(_chain(n.func)), n.lineno))
    for pat, why in REMOVED_PATTERNS.items():
        for m in re.finditer(pat, code):
            out.append(Finding("IIH", "v1_removed_api", why, code[:m.start()].count("\n") + 1))
    return {"parses": True, "findings": [asdict(x) for x in out],
            "is_iih_v1": any(x.rule.startswith("v1_") for x in out),
            "is_iih_ext": len(out) > 0}
