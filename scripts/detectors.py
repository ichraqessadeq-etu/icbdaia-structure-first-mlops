# -*- coding: utf-8 -*-
"""Operational-hallucination detectors (RVH, IIH) and summary metrics.

Copied verbatim from the MMC study (mmc_hallucination_study.py, PART 1) so that this
study uses exactly the same oracle. Standard library only, deterministic, no network.
"""
from __future__ import annotations
import re, ast
from dataclasses import dataclass, asdict
from typing import Optional

# =============================================================================
#  PART 1 - DETECTORS  (pure standard-library, no network, deterministic)
# =============================================================================

# ---- code extraction --------------------------------------------------------
_FENCE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<body>.*?)```", re.DOTALL)

def extract_code_blocks(text: str, lang: Optional[str] = None) -> list:
    """Return fenced code blocks from an LLM response. If ``lang`` is given
    (e.g. 'python','yaml'), only matching blocks are returned; 'py'=='python'."""
    blocks = []
    for m in _FENCE.finditer(text):
        label = (m.group("lang") or "").strip().lower()
        if label in ("py", "python3"):
            label = "python"
        if lang is None or label == lang.lower():
            blocks.append(m.group("body"))
    return blocks


@dataclass
class Violation:
    kind: str          # 'RVH' | 'IIH'
    rule: str
    detail: str
    lineno: int = 0


def _attr_chain(node) -> tuple:
    """Dotted name of a Call's func as a tuple: np.random.shuffle ->
    ('np','random','shuffle'). Returns () if unresolved."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ()
    return tuple(reversed(parts))


# ---- RVH: Reproducibility Violation Hallucination ---------------------------
_NEEDS_SEED_CALL = {
    "train_test_split": ("random_state",),
    "RandomForestClassifier": ("random_state",),
    "RandomForestRegressor": ("random_state",),
    "ExtraTreesClassifier": ("random_state",),
    "GradientBoostingClassifier": ("random_state",),
    "DecisionTreeClassifier": ("random_state",),
    "XGBClassifier": ("random_state", "seed"),
    "LGBMClassifier": ("random_state", "seed"),
    "KMeans": ("random_state",),
    "MLPClassifier": ("random_state",),
    "KFold": ("random_state",),
    "StratifiedKFold": ("random_state",),
    "ShuffleSplit": ("random_state",),
}
_GLOBAL_SEED_SETTERS = {
    ("np", "random", "seed"), ("numpy", "random", "seed"),
    ("random", "seed"), ("torch", "manual_seed"), ("tf", "random", "set_seed"),
}
_RNG_PREFIXES = (("np", "random"), ("numpy", "random"), ("random",))
_SEEDED_METHODS = {"sample"}


def detect_rvh(code: str) -> dict:
    """{is_rvh, parse_error, has_global_seed, violations:[...]}. A parse error
    means the artifact is not analyzable as Python (reported, not counted RVH)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"is_rvh": False, "parse_error": str(e),
                "has_global_seed": False, "violations": []}

    has_global_seed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            if chain in _GLOBAL_SEED_SETTERS or chain[-2:] == ("random", "seed"):
                has_global_seed = True
                break

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attr_chain(node.func)
        if not chain:
            continue
        fname = chain[-1]
        kwargs = {kw.arg for kw in node.keywords if kw.arg}

        if fname in _NEEDS_SEED_CALL:
            accepted = _NEEDS_SEED_CALL[fname]
            if not (set(accepted) & kwargs):
                violations.append(Violation(
                    "RVH", "unseeded_estimator",
                    f"{fname}(...) called without {' or '.join(accepted)}",
                    getattr(node, "lineno", 0)))

        if fname in _SEEDED_METHODS and "random_state" not in kwargs:
            violations.append(Violation(
                "RVH", "unseeded_sample",
                f".{fname}(...) called without random_state",
                getattr(node, "lineno", 0)))

        if not has_global_seed:
            for pref in _RNG_PREFIXES:
                if chain[:len(pref)] == pref and chain[-1] != "seed" \
                        and len(chain) > len(pref):
                    violations.append(Violation(
                        "RVH", "unseeded_rng",
                        f"{'.'.join(chain)}(...) with no global seed set",
                        getattr(node, "lineno", 0)))
                    break

    return {"is_rvh": len(violations) > 0, "parse_error": None,
            "has_global_seed": has_global_seed,
            "violations": [asdict(v) for v in violations]}


# ---- IIH: Infrastructure Inconsistency Hallucination ------------------------
# Heavy third-party frameworks whose presence signals an environment mismatch
# if they are NOT in the pinned env. Local/first-party imports (the pipeline
# module the LLM itself generates, 'src', 'utils', ...) are intentionally NOT
# flagged -- a test file importing the generated pipeline is legitimate.
_ENV_SENSITIVE_PKGS = {
    "tensorflow", "tensorflow_addons", "keras", "torch", "torchvision",
    "jax", "flax", "transformers", "lightgbm", "catboost", "xgboost",
    "cupy", "cudf", "cuml", "pyspark", "dask", "h2o", "paddle", "mxnet",
    "theano", "caffe",
}
_BANNED_APIS = {
    ("tf", "Session"): "TF1 API (tf.Session) under a TF2 environment",
    ("tf", "placeholder"): "TF1 API (tf.placeholder) under a TF2 environment",
    ("tf", "global_variables_initializer"): "TF1 API under a TF2 environment",
}
_BANNED_PATTERNS = {
    r"\bsklearn\.externals\.joblib\b": "sklearn.externals.joblib removed in sklearn>=0.23",
    r"\.ix\[": "DataFrame.ix removed in pandas>=1.0",
    r"\bnp\.int\b": "np.int alias removed in numpy>=1.24",
    r"\bnp\.float\b": "np.float alias removed in numpy>=1.24",
}

def _top_level_module(name: str) -> str:
    return name.split(".")[0]


def detect_iih(code: str, allowed_imports: set) -> dict:
    """{is_iih, parse_error, violations:[...]} against a pinned environment.
    Flags: (1) env-sensitive frameworks imported but not pinned; (2) known
    version-incompatible API calls; (3) code that does not parse. Local/
    first-party module imports are NOT flagged."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # Unparseable code is NOT an infrastructure inconsistency; it is an
        # invalid/incomplete artifact. Report the parse error, do not flag IIH.
        return {"is_iih": False, "parse_error": str(e), "violations": []}

    violations = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                m = _top_level_module(alias.name)
                if m in _ENV_SENSITIVE_PKGS and m not in allowed_imports:
                    violations.append(Violation(
                        "IIH", "framework_not_in_env",
                        f"import '{alias.name}' (env-sensitive) not in pinned environment",
                        getattr(node, "lineno", 0)))
        elif isinstance(node, ast.ImportFrom) and node.module:
            m = _top_level_module(node.module)
            if m in _ENV_SENSITIVE_PKGS and m not in allowed_imports:
                violations.append(Violation(
                    "IIH", "framework_not_in_env",
                    f"from '{node.module}' import ... (env-sensitive) not in pinned environment",
                    getattr(node, "lineno", 0)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            for banned, why in _BANNED_APIS.items():
                if chain[-len(banned):] == banned:
                    violations.append(Violation(
                        "IIH", "incompatible_api", why, getattr(node, "lineno", 0)))

    for pat, why in _BANNED_PATTERNS.items():
        for m in re.finditer(pat, code):
            line = code[:m.start()].count("\n") + 1
            violations.append(Violation("IIH", "removed_api", why, line))

    return {"is_iih": len(violations) > 0, "parse_error": None,
            "violations": [asdict(v) for v in violations]}


def summarize(per_artifact: list) -> dict:
    """Three orthogonal metrics with explicit denominators:
      - invalid_rate : share of artifacts with no parseable Python (over n)
      - rvh_rate     : share flagged RVH among VALID artifacts (over n_valid)
      - iih_rate     : share flagged real IIH among VALID artifacts (over n_valid)
    RVH/IIH are undefined (None) when no artifact is valid."""
    n = len(per_artifact)
    if n == 0:
        return {"n": 0, "n_valid": 0, "invalid_rate": None,
                "rvh_rate": None, "iih_rate": None}
    invalid = sum(1 for r in per_artifact if r.get("is_invalid"))
    valid = [r for r in per_artifact if not r.get("is_invalid")]
    n_valid = len(valid)
    rvh = sum(1 for r in valid if r["is_rvh"])
    iih = sum(1 for r in valid if r["is_iih"])
    return {"n": n, "n_valid": n_valid,
            "invalid_rate": round(invalid / n, 3),
            "rvh_rate": round(rvh / n_valid, 3) if n_valid else None,
            "iih_rate": round(iih / n_valid, 3) if n_valid else None}
