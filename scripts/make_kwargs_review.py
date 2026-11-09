# -*- coding: utf-8 -*-
"""Review sheet for the manual check of the estimator calls built from an unpacked dict (**params).

For each flagged call it prints the function that contains it, marks the call line with >>> and
the lines that mention random_state with ***. Output: kwargs_review.md (no API call).

    py -3.13 make_kwargs_review.py
"""
import ast, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import extract_code_blocks

out = ["# Manual check of the unpacked-dictionary estimator calls", "",
       "For each case: read the function, follow the dictionary passed with `**`, and answer the question:",
       "**is `random_state` a key of the dictionary that actually reaches this call, in the default path?**",
       "Write OK (seeded: false positive of the detector), NO (unseeded: true violation) or ? (cannot tell).",
       "`>>>` marks the flagged call, `***` marks every line that mentions random_state.", ""]
n = 0
for root, label, lim in [("runs_claude", "Claude Sonnet 4.5", 10),
                         ("runs", "GPT-4o batch 1", 99), ("runs_gpt4o_batch1", "GPT-4o batch 1", 99),
                         ("runs_gpt4o_batch2", "GPT-4o batch 2", 99)]:
    if not os.path.isdir(root):
        continue
    out += [f"## {label} (`{root}`)", ""]
    for d in sorted(glob.glob(f"{root}/*__*/run_*")):
        if int(d[-2:]) > lim:
            continue
        a = json.load(open(f"{d}/analysis.json", encoding="utf-8"))
        f = f"{d}/final_artifact.txt" if os.path.exists(f"{d}/final_artifact.txt") else f"{d}/response.txt"
        py = extract_code_blocks(open(f, encoding="utf-8").read(), "python")
        for i, bl in enumerate(a["blocks"]):
            if not bl["parses"]:
                continue
            src = py[i]; lines = src.splitlines(); tree = ast.parse(src)
            flagged = {v["lineno"] for v in bl["rvh"]["violations"] if v["rule"] == "unseeded_estimator"}
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and node.lineno in flagged
                        and any(k.arg is None for k in node.keywords)):
                    continue
                n += 1
                funcs = [fn for fn in ast.walk(tree)
                         if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and fn.lineno <= node.lineno <= fn.end_lineno]
                fn = min(funcs, key=lambda x: x.end_lineno - x.lineno) if funcs else None
                lo, hi = (fn.lineno, fn.end_lineno) if fn else (max(1, node.lineno - 30), node.lineno + 5)
                where = f"function `{fn.name}`" if fn else "module level"
                out += [f"### Case {n}: `{d.replace(os.sep, '/')}`, code block {i}, line {node.lineno} ({where})", "",
                        "Answer: ____", "", "```python"]
                for k in range(lo, hi + 1):
                    t = lines[k - 1]
                    mark = ">>>" if k == node.lineno else ("***" if "random_state" in t else "   ")
                    out.append(f"{mark} {k:4d}  {t}")
                out += ["```", ""]
                if fn and not any("random_state" in lines[k - 1] for k in range(lo, hi + 1)):
                    out += ["Note: no line of this function mentions random_state; the dictionary comes from the "
                            "caller or from a config: search the file for the dictionary's name.", ""]
open("kwargs_review.md", "w", encoding="utf-8").write("\n".join(out))
print(f"{n} cases -> kwargs_review.md")
