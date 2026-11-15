# -*- coding: utf-8 -*-
"""Fraud task data: ULB credit-card fraud (Dal Pozzolo et al., 2015), OpenML dataset 1597 (ARFF, md5
178bcf9bb1f31a3dfe12d0e577884add). Keeps every fraud and a random sample of legitimate transactions
(seed 42) so that one pytest run trains in seconds on a CPU laptop.

    py -3.13 -I prepare_fraud.py ../data_raw/creditcard.arff ../data/fraud.csv
"""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

N_TOTAL, SEED = 10_000, 42
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
assert hashlib.md5(src.read_bytes()).hexdigest() == "178bcf9bb1f31a3dfe12d0e577884add", "unexpected file"
lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
cols = [re.match(r"@attribute\s+'?([^'\s]+)'?", l, re.I).group(1) for l in lines if l.lower().startswith("@attribute")]
start = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data") + 1
rows = [l for l in lines[start:] if l.strip() and not l.startswith("%")]
d = pd.DataFrame([r.split(",") for r in rows], columns=cols)
d["Class"] = d["Class"].str.strip("'\" ").astype(int)
for c in cols[:-1]:
    d[c] = pd.to_numeric(d[c])
fraud, legit = d[d["Class"] == 1], d[d["Class"] == 0]
legit = legit.sample(n=N_TOTAL - len(fraud), random_state=SEED)
out = pd.concat([fraud, legit]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
dst.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dst, index=False)
info = {"source_rows": len(d), "source_frauds": int(d["Class"].sum()), "rows": len(out),
        "frauds": int(out["Class"].sum()), "prevalence": round(float(out["Class"].mean()), 4),
        "columns": list(out.columns), "seed": SEED,
        "csv_sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
(dst.parent / "fraud_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in info.items() if k != "columns"}, indent=2))
