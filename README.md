# Staged generation of MLOps pipelines with LLMs

[![DOI](https://zenodo.org/badge/1451885468.svg)](https://doi.org/10.5281/zenodo.24093463)

Code and data for **"Staged Generation of MLOps Pipelines with Large Language Models: More Tests, Broken
Interfaces, and No Gain in Reproducibility"** (I. Essadeq, S. Nouh, K. Kandali), submitted to ICBDAIA 2027
(Springer LNNS). Version 1.0.0 of this repository accompanied an earlier draft entitled *"Structure-First
Generation of MLOps Pipelines: Does Engineering-Style Staging Reduce Operational Hallucinations in LLM-Generated
CI/CD Artifacts?"*; version 2.0.0 accompanies the revised paper.

The study asks an LLM for the same three deliverables (a Python training pipeline with MLflow, a pytest file and a
GitHub Actions workflow) for a credit-card fraud detection model, under five generation conditions:

| Condition | Calls | Folder name |
|---|---|---|
| one-shot (OS) | 1: task → module, tests, workflow | `one_shot__*` |
| OS + seed instruction | 1: same, plus "seed every source of randomness …" | `one_shot_seed__*` |
| two calls, no skeleton | 2: task → module; module → tests, workflow | `two_call__*` |
| structure-first (SF) | 3: task → skeleton; skeleton → module; module → tests, workflow | `structure_first__*` |
| SF + skeleton to step 3 | 3: as SF, the skeleton is also given to step 3 | `structure_first_plan__*` |

Each condition is crossed with a specified and an under-specified prompt; temperature 0.7. GPT-4o: 40 artifacts per
cell for OS and SF (two batches of 20), 30 per cell for the three ablation conditions; Claude Sonnet 4.5: 10 per
cell for OS and SF. 380 artifacts in total.

All raw model responses are included, so **every number of the paper can be recomputed without any API call**.

## Layout

```
scripts/structure_first_study.py   generation (OpenAI or Anthropic), all five conditions, cost, model snapshot
scripts/detectors.py               RVH / IIH detectors of the earlier MMC study (v1), unchanged
scripts/detectors_v2.py            revised detectors (v2): RVH rules R1-R7, unresolved first-party imports,
                                   packages outside the pinned environment
scripts/reanalyze_v2.py            v1 + v2 detectors, tests, LOC, workflows on all 380 artifacts -> artifacts_v2.csv
scripts/stats_v2.py                statistics of the revised paper -> stats_v2.json
scripts/execute_artifacts.py       executes each artifact 3 times with pytest in a pinned environment
scripts/make_tables.py             rows of Table 1 -> figures/table_main.tex
scripts/make_figure_v2.py          execution outcomes -> figures/fig_outcomes.pdf / .png
scripts/make_annotation_sheet.py   blind annotation sheet of the RVH oracle (shuffled, detector hidden)
scripts/annotation_metrics.py      Cohen's kappa, precision of v2, removed v1 findings, missed violations
scripts/prepare_fraud.py           10,000-row sample of the credit-card data used for execution
scripts/analyze_results.py,        statistics and figure of the first draft (v1 detectors, OS vs SF only)
scripts/make_figure.py, make_kwargs_review.py

results/runs_gpt4o_batch1/         GPT-4o, OS and SF, first batch (20 per cell)
results/runs_gpt4o_batch2/         GPT-4o, OS and SF, second, independent batch (20 per cell)
results/runs_claude/               Claude Sonnet 4.5, OS and SF (10 per cell)
results/runs_ablation/             GPT-4o, the three ablation conditions (30 per cell, 15 Nov. 2026,
                                   snapshot gpt-4o-2024-08-06)
results/artifacts_v2.csv / .json   one row per artifact: detectors v1 and v2, tests, LOC, workflow checks
results/execution_results.json     execution of the 200 OS/SF artifacts (3 pytest runs each)
results/execution_results_ablation.json   execution of the 180 ablation artifacts
results/artifacts_exec.csv         execution outcome per artifact (merged view)
results/stats_v2.json              statistics of the revised paper
results/kwargs_manual_check.csv    manual tracing of the 27 unpacked-dictionary estimator calls of Claude
results/kwargs_manual_check_gpt4o.csv   same for the 6 calls of GPT-4o
results/annotation/annotation_verdicts.csv   two independent annotations of 172 oracle items (anonymised)
results/annotation_metrics.json    kappa, precision, recall check of the RVH oracle
results/fraud_sample_info.json     provenance and SHA-256 of the data sample used for execution
results/statistics*.json, kwargs_cases.txt, kwargs_review.md   first draft (v1)
figures/                           figures and table of the revised paper (and fig1 of the first draft)
```

Each run folder holds `results.json` (settings, script hash, per-artifact summary) and one sub-folder per cell with
one `run_XX/` per artifact: `response.txt` (one call) or the step files and `final_artifact.txt` (staged
conditions), plus `analysis.json` (v1 detector output, tests, tokens, errors). In the data files, artifacts are
named by their folder, e.g. `runs_gpt4o_batch1/structure_first__specified/run_03`.

## Reproducing the paper's numbers (no API key needed)

```bash
pip install -r requirements.txt
cd results
python ../scripts/reanalyze_v2.py        # -> artifacts_v2.csv / .json
python ../scripts/stats_v2.py            # -> stats_v2.json (Section 4)
python ../scripts/make_tables.py         # -> ../figures/table_main.tex (Table 1)
python ../scripts/make_figure_v2.py      # -> ../figures/fig_outcomes.pdf (Fig. 2)
python ../scripts/annotation_metrics.py  # -> annotation_metrics.json (Section 3.4, validation)
```

The table and the figure read the saved execution results; they do not re-execute the artifacts. The bootstrap
uses a fixed seed, so the confidence intervals are identical on every run.

## Executing the artifacts

`execute_artifacts.py` writes each artifact to files, places the data sample at `creditcard.csv`,
`data/creditcard.csv` and `data/raw/creditcard.csv`, and runs pytest three times. The paper used Python 3.7.7 with
scikit-learn 1.0.2, pandas 1.3.5, NumPy 1.21.6, SciPy 1.7.3, MLflow 1.30.1 and pytest 7.4.4.

```bash
python scripts/prepare_fraud.py creditcard.arff results/data/creditcard_sample.csv   # OpenML 1597 (ARFF)
cd results
PINNED_PYTHON=/path/to/python3.7 FRAUD_CSV=data/creditcard_sample.csv \
    python ../scripts/execute_artifacts.py --workers 2 --runs 3 --out my_execution.json
python ../scripts/execute_artifacts.py --roots runs_ablation --out my_execution_ablation.json   # same variables
```

The data are not redistributed here: download the ULB credit-card dataset as OpenML dataset 1597;
`prepare_fraud.py` checks its MD5 and `results/fraud_sample_info.json` gives the SHA-256 of the expected sample.

## Generating new artifacts (paid API calls)

```bash
cp .env.example .env          # then fill in your own key(s); .env is git-ignored
cd results
python ../scripts/structure_first_study.py --dry-run                      # offline check, fake LLM, no key
python ../scripts/structure_first_study.py --strategies two_call,structure_first_plan,one_shot_seed \
       --n 30 --out runs_ablation_new                                     # OpenAI, gpt-4o
```

The script stops at once if the API reports insufficient credit, and records the dated model snapshot returned by
the API. Claude Sonnet 4.5 is retired on 30 November 2026: after that date its artifacts can only be re-analysed,
not regenerated, which is why every raw response is kept here.

## Annotation of the oracle

Two annotators independently labelled 172 items (every v2 finding, every v1 finding removed by v2, and 15 artifacts
without findings), shuffled and without knowing which detector produced each item. `annotation_verdicts.csv` keeps
only the verdicts (TP / FP / ?) and the missed violations reported for the 15 negative artifacts; names and free-text
comments are not published.

## Citation and license

Archived on Zenodo: [10.5281/zenodo.24093463](https://doi.org/10.5281/zenodo.24093463) (concept DOI, always resolves
to the latest version). See `CITATION.cff`. Code: MIT (see `LICENSE`). The generated artifacts in `results/` are
model outputs released for research reproducibility.
