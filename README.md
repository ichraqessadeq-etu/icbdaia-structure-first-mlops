# Structure-first generation of MLOps pipelines

[![DOI](https://zenodo.org/badge/1451885468.svg)](https://doi.org/10.5281/zenodo.24093463)

Code and data for **"Structure-First Generation of MLOps Pipelines: Does Engineering-Style Staging Reduce
Operational Hallucinations in LLM-Generated CI/CD Artifacts?"** (I. Essadeq, S. Nouh, K. Kandali),
submitted to ICBDAIA 2027 (Springer LNNS).

The study compares two ways of asking an LLM for the same three deliverables (a Python training pipeline with
MLflow, a pytest file and a GitHub Actions workflow) for a credit-card fraud detection model:

- **one-shot**: one call requests the three files;
- **structure-first**: three calls, skeleton (file layout, signatures, test names, CI jobs) → implementation →
  tests and workflow.

2 × 2 design: strategy × prompt (specified vs under-specified), temperature 0.7. Defects are measured with the
deterministic static detectors of our earlier study (repository `mmc-operational-hallucination-detection`),
copied unchanged in `scripts/detectors.py`:

- **RVH (reproducibility violation)**: stochastic estimator or splitter without `random_state`, `.sample()` without
  `random_state`, NumPy / `random` generator used with no global seed.
- **IIH (infrastructure inconsistency)**: unpinned environment-sensitive framework, TF1 API, removed APIs.

All raw model responses are included, so **every number of the paper can be recomputed without any API call**.

## Layout

```
scripts/structure_first_study.py   generation (OpenAI or Anthropic), assembly of the artifact, detectors, cost
scripts/detectors.py               RVH / IIH detectors of the MMC study, unchanged
scripts/analyze_results.py         statistics of the paper (Fisher, Mann-Whitney, cluster bootstrap)
scripts/make_figure.py             Figure 1
results/runs_gpt4o_batch1/         GPT-4o, first batch: 20 artifacts per cell
results/runs_gpt4o_batch2/         GPT-4o, second, independent batch: 20 artifacts per cell
results/runs_claude/               Claude Sonnet 4.5: 10 artifacts per cell (10 complete rounds, budget-capped)
results/statistics.json            statistics, detector as published (primary analysis)
results/statistics_strict.json     statistics, sensitivity analysis (see below)
results/kwargs_cases.txt           the estimator calls removed by the sensitivity analysis, for manual checking
results/kwargs_review.md           review sheet: each removed call shown inside its function (scripts/make_kwargs_review.py)
results/kwargs_manual_check.csv    verdict of the manual check of the 27 Claude cases (25 false positives, 2 undetermined)
figures/fig1.pdf, fig1.png         Figure 1
```

Each run folder holds `results.json` (settings, script hash, per-artifact summary) and one sub-folder per cell
(`one_shot__specified`, `structure_first__under_specified`, ...) with one `run_XX/` per artifact:
`response.txt` (one-shot) or `step1_skeleton.txt`, `step2_implementation.txt`, `step3_tests_ci.txt` and
`final_artifact.txt` (structure-first), plus `analysis.json` (detector output, tests, tokens, errors).

## Reproducing the paper's numbers (no API key needed)

```bash
pip install -r requirements.txt
cd results
python ../scripts/analyze_results.py               # -> statistics.json   (Table 1, Section 4)
STRICT=1 python ../scripts/analyze_results.py      # -> statistics_strict.json   (sensitivity analysis)
python ../scripts/make_figure.py                   # -> ../figures/fig1.pdf
```

On Windows (cmd), use `set STRICT=1` before the second command, and `set STRICT=` afterwards.
The bootstrap uses a fixed seed (42) per comparison, so the confidence intervals are identical on every run.

## Sensitivity analysis

The RVH detector reads the keyword arguments written at the call site. An estimator built from an unpacked
dictionary, e.g. `RandomForestClassifier(**hyperparameters)`, is flagged even when `random_state` is a key of
that dictionary. Claude Sonnet 4.5 writes most of its estimators this way: all 27 of its `unseeded_estimator`
flags are of this form, against 6 of 34 for GPT-4o. `STRICT=1` drops these flags (the detector itself is not
modified); `results/kwargs_cases.txt` lists every dropped call with its file and line.

## Generating new artifacts (paid API calls)

```bash
cp .env.example .env          # then fill in your own key(s); .env is git-ignored
cd results
python ../scripts/structure_first_study.py --dry-run                      # offline check, fake LLM, no key
python ../scripts/structure_first_study.py --n 20 --out runs_gpt4o_new    # OpenAI, gpt-4o
python ../scripts/structure_first_study.py --provider anthropic --model claude-sonnet-4-5 \
       --max-tokens 64000 --budget-usd 16 --out runs_claude_new           # Anthropic, budget-capped rounds
```

The Claude run uses API version `2023-06-01` (pinned in the script) and streams responses above 8,192 output
tokens; the script stops at the first truncated response, and with `--budget-usd` it stops before a round that
could exceed the budget. Claude Sonnet 4.5 is retired on 30 November 2026: after that date its artifacts can only
be re-analysed, not regenerated, which is why every raw response is kept here.

## Script versions

`results.json` records the SHA-256 of the generation script used for each folder. The two GPT-4o batches were
generated with earlier revisions of `structure_first_study.py`; the revisions that followed added the Anthropic
client, round-based scheduling, the budget guard and truncation checks. The Claude run used the released version.
All folders are scored by the same `detectors.py`.

## Citation and license

Archived on Zenodo: [10.5281/zenodo.24093463](https://doi.org/10.5281/zenodo.24093463) (this DOI always resolves to the latest version). See `CITATION.cff`. Code: MIT (see `LICENSE`). The generated artifacts in `results/` are model outputs released
for research reproducibility.
