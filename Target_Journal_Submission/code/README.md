# PAFT — prevalence-adaptive F-measure thresholding (code and reproduction)

This folder reproduces every number, table and figure of the manuscript
*"Prevalence-adaptive F-measure thresholding for imbalanced tabular classification under label shift: a guarded decision layer and its limits"*.

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate      # Python >= 3.11 (developed with 3.13)
pip install -r requirements.txt                          # exact versions used: requirements-lock.txt
```
No GPU is needed. The device (CPU / CUDA / Apple MPS) is detected automatically (`src/models/zoo.py::detect_device`);
the reported run used 4 CPU cores and 15 GB RAM.
Building the manuscript additionally needs `pandoc`; the PDF needs Node.js with Playwright/Chromium.

## 2. Reproduce
```bash
python run_all.py --mode quick     # smoke test of the whole pipeline on 3 datasets, 1 seed  (~1 min after download)
python run_all.py --mode full      # the complete study                                         (~1 h on 4 cores)
python scripts/build_manuscript.py --root ..   # regenerate manuscript + supplementary (DOCX, PDF) from the results
python scripts/build_cover_letter.py ..
python scripts/check_consistency.py            # final consistency audit (manuscript <-> results)
python -m pytest -q tests                      # 10 unit tests
```
`quick` writes to `code/quick_run/` and never overwrites the reported results; `full` writes to the parent folder
(`../results`, `../tables`, `../figures`). Every stage is resumable: trained scores are cached in `<root>/scores/`.

## 3. Data (all public, downloaded by `scripts/download_data.py`)
| Source | Used for | Access |
|---|---|---|
| OpenML, 18 datasets (ids in `src/data/openml_suite.py`) | controlled-shift benchmark | `sklearn.datasets.fetch_openml` |
| UCI Bank Marketing (`bank-additional-full.csv`) | bank-time | archive.ics.uci.edu |
| OpenML 1597 (credit-card fraud), OpenML 151 (electricity) | creditcard-time, electricity-time | OpenML |
| Elliptic Bitcoin data set | elliptic-time | data.pyg.org mirror |
| US Census ACS 2018 1-Year PUMS, person files, 49 states + PR + CA | acs-income-states | www2.census.gov |
| KDD Cup 1999 (10 % subset) | kdd99-time | `sklearn.datasets.fetch_kddcup99` |

About 1 GB in total. Check each provider's licence/terms before redistribution; raw data are not stored in this repository.

## 4. Layout
```
config/            default.yaml (full protocol), quick.yaml (smoke test)
src/data/          dataset loaders, duplicate-aware splitting, real-shift suites
src/preprocessing/ leakage-free encoders (fitted on training data only)
src/models/        LightGBM, XGBoost, logistic regression, PyTorch MLP; device detection
src/training/      training + score cache (checkpoints = cached validation/test scores)
src/methods/       score_model.py (ROC surface, histograms), quantify.py (prevalence estimators),
                   rules.py (all decision rules: baselines, PAFT-Plug/Bayes/Guard, PAFT-L)
src/evaluation/    batch evaluation, artificial-prevalence protocol
src/statistics/    Wilcoxon + Holm, bootstrap CI, rank-biserial, Friedman/Nemenyi
experiments/       train_controlled, run_controlled, run_real, run_efficiency, run_labelled, run_guard_calibration
scripts/           download_data, make_tables, make_figures, build_manuscript, build_cover_letter, check_consistency
tests/             unit tests
refs.py            verified bibliography
```

## 5. Experiments ↔ scripts ↔ outputs
| Experiment | Script / condition | Output |
|---|---|---|
| 1–2 main comparison, baselines | `run_controlled.py` (`main`) | `results/controlled/main.csv`, Tables 4–5, Figs 3–4 |
| 3 ablation, 7 sensitivity | `run_controlled.py` (`ablation`) | Table 7, Fig. 5 |
| 4 robustness | `noise_*`, `valfrac_*`, `calib_platt` | Table 8, Fig. 6 |
| 5 generalisation | `metric_*`, other learners in `main`, `run_real.py` | Tables 6, 10, Fig. 7 |
| 6 efficiency | `run_efficiency.py` | Table 12 |
| 8 error analysis | `main__batches.csv.gz` → `make_tables.py` | Table 13, Fig. 8 |
| 9 label-assisted (post-hoc) | `run_labelled.py` | Table 11 |
| 10 guard calibration (review round 1) | `run_guard_calibration.py` | Supplementary Table S16 |

## 6. Determinism
Seeds 0–4 are fixed in the configuration. Each deployment batch is drawn from a generator seeded by a CRC-32 hash of
(dataset, model, seed, condition, metric, odds multiplier, batch size). A second full run of the main experiment reproduced
all 90,720 aggregated rows exactly. Results can differ in late decimals with other library versions.

## 7. Using the method on your own scores
```python
from src.methods.rules import DecisionLayer
dl = DecisionLayer(val_scores, val_labels, metric="f1")      # once, on labelled validation data
decisions, info = dl.decide(batch_scores)                     # on each unlabelled batch
kind, threshold = decisions["paft_guard"]                    # ('t', threshold): predict positive iff score >= threshold
print(info["pi_mean"], info["guard_p"])                     # estimated prevalence, label-shift goodness-of-fit p-value
```
Read the manuscript's Sections 6.6 and 8 first: on six real shifts unguarded adaptation did not help.
