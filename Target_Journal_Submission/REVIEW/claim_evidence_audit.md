# Claim–evidence audit (Phase 22)

Every number in the manuscript is injected by `code/scripts/build_manuscript.py` from the files below; the template
contains no result values. `code/scripts/check_consistency.py` re-derives the manuscript and re-checks the claims marked ✔ auto.

| # | Claim (manuscript location) | Evidence | Experiment | Table / figure / file | Check |
|---|---|---|---|---|---|
| 1 | Static threshold loses F1 under prevalence shift; loss grows with shift size (6.1) | Static-F 0.496 vs Oracle-π 0.537 (ρ ≠ 1); 0.674 vs 0.766 at ρ = 8 | Exp. 1 | Table 4, Fig. 3; `tables/table2_main_by_shift.csv` | values injected |
| 2 | PAFT-Bayes beats Static-F on all 18 datasets, +0.036, CI +0.027..+0.045, Holm p < 0.001 (abstract, 6.1) | paired Wilcoxon, n = 18 | Exp. 1–2 | Table 5; `table3_stats_shifted.csv` | ✔ auto |
| 3 | PAFT-Bayes is equivalent to DTA-EM in mean F1 (margin 0.01) and does not outperform it (abstract, 6.1) | −0.001, p = 0.099; 90 % CI −0.006..+0.002 | Exp. 2 + round-3 analysis | Table 5; `statistics.json: equivalence_paft_vs_dta` | ✔ auto |
| 4 | EM-Fixed is biased as Corollary 1 predicts (6.1) | EM-Fixed ≤ Static-F for ρ < 1, close to PAFT for ρ > 1 | Exp. 1 | Table 4 | values injected |
| 5 | Cost at ρ = 1 is 0.003 (6.1) | −0.0035, Holm p = 0.002 | Exp. 1 | `table3b_stats_noshift.csv` | values injected |
| 6 | Same pattern for XGBoost, logistic regression, MLP (6.2) | gains +0.035/+0.034/+0.034, Holm p < 0.001 | Exp. 5 | Table S1/S1b | ✔ auto |
| 7 | Works for F0.5, F2, MCC (6.2) | +0.026/+0.040/+0.014, Holm p < 0.01 | Exp. 5 | Table 6, S12 | ✔ auto |
| 8 | Posterior-expected rule beats plug-in, mostly in small batches (6.3) | +0.010, 18/18 | Exp. 3 | Table 7, S11 | ✔ auto |
| 9 | Prior width matters little for σ ≤ 2; σ = 1 marginally better than the pre-specified σ = 2 (6.3) | Table 7 | Exp. 3/7 | Table 7, Fig. 5 | values injected |
| 10 | With 10 % validation data DTA-EM is significantly better (6.4) | −0.004, Holm p = 0.021 | Exp. 4 | Table 8, S6 | values injected |
| 11 | Under feature noise unguarded rules lose their advantage; PAFT-Guard has the highest label-free mean, not significantly (6.4) | Table 8 | Exp. 4 | Table 8, S6, Fig. 6 | ✔ auto (ordering) |
| 12 | Guard over-fires under the resampling protocol for the rarest classes but is conservative on fresh samples (6.4) | 100 % on creditcard; simulation ≤ 1 % in 96 % of cells | Exp. 10 | Table S5, S16 | values injected |
| 13 | On real suites label-free adaptation gives no reliable gain; significant loss on ACS and electricity; loss on KDD99 (6.6) | Table 10; S7 | Exp. 5 (real) | Table 10, Fig. 7 | ✔ auto |
| 14 | On ACS the threshold map transfers but quantification fails (6.6) | Oracle-π +0.011 over static; MAE of π̂ 0.13 | Exp. 5 (real) | Tables 3, 10 | values injected |
| 15 | Guard fires on 95–100 % of real batches and restores static performance (6.6) | Table 3 | Exp. 5 (real) | Table 3, 10 | values injected |
| 16 | PAFT-L helps on ACS, harms on KDD99, loses to re-tuning under concept change (6.7) | Table 11, S13b | Exp. 9 (post-hoc) | Table 11 | ✔ auto (ACS) |
| 17 | Computational cost negligible (6.8) | 0.56 ms / 2,000 scores; 2.5 MB | Exp. 6 | Table 12, S15, Fig. S1 | values injected |
| 18 | Regret driven by prevalence error and few positives; creditcard is the worst case (6.9) | Table 13, S8, S9 | Exp. 8 | Table 13, Fig. 8 | values injected |
| 19 | Zero cross-split duplicates (5.2) | split audit | — | Table 2; `table1_datasets.csv` | values injected |
| 20 | Exact reproducibility (Supplement S1) | re-run of main experiment: 90,720 rows, max abs difference 0.0 | — | this session's log; `results/experiment_logs/` | manual |

## Claims removed or weakened during the audit
* "PAFT improves deployed classifiers" – never stated; real-shift evidence contradicts it.
* "Bank-time shows adaptation helps" – replaced by an explicit warning that the classifier is uninformative there.
* "PAFT-L is useful when the guard fires" – weakened to "helped on one suite, harmed on one; exploratory".
* "The guard keeps its nominal level" – replaced by the measured rates and the simulation.
* "Equivalent" – now tied to a stated margin and a confidence interval.
* Literature claims are limited to what the cited sources state; two 2026 preprints were read at abstract level only.
