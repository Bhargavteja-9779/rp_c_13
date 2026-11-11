# Review round 4 — Reviewer #4 (ML methodology, implementation, code–paper consistency)

The code was read side by side with Sections 4–5.

| # | Severity | Issue | Outcome |
|---|---|---|---|
| 4.1 | MAJOR | Per-batch records lacked the base-learner identifier, contaminating the error analysis. | Fixed in round 1 (see 1.4); regression-checked by exact reproduction of the aggregated results. |
| 4.2 | MODERATE | Paper: candidate thresholds "thinned to at most 1,024 quantiles per class". Code: `max_thr=1024` — consistent. Static-Bag, however, uses 512 and 8 bins, which the paper did not say. | Now stated in Supplementary S1. |
| 4.3 | MODERATE | Paper: prevalence grid "±9 logits". Code additionally clips logits to ±13.8. | Clipping is inactive for all datasets except at the extreme lower end for creditcard (logit π0 − 9 = −15.4); stated here; the effect is a slightly truncated grid below π = 10⁻⁶, which cannot affect the optimum. |
| 4.4 | MODERATE | EM-Fixed maps the static threshold through the isotonic calibrator; ties inside an isotonic block make the posterior-space rule predict more positives than the score-space rule. Is the baseline handicapped? | Checked with the strictly monotone Platt calibrator: EM-Fixed 0.504 vs 0.507 (isotonic). The weakness of EM-Fixed is the fixed posterior threshold (Corollary 1), not tie handling. |
| 4.5 | MODERATE | DTA-EM uses the ratio-of-expectations approximation of expected F. | Disclosed (2.6). A second DTA variant on PAFT's histogram posterior was run as ablation (iv). |
| 4.6 | MODERATE | Base learners are untuned and the perceptron is small; "strong deep baseline" would be an overstatement. | The paper claims model-agnosticism of the *decision layer*, not competitive tabular deep learning. Limitation 6. |
| 4.7 | MODERATE | Unit tests covered only the core. | Tests added for the label-assisted rule, re-tuning baseline, duplicate-aware split, training-only encoder statistics and EM prior recovery (10 tests, all passing). |
| 4.8 | MINOR | `lightgbm` deprecation warning for `eval_set`. | Harmless; warnings suppressed inside `fit_predict`. |
| 4.9 | MINOR | Hardware detection. | `detect_device()` returns cuda / mps / cpu; the run used CPU. Logged in `results/experiment_logs/environment.json` by `run_all.py`. |
| 4.10 | MINOR | Equation (8) compares with the mixture fitted at the MLE, while (7) uses the posterior. | Intended and stated ("π̂ maximises (5)"). |

No contradiction between an equation and its implementation remains to my knowledge: (2) ↔ `fbeta_from_rates`; (5) ↔ `binned_log_lik`;
(6) ↔ `posterior`; (7) ↔ `DecisionLayer.paft`; (8) ↔ `mixture_ks_pvalue`; Proposition 1 is checked numerically by
`test_lipton_condition_generalised` for β ∈ {0.5, 1, 2}.
