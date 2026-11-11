# Review round 1 — Reviewer #1 (general referee, hostile but fair)

Manuscript version reviewed: first complete draft (five real suites, no guard-calibration study).

**Recommendation at this stage: major revision.**

## Summary of the submission
A decision-layer method (PAFT) that re-derives an F-measure threshold from validation ROC information and an estimated
deployment prevalence, with a goodness-of-fit guard; evaluated on 18 tabular datasets under simulated label shift and on
five real shifts.

## Issues

| # | Severity | Issue | Required fix | Outcome |
|---|---|---|---|---|
| 1.1 | MAJOR | The guard rejects on 100 % of creditcard batches and 20 % of APSFailure batches *under pure label shift*, far above its nominal 1 % level. Either the test is mis-calibrated or the protocol is at fault; the paper cannot say which. | Establish the test's size and power on data where every batch is a fresh sample. | **Fixed experimentally.** New Experiment 10 (`experiments/run_guard_calibration.py`): binormal scores, fresh draws, 24 null cells. Size ≤ 1 % in 96 % of cells (max 6.8 % at 50 validation positives, m = 10,000); power ≥ 99.6 % for a 0.5 s.d. score displacement with ≥ 500 validation positives and m ≥ 2,000. The excess firing is therefore an artefact of resampling a tiny positive pool with replacement. Section 6.4 rewritten; Supplementary Table S16 added. |
| 1.2 | MAJOR | The headline method does not beat the strongest baseline (DTA-EM). The draft's title and abstract must not read as a "new method wins" paper. | State the tie in the abstract, the contribution list and the conclusion. | **Fixed.** Abstract, contribution 3, Section 6.1 heading "PAFT does not outperform DTA-EM", Discussion and Limitations (1). |
| 1.3 | MAJOR | All real-shift results are null or negative, and bank-time is reported as a "gain" although the classifier has AUC ≈ 0.5 after the shift. | Do not count bank-time as a success; report deployment AUC per suite. | **Fixed.** Table 3 reports deployment AUC; text states the bank-time gain is spurious. |
| 1.4 | MODERATE | The error-analysis tables were computed on batches pooled over all four base learners while the text says "LightGBM". | Recompute on LightGBM only. | **Fixed (code bug).** `model` column added to per-batch records; main experiment re-run; all 90,720 aggregated rows reproduced with zero difference; Tables 13, S8, S9 and Fig. 8 regenerated. |
| 1.5 | MODERATE | The no-shift cost of adaptation is not quantified with a test. | Add paired test at ρ = 1. | **Fixed.** −0.003, 95 % CI −0.006 to −0.002, Holm p = 0.002 (Section 6.1). |
| 1.6 | MODERATE | "Threshold moves with prevalence" statistic had the wrong sign convention (2 % agreement). | Check the direction against Corollary 2. | **Fixed.** On the raw score the optimal threshold falls as prevalence rises; agreement 89 %. |
| 1.7 | MODERATE | σ = 2 was "pre-specified", but σ = 1 is slightly better in the ablation. Reporting σ = 1 as the method would be post-hoc tuning. | Keep σ = 2 and report the sensitivity honestly. | **Done.** Section 6.3 (ii). |
| 1.8 | MINOR | Abstract exceeded 250 words; tables were not numbered in order of appearance; headings were coloured. | Fix. | **Fixed.** 249 words; tables re-ordered; styles corrected. |
| 1.9 | MINOR | The PDF produced through LibreOffice silently dropped every equation. | Use another PDF route and verify visually. | **Fixed.** PDF is now rendered from HTML/MathML with headless Chromium; pages inspected. The DOCX contains 112 native Word equation objects. |

## Not fixable
* The absence of a positive real-data result is a property of the evidence, not of the write-up (see rounds 5 and 10).
