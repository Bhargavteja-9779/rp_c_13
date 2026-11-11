# Review round 5 — Reviewer #5 (dataset validity, generalisation, external validity, deployment)

Goal of this round: invalidate the central claim.

| # | Severity | Issue | Outcome |
|---|---|---|---|
| 5.1 | **FATAL for any claim of practical benefit** | The only setting in which the method helps is one where label shift is true by construction. On every real suite it does not help. A claim that PAFT "improves deployed classifiers" would be false. | **The claim is not made.** The paper claims effectiveness under label shift and the need for a guard otherwise. To test whether the negative result is due to an unlucky choice of suites, a sixth suite was added from a domain where the class plausibly causes the features (network intrusion, KDD Cup 1999, time-ordered; split rule fixed before running). Result: again negative (PAFT-Bayes 0.947 vs Static-F 0.973; even Oracle-π is below static). Reported in full. |
| 5.2 | MAJOR | Where exactly does the chain break on real data? | Oracle decomposition added: Oracle-π isolates the threshold map, the quantifier MAE isolates prevalence estimation. ACS: map transfers, quantifier fails. Electricity, KDD99: map fails. Section 6.6, Table 3. |
| 5.3 | MAJOR | If quantification is the weak link, test a label-based estimate. | New Experiment 9 (label-assisted PAFT-L). Helps on ACS (+0.009 with 50 labels), neutral on three suites, **harmful on KDD99 (−0.022)**, and inferior to direct re-tuning under concept change. Reported with all signs. |
| 5.4 | MODERATE | Dataset selection could favour the method. | Criteria fixed a priori; screening list of 30 datasets with reasons added (Table S17); one eligible dataset could not be downloaded and this is disclosed. |
| 5.5 | MODERATE | Feature-noise experiment as a proxy for covariate shift. | Kept as a controlled violation; it agrees qualitatively with the real suites (unguarded adaptation loses its advantage; the guard fires). |
| 5.6 | MODERATE | Creditcard-time relies on row order being time order (the OpenML copy lacks the Time column). | The first row of OpenML 1597 matches the first row of the original file (V1 = −1.3598, Amount = 149.62); stated as an assumption in the loader docstring. |
| 5.7 | MINOR | Electricity has five non-monotone date steps out of 45,312. | Negligible; noted here. |

**Conclusion of the round.** The central *empirical* claims survive because they were narrowed to what the data show. The work
does not demonstrate external validity of prevalence adaptation; it demonstrates its absence on six suites, which is itself the finding.
