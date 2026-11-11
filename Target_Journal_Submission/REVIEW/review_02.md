# Review round 2 — Reviewer #2 (novelty, prior art, contribution differentiation)

**Recommendation at this stage: major revision; novelty is modest and must be claimed accordingly.**

| # | Severity | Issue | Required fix | Outcome |
|---|---|---|---|---|
| 2.1 | MAJOR | Equation (2) and the dependence of the F-optimal operating point on class skew are textbook ROC analysis (Provost & Fawcett 2001; Fawcett 2006; Flach & Kull 2015). Presenting the "metric surface" as a contribution would be overclaiming. | Attribute the identity to prior work; restrict the analytical claim to the shifted-prior optimality condition and its corollaries. | **Fixed.** Section 2 ("classical"), Section 7 ("Our analytical additions are modest"). |
| 2.2 | MAJOR | The plug-in rule is nothing but importance-weighted re-tuning of the threshold on validation data. A reader should not have to discover this. | Say so explicitly. | **Fixed.** Sentence added to Section 4.1. |
| 2.3 | MAJOR | The quantifier (mixture maximum likelihood on binned scores) is the Saerens EM estimator on histogram-calibrated scores, closely related to HDy/DyS. | Do not present it as new; compare against existing quantifiers. | **Fixed.** Section 4.2 disclaims novelty; ablation (iii) compares ACC, PACC, EM and HDy plug-ins. |
| 2.4 | MAJOR | Is the gap genuine? Godau et al. (MICCAI 2023 / Med. Image Anal. 2025) already propose a prevalence-aware workflow. | Read the paper and state precisely what it does and does not do. | **Checked against the full text.** They re-calibrate with estimated/known prevalences and use the Bayes rule for expected cost; F1 is evaluated but no F-optimal rule under estimated prevalence is given; imbalance ratio ≤ 10; simulated shift only; no validity test. Table 1 states this. The journal version (Med. Image Anal. 102:103504) is cited instead of the preprint. |
| 2.5 | MODERATE | Two 2026 preprints are close in spirit (online Bayes threshold for cost-sensitive decisions; 45-task threshold-tuning study). | Cite and differentiate. | **Done** (refs. Alsulaimawi 2026; Musaev 2026). Only abstracts were read for these two; the text attributes to them only what the abstracts state. |
| 2.6 | MODERATE | The decision-theoretic approach (Ye et al. 2012) on prior-corrected posteriors is the obvious competitor and is absent from most prior-shift papers. Is it implemented faithfully? | Document the approximation used. | **Done.** Section 5.4 gives the plug-in expected-F formula (ratio-of-expectations approximation, not the exact O(n³) algorithm). This deviation from the exact method is disclosed. |
| 2.7 | MODERATE | An arXiv search for "label shift" + F1/F-measure/threshold returned only tangential work; that supports the gap, but an absence of hits is weak evidence. | Phrase the gap with "to our knowledge". | **Done.** No "first" or "no one has" claims remain. |
| 2.8 | MINOR | Jansche (2005) was cited for the decision-theoretic approach although it concerns training. | Correct the attribution. | **Fixed.** |

## Residual novelty assessment
The defensible novel elements are: (a) the shifted-prior F_β optimality condition with two corollaries that the experiments
confirm; (b) the posterior-expected threshold rule; (c) the label-shift validity guard in a thresholding pipeline; (d) the
two-sided benchmark. None is a breakthrough; together they are an incremental-to-moderate contribution whose main value is
the evaluation. This is recorded as a remaining rejection risk.
