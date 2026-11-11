# Review round 8 — Reviewer #8 (a competing researcher who works on prior-shift adaptation)

"I developed EM-based prior correction with decision-theoretic F maximisation. Show me why I need your method."

| # | Severity | Attack | Assessment and outcome |
|---|---|---|---|
| 8.1 | MAJOR | *An existing method already solves the problem.* DTA on EM-corrected posteriors equals PAFT in F1. | **Valid.** The paper now says so in the abstract. What PAFT adds is orthogonal: an explicit threshold, no calibration requirement, and metrics beyond F_β (MCC: +0.014 over static, where DTA has no formula). |
| 8.2 | MAJOR | *The improvement comes from the quantifier, not from your decision rule.* | **Partly valid, tested.** Ablation (iii): quantifier choice changes little. Ablation (i): the posterior-expected rule beats the plug-in with the same likelihood on 18/18 datasets (+0.010), mostly in small batches. Ablation (iv): DTA on PAFT's own posterior is not better than rule (7). |
| 8.3 | MAJOR | *Unfair baseline: you gave EM an isotonic calibrator, known to be poor for EM.* | Tested with the Platt-type calibrator σ(a·logit(s) + b) fitted by maximum likelihood on validation data: DTA-EM 0.532 vs PAFT 0.534 (p = 0.35). For a binary problem this two-parameter map *is* temperature scaling with a bias term, i.e. the bias-corrected temperature scaling recommended for EM by Alexandari et al.; no further calibrator was needed. Vector scaling variants coincide with it in the binary case. |
| 8.4 | MODERATE | *The comparison with Static-F is a straw man; nobody keeps a threshold fixed for years.* | Experiment 9 compares with periodic re-tuning on labelled deployment samples; re-tuning wins whenever a few hundred labels with enough positives are available and the concept has changed. Stated. |
| 8.5 | MODERATE | *Your datasets favour you.* | Selection criteria and screening list published (Table S17). On the two datasets with the fewest validation positives PAFT loses to DTA-EM, and this is reported in the main text. |
| 8.6 | MODERATE | *Your guard is just a KS test.* | Correct; it is presented as a simple, cheap check, with its limitations (approximate null, over-powered for large m). Its value is empirical: it removes the losses on ACS, electricity and KDD99. |
| 8.7 | MODERATE | *With 10 % of the validation data my method is significantly better.* | Reported (−0.004, Holm p = 0.021). |
| 8.8 | MINOR | *With cells of ≥ 5 expected positives PAFT is nominally better (p = 0.009); why not claim it?* | Because the effect is +0.001 F1, depends on a post-hoc restriction, and reverses when all cells are used. Not claimed. |
