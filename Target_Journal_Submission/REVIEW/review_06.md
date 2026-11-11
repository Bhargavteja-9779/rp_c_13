# Review round 6 — Reviewer #6 (complexity, latency, memory, scalability, deployment)

| # | Severity | Issue | Outcome |
|---|---|---|---|
| 6.1 | MODERATE | Is the improvement worth its cost? | Experiment 6: PAFT-Bayes 0.56 ms per 2,000-score batch and 3.2 ms per 100,000 on one CPU thread; guard 0.19 ms / 4.3 ms; stored state 2.5 MB; retraining the base model 1.5 s. The cost is negligible relative to scoring. Section 6.8, Table 12, Fig. S1. |
| 6.2 | MODERATE | PAFT is slower than DTA-EM for small batches. | Stated: fixed O(G·T) cost of the expectation dominates below m ≈ 10,000. |
| 6.3 | MODERATE | Development-stage time (3.7 s) is dominated by something that is not PAFT. | Stated: 50 bootstrap thresholds of the Static-Bag baseline. |
| 6.4 | MINOR | Energy and FLOPs are not reported. | Not measured; at sub-millisecond CPU latency they would be noise. No energy claim is made. |
| 6.5 | MINOR | Scalability in the number of thresholds and grid points. | G = 161, T ≤ 2,048 by construction, so cost per batch is bounded independently of the validation size. |
| 6.6 | MINOR | Streaming use: PAFT needs a batch to estimate prevalence. | Stated in the Discussion ("after a warm-up batch"); sequential dependence is Limitation 8. |
| 6.7 | MINOR | Timing was measured on one dataset. | Disclosed in the table caption context; latency depends on m, G, T and B only, not on the data. |

No additional experiment was judged necessary.
