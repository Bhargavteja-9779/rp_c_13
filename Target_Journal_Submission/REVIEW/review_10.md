# Review round 10 — Final adversarial review (independent of rounds 1–9; objective: reject)

## One candidate fatal reason
**"The proposed method is no better than an existing baseline under ideal conditions and no better than doing nothing under
realistic conditions."**
This is an accurate one-sentence summary of the F1 results and is the most likely ground for rejection. It is not fixable by
writing or by further analysis of the present data; fixing it would require either a method that genuinely improves on
DTA-EM or a real deployment in which label shift holds. We searched for the latter (six suites, the last chosen specifically
for that purpose) and did not find it. The manuscript is therefore positioned as a study whose findings are (i) the size of
the static-threshold loss, (ii) the equivalence of two adaptive rules, (iii) the necessity and effectiveness of a guard,
(iv) the failure of label-free prevalence adaptation on real tabular shifts. Whether that suffices is an editorial judgement.

## Major weaknesses (genuine ones only; fewer than ten exist, and one of the seven below turned out to be already addressed)
1. No superiority over DTA-EM (acknowledged).
2. No real-data benefit of label-free adaptation (acknowledged; six suites).
3. Controlled benchmark is simulation of the method's own assumption (acknowledged; standard protocol in quantification).
4. Guard null distribution is approximate; over-rejects under the resampling protocol for the rarest classes (acknowledged; calibration study added).
5. The label-assisted variant is post-hoc, helps once and harms once (acknowledged; labelled exploratory).
6. Is the calibrator recommended for EM by Alexandari et al. (bias-corrected temperature scaling) among the baselines? On inspection, yes: in the binary case it is identical to the Platt-type calibrator on the logit score used in the robustness experiment (review 8.3). Resolved; the manuscript now says so.
7. Binary classification only (acknowledged).

## Moderate weaknesses
8. One suite (KDD99) was added after seeing results; disclosed, but it is a deviation from the pre-specified design.
9. Real-suite p-values ignore temporal dependence (caveat added).
10. Base learners untuned.
11. Timing on a single machine and dataset.
12. Journal-specific formatting unverified.
13. Two preprints cited from abstracts only.
14. The Remark on second-order regret is not proved.
15. Screening of candidate datasets was a list of well-known sets, not an exhaustive OpenML query; one eligible set failed to download.
16. Manuscript length.

## Minor weaknesses
17. Exploratory sequential variant only in the supplement.
18. Fig. 7 has six narrow columns and small tick labels.
19. Table 11 is long.
20. "Bank-time" could be dropped from the main table since the classifier is uninformative; kept for transparency.

## Fixes applied in this round
* Fig. 7 regenerated for six suites and inspected.
* Final wording pass on abstract, contributions, Discussion and Conclusion so that no sentence implies a practical benefit the data do not show.
* Limitation 2 now discloses that KDD99 was added after the first results.

## Items deliberately not "fixed"
* The fatal-reason candidate: cannot be fixed honestly.
