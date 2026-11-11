# Selected target journal

| Field | Value | Verification status |
|---|---|---|
| TARGET JOURNAL | Journal of King Saud University – Computer and Information Sciences (JKSUCIS) | — |
| PUBLISHER | Springer Nature (journal no. 44443; published by Elsevier until the end of 2024) | Crossref publisher field of 2025–2026 articles = "Springer Science and Business Media LLC" (measured by us) |
| ISSN | 1319-1578 (print), 2213-1248 (electronic) | Crossref |
| JCR STATUS | Indexed (SCIE) | Secondary sources only – JCR itself is subscription-only and was **not** opened by us |
| JCR QUARTILE | Q1, category "Computer Science, Information Systems" | Secondary source (Peeref's reproduction of the latest JCR table). **Must be confirmed in JCR before submission** |
| JOURNAL IMPACT FACTOR | 6.4 (2025 JIF, as shown on the Springer journal page according to a search-engine summary); 6.1 was the previous year's value | Secondary. We could not load the Springer page ourselves (it is behind a bot challenge that we did not bypass) |
| VERIFIED REVIEW / FIRST-DECISION TIMELINE | **Not verified.** No publisher-stated "submission to first decision" figure was retrieved | Unverified |
| VERIFIED ACCEPTANCE TIMELINE | **Median 68 days received→accepted** for the 904 articles accepted in 2026 (IQR 53–94 days; 56 % ≤ 72 days; 38 % ≤ 60 days). For articles accepted since May 2026: median 67 days (n = 553). For 2025 acceptances: median 87 days (n = 455) | **Independently measured by us** from the `received` / `accepted` assertions that Springer deposits in Crossref (script and raw data in `evidence/`) |
| APC | Open-access journal with an article-processing charge. **Amount not verified** (third-party listings conflict) | Unverified – check the journal's "How to publish with us" page |
| PEER REVIEW MODEL | Double-anonymised (search-engine summary of the Springer guidelines) | Secondary |
| SCOPE FIT | "All aspects of both foundations of computer [science] and its practical applications". 2026 articles sampled by us include imbalanced learning, fraud/intrusion detection, concept drift and trust/uncertainty methods | Scope sentence: secondary; article sample: measured |
| RESEARCH FIT | A model-agnostic decision-layer method for imbalanced classification under prevalence drift, evaluated on fraud, churn, intrusion-style and socio-economic tabular data | Our judgement |

## What the 72-day requirement does and does not mean here

* The measured figure is **submission(received)-to-acceptance**, not first decision and not publication.
* 68 days is a **median over accepted papers**. About 44 % of papers accepted in 2026 took longer than 72 days, and
  the figure says nothing about rejected papers. It is **not** a promise that this manuscript will be accepted in 72 days.
* No candidate we could measure does better: of 100+ Q1-range journals probed (Springer Nature, Wiley, IET, Taylor & Francis)
  JKSUCIS was the only one with a median at or below 72 days.
* Elsevier and Oxford University Press do not deposit received/accepted dates in Crossref, so their journals could not be
  measured this way; the Elsevier journals for which the publisher's own "submission to acceptance" figure surfaced were all
  far above 72 days (130–183 days). No OUP journal with a verifiable ≤ 72-day acceptance time was found.

## Key author guidelines used for the manuscript (Springer Nature journal style)
Title page separated from the anonymised manuscript (double-anonymised review); abstract of 150–250 words; 4–6 keywords;
numbered sections; numbered references in order of citation; figures/tables cited in order with captions;
mandatory end-matter "Declarations" (funding, competing interests, data availability, code availability, author contributions,
ethics approval where applicable). **These follow the generic Springer Nature journal guidelines; the journal-specific page
could not be loaded by us and must be checked before submission.**

## Key rejection risks
1. Reviewer taste: the journal mostly publishes named deep-learning architectures; a decision-layer method paper is less typical.
2. "Combination of known ingredients" objection (ROC-space prevalence reweighting and quantification are both classical).
3. Results on real (non-simulated) shift are mixed – see the manuscript and `REVIEW/final_audit.md`.
4. Unverified items above (JCR quartile, APC, journal-specific formatting).
