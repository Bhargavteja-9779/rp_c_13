# Target_Journal_Submission

Submission package for **Journal of King Saud University – Computer and Information Sciences** (Springer Nature).

> **Read first:** `REVIEW/final_audit.md` (what a human must verify and complete before submitting, and the honest
> rejection-risk assessment) and `journal_selection/selected_journal.md` (what could and could not be verified about the journal).

| Folder | Content |
|---|---|
| `journal_selection/` | selected journal, comparison of candidates, verification log, measured acceptance-time evidence |
| `manuscript/` | `Final_Manuscript.docx` (anonymised, native Word equations, line numbers), `Final_Manuscript.pdf`, `Title_Page.docx`, `cover_letter.docx`, the template and the generated Markdown source |
| `supplementary/` | `Supplementary_Information.docx/.pdf` (implementation details, Tables S1–S18, Fig. S1) |
| `code/` | complete source, configs, tests, `run_all.py`, reproduction README |
| `results/` | `raw_results.csv`, `metrics.json`, `statistics.json`, per-experiment result files, `experiment_logs/` |
| `tables/` | every table as CSV and Markdown (machine-generated) |
| `figures/` | every figure (600 dpi PNG, machine-generated) |
| `REVIEW/` | ten review rounds with outcomes, claim–evidence audit, citation audit, consistency-check output, final audit |

## Result in three sentences
Under controlled label shift, re-deriving the F-measure threshold from an estimated prevalence beats the static threshold on
all 18 datasets (+0.036 F1), and the proposed rule is equivalent to — not better than — the strongest existing rule.
On six real temporal or cross-site shifts, label-free adaptation by any method did not help and sometimes hurt; the proposed
validity guard detected this and prevented the loss.
No acceptance outcome or timeline can be guaranteed.
