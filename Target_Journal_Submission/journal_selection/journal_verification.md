# Journal verification log

## Method
1. **Acceptance time (primary evidence, measured).** For each candidate ISSN we queried the Crossref REST API
   (`/journals/{issn}/works`, journal articles published from 2026-01-01) and read the `received` and `accepted`
   assertions deposited by the publisher. Days = accepted − received. Script: `evidence/crossref_acceptance_time_probe.py`.
   For the selected journal we downloaded *all* 2025–2026 records (1,361 with both dates):
   `evidence/jksucis_received_accepted_2025_2026.json`.
2. **Impact factor / quartile (secondary).** Clarivate JCR is subscription-only; we did not access it. Values come from
   search-engine summaries of the publisher page and of JCR-reproducing aggregators, which disagree with each other
   for several journals. They are labelled "indicative" everywhere.
3. **Publisher pages.** ScienceDirect returned HTTP 403 and Springer Link a JavaScript bot challenge. We did **not**
   bypass either. Guideline details therefore rely on search-engine summaries and generic Springer Nature rules.

## Selected journal – measured acceptance-time distribution
| Acceptance year | n | median | Q1 | Q3 | ≤ 60 d | ≤ 72 d |
|---|---|---|---|---|---|---|
| 2025 | 455 | 87 | 68 | 116 | 17 % | 32 % |
| 2026 | 904 | 68 | 53 | 94 | 38 % | 56 % |
| 2026, May onwards | 553 | 67 | 52 | 91 | 39 % | 58 % |

## Other journals measured (median received→accepted days, 2026 articles)
Energy Informatics 76 · AI and Ethics 100 · J. Eng. Appl. Sci. 100 · J. Real-Time Image Proc. 102 · BioData Mining 109 ·
IET Smart Grid 111 · J. Transl. Med. 113 · IET GTD 119 · High Voltage 121 · Discover AI 126 · GIScience & RS 126 ·
Scientific Reports 127 · Adv. Intell. Syst. 130 · Brain Informatics 130 · Concurrency & Computation 130 ·
Systems Sci. & Control Eng. 131 · Engineering Reports 132 · J. Cheminformatics 133 · Cybersecurity 134 ·
Int. J. Comput. Intell. Syst. 134 · BMC Bioinformatics 139 · Int. J. Digital Earth 142 · Complex & Intell. Syst. 150 ·
Engineering with Computers 151 · IET RPG 155 · J. Intell. Manuf. 159 · CAAI TRIT 160 · Int. J. Intell. Syst. 162 ·
J. Cloud Computing 163 · Expert Systems 166 · npj Digital Medicine 170 · AI Review 177 · J. Supercomputing 178 ·
BMC Med. Inform. Decis. Mak. 181 · IET ITS 183 · EPJ Data Science 184 · Machine Learning 205 · Connection Science 211 ·
Applied Intelligence 237 · DMKD 242 · J. Big Data 258 · Neural Comput. & Appl. 258 · IJMLC 290 · Cluster Computing 347 ·
Multimedia Tools & Appl. 364 · Financial Innovation 524.

Publishers that deposit no received/accepted dates (cannot be measured this way): Elsevier, Oxford University Press,
PeerJ, The Royal Society.

## Legitimacy / integrity notes on the selected journal
* Long-established university journal (King Saud University), formerly hosted by Elsevier, now Springer Nature.
* It is a **high-volume, fast, APC-funded** journal (≈ 950 articles indexed for 2026 at the time of the probe).
  Such journals are periodically re-evaluated by indexers; **confirm current SCIE/JCR status on the day of submission.**
* We found no retraction-wave or delisting report in our searches, but the search was not exhaustive.

## Requirements that could not be satisfied or verified
* "Oxford-type" publisher with ≤ 72-day verified acceptance: **none found**.
* JCR quartile and JIF: **secondary evidence only**.
* APC amount, journal-specific template and length limits: **not verified**.
