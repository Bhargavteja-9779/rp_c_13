"""Cover letter (DOCX). Numbers are read from the generated manuscript_numbers.json."""
import os, sys, json
from docx import Document
from docx.shared import Pt, Inches
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
P = json.load(open(os.path.join(root, "manuscript", "manuscript_numbers.json")))
d = Document(); st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11.5)
for s in d.sections: s.left_margin = s.right_margin = Inches(1.1); s.top_margin = s.bottom_margin = Inches(1)
def para(t, bold=False, after=8):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = bold; p.paragraph_format.space_after = Pt(after); return p
para("[Corresponding author name]\n[Affiliation]\n[Postal address]\n[E-mail] · [ORCID]"); para("[Date]")
para("To the Editor-in-Chief\nJournal of King Saud University – Computer and Information Sciences\nSpringer Nature")
para("Submission of an original research article", bold=True)
para("Dear Editor,")
para("We submit for your consideration the manuscript “Prevalence-adaptive F-measure thresholding for imbalanced tabular classification under label shift: a guarded decision layer and its limits”.")
para("What the paper is about. Classifiers for rare events are almost always deployed with a decision threshold tuned for the F-measure on validation data. That threshold is optimal only for the validation prevalence, and prevalence changes after deployment. We ask how much is lost, how much can be recovered without labels, and when recovery fails.")
para(f"What we contribute. (1) A characterisation of the F-beta-optimal threshold under a shifted class prior, which explains why applying a fixed threshold to prior-corrected posteriors is biased. (2) PAFT, a model-agnostic decision layer that infers the deployment prevalence from unlabelled scores, chooses the threshold with the highest posterior-expected metric, and falls back to the static threshold when a goodness-of-fit test rejects the label-shift assumption. (3) An evaluation on {P['n_ds']} public tabular datasets with four base learners and {P['n_batches_main']} simulated deployment batches, with dataset-level paired tests, effect sizes, ablations, robustness, cost and error analyses. (4) An evaluation on six real temporal or cross-site shifts.")
para(f"What we found, including what did not work. Under controlled label shift, adapting the threshold improves mean F1 over the static threshold on all {P['n_ds']} datasets ({P['gain_static']}; Holm-adjusted p {P['p_static']}). Our rule is statistically equivalent to, and not better than, the strongest existing alternative (expected-F maximisation on EM-corrected posteriors); it differs in returning an explicit threshold, needing no calibrated probabilities and covering other metrics such as the Matthews correlation coefficient. On the six real shifts, label-free adaptation by any method gave no reliable benefit and lowered F1 on several; the proposed guard detected the violated assumption and prevented the loss. We report these negative results in the abstract because we believe they are the part of the study most useful to practitioners.")
para("Why this journal. The work addresses a general problem of deployed machine-learning systems in the journal’s application areas (fraud and intrusion detection, churn, credit, socio-economic prediction), is model-agnostic, and is fully reproducible from public data. We think the journal’s readers, many of whom report F-measures for imbalanced problems, will find a tested protocol and a clear statement of its limits useful.")
para("Declarations. The manuscript is original, has not been published, and is not under consideration elsewhere. All authors have approved the submission. The study uses only public, de-identified data and required no ethics approval. [Funding statement.] [Competing-interests statement.] Source code, configurations, raw results and the scripts that regenerate every table and figure accompany the submission.")
para("Use of AI tools. [The authors must describe here, truthfully and in line with Springer Nature policy, the role that generative AI tools played in the design, coding, analysis and drafting of this work, and confirm that the human authors have verified the content and take full responsibility for it.]")
para("The manuscript has been prepared for double-anonymised review; author details are on the separate title page.")
para("Yours sincerely,\n\n[Corresponding author name], on behalf of all authors")
d.save(os.path.join(root, "manuscript", "cover_letter.docx")); print("cover letter written")
