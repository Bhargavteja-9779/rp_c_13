"""Final consistency audit (Phase 34): manuscript <-> generated results.

Checks that (1) the manuscript build is deterministic, (2) every injected value is traceable to a table/statistics file
(by construction: the template has no numbers of its own in results sentences), (3) every figure and table exists,
(4) the DOCX contains the expected equation / table / figure objects, (5) every citation key resolves and every reference is cited."""
from __future__ import annotations
import os, sys, re, json, hashlib, zipfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from refs import REFS
import build_manuscript as bm

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (" — " + detail if detail else ""))

md = os.path.join(ROOT, "manuscript", "Final_Manuscript.md")
h0 = hashlib.sha256(open(md, "rb").read()).hexdigest(); bm.build(ROOT); h1 = hashlib.sha256(open(md, "rb").read()).hexdigest()
check("manuscript regenerates identically from result files", h0 == h1)
t = open(md, encoding="utf8").read(); tpl = open(os.path.join(ROOT, "manuscript", "manuscript_template.md"), encoding="utf8").read()
check("no unfilled placeholders", not re.findall(r"⟦[^⟧]*⟧", t))
res = tpl[tpl.index("# 6 Results"):tpl.index("# 7 Discussion")]
hard = [m for m in re.finditer(r"(?<![\w⟧.])[0-9]\.[0-9]{3}(?![0-9])", res) if not res[max(0, m.start() - 12):m.start()].rstrip().endswith(("<", "between"))]
check("no hand-typed 3-decimal results in the Results template", len(hard) == 0, f"{len(hard)} found")
figs = re.findall(r"\(\.\./figures/([\w.]+)\)", t)
check("all referenced figures exist", all(os.path.exists(os.path.join(ROOT, "figures", f)) for f in figs), f"{len(figs)} figures")
tn = list(map(int, re.findall(r"\*\*Table (\d+)\*\*", t))); fn = list(map(int, re.findall(r"\*\*Fig\. (\d+)\*\*", t)))
check("tables numbered consecutively in order of appearance", tn == list(range(1, len(tn) + 1)), str(tn))
check("figures numbered consecutively in order of appearance", fn == list(range(1, len(fn) + 1)), str(fn))
for n in tn: check(f"Table {n} is cited in the text", re.search(rf"Table {n}\b(?!\*)", t) is not None)
for n in fn: check(f"Fig. {n} is cited in the text", re.search(rf"Fig\. {n}\b(?!\*)", t) is not None)
body, refs = t.split("# References")
cited = set(int(x) for g in re.findall(r"\[(\d+(?:, \d+)*)\]", body) for x in g.split(", ")); nref = len(re.findall(r"^\d+\. ", refs, flags=re.M))
check("every reference is cited and every citation resolves", cited == set(range(1, nref + 1)), f"{nref} references")
x = zipfile.ZipFile(os.path.join(ROOT, "manuscript", "Final_Manuscript.docx")).read("word/document.xml").decode()
check("DOCX has native equations, tables and figures", x.count("<m:oMath>") > 50 and x.count("<w:tbl>") == len(tn) and x.count("<pic:pic") == len(fn), f"{x.count('<m:oMath>')} equations, {x.count('<w:tbl>')} tables, {x.count('<pic:pic')} figures")
check("DOCX has continuous line numbering", "w:lnNumType" in x)
S = json.load(open(os.path.join(ROOT, "results", "statistics.json"))); P = json.load(open(os.path.join(ROOT, "manuscript", "manuscript_numbers.json")))
check("abstract headline numbers equal statistics.json", f"{S['main']['PAFT-Bayes'] - S['main']['Static-F']:+.3f}" == P["gain_static"].replace("−", "-"), P["gain_static"])
import pandas as pd
T = lambda n: pd.read_csv(os.path.join(ROOT, "tables", n + ".csv"))
t4b = T("table4b_stats_by_model"); check("claim: PAFT vs Static-F Holm p < 0.001 for every base learner", (t4b[t4b.baseline == "Static-F"].p_holm < 0.001).all())
t8b = T("table8b_other_metrics_stats"); check("claim: PAFT vs Static-F Holm p < 0.01 for F0.5, F2, MCC", (t8b[t8b.baseline == "static"].p_holm < 0.01).all())
t6b = T("table6b_ablation_stats").set_index("baseline"); check("claim: Bayes vs plug-in Holm p < 0.001, 18/18 wins", t6b.loc["paft_plug"].p_holm < 0.001 and t6b.loc["paft_plug"].wins == 18)
t9b = T("table9b_real_stats"); g = lambda su: t9b[(t9b.suite == su) & (t9b.baseline == "static")].iloc[0]
check("claim: significant loss vs Static-F on ACS and electricity (Holm p < 0.001)", g("acs_income_states").p_holm < 0.001 and g("electricity_time").p_holm < 0.001 and g("acs_income_states").mean_diff < 0 and g("electricity_time").mean_diff < 0)
t7 = T("table7_robustness").set_index("method"); lf = [i for i in t7.index if not i.startswith("Oracle")]
check("claim: PAFT-Guard has the highest label-free mean F1 at every noise level", all(t7.loc[lf, c].idxmax() == "PAFT-Guard" for c in t7.columns if c.startswith("noise")))
t12b = T("table12b_labelled_stats"); x = t12b[(t12b.suite == "acs_income_states") & (t12b.n_lab == 50) & (t12b.baseline == "static")].iloc[0]; check("claim: PAFT-L (50 labels) beats Static-F on ACS, Holm p < 0.001", x.p_holm < 0.001 and x.mean_diff > 0)
t3 = T("table3_stats_shifted").set_index("baseline"); check("claim: PAFT-Bayes beats Static-F on all datasets", t3.loc["Static-F"].losses == 0 and t3.loc["Static-F"].ties == 0)
check("claim: equivalence with DTA-EM at margin 0.01 (90 % CI)", S["equivalence_paft_vs_dta"]["equivalent_at_margin"])
r = subprocess.run([sys.executable, "-m", "pytest", "-q", os.path.join(HERE, "..", "tests")], capture_output=True, text=True)
check("unit tests pass", r.returncode == 0, r.stdout.strip().splitlines()[-1])
print("\nOVERALL:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
