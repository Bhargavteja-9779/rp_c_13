"""Build the manuscript (Markdown -> DOCX with native equations -> PDF) and the supplementary file.

Every number, table and figure in the text is injected from the machine-generated files in
<root>/tables, <root>/results and <root>/figures; the template contains no results."""
from __future__ import annotations
import os, sys, re, json, glob, subprocess, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from refs import REFS  # noqa: E402


def f3(x): return f"{x:.3f}"
def s3(x): return f"{x:+.3f}".replace("-", "−")
def ci(lo, hi): return f"{s3(lo)} to {s3(hi)}"
def pv(p): return "< 0.001" if p < 0.001 else f"{p:.3f}"
def pvs(p): return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def md_table(df, caption, fmt=None, index=True, note=None):
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    def cell(v, c):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v): return "–"
            f = (fmt or {}).get(c, "{:.3f}")
            return f.format(v).replace("-", "−")
        if isinstance(v, (int, np.integer)): return f"{v:,}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(cell(r[c], str(c)) for c in d.columns) + " |")
    out = "\n".join(lines) + "\n\n: " + caption + "\n"
    if note:
        out += "\n*Note.* " + note + "\n"
    return out


def build(root):
    T = lambda n, **k: pd.read_csv(os.path.join(root, "tables", n + ".csv"), **k)
    S = json.load(open(os.path.join(root, "results", "statistics.json")))
    P = {}
    t1 = T("table1_datasets", index_col=0); t2 = T("table2_main_by_shift", index_col=0)
    t3 = T("table3_stats_shifted").set_index("baseline"); t3b = T("table3b_stats_noshift").set_index("baseline")
    ranks = T("table3c_ranks", index_col=0)["average rank"]
    t4b = T("table4b_stats_by_model"); t6 = T("table6_ablation", index_col=0); t6b = T("table6b_ablation_stats").set_index("baseline")
    t5 = T("table5_by_batch_size", index_col=0); t7 = T("table7_robustness", index_col=0); t7c = T("table7c_robustness_stats")
    t8 = T("table8_other_metrics", index_col=0); t8b = T("table8b_other_metrics_stats")
    t9 = T("table9_real_suites", index_col=0); t9b = T("table9b_real_stats"); t9c = T("table9c_real_suite_info", index_col=0)
    t10 = T("table10_error_by_positives", index_col=0); t10b = T("table10b_error_by_shift", index_col=0); t10c = T("table10c_error_by_dataset", index_col=0)
    t11 = T("table11_efficiency"); t12 = T("table12_labelled", index_col=[0, 1]); t12b = T("table12b_labelled_stats"); t13 = T("table13_guard_alpha", index_col=0)
    gf = T("tableS_guard_fire_by_dataset", index_col=0).iloc[:, 0]; fq = pd.read_csv(os.path.join(root, "results", "controlled", "main__quant.csv"))
    main = pd.read_csv(os.path.join(root, "results", "controlled", "main.csv"))
    sh = "shifted (ρ≠1)"
    n_ds = S["n_datasets"]; n_fits = main.groupby(["dataset", "model", "seed"]).ngroups
    P.update(n_ds=n_ds, n_seeds=main.seed.nunique(), n_fits=n_fits, n_batches_fit=f"{int(main[(main.method=='static')&(main.dataset==main.dataset.iloc[0])&(main.model=='lgbm')&(main.seed==0)].n_batches.sum()):,}",
             n_batches_main=f"{int(main[main.method=='static'].n_batches.sum()):,}", prev_min=f"{t1['prevalence_%'].min():.2f}", prev_max=f"{t1['prevalence_%'].max():.1f}", dup_max=f"{int(t1.duplicate_rows.max()):,}")
    a = t3.loc["Static-F"]; d = t3.loc["DTA-EM"]
    P.update(gain_static=s3(a.mean_diff), gain_static_ci=ci(a.ci95_lo, a.ci95_hi), p_static=pv(a.p_holm), rb_static=f"{a.rank_biserial:.2f}",
             gain_dta=s3(d.mean_diff), gain_dta_abs=f3(abs(d.mean_diff)), p_dta=f"{d.p_wilcoxon:.3f}", wins_dta=int(d.wins), med_dta=s3(d.median_diff),
             gain_emfixed=s3(t3.loc["EM-Fixed"].mean_diff), gain_prevmatch=s3(t3.loc["PrevMatch"].mean_diff),
             f_static=f3(t2.loc["Static-F", sh]), f_oraclepi=f3(t2.loc["Oracle-π", sh]), f_oraclethr=f3(t2.loc["Oracle-threshold", sh]), f_paft=f3(t2.loc["PAFT-Bayes", sh]), f_dta=f3(t2.loc["DTA-EM", sh]),
             f_static_8=f3(t2.loc["Static-F", "8.0"]), f_oraclepi_8=f3(t2.loc["Oracle-π", "8.0"]), f_dta_1=f3(t2.loc["DTA-EM", "1.0"]), f_paft_1=f3(t2.loc["PAFT-Bayes", "1.0"]),
             gap_closed=f"{100*(t2.loc['PAFT-Bayes', sh]-t2.loc['Static-F', sh])/(t2.loc['Oracle-π', sh]-t2.loc['Static-F', sh]):.0f}",
             rank_paft=f"{ranks['PAFT-Bayes']:.2f}", rank_dta=f"{ranks['DTA-EM']:.2f}", cd=f"{S['friedman']['nemenyi_cd']:.2f}", friedman_chi=f"{S['friedman']['chi2']:.1f}", friedman_p=pv(S["friedman"]["p"]),
             seed_gain=f"mean {s3(S['seed_gain_vs_static']['mean'])}, standard deviation {S['seed_gain_vs_static']['sd']:.4f} over the five seeds")
    b = t3b.loc["Static-F"]; P.update(noshift_cost=f3(abs(b.mean_diff)), noshift_ci=ci(b.ci95_lo, b.ci95_hi), noshift_p=f"{b.p_holm:.3f}")
    for m in ["xgb", "logreg", "mlp"]:
        P["gain_" + m] = s3(t4b[(t4b.model == m) & (t4b.baseline == "Static-F")].mean_diff.iloc[0])
    for m in ["xgb", "mlp"]:
        P["p_dta_" + m] = f"{t4b[(t4b.model == m) & (t4b.baseline == 'DTA-EM')].p_wilcoxon.iloc[0]:.3f}"
    eq = S["equivalence_paft_vs_dta"]; P.update(eq_ci=ci(eq["ci90_lo"], eq["ci90_hi"]), eq_max=f3(eq["max_abs_dataset_diff"]), eq_ds=eq["dataset_of_max"])
    sx = T("tableS_sensitivity_expected_positives"); gsx = lambda c, bl: s3(sx[(sx.cells == c) & (sx.baseline == bl)].mean_diff.iloc[0])
    P.update(sens5_static=gsx("expected positives >= 5", "static"), sens20_static=gsx("expected positives >= 20", "static"), sens5_dta=gsx("expected positives >= 5", "dta_em"), sens20_dta=gsx("expected positives >= 20", "dta_em"))
    g8 = lambda mt, bl, c: t8b[(t8b.metric == mt) & (t8b.baseline == bl)][c].iloc[0]
    P.update(gain_f05=s3(g8("f0.5", "static", "mean_diff")), gain_f2=s3(g8("f2", "static", "mean_diff")), gain_mcc=s3(g8("mcc", "static", "mean_diff")),
             dta_f05=f3(t8.loc["DTA-EM", "f0.5"]), paft_f05=f3(t8.loc["PAFT-Bayes", "f0.5"]), p_f05_dta=f"{g8('f0.5', 'dta_em', 'p_wilcoxon'):.2f}")
    r = lambda k: t6.loc[k]
    pl, by = r("Plug-in, mixture MLE (PAFT-Plug)"), r("Bayes, σ=2 (PAFT-Bayes)")
    P.update(abl_plug=s3(t6b.loc["paft_plug"].mean_diff), abl_plug_100=s3(by["100"] - pl["100"]), abl_plug_10000=s3(by["10000"] - pl["10000"]),
             abl_flat=f3(t6b.loc["bayes_sdNone"].mean_diff), abl_sd1=s3(-t6b.loc["bayes_sd1.0"].mean_diff), abl_sd1_p=f"{t6b.loc['bayes_sd1.0'].p_holm:.3f}",
             abl_dtahist=s3(-t6b.loc["dta_hist"].mean_diff), abl_dtahist_p=f"{t6b.loc['dta_hist'].p_wilcoxon:.3f}", abl_guard=f3(abs(t3.loc["PAFT-Guard"].mean_diff)),
             gain_m100=s3(t5.loc["PAFT-Bayes", "100"] - t5.loc["Static-F", "100"]), gain_m10000=s3(t5.loc["PAFT-Bayes", "10000"] - t5.loc["Static-F", "10000"]))
    g7 = lambda cnd, bl, c: t7c[(t7c.condition == cnd) & (t7c.baseline == bl)][c].iloc[0]
    P.update(val25_gain=s3(g7("valfrac_0.25", "static", "mean_diff")), val10_gain=s3(g7("valfrac_0.1", "static", "mean_diff")), val10_dta=s3(g7("valfrac_0.1", "dta_em", "mean_diff")), val10_dta_p=f"{g7('valfrac_0.1', 'dta_em', 'p_holm'):.3f}",
             platt_dta=f3(t7.loc["DTA-EM", "calib_platt"]), platt_paft=f3(t7.loc["PAFT-Bayes", "calib_platt"]), platt_p=f"{g7('calib_platt', 'dta_em', 'p_wilcoxon'):.2f}",
             noise01_paft=f3(t7.loc["PAFT-Bayes", "noise_0.1"]), noise01_dta=f3(t7.loc["DTA-EM", "noise_0.1"]), noise01_static=f3(t7.loc["Static-F", "noise_0.1"]),
             noise01_guard=f3(t7.loc["PAFT-Guard", "noise_0.1"]), noise025_guard=f3(t7.loc["PAFT-Guard", "noise_0.25"]), noise05_guard=f3(t7.loc["PAFT-Guard", "noise_0.5"]))
    gfr = T("table7b_guard_fire_rates").set_index("condition")["guard_fire_rate"]
    P["noise01_fire"] = f"{100*gfr['noise_0.1']:.0f}"
    for m_ in [500, 2000, 10000]:
        P[f"fire_{m_}"] = f"{100*t13.loc[0.01, f'fire_rate_m{m_}']:.1f}"
    z = np.load(os.path.join(root, "scores", "controlled", "creditcard__lgbm__s0.npz"))
    P.update(fire_aps=f"{100*gf['APSFailure']:.0f}", fire_median16=f"{100*gf.drop(['creditcard', 'APSFailure']).median():.1f}", cc_valpos=int(z["y_val"].sum()), cc_testpos=int(z["y_test"].sum()))
    # real
    g9 = lambda su, bl, c: t9b[(t9b.suite == su) & (t9b.baseline == bl)][c].iloc[0]
    A = "acs_income_states"; E = "electricity_time"; Bk = "bank_time"
    P.update(acs_pimin=f"{t9c.loc[A, 'pi_min']:.2f}", acs_pimax=f"{t9c.loc[A, 'pi_max']:.2f}", acs_pi0=f"{t9c.loc[A, 'pi0']:.2f}", acs_loss=f3(abs(g9(A, "static", "mean_diff"))),
             acs_loss_ci=ci(g9(A, "static", "ci95_lo"), g9(A, "static", "ci95_hi")), acs_oraclegain=s3(t9.loc["Oracle-π", A] - t9.loc["Static-F", A]), acs_mae=f"{t9c.loc[A, 'MAE_pi_hat']:.2f}",
             elec_loss=f3(abs(g9(E, "static", "mean_diff"))), elec_oraclepi=f3(t9.loc["Oracle-π", E]), elec_static=f3(t9.loc["Static-F", E]), elec_oraclethr=f3(t9.loc["Oracle-threshold", E]),
             bank_gain=s3(g9(Bk, "static", "mean_diff")), bank_auc=f"{t9c.loc[Bk, 'auc']:.2f}", bank_pimax=f"{t9c.loc[Bk, 'pi_max']:.2f}", real_fire_min=f"{100*t9c.guard_fire_rate.min():.0f}")
    K = "kdd99_time"
    P.update(kdd_pimin=f"{t9c.loc[K, 'pi_min']:.2f}", kdd_pimax=f"{t9c.loc[K, 'pi_max']:.2f}", kdd_auc=f"{t9c.loc[K, 'auc']:.3f}", kdd_paft=f3(t9.loc["PAFT-Bayes", K]), kdd_static=f3(t9.loc["Static-F", K]),
             kdd_loss=s3(g9(K, "static", "mean_diff")), kdd_loss_p=f"{g9(K, 'static', 'p_wilcoxon'):.3f}", kdd_loss_holm=f"{g9(K, 'static', 'p_holm'):.2f}", kdd_oraclepi=f3(t9.loc["Oracle-π", K]),
             gsim_share=f"{100*S['guard_sim']['share_cells_le_nominal']:.0f}", gsim_max=f"{100*S['guard_sim']['max_fire_h0']:.1f}", gsim_power=f"{100*S['guard_sim']['min_power_shift05_n500']:.1f}")
    gl = lambda su, nl, bl, c: t12b[(t12b.suite == su) & (t12b.n_lab == nl) & (t12b.baseline == bl)][c].iloc[0]
    P.update(lab_acs50=s3(gl(A, 50, "static", "mean_diff")), lab_acs50_ci=ci(gl(A, 50, "static", "ci95_lo"), gl(A, 50, "static", "ci95_hi")), lab_acs50_retune=s3(gl(A, 50, "retune_labels", "mean_diff")),
             lab_acs1000_retune=s3(gl(A, 1000, "retune_labels", "mean_diff")), lab_ctl50=s3(gl("controlled", 50, "static", "mean_diff")), lab_cc200_retune=s3(gl("creditcard_time", 200, "retune_labels", "mean_diff")),
             lab_elec_retune=f3(t12.loc[(E, 50), "Re-tune on labels"]))
    P.update(lab_kdd50=s3(gl(K, 50, "static", "mean_diff")), lab_kdd50_p=f"{gl(K, 50, 'static', 'p_holm'):.3f}")
    ge = lambda comp, m_: t11[(t11.component.str.startswith(comp)) & (t11.m == m_)].ms.iloc[0]
    P.update(eff_paft_2000=f"{ge('PAFT-Bayes', 2000):.2f}", eff_paft_100000=f"{ge('PAFT-Bayes', 100000):.1f}", eff_guard_2000=f"{ge('PAFT guard', 2000):.2f}", eff_guard_100000=f"{ge('PAFT guard', 100000):.1f}",
             eff_dta_2000=f"{ge('DTA-EM', 2000):.2f}", eff_dta_100000=f"{ge('DTA-EM', 100000):.1f}", eff_mem=f"{t11[t11.component.str.startswith('stored')].peak_mb.iloc[0]:.1f}",
             eff_fit=f"{t11[t11.component.str.startswith('fit decision')].ms.iloc[0]/1000:.1f}", eff_retrain=f"{t11[t11.component.str.startswith('retrain')].ms.iloc[0]/1000:.1f}")
    e = S["error"]
    P.update(err_win8=f"{100*t10b.loc[8.0, 'win']:.0f}", err_win05=f"{100*t10b.loc[0.5, 'win']:.0f}", err_loss05=f"{100*t10b.loc[0.5, 'loss']:.0f}", err_worstdecile=s3(t10b.worst_decile_gain.min()),
             err_regret_15=f3(t10.loc["1-5", "regret"]), err_regret_1000=f3(t10.loc[">1000", "regret"]), err_rel_15=f"{t10.loc['1-5', 'rel_err_pi']:.2f}", err_rel_1000=f"{t10.loc['>1000', 'rel_err_pi']:.2f}",
             err_paft_15=f3(t10.loc["1-5", "F1_paft"]), err_static_15=f3(t10.loc["1-5", "F1_static"]), err_dta_15=f3(t10.loc["1-5", "F1_dta"]),
             err_cc_gain=s3(t10c.loc["creditcard", "gain_vs_static"]), err_cc_dta=f3(abs(t10c.loc["creditcard", "gain_vs_dta"])),
             err_sp_auc=f"{e['spearman_gain_auc'][0]:.2f}", err_sp_auc_p=f"{e['spearman_gain_auc'][1]:.2f}", err_sp_prev=f"{e['spearman_gain_prev'][0]:.2f}", err_sp_prev_p=f"{e['spearman_gain_prev'][1]:.2f}",
             err_dir=f"{100*e['threshold_moves_in_expected_direction_rate']:.0f}")

    # ------------------------------------------------------------------ tables
    TB = {}
    gap = pd.DataFrame([
        ["Lipton et al. [⟦c:lipton2014⟧]", "F1-optimal threshold", "No", "–", "–", "–"],
        ["Saerens et al. [⟦c:saerens2002⟧]; Alexandari et al. [⟦c:alexandari2020⟧]", "Posterior correction", "Estimated (EM)", "No (accuracy, calibration)", "No", "Simulated"],
        ["Forman [⟦c:forman2008⟧]; Esuli et al. [⟦c:esuli2023⟧]", "Prevalence estimation", "Estimated", "No decision rule", "No", "Simulated, some real"],
        ["Ye et al. [⟦c:ye2012⟧]", "Expected-F maximisation", "No", "Yes", "–", "–"],
        ["Esposito et al. [⟦c:esposito2021⟧]", "Static threshold tuning", "No", "No (kappa)", "–", "–"],
        ["Godau et al. [⟦c:godau2025⟧]", "Re-calibration + Bayes rule", "Assumed known / external", "No (expected cost)", "No", "Simulated (images)"],
        ["Alsulaimawi [⟦c:alsulaimawi2026⟧]", "Online Bayes threshold", "Estimated online", "No (cost-sensitive)", "No", "Simulated"],
        ["This work", "Threshold from ROC surface", "Estimated, with posterior", "Yes (F~β~, MCC)", "Yes", "Simulated and six real suites"],
    ], columns=["Work", "Decision mechanism", "Deployment prevalence", "F-optimal under shift", "Validity check", "Shift in evaluation"])
    TB["gap"] = md_table(gap, "Position of this work relative to the closest prior work", index=False)
    d1 = t1.rename(columns={"n": "Instances", "features": "Features", "prevalence_%": "Prevalence (%)", "duplicate_rows": "Duplicate rows", "AUC_mean": "AUC", "AP_mean": "AP"})[["Instances", "Features", "Prevalence (%)", "Duplicate rows", "AUC", "AP"]]
    d1.index.name = "Dataset"; d1["Instances"] = d1["Instances"].astype(int); d1["Features"] = d1["Features"].astype(int); d1["Duplicate rows"] = d1["Duplicate rows"].astype(int)
    TB["datasets"] = md_table(d1, f"Datasets of the controlled-shift benchmark, ordered by minority prevalence. AUC and average precision (AP) are test-pool means over five seeds for LightGBM", fmt={"Prevalence (%)": "{:.2f}"},
                              note="Duplicate rows are rows whose feature vector occurs more than once; the group-aware split keeps them in one partition.")
    d2 = t2.copy(); d2.columns = ["ρ=1/8", "ρ=1/4", "ρ=1/2", "ρ=1", "ρ=2", "ρ=4", "ρ=8", "Shifted", "All", "Regret"]; d2.index.name = "Rule"
    TB["main"] = md_table(d2, f"Mean F1 by odds multiplier ρ (LightGBM; mean over {n_ds} datasets, five seeds, three batch sizes and 30 batches per cell)",
                          note="Shifted = mean over ρ ≠ 1. Regret = mean difference to Oracle-threshold over ρ ≠ 1. Oracle rows use deployment labels and are bounds, not competitors.")
    d3 = t3.reset_index()[["baseline", "mean_diff", "ci95_lo", "ci95_hi", "wins", "losses", "rank_biserial", "p_wilcoxon", "p_holm"]]
    d3["W/L"] = d3.wins.astype(str) + "/" + d3.losses.astype(str); d3["95% CI"] = [ci(a_, b_) for a_, b_ in zip(d3.ci95_lo, d3.ci95_hi)]
    d3["p"] = d3.p_wilcoxon.map(pv); d3["p (Holm)"] = d3.p_holm.map(pv)
    d3 = d3.rename(columns={"baseline": "PAFT-Bayes versus", "mean_diff": "Mean ΔF1", "rank_biserial": "Rank-biserial"})[["PAFT-Bayes versus", "Mean ΔF1", "95% CI", "W/L", "Rank-biserial", "p", "p (Holm)"]]
    TB["stats"] = md_table(d3, f"Paired comparison of PAFT-Bayes with each other rule under shift (ρ ≠ 1); unit of analysis: dataset (n = {n_ds})", index=False, fmt={"Mean ΔF1": "{:+.4f}", "Rank-biserial": "{:.2f}"},
                           note="Two-sided Wilcoxon signed-rank test; Holm correction over the nine comparisons; bootstrap percentile confidence interval of the mean difference; W/L = datasets won/lost by PAFT-Bayes.")
    d8 = t8.rename(columns={"f0.5": "F0.5", "f2": "F2", "mcc": "MCC"}); d8.index.name = "Rule"
    TB["metrics"] = md_table(d8, "Other target metrics under shift (LightGBM, m = 2,000, ρ ≠ 1). Each rule optimises the metric of the column", note="DTA-EM is defined for F-measures only.")
    d6 = t6.copy(); d6.columns = ["m=100", "m=500", "m=2,000", "m=10,000", "No shift (ρ=1)"]; d6.index.name = "Variant"
    TB["ablation"] = md_table(d6, "Ablation of PAFT components: mean F1 under shift (ρ ≠ 1) by batch size, and without shift (LightGBM)")
    d7 = t7.rename(columns={"clean (m=2000)": "Clean", "calib_platt": "Platt calibration", "noise_0.1": "Noise 0.1", "noise_0.25": "Noise 0.25", "noise_0.5": "Noise 0.5", "valfrac_0.1": "10% validation", "valfrac_0.25": "25% validation"}); d7.index.name = "Rule"
    TB["robust"] = md_table(d7[["Clean", "25% validation", "10% validation", "Platt calibration", "Noise 0.1", "Noise 0.25", "Noise 0.5"]], "Robustness: mean F1 under shift (LightGBM, m = 2,000, ρ ≠ 1)",
                            note="Noise columns add Gaussian feature noise (in units of each feature's standard deviation) to deployment data, which violates the label-shift assumption.")
    d13 = t13[["controlled_F1_shifted", "controlled_F1_noshift", "fire_rate_m500", "fire_rate_m2000", "fire_rate_m10000", "real_acs_income_states", "real_electricity_time", "real_kdd99_time"]].copy()
    d13.columns = ["F1 shifted", "F1 no shift", "Fire rate m=500", "Fire rate m=2,000", "Fire rate m=10,000", "F1 ACS", "F1 electricity", "F1 KDD99"]; d13.index = [f"{a_:g}" for a_ in d13.index]; d13.index.name = "α_g"
    TB["guard"] = md_table(d13, "Sensitivity of PAFT-Guard to the guard level (controlled benchmark and three real suites)", note="Fire rates are fractions of batches under pure label shift on which the guard rejects.")
    d9c = t9c.copy(); d9c["m"] = d9c.m_min.astype(int).map("{:,}".format) + "–" + d9c.m_max.astype(int).map("{:,}".format)
    d9c = d9c.rename(columns={"batches": "Batches", "pi0": "π0 (dev.)", "pi_min": "π min", "pi_max": "π max", "auc": "AUC (deploy.)", "MAE_pi_hat": "MAE of π̂", "guard_fire_rate": "Guard fire rate"})[["Batches", "m", "π0 (dev.)", "π min", "π max", "AUC (deploy.)", "MAE of π̂", "Guard fire rate"]]
    d9c.index.name = "Suite"; d9c["Batches"] = d9c["Batches"].astype(int)
    TB["realinfo"] = md_table(d9c, "Real-shift suites (LightGBM, means over five seeds)", fmt={"π0 (dev.)": "{:.4f}", "π min": "{:.4f}", "π max": "{:.4f}", "MAE of π̂": "{:.4f}", "Guard fire rate": "{:.2f}"},
                              note="AUC is the mean per-batch area under the ROC curve on deployment data; MAE of π̂ is the mean absolute error of the PAFT posterior-mean prevalence.")
    d9 = t9.drop(index=["PAFT-Seq (exploratory)"]).rename(columns=lambda c: c.replace("_", "-")); d9.index.name = "Rule"
    TB["real"] = md_table(d9[["acs-income-states", "bank-time", "creditcard-time", "electricity-time", "elliptic-time", "kdd99-time"]], "Mean F1 over deployment batches on the real-shift suites (LightGBM, five seeds)",
                          note="Bank-time values reflect a classifier with deployment AUC of about 0.5 and are not evidence of useful adaptation (see text).")
    d12 = t12.reset_index(); d12 = d12[d12.n_lab.isin([50, 200])].rename(columns={"suite": "Suite", "n_lab": "Labels"}); d12["Suite"] = d12["Suite"].str.replace("_", "-")
    TB["labelled"] = md_table(d12, "Label-assisted adaptation: mean F1 on the unlabelled remainder of each batch when n_L instances are labelled (LightGBM)", index=False,
                              note="Controlled rows: m = 2,000, ρ ≠ 1. PAFT-L uses the labels only to estimate prevalence; 'Re-tune on labels' picks the F1-maximising threshold on the labelled instances.")
    d11 = t11[t11.component.isin(["static threshold (comparison only)", "EM prior correction (SLD)", "DTA-EM (EM + sort + expected-F scan)", "PAFT-Plug", "PAFT-Bayes", "PAFT guard (KS test)"])].pivot(index="component", columns="m", values="ms")
    d11.columns = [f"m={int(c):,}" for c in d11.columns]; d11.index.name = "Component"
    TB["eff"] = md_table(d11, "Latency per deployment batch in milliseconds (one CPU thread, mean of 30 runs)", fmt={c: "{:.3f}" for c in d11.columns})
    d10 = t10.rename(columns={"batches": "Batches", "F1_static": "Static-F", "F1_paft": "PAFT-Bayes", "F1_dta": "DTA-EM", "loss_rate_vs_static": "Share worse than static", "regret": "Regret", "rel_err_pi": "Median rel. error of π̂"}); d10.index.name = "Positives in batch"
    d10["Batches"] = d10["Batches"].astype(int)
    TB["error"] = md_table(d10, "Error analysis by number of positives in the batch (LightGBM, controlled benchmark, all shift levels)", fmt={"Share worse than static": "{:.2f}", "Median rel. error of π̂": "{:.2f}"})

    FIG = {"workflow": ("fig1_workflow.png", "Workflow of PAFT. The development stage runs once on the labelled validation split; the deployment stage runs on every unlabelled batch"),
           "mechanism": ("fig2_mechanism.png", "Why a static threshold is suboptimal under prevalence shift (bank-marketing data, LightGBM, seed 0). a F1 as a function of the threshold for three prevalences. b The F1-optimal threshold on the raw score as a function of prevalence. c F1 regret of the static threshold. Dotted lines mark the validation prevalence and its threshold"),
           "main": ("fig3_main.png", "Controlled label shift (LightGBM). a Mean difference in F1 to the static threshold as a function of the odds multiplier. b Per-dataset difference between PAFT-Bayes and Static-F averaged over shifted levels"),
           "ranks": ("fig4_ranks.png", "Average ranks of the ten label-free rules over the 18 datasets under shift, with the Nemenyi critical difference at the 0.05 level"),
           "ablation": ("fig5_ablation.png", "Ablation. a Mean F1 under shift by batch size. b Gain over Static-F as a function of the prior standard deviation σ"),
           "robust": ("fig6_robustness.png", "Robustness. a Feature noise on deployment data violates label shift. b Reduced validation sets"),
           "real": ("fig7_real.png", "Real-shift suites (LightGBM, mean over five seeds). Top: true and estimated prevalence per batch; the dotted line is the development prevalence. Bottom: F1 per batch. PAFT-Guard coincides with Static-F wherever the guard fires"),
           "error": ("fig8_error.png", "Error analysis on the controlled benchmark (LightGBM, ρ ≠ 1). a Distribution of per-batch differences between PAFT-Bayes and Static-F. b Regret against the oracle threshold by decile of relative prevalence error. c Share of batches better and worse than Static-F by number of positives")}

    txt = open(os.path.join(root, "manuscript", "manuscript_template.md"), encoding="utf8").read()
    # tables: explicit placement, numbering by first mention
    for k, v in TB.items():
        txt = txt.replace(f"⟦TABLE:{k}⟧", v)
    txt_tables = {k: v for k, v in TB.items()}
    order = []
    for m in re.finditer(r"⟦T:(\w+)⟧", txt):
        if m.group(1) not in order: order.append(m.group(1))
    # tables referenced but not explicitly placed -> after the paragraph of first mention
    for k in order:
        if txt_tables[k] not in txt:
            i = txt.index(f"⟦T:{k}⟧"); j = txt.index("\n\n", i)
            txt = txt[:j] + "\n\n" + txt_tables[k] + txt[j:]
    for n, k in enumerate(order, 1):
        txt = txt.replace(f"⟦T:{k}⟧", str(n))
    # caption prefixes "Table n"
    def number_caption(m, _c=[0]):
        _c[0] += 1; return f"\n: **Table {_c[0]}** "
    # captions appear in document order; make sure order of placement == order of first mention
    placed = [k for k in sorted(TB, key=lambda k: txt.find(TB[k].split("\n\n: ")[1][:40])) if TB[k].split("\n\n: ")[1][:40] in txt]
    for k in placed:
        cap = TB[k].split("\n\n: ")[1].split("\n")[0]
        txt = txt.replace("\n: " + cap, f"\n: **Table {order.index(k)+1}** " + cap)
    forder = []
    for m in re.finditer(r"⟦F:(\w+)⟧", txt):
        if m.group(1) not in forder: forder.append(m.group(1))
    for n, k in enumerate(forder, 1):
        i = txt.index(f"⟦F:{k}⟧"); j = txt.index("\n\n", i)
        fn, cap = FIG[k]
        txt = txt[:j] + f"\n\n![**Fig. {n}** {cap}](../figures/{fn}){{width=6.3in}}" + txt[j:]
        txt = txt.replace(f"⟦F:{k}⟧", str(n))
    for k, v in P.items():
        txt = txt.replace(f"⟦{k}⟧", str(v))
    # citations numbered in order of first appearance
    corder = []
    for m in re.finditer(r"⟦c:(\w+)⟧", txt):
        if m.group(1) not in corder: corder.append(m.group(1))
    missing = [c for c in corder if c not in REFS]
    assert not missing, f"unknown citation keys: {missing}"
    for n, k in enumerate(corder, 1):
        txt = txt.replace(f"⟦c:{k}⟧", str(n))
    txt = re.sub(r"\[(\d+(?:, \d+)*)\]", lambda m: "[" + ", ".join(map(str, sorted(map(int, m.group(1).split(", "))))) + "]", txt)
    txt = txt.replace("⟦REFERENCES⟧", "\n\n".join(f"{n}. {REFS[k]}" for n, k in enumerate(corder, 1)))
    left = re.findall(r"⟦[^⟧]*⟧", txt)
    assert not left, f"unfilled placeholders: {sorted(set(left))[:20]}"
    out_md = os.path.join(root, "manuscript", "Final_Manuscript.md")
    open(out_md, "w", encoding="utf8").write(txt)
    json.dump({k: str(v) for k, v in P.items()}, open(os.path.join(root, "manuscript", "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)
    print(f"manuscript: {len(corder)} references, {len(order)} tables, {len(forder)} figures, {len(P)} injected values")
    return out_md


def supplementary(root):
    T = lambda n, **k: pd.read_csv(os.path.join(root, "tables", n + ".csv"), **k)
    parts = ["---\ntitle: \"Supplementary Information: Prevalence-adaptive F-measure thresholding for imbalanced tabular classification under label shift\"\n---\n",
             "# S1 Implementation details\n\n"
             "**Base learners (identical for all datasets).** LightGBM: up to 2,000 trees, learning rate 0.05, 31 leaves, row and column subsampling 0.8, minimum 20 samples per leaf, early stopping after 50 rounds on a 10% slice of the training split. "
             "XGBoost: histogram method, depth 6, learning rate 0.05, subsampling 0.8, up to 2,000 trees, early stopping after 50 rounds. Logistic regression: L2 penalty with C = 1, at most 300 L-BFGS iterations. "
             "Perceptron: 256 and 128 ReLU units, dropout 0.1, AdamW (learning rate 0.001, weight decay 1e-5), batch size 512, at most 40 epochs, patience 5.\n\n"
             "**Pre-processing.** Categorical features are ordinal-encoded for tree models and frequency-encoded for the linear and neural models; the latter also receive median imputation and standardisation. All statistics are fitted on the training split.\n\n"
             "**Decision layer.** Candidate thresholds: validation scores of each class, thinned to at most 1,024 class-wise quantiles. Histogram: B = min(32, max(8, ⌊n1/10⌋)) class-balanced quantile bins, Laplace smoothing 0.5. Prevalence grid: 161 points uniform in logit(π0) ± 9. Prior: σ = 2. Guard level: 0.01. "
             "EM baseline: at most 200 iterations, tolerance 1e-7, on isotonic-calibrated posteriors.\n\n"
             "**Random seeds.** Splits and learners use seeds 0–4. Static-Bag thresholds are computed on candidate thresholds thinned to 512 quantiles per class. Deployment batches use a generator seeded by a CRC-32 hash of (dataset, model, seed, condition, metric, ρ, m), so every result is exactly reproducible; a complete re-run of the main experiment reproduced all 90,720 result rows with zero difference.\n\n"
             "**Hardware and software.** 4 CPU cores, 15 GB RAM, no GPU; Python 3.13, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, PyTorch 2.15.0 (CPU). The device (CPU, CUDA or Apple MPS) is detected automatically.\n"]
    def add(title, df, cap, index=True, fmt=None):
        parts.append(f"\n# {title}\n\n" + md_table(df, cap, index=index, fmt=fmt))
    add("S2 Results by base learner", T("table4_by_model", index_col=0), "**Table S1** Mean F1 under shift (ρ ≠ 1) by base learner")
    add("S2 (continued)", T("table4b_stats_by_model").drop(columns=["reference", "median_diff"]), "**Table S1b** Paired tests of PAFT-Bayes by base learner (unit: dataset)", index=False, fmt={"p_wilcoxon": "{:.5f}", "p_holm": "{:.5f}", "mean_diff": "{:+.4f}", "ci95_lo": "{:+.4f}", "ci95_hi": "{:+.4f}"})
    add("S3 Batch size", T("table5_by_batch_size", index_col=0), "**Table S2** Mean F1 under shift by batch size (LightGBM)")
    add("S4 Seed variability", T("tableS_seed_variability", index_col=0), "**Table S3** Mean F1 under shift: mean, standard deviation, minimum and maximum over the five seeds (each seed value is a mean over 18 datasets)", fmt={"sd_over_seeds": "{:.4f}"})
    r9 = T("table9_real_suites", index_col=0); add("S5 Real suites: all rules including the exploratory sequential variant", r9, "**Table S4** Mean F1 on real-shift suites. PAFT-Seq centres the prior on the previous batch's posterior mean (σ = 1); it was not part of the pre-specified comparison")
    add("S5 (continued)", T("table9_real_suites_sd", index_col=0), "**Table S4b** Standard deviation over seeds of the values in Table S4", fmt=None)
    add("S5 (continued) ", T("tableS_real_by_model"), "**Table S4c** Real-shift suites by base learner", index=False)
    add("S6 Guard firing rate by dataset", T("tableS_guard_fire_by_dataset", index_col=0), "**Table S5** Fraction of batches under pure label shift on which the guard rejects (α = 0.01, LightGBM, all batch sizes)", fmt={"guard_fire_rate_under_pure_label_shift": "{:.4f}"})
    add("S7 Robustness: paired tests", T("table7c_robustness_stats").drop(columns=["reference", "median_diff"]), "**Table S6** Paired tests of PAFT-Bayes under each robustness condition (unit: dataset)", index=False, fmt={"p_wilcoxon": "{:.4f}", "p_holm": "{:.4f}", "mean_diff": "{:+.4f}", "ci95_lo": "{:+.4f}", "ci95_hi": "{:+.4f}"})
    add("S8 Real suites: paired tests", T("table9b_real_stats").drop(columns=["reference", "median_diff"]), "**Table S7** Paired tests of PAFT-Bayes on real suites (unit: deployment batch, averaged over seeds)", index=False, fmt={"p_wilcoxon": "{:.4f}", "p_holm": "{:.4f}", "mean_diff": "{:+.4f}", "ci95_lo": "{:+.4f}", "ci95_hi": "{:+.4f}"})
    add("S9 Error analysis by shift level", T("table10b_error_by_shift", index_col=0), "**Table S8** Per-batch win and loss rates of PAFT-Bayes against Static-F by odds multiplier")
    add("S9 (continued)", T("table10c_error_by_dataset", index_col=0), "**Table S9** Per-dataset gains, relative prevalence error and regret (ρ ≠ 1)")
    add("S10 Quantifier error", T("table6c_quantifier_mae", index_col=0), "**Table S10** Mean absolute error of prevalence estimators by batch size (the row guard_fire_rate is the guard's rejection rate)", fmt=None)
    add("S11 Ablation: paired tests", T("table6b_ablation_stats").drop(columns=["reference", "median_diff"]), "**Table S11** Paired tests of PAFT-Bayes against ablated variants (all batch sizes)", index=False, fmt={"p_wilcoxon": "{:.5f}", "p_holm": "{:.5f}", "mean_diff": "{:+.4f}", "ci95_lo": "{:+.4f}", "ci95_hi": "{:+.4f}"})
    add("S12 Other metrics: paired tests", T("table8b_other_metrics_stats").drop(columns=["reference", "median_diff"]), "**Table S12** Paired tests for F0.5, F2 and MCC", index=False, fmt={"p_wilcoxon": "{:.5f}", "p_holm": "{:.5f}", "mean_diff": "{:+.4f}", "ci95_lo": "{:+.4f}", "ci95_hi": "{:+.4f}"})
    add("S13 Label-assisted variant: all budgets and paired tests", T("table12_labelled"), "**Table S13** Label-assisted adaptation for all label budgets", index=False)
    add("S13 (continued)", T("table12b_labelled_stats").drop(columns=["reference", "median_diff"]), "**Table S13b** Paired tests of guarded PAFT-L", index=False, fmt={"p_wilcoxon": "{:.4f}", "p_holm": "{:.4f}", "mean_diff": "{:+.4f}", "ci95_lo": "{:+.4f}", "ci95_hi": "{:+.4f}"})
    add("S14 Per-dataset results", T("tableS_per_dataset_shifted", index_col=0), "**Table S14** Mean F1 under shift per dataset (LightGBM)")
    add("S15 Full efficiency measurements", T("table11_efficiency"), "**Table S15** All latency and memory measurements", index=False, fmt={"ms": "{:.3f}", "peak_mb": "{:.2f}"})
    add("S16 Guard calibration on synthetic scores (Experiment 10)", T("table14_guard_calibration"), "**Table S16** Rejection rate of the guard (level 0.01) on binormal scores (class separation 2 s.d., development prevalence 0.05) when every batch is a fresh sample. score_shift = 0 is pure label shift (rate = size); score_shift > 0 displaces all deployment scores by that many s.d. (rate = power). 500 batches per cell over 10 validation draws", index=False)
    scr = pd.DataFrame([("adult", 1590, "included"), ("bank-marketing", 1461, "included"), ("creditcard", 1597, "included"), ("default-of-credit-card-clients", 42477, "included"), ("APSFailure", 41138, "included"),
                        ("KDDCup09_appetency", 1111, "included"), ("KDDCup09_churn", 1112, "included"), ("KDDCup09_upselling", 1114, "included"), ("Click_prediction_small", 1220, "included"), ("nomao", 1486, "included"),
                        ("MiniBooNE", 41150, "included"), ("jm1", 1053, "included"), ("Amazon_employee_access", 4135, "included"), ("kick", 41162, "included"), ("Diabetes130US", 4541, "included (binarised: readmission < 30 days)"),
                        ("online_shoppers", 45060, "included"), ("porto-seguro", 42742, "included"), ("Census-Income-KDD", 42750, "included"),
                        ("MagicTelescope", 1120, "excluded: minority prevalence 35 %"), ("eeg-eye-state", 1471, "excluded: minority prevalence 45 %"), ("electricity", 151, "excluded: minority prevalence 42 % (used in the temporal suite)"),
                        ("PhishingWebsites", 4534, "excluded: minority prevalence 44 %"), ("airlines", 1169, "excluded: minority prevalence 45 %"), ("higgs", 23512, "excluded: minority prevalence 47 %"),
                        ("numerai28.6", 23517, "excluded: minority prevalence 49 %"), ("mammography", 310, "excluded: 260 minority instances"), ("road-safety", 42803, "excluded: target not binary"),
                        ("data id 45557", 45557, "excluded: fewer than 10,000 instances"), ("SantanderCustomerSatisfaction", 42395, "eligible by description but download failed repeatedly; not included")], columns=["Dataset", "OpenML id", "Decision"])
    add("S17 Dataset screening", scr, "**Table S17** OpenML datasets screened against the inclusion criteria (binary target, at least 10,000 instances, at least 400 minority instances, minority prevalence at most 30 %)", index=False)
    add("S18 Sensitivity to cells with few expected positives", T("tableS_sensitivity_expected_positives").drop(columns=["reference", "median_diff"]), "**Table S18** Paired tests of PAFT-Bayes restricted to benchmark cells whose expected number of positives (target prevalence times batch size) exceeds a minimum", index=False, fmt={"p_wilcoxon": "{:.5f}", "p_holm": "{:.5f}", "mean_diff": "{:+.4f}", "ci95_lo": "{:+.4f}", "ci95_hi": "{:+.4f}", "share_of_cells": "{:.2f}"})
    parts.append("\n![**Fig. S1** Latency of the decision rules as a function of batch size (one CPU thread)](../figures/fig9_efficiency.png){width=3.5in}\n")
    p = os.path.join(root, "supplementary", "Supplementary_Information.md"); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf8").write("\n".join(parts)); return p


def to_docx(md, out, root):
    ref = os.path.join(root, "manuscript", "reference.docx")
    if not os.path.exists(ref):
        subprocess.run(f'pandoc -o "{ref}" --print-default-data-file reference.docx', shell=True, check=True)
        _style_reference(ref)
    subprocess.run(["pandoc", md, "-o", out, "--reference-doc", ref, "--from", "markdown+pipe_tables+table_captions+implicit_figures+tex_math_dollars+subscript+superscript", "--resource-path", os.path.dirname(md)], check=True)
    _postprocess(out)


CSS = """
body{font-family:'Liberation Serif','Times New Roman',serif;font-size:11pt;line-height:1.45;max-width:none;margin:0;color:#000}
h1.title{font-size:16pt;text-align:center;margin:0 0 14pt}
h1{font-size:13pt;margin:16pt 0 6pt} h2{font-size:11.5pt;margin:12pt 0 4pt}
table{border-collapse:collapse;margin:8pt auto;font-size:8pt;line-height:1.2;border-top:1.2px solid #000;border-bottom:1.2px solid #000;page-break-inside:avoid}
th{border-bottom:0.8px solid #000;padding:2pt 5pt;text-align:right} td{padding:1.5pt 5pt;text-align:right} th:first-child,td:first-child{text-align:left}
caption{caption-side:top;font-size:9.5pt;text-align:left;margin-bottom:3pt}
figure{margin:10pt 0;text-align:center;page-break-inside:avoid} figure img{max-width:100%} figcaption{font-size:9.5pt;text-align:left}
p{margin:0 0 6pt;text-align:justify} math[display=block]{margin:8pt 0}
ol{padding-left:22pt} li p{margin:0 0 3pt}
"""


def to_pdf(md, out):
    """PDF through HTML + MathML rendered by headless Chromium (LibreOffice here cannot render Word equations)."""
    html = out[:-4] + ".html"; css = out[:-4] + ".css"
    open(css, "w").write(CSS)
    subprocess.run(["pandoc", md, "-o", html, "-s", "--mathml", "--embed-resources", "--css", css, "--metadata", "lang=en", "--resource-path", os.path.dirname(md),
                    "--from", "markdown+pipe_tables+table_captions+implicit_figures+tex_math_dollars+subscript+superscript"], check=True)
    js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
          "await p.goto('file://'+process.argv[1]);await p.pdf({path:process.argv[2],format:'A4',margin:{top:'25mm',bottom:'25mm',left:'25mm',right:'25mm'},"
          "displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style=\"font-size:8px;width:100%;text-align:center\"><span class=\"pageNumber\"></span></div>'});await b.close();})();")
    r = subprocess.run(["node", "-e", js, os.path.abspath(html), os.path.abspath(out)], capture_output=True, text=True)
    if r.returncode != 0:
        print("PDF generation failed:", r.stderr[:300])
    for f in (html, css):
        if os.path.exists(f): os.remove(f)



def _style_reference(path):
    from docx import Document
    from docx.shared import Pt
    d = Document(path)
    from docx.shared import RGBColor
    for st in d.styles:
        try:
            if st.font is not None:
                st.font.name = "Times New Roman"; st.font.color.rgb = RGBColor(0, 0, 0)
                rpr = st.element.get_or_add_rPr(); rf = rpr.get_or_add_rFonts()
                for at in ('ascii', 'hAnsi', 'cs', 'eastAsia'):
                    rf.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}' + at, 'Times New Roman')
                for at in ('asciiTheme', 'hAnsiTheme', 'cstheme', 'eastAsiaTheme'):
                    k = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}' + at
                    if k in rf.attrib: del rf.attrib[k]
                st.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Times New Roman") if st.element.rPr is not None and st.element.rPr.rFonts is not None else None
        except Exception:
            pass
    for name, size, bold in [("Normal", 11, None), ("Body Text", 11, None), ("First Paragraph", 11, None), ("Compact", 9, None), ("Title", 16, True), ("Heading 1", 13, True), ("Heading 2", 11.5, True), ("Heading 3", 11, True), ("Image Caption", 9.5, None), ("Table Caption", 9.5, None), ("Caption", 9.5, None)]:
        try:
            s = d.styles[name]; s.font.size = Pt(size)
            if bold is not None: s.font.bold = bold
            from docx.shared import RGBColor
            s.font.color.rgb = RGBColor(0, 0, 0)
            if name in ("Body Text", "First Paragraph", "Normal"):
                s.paragraph_format.line_spacing = 1.5; s.paragraph_format.space_after = Pt(6)
        except KeyError:
            pass
    d.save(path)


def _postprocess(path):
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    d = Document(path)
    for sec in d.sections:
        sec.left_margin = sec.right_margin = Inches(1); sec.top_margin = sec.bottom_margin = Inches(1)
        ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); sec._sectPr.append(ln)   # continuous line numbers for review
        # page number in footer
        p = sec.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(); f1 = OxmlElement("w:fldSimple"); f1.set(qn("w:instr"), "PAGE"); r._r.append(f1)
    for t in d.tables:
        t.alignment = 1
        tblPr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom"):
            e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "8"); e.set(qn("w:color"), "000000"); borders.append(e)
        tblPr.append(borders)
        wide = len(t.columns) >= 8
        nc = len(t.columns); first = Inches(1.9 if nc <= 6 else 1.25); rest = (Inches(6.4) - first) // max(nc - 1, 1)
        t.autofit = False
        for ci_, col in enumerate(t.columns):
            for c in col.cells:
                c.width = first if ci_ == 0 else rest
        for i, row in enumerate(t.rows):
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0); p.paragraph_format.line_spacing = 1.0
                    for r in p.runs:
                        r.font.size = Pt(7 if wide else 8); r.font.name = "Times New Roman"
                        if i == 0: r.font.bold = True
                if i == 0:
                    tcPr = c._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders"); e = OxmlElement("w:bottom"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000"); b.append(e); tcPr.append(b)
    for p in d.paragraphs:
        if p.style.name in ("Image Caption", "Table Caption", "Captioned Figure"):
            p.paragraph_format.keep_with_next = p.style.name == "Table Caption"
        if p.style.name.startswith("Heading"):
            p.paragraph_format.keep_with_next = True
    d.save(path)


def title_page(root):
    from docx import Document
    from docx.shared import Pt
    d = Document(); st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
    d.add_heading("Title page (submitted separately for double-anonymised review)", 2)
    d.add_paragraph("Title: Prevalence-adaptive F-measure thresholding for imbalanced tabular classification under label shift: a guarded decision layer and its limits")
    for l in ["Authors: [Author 1 full name]¹, [Author 2 full name]², …", "Affiliations: ¹[Department, Institution, City, Country]; ²[Department, Institution, City, Country]",
              "Corresponding author: [Name], [e-mail address], [postal address]", "ORCID: [0000-0000-0000-0000] for each author",
              "Acknowledgements: [To be completed, or 'Not applicable']", "Funding: [To be completed]", "Competing interests: [To be completed]",
              "Author contributions (CRediT): [To be completed]",
              "Use of AI tools: [The authors must disclose here, in line with Springer Nature policy, how generative AI tools were used in conducting the research, producing code and drafting the manuscript, and confirm that the human authors take full responsibility for the content. An AI tool cannot be listed as an author.]"]:
        d.add_paragraph(l)
    d.save(os.path.join(root, "manuscript", "Title_Page.docx"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".."); a = ap.parse_args()
    md = build(a.root)
    to_docx(md, os.path.join(a.root, "manuscript", "Final_Manuscript.docx"), a.root)
    sp = supplementary(a.root)
    to_docx(sp, os.path.join(a.root, "supplementary", "Supplementary_Information.docx"), a.root)
    title_page(a.root)
    to_pdf(md, os.path.join(a.root, "manuscript", "Final_Manuscript.pdf"))
    to_pdf(sp, os.path.join(a.root, "supplementary", "Supplementary_Information.pdf"))
    print("built")
