"""Generate every manuscript table and results/statistics.json from machine-written result files.
No number in the manuscript is typed by hand: the DOCX builder reads the files written here."""
from __future__ import annotations
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from src.statistics.tests import paired_table, friedman_ranks, boot_ci

NAMES = {"default_0.5": "Default 0.5", "static": "Static-F", "static_bag": "Static-Bag", "em_0.5": "EM-0.5", "em_fixed": "EM-Fixed",
         "prev_match": "PrevMatch", "dta_em": "DTA-EM", "paft_plug": "PAFT-Plug", "paft_bayes": "PAFT-Bayes", "paft_guard": "PAFT-Guard",
         "paft_seq": "PAFT-Seq (exploratory)", "oracle_pi": "Oracle-π", "oracle_thr": "Oracle-threshold"}
ORDER = ["default_0.5", "static", "static_bag", "em_0.5", "em_fixed", "prev_match", "dta_em", "paft_plug", "paft_bayes", "paft_guard", "oracle_pi", "oracle_thr"]
BASE = ["default_0.5", "static", "static_bag", "em_0.5", "em_fixed", "prev_match", "dta_em"]
REF = "paft_bayes"


def save(df, name, T, index=True, floatfmt="%.3f"):
    df.to_csv(os.path.join(T, name + ".csv"), index=index, float_format="%.6g")
    with open(os.path.join(T, name + ".md"), "w") as f:
        f.write(df.to_markdown(index=index, floatfmt=".3f") if hasattr(df, "to_markdown") else df.to_string())


def main(root):
    R, T = os.path.join(root, "results"), os.path.join(root, "tables"); os.makedirs(T, exist_ok=True)
    S = {}
    C = lambda n: pd.read_csv(os.path.join(R, "controlled", n + ".csv"))
    main_df = C("main")
    mm = "lgbm"
    lg = main_df[main_df.model == mm]

    # ---- T1 datasets ---------------------------------------------------------------------------
    metas = []
    for f in glob.glob(os.path.join(root, "scores", "controlled", "*.npz")):
        metas.append(json.loads(str(np.load(f)["meta"])))
    M = pd.DataFrame(metas)
    g = M[M.model == mm].groupby("dataset")
    t1 = pd.DataFrame({"n": g["n"].first(), "features": g["n_features"].first(), "prevalence_%": 100 * g["prevalence"].first(),
                       "duplicate_rows": g["n_dup_rows"].first(), "cross_split_duplicates": g["cross_split_dups"].max(),
                       "AUC_mean": g["auc_test"].mean(), "AUC_sd": g["auc_test"].std(), "AP_mean": g["ap_test"].mean(), "AP_sd": g["ap_test"].std()}).sort_values("prevalence_%")
    save(t1, "table1_datasets", T)
    t1b = M.groupby(["dataset", "model"])["auc_test"].mean().unstack(); save(t1b, "tableS_auc_by_model", T)
    S["n_datasets"] = int(t1.shape[0]); S["cross_split_duplicates_max"] = int(t1.cross_split_duplicates.max())

    # ---- T2 main result by shift -----------------------------------------------------------------
    def by_shift(df):
        d = df.groupby(["dataset", "rho", "method"])["value"].mean().reset_index()      # average over seeds, m
        t = d.groupby(["method", "rho"])["value"].mean().unstack()
        t["shifted (ρ≠1)"] = d[d.rho != 1].groupby("method")["value"].mean()
        t["all"] = d.groupby("method")["value"].mean()
        rg = df.groupby(["dataset", "rho", "method"])["regret"].mean().reset_index()
        t["regret (ρ≠1)"] = rg[rg.rho != 1].groupby("method")["regret"].mean()
        return t.reindex([m for m in ORDER if m in t.index]).rename(index=NAMES)
    t2 = by_shift(lg); save(t2, "table2_main_by_shift", T)

    # ---- T3 statistics: unit = dataset -------------------------------------------------------------
    def unit_wide(df, sel):
        return df[sel(df)].groupby(["dataset", "method"])["value"].mean().unstack()
    ws = unit_wide(lg, lambda d: d.rho != 1)
    w1 = unit_wide(lg, lambda d: d.rho == 1)
    t3 = paired_table(ws, REF, BASE + ["paft_plug", "paft_guard"]); t3["baseline"] = t3["baseline"].map(NAMES); save(t3, "table3_stats_shifted", T, index=False)
    t3b = paired_table(w1, REF, BASE + ["paft_plug", "paft_guard"]); t3b["baseline"] = t3b["baseline"].map(NAMES); save(t3b, "table3b_stats_noshift", T, index=False)
    cols = BASE + ["paft_plug", "paft_bayes", "paft_guard"]
    ranks, chi, p, cd = friedman_ranks(ws[cols])
    save(ranks.rename(index=NAMES).to_frame("average rank"), "table3c_ranks", T)
    S["friedman"] = dict(chi2=chi, p=p, nemenyi_cd=cd, k=len(cols), n=int(ws.shape[0]))
    S["main"] = {NAMES[k]: float(ws[k].mean()) for k in ws.columns if k in NAMES}
    S["main_noshift"] = {NAMES[k]: float(w1[k].mean()) for k in w1.columns if k in NAMES}
    per_ds = (ws[[REF, "static", "dta_em", "em_fixed", "oracle_pi", "oracle_thr"]]).rename(columns=NAMES); save(per_ds, "tableS_per_dataset_shifted", T)

    # ---- T4 base models -----------------------------------------------------------------------------
    d = main_df[main_df.rho != 1].groupby(["model", "dataset", "method"])["value"].mean().reset_index()
    t4 = d.groupby(["method", "model"])["value"].mean().unstack().reindex(ORDER).rename(index=NAMES); save(t4, "table4_by_model", T)
    st4 = []
    for m in main_df.model.unique():
        w = unit_wide(main_df[main_df.model == m], lambda d: d.rho != 1)
        x = paired_table(w, REF, ["static", "dta_em", "em_fixed"]); x.insert(0, "model", m); st4.append(x)
    st4 = pd.concat(st4); st4["baseline"] = st4["baseline"].map(NAMES); save(st4, "table4b_stats_by_model", T, index=False)

    # ---- by batch size --------------------------------------------------------------------------------
    ab = C("ablation")
    d = ab[ab.rho != 1].groupby(["m", "dataset", "method"])["value"].mean().reset_index()
    keep = ORDER[:10] + ["oracle_pi", "oracle_thr"]
    t5 = d.groupby(["method", "m"])["value"].mean().unstack().reindex(keep).rename(index=NAMES); save(t5, "table5_by_batch_size", T)
    # ---- ablation ---------------------------------------------------------------------------------------
    abl = ["static", "paft_plug", "plug_acc", "plug_pacc", "plug_em", "plug_hdy", "bayes_sd0.5", "bayes_sd1.0", "paft_bayes", "bayes_sd4.0", "bayes_sdNone", "paft_guard", "dta_em", "dta_hist"]
    lab = {"plug_acc": "Plug-in, ACC quantifier", "plug_pacc": "Plug-in, PACC quantifier", "plug_em": "Plug-in, EM (isotonic) quantifier", "plug_hdy": "Plug-in, HDy quantifier",
           "paft_plug": "Plug-in, mixture MLE (PAFT-Plug)", "bayes_sd0.5": "Bayes, σ=0.5", "bayes_sd1.0": "Bayes, σ=1", "paft_bayes": "Bayes, σ=2 (PAFT-Bayes)",
           "bayes_sd4.0": "Bayes, σ=4", "bayes_sdNone": "Bayes, flat prior", "paft_guard": "Bayes σ=2 + guard (PAFT-Guard)", "static": "Static-F", "dta_em": "DTA-EM", "dta_hist": "DTA on PAFT histogram posterior"}
    da = ab.groupby(["m", "dataset", "rho", "method"])["value"].mean().reset_index()
    t6 = da[da.rho != 1].groupby(["method", "m"])["value"].mean().unstack()
    t6["no shift (ρ=1), all m"] = da[da.rho == 1].groupby("method")["value"].mean()
    t6 = t6.reindex(abl).rename(index=lab); save(t6, "table6_ablation", T)
    wab = ab[(ab.rho != 1)].groupby(["dataset", "method"])["value"].mean().unstack()
    s6 = paired_table(wab, "paft_bayes", ["paft_plug", "paft_guard", "bayes_sd0.5", "bayes_sd1.0", "bayes_sd4.0", "bayes_sdNone", "plug_acc", "plug_em", "dta_hist"]); save(s6, "table6b_ablation_stats", T, index=False)
    q = pd.read_csv(os.path.join(R, "controlled", "ablation__quant.csv"))
    tq = q.groupby(["quantifier", "m"])["mae"].mean().unstack(); save(tq, "table6c_quantifier_mae", T)

    # ---- robustness --------------------------------------------------------------------------------------
    rows = {}
    ref2000 = lg[lg.m == 2000]
    def summ(df):
        d = df[df.rho != 1].groupby(["dataset", "method"])["value"].mean().unstack()
        return d.mean()
    rows["clean (m=2000)"] = summ(ref2000)
    for f in sorted(glob.glob(os.path.join(R, "controlled", "noise_*.csv")) + glob.glob(os.path.join(R, "controlled", "valfrac_*.csv")) + glob.glob(os.path.join(R, "controlled", "calib_platt.csv"))):
        if "__" in os.path.basename(f): continue
        rows[os.path.basename(f)[:-4]] = summ(pd.read_csv(f))
    t7 = pd.DataFrame(rows).reindex(ORDER).rename(index=NAMES); save(t7, "table7_robustness", T)
    gq = []
    for f in sorted(glob.glob(os.path.join(R, "controlled", "*__quant.csv"))):
        x = pd.read_csv(f); x = x[x.quantifier == "guard_fire_rate"]
        gq.append(dict(condition=os.path.basename(f).split("__")[0], guard_fire_rate=x.mae.mean()))
    save(pd.DataFrame(gq), "table7b_guard_fire_rates", T, index=False)
    rs = []
    for cnd in rows:
        if cnd.startswith("clean"): continue
        w = pd.read_csv(os.path.join(R, "controlled", cnd + ".csv")); w = w[w.rho != 1].groupby(["dataset", "method"])["value"].mean().unstack()
        x = paired_table(w, REF, ["static", "dta_em", "paft_guard"]); x.insert(0, "condition", cnd); rs.append(x)
    save(pd.concat(rs), "table7c_robustness_stats", T, index=False)

    # ---- other metrics -------------------------------------------------------------------------------------
    tm, sm_ = {}, []
    for f in sorted(glob.glob(os.path.join(R, "controlled", "metric_*.csv"))):
        if "__" in os.path.basename(f): continue
        x = pd.read_csv(f); name = os.path.basename(f)[7:-4]
        w = x[x.rho != 1].groupby(["dataset", "method"])["value"].mean().unstack()
        tm[name] = w.mean()
        oth = [c for c in ["static", "dta_em", "em_fixed"] if c in w.columns]
        y = paired_table(w, REF, oth); y.insert(0, "metric", name); sm_.append(y)
    t8 = pd.DataFrame(tm).reindex(ORDER).rename(index=NAMES); save(t8, "table8_other_metrics", T)
    save(pd.concat(sm_), "table8b_other_metrics_stats", T, index=False)

    # ---- real suites -----------------------------------------------------------------------------------------
    rp = os.path.join(R, "real", "real_batches.csv")
    if os.path.exists(rp):
        rb = pd.read_csv(rp)
        rl = rb[rb.model == mm]
        d = rl.groupby(["suite", "seed", "method"])["value"].mean().reset_index()
        t9 = d.groupby(["method", "suite"])["value"].mean().unstack().reindex(ORDER[:10] + ["paft_seq", "oracle_pi", "oracle_thr"]).rename(index=NAMES)
        t9sd = d.groupby(["method", "suite"])["value"].std().unstack().reindex(ORDER[:10] + ["paft_seq", "oracle_pi", "oracle_thr"]).rename(index=NAMES)
        save(t9, "table9_real_suites", T); save(t9sd, "table9_real_suites_sd", T)
        rs = []
        for su, x in rl.groupby("suite"):
            w = x.groupby(["batch", "method"])["value"].mean().unstack()         # unit = batch (mean over seeds)
            y = paired_table(w, REF, ["static", "dta_em", "em_fixed", "paft_guard", "paft_plug"]); y.insert(0, "suite", su); rs.append(y)
        save(pd.concat(rs), "table9b_real_stats", T, index=False)
        info = rl[rl.method == "static"].groupby("suite").agg(batches=("batch", "nunique"), m_min=("m", "min"), m_max=("m", "max"), pi0=("pi0", "mean"),
                                                             pi_min=("pi_true", "min"), pi_max=("pi_true", "max"), auc=("auc", "mean"),
                                                             mae_pi_mle=("pi_true", lambda s: np.nan)).drop(columns="mae_pi_mle")
        z = rl[rl.method == "static"].copy(); z["ae"] = (z.pi_mean - z.pi_true).abs(); z["fire"] = z.guard_p < 0.01
        info["MAE_pi_hat"] = z.groupby("suite")["ae"].mean(); info["guard_fire_rate"] = z.groupby("suite")["fire"].mean()
        save(info, "table9c_real_suite_info", T)
        d2 = rb.groupby(["model", "suite", "method"])["value"].mean().reset_index()
        t9m = d2[d2.method.isin(["static", "dta_em", "paft_bayes", "paft_guard"])].pivot_table(index=["suite", "model"], columns="method", values="value").rename(columns=NAMES)
        save(t9m, "tableS_real_by_model", T)
        S["real"] = {su: {NAMES.get(k, k): float(v) for k, v in t9[su].items()} for su in t9.columns}

    # ---- error analysis ----------------------------------------------------------------------------------------
    bp = os.path.join(R, "controlled", "main__batches.csv.gz")
    if os.path.exists(bp):
        b = pd.read_csv(bp); b = b[b.model == mm]
        b["gain_vs_static"] = b.v_paft_bayes - b.v_static
        b["gain_vs_dta"] = b.v_paft_bayes - b.v_dta_em
        b["abs_err"] = (b.pi_mean - b.pi_true).abs(); b["rel_err"] = b.abs_err / b.pi_true
        b["regret"] = b.v_oracle_thr - b.v_paft_bayes
        b["npos_bin"] = pd.cut(b.n_pos, [0, 5, 20, 100, 1000, 1e9], labels=["1-5", "6-20", "21-100", "101-1000", ">1000"])
        e1 = b.groupby("npos_bin", observed=True).agg(batches=("regret", "size"), F1_static=("v_static", "mean"), F1_paft=("v_paft_bayes", "mean"), F1_dta=("v_dta_em", "mean"),
                                                       loss_rate_vs_static=("gain_vs_static", lambda s: (s < 0).mean()), regret=("regret", "mean"), rel_err_pi=("rel_err", "median"))
        save(e1, "table10_error_by_positives", T)
        e2 = b.groupby("rho").agg(F1_static=("v_static", "mean"), F1_paft=("v_paft_bayes", "mean"), win=("gain_vs_static", lambda s: (s > 0).mean()), loss=("gain_vs_static", lambda s: (s < 0).mean()),
                                  worst_decile_gain=("gain_vs_static", lambda s: s.quantile(0.1)), regret=("regret", "mean"))
        save(e2, "table10b_error_by_shift", T)
        ds = b[b.rho != 1].groupby("dataset").agg(gain_vs_static=("gain_vs_static", "mean"), gain_vs_dta=("gain_vs_dta", "mean"), rel_err_pi=("rel_err", "median"), regret=("regret", "mean"))
        ds = ds.join(t1[["prevalence_%", "AUC_mean"]]).sort_values("gain_vs_static"); save(ds, "table10c_error_by_dataset", T)
        from scipy.stats import spearmanr
        S["error"] = dict(spearman_gain_auc=list(map(float, spearmanr(ds.AUC_mean, ds.gain_vs_static))), spearman_gain_prev=list(map(float, spearmanr(ds["prevalence_%"], ds.gain_vs_static))),
                          spearman_regret_relerr=list(map(float, spearmanr(b.rel_err, b.regret))))
        # threshold direction: how often does the adaptive threshold move the "right" way
        b["dir_ok"] = np.sign(b.t_bayes - b.t_static) == -np.sign(np.log(b.rho))   # raw-score threshold falls when prevalence rises
        S["error"]["threshold_moves_in_expected_direction_rate"] = float(b[b.rho != 1].dir_ok.mean())


    # ---- label-assisted variant (Experiment 9) -----------------------------------------------------------------------
    lp = os.path.join(R, "labelled", "labelled.csv")
    if os.path.exists(lp):
        L = pd.read_csv(lp)
        LN = {"static": "Static-F", "paft_bayes": "PAFT-Bayes (no labels)", "retune_labels": "Re-tune on labels", "paft_L": "PAFT-L (labels → π)",
              "paft_L_both": "PAFT-L + mixture", "paft_L_guarded": "PAFT-L guarded", "oracle_pi": "Oracle-π", "oracle_thr": "Oracle-threshold"}
        real = L[L.suite != "controlled"].groupby(["suite", "n_lab", "unit", "method"])["value"].mean().reset_index()
        ctl = L[(L.suite == "controlled") & (L.rho != 1)].groupby(["suite", "n_lab", "unit", "method"])["value"].mean().reset_index()
        allu = pd.concat([ctl, real])
        t12 = allu.groupby(["suite", "n_lab", "method"])["value"].mean().unstack()[list(LN)].rename(columns=LN); save(t12, "table12_labelled", T)
        rs = []
        for (su, nl), x in allu.groupby(["suite", "n_lab"]):
            w = x.pivot(index="unit", columns="method", values="value")
            y = paired_table(w, "paft_L_guarded", ["static", "retune_labels", "paft_bayes"]); y.insert(0, "n_lab", nl); y.insert(0, "suite", su); rs.append(y)
        save(pd.concat(rs), "table12b_labelled_stats", T, index=False)

    # ---- seed variability, guard calibration and guard-level sensitivity -----------------------------------------------
    sv = lg[lg.rho != 1].groupby(["seed", "dataset", "method"])["value"].mean().unstack().groupby("seed").mean()
    sv = pd.DataFrame({"mean": sv.mean(), "sd_over_seeds": sv.std(), "min": sv.min(), "max": sv.max()}).reindex(ORDER).rename(index=NAMES); save(sv, "tableS_seed_variability", T)
    gain_seed = lg[lg.rho != 1].groupby(["seed", "dataset", "method"])["value"].mean().unstack()
    gs_ = (gain_seed["paft_bayes"] - gain_seed["static"]).groupby("seed").mean()
    S["seed_gain_vs_static"] = dict(mean=float(gs_.mean()), sd=float(gs_.std()), values=[float(v) for v in gs_])
    gd = (gain_seed["paft_bayes"] - gain_seed["dta_em"]).groupby("seed").mean()
    S["seed_gain_vs_dta"] = dict(mean=float(gd.mean()), sd=float(gd.std()), values=[float(v) for v in gd])
    if os.path.exists(bp):
        b = pd.read_csv(bp); b = b[b.model == mm]
        rows = []
        for al in [0.001, 0.01, 0.05, 0.2]:
            v = np.where(b.guard_p >= al, b.v_paft_bayes, b.v_static)
            x = b.assign(v=v, fire=b.guard_p < al)
            rows.append(dict(alpha=al, controlled_F1_shifted=x[x.rho != 1].groupby("dataset")["v"].mean().mean(), controlled_F1_noshift=x[x.rho == 1].groupby("dataset")["v"].mean().mean(),
                             fire_rate_m500=x[x.m == 500].fire.mean(), fire_rate_m2000=x[x.m == 2000].fire.mean(), fire_rate_m10000=x[x.m == 10000].fire.mean()))
        ga = pd.DataFrame(rows).set_index("alpha")
        if os.path.exists(rp):
            w = rl.pivot_table(index=["suite", "seed", "batch"], columns="method", values="value"); gp = rl[rl.method == "static"].set_index(["suite", "seed", "batch"])["guard_p"]
            for al in ga.index:
                v = np.where(gp.reindex(w.index) >= al, w["paft_bayes"], w["static"])
                for su, val in pd.Series(v, index=w.index).groupby("suite").mean().items():
                    ga.loc[al, "real_" + su] = val
        save(ga, "table13_guard_alpha", T)
        fr = b.assign(fire=b.guard_p < 0.01).groupby("dataset")["fire"].mean().sort_values(); save(fr.to_frame("guard_fire_rate_under_pure_label_shift"), "tableS_guard_fire_by_dataset", T)

    # ---- efficiency ------------------------------------------------------------------------------------------------
    ep = os.path.join(R, "efficiency", "efficiency.csv")
    if os.path.exists(ep):
        e = pd.read_csv(ep); save(e, "table11_efficiency", T, index=False)
    json.dump(S, open(os.path.join(R, "statistics.json"), "w"), indent=1)
    # consolidated machine-readable outputs required by the manifest
    main_df.to_csv(os.path.join(R, "raw_results.csv"), index=False)
    json.dump({"controlled_shifted_mean_F1_lgbm": S["main"], "controlled_noshift_mean_F1_lgbm": S["main_noshift"], "real": S.get("real", {})}, open(os.path.join(R, "metrics.json"), "w"), indent=1)
    print("tables written to", T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="..")
    main(ap.parse_args().root)
