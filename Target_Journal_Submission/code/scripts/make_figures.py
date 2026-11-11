"""Publication figures, generated only from machine-written result files."""
from __future__ import annotations
import os, sys, json, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.methods.score_model import ScoreModel
from src.methods.rules import DecisionLayer
from src.methods import quantify as Q

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "legend.frameon": False})
# colour-blind-safe palette (Okabe-Ito)
COL = {"static": "#555555", "em_fixed": "#E69F00", "dta_em": "#0072B2", "paft_bayes": "#D55E00", "paft_plug": "#CC79A7", "paft_guard": "#009E73",
       "oracle_pi": "#999999", "oracle_thr": "#000000", "default_0.5": "#BBBBBB", "prev_match": "#56B4E9", "em_0.5": "#F0E442", "static_bag": "#888888"}
LAB = {"default_0.5": "Default 0.5", "static": "Static-F", "static_bag": "Static-Bag", "em_0.5": "EM-0.5", "em_fixed": "EM-Fixed", "prev_match": "PrevMatch",
       "dta_em": "DTA-EM", "paft_plug": "PAFT-Plug", "paft_bayes": "PAFT-Bayes", "paft_guard": "PAFT-Guard", "oracle_pi": "Oracle-π", "oracle_thr": "Oracle-threshold"}
MK = {"static": "s", "em_fixed": "v", "dta_em": "D", "paft_bayes": "o", "paft_guard": "^", "oracle_pi": "x", "paft_plug": "P"}


def fig_workflow(F):
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=c, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
    def arr(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
    ax.text(1, 37.5, "Development (labels available, once)", fontsize=8, weight="bold")
    ax.text(1, 16.5, "Deployment (no labels, every batch)", fontsize=8, weight="bold")
    box(1, 24, 17, 10, "Trained scorer s(x)\n(any model)", "#EEEEEE")
    box(23, 24, 20, 10, "Validation scores\n→ TPR(t), FPR(t)\n→ histograms h₁, h₀", "#DDEBF7")
    box(48, 24, 22, 10, "Metric surface\nM(t; π) on a\nthreshold × prevalence grid", "#DDEBF7")
    box(75, 24, 23, 10, "Static threshold\nt₀ = argmax M(t; π₀)", "#EEEEEE")
    arr((18.5, 29), (22.5, 29)); arr((43.5, 29), (47.5, 29)); arr((70.5, 29), (74.5, 29))
    box(1, 3, 17, 10, "Unlabelled batch\nscores s₁…s_m", "#EEEEEE")
    box(23, 3, 20, 10, "Mixture likelihood\n+ logit-normal prior\n→ posterior p(π | batch)", "#FCE4D6")
    box(48, 3, 22, 10, "Bayes threshold\nt̂ = argmax E[M(t; π)]", "#FCE4D6")
    box(75, 3, 23, 10, "Validity guard (KS)\npass → t̂   fail → t₀", "#E2EFDA")
    arr((18.5, 8), (22.5, 8)); arr((43.5, 8), (47.5, 8)); arr((70.5, 8), (74.5, 8))
    arr((33, 23.5), (33, 13.5)); arr((59, 23.5), (59, 13.5)); arr((86.5, 23.5), (86.5, 13.5))
    fig.savefig(os.path.join(F, "fig1_workflow.png")); plt.close(fig)


def fig_mechanism(root, F):
    z = np.load(os.path.join(root, "scores", "controlled", "bank_marketing__lgbm__s0.npz"))
    sm = ScoreModel(z["s_val"], z["y_val"])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    pis = [sm.pi0 / 4, sm.pi0, min(4 * sm.pi0, 0.6)]
    for pi, c in zip(pis, ["#0072B2", "#555555", "#D55E00"]):
        f = sm.surface(pi); i = int(np.argmax(f))
        ax[0].plot(sm.thr, f, color=c, lw=1.2, label=f"π = {pi:.3f}"); ax[0].plot(sm.thr[i], f[i], "o", color=c, ms=4)
    t0 = sm.best_threshold(sm.pi0); ax[0].axvline(t0, color="#555555", ls=":", lw=0.8)
    ax[0].set_xlabel("threshold t on the score"); ax[0].set_ylabel("F1(t; π)"); ax[0].legend(fontsize=7); ax[0].set_title("(a) F1 surface slices", fontsize=8)
    g = Q.pi_grid(sm.pi0, 161); S = sm.surface(g)
    best = S.max(1); stat = S[:, int(np.argmax(sm.surface(sm.pi0)))]
    ax[1].plot(g, sm.thr[S.argmax(1)], color="#D55E00", lw=1.2); ax[1].axhline(t0, color="#555555", ls=":", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_xlim(sm.pi0 / 20, 0.9); ax[1].set_xlabel("deployment prevalence π"); ax[1].set_ylabel("F1-optimal threshold t*(π)"); ax[1].set_title("(b) optimal threshold moves", fontsize=8)
    ax[2].plot(g, best - stat, color="#D55E00", lw=1.2); ax[2].axvline(sm.pi0, color="#555555", ls=":", lw=0.8)
    ax[2].set_xscale("log"); ax[2].set_xlim(sm.pi0 / 20, 0.9); ax[2].set_xlabel("deployment prevalence π"); ax[2].set_ylabel("F1 regret of static threshold"); ax[2].set_title("(c) cost of not adapting", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_mechanism.png")); plt.close(fig)


def fig_main(R, F):
    d = pd.read_csv(os.path.join(R, "controlled", "main.csv")); d = d[d.model == "lgbm"]
    g = d.groupby(["dataset", "rho", "method"])["value"].mean().reset_index()
    piv = g.groupby(["rho", "method"])["value"].mean().unstack()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for m in ["static", "em_fixed", "dta_em", "paft_bayes", "oracle_pi"]:
        ax[0].plot(piv.index, piv[m] - piv["static"], marker=MK[m], ms=4, lw=1.1, color=COL[m], label=LAB[m])
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(piv.index); ax[0].set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax[0].set_xlabel("odds multiplier ρ (deployment vs. development)"); ax[0].set_ylabel("ΔF1 vs. Static-F"); ax[0].legend(fontsize=7); ax[0].set_title("(a) mean over 18 datasets", fontsize=8)
    w = g[g.rho != 1].groupby(["dataset", "method"])["value"].mean().unstack()
    diff = (w["paft_bayes"] - w["static"]).sort_values()
    ax[1].barh(range(len(diff)), diff.values, color=["#D55E00" if v > 0 else "#0072B2" for v in diff.values])
    ax[1].set_yticks(range(len(diff))); ax[1].set_yticklabels(diff.index, fontsize=6.5); ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_xlabel("ΔF1, PAFT-Bayes − Static-F (ρ ≠ 1)"); ax[1].set_title("(b) per dataset", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_main.png")); plt.close(fig)


def fig_ranks(root, F):
    r = pd.read_csv(os.path.join(root, "tables", "table3c_ranks.csv"), index_col=0)["average rank"].sort_values()
    S = json.load(open(os.path.join(root, "results", "statistics.json")))["friedman"]
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.barh(range(len(r)), r.values, color=["#D55E00" if "PAFT" in i else "#777777" for i in r.index]); ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index, fontsize=7.5); ax.invert_yaxis()
    ax.plot([r.values[0], r.values[0] + S["nemenyi_cd"]], [-0.8, -0.8], color="k", lw=1.5); ax.text(r.values[0] + S["nemenyi_cd"] / 2, -1.0, f"Nemenyi CD = {S['nemenyi_cd']:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel(f"average rank over {S['n']} datasets (lower is better)"); ax.set_ylim(len(r) - 0.4, -1.8)
    fig.savefig(os.path.join(F, "fig4_ranks.png")); plt.close(fig)


def fig_ablation(R, F):
    d = pd.read_csv(os.path.join(R, "controlled", "ablation.csv"))
    g = d[d.rho != 1].groupby(["m", "dataset", "method"])["value"].mean().reset_index().groupby(["m", "method"])["value"].mean().unstack()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for m in ["static", "dta_em", "paft_plug", "paft_bayes", "oracle_pi"]:
        ax[0].plot(g.index, g[m], marker=MK[m], ms=4, lw=1.1, color=COL[m], label=LAB[m])
    ax[0].set_xscale("log"); ax[0].set_xlabel("batch size m"); ax[0].set_ylabel("mean F1 (ρ ≠ 1)"); ax[0].legend(fontsize=7); ax[0].set_title("(a) batch size", fontsize=8)
    sd = ["bayes_sd0.5", "bayes_sd1.0", "paft_bayes", "bayes_sd4.0", "bayes_sdNone"]; xs = ["0.5", "1", "2", "4", "flat"]
    gs = d.groupby(["m", "dataset", "rho", "method"])["value"].mean().reset_index()
    for m_, c in zip(sorted(d.m.unique()), ["#0072B2", "#56B4E9", "#E69F00", "#D55E00"]):
        x = gs[(gs.m == m_) & (gs.rho != 1)].groupby("method")["value"].mean()
        ax[1].plot(xs, [x[s] - x["static"] for s in sd], marker="o", ms=3.5, lw=1.1, color=c, label=f"m = {m_}")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("prior s.d. σ on logit π"); ax[1].set_ylabel("ΔF1 vs. Static-F (ρ ≠ 1)"); ax[1].legend(fontsize=7); ax[1].set_title("(b) prior strength", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig5_ablation.png")); plt.close(fig)


def fig_robust(root, F):
    t = pd.read_csv(os.path.join(root, "tables", "table7_robustness.csv"), index_col=0)
    cols = ["clean (m=2000)"] + [c for c in t.columns if c.startswith("noise")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = [0] + [float(c.split("_")[1]) for c in cols[1:]]
    for k in ["static", "em_fixed", "dta_em", "paft_bayes", "paft_guard"]:
        ax[0].plot(x, t.loc[LAB[k], cols].values, marker=MK[k], ms=4, lw=1.1, color=COL[k], label=LAB[k])
    ax[0].set_xlabel("feature-noise level (× feature s.d.)"); ax[0].set_ylabel("mean F1 (ρ ≠ 1, m = 2000)"); ax[0].legend(fontsize=7); ax[0].set_title("(a) violated label-shift assumption", fontsize=8)
    cols = [c for c in t.columns if c.startswith("valfrac")][::1] + ["clean (m=2000)"]
    x = [float(c.split("_")[1]) for c in cols[:-1]] + [1.0]
    for k in ["static", "dta_em", "paft_bayes"]:
        ax[1].plot(x, t.loc[LAB[k], cols].values, marker=MK[k], ms=4, lw=1.1, color=COL[k], label=LAB[k])
    ax[1].set_xscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([str(v) for v in x]); ax[1].set_xlabel("fraction of validation split used"); ax[1].set_ylabel("mean F1 (ρ ≠ 1, m = 2000)"); ax[1].legend(fontsize=7); ax[1].set_title("(b) small validation sets", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig6_robustness.png")); plt.close(fig)


def fig_real(R, F):
    p = os.path.join(R, "real", "real_batches.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p); d = d[d.model == "lgbm"]
    suites = list(d.suite.unique())
    fig, axes = plt.subplots(2, len(suites), figsize=(7.2, 3.6), sharex="col")
    for j, su in enumerate(suites):
        x = d[d.suite == su]
        order = list(dict.fromkeys(x.batch)) if su != "acs_income_states" else list(x[x.method == "static"].groupby("batch")["pi_true"].mean().sort_values().index)
        w = x.groupby(["batch", "method"])["value"].mean().unstack().reindex(order)
        pt = x[x.method == "static"].groupby("batch")[["pi_true", "pi_mean"]].mean().reindex(order)
        a = axes[0, j]; a.plot(range(len(order)), pt.pi_true, color="k", lw=1, label="true π"); a.plot(range(len(order)), pt.pi_mean, color="#D55E00", lw=1, ls="--", label="estimated π")
        a.axhline(x.pi0.mean(), color="#777777", ls=":", lw=0.8); a.set_title(su.replace("_", " "), fontsize=7.5); a.tick_params(labelsize=6)
        b = axes[1, j]
        for k in ["static", "dta_em", "paft_bayes", "paft_guard"]:
            b.plot(range(len(order)), w[k], color=COL[k], lw=1, label=LAB[k])
        b.tick_params(labelsize=6); b.set_xlabel("batch (time)" if su != "acs_income_states" else "state (by π)", fontsize=7)
    axes[0, 0].set_ylabel("prevalence", fontsize=7.5); axes[1, 0].set_ylabel("F1", fontsize=7.5)
    axes[0, 0].legend(fontsize=6); axes[1, 0].legend(fontsize=5.5)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig7_real.png")); plt.close(fig)


def fig_error(R, F):
    p = os.path.join(R, "controlled", "main__batches.csv.gz")
    if not os.path.exists(p): return
    b = pd.read_csv(p); b = b[(b.rho != 1) & (b.model == "lgbm")]
    b["gain"] = b.v_paft_bayes - b.v_static; b["rel"] = (b.pi_mean - b.pi_true).abs() / b.pi_true; b["regret"] = b.v_oracle_thr - b.v_paft_bayes
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax[0].hist(b.gain, bins=80, color="#D55E00"); ax[0].axvline(0, color="k", lw=0.6); ax[0].set_yscale("log"); ax[0].set_xlabel("per-batch ΔF1 (PAFT-Bayes − Static-F)"); ax[0].set_ylabel("batches"); ax[0].set_title("(a) gain distribution", fontsize=8)
    q = pd.qcut(b.rel, 10, duplicates="drop"); g = b.groupby(q, observed=True).agg(x=("rel", "median"), y=("regret", "mean"))
    ax[1].plot(g.x, g.y, "o-", color="#0072B2", ms=3.5, lw=1); ax[1].set_xscale("log"); ax[1].set_xlabel("relative prevalence error |π̂−π|/π"); ax[1].set_ylabel("mean regret vs. oracle threshold"); ax[1].set_title("(b) quantification error drives regret", fontsize=8)
    bins = [0, 5, 20, 100, 1000, 1e9]; lab = ["1–5", "6–20", "21–100", "101–1000", ">1000"]
    g = b.groupby(pd.cut(b.n_pos, bins, labels=lab), observed=True).agg(loss=("gain", lambda s: (s < 0).mean()), win=("gain", lambda s: (s > 0).mean()))
    ax[2].bar(np.arange(len(g)) - 0.2, g.win, 0.4, color="#D55E00", label="better than static"); ax[2].bar(np.arange(len(g)) + 0.2, g.loss, 0.4, color="#0072B2", label="worse than static")
    ax[2].set_xticks(range(len(g))); ax[2].set_xticklabels(g.index, fontsize=6.5); ax[2].set_xlabel("positives in the batch"); ax[2].set_ylabel("fraction of batches"); ax[2].legend(fontsize=6.5); ax[2].set_title("(c) win / loss rate", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig8_error.png")); plt.close(fig)


def fig_efficiency(R, F):
    p = os.path.join(R, "efficiency", "efficiency.csv")
    if not os.path.exists(p): return
    e = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for k, c in [("static threshold (comparison only)", "#555555"), ("EM prior correction (SLD)", "#E69F00"), ("DTA-EM (EM + sort + expected-F scan)", "#0072B2"), ("PAFT-Bayes", "#D55E00"), ("PAFT guard (KS test)", "#009E73")]:
        x = e[e.component == k]; ax.plot(x.m, x.ms, "o-", ms=3.5, lw=1.1, color=c, label=k.split(" (")[0])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("batch size m"); ax.set_ylabel("latency per batch (ms, 1 CPU thread)"); ax.legend(fontsize=6.5)
    fig.savefig(os.path.join(F, "fig9_efficiency.png")); plt.close(fig)


def main(root):
    R, F = os.path.join(root, "results"), os.path.join(root, "figures"); os.makedirs(F, exist_ok=True)
    fig_workflow(F); fig_mechanism(root, F); fig_main(R, F); fig_ranks(root, F); fig_ablation(R, F); fig_robust(root, F); fig_real(R, F); fig_error(R, F); fig_efficiency(R, F)
    print("figures written to", F)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="..")
    main(ap.parse_args().root)
