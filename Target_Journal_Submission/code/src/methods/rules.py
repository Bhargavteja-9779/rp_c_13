"""Decision rules. Every rule maps an unlabelled batch of scores to a threshold
(predict positive iff score >= threshold) or to a number k of top-scored instances.

Proposed: PAFT (prevalence-adaptive F-measure thresholding)
  paft_plug  : plug-in   t*(pi_hat),        pi_hat = mixture-likelihood MLE
  paft_bayes : Bayes     argmax_t E_{pi|batch}[ M(t; pi) ]
  paft_guard : paft_bayes with a label-shift validity guard (falls back to static)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import kstwobign
from .score_model import ScoreModel
from . import quantify as Q


class Calibrator:
    """Monotone probability calibrator fitted on validation scores ('isotonic' | 'platt')."""

    def __init__(self, s_val, y_val, kind="isotonic"):
        self.kind = kind
        eps = 1e-6
        if kind == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.m = IsotonicRegression(y_min=eps, y_max=1 - eps, out_of_bounds="clip").fit(s_val, y_val)
        else:
            from sklearn.linear_model import LogisticRegression
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(s_val), y_val)

    @staticmethod
    def _z(s):
        s = np.clip(np.asarray(s, float), 1e-7, 1 - 1e-7)
        return np.log(s / (1 - s)).reshape(-1, 1)

    def __call__(self, s):
        if self.kind == "isotonic":
            return self.m.predict(s)
        return np.clip(self.m.predict_proba(self._z(s))[:, 1], 1e-6, 1 - 1e-6)


def mixture_ks_pvalue(sm: ScoreModel, s_sorted, pi):
    """Goodness of fit of the label-shift mixture pi*F1 + (1-pi)*F0 to the batch scores.

    Kolmogorov-Smirnov distance between the batch ECDF and the fitted mixture CDF, with
    the two-sample effective size m*n_eff/(m+n_eff), n_eff = 1/(pi^2/n1 + (1-pi)^2/n0).
    Conservative because pi is fitted on the batch."""
    m = len(s_sorted)
    f1 = np.searchsorted(sm.s1, s_sorted, side="right") / sm.n1
    f0 = np.searchsorted(sm.s0, s_sorted, side="right") / sm.n0
    mix = pi * f1 + (1 - pi) * f0
    hi = np.arange(1, m + 1) / m
    lo = np.arange(0, m) / m
    d = max(np.max(np.abs(hi - mix)), np.max(np.abs(lo - mix)))
    n_eff = 1.0 / (pi ** 2 / sm.n1 + (1 - pi) ** 2 / sm.n0)
    en = np.sqrt(m * n_eff / (m + n_eff))
    return float(kstwobign.sf((en + 0.12 + 0.11 / en) * d)), float(d)


def dta_topk(q_sorted_desc, beta=1.0):
    """Decision-theoretic rule: k maximising the plug-in expected F_beta
    (1+b^2) sum_{i<=k} q_i / (b^2 sum_i q_i + k) on (prior-corrected) posteriors."""
    b2 = beta * beta
    cs = np.cumsum(q_sorted_desc)
    k = np.arange(1, len(cs) + 1)
    return int(np.argmax((1 + b2) * cs / (b2 * cs[-1] + k)) + 1)


class DecisionLayer:
    """Fits all validation-side objects once; `decide` evaluates every rule on a batch."""

    def __init__(self, s_val, y_val, metric="f1", prior_sd=2.0, guard_alpha=0.01,
                 n_bins=None, n_grid=161, calibrator="isotonic", n_bag=50, seed=0):
        self.metric, self.prior_sd, self.guard_alpha = metric, prior_sd, guard_alpha
        self.sm = sm = ScoreModel(s_val, y_val, n_bins=n_bins)
        self.grid = Q.pi_grid(sm.pi0, n_grid)
        self.surf = sm.surface(self.grid, metric)          # [G, T], reused for every batch
        self.t_static = sm.best_threshold(sm.pi0, metric)
        self.cal = Calibrator(s_val, y_val, calibrator)
        p_val = self.cal(np.asarray(s_val, float))
        y_val = np.asarray(y_val).astype(int)
        self.pv1, self.pv0 = float(p_val[y_val == 1].mean()), float(p_val[y_val == 0].mean())
        self.tau_static = float(self.cal(np.array([self.t_static]))[0])   # static threshold in posterior space
        # bagged static threshold (robust static baseline, in the spirit of GHOST)
        rng = np.random.default_rng(seed)
        ts = []
        s_val = np.asarray(s_val, float)
        i1, i0 = np.where(y_val == 1)[0], np.where(y_val == 0)[0]
        for _ in range(n_bag):
            idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
            ts.append(ScoreModel(s_val[idx], y_val[idx], n_bins=8, max_thr=512).best_threshold(sm.pi0, metric))
        self.t_bag = float(np.median(ts))
        self.beta = {"f1": 1.0, "f0.5": 0.5, "f2": 2.0}.get(metric, None)

    # ------------------------------------------------------------------------
    def paft(self, s, prior_sd="default", prior_center=None, temper=1.0):
        """Returns dict with plug-in / Bayes thresholds and posterior summaries."""
        sm = self.sm
        sd = self.prior_sd if prior_sd == "default" else prior_sd
        w, ll = Q.posterior(sm, s, self.grid, sd, prior_center, temper)
        pi_mle = float(self.grid[int(np.argmax(ll))])
        pi_mean = float((w * self.grid).sum())
        t_plug = float(sm.thr[int(np.argmax(self.surf[int(np.argmax(ll))]))])
        t_bayes = float(sm.thr[int(np.argmax(w @ self.surf))])
        return dict(pi_mle=pi_mle, pi_mean=pi_mean, t_plug=t_plug, t_bayes=t_bayes, w=w)

    def decide(self, s, extra=False):
        """s: unlabelled batch scores. Returns {rule: ('t', thr) | ('k', k)}, info dict."""
        sm = self.sm
        s = np.asarray(s, float)
        m = len(s)
        out, info = {}, {}
        out["default_0.5"] = ("t", float(np.min(sm.thr[self.cal(sm.thr) >= 0.5], initial=np.inf)))
        out["static"] = ("t", self.t_static)
        out["static_bag"] = ("t", self.t_bag)
        # --- prior-correction baselines on calibrated posteriors -------------
        p = self.cal(s)
        pi_em, q = Q.q_em(p, sm.pi0)
        order = np.argsort(-q, kind="stable")
        out["em_0.5"] = ("k", int(np.sum(q >= 0.5)))
        out["em_fixed"] = ("k", int(np.sum(q >= self.tau_static)))
        pi_acc = Q.q_acc(sm, s, self.t_static)
        out["prev_match"] = ("k", int(round(pi_acc * m)))
        if self.beta is not None:
            out["dta_em"] = ("k", dta_topk(q[order], self.beta))
        # --- proposed ---------------------------------------------------------
        r = self.paft(s)
        out["paft_plug"] = ("t", r["t_plug"])
        out["paft_bayes"] = ("t", r["t_bayes"])
        pval, d = mixture_ks_pvalue(sm, np.sort(s), r["pi_mle"])
        out["paft_guard"] = ("t", r["t_bayes"] if pval >= self.guard_alpha else self.t_static)
        info.update(pi_em=pi_em, pi_acc=pi_acc, pi_mle=r["pi_mle"], pi_mean=r["pi_mean"],
                    guard_p=pval, guard_d=d, t_static=self.t_static, t_bayes=r["t_bayes"])
        if extra:   # quantifier / prior ablations
            info["pi_cc"] = Q.q_cc(sm, s, self.t_static)
            info["pi_pcc"] = Q.q_pcc(p)
            info["pi_pacc"] = Q.q_pacc(p, self.pv1, self.pv0)
            info["pi_hdy"] = Q.q_hdy(sm, s, self.grid)
            for name, pi in [("acc", pi_acc), ("pacc", info["pi_pacc"]), ("em", pi_em), ("hdy", info["pi_hdy"])]:
                out[f"plug_{name}"] = ("t", sm.best_threshold(float(np.clip(pi, 1e-6, 1 - 1e-6)), self.metric))
            for sd in [0.5, 1.0, 4.0, None]:
                out[f"bayes_sd{sd}"] = ("t", self.paft(s, prior_sd=sd)["t_bayes"])
            # EM-corrected hist posterior + DTA (same likelihood as PAFT, transductive decision)
            if self.beta is not None:
                qh = sm.hist_posterior(s, r["pi_mle"])
                out["dta_hist"] = ("k", dta_topk(np.sort(qh)[::-1], self.beta))
        return out, info

    def oracle_pi_threshold(self, pi_true):
        return self.sm.best_threshold(float(np.clip(pi_true, 1e-6, 1 - 1e-6)), self.metric)


# ----------------------------------------------------------------------------------------------
# Label-assisted variant (added after the real-shift analysis, see manuscript Section 4.5)
# ----------------------------------------------------------------------------------------------
def paft_labelled(dl: "DecisionLayer", s_unl, k_pos, n_lab, mode="labels"):
    """Threshold from a posterior over pi that uses a small labelled audit sample.

    mode = 'labels' : prior x Binomial(k_pos | n_lab, pi)                    (robust to any shift in P(x|y) for *quantification*)
           'both'   : prior x Binomial x mixture likelihood of unlabelled scores
           'guarded': 'both' if the label-shift guard passes on the unlabelled scores, else 'labels'
    Returns (threshold, posterior mean of pi)."""
    from scipy.special import logit, logsumexp
    g = dl.grid
    lp = k_pos * np.log(g) + (n_lab - k_pos) * np.log1p(-g)
    if dl.prior_sd is not None:
        lp = lp - 0.5 * ((logit(g) - logit(dl.sm.pi0)) / dl.prior_sd) ** 2
    use_mix = mode == "both"
    if mode == "guarded":
        pi_mle = Q.q_mle(dl.sm, s_unl, g)
        use_mix = mixture_ks_pvalue(dl.sm, np.sort(s_unl), pi_mle)[0] >= dl.guard_alpha
    if use_mix:
        lp = lp + Q.binned_log_lik(dl.sm, s_unl, g)
    w = np.exp(lp - logsumexp(lp))
    return float(dl.sm.thr[int(np.argmax(w @ dl.surf))]), float((w * g).sum())


def retune_on_labels(s_lab, y_lab, fallback, metric="f1"):
    """Natural label-using baseline: pick the threshold that maximises the metric on the labelled audit sample."""
    from ..evaluation.batch_eval import Batch, _metric_from_counts
    if y_lab.sum() == 0 or y_lab.sum() == len(y_lab):
        return fallback
    b = Batch(np.asarray(s_lab, float), np.asarray(y_lab))
    ks = np.arange(1, b.m + 1)
    v = _metric_from_counts(b.ctp[ks], ks, b.npos, b.m, metric)
    k = int(ks[int(np.argmax(v))])
    return float(b.s[k - 1])
