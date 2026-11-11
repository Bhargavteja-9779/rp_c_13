"""Statistical validation. Unit of analysis = dataset (controlled suite) or deployment batch
(real suites): per-unit values are first averaged over seeds / repetitions / cells, so the
paired samples are independent across units."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (Kerby 2014): (W+ - W-)/(W+ + W-)."""
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def boot_ci(d, stat=np.mean, n=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    b = stat(d[rng.integers(0, len(d), (n, len(d)))], axis=1)
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def paired_table(wide: pd.DataFrame, ref: str, others):
    """wide: units x methods. Two-sided Wilcoxon signed-rank test of ref vs each other method,
    Holm-corrected over the family `others`."""
    rows = []
    for o in others:
        d = (wide[ref] - wide[o]).to_numpy()
        try:
            p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        lo, hi = boot_ci(d)
        rows.append(dict(reference=ref, baseline=o, n_units=len(d), mean_diff=d.mean(), median_diff=np.median(d), ci95_lo=lo, ci95_hi=hi,
                         wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()), p_wilcoxon=p, rank_biserial=rank_biserial(d)))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t["p_wilcoxon"]); return t


def friedman_ranks(wide: pd.DataFrame):
    r = wide.rank(axis=1, ascending=False).mean().sort_values()
    st = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    k, n = wide.shape[1], wide.shape[0]
    # Nemenyi critical difference at alpha = 0.05 (studentised-range based)
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return r, float(st.statistic), float(st.pvalue), float(cd)
