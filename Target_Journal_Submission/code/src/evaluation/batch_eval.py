"""Evaluate decision rules on a labelled deployment batch."""
from __future__ import annotations
import numpy as np


def _metric_from_counts(tp, k, npos, m, metric):
    tp = np.asarray(tp, float); k = np.asarray(k, float)
    fp, fn = k - tp, npos - tp
    tn = m - k - fn
    if metric == "mcc":
        den = np.sqrt(np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
        return np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-300), 0.0)
    b2 = {"f1": 1.0, "f0.5": 0.25, "f2": 4.0}[metric]
    den = (1 + b2) * tp + b2 * fn + fp
    return np.where(den > 0, (1 + b2) * tp / np.maximum(den, 1e-300), 0.0)


class Batch:
    """Sorted view of a labelled batch: any rule reduces to 'predict the top-k scores'."""

    def __init__(self, s, y):
        o = np.argsort(-s, kind="stable")
        self.s = s[o]; self.y = y[o].astype(int)
        self.m, self.npos = len(s), int(self.y.sum())
        self.ctp = np.concatenate([[0], np.cumsum(self.y)])

    def k_of_threshold(self, t):
        return int(np.searchsorted(-self.s, -t, side="right"))    # #(s >= t)

    def score(self, k, metric="f1"):
        k = int(np.clip(k, 0, self.m))
        tp = self.ctp[k]
        val = float(_metric_from_counts(tp, k, self.npos, self.m, metric))
        prec = tp / k if k > 0 else 0.0
        rec = tp / self.npos if self.npos > 0 else 0.0
        return val, prec, rec

    def oracle(self, metric="f1"):
        """Best achievable value over all thresholds on this batch (ties in score respected)."""
        ks = np.concatenate([np.where(np.diff(self.s) != 0)[0] + 1, [self.m]])
        vals = _metric_from_counts(self.ctp[ks], ks, self.npos, self.m, metric)
        return float(np.max(vals))


def evaluate_rules(decisions, batch: Batch, metric="f1"):
    res = {}
    for name, (kind, v) in decisions.items():
        k = batch.k_of_threshold(v) if kind == "t" else v
        res[name] = batch.score(k, metric) + (k,)
    return res
