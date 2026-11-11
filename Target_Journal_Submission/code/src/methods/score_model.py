"""Validation-side score model shared by every decision rule.

Under prior-probability (label) shift the class-conditional score distributions
P(s | y) are invariant, so a held-out validation sample fully determines
  * the ROC operating points  TPR(t) = P(s >= t | y=1),  FPR(t) = P(s >= t | y=0)
  * the binned class-conditional likelihoods h1, h0 used for prevalence inference.
Any confusion-matrix metric at deployment prevalence pi is a closed-form function
of (pi, TPR(t), FPR(t)); see `metric_surface`.
"""
from __future__ import annotations
import numpy as np


def fbeta_from_rates(pi, tpr, fpr, beta=1.0):
    """F_beta as a function of prevalence and ROC point (Eq. 3 of the paper)."""
    b2 = beta * beta
    num = (1.0 + b2) * pi * tpr
    den = b2 * pi + pi * tpr + (1.0 - pi) * fpr
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def mcc_from_rates(pi, tpr, fpr):
    tp = pi * tpr
    fp = (1 - pi) * fpr
    fn = pi * (1 - tpr)
    tn = (1 - pi) * (1 - fpr)
    den = np.sqrt(np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    return np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-300), 0.0)


def metric_surface(pi, tpr, fpr, metric="f1"):
    if metric == "mcc":
        return mcc_from_rates(pi, tpr, fpr)
    beta = {"f1": 1.0, "f0.5": 0.5, "f2": 2.0}[metric]
    return fbeta_from_rates(pi, tpr, fpr, beta)


class ScoreModel:
    """Everything the decision layer learns from the labelled validation split."""

    def __init__(self, s_val, y_val, n_bins=None, max_thr=1024, laplace=0.5):
        s_val = np.asarray(s_val, dtype=float)
        y_val = np.asarray(y_val).astype(int)
        self.s1 = np.sort(s_val[y_val == 1])
        self.s0 = np.sort(s_val[y_val == 0])
        self.n1, self.n0 = len(self.s1), len(self.s0)
        if self.n1 < 2 or self.n0 < 2:
            raise ValueError("validation split needs >=2 samples per class")
        self.pi0 = self.n1 / (self.n1 + self.n0)
        # --- candidate thresholds: class-adaptive resolution -------------------
        q = np.linspace(0, 1, max_thr)
        cand = np.concatenate([
            self.s1 if self.n1 <= max_thr else np.quantile(self.s1, q),
            self.s0 if self.n0 <= max_thr else np.quantile(self.s0, q),
        ])
        self.thr = np.unique(cand)
        self.tpr = 1.0 - np.searchsorted(self.s1, self.thr, side="left") / self.n1
        self.fpr = 1.0 - np.searchsorted(self.s0, self.thr, side="left") / self.n0
        # --- binned class-conditional likelihoods ------------------------------
        if n_bins is None:
            n_bins = int(np.clip(self.n1 // 10, 8, 32))
        self.n_bins = n_bins
        # class-balanced quantile edges so that both tails are resolved
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.concatenate([np.quantile(self.s1, qs[::2]), np.quantile(self.s0, qs[1::2])]))
        self.edges = edges
        nb = len(edges) + 1
        c1 = np.bincount(np.searchsorted(edges, self.s1, side="right"), minlength=nb).astype(float)
        c0 = np.bincount(np.searchsorted(edges, self.s0, side="right"), minlength=nb).astype(float)
        self.h1 = (c1 + laplace) / (c1.sum() + laplace * nb)
        self.h0 = (c0 + laplace) / (c0.sum() + laplace * nb)

    # ---- helpers ---------------------------------------------------------------
    def bin_counts(self, s):
        return np.bincount(np.searchsorted(self.edges, s, side="right"), minlength=len(self.h1)).astype(float)

    def surface(self, pi, metric="f1"):
        """metric value for every candidate threshold; pi scalar -> [T], array [G] -> [G, T]."""
        pi = np.asarray(pi, dtype=float)
        if pi.ndim == 0:
            return metric_surface(pi, self.tpr, self.fpr, metric)
        return metric_surface(pi[:, None], self.tpr[None, :], self.fpr[None, :], metric)

    def best_threshold(self, pi, metric="f1"):
        return float(self.thr[int(np.argmax(self.surface(pi, metric)))])

    def hist_posterior(self, s, pi=None):
        """Histogram-binning calibrated P(y=1|s) at prevalence pi (default: validation)."""
        pi = self.pi0 if pi is None else pi
        b = np.searchsorted(self.edges, s, side="right")
        a1, a0 = pi * self.h1[b], (1 - pi) * self.h0[b]
        return a1 / (a1 + a0)
