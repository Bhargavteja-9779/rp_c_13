"""Prevalence estimators (quantifiers) operating on an unlabelled batch of scores."""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logit, logsumexp
from .score_model import ScoreModel


def pi_grid(pi0, n=161, half_width=9.0):
    """Grid that is uniform on the logit scale, centred on the validation prevalence."""
    z = np.clip(logit(pi0) + np.linspace(-half_width, half_width, n), -13.8, 13.8)
    return expit(z)


def binned_log_lik(sm: ScoreModel, s, grid):
    """log L(pi) = sum_b n_b log(pi h1_b + (1-pi) h0_b)  (mixture likelihood, Eq. 6)."""
    nb = sm.bin_counts(s)
    mix = grid[:, None] * sm.h1[None, :] + (1 - grid[:, None]) * sm.h0[None, :]
    return (np.log(mix) * nb[None, :]).sum(1)


def posterior(sm: ScoreModel, s, grid, prior_sd=2.0, prior_center=None, temper=1.0):
    """Grid posterior over pi with a Normal(logit pi_c, prior_sd^2) prior on logit(pi).

    prior_sd=None gives a flat prior on the logit scale. Returns (weights, loglik)."""
    ll = binned_log_lik(sm, s, grid) * temper
    lp = ll.copy()
    if prior_sd is not None:
        c = logit(sm.pi0 if prior_center is None else prior_center)
        lp = lp - 0.5 * ((logit(grid) - c) / prior_sd) ** 2
    w = np.exp(lp - logsumexp(lp))
    return w, ll


def q_mle(sm, s, grid):
    return float(grid[int(np.argmax(binned_log_lik(sm, s, grid)))])


def q_cc(sm, s, t):
    return float(np.mean(s >= t))


def q_acc(sm, s, t):
    i = int(np.searchsorted(sm.thr, t))
    i = min(i, len(sm.thr) - 1)
    tpr, fpr = sm.tpr[i], sm.fpr[i]
    if tpr - fpr < 1e-6:
        return q_cc(sm, s, t)
    return float(np.clip((np.mean(s >= t) - fpr) / (tpr - fpr), 0.0, 1.0))


def q_pcc(p):
    return float(np.mean(p))


def q_pacc(p, p_val1_mean, p_val0_mean):
    d = p_val1_mean - p_val0_mean
    if d < 1e-6:
        return q_pcc(p)
    return float(np.clip((np.mean(p) - p_val0_mean) / d, 0.0, 1.0))


def q_em(p, pi0, max_iter=200, tol=1e-7):
    """Saerens-Latinne-Decaestecker EM on calibrated posteriors p (trained at prior pi0).
    Returns (pi_hat, corrected posteriors)."""
    pi = pi0
    r1, r0 = p / pi0, (1 - p) / (1 - pi0)
    for _ in range(max_iter):
        a1, a0 = pi * r1, (1 - pi) * r0
        q = a1 / (a1 + a0)
        new = float(np.mean(q))
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    pi = float(np.clip(pi, 1e-6, 1 - 1e-6))
    a1, a0 = pi * r1, (1 - pi) * r0
    return pi, a1 / (a1 + a0)


def q_hdy(sm: ScoreModel, s, grid):
    """Hellinger-distance mixture matching on the same bins (Gonzalez-Castro et al.)."""
    nb = sm.bin_counts(s)
    u = np.sqrt(nb / nb.sum())
    mix = np.sqrt(grid[:, None] * sm.h1[None, :] + (1 - grid[:, None]) * sm.h0[None, :])
    return float(grid[int(np.argmax((mix * u[None, :]).sum(1)))])
