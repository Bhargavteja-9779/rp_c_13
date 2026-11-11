"""Artificial-prevalence protocol on cached scores (controlled label shift) and the
generic batch loop shared with the real-shift suites."""
from __future__ import annotations
import json, zlib
import numpy as np
from ..methods.rules import DecisionLayer
from .batch_eval import Batch, evaluate_rules

QUANT = ["pi_acc", "pi_em", "pi_mle", "pi_mean", "pi_cc", "pi_pcc", "pi_pacc", "pi_hdy"]


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def eval_batch(dl: DecisionLayer, s, y, metric, extra=False):
    """All rules + oracles on one labelled batch. Returns {rule: (value, prec, rec, k)}, info."""
    b = Batch(np.asarray(s, float), np.asarray(y))
    dec, info = dl.decide(b.s, extra=extra)
    pi_true = b.npos / b.m
    dec["oracle_pi"] = ("t", dl.oracle_pi_threshold(pi_true))
    res = evaluate_rules(dec, b, metric)
    orc = b.oracle(metric)
    res["oracle_thr"] = (orc, np.nan, np.nan, -1)
    info["pi_true"] = pi_true
    return res, info


def subsample_val(s_val, y_val, frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y_val == c)[0], max(int(round(frac * np.sum(y_val == c))), 5), replace=False) for c in (0, 1)])
    return s_val[idx], y_val[idx]


def run_controlled_file(path, cfg, condition="main", metric="f1", batch_sizes=None, extra=False,
                        score_key="s_test", val_frac=None, calibrator=None, keep_batches=False, reps=None):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    ds, model, seed = meta["dataset"], meta["model"], meta["seed"]
    s_val, y_val, s_te, y_te = z["s_val"], z["y_val"], z[score_key], z["y_test"]
    if val_frac is not None:
        s_val, y_val = subsample_val(s_val, y_val, val_frac, stable_seed(ds, seed, "val", val_frac))
    mc = cfg["method"]
    dl = DecisionLayer(s_val, y_val, metric=metric, prior_sd=mc["prior_sd"], guard_alpha=mc["guard_alpha"],
                       n_grid=mc["n_grid"], calibrator=calibrator or mc["calibrator"], seed=seed)
    pos, neg = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
    pi0 = dl.sm.pi0
    odds0 = pi0 / (1 - pi0)
    cc = cfg["controlled"]
    rows, qrows, batches = [], [], []
    for rho in cc["odds_multipliers"]:
        pi_t = rho * odds0 / (1 + rho * odds0)
        for m in (batch_sizes or cc["batch_sizes"]):
            rng = np.random.default_rng(stable_seed(ds, model, seed, condition, metric, rho, m))
            acc, qacc, guard = {}, {}, []
            R = reps or cc["reps"]
            for r in range(R):
                n_pos = 0
                while not (1 <= n_pos <= m - 1):
                    n_pos = rng.binomial(m, pi_t)
                idx = np.concatenate([rng.choice(pos, n_pos), rng.choice(neg, m - n_pos)])
                res, info = eval_batch(dl, s_te[idx], y_te[idx], metric, extra)
                orc = res["oracle_thr"][0]
                for k, v in res.items():
                    acc.setdefault(k, []).append((v[0], v[1], v[2], orc - v[0]))
                for q in QUANT:
                    if q in info:
                        qacc.setdefault(q, []).append(abs(info[q] - info["pi_true"]))
                guard.append(info["guard_p"] < mc["guard_alpha"])
                if keep_batches:
                    batches.append(dict(dataset=ds, model=model, seed=seed, rho=rho, m=m, rep=r, n_pos=n_pos, pi_true=info["pi_true"],
                                        pi_mle=info["pi_mle"], pi_mean=info["pi_mean"], pi_em=info["pi_em"], guard_p=info["guard_p"],
                                        t_static=info["t_static"], t_bayes=info["t_bayes"],
                                        **{f"v_{k}": v[0] for k, v in res.items()}, **{f"k_{k}": v[3] for k, v in res.items()}))
            base = dict(dataset=ds, model=model, seed=seed, condition=condition, metric=metric, rho=rho, m=m, pi0=pi0, pi_target=pi_t, n_batches=R)
            for k, v in acc.items():
                a = np.array(v, float)
                rows.append(dict(base, method=k, value=a[:, 0].mean(), precision=np.nanmean(a[:, 1]), recall=np.nanmean(a[:, 2]), regret=a[:, 3].mean()))
            for q, v in qacc.items():
                qrows.append(dict(base, quantifier=q, mae=float(np.mean(v))))
            qrows.append(dict(base, quantifier="guard_fire_rate", mae=float(np.mean(guard))))
    return rows, qrows, batches
