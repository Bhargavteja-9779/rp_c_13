"""Experiment 10 (added in review round 1): size and power of the label-shift guard on synthetic scores
where every deployment batch consists of *fresh* draws (no resampling from a finite test pool)."""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.methods.score_model import ScoreModel
from src.methods.rules import mixture_ks_pvalue
from src.methods import quantify as Q


def draw(n1, n0, rng, sep=2.0, shift=0.0, scale=1.0):
    z = np.concatenate([rng.normal(sep + shift, scale, n1), rng.normal(shift, scale, n0)])
    return 1 / (1 + np.exp(-(z - sep / 2))), np.concatenate([np.ones(n1, int), np.zeros(n0, int)])


def cell(n1_val, m, rho, shift, pi0=0.05, reps=500, n_models=10, seed=0):
    rng = np.random.default_rng(seed); fires = []
    odds = rho * pi0 / (1 - pi0); pi_t = odds / (1 + odds)
    for _ in range(n_models):                       # average over validation draws
        s, y = draw(n1_val, int(n1_val * (1 - pi0) / pi0), rng)
        sm = ScoreModel(s, y); grid = Q.pi_grid(sm.pi0)
        for _ in range(reps // n_models):
            k = max(rng.binomial(m, pi_t), 1)
            sb, _ = draw(k, m - k, rng, shift=shift)
            fires.append(mixture_ks_pvalue(sm, np.sort(sb), Q.q_mle(sm, sb, grid))[0] < 0.01)
    return dict(n1_val=n1_val, m=m, rho=rho, score_shift=shift, fire_rate=float(np.mean(fires)), n=len(fires))


def main(root, reps):
    jobs = [(n1, m, rho, sh) for n1 in [50, 100, 500, 5000] for m in [500, 2000, 10000] for rho in [1.0, 4.0] for sh in [0.0, 0.25, 0.5]]
    res = Parallel(n_jobs=4)(delayed(cell)(*j, reps=reps, seed=i) for i, j in enumerate(jobs))
    out = os.path.join(root, "results", "guard_calibration"); os.makedirs(out, exist_ok=True)
    d = pd.DataFrame(res); d.to_csv(os.path.join(out, "guard_calibration.csv"), index=False)
    print(d.pivot_table(index=["score_shift", "n1_val"], columns=["rho", "m"], values="fire_rate").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".."); ap.add_argument("--reps", type=int, default=500)
    a = ap.parse_args(); main(a.root, a.reps)
