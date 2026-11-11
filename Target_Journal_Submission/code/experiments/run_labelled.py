"""Experiment 9 (post-hoc, motivated by the real-shift analysis): how much does a small labelled
audit sample of the deployment batch help, if it is spent only on estimating the prevalence?"""
from __future__ import annotations
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, yaml
from joblib import Parallel, delayed
from src.methods.rules import DecisionLayer, paft_labelled, retune_on_labels
from src.evaluation.batch_eval import Batch
from src.evaluation.protocol import stable_seed

N_LAB = [50, 200, 1000]


def _one(dl, s, y, n_lab, rng):
    idx = rng.permutation(len(s)); L, U = idx[:n_lab], idx[n_lab:]
    b = Batch(s[U].astype(float), y[U])
    if b.npos == 0 or b.npos == b.m:
        return None
    k = int(y[L].sum())
    r = dl.paft(b.s)
    thr = {"static": dl.t_static, "paft_bayes": r["t_bayes"], "retune_labels": retune_on_labels(s[L], y[L], dl.t_static),
           "paft_L": paft_labelled(dl, b.s, k, n_lab, "labels")[0], "paft_L_both": paft_labelled(dl, b.s, k, n_lab, "both")[0],
           "paft_L_guarded": paft_labelled(dl, b.s, k, n_lab, "guarded")[0], "oracle_pi": dl.oracle_pi_threshold(b.npos / b.m)}
    out = {m: b.score(b.k_of_threshold(t))[0] for m, t in thr.items()}
    out["oracle_thr"] = b.oracle()
    return out


def run_real(path, cfg, reps=20):
    z = np.load(path); meta = json.loads(str(z["meta"])); mc = cfg["method"]
    dl = DecisionLayer(z["s_val"], z["y_val"], prior_sd=mc["prior_sd"], guard_alpha=mc["guard_alpha"], n_grid=mc["n_grid"], calibrator=mc["calibrator"], seed=meta["seed"], n_bag=1)
    rows, a = [], 0
    for name, n in zip(meta["batch_names"], meta["batch_sizes"]):
        s, y = z["s_dep"][a:a + n], z["y_dep"][a:a + n]; a += n
        for nl in N_LAB:
            if nl > n // 2: continue
            rng = np.random.default_rng(stable_seed(meta["suite"], meta["seed"], name, nl))
            res = [r for r in (_one(dl, s, y, nl, rng) for _ in range(reps)) if r]
            if res:
                for m, v in pd.DataFrame(res).mean().items():
                    rows.append(dict(suite=meta["suite"], seed=meta["seed"], unit=name, n_lab=nl, method=m, value=v))
    return rows


def run_controlled(path, cfg, m=2000, reps=20):
    z = np.load(path); meta = json.loads(str(z["meta"])); mc = cfg["method"]
    dl = DecisionLayer(z["s_val"], z["y_val"], prior_sd=mc["prior_sd"], guard_alpha=mc["guard_alpha"], n_grid=mc["n_grid"], calibrator=mc["calibrator"], seed=meta["seed"], n_bag=1)
    s_te, y_te = z["s_test"], z["y_test"]; pos, neg = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
    odds0 = dl.sm.pi0 / (1 - dl.sm.pi0); rows = []
    for rho in cfg["controlled"]["odds_multipliers"]:
        pi_t = rho * odds0 / (1 + rho * odds0)
        for nl in [50, 200]:
            rng = np.random.default_rng(stable_seed(meta["dataset"], meta["seed"], "lab", rho, nl)); res = []
            for _ in range(reps):
                n_pos = 0
                while not (1 <= n_pos <= m - 1): n_pos = rng.binomial(m, pi_t)
                idx = np.concatenate([rng.choice(pos, n_pos), rng.choice(neg, m - n_pos)])
                r = _one(dl, s_te[idx], y_te[idx], nl, rng)
                if r: res.append(r)
            for mth, v in pd.DataFrame(res).mean().items():
                rows.append(dict(suite="controlled", seed=meta["seed"], unit=meta["dataset"], rho=rho, n_lab=nl, method=mth, value=v))
    return rows


def main(cfg_path, root):
    cfg = yaml.safe_load(open(cfg_path)); mm = cfg["main_model"]
    fr = [os.path.join(root, "scores", "real", f"{su}__{mm}__s{s}.npz") for su in cfg["real"]["suites"] for s in cfg["seeds"]]
    fc = [os.path.join(root, "scores", "controlled", f"{d}__{mm}__s{s}.npz") for d in cfg["controlled"]["datasets"] for s in cfg["seeds"]]
    res = Parallel(n_jobs=cfg["n_jobs"])([delayed(run_real)(f, cfg) for f in fr] + [delayed(run_controlled)(f, cfg) for f in fc])
    out = os.path.join(root, "results", "labelled"); os.makedirs(out, exist_ok=True)
    pd.DataFrame([r for x in res for r in x]).to_csv(os.path.join(out, "labelled.csv"), index=False); print("[labelled] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/default.yaml"); ap.add_argument("--root", default="..")
    a = ap.parse_args(); main(a.config, a.root)
