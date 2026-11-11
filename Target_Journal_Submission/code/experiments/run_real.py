"""Experiment 5 (generalisation under real, non-simulated shift): train on the development
period / site, deploy on naturally occurring batches."""
from __future__ import annotations
import os, sys, argparse, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from src.data.real_suites import SUITES
from src.data.splits import make_split
from src.preprocessing.encode import TabularEncoder
from src.models.zoo import fit_predict
from src.methods.rules import DecisionLayer
from src.evaluation.protocol import eval_batch

TREE = {"lgbm", "xgb"}


def _train(suite, Xd, yd, batches, model, seed, root):
    p = os.path.join(root, "scores", "real", f"{suite}__{model}__s{seed}.npz")
    if os.path.exists(p):
        return
    tr, va, _, audit = make_split(Xd, yd, seed, n_folds=7, train_folds=5, val_folds=2)
    enc = TabularEncoder().fit(Xd.iloc[tr])
    f = enc.tree if model in TREE else enc.linear
    t0 = time.time()
    outs = fit_predict(model, f(Xd.iloc[tr]), yd[tr], [f(Xd.iloc[va])] + [f(b[1]) for b in batches], seed)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    meta = dict(suite=suite, model=model, seed=seed, n_train=len(tr), n_val=len(va), fit_seconds=time.time() - t0,
                batch_names=[b[0] for b in batches], batch_sizes=[len(b[2]) for b in batches], **audit)
    np.savez_compressed(p, s_val=outs[0], y_val=yd[va], s_dep=np.concatenate(outs[1:]), y_dep=np.concatenate([b[2] for b in batches]), meta=json.dumps(meta))
    print(f"[real-train] {suite} {model} s{seed} {time.time()-t0:.0f}s", flush=True)


def _evaluate(path, cfg):
    z = np.load(path); meta = json.loads(str(z["meta"]))
    mc = cfg["method"]
    dl = DecisionLayer(z["s_val"], z["y_val"], prior_sd=mc["prior_sd"], guard_alpha=mc["guard_alpha"], n_grid=mc["n_grid"],
                       calibrator=mc["calibrator"], seed=meta["seed"])
    rows, a, prev = [], 0, None
    for name, n in zip(meta["batch_names"], meta["batch_sizes"]):
        s, y = z["s_dep"][a:a + n], z["y_dep"][a:a + n]; a += n
        res, info = eval_batch(dl, s, y, "f1")
        # exploratory sequential variant: prior centred on the previous batch's posterior mean
        from src.evaluation.batch_eval import Batch
        b = Batch(s.astype(float), y)
        r = dl.paft(b.s, prior_sd=1.0, prior_center=prev)
        res["paft_seq"] = b.score(b.k_of_threshold(r["t_bayes"]), "f1") + (0,)
        prev = float(np.clip(r["pi_mean"], 1e-5, 1 - 1e-5))
        auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
        for k, v in res.items():
            rows.append(dict(suite=meta["suite"], model=meta["model"], seed=meta["seed"], batch=name, m=n, pi_true=info["pi_true"], pi0=dl.sm.pi0,
                             pi_mle=info["pi_mle"], pi_mean=info["pi_mean"], pi_em=info["pi_em"], pi_acc=info["pi_acc"], guard_p=info["guard_p"], auc=auc,
                             method=k, value=v[0], precision=v[1], recall=v[2], regret=res["oracle_thr"][0] - v[0]))
    return rows


def main(cfg_path, root):
    cfg = yaml.safe_load(open(cfg_path))
    for suite in cfg["real"]["suites"]:
        Xd, yd, batches = SUITES[suite]()
        Parallel(n_jobs=cfg["n_jobs"])(delayed(_train)(suite, Xd, yd, batches, m, s, root) for m in cfg["models"] for s in cfg["seeds"])
    files = [os.path.join(root, "scores", "real", f"{su}__{m}__s{s}.npz") for su in cfg["real"]["suites"] for m in cfg["models"] for s in cfg["seeds"]]
    res = Parallel(n_jobs=cfg["n_jobs"])(delayed(_evaluate)(f, cfg) for f in files)
    out = os.path.join(root, "results", "real"); os.makedirs(out, exist_ok=True)
    pd.DataFrame([r for x in res for r in x]).to_csv(os.path.join(out, "real_batches.csv"), index=False)
    print("[real] done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/default.yaml"); ap.add_argument("--root", default="..")
    a = ap.parse_args(); main(a.config, a.root)
