"""Train base classifiers and cache validation / test-pool scores (checkpoint = npz of scores)."""
from __future__ import annotations
import os, time, json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from ..data import openml_suite
from ..data.splits import make_split
from ..preprocessing.encode import TabularEncoder, add_feature_noise
from ..models.zoo import fit_predict

TREE = {"lgbm", "xgb"}


def cache_path(root, suite, ds, model, seed):
    return os.path.join(root, "scores", suite, f"{ds}__{model}__s{seed}.npz")


def train_controlled(ds, seed, models, cfg, root, noise_levels=(), main_model="lgbm"):
    todo = [m for m in models if not os.path.exists(cache_path(root, "controlled", ds, m, seed))]
    if not todo:
        return
    X, y = openml_suite.load(ds)
    sp = cfg["split"]
    tr, va, te, audit = make_split(X, y, seed, sp["n_folds"], sp["train_folds"], sp["val_folds"])
    enc = TabularEncoder().fit(X.iloc[tr])
    for model in todo:
        f = enc.tree if model in TREE else enc.linear
        Xs = [f(X.iloc[va]), f(X.iloc[te])]
        names = ["s_val", "s_test"]
        if model == main_model:
            for sg in noise_levels:
                Xs.append(f(add_feature_noise(X.iloc[te], enc, sg, np.random.default_rng(10_000 + seed))))
                names.append(f"s_test_noise{sg}")
        t0 = time.time()
        outs = fit_predict(model, f(X.iloc[tr]), y[tr], Xs, seed)
        meta = dict(dataset=ds, model=model, seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te),
                    prevalence=float(y.mean()), n_features=int(X.shape[1]), fit_seconds=time.time() - t0,
                    auc_test=float(roc_auc_score(y[te], outs[1])), ap_test=float(average_precision_score(y[te], outs[1])), **audit)
        p = cache_path(root, "controlled", ds, model, seed)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p, y_val=y[va], y_test=y[te], meta=json.dumps(meta), **dict(zip(names, outs)))
        print(f"[train] {ds:16s} {model:6s} seed={seed} AUC={meta['auc_test']:.4f} AP={meta['ap_test']:.4f} {meta['fit_seconds']:.0f}s", flush=True)
