"""Base classifiers. Fixed, untuned-but-sensible hyper-parameters shared by all datasets;
the decision layer under study is model-agnostic and only consumes scores."""
from __future__ import annotations
import numpy as np


def detect_device():
    """CPU / CUDA / Apple-Silicon MPS auto-detection (no GPU is assumed)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _es_split(n, seed, frac=0.1):
    rng = np.random.default_rng(seed + 1000)
    idx = rng.permutation(n)
    k = max(int(frac * n), 1)
    return idx[k:], idx[:k]


def fit_predict(model, Xtr, ytr, Xs, seed, n_threads=1):
    import warnings
    warnings.filterwarnings("ignore")
    """Train `model` on (Xtr, ytr); return list of positive-class scores for each array in Xs.
    Early stopping uses a 10 % slice of the *training* split, never validation/test."""
    fit, es = _es_split(len(ytr), seed)
    if model == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, min_child_samples=20, random_state=seed, n_jobs=n_threads, verbose=-1)
        m.fit(Xtr[fit], ytr[fit], eval_set=[(Xtr[es], ytr[es])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return [m.predict_proba(X)[:, 1] for X in Xs]
    if model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", early_stopping_rounds=50, eval_metric="logloss", random_state=seed, n_jobs=n_threads)
        m.fit(Xtr[fit], ytr[fit], eval_set=[(Xtr[es], ytr[es])], verbose=False)
        return [m.predict_proba(X)[:, 1] for X in Xs]
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1.0, max_iter=300).fit(Xtr, ytr)
        return [m.predict_proba(X)[:, 1] for X in Xs]
    if model == "mlp":
        return _mlp(Xtr, ytr, Xs, fit, es, seed, n_threads)
    raise ValueError(model)


def _mlp(Xtr, ytr, Xs, fit, es, seed, n_threads, epochs=40, patience=5, bs=512):
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(n_threads)
    dev = detect_device()
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                        nn.Dropout(0.1), nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xf = torch.tensor(Xtr[fit]); yf = torch.tensor(ytr[fit], dtype=torch.float32)
    Xe = torch.tensor(Xtr[es]).to(dev); ye = torch.tensor(ytr[es], dtype=torch.float32).to(dev)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(yf), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(Xf[b].to(dev)).squeeze(1), yf[b].to(dev)).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            l = float(lossf(net(Xe).squeeze(1), ye))
        if l < best - 1e-5:
            best, bad = l, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    outs = []
    with torch.no_grad():
        for X in Xs:
            outs.append(torch.cat([torch.sigmoid(net(torch.tensor(X[i:i + 65536]).to(dev)).squeeze(1)).cpu()
                                   for i in range(0, len(X), 65536)]).numpy().astype(float))
    return outs
