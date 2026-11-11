"""Duplicate-aware stratified splitting and leakage audit."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def row_groups(X: pd.DataFrame):
    """Identical feature vectors share a group id, so exact duplicates can never
    straddle train / validation / test."""
    h = pd.util.hash_pandas_object(X.astype(str) if any(not pd.api.types.is_numeric_dtype(t) for t in X.dtypes) else X, index=False)
    return pd.factorize(h)[0]


def make_split(X, y, seed, n_folds=10, train_folds=5, val_folds=2):
    g = row_groups(X)
    folds = np.empty(len(y), int)
    for i, (_, te) in enumerate(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g)):
        folds[te] = i
    tr = np.where(folds < train_folds)[0]
    va = np.where((folds >= train_folds) & (folds < train_folds + val_folds))[0]
    te = np.where(folds >= train_folds + val_folds)[0]
    audit = dict(n=len(y), n_dup_rows=int(len(y) - len(np.unique(g))),
                 cross_split_dups=int(len(set(g[tr]) & set(g[te])) + len(set(g[tr]) & set(g[va])) + len(set(g[va]) & set(g[te]))))
    return tr, va, te, audit
