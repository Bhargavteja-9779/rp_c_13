"""Leakage-free preprocessing: every statistic is fitted on the training split only."""
from __future__ import annotations
import numpy as np, pandas as pd


class TabularEncoder:
    """Numeric columns pass through (NaN kept for tree models); categorical columns are
    ordinal-encoded for tree models and frequency-encoded for linear/neural models.
    Linear/neural models additionally get median imputation + standardisation."""

    def fit(self, X: pd.DataFrame):
        self.cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
        self.num = [c for c in X.columns if c not in self.cat]
        self.maps, self.freq = {}, {}
        for c in self.cat:
            v = X[c].astype(str)
            vc = v.value_counts()
            self.maps[c] = {k: i for i, k in enumerate(vc.index)}
            self.freq[c] = (vc / len(v)).to_dict()
        T = self._tree(X)
        self.keep = ~np.all(np.isnan(T), axis=0)
        T = T[:, self.keep]
        L = self._lin_raw(X)[:, self.keep]
        self.med = np.nan_to_num(np.nanmedian(L, axis=0))
        L = np.where(np.isnan(L), self.med, L)
        self.mu, self.sd = L.mean(0), L.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        self.num_mask = np.array([True] * len(self.num) + [False] * len(self.cat))[self.keep]
        return self

    def _num(self, X):
        return X[self.num].to_numpy(dtype=float, na_value=np.nan) if self.num else np.zeros((len(X), 0))

    def _tree(self, X):
        cats = [X[c].astype(str).map(self.maps[c]).fillna(-1).to_numpy(float) for c in self.cat]
        return np.column_stack([self._num(X)] + cats) if cats else self._num(X)

    def _lin_raw(self, X):
        cats = [X[c].astype(str).map(self.freq[c]).fillna(0.0).to_numpy(float) for c in self.cat]
        return np.column_stack([self._num(X)] + cats) if cats else self._num(X)

    def tree(self, X):
        return self._tree(X)[:, self.keep].astype(np.float32)

    def linear(self, X):
        L = self._lin_raw(X)[:, self.keep]
        L = np.where(np.isnan(L), self.med, L)
        return np.clip((L - self.mu) / self.sd, -10, 10).astype(np.float32)


def add_feature_noise(X: pd.DataFrame, enc: TabularEncoder, sigma, rng):
    """Label-shift violation: Gaussian noise (sigma x training SD) on every numeric feature."""
    X = X.copy()
    for c in enc.num:
        v = X[c].to_numpy(dtype=float, na_value=np.nan)
        sd = np.nanstd(v)
        X[c] = v + rng.normal(0, sigma * (sd if sd > 0 else 1.0), len(v))
    return X
