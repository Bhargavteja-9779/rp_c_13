"""Controlled-shift suite: 18 public binary tabular datasets (OpenML).

Inclusion criteria (fixed a priori): binary target (or a documented standard
binarisation), >= 10,000 instances, >= 400 minority instances, minority
prevalence <= 30 %. The minority class is always the positive class."""
from __future__ import annotations
import os, warnings
import numpy as np, pandas as pd

DATA_HOME = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "openml")

# name -> (OpenML data id, function mapping raw target -> {0,1} or None for "minority = 1")
REGISTRY = {
    "creditcard": (1597, None), "APSFailure": (41138, None), "KDD09_appetency": (1111, None),
    "porto_seguro": (42742, None), "Amazon_employee": (4135, None), "census_income": (42750, None),
    "KDD09_churn": (1112, None), "KDD09_upselling": (1114, None),
    "Diabetes130US": (4541, lambda y: (y.astype(str) == "<30").astype(int)),   # 30-day readmission
    "bank_marketing": (1461, None), "kick": (41162, None), "online_shoppers": (45060, None),
    "Click_small": (1220, None), "jm1": (1053, None), "default_credit": (42477, None),
    "adult": (1590, None), "MiniBooNE": (41150, None), "nomao": (1486, None),
    "electricity": (151, lambda y: (y.astype(str) == "UP").astype(int)),       # used by the temporal suite
}


def load(name):
    from sklearn.datasets import fetch_openml
    did, fn = REGISTRY[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fetch_openml(data_id=did, as_frame=True, data_home=DATA_HOME, parser="auto")
    X, y = d.data.copy(), d.target
    if fn is not None:
        y = fn(y)
    else:
        vc = y.astype(str).value_counts()
        assert len(vc) == 2, f"{name}: target is not binary"
        y = (y.astype(str) == vc.index[-1]).astype(int)      # minority -> 1
    y = np.asarray(y).astype(int)
    X = X.reset_index(drop=True)
    return X, y
