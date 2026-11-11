"""Real (non-simulated) shift suites. Each loader returns
    X_dev, y_dev            development data (i.i.d. split into train / validation later)
    batches                 list of (batch_name, X, y) deployment batches, in natural order
Nothing from the deployment batches is ever used for fitting."""
from __future__ import annotations
import os, io, zipfile, glob
import numpy as np, pandas as pd
from . import openml_suite

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def _chunks(X, y, size, prefix, min_pos=1):
    out = []
    for i, a in enumerate(range(0, len(y) - size + 1, size)):
        yy = y[a:a + size]
        if min_pos <= yy.sum() <= len(yy) - 1:
            out.append((f"{prefix}{i:03d}", X.iloc[a:a + size].reset_index(drop=True), yy))
    return out


def bank_time():
    """UCI Bank Marketing (bank-additional-full, rows in chronological order May 2008 - Nov 2010).
    'duration' is dropped (known target leak). Dev = first 50 % of calls; then batches of 2,000 calls."""
    with zipfile.ZipFile(os.path.join(RAW, "bank_marketing.zip")) as z:
        with zipfile.ZipFile(io.BytesIO(z.read("bank-additional.zip"))) as z2:
            name = [n for n in z2.namelist() if n.endswith("bank-additional-full.csv")][0]
            df = pd.read_csv(io.BytesIO(z2.read(name)), sep=";")
    y = (df.pop("y") == "yes").astype(int).to_numpy()
    X = df.drop(columns=["duration"])
    n = len(y) // 2
    return X.iloc[:n].reset_index(drop=True), y[:n], _chunks(X.iloc[n:], y[n:], 2000, "b")


def creditcard_time():
    """ULB credit-card fraud (OpenML 1597, rows in transaction-time order over two days).
    Dev = first 50 % of transactions; then batches of 10,000 transactions."""
    X, y = openml_suite.load("creditcard")
    n = len(y) // 2
    return X.iloc[:n].reset_index(drop=True), y[:n], _chunks(X.iloc[n:], y[n:], 10000, "b")


def electricity_time():
    """Elec2 (OpenML 151, half-hourly, chronological). Positive = price UP.
    Dev = first 30 %; then batches of 1,008 half-hours (three weeks)."""
    X, y = openml_suite.load("electricity")
    n = int(0.3 * len(y))
    return X.iloc[:n].reset_index(drop=True), y[:n], _chunks(X.iloc[n:], y[n:], 1008, "b")


def elliptic_time():
    """Elliptic Bitcoin transactions (Weber et al. 2019): 49 time steps, labelled nodes only.
    Positive = illicit. Dev = time steps 1-34 (the standard temporal split); one batch per step 35-49."""
    f = pd.read_csv(os.path.join(RAW, "elliptic_features.zip"), header=None)
    c = pd.read_csv(os.path.join(RAW, "elliptic_classes.zip"))
    f = f.rename(columns={0: "txId", 1: "t"}).merge(c, on="txId")
    f = f[f["class"].astype(str) != "unknown"].sort_values("t", kind="stable")
    y = (f["class"].astype(str) == "1").astype(int).to_numpy()
    t = f["t"].to_numpy()
    X = f.drop(columns=["txId", "t", "class"]).reset_index(drop=True)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    dev = t <= 34
    batches = [(f"t{k:02d}", X[t == k].reset_index(drop=True), y[t == k]) for k in range(35, 50) if 1 <= y[t == k].sum()]
    return X[dev].reset_index(drop=True), y[dev], batches


ACS_COLS = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]


def _acs_state(path):
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(io.BytesIO(z.read(name)), usecols=ACS_COLS + ["PINCP", "PWGTP"])
    df = df[(df.AGEP > 16) & (df.PINCP > 100) & (df.WKHP > 0) & (df.PWGTP >= 1)]   # ACSIncome filter (Ding et al. 2021)
    y = (df.PINCP > 50000).astype(int).to_numpy()
    X = df[ACS_COLS].reset_index(drop=True).copy()
    for c in ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]:
        X[c] = X[c].astype("Int64").astype(str)
    return X, y


def acs_income_states():
    """ACSIncome task (Ding et al. 2021) on 2018 1-Year ACS PUMS. Dev = California;
    one deployment batch per other state / DC / Puerto Rico (51 batches)."""
    files = sorted(glob.glob(os.path.join(RAW, "acs", "csv_p*.zip")))
    Xd, yd = _acs_state(os.path.join(RAW, "acs", "csv_pca.zip"))
    batches = []
    for f in files:
        st = os.path.basename(f)[5:7].upper()
        if st != "CA":
            X, y = _acs_state(f)
            batches.append((st, X, y))
    return Xd, yd, batches


SUITES = dict(bank_time=bank_time, creditcard_time=creditcard_time, electricity_time=electricity_time,
              elliptic_time=elliptic_time, acs_income_states=acs_income_states)
