"""Experiment 6: computational cost of the decision layer (single CPU thread)."""
from __future__ import annotations
import os, sys, time, argparse, json, glob, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, yaml
from src.methods.rules import DecisionLayer, dta_topk, mixture_ks_pvalue
from src.methods import quantify as Q


def main(cfg_path, root):
    cfg = yaml.safe_load(open(cfg_path))
    f = sorted(glob.glob(os.path.join(root, "scores", "controlled", f"*__{cfg['main_model']}__s0.npz")))
    f = ([x for x in f if "adult" in x] or f)[0]
    z = np.load(f); s_val, y_val, s_te = z["s_val"], z["y_val"], z["s_test"]
    meta = json.loads(str(z["meta"]))
    t0 = time.perf_counter(); tracemalloc.start()
    dl = DecisionLayer(s_val, y_val, seed=0)
    fit_s = time.perf_counter() - t0; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    rows = [dict(component="fit decision layer (once, incl. 50 bootstrap static thresholds)", m=len(y_val), ms=1000 * fit_s, peak_mb=peak / 2 ** 20)]
    rng = np.random.default_rng(0)
    for m in [100, 500, 2000, 10000, 100000]:
        s = rng.choice(s_te, m)
        def timeit(fn, n=30):
            fn(); t = time.perf_counter()
            for _ in range(n): fn()
            return 1000 * (time.perf_counter() - t) / n
        p = dl.cal(s)
        comps = {
            "static threshold (comparison only)": lambda: s >= dl.t_static,
            "EM prior correction (SLD)": lambda: Q.q_em(dl.cal(s), dl.sm.pi0),
            "DTA-EM (EM + sort + expected-F scan)": lambda: dta_topk(np.sort(Q.q_em(dl.cal(s), dl.sm.pi0)[1])[::-1]),
            "PAFT-Plug": lambda: dl.sm.thr[np.argmax(dl.surf[np.argmax(Q.binned_log_lik(dl.sm, s, dl.grid))])],
            "PAFT-Bayes": lambda: dl.paft(s),
            "PAFT guard (KS test)": lambda: mixture_ks_pvalue(dl.sm, np.sort(s), 0.2),
            "all rules (decide)": lambda: dl.decide(s),
        }
        for k, fn in comps.items():
            rows.append(dict(component=k, m=m, ms=timeit(fn), peak_mb=np.nan))
    rows.append(dict(component="retrain LightGBM on adult (reference)", m=meta["n_train"], ms=1000 * meta["fit_seconds"], peak_mb=np.nan))
    rows.append(dict(component="stored state: thresholds T x grid G surface (float64)", m=dl.surf.size, ms=np.nan, peak_mb=dl.surf.nbytes / 2 ** 20))
    out = os.path.join(root, "results", "efficiency"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "efficiency.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/default.yaml"); ap.add_argument("--root", default="..")
    a = ap.parse_args(); main(a.config, a.root)
