"""Experiments 1-5,7: controlled label-shift benchmark on cached scores."""
from __future__ import annotations
import os, sys, glob, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml, pandas as pd
from joblib import Parallel, delayed
from src.evaluation.protocol import run_controlled_file


def conditions(cfg):
    cc, mm = cfg["controlled"], cfg["main_model"]
    conds = [dict(name="main", models=cfg["models"], kw=dict(metric="f1", keep_batches=True))]
    conds.append(dict(name="ablation", models=[mm], kw=dict(metric="f1", extra=True, batch_sizes=cc["extra_batch_sizes"] + cc["batch_sizes"])))
    for mt in cc["metrics_extra"]:
        conds.append(dict(name=f"metric_{mt}", models=[mm], kw=dict(metric=mt, batch_sizes=[2000])))
    for sg in cc["noise_levels"]:
        conds.append(dict(name=f"noise_{sg}", models=[mm], kw=dict(metric="f1", batch_sizes=[2000], score_key=f"s_test_noise{sg}")))
    for vf in cc["val_fractions"]:
        conds.append(dict(name=f"valfrac_{vf}", models=[mm], kw=dict(metric="f1", batch_sizes=[2000], val_frac=vf)))
    conds.append(dict(name="calib_platt", models=[mm], kw=dict(metric="f1", batch_sizes=[2000], calibrator="platt")))
    return conds


def main(cfg_path, root, only=None):
    cfg = yaml.safe_load(open(cfg_path))
    out = os.path.join(root, "results", "controlled"); os.makedirs(out, exist_ok=True)
    for c in conditions(cfg):
        if only and c["name"] not in only:
            continue
        files = [f for d in cfg["controlled"]["datasets"] for m in c["models"] for s in cfg["seeds"]
                 for f in glob.glob(os.path.join(root, "scores", "controlled", f"{d}__{m}__s{s}.npz"))]
        t0 = time.time()
        res = Parallel(n_jobs=cfg["n_jobs"])(delayed(run_controlled_file)(f, cfg, condition=c["name"], **c["kw"]) for f in files)
        pd.DataFrame([r for x in res for r in x[0]]).to_csv(os.path.join(out, f"{c['name']}.csv"), index=False)
        pd.DataFrame([r for x in res for r in x[1]]).to_csv(os.path.join(out, f"{c['name']}__quant.csv"), index=False)
        b = [r for x in res for r in x[2]]
        if b:
            pd.DataFrame(b).to_csv(os.path.join(out, f"{c['name']}__batches.csv.gz"), index=False)
        print(f"[controlled] {c['name']:14s} files={len(files)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/default.yaml"); ap.add_argument("--root", default="..")
    ap.add_argument("--only", nargs="*", help="run only these conditions")
    a = ap.parse_args(); main(a.config, a.root, a.only)
