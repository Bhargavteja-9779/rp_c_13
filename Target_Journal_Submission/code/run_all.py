#!/usr/bin/env python3
"""Reproduce the whole study.

    python run_all.py --mode quick     # pipeline smoke test, a few minutes, reduced computation
    python run_all.py --mode full      # the complete scientific experiment (about 1 h on 4 CPU cores)

Stages: data download -> base-model training (score cache) -> controlled label-shift benchmark ->
real-shift suites -> efficiency -> label-assisted experiment -> guard calibration -> statistics + tables -> figures. Every stage is resumable.
Outputs go to <root>/results, <root>/tables, <root>/figures (root = parent folder for `full`,
`quick_run/` for `quick`, so a smoke test never overwrites the reported results)."""
from __future__ import annotations
import argparse, os, sys, subprocess, time, platform, json

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(args, log):
    print(">>", " ".join(args), flush=True)
    with open(log, "a") as f:
        r = subprocess.run([sys.executable] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"stage failed, see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    cfg = os.path.join("config", "quick.yaml" if a.mode == "quick" else "default.yaml")
    root = os.path.join(HERE, "quick_run") if a.mode == "quick" else os.path.dirname(HERE)
    logs = os.path.join(root, "results", "experiment_logs"); os.makedirs(logs, exist_ok=True)
    sys.path.insert(0, HERE)
    from src.models.zoo import detect_device
    env = dict(python=platform.python_version(), platform=platform.platform(), device=detect_device(), cpu_count=os.cpu_count(), mode=a.mode)
    json.dump(env, open(os.path.join(logs, "environment.json"), "w"), indent=1); print(env)
    t0 = time.time()
    if not a.skip_download:
        sh(["scripts/download_data.py", "--config", cfg], os.path.join(logs, "download.log"))
    sh(["experiments/train_controlled.py", "--config", cfg, "--root", root], os.path.join(logs, "train_controlled.log"))
    sh(["experiments/run_controlled.py", "--config", cfg, "--root", root], os.path.join(logs, "run_controlled.log"))
    sh(["experiments/run_real.py", "--config", cfg, "--root", root], os.path.join(logs, "run_real.log"))
    sh(["experiments/run_efficiency.py", "--config", cfg, "--root", root], os.path.join(logs, "run_efficiency.log"))
    sh(["experiments/run_labelled.py", "--config", cfg, "--root", root], os.path.join(logs, "run_labelled.log"))
    sh(["experiments/run_guard_calibration.py", "--root", root, "--reps", "100" if a.mode == "quick" else "500"], os.path.join(logs, "run_guard_calibration.log"))
    sh(["scripts/make_tables.py", "--root", root], os.path.join(logs, "make_tables.log"))
    sh(["scripts/make_figures.py", "--root", root], os.path.join(logs, "make_figures.log"))
    print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {root}")


if __name__ == "__main__":
    main()
