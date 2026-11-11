"""Train every base classifier of the controlled suite and cache scores."""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
from joblib import Parallel, delayed
from src.training.score_cache import train_controlled


def main(cfg_path, root):
    cfg = yaml.safe_load(open(cfg_path))
    cc = cfg["controlled"]
    Parallel(n_jobs=cfg["n_jobs"])(delayed(train_controlled)(d, s, cfg["models"], cfg, root, cc["noise_levels"], cfg["main_model"])
                                   for d in cc["datasets"] for s in cfg["seeds"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/default.yaml"); ap.add_argument("--root", default="..")
    a = ap.parse_args(); main(a.config, a.root)
