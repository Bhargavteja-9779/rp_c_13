"""Download every public dataset used in the study (idempotent)."""
from __future__ import annotations
import os, sys, argparse, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
STATES = "al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy pr".split()
FILES = {
    "bank_time": [("bank_marketing.zip", "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip")],
    "elliptic_time": [("elliptic_features.zip", "https://data.pyg.org/datasets/elliptic/elliptic_txs_features.csv.zip"),
                      ("elliptic_classes.zip", "https://data.pyg.org/datasets/elliptic/elliptic_txs_classes.csv.zip")],
    "acs_income_states": [(f"acs/csv_p{s}.zip", f"https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_p{s}.zip") for s in STATES],
}


def get(name, url):
    p = os.path.join(RAW, name); os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return
    print("downloading", url, flush=True)
    urllib.request.urlretrieve(url, p)


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    for su in cfg["real"]["suites"]:
        for name, url in FILES.get(su, []):
            get(name, url)
    from src.data import openml_suite
    need = set(cfg["controlled"]["datasets"]) | ({"creditcard"} if "creditcard_time" in cfg["real"]["suites"] else set()) | ({"electricity"} if "electricity_time" in cfg["real"]["suites"] else set())
    for d in sorted(need):
        X, y = openml_suite.load(d); print(f"{d:18s} n={len(y):7d} d={X.shape[1]:4d} prevalence={y.mean():.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/default.yaml")
    main(ap.parse_args().config)
