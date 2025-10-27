#!/usr/bin/env python3
# make_globals_from_fits.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEEP_AS_GLOBAL = ["r_C","u","k_B","g","K_B","d0","chi","H_I","eta","H_on","H_off","eps","k_on","k_off","p_min","alpha","beta"]
DEFAULTS = {
  "r_C":0.7,"u":0.05,"k_B":0.08,"g":1.0,"K_B":0.6,"d0":0.08,"chi":0.4,"H_I":0.5,"eta":6.0,
  "H_on":0.3,"H_off":0.7,"eps":0.05,"k_on":0.2,"k_off":0.2,"p_min":0.2,"alpha":0.4,"beta":0.4
}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_json", default="globals_cohort.json")
    ap.add_argument("--max_bound_hits", type=int, default=6)
    ap.add_argument("--min_r2b", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    # Exclude poor fits
    mask = (df.get("bound_hits", 0) <= args.max_bound_hits) & (df.get("R2_B", -1e9) >= args.min_r2b)
    use = df.loc[mask].copy()
    fixed = {}

    for k in KEEP_AS_GLOBAL:
        if k in use.columns and use[k].notna().any():
            fixed[k] = float(use[k].median())
        else:
            fixed[k] = DEFAULTS.get(k)

    # Always keep these constants
    fixed["K_M"] = 1.0
    fixed["K_C"] = 1.0
    fixed["n"] = 2.0

    Path(args.out_json).write_text(json.dumps({"fixed": fixed}, indent=2))
    print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    main()
