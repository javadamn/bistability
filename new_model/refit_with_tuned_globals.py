#!/usr/bin/env python3
# refit_with_tuned_globals.py
import subprocess
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--globals", default="globals_cohort_tuned.json")
    ap.add_argument("--outdir", default="outputs_slim_tuned")
    ap.add_argument("--n_starts", type=int, default=12)
    ap.add_argument("--max_nfev", type=int, default=350)
    ap.add_argument("--logbase", choices=["e","10"], default="10")
    ap.add_argument("--rawB", action="store_true")
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--min_obs_b", type=int, default=4)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    cmd = [
        "python", "fit_all_subjects_slim.py",
        "--csv", args.csv,
        "--globals", args.globals,
        "--outdir", args.outdir,
        "--n_starts", str(args.n_starts),
        "--max_nfev", str(args.max_nfev),
        "--min_obs_b", str(args.min_obs_b),
        "--logbase", args.logbase
    ]
    if args.rawB: cmd.append("--rawB")
    if args.save_preds: cmd.append("--save_preds")
    if args.subjects:
        cmd += ["--subjects"] + args.subjects

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
