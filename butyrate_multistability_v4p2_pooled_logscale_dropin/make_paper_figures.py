#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess


def safe_mkdir(p):
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        print(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(r.returncode)


def main():
    ap = argparse.ArgumentParser(description="Generate paper-ready figures from a chosen run.")
    ap.add_argument("--run_dir", required=True, help="One sweep run folder containing predictions/fit_summary/params/global_json")
    ap.add_argument("--meta_csv", required=False, default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--r2_thresh", type=float, default=0.40)
    ap.add_argument("--exemplars", type=str, default="", help="Comma-separated subject IDs; if empty uses top 2 by R2.")
    ap.add_argument("--center", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--h_amp", type=float, default=0.2)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = safe_mkdir(args.outdir)

    # 1) cohort selection + histogram
    run(["python", "analyze_v4_and_select_cohort.py",
         "--fit_v4", str(run_dir / "fit_summary_v4_cf.csv"),
         "--global_json", str(run_dir / "global_fit_v4_cf.json"),
         "--outdir", str(out / "cohort"),
         "--r2_thresh", str(args.r2_thresh)] + (["--meta_csv", args.meta_csv] if args.meta_csv else []))

    # pick exemplars
    cohort = pd.read_csv(out / "cohort" / "cohort_subjects.csv")
    if args.exemplars.strip():
        exemplars = [x.strip() for x in args.exemplars.split(",") if x.strip()]
    else:
        # take top 2 by R2
        subj_col = cohort.columns[0]
        exemplars = cohort.sort_values(cohort.columns[1], ascending=False)[subj_col].astype(str).head(2).tolist()

    # write a small exemplars file for plotting
    ex_df = pd.DataFrame({"subject": exemplars})
    ex_df.to_csv(out / "exemplars.csv", index=False)

    # 2) time-series panels for exemplars
    run(["python", "plot_timeseries_panels.py",
         "--pred_v4", str(run_dir / "predictions_v4_cf.csv"),
         "--subjects_csv", str(out / "exemplars.csv"),
         "--max_plots", str(len(exemplars)),
         "--outdir", str(out / "panels")] + (["--meta_csv", args.meta_csv] if args.meta_csv else []))

    # 3) multistability figures for exemplars (batch)
    run(["python", "multistability_analysis.py",
         "--params_csv", str(run_dir / "params_v4_cf.csv"),
         "--pred_v4", str(run_dir / "predictions_v4_cf.csv"),
         "--global_json", str(run_dir / "global_fit_v4_cf.json"),
         "--subjects_csv", str(out / "exemplars.csv"),
         "--center", str(args.center),
         "--gap", str(args.gap),
         "--h_amp", str(args.h_amp),
         "--outdir", str(out / "multistability")])

    print("Figures written under:", out)
    print("Exemplars:", exemplars)


if __name__ == "__main__":
    main()
