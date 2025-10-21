#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add RMSE/NRMSE to an existing fit_summary_all_subjects.csv, and shortlist subjects.

Usage:
  python postprocess_fit_summary.py \
    --summary out_prebiotic/fits/fit_summary_all_subjects.csv \
    --timeseries out_prebiotic/prebiotic_scfa_timeseries.csv \
    --out out_prebiotic/fits/fit_summary_with_errors.csv \
    --shortlist out_prebiotic/fits/shortlist_subjects.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--rmse_norm", choices=["range","mean"], default="range",
                    help="NRMSE denominator: 'range' (default) or 'mean'")
    args = ap.parse_args()

    summ = pd.read_csv(args.summary)
    tidy = pd.read_csv(args.timeseries)

    # Count points per subject (usable butyrate observations)
    if "subject_id" not in tidy.columns:
        raise SystemExit("timeseries file must have subject_id column.")
    if "time_hr" not in tidy.columns or "butyrate" not in tidy.columns:
        raise SystemExit("timeseries file must have time_hr and butyrate columns.")

    counts = (tidy.dropna(subset=["time_hr","butyrate"])
                  .groupby("subject_id")["butyrate"]
                  .agg(N="count", mean="mean", vmin="min", vmax="max")
                  .reset_index())

    df = summ.merge(counts, on="subject_id", how="left")

    # Coerce numeric
    for c in ["cost","N","vmin","vmax","mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # RMSE and NRMSE
    # least_squares reports cost = 0.5 * sum(residual^2)
    df["RMSE"] = np.sqrt(2.0 * df["cost"] / df["N"])
    denom = (df["vmax"] - df["vmin"]) if args.rmse_norm == "range" else df["mean"]
    df["NRMSE"] = df["RMSE"] / denom.replace(0, np.nan)

    # Derived hysteresis metrics if present
    if {"p_high","p_low","H_on","H_off"}.issubset(df.columns):
        df["p_gap"] = df["p_high"] - df["p_low"]
        df["H_band"] = df["H_off"] - df["H_on"]

    # Heuristic flags
    med_cost = df["cost"].median(skipna=True)
    df["flag_high_cost"]   = df["cost"] > 2.0 * med_cost
    df["flag_small_p_gap"] = df.get("p_gap", np.nan) < 0.2
    df["flag_narrow_band"] = df.get("H_band", np.nan) < 0.05
    df["flag_bad_rmse"]    = df["NRMSE"] > 0.35
    df["flag_any"] = df[[c for c in df.columns if c.startswith("flag_")]].any(axis=1)

    df = df.sort_values(["success","NRMSE","cost"], ascending=[False, True, True])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Shortlist: good fits w/ decent hysteresis structure
    keep = df[
        (df["success"] == True) &
        (df["NRMSE"] <= 0.35) &
        (df.get("p_gap", 1.0) >= 0.5) &
        (df.get("H_band", 1.0) >= 0.08)
    ].copy()
    keep.to_csv(args.shortlist, index=False)

    print(f"[write] {args.out}")
    print(f"[write] {args.shortlist}")
    print(f"[info] kept {len(keep)}/{len(df)} subjects for hysteresis analysis.")

if __name__ == "__main__":
    main()
