#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hysteresis Re-Shortlist & Group Summary

What this script does
---------------------
1) Loads `fit_summary_all_subjects_refit.csv` (post-refit results).
2) Applies stricter keeper thresholds:
   - NRMSE ≤ 0.40
   - p_gap ≥ 1.0
   - H_band ≥ 0.10
   - 0.50 ≤ u ≤ 0.85  (h^-1)
   - 0.5 ≤ tau_q ≤ 20 (h)
3) Writes final shortlist:  group_outputs/final_shortlist.csv
4) Computes group medians & IQRs: group_outputs/group_summary.csv
5) Produces figures:
   - group_outputs/violin_H_on_off.png
   - group_outputs/density_p_gap.png
   - group_outputs/violin_tau_q.png

Usage
-----
python hysteresis_group_summary.py \
  --input fit_summary_all_subjects_refit.csv \
  --outdir group_outputs
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_refit_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce types
    num_cols = [
        "cost","RMSE","NRMSE","r_max","K_M","c","d","g","u",
        "p_low","p_high","H_on","H_off","tau_q","alpha","p_gap","H_band"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "success" in df.columns:
        df["success"] = df["success"].astype(bool)
    return df

def shortlist(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (df.get("success", True) == True) &
        (df["NRMSE"] <= 0.40) &
        (df["p_gap"] >= 1.0) &
        (df["H_band"] >= 0.10) &
        (df["u"].between(0.50, 0.85, inclusive="both")) &
        (df["tau_q"].between(0.5, 20.0, inclusive="both"))
    )
    out = df.loc[keep].copy().sort_values(
        ["NRMSE","p_gap","H_band"], ascending=[True, False, False]
    )
    return out

def group_summary(short: pd.DataFrame) -> pd.DataFrame:
    def iqr(arr):
        return np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)
    keys = [
        "NRMSE","p_low","p_high","p_gap","H_on","H_off","H_band",
        "tau_q","u","g","d","r_max","K_M","alpha"
    ]
    rows = []
    for k in keys:
        if k in short.columns:
            x = short[k].astype(float)
            med = np.nanmedian(x)
            q1  = np.nanpercentile(x, 25)
            q3  = np.nanpercentile(x, 75)
            rows.append({"metric": k, "median": med, "q1": q1, "q3": q3, "IQR": q3 - q1})
    return pd.DataFrame(rows)

def make_figures(short: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # H_on / H_off violin
    data, labels = [], []
    if "H_on" in short.columns and short["H_on"].notna().any():
        data.append(short["H_on"].dropna().values); labels.append("H_on")
    if "H_off" in short.columns and short["H_off"].notna().any():
        data.append(short["H_off"].dropna().values); labels.append("H_off")
    if data:
        plt.figure(figsize=(6,4))
        plt.violinplot(data, showmeans=True, showmedians=False)
        plt.xticks(range(1, len(labels)+1), labels)
        plt.ylabel("Threshold value")
        plt.title("Hysteresis thresholds")
        plt.tight_layout()
        plt.savefig(outdir / "violin_H_on_off.png", dpi=200)
        plt.close()

    # p_gap density-style histogram
    if "p_gap" in short.columns and short["p_gap"].notna().any():
        plt.figure(figsize=(6,4))
        plt.hist(short["p_gap"].dropna().values, bins=15, density=True)
        plt.xlabel("p_high - p_low")
        plt.ylabel("Density")
        plt.title("Production gap distribution")
        plt.tight_layout()
        plt.savefig(outdir / "density_p_gap.png", dpi=200)
        plt.close()

    # tau_q violin
    if "tau_q" in short.columns and short["tau_q"].notna().any():
        plt.figure(figsize=(6,4))
        plt.violinplot(short["tau_q"].dropna().values, showmeans=True, showmedians=False)
        plt.xticks([1], ["tau_q (h)"])
        plt.ylabel("Hours")
        plt.title("Memory timescale (tau_q)")
        plt.tight_layout()
        plt.savefig(outdir / "violin_tau_q.png", dpi=200)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="fit_summary_all_subjects_refit.csv")
    ap.add_argument("--outdir", default="group_outputs", help="Output directory")
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_refit_summary(inp)
    print(f"[info] loaded {len(df)} rows; success={int(df.get('success', pd.Series([True]*len(df))).sum())}")

    short = shortlist(df)
    short.to_csv(outdir / "final_shortlist.csv", index=False)
    print(f"[write] {outdir / 'final_shortlist.csv'}  (kept {len(short)}/{len(df)})")

    gs = group_summary(short)
    gs.to_csv(outdir / "group_summary.csv", index=False)
    print(f"[write] {outdir / 'group_summary.csv'}")

    make_figures(short, outdir)
    print(f"[write] figures in {outdir}/ (H_on/H_off violin, p_gap density, tau_q violin)")

if __name__ == "__main__":
    main()
