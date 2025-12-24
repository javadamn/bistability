#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        raise RuntimeError(f"Command failed with code {r.returncode}")
    return r


def summarize_fit(outdir: Path):
    fit_sum = pd.read_csv(outdir / "fit_summary_v4_cf.csv")
    r2 = pd.to_numeric(fit_sum["R2_logspace"], errors="coerce").dropna().to_numpy()
    summary = {
        "n_subjects": int(fit_sum["subject"].nunique()) if "subject" in fit_sum.columns else int(len(fit_sum)),
        "mean_R2": float(np.mean(r2)) if len(r2) else np.nan,
        "median_R2": float(np.median(r2)) if len(r2) else np.nan,
        "p75_R2": float(np.percentile(r2, 75)) if len(r2) else np.nan,
        "p90_R2": float(np.percentile(r2, 90)) if len(r2) else np.nan,
        "n_ge_0p3": int(np.sum(r2 >= 0.3)) if len(r2) else 0,
        "n_ge_0p4": int(np.sum(r2 >= 0.4)) if len(r2) else 0,
        "n_ge_0p5": int(np.sum(r2 >= 0.5)) if len(r2) else 0,
    }
    with open(outdir / "global_fit_v4_cf.json", "r") as f:
        gj = json.load(f)
    best = gj.get("best", {})
    summary.update({
        "lambda_B": best.get("lambda_B", np.nan),
        "alpha_F": best.get("alpha_F", np.nan),
        "alpha_A": best.get("alpha_A", np.nan),
        "k_LB": best.get("k_LB", np.nan),
        "loss_huber": best.get("loss_huber", np.nan),
        "model": gj.get("model", "unknown"),
        "lags": str(gj.get("lags", {})),
        "within_subject_center": gj.get("within_subject_center", None),
    })
    return summary, fit_sum


def main():
    ap = argparse.ArgumentParser(description="Run a small model sweep and compare cohort-level fit.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fit_script", default="fit_butyrate_only_aux_v4_cf.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--use_smoothed", action="store_true")
    ap.add_argument("--min_obs", type=int, default=6)
    ap.add_argument("--min_sd_zB", type=float, default=0.25)
    ap.add_argument("--lambda_grid", type=str, default="0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3,0.45,0.6")
    ap.add_argument("--nonneg_k", action="store_true")
    ap.add_argument("--centered", action="store_true", help="Use within-subject centering (recommended).")
    ap.add_argument("--models", type=str, default="full,noAL,noL",
                    help="Comma list among: full,noAL,noL,noA,noF")
    ap.add_argument("--lags", type=str, default="1:0:1,0:0:1",
                    help="Comma list of lag triplets lagF:lagA:lagL, e.g. 1:0:1,0:0:1")
    args = ap.parse_args()

    out = safe_mkdir(args.outdir)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lags = [t.strip() for t in args.lags.split(",") if t.strip()]

    all_rows = []
    # Also store R2 distributions for plotting
    r2_store = {}

    for m in models:
        for lag_trip in lags:
            lagF, lagA, lagL = lag_trip.split(":")
            tag = f"model_{m}__lags_{lagF}_{lagA}_{lagL}__centered_{int(args.centered)}"
            od = out / tag
            od.mkdir(parents=True, exist_ok=True)

            cmd = ["python", args.fit_script,
                   "--csv", args.csv,
                   "--outdir", str(od),
                   "--min_obs", str(args.min_obs),
                   "--min_sd_zB", str(args.min_sd_zB),
                   "--lambda_grid", args.lambda_grid,
                   "--lag_F", str(lagF),
                   "--lag_A", str(lagA),
                   "--lag_L", str(lagL),
                   "--model", m]
            if args.use_smoothed:
                cmd.append("--use_smoothed")
            if args.nonneg_k:
                cmd.append("--nonneg_k")
            if args.centered:
                cmd.append("--within_subject_center")
            else:
                cmd.append("--no_within_subject_center")

            run(cmd)

            summ, fit_sum = summarize_fit(od)
            summ["run_tag"] = tag
            all_rows.append(summ)

            r2_store[tag] = pd.to_numeric(fit_sum["R2_logspace"], errors="coerce").dropna().to_numpy()

    comp = pd.DataFrame(all_rows)
    comp = comp.sort_values(["median_R2", "n_ge_0p4", "mean_R2"], ascending=False)
    comp.to_csv(out / "model_comparison_summary.csv", index=False)

    # Plot: median R2 by model-run
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(comp)), comp["median_R2"].to_numpy())
    plt.xticks(np.arange(len(comp)), comp["run_tag"].to_numpy(), rotation=75, ha="right", fontsize=7)
    plt.ylabel("Median R2 (log-space)")
    plt.title("Model sweep: median R2 across subjects")
    plt.tight_layout()
    plt.savefig(out / "model_sweep_medianR2.png", dpi=200)
    plt.close()

    # Plot: overlapping histograms for top 3 runs
    top = comp.head(3)["run_tag"].tolist()
    plt.figure(figsize=(7, 4))
    for tag in top:
        plt.hist(r2_store[tag], bins=25, alpha=0.4, label=tag)
    plt.xlabel("R2 (log-space)")
    plt.ylabel("count")
    plt.title("Top 3 runs: R2 distributions")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "model_sweep_top3_r2_hists.png", dpi=200)
    plt.close()

    print(f"Wrote: {out/'model_comparison_summary.csv'}")
    print(f"Wrote: {out/'model_sweep_medianR2.png'}")
    print(f"Wrote: {out/'model_sweep_top3_r2_hists.png'}")
    print("Top run:", comp.iloc[0]["run_tag"])


if __name__ == "__main__":
    main()
