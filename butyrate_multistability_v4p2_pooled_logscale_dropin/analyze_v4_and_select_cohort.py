#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p: str) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarize per-subject R2, select cohort, and plot histograms.")
    ap.add_argument("--fit_v4", required=True, help="fit_summary_v4_cf.csv")
    ap.add_argument("--global_json", default=None, help="Optional global_fit_v4_cf.json (accepted for drop-in compatibility)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--r2_thresh", type=float, default=0.40)
    ap.add_argument("--meta_csv", default=None, help="Optional metadata CSV for stratification summaries (ABX/diet as labels)")
    args = ap.parse_args()

    outdir = safe_mkdir(args.outdir)

    fit = pd.read_csv(args.fit_v4)
    # detect subject col
    subj_col = None
    for c in fit.columns:
        if "subject" in str(c).lower():
            subj_col = c
            break
    if subj_col is None:
        raise SystemExit("ERROR: Could not find subject column in fit_v4.")

    if "R2_logspace" in fit.columns:
        r2_df = fit[[subj_col, "R2_logspace"]].copy()
        # if multiple rows per subject, keep best or mean; here mean
        r2_df = r2_df.groupby(subj_col, as_index=False)["R2_logspace"].mean()
        counts = fit.groupby(subj_col).size().reset_index(name="n_aligned")
        r2_df = r2_df.merge(counts, on=subj_col, how="left")
    else:
        # If fit_v4 is actually predictions-like, compute R2 from z_obs/z_hat or z_next/z_hat
        z_obs_col = "z_obs" if "z_obs" in fit.columns else ("z_next" if "z_next" in fit.columns else None)
        z_hat_col = "z_hat" if "z_hat" in fit.columns else None
        if z_obs_col is None or z_hat_col is None:
            raise SystemExit("ERROR: fit_v4 must contain R2_logspace OR (z_obs/z_hat) or (z_next/z_hat).")

        rows = []
        for sid, g in fit.groupby(subj_col, sort=False):
            y = pd.to_numeric(g[z_obs_col], errors="coerce").to_numpy(dtype=float)
            yh = pd.to_numeric(g[z_hat_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(y) & np.isfinite(yh)
            y, yh = y[m], yh[m]
            if len(y) < 2:
                continue
            ss_res = float(np.sum((y - yh) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = np.nan if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
            rows.append((sid, r2, len(y)))
        r2_df = pd.DataFrame(rows, columns=[subj_col, "R2_logspace", "n_aligned"])

    r2_vals = pd.to_numeric(r2_df["R2_logspace"], errors="coerce").dropna().to_numpy()
    summary = {
        "n_subjects": int(len(r2_df)),
        "mean_R2": float(np.nanmean(r2_vals)) if r2_vals.size else np.nan,
        "median_R2": float(np.nanmedian(r2_vals)) if r2_vals.size else np.nan,
        "p75_R2": float(np.nanpercentile(r2_vals, 75)) if r2_vals.size else np.nan,
        "p90_R2": float(np.nanpercentile(r2_vals, 90)) if r2_vals.size else np.nan,
        "n_ge_0p30": int(np.sum(r2_vals >= 0.30)) if r2_vals.size else 0,
        "n_ge_0p40": int(np.sum(r2_vals >= 0.40)) if r2_vals.size else 0,
        "n_ge_0p50": int(np.sum(r2_vals >= 0.50)) if r2_vals.size else 0,
    }
    pd.DataFrame([summary]).to_csv(outdir / "r2_summary.csv", index=False)

    cohort = r2_df[pd.to_numeric(r2_df["R2_logspace"], errors="coerce") >= float(args.r2_thresh)].copy()
    cohort = cohort.sort_values("R2_logspace", ascending=False)
    cohort.to_csv(outdir / "cohort_subjects.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.hist(r2_vals, bins=20)
    plt.xlabel("R2 (log-space)")
    plt.ylabel("count")
    plt.title("R2 distribution")
    plt.tight_layout()
    plt.savefig(outdir / "r2_hist_v4.png", dpi=200)
    plt.close()

    # Optional stratification summary (very lightweight; no proxy scoring)
    if args.meta_csv:
        meta = pd.read_csv(args.meta_csv)
        meta_subj = None
        for c in meta.columns:
            if "subject" in str(c).lower():
                meta_subj = c
                break
        if meta_subj is not None:
            # crude antibiotic col detection
            abx_col = None
            for c in meta.columns:
                if "antibiotic" in str(c).lower():
                    abx_col = c
                    break
            strat_rows = []
            if abx_col is not None:
                # map subject -> any_yes
                s_any = meta.groupby(meta_subj)[abx_col].apply(
                    lambda s: any(str(x).strip().lower() in ["yes", "y", "true", "1", "t"] for x in s.dropna())
                ).reset_index(name="any_antibiotic_yes")
                merged = r2_df.merge(s_any, left_on=subj_col, right_on=meta_subj, how="left")
                for flag, gg in merged.groupby("any_antibiotic_yes", dropna=False):
                    vals = pd.to_numeric(gg["R2_logspace"], errors="coerce").dropna().to_numpy()
                    strat_rows.append({
                        "group": f"any_antibiotic_yes={flag}",
                        "n": int(len(vals)),
                        "mean_R2": float(np.mean(vals)) if len(vals) else np.nan,
                        "median_R2": float(np.median(vals)) if len(vals) else np.nan,
                    })
                pd.DataFrame(strat_rows).to_csv(outdir / "r2_stratified_summary.csv", index=False)

    print(f"Wrote: {outdir/'r2_summary.csv'}")
    print(f"Wrote: {outdir/'cohort_subjects.csv'}")
    print(f"Wrote: {outdir/'r2_hist_v4.png'}")


if __name__ == "__main__":
    main()
