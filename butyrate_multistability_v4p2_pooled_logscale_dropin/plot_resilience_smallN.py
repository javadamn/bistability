#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curv_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", default="kappa_norm", choices=["kappa", "kappa_norm"])
    ap.add_argument("--top_n", type=int, default=50,
                    help="Max subjects to plot (sorted by metric). For n=4, keep default.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    df = pd.read_csv(args.curv_csv)

    metric = args.metric
    lo = metric + "_lo"
    hi = metric + "_hi"

    for c in [metric, lo, hi]:
        if c not in df.columns:
            raise SystemExit(f"ERROR: missing column '{c}' in {args.curv_csv}")

    if "subject" not in df.columns:
        raise SystemExit("ERROR: missing 'subject' column")

    df["subject"] = df["subject"].astype(str)
    if "abx_any" in df.columns:
        df["abx_any"] = pd.to_numeric(df["abx_any"], errors="coerce").fillna(0).astype(int)
    else:
        df["abx_any"] = 0

    # keep only finite metric
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df[lo] = pd.to_numeric(df[lo], errors="coerce")
    df[hi] = pd.to_numeric(df[hi], errors="coerce")

    df = df[np.isfinite(df[metric])].copy()
    if len(df) == 0:
        raise SystemExit(f"ERROR: no finite {metric} values to plot.")

    # sort for clean plotting
    df = df.sort_values(metric, ascending=True).head(args.top_n)

    # --- Figure: dot + CI by subject (sorted) ---
    y = np.arange(len(df))
    x = df[metric].to_numpy(float)

    # error bars: if CI missing, fall back to 0
    xlo = df[lo].to_numpy(float)
    xhi = df[hi].to_numpy(float)
    err_left = np.where(np.isfinite(xlo), x - xlo, 0.0)
    err_right = np.where(np.isfinite(xhi), xhi - x, 0.0)
    xerr = np.vstack([np.clip(err_left, 0, None), np.clip(err_right, 0, None)])

    plt.figure(figsize=(6.2, max(2.6, 0.45 * len(df) + 1.2)))
    # color by ABX
    colors = np.where(df["abx_any"].to_numpy(int) == 1, "tab:orange", "tab:blue")
    for i in range(len(df)):
        plt.errorbar(x[i], y[i], xerr=xerr[:, i:i+1], fmt="o", capsize=3)

    # overlay points with ABX color
    plt.scatter(x, y, s=60, c=colors)

    plt.yticks(y, df["subject"].tolist())
    plt.axvline(np.median(x), linestyle="--", linewidth=1)
    plt.xlabel(f"{metric} (with 95% bootstrap CI where available)")
    plt.title(f"Resilience metric by subject (n={len(df)})")
    plt.tight_layout()
    plt.savefig(Path(args.outdir) / f"{metric}_by_subject_ci.png", dpi=220)
    plt.close()

    # --- Figure: ABX group means (descriptive; tiny n) ---
    g0 = df[df["abx_any"] == 0][metric].to_numpy(float)
    g1 = df[df["abx_any"] == 1][metric].to_numpy(float)

    plt.figure(figsize=(4.8, 3.2))
    # jitter points
    rng = np.random.default_rng(0)
    if len(g0):
        plt.scatter(rng.normal(0, 0.04, size=len(g0)), g0, s=60, label=f"No ABX (n={len(g0)})")
    if len(g1):
        plt.scatter(1 + rng.normal(0, 0.04, size=len(g1)), g1, s=60, label=f"ABX ever (n={len(g1)})")

    plt.xticks([0, 1], ["No ABX", "ABX ever"])
    plt.ylabel(metric)
    plt.title("Antibiotics stratification (descriptive; small n)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.outdir) / f"{metric}_abx_points.png", dpi=220)
    plt.close()

    # save table used for plotting
    df.to_csv(Path(args.outdir) / f"{metric}_plot_table.csv", index=False)
    print("Wrote small-N plots to:", args.outdir)


if __name__ == "__main__":
    main()
