#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)

def finite_array(df, col):
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return x

def fd_bins(x):
    """Freedman–Diaconis bin rule with safeguards for small n."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return 1
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 0:
        return min(10, max(1, int(np.sqrt(n))))
    bw = 2 * iqr / (n ** (1/3))
    if bw <= 0:
        return min(10, max(1, int(np.sqrt(n))))
    bins = int(np.ceil((x.max() - x.min()) / bw)) if x.max() > x.min() else 1
    return int(np.clip(bins, 1, 30))

def plot_ecdf(x, xlabel, title, outpath):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    y = np.arange(1, n + 1) / n
    plt.figure(figsize=(5.2, 3.6))
    plt.step(x, y, where="post")
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.title(f"{title} (n={n})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def plot_points(x, xlabel, title, outpath):
    x = np.asarray(x, float)
    n = len(x)
    # small jitter on y just to separate points visually
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.02, size=n)
    plt.figure(figsize=(5.2, 2.6))
    plt.scatter(x, y, s=30)
    plt.yticks([])
    plt.xlabel(xlabel)
    plt.title(f"{title} (n={n})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def plot_strip_two_groups(x0, x1, label0, label1, ylabel, title, outpath):
    x0 = np.asarray(x0, float); x1 = np.asarray(x1, float)
    rng = np.random.default_rng(0)
    y0 = 0 + rng.normal(0.0, 0.06, size=len(x0))
    y1 = 1 + rng.normal(0.0, 0.06, size=len(x1))

    plt.figure(figsize=(5.0, 3.2))
    if len(x0): plt.scatter(x0, y0, s=35, label=label0)
    if len(x1): plt.scatter(x1, y1, s=35, label=label1)
    plt.yticks([0, 1], [label0, label1])
    plt.xlabel(ylabel)
    plt.title(title + f" (n0={len(x0)}, n1={len(x1)})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curv_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", default="kappa_norm",
                    help="Which metric to plot: kappa_norm (default) or kappa.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    df = pd.read_csv(args.curv_csv)

    metric = args.metric
    if metric not in df.columns:
        # fallback to kappa if kappa_norm isn't present
        if "kappa" in df.columns:
            metric = "kappa"
        else:
            raise SystemExit("ERROR: curv_csv must contain kappa_norm or kappa.")

    x = finite_array(df, metric)
    if len(x) == 0:
        raise SystemExit(f"ERROR: No finite values for {metric}.")

    # 1) ECDF (best for small n)
    plot_ecdf(
        x,
        xlabel=metric,
        title=f"Cohort distribution of {metric}",
        outpath=Path(args.outdir) / f"{metric}_ecdf.png",
    )

    # 2) point plot (shows actual sample size clearly)
    plot_points(
        x,
        xlabel=metric,
        title=f"Cohort values of {metric}",
        outpath=Path(args.outdir) / f"{metric}_points.png",
    )

    # 3) histogram (optional, but adaptive bins)
    bins = fd_bins(x)
    plt.figure(figsize=(5.2, 3.6))
    plt.hist(x, bins=bins)
    plt.xlabel(metric)
    plt.ylabel("count")
    plt.title(f"{metric} histogram (adaptive bins={bins}, n={len(x)})")
    plt.tight_layout()
    plt.savefig(Path(args.outdir) / f"{metric}_hist_adaptive.png", dpi=220)
    plt.close()

    # 4) ABX stratification: use strip plot (NOT boxplot for tiny n)
    if "abx_any" in df.columns:
        df["abx_any"] = pd.to_numeric(df["abx_any"], errors="coerce").fillna(0).astype(int)
        d0 = df[df["abx_any"] == 0]
        d1 = df[df["abx_any"] == 1]
        x0 = finite_array(d0, metric)
        x1 = finite_array(d1, metric)

        if len(x0) + len(x1) > 0:
            plot_strip_two_groups(
                x0, x1,
                label0="No ABX",
                label1="ABX ever",
                ylabel=metric,
                title=f"{metric} stratified by antibiotics",
                outpath=Path(args.outdir) / f"{metric}_abx_strip.png",
            )

    # summary table with counts (crucial for transparency)
    summ = {
        "metric": metric,
        "n_subjects_total": int(df["subject"].nunique()) if "subject" in df.columns else int(len(df)),
        "n_finite_metric": int(len(x)),
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)) if len(x) >= 4 else float(np.min(x)),
        "p75": float(np.percentile(x, 75)) if len(x) >= 4 else float(np.max(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    pd.DataFrame([summ]).to_csv(Path(args.outdir) / f"{metric}_summary.csv", index=False)

    print("Wrote ECDF/points/plots to:", args.outdir)


if __name__ == "__main__":
    main()
