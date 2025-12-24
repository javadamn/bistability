#!/usr/bin/env python3
"""
Estimate effective potential landscapes for butyrate dynamics.

v3 (small-N robust):
- Avoids binning failures by fitting a smooth drift function dx(x) with low-degree polynomial.
- Works well even with ~5-10 transitions per subject.
- Uses (B_prev,B_next) if available; else (z_prev,z_obs).
- Writes: potential_summary.csv, potential_debug.csv, and per-subject potential PNGs.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def polyfit_drift(x_prev, x_next, degree=1):
    x_prev = np.asarray(x_prev, dtype=float)
    dx = np.asarray(x_next, dtype=float) - x_prev
    m = np.isfinite(x_prev) & np.isfinite(dx)
    x = x_prev[m]
    y = dx[m]
    if len(x) < max(4, degree + 2):
        return None, None, None
    # center/scale x to reduce conditioning issues
    x_mu = float(np.mean(x))
    x_sd = float(np.std(x)) if float(np.std(x)) > 1e-12 else 1.0
    xs = (x - x_mu) / x_sd
    coef = np.polyfit(xs, y, deg=degree)  # highest power first
    return coef, x_mu, x_sd


def drift_eval(coef, x, x_mu, x_sd):
    xs = (x - x_mu) / x_sd
    return np.polyval(coef, xs)


def potential_from_drift(x_grid, drift):
    # V = -∫ drift dx (numerical cumulative)
    dx = np.gradient(x_grid)
    V = -np.cumsum(drift * dx)
    V = V - float(np.min(V))
    return V


def count_minima(x, V):
    if len(V) < 5:
        return [], []
    dV = np.gradient(V)
    idx = []
    for i in range(1, len(dV) - 1):
        if dV[i - 1] < 0 and dV[i + 1] > 0:
            idx.append(i)
    return idx, [float(x[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--subjects_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_pairs", type=int, default=5)
    ap.add_argument("--degree", type=int, default=1, choices=[1, 2],
                    help="Polynomial degree for drift fit (1=linear, 2=quadratic). Start with 1 for stability.")
    ap.add_argument("--grid_n", type=int, default=60, help="Grid points for potential curve.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    P = pd.read_csv(args.pred_csv)
    S = pd.read_csv(args.subjects_csv)

    if "subject" not in P.columns:
        raise SystemExit("ERROR: predictions CSV must contain 'subject'.")
    P["subject"] = P["subject"].astype(str).str.strip()

    subj_list = S.iloc[:, 0].astype(str).str.strip().tolist()
    subj_list = [s for s in subj_list if s != ""]
    if len(subj_list) == 0:
        raise SystemExit("ERROR: subjects_csv has no subject IDs.")

    # choose working variables
    if ("B_prev" in P.columns) and ("B_next" in P.columns):
        mode = "B"
        P["x_prev"] = pd.to_numeric(P["B_prev"], errors="coerce")
        P["x_next"] = pd.to_numeric(P["B_next"], errors="coerce")
        x_label = "Butyrate B"
    elif ("z_prev" in P.columns) and ("z_obs" in P.columns):
        mode = "z"
        P["x_prev"] = pd.to_numeric(P["z_prev"], errors="coerce")
        P["x_next"] = pd.to_numeric(P["z_obs"], errors="coerce")
        x_label = "z = log1p(B)"
    else:
        raise SystemExit("ERROR: predictions CSV must have either (B_prev,B_next) or (z_prev,z_obs).")

    debug_rows = []
    summary_rows = []

    for sid in subj_list:
        g = P[P["subject"] == sid].copy()
        n_rows = int(len(g))
        x_prev = g["x_prev"].to_numpy(dtype=float)
        x_next = g["x_next"].to_numpy(dtype=float)
        n_pairs = int(np.sum(np.isfinite(x_prev) & np.isfinite(x_next)))

        if n_rows == 0:
            debug_rows.append({"subject": sid, "n_rows": n_rows, "n_pairs": n_pairs,
                               "status": "skipped", "reason": "no_rows_for_subject"})
            continue
        if n_pairs < args.min_pairs:
            debug_rows.append({"subject": sid, "n_rows": n_rows, "n_pairs": n_pairs,
                               "status": "skipped", "reason": f"too_few_pairs({n_pairs}<{args.min_pairs})"})
            continue

        coef, x_mu, x_sd = polyfit_drift(x_prev, x_next, degree=args.degree)
        if coef is None:
            debug_rows.append({"subject": sid, "n_rows": n_rows, "n_pairs": n_pairs,
                               "status": "skipped", "reason": "polyfit_failed_smallN"})
            continue

        # grid over observed range
        x_ok = x_prev[np.isfinite(x_prev)]
        lo, hi = float(np.min(x_ok)), float(np.max(x_ok))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
            debug_rows.append({"subject": sid, "n_rows": n_rows, "n_pairs": n_pairs,
                               "status": "skipped", "reason": "degenerate_range"})
            continue

        x_grid = np.linspace(lo, hi, int(args.grid_n))
        drift = drift_eval(coef, x_grid, x_mu, x_sd)
        V = potential_from_drift(x_grid, drift)

        idx_min, minima_pos = count_minima(x_grid, V)

        summary_rows.append({
            "subject": sid,
            "mode": mode,
            "n_rows": n_rows,
            "n_pairs": n_pairs,
            "degree": int(args.degree),
            "n_minima": int(len(idx_min)),
            "minima_positions": minima_pos,
            "coef": coef.tolist(),
            "x_center": float(x_mu),
            "x_scale": float(x_sd),
        })

        # plot
        plt.figure(figsize=(4.4, 3.3))
        plt.plot(x_grid, V, "-")
        for i in idx_min:
            plt.scatter([x_grid[i]], [V[i]])
        plt.xlabel(x_label)
        plt.ylabel("Effective potential V")
        plt.title(f"{sid}  (minima={len(idx_min)}, deg={args.degree})")
        plt.tight_layout()
        plt.savefig(Path(args.outdir) / f"{sid}_potential.png", dpi=220)
        plt.close()

        debug_rows.append({"subject": sid, "n_rows": n_rows, "n_pairs": n_pairs,
                           "status": "ok", "reason": ""})

    pd.DataFrame(debug_rows).to_csv(Path(args.outdir) / "potential_debug.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(Path(args.outdir) / "potential_summary.csv", index=False)

    print(f"Wrote: {Path(args.outdir) / 'potential_summary.csv'}  (rows={len(summary_rows)})")
    print(f"Wrote: {Path(args.outdir) / 'potential_debug.csv'}")


if __name__ == "__main__":
    main()
