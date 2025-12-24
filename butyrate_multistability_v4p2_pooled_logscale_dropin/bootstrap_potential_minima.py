#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def fit_poly_drift(x_prev, x_next, degree=1):
    x_prev = np.asarray(x_prev, dtype=float)
    dx = np.asarray(x_next, dtype=float) - x_prev
    m = np.isfinite(x_prev) & np.isfinite(dx)
    x = x_prev[m]
    y = dx[m]
    if len(x) < max(4, degree + 2):
        return None

    mu = float(np.mean(x))
    sd = float(np.std(x)) if float(np.std(x)) > 1e-12 else 1.0
    xs = (x - mu) / sd
    coef = np.polyfit(xs, y, deg=degree)
    return coef, mu, sd


def eval_poly(coef, x, mu, sd):
    xs = (x - mu) / sd
    return np.polyval(coef, xs)


def potential_from_drift(x_grid, drift):
    dx = np.gradient(x_grid)
    V = -np.cumsum(drift * dx)
    V = V - float(np.min(V))
    return V


def minima_with_barrier(x, V, min_sep_frac=0.08, min_barrier_frac=0.05):
    """Return minima indices that are meaningfully separated and have a barrier."""
    if len(V) < 10:
        return []

    # candidates by local neighborhood comparison
    cand = []
    for i in range(1, len(V) - 1):
        if V[i] <= V[i - 1] and V[i] <= V[i + 1]:
            cand.append(i)
    if len(cand) <= 1:
        return cand

    # filter by separation and barrier
    rng = float(np.max(x) - np.min(x))
    if rng <= 0:
        return [cand[0]]

    min_sep = min_sep_frac * rng
    kept = [cand[0]]
    for i in cand[1:]:
        if abs(x[i] - x[kept[-1]]) >= min_sep:
            kept.append(i)

    if len(kept) <= 1:
        return [kept[0]]

    # barrier criterion: between two minima must rise by at least min_barrier_frac * range(V)
    Vrng = float(np.max(V) - np.min(V))
    min_bar = min_barrier_frac * Vrng
    final = [kept[0]]
    for a, b in zip(kept[:-1], kept[1:]):
        mid_max = float(np.max(V[min(a, b):max(a, b) + 1]))
        if mid_max - max(V[a], V[b]) >= min_bar:
            final.append(b)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--subjects_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--degree", type=int, default=1, choices=[1, 2])
    ap.add_argument("--grid_n", type=int, default=80)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--min_pairs", type=int, default=5)
    ap.add_argument("--min_sep_frac", type=float, default=0.08)
    ap.add_argument("--min_barrier_frac", type=float, default=0.05)
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    P = pd.read_csv(args.pred_csv)
    S = pd.read_csv(args.subjects_csv)

    P["subject"] = P["subject"].astype(str).str.strip()
    subjects = S.iloc[:, 0].astype(str).str.strip().tolist()

    # choose z-scale if present; else compute z from B
    if ("z_prev" in P.columns) and ("z_obs" in P.columns):
        P["x_prev"] = pd.to_numeric(P["z_prev"], errors="coerce")
        P["x_next"] = pd.to_numeric(P["z_obs"], errors="coerce")
        x_label = "z = log1p(B)"
    else:
        Bp = pd.to_numeric(P["B_prev"], errors="coerce")
        Bn = pd.to_numeric(P["B_next"], errors="coerce")
        P["x_prev"] = np.log1p(np.clip(Bp, 0, None))
        P["x_next"] = np.log1p(np.clip(Bn, 0, None))
        x_label = "z = log1p(B)"

    rows = []

    for sid in subjects:
        g = P[P["subject"] == sid].copy()
        x_prev = g["x_prev"].to_numpy(dtype=float)
        x_next = g["x_next"].to_numpy(dtype=float)
        m = np.isfinite(x_prev) & np.isfinite(x_next)
        x_prev = x_prev[m]
        x_next = x_next[m]
        n = len(x_prev)
        if n < args.min_pairs:
            rows.append({"subject": sid, "n_pairs": n, "p_bistable": np.nan, "note": "too_few_pairs"})
            continue

        lo, hi = float(np.min(x_prev)), float(np.max(x_prev))
        if (hi - lo) < 1e-8:
            rows.append({"subject": sid, "n_pairs": n, "p_bistable": np.nan, "note": "degenerate_range"})
            continue

        x_grid = np.linspace(lo, hi, args.grid_n)

        # bootstrap
        bist = 0
        for b in range(args.boot):
            idx = np.random.randint(0, n, size=n)
            coef_pack = fit_poly_drift(x_prev[idx], x_next[idx], degree=args.degree)
            if coef_pack is None:
                continue
            coef, mu, sd = coef_pack
            drift = eval_poly(coef, x_grid, mu, sd)
            V = potential_from_drift(x_grid, drift)
            mins = minima_with_barrier(x_grid, V, args.min_sep_frac, args.min_barrier_frac)
            if len(mins) >= 2:
                bist += 1

        p_bi = bist / float(args.boot)

        # also plot one “full-data” curve
        coef_pack = fit_poly_drift(x_prev, x_next, degree=args.degree)
        if coef_pack is not None:
            coef, mu, sd = coef_pack
            drift = eval_poly(coef, x_grid, mu, sd)
            V = potential_from_drift(x_grid, drift)
            mins = minima_with_barrier(x_grid, V, args.min_sep_frac, args.min_barrier_frac)

            plt.figure(figsize=(4.3, 3.2))
            plt.plot(x_grid, V, "-")
            for i in mins:
                plt.scatter([x_grid[i]], [V[i]])
            plt.xlabel(x_label)
            plt.ylabel("Effective potential V")
            plt.title(f"{sid}  p(bistable)={p_bi:.2f}")
            plt.tight_layout()
            plt.savefig(Path(args.outdir) / f"{sid}_potential_boot.png", dpi=220)
            plt.close()

        rows.append({"subject": sid, "n_pairs": n, "p_bistable": p_bi, "note": ""})

    out = pd.DataFrame(rows)
    out.to_csv(Path(args.outdir) / "bootstrap_bistability_summary.csv", index=False)
    print("Wrote:", Path(args.outdir) / "bootstrap_bistability_summary.csv")


if __name__ == "__main__":
    main()
