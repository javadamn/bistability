#!/usr/bin/env python3
"""
Toy-model synthetic analysis:
- Simulate a bistable SDE (double-well potential) and a monostable OU SDE
- Reconstruct drift and effective potential from dense vs sparse samples
- Show how sparse sampling can obscure bistability

Dependencies: numpy, pandas, matplotlib only.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


# -----------------------------
# SDE definitions
# -----------------------------
def drift_bistable(z, a=0.0):
    """
    Double-well potential: V(z) = (z^2 - 1)^2 / 4 - a z
    Then dV/dz = z(z^2 - 1) - a
    Drift f(z) = -dV/dz = -z(z^2 - 1) + a = -z^3 + z + a
    """
    return -z**3 + z + a


def drift_ou(z, theta=1.0, mu=0.0):
    """Ornstein-Uhlenbeck drift: f(z) = -theta (z - mu)."""
    return -theta * (z - mu)


def simulate_em(drift_fn, z0, dt, n_steps, sigma, rng, drift_kwargs=None):
    """Euler–Maruyama simulation."""
    if drift_kwargs is None:
        drift_kwargs = {}
    z = np.zeros(n_steps + 1, dtype=float)
    z[0] = float(z0)
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        f = drift_fn(z[t], **drift_kwargs)
        z[t + 1] = z[t] + f * dt + sigma * sqrt_dt * rng.normal()
    return z


# -----------------------------
# Reconstruction: drift -> potential
# -----------------------------
def reconstruct_drift_from_samples(z, dt_eff, n_bins=30, z_clip_quantiles=(1, 99)):
    """
    Estimate drift f(z) ~ E[Δz]/dt from samples (z_t -> z_{t+1}).
    Uses binning over z_t.
    """
    z_t = z[:-1]
    dz = z[1:] - z[:-1]

    # clip extreme z for stable binning
    lo, hi = np.percentile(z_t, z_clip_quantiles)
    mask = (z_t >= lo) & (z_t <= hi)
    z_t = z_t[mask]
    dz = dz[mask]

    if len(z_t) < max(20, n_bins * 2):
        # too few samples, reduce bins
        n_bins = max(8, int(np.sqrt(max(1, len(z_t)))))

    edges = np.linspace(z_t.min(), z_t.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fhat = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    bin_idx = np.clip(np.digitize(z_t, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = bin_idx == b
        counts[b] = int(sel.sum())
        if counts[b] >= 10:
            fhat[b] = np.mean(dz[sel]) / dt_eff

    # keep bins with estimates
    ok = np.isfinite(fhat)
    return centers[ok], fhat[ok], counts[ok]


def integrate_potential(z_grid, fhat, anchor="min"):
    """
    Given drift f(z), define potential up to constant:
    V(z) = -∫ f(z) dz.
    Use cumulative trapezoid.
    """
    z_grid = np.asarray(z_grid, float)
    fhat = np.asarray(fhat, float)

    # sort by z
    order = np.argsort(z_grid)
    z = z_grid[order]
    f = fhat[order]

    V = np.zeros_like(z)
    for i in range(1, len(z)):
        dz = z[i] - z[i - 1]
        V[i] = V[i - 1] - 0.5 * (f[i] + f[i - 1]) * dz

    # anchor for plotting
    if anchor == "min":
        V = V - np.min(V)
    elif anchor == "mean":
        V = V - np.mean(V)
    return z, V


def count_minima(z, V, smooth_window=5):
    """Count local minima in a smoothed potential curve."""
    z = np.asarray(z)
    V = np.asarray(V)

    if len(V) < 7:
        return 0

    # simple moving average smoothing
    w = int(max(3, smooth_window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    Vp = np.pad(V, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    Vs = np.convolve(Vp, kernel, mode="valid")

    # local minima count
    mins = 0
    for i in range(1, len(Vs) - 1):
        if Vs[i] < Vs[i - 1] and Vs[i] < Vs[i + 1]:
            mins += 1
    return mins


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)

    # Bistable params
    ap.add_argument("--a", type=float, default=0.0, help="tilt parameter for bistable drift")
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--T", type=float, default=400.0, help="total simulated time")

    # Sampling: dense vs sparse
    ap.add_argument("--dense_every", type=int, default=1, help="keep every k step (dense)")
    ap.add_argument("--sparse_every", type=int, default=80, help="keep every k step (sparse)")
    ap.add_argument("--n_bins", type=int, default=30)

    args = ap.parse_args()
    safe_mkdir(args.outdir)
    rng = np.random.default_rng(args.seed)

    n_steps = int(args.T / args.dt)

    # 1) simulate bistable
    z_bi = simulate_em(
        drift_bistable, z0=-1.0, dt=args.dt, n_steps=n_steps,
        sigma=args.sigma, rng=rng, drift_kwargs={"a": args.a}
    )

    # 2) simulate monostable OU as control
    z_ou = simulate_em(
        drift_ou, z0=0.6, dt=args.dt, n_steps=n_steps,
        sigma=args.sigma, rng=rng, drift_kwargs={"theta": 1.2, "mu": 0.0}
    )

    # sampling helper
    def sample_series(z, every):
        idx = np.arange(0, len(z), every, dtype=int)
        zs = z[idx]
        return zs, idx

    z_bi_dense, idx_bi_dense = sample_series(z_bi, args.dense_every)
    z_bi_sparse, idx_bi_sparse = sample_series(z_bi, args.sparse_every)

    z_ou_dense, idx_ou_dense = sample_series(z_ou, args.dense_every)
    z_ou_sparse, idx_ou_sparse = sample_series(z_ou, args.sparse_every)

    # effective dt for the sampled series
    dt_dense = args.dt * args.dense_every
    dt_sparse = args.dt * args.sparse_every

    # reconstruct drift/potential
    def recon(zs, dt_eff):
        zg, fhat, counts = reconstruct_drift_from_samples(zs, dt_eff, n_bins=args.n_bins)
        zV, V = integrate_potential(zg, fhat, anchor="min")
        nmins = count_minima(zV, V)
        return zg, fhat, zV, V, nmins

    bi_d = recon(z_bi_dense, dt_dense)
    bi_s = recon(z_bi_sparse, dt_sparse)
    ou_d = recon(z_ou_dense, dt_dense)
    ou_s = recon(z_ou_sparse, dt_sparse)

    # Save a compact CSV summary
    rows = []
    for name, pack in [
        ("bistable_dense", bi_d), ("bistable_sparse", bi_s),
        ("ou_dense", ou_d), ("ou_sparse", ou_s),
    ]:
        zg, fhat, zV, V, nmins = pack
        rows.append({
            "case": name,
            "n_samples": int(len(zg)),
            "minima_count": int(nmins),
            "z_min": float(np.min(zV)) if len(zV) else np.nan,
            "z_max": float(np.max(zV)) if len(zV) else np.nan,
        })
    pd.DataFrame(rows).to_csv(Path(args.outdir) / "toy_model_summary.csv", index=False)

    # Multi-panel figure
    fig = plt.figure(figsize=(12, 8))

    # Panel A: bistable time series (dense vs sparse)
    ax = fig.add_subplot(2, 3, 1)
    t_dense = idx_bi_dense * args.dt
    ax.plot(t_dense[:4000], z_bi_dense[:4000])
    ax.set_title("Bistable SDE: dense sampling")
    ax.set_xlabel("time")
    ax.set_ylabel("z")

    ax = fig.add_subplot(2, 3, 4)
    t_sparse = idx_bi_sparse * args.dt
    ax.plot(t_sparse[:800], z_bi_sparse[:800], marker="o", linestyle="-", markersize=2)
    ax.set_title("Bistable SDE: sparse sampling")
    ax.set_xlabel("time")
    ax.set_ylabel("z")

    # Panel B: reconstructed potential (bistable)
    ax = fig.add_subplot(2, 3, 2)
    _, _, zV, V, nmins = bi_d
    ax.plot(zV, V)
    ax.set_title(f"Bistable: V(z) dense (minima={nmins})")
    ax.set_xlabel("z")
    ax.set_ylabel("V")

    ax = fig.add_subplot(2, 3, 5)
    _, _, zV, V, nmins = bi_s
    ax.plot(zV, V)
    ax.set_title(f"Bistable: V(z) sparse (minima={nmins})")
    ax.set_xlabel("z")
    ax.set_ylabel("V")

    # Panel C: reconstructed potential (OU control)
    ax = fig.add_subplot(2, 3, 3)
    _, _, zV, V, nmins = ou_d
    ax.plot(zV, V)
    ax.set_title(f"OU control: V(z) dense (minima={nmins})")
    ax.set_xlabel("z")
    ax.set_ylabel("V")

    ax = fig.add_subplot(2, 3, 6)
    _, _, zV, V, nmins = ou_s
    ax.plot(zV, V)
    ax.set_title(f"OU control: V(z) sparse (minima={nmins})")
    ax.set_xlabel("z")
    ax.set_ylabel("V")

    plt.tight_layout()
    out_png = Path(args.outdir) / "toy_models_sparse_sampling.png"
    plt.savefig(out_png, dpi=220)
    plt.close()

    print("Wrote:", out_png)
    print("Wrote:", Path(args.outdir) / "toy_model_summary.csv")


if __name__ == "__main__":
    main()
