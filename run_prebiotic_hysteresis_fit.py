#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end pipeline for the prebiotic crossover dataset (Microbiome 2022 supplement)
File: 40168_2022_1307_MOESM8_ESM.csv

What this script does:
1) Load the CSV, tidy & rename columns:
   - subject_id (from 'ID#'), Week, Prebiotic, Treatment,
     acetate, propionate, butyrate (assumed mM)
2) Build per-subject time axis in hours:
   time_hr = (Week - subject's first Week) * 7 * 24
3) For each subject with >= 4 time points:
   - Fit a hysteretic ODE model to butyrate trajectory (B only; M/H latent)
   - Save fitted parameters, a diagnostic plot, and simulated time series
4) Write combined tidy table and per-subject ODE-ready tables.

Usage:
    python run_prebiotic_hysteresis_fit.py \
        --input 40168_2022_1307_MOESM8_ESM.csv \
        --outdir out_prebiotic

Outputs (in outdir/):
  - prebiotic_scfa_timeseries.csv
  - ode_ready/subject_<ID>_data_timeseries.csv
  - fits/subject_<ID>_fit_params.csv
  - fits/subject_<ID>_fit_plot.png
  - fits/subject_<ID>_simulated.csv
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ------------- ODE model (microbe M, host H (latent), butyrate B, memory q) -------------
def microbiome_model_mem(t, y, pars):
    """
    State y = [M, H, B, q]
    Parameters pars = (r_max, K_M, c, d, g, u, p_low, p_high, H_on, H_off, tau_q)

    dM/dt = (r_max - c * p_B) * M * (1 - M / K_M)
    dH/dt = g * B * (1 - H) - d * H
    dB/dt = p_B * M - u * H * B
    dq/dt = (q_inf - q) / tau_q, with hysteretic target q_inf:
            q_inf = 1 if H < H_on; q_inf = 0 if H > H_off; else q_inf = q
    """
    M, H, B, q = y
    r_max, K_M, c, d, g, u, p_low, p_high, H_on, H_off, tau_q = pars

    # hysteretic production policy: p_B in [p_low, p_high] controlled by q
    p_B = p_low + (p_high - p_low) * np.clip(q, 0.0, 1.0)

    dMdt = (r_max - c * p_B) * M * (1 - M / K_M)
    dHdt = g * B * (1 - H) - d * H
    dBdt = p_B * M - u * H * B

    # hysteretic target q_inf
    if H < H_on:
        q_inf = 1.0
    elif H > H_off:
        q_inf = 0.0
    else:
        q_inf = q
    dqdt = (q_inf - q) / tau_q
    return [dMdt, dHdt, dBdt, dqdt]

# ------------- Utilities -------------
def robust_num(x):
    try:
        return pd.to_numeric(x, errors="coerce")
    except Exception:
        return np.nan

def build_time_axis(df):
    """
    Creates a per-subject time index in hours using Week as the base:
    time_hr = (Week - min Week per subject) * 7 * 24
    """
    df = df.copy()
    first_week = df.groupby("subject_id")["week"].transform("min")
    df["week_index"] = df["week"] - first_week
    df["time_hr"] = df["week_index"] * 7.0 * 24.0
    return df

def make_outdirs(base):
    base = Path(base)
    (base / "fits").mkdir(parents=True, exist_ok=True)
    (base / "ode_ready").mkdir(parents=True, exist_ok=True)
    return base

# ------------- Fitting per subject (SCFA-only: fit to butyrate time series) -------------
def fit_subject(time_hr, B_mM, subject_id, outdir, default_init=None):
    """
    Fit model parameters by minimizing squared error between simulated B(t) (in mM)
    and observed B_mM (also mM). H and M are latent; we regularize and constrain bounds.

    Returns dict of fitted parameters and writes plots/CSVs.
    """

    # --- scaling: model B "units" = mM directly (no extra scale)
    # Reasonable bounds/priors from literature and earlier toy model
    B0 = max(B_mM[0], 1e-3)
    H0_guess = 0.65 if B0 >= np.nanmedian(B_mM) else 0.5  # heuristic
    M0_guess = 0.2
    q0_guess = 1.0 if H0_guess < 0.6 else 0.0

    # parameter vector (11 params)
    # [r_max, K_M, c, d, g, u, p_low, p_high, H_on, H_off, tau_q]
    x0 = np.array([0.30, 1.0, 0.10, 0.12, 0.45, 0.60, 0.10, 2.50, 0.55, 0.70, 4.0])
    lb = np.array([0.05, 0.3, 0.02, 0.01, 0.05, 0.20, 0.00, 0.40, 0.20, 0.30, 0.5])
    ub = np.array([0.60, 2.0, 0.25, 0.40, 1.20, 1.20, 0.80, 4.00, 0.80, 0.95, 24.0])

    # initial conditions:
    def make_y0(x):
        H_on = x[8]
        H0 = H0_guess
        q0 = 1.0 if (H0 < H_on) else 0.0
        M0 = M0_guess
        return np.array([M0, H0, B0, q0], dtype=float)

    t0, t1 = float(time_hr.min()), float(time_hr.max())
    t_eval = np.linspace(t0, t1, max(30, len(time_hr) * 8))

    def simulate(pars):
        y0 = make_y0(pars)
        sol = solve_ivp(microbiome_model_mem, (t0, t1), y0, args=(pars,),
                        method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.5)
        return sol

    # residuals: fit ONLY B
    def residuals(x):
        # soft constraint: H_off > H_on
        if x[9] <= x[8]:
            return np.ones_like(time_hr) * 1e3
        sol = simulate(x)
        if not sol.success:
            return np.ones_like(time_hr) * 1e3
        Bmod = np.interp(time_hr, sol.t, sol.y[2])
        resB = (Bmod - B_mM)
        # small regularization to maintain margin p_high - p_low >= 0.2
        margin = max(0.0, 0.2 - (x[7] - x[6]))
        res = np.concatenate([resB, [10.0 * margin]])
        return res

    fit = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=400, verbose=0)
    pars_hat = fit.x
    sol_hat = simulate(pars_hat)

    # --- outputs
    names = ["r_max","K_M","c","d","g","u","p_low","p_high","H_on","H_off","tau_q"]
    params_df = pd.Series(pars_hat, index=names).to_frame(name="value")
    params_df["subject_id"] = subject_id

    # save params
    pth_params = Path(outdir) / "fits" / f"subject_{subject_id}_fit_params.csv"
    params_df.to_csv(pth_params)

    # save simulated trajectory
    sim_df = pd.DataFrame({
        "t_hr": sol_hat.t,
        "M": sol_hat.y[0],
        "H": sol_hat.y[1],
        "B_mM_model": sol_hat.y[2],
        "q": sol_hat.y[3],
    })
    pth_sim = Path(outdir) / "fits" / f"subject_{subject_id}_simulated.csv"
    sim_df.to_csv(pth_sim, index=False)

    # plot
    fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=True)
    # Butyrate fit
    ax[0].plot(sol_hat.t, sol_hat.y[2], lw=2, label="Model B (mM)")
    ax[0].scatter(time_hr, B_mM, c="k", s=35, label="Observed B (mM)")
    ax[0].set_ylabel("Butyrate (mM)"); ax[0].grid(True, ls=":", alpha=0.6)
    ax[0].legend()
    # Host H & thresholds (for intuition)
    ax[1].plot(sol_hat.t, sol_hat.y[1], lw=2, label="H (latent)")
    ax[1].axhline(pars_hat[8], ls=":", c="gray", label="H_on")
    ax[1].axhline(pars_hat[9], ls="--", c="gray", label="H_off")
    ax[1].set_xlabel("Time (hr)"); ax[1].set_ylabel("Host H"); ax[1].grid(True, ls=":", alpha=0.6)
    ax[1].legend()
    fig.suptitle(f"Subject {subject_id} — Butyrate fit with hysteresis", y=0.98)
    fig.tight_layout()
    pth_fig = Path(outdir) / "fits" / f"subject_{subject_id}_fit_plot.png"
    fig.savefig(pth_fig, dpi=200)
    plt.close(fig)

    return {
        "subject_id": subject_id,
        "params": dict(zip(names, pars_hat)),
        "params_path": str(pth_params),
        "sim_path": str(pth_sim),
        "fig_path": str(pth_fig),
        "success": fit.success,
        "cost": fit.cost
    }

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to 40168_2022_1307_MOESM8_ESM.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min_points", type=int, default=4, help="Minimum time points per subject to fit")
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = make_outdirs(args.outdir)

    # --- load data ---
    df = pd.read_csv(inp)
    # Normalize column names a bit
    df.columns = [c.strip().replace("#","_id") for c in df.columns]
    # Try to map expected columns
    colmap = {}
    # subject id
    for cand in ["ID_id","ID","Subject","Participant","participant_id","subject_id"]:
        if cand in df.columns:
            colmap["subject_id"] = cand
            break
    # time (Week)
    for cand in ["Week","week","Study week","study_week"]:
        if cand in df.columns:
            colmap["week"] = cand
            break
    # prebiotic arm
    for cand in ["Prebiotic","prebiotic","Arm","arm","Diet","diet"]:
        if cand in df.columns:
            colmap["prebiotic"] = cand
            break
    # treatment (as provided in supplement; can be same as prebiotic)
    for cand in ["Treatment","treatment","Group","group"]:
        if cand in df.columns:
            colmap["treatment"] = cand
            break
    # SCFAs
    for key, cands in {
        "acetate": ["acetate","Acetate","Acetate_mM","acetate_mM"],
        "propionate": ["propionate","Propionate","Propionate_mM","propionate_mM"],
        "butyrate": ["butyrate","Butyrate","Butyrate_mM","butyrate_mM"],
    }.items():
        for cand in cands:
            if cand in df.columns:
                colmap[key] = cand
                break

    required = ["subject_id","week","butyrate"]
    missing = [k for k in required if k not in colmap]
    if missing:
        raise RuntimeError(f"Could not find required columns in CSV: {missing}. "
                           f"Columns seen: {list(df.columns)[:30]} ...")

    # Apply renames
    df = df.rename(columns={v:k for k,v in colmap.items()})

    # Coerce numerics for SCFAs & week
    for c in ["week","acetate","propionate","butyrate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build time axis
    df = build_time_axis(df)

    # Keep tidy columns and sort
    keep = ["subject_id","week","week_index","time_hr","prebiotic","treatment","acetate","propionate","butyrate"]
    keep = [c for c in keep if c in df.columns]
    tidy = df[keep].sort_values(["subject_id","time_hr"])
    tidy.to_csv(outdir / "prebiotic_scfa_timeseries.csv", index=False)
    print(f"[write] {outdir / 'prebiotic_scfa_timeseries.csv'}")

    # Also write ODE-ready per-subject CSVs (butyrate only)
    ode_ready_dir = outdir / "ode_ready"
    for sid, g in tidy.groupby("subject_id"):
        sub = g[["time_hr","butyrate"]].dropna().rename(columns={"butyrate":"butyrate_mM"})
        if len(sub) >= 2:
            sub.to_csv(ode_ready_dir / f"subject_{sid}_data_timeseries.csv", index=False)

    # Fit per subject
    fit_summaries = []
    for sid, g in tidy.groupby("subject_id"):
        g = g.dropna(subset=["time_hr","butyrate"])
        if len(g) < max(2, args.min_points):
            print(f"[skip] subject {sid}: only {len(g)} usable points (<{args.min_points})")
            continue
        res = fit_subject(g["time_hr"].values.astype(float),
                          g["butyrate"].values.astype(float),
                          subject_id=sid,
                          outdir=outdir)
        fit_summaries.append(res)
        print(f"[fit] subject {sid}: success={res['success']} cost={res['cost']:.4f}")

    # Write a combined summary of fitted parameters
    if fit_summaries:
        rows = []
        for r in fit_summaries:
            row = {"subject_id": r["subject_id"], "success": r["success"], "cost": r["cost"]}
            row.update({k: v for k, v in r["params"].items()})
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(outdir / "fits" / "fit_summary_all_subjects.csv", index=False)
        print(f"[write] {outdir / 'fits' / 'fit_summary_all_subjects.csv'}")
    else:
        print("[warn] No subjects were fitted (insufficient points or parsing issue).")

    print("\nDone. Outputs are in:", outdir)

if __name__ == "__main__":
    main()
