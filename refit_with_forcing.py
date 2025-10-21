#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refit per subject with:
- tighter priors on u,d (bounds),
- hysteresis margins (H_off > H_on + m; p_high > p_low + m),
- prebiotic forcing: p_B = p_low + (p_high - p_low)*q + alpha * U(t), alpha>=0, U in {0,1},
- multi-start optimizer,
- outputs RMSE/NRMSE, params, plots, and a new summary.

Usage:
  python refit_with_forcing.py \
    --input 40168_2022_1307_MOESM8_ESM.csv \
    --outdir out_prebiotic_refit \
    --n_starts 7 \
    --min_points 4
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

# ---------------- data helpers ----------------
def load_and_tidy(inp_path: Path):
    df = pd.read_csv(inp_path)
    df.columns = [c.strip().replace("#","_id") for c in df.columns]

    # Column mapping
    def pick(cols, candidates):
        for c in candidates:
            if c in cols: return c
        return None

    cols = set(df.columns)
    sid = pick(cols, ["ID_id","ID","Subject","Participant","participant_id","subject_id"])
    week = pick(cols, ["Week","week","Study week","study_week"])
    preb = pick(cols, ["Prebiotic","prebiotic","Arm","arm","Diet","diet"])
    trt  = pick(cols, ["Treatment","treatment","Group","group"])
    ac   = pick(cols, ["acetate","Acetate","Acetate_mM","acetate_mM"])
    pr   = pick(cols, ["propionate","Propionate","Propionate_mM","propionate_mM"])
    bu   = pick(cols, ["butyrate","Butyrate","Butyrate_mM","butyrate_mM"])

    if sid is None or week is None or bu is None:
        raise SystemExit("Required columns missing: subject_id/Week/butyrate")

    df = df.rename(columns={sid:"subject_id", week:"week"})
    if preb: df = df.rename(columns={preb:"prebiotic"})
    if trt:  df = df.rename(columns={trt:"treatment"})
    if ac:   df = df.rename(columns={ac:"acetate"})
    if pr:   df = df.rename(columns={pr:"propionate"})
    df = df.rename(columns={bu:"butyrate"})

    for c in ["week","acetate","propionate","butyrate"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # time axis in hours from first week per subject
    first_week = df.groupby("subject_id")["week"].transform("min")
    df["week_index"] = df["week"] - first_week
    df["time_hr"] = df["week_index"] * 7.0 * 24.0

    # binary U(t) forcing: 1 during non-baseline prebiotic weeks, else 0
    # Adjust this rule to match the paper's baseline coding if needed.
    if "prebiotic" in df.columns:
        df["U"] = (df["prebiotic"].astype(str).str.lower().isin(
            ["inulin","gos","dextrin","arabixylan","fibre","fiber","resistant starch","prebiotic"]
        )).astype(float)
    else:
        df["U"] = 0.0

    tidy_cols = ["subject_id","week","week_index","time_hr","prebiotic","treatment","U","acetate","propionate","butyrate"]
    tidy_cols = [c for c in tidy_cols if c in df.columns]
    tidy = df[tidy_cols].dropna(subset=["time_hr","butyrate"]).sort_values(["subject_id","time_hr"])
    return tidy

# ---------------- ODE with forcing & hysteresis ----------------
def ode_rhs(t, y, pars, U_fun):
    """
    y = [M, H, B, q]
    pars = (r_max, K_M, c, d, g, u, p_low, p_high, H_on, H_off, tau_q, alpha)
    U_fun(t) in [0,1] indicates prebiotic weeks.
    """
    M, H, B, q = y
    r_max, K_M, c, d, g, u, p_low, p_high, H_on, H_off, tau_q, alpha = pars
    U = U_fun(t)

    pB_base = p_low + (p_high - p_low) * np.clip(q, 0, 1)
    p_B = pB_base + alpha * U
    p_B = max(0.0, p_B)

    dMdt = (r_max - c * p_B) * M * (1 - M / K_M)
    dHdt = g * B * (1 - H) - d * H
    dBdt = p_B * M - u * H * B

    # hysteretic q target
    if H < H_on:
        q_inf = 1.0
    elif H > H_off:
        q_inf = 0.0
    else:
        q_inf = q
    dqdt = (q_inf - q) / tau_q
    return [dMdt, dHdt, dBdt, dqdt]

# piecewise-constant U(t) from (time_hr, U) samples
def make_U_fun(times, Us):
    times = np.asarray(times, float)
    Us = np.asarray(Us, float)
    order = np.argsort(times)
    t_sorted = times[order]
    u_sorted = Us[order]
    def U_of_t(t):
        # hold last observed value at/left of t
        idx = np.searchsorted(t_sorted, t, side="right") - 1
        if idx < 0: idx = 0
        if idx >= len(u_sorted): idx = len(u_sorted) - 1
        return float(u_sorted[idx])
    return U_of_t

# ---------------- fitting ----------------
def fit_subject(gsub, outdir, n_starts=7, min_band=0.05, min_gap=0.2):
    sid = gsub["subject_id"].iloc[0]
    t_obs = gsub["time_hr"].values.astype(float)
    B_obs = gsub["butyrate"].values.astype(float)
    U_obs = gsub["U"].values.astype(float)
    t0, t1 = float(t_obs.min()), float(t_obs.max())
    U_fun = make_U_fun(t_obs, U_obs)

    # initial conditions heuristic
    B0 = max(B_obs[0], 1e-3)
    H0_guess = 0.6 if B0 >= np.median(B_obs) else 0.5
    M0_guess = 0.2
    def make_y0(x):
        H_on = x[8]; H0 = H0_guess; q0 = 1.0 if H0 < H_on else 0.0
        return np.array([M0_guess, H0, B0, q0], float)

    # parameters with forcing alpha
    #   [0] r_max, [1] K_M, [2] c, [3] d, [4] g, [5] u,
    #   [6] p_low, [7] p_high, [8] H_on, [9] H_off, [10] tau_q, [11] alpha
    x0 = np.array([0.30, 1.0, 0.10, 0.12, 0.45, 0.60, 0.10, 2.50, 0.55, 0.70, 4.0, 0.6])
    lb = np.array([0.05, 0.3, 0.02, 0.08, 0.10, 0.50, 0.00, 0.40, 0.30, 0.40, 0.5, 0.0])
    ub = np.array([0.60, 2.0, 0.25, 0.20, 1.20, 0.85, 0.80, 4.00, 0.75, 0.95, 24.0, 2.0])

    t_eval = np.linspace(t0, t1, max(40, len(t_obs)*10))

    def simulate(x):
        y0 = make_y0(x)
        sol = solve_ivp(ode_rhs, (t0, t1), y0, args=(x, U_fun),
                        method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.5)
        return sol

    def residuals(x):
        # enforce hysteresis margins via soft penalties
        band = x[9] - x[8]
        gap  = x[7] - x[6]
        pen_band = max(0.0, min_band - band)
        pen_gap  = max(0.0, min_gap  - gap)

        # quick reject if margins violated badly
        if pen_band > 0.05 or pen_gap > 0.1:
            return np.ones_like(t_obs) * 1e3

        sol = simulate(x)
        if not sol.success:
            return np.ones_like(t_obs) * 1e3

        Bmod = np.interp(t_obs, sol.t, sol.y[2])
        res = (Bmod - B_obs)
        # small regularization to keep margins
        res = np.concatenate([res, [50.0*pen_band, 20.0*pen_gap]])
        return res

    # multistart
    rng = np.random.default_rng(42)
    starts = [x0]
    for _ in range(max(1, n_starts-1)):
        jitter = rng.normal(0, 0.08, size=len(x0))
        xj = np.clip(x0 * (1.0 + jitter), lb, ub)
        # ensure H_off > H_on by at least min_band in start
        if xj[9] <= xj[8] + min_band:
            xj[9] = min(ub[9], xj[8] + (min_band + 0.05))
        # ensure p_high - p_low > min_gap in start
        if xj[7] <= xj[6] + min_gap:
            xj[7] = min(ub[7], xj[6] + (min_gap + 0.2))
        starts.append(xj)

    best = None
    for x_init in starts:
        fit = least_squares(residuals, x_init, bounds=(lb, ub), max_nfev=600, verbose=0)
        if (best is None) or (fit.cost < best.cost):
            best = fit

    x_hat = best.x
    sol = simulate(x_hat)

    # errors
    N = len(t_obs)
    # cost = 0.5 * sum(res^2)
    RMSE = np.sqrt(2.0 * best.cost / N)
    denom = (np.nanmax(B_obs) - np.nanmin(B_obs))
    NRMSE = RMSE / denom if denom > 0 else np.nan

    # save
    names = ["r_max","K_M","c","d","g","u","p_low","p_high","H_on","H_off","tau_q","alpha"]
    params_df = pd.Series(x_hat, index=names).to_frame(name="value")
    params_df["subject_id"] = sid

    outdir = Path(outdir)
    (outdir / "fits").mkdir(parents=True, exist_ok=True)

    params_df.to_csv(outdir / "fits" / f"subject_{sid}_fit_params_refit.csv")

    sim_df = pd.DataFrame({
        "t_hr": sol.t,
        "M": sol.y[0],
        "H": sol.y[1],
        "B_mM_model": sol.y[2],
        "q": sol.y[3],
        "U": [float(make_U_fun(t_obs, U_obs)(t)) for t in sol.t],
    })
    sim_df.to_csv(outdir / "fits" / f"subject_{sid}_simulated_refit.csv", index=False)

    # plot
    fig, ax = plt.subplots(3,1, figsize=(8,7), sharex=True)
    ax[0].plot(sol.t, sol.y[2], lw=2, label="Model B")
    ax[0].scatter(t_obs, B_obs, c="k", s=28, label="Observed B")
    ax[0].set_ylabel("Butyrate (mM)"); ax[0].legend(); ax[0].grid(ls=":", alpha=0.6)

    ax[1].plot(sol.t, sol.y[1], lw=2, label="H")
    ax[1].axhline(x_hat[8], ls=":", c="gray", label="H_on")
    ax[1].axhline(x_hat[9], ls="--", c="gray", label="H_off")
    ax[1].set_ylabel("Host H"); ax[1].legend(); ax[1].grid(ls=":", alpha=0.6)

    U_line = [make_U_fun(t_obs, U_obs)(t) for t in sol.t]
    ax[2].step(sol.t, U_line, where="post")
    ax[2].set_xlabel("Time (hr)"); ax[2].set_ylabel("Prebiotic U(t)")
    ax[2].grid(ls=":", alpha=0.6)

    fig.suptitle(f"Subject {sid} — Refit with forcing (RMSE={RMSE:.2f}, NRMSE={NRMSE:.2f})", y=0.98)
    fig.tight_layout()
    fig.savefig(outdir / "fits" / f"subject_{sid}_fit_plot_refit.png", dpi=200)
    plt.close(fig)

    row = {"subject_id": sid, "success": best.success, "cost": best.cost, "RMSE": RMSE, "NRMSE": NRMSE}
    row.update({k:v for k,v in zip(names, x_hat)})
    row["p_gap"] = x_hat[7] - x_hat[6]
    row["H_band"] = x_hat[9] - x_hat[8]
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_starts", type=int, default=7)
    ap.add_argument("--min_points", type=int, default=4)
    ap.add_argument("--min_band", type=float, default=0.05)
    ap.add_argument("--min_gap", type=float, default=0.2)
    args = ap.parse_args()

    outdir = Path(args.outdir); (outdir / "fits").mkdir(parents=True, exist_ok=True)
    tidy = load_and_tidy(Path(args.input))

    # Write updated tidy table for reference
    tidy.to_csv(outdir / "prebiotic_scfa_timeseries_refit_basis.csv", index=False)

    rows = []
    for sid, g in tidy.groupby("subject_id"):
        g = g.dropna(subset=["time_hr","butyrate"])
        if len(g) < max(2, args.min_points):
            print(f"[skip] subject {sid}: only {len(g)} usable points")
            continue
        row = fit_subject(g, outdir, n_starts=args.n_starts,
                          min_band=args.min_band, min_gap=args.min_gap)
        rows.append(row)
        print(f"[fit] subject {sid}: RMSE={row['RMSE']:.3f} NRMSE={row['NRMSE']:.3f} "
              f"p_gap={row['p_gap']:.2f} H_band={row['H_band']:.2f} alpha={row['alpha']:.2f}")

    if rows:
        summary = pd.DataFrame(rows).sort_values(["success","NRMSE","cost"], ascending=[False,True,True])
        summary.to_csv(outdir / "fits" / "fit_summary_all_subjects_refit.csv", index=False)
        # Shortlist based on improved criteria
        keep = summary[
            (summary["success"]==True) &
            (summary["NRMSE"]<=0.35) &
            (summary["p_gap"]>=0.5) &
            (summary["H_band"]>=0.08)
        ]
        keep.to_csv(outdir / "fits" / "shortlist_subjects_refit.csv", index=False)
        print(f"[write] {outdir / 'fits' / 'fit_summary_all_subjects_refit.csv'}")
        print(f"[write] {outdir / 'fits' / 'shortlist_subjects_refit.csv'}")
        print(f"[info] kept {len(keep)}/{len(summary)} subjects post-refit")

if __name__ == "__main__":
    main()
