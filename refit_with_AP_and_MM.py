#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refit butyrate time-series with:
 - Acetate & propionate drivers in p_B (co-substrate & competition) using Hill terms
 - Michaelis–Menten uptake for host (u_max, K_B)
 - Choice of hysteresis: relay band (H_on/H_off) or smooth logistic switch
 - Multi-start least-squares with soft constraints (margin on band & production gap)
 - Gentle L2 priors on alpha (forcing), beta_A (acetate), beta_P (propionate)

Inputs:
  - Prebiotic crossover supplement CSV (e.g., 40168_2022_1307_MOESM8_ESM.csv)

Outputs (in --outdir):
  - fits/subject_<ID>_fit_params_APMM.csv
  - fits/subject_<ID>_simulated_APMM.csv
  - fits/subject_<ID>_fit_plot_APMM.png
  - fits/fit_summary_all_subjects_APMM.csv
  - fits/shortlist_subjects_APMM.csv

Usage:
  python refit_with_AP_and_MM.py \
    --input 40168_2022_1307_MOESM8_ESM.csv \
    --outdir out_prebiotic_APMM \
    --hysteresis smooth \
    --n_starts 10 \
    --min_points 4
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ---------------------- data helpers ----------------------
def load_and_tidy(inp_path: Path):
    df = pd.read_csv(inp_path)
    df.columns = [c.strip().replace("#","_id") for c in df.columns]

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

    # numerics
    for c in ["week","acetate","propionate","butyrate"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # time axis (hours from first week per subject)
    first_week = df.groupby("subject_id")["week"].transform("min")
    df["week_index"] = df["week"] - first_week
    df["time_hr"] = df["week_index"] * 7.0 * 24.0

    # U(t) flag for intervention weeks (adjust strings if needed)
    if "prebiotic" in df.columns:
        df["prebiotic"] = df["prebiotic"].astype(str).str.lower()
        pos = ("inulin","gos","dextrin","dex","arabixylan","resistant starch","resistant-starch","prebiotic","fiber","fibre")
        df["U"] = df["prebiotic"].apply(lambda s: float(any(p in s for p in pos)))
    else:
        df["U"] = 0.0

    tidy_cols = ["subject_id","week","week_index","time_hr","prebiotic","treatment","U","acetate","propionate","butyrate"]
    tidy_cols = [c for c in tidy_cols if c in df.columns]
    tidy = df[tidy_cols].dropna(subset=["time_hr","butyrate"]).sort_values(["subject_id","time_hr"])
    return tidy

def make_series_fun(times, values, kind="linear"):
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    valid = ~np.isnan(times) & ~np.isnan(values)
    t = times[valid]; v = values[valid]
    if len(t) == 0:
        return lambda x: 0.0
    if len(t) == 1:
        val = float(v[0])
        return lambda x: val
    return interp1d(t, v, kind=kind, fill_value=(v[0], v[-1]), bounds_error=False)

def make_step_fun(times, values):
    times = np.asarray(times, float); values = np.asarray(values, float)
    order = np.argsort(times)
    t_sorted = times[order]; v_sorted = values[order]
    def f(t):
        idx = np.searchsorted(t_sorted, t, side="right") - 1
        if idx < 0: idx = 0
        if idx >= len(v_sorted): idx = len(v_sorted) - 1
        return float(v_sorted[idx])
    return f

# ---------------------- ODEs ----------------------
def q_infty_relay(H, H_on, H_off, q):
    if H < H_on: return 1.0
    if H > H_off: return 0.0
    return q

def q_infty_smooth(H, H_on, H_off):
    # logistic center + width from band
    Hc = 0.5 * (H_on + H_off)
    kappa = max(1e-3, 0.25 * (H_off - H_on))  # slope ∝ band width
    z = (Hc - H) / kappa
    return 1.0 / (1.0 + np.exp(-z))

def ode_rhs_APMM(t, y, pars, U_fun, A_fun, P_fun, hysteresis="relay"):
    """
    y = [M, H, B, q]
    pars =
      r_max, K_M, c, d, g,
      u_max, K_B,
      p_low, p_high, H_on, H_off, tau_q,
      alpha,
      beta_A, K_A,
      beta_P, K_P
    (We keep Hill exponents fixed at 1.5 implicitly via shaping of KA/KP; could be added if needed.)
    """
    M, H, B, q = y

    (r_max, K_M, c, d, g,
     u_max, K_B,
     p_low, p_high, H_on, H_off, tau_q,
     alpha,
     beta_A, K_A,
     beta_P, K_P) = pars

    U = float(U_fun(t))
    A = max(0.0, float(A_fun(t)))
    P = max(0.0, float(P_fun(t)))

    # Base hysteretic production
    q_base = q
    if hysteresis == "relay":
        q_inf = q_infty_relay(H, H_on, H_off, q)
    else:  # smooth
        q_inf = q_infty_smooth(H, H_on, H_off)
    dqdt = (q_inf - q) / tau_q

    pB_base = p_low + (p_high - p_low) * np.clip(q, 0.0, 1.0)
    # Hill exponent ~1.5 approximated by squashing via KA/KP; keep simple Hill=1 for stability
    A_term = beta_A * (A / (K_A + A + 1e-12))
    P_term = beta_P * (P / (K_P + P + 1e-12))
    p_B = pB_base + alpha*U + A_term - P_term
    p_B = max(0.0, p_B)

    dMdt = (r_max - c * p_B) * M * (1 - M / K_M)
    dHdt = g * B * (1 - H) - d * H
    uptake = u_max * H * B / (K_B + B + 1e-9)  # saturable uptake
    dBdt = p_B * M - uptake
    return [dMdt, dHdt, dBdt, dqdt]

# ---------------------- fitting ----------------------
def fit_subject_APMM(gsub, outdir, hysteresis="relay", n_starts=7,
                     min_band=0.05, min_gap=0.2,
                     prior_alpha=0.2, prior_betaA=0.1, prior_betaP=0.1,
                     bounds=None, seed=42):
    sid = gsub["subject_id"].iloc[0]
    t_obs = gsub["time_hr"].values.astype(float)
    B_obs = gsub["butyrate"].values.astype(float)
    U_obs = gsub["U"].values.astype(float)
    A_obs = gsub["acetate"].values.astype(float) if "acetate" in gsub.columns else np.zeros_like(t_obs)
    P_obs = gsub["propionate"].values.astype(float) if "propionate" in gsub.columns else np.zeros_like(t_obs)

    # functions in time
    U_fun = make_step_fun(t_obs, U_obs)
    A_fun = make_series_fun(t_obs, A_obs, kind="linear")
    P_fun = make_series_fun(t_obs, P_obs, kind="linear")

    t0, t1 = float(t_obs.min()), float(t_obs.max())

    # Initial conditions
    B0 = max(B_obs[0], 1e-3)
    H0 = 0.6 if B0 >= np.nanmedian(B_obs) else 0.5
    M0 = 0.25
    q0 = 1.0 if H0 < 0.55 else 0.0

    # Params:
    # [0] r_max, [1] K_M, [2] c, [3] d, [4] g,
    # [5] u_max, [6] K_B,
    # [7] p_low, [8] p_high, [9] H_on, [10] H_off, [11] tau_q,
    # [12] alpha, [13] beta_A, [14] K_A, [15] beta_P, [16] K_P
    x0 = np.array([
        0.30, 1.10, 0.10, 0.12, 0.45,
        0.70, 5.0,
        0.10, 2.50, 0.55, 0.70, 8.0,
        0.6,  1.0,  10.0, 0.7,  10.0
    ], dtype=float)
    lb = np.array([
        0.05, 0.5,  0.02, 0.08, 0.10,
        0.40, 0.5,
        0.00, 0.40, 0.30, 0.40, 0.5,
        0.0,  0.0,  1.0,  0.0,  1.0
    ], dtype=float)
    ub = np.array([
        0.60, 2.5,  0.25, 0.20, 1.20,
        1.10, 30.0,
        0.80, 5.00, 0.80, 0.98, 24.0,
        2.5,  5.0,  60.0, 3.0,  60.0
    ], dtype=float)
    if bounds is not None:
        lb, ub = bounds

    def make_y0(pars):
        H_on = pars[9]
        q0_local = 1.0 if H0 < H_on else 0.0
        return np.array([M0, H0, B0, q0_local], float)

    t_eval = np.linspace(t0, t1, max(50, len(t_obs)*12))

    def simulate(pars):
        y0 = make_y0(pars)
        sol = solve_ivp(
            ode_rhs_APMM, (t0, t1), y0,
            args=(pars, U_fun, A_fun, P_fun, hysteresis),
            t_eval=t_eval, rtol=1e-6, atol=1e-8, method="RK45", max_step=1.0
        )
        return sol

    # Priors: weak L2 on alpha, beta_A, beta_P around 0
    # weight chosen so that ~1.0 in these params adds about same order as 1 mM residual over a few points
    lam_alpha = prior_alpha
    lam_betaA = prior_betaA
    lam_betaP = prior_betaP

    def residuals(x):
        # enforce margins softly
        band = x[10] - x[9]           # H_off - H_on
        gap  = x[8]  - x[7]           # p_high - p_low
        pen_band = max(0.0, min_band - band)
        pen_gap  = max(0.0, min_gap  - gap)
        if pen_band > 0.06 or pen_gap > 0.12:
            return np.ones_like(t_obs) * 1e3

        sol = simulate(x)
        if not sol.success:
            return np.ones_like(t_obs) * 1e3

        Bmod = np.interp(t_obs, sol.t, sol.y[2])
        res = (Bmod - B_obs)

        # small penalties to keep margins and shrink overly large driver gains/alpha
        reg = np.array([
            80.0*pen_band,
            40.0*pen_gap,
            lam_alpha * x[12],
            lam_betaA * x[13],
            lam_betaP * x[15]
        ], dtype=float)
        return np.concatenate([res, reg])

    # multi-start
    rng = np.random.default_rng(42)
    starts = [x0]
    for _ in range(max(1, n_starts-1)):
        jitter = rng.normal(0, 0.10, size=len(x0))
        xj = np.clip(x0 * (1.0 + jitter), lb, ub)
        # seed margins
        if xj[10] <= xj[9] + min_band:
            xj[10] = min(ub[10], xj[9] + (min_band + 0.06))
        if xj[8] <= xj[7] + min_gap:
            xj[8] = min(ub[8],  xj[7] + (min_gap + 0.2))
        starts.append(xj)

    best = None
    for x_init in starts:
        fit = least_squares(residuals, x_init, bounds=(lb, ub), max_nfev=800, verbose=0)
        if (best is None) or (fit.cost < best.cost):
            best = fit

    x_hat = best.x
    sol = simulate(x_hat)

    # errors
    N = len(t_obs)
    RMSE = float(np.sqrt(2.0 * best.cost / N))
    rngB = float(np.nanmax(B_obs) - np.nanmin(B_obs)) if N > 1 else np.nan
    NRMSE = float(RMSE / rngB) if rngB and rngB > 0 else np.nan

    # save
    names = ["r_max","K_M","c","d","g","u_max","K_B",
             "p_low","p_high","H_on","H_off","tau_q",
             "alpha","beta_A","K_A","beta_P","K_P"]
    params_df = pd.Series(x_hat, index=names).to_frame(name="value")
    params_df["subject_id"] = sid

    outdir = Path(outdir)
    (outdir / "fits").mkdir(parents=True, exist_ok=True)

    params_df.to_csv(outdir / "fits" / f"subject_{sid}_fit_params_APMM.csv")

    sim_df = pd.DataFrame({
        "t_hr": sol.t,
        "M": sol.y[0],
        "H": sol.y[1],
        "B_mM_model": sol.y[2],
        "q": sol.y[3],
        "U": [float(U_fun(t)) for t in sol.t],
        "A": [float(A_fun(t)) for t in sol.t],
        "P": [float(P_fun(t)) for t in sol.t],
    })
    sim_df.to_csv(outdir / "fits" / f"subject_{sid}_simulated_APMM.csv", index=False)

    # plot
    fig, ax = plt.subplots(4,1, figsize=(8,9), sharex=True)
    ax[0].plot(sol.t, sol.y[2], lw=2, label="Model B")
    ax[0].scatter(t_obs, B_obs, c="k", s=28, label="Observed B")
    ax[0].set_ylabel("Butyrate (mM)"); ax[0].legend(); ax[0].grid(ls=":", alpha=0.6)

    ax[1].plot(sol.t, sol.y[1], lw=2, label="H")
    ax[1].axhline(x_hat[9],  ls=":", c="gray", label="H_on")
    ax[1].axhline(x_hat[10], ls="--", c="gray", label="H_off")
    ax[1].set_ylabel("Host H"); ax[1].legend(); ax[1].grid(ls=":", alpha=0.6)

    ax[2].plot(sol.t, sim_df["A"], lw=1.5, label="Acetate A(t)")
    ax[2].plot(sol.t, sim_df["P"], lw=1.5, label="Propionate P(t)")
    ax[2].set_ylabel("mM"); ax[2].legend(); ax[2].grid(ls=":", alpha=0.6)

    ax[3].step(sol.t, sim_df["U"], where="post", label="U(t) prebiotic")
    ax[3].set_xlabel("Time (hr)"); ax[3].set_ylabel("U(t)")
    ax[3].grid(ls=":", alpha=0.6)

    fig.suptitle(f"Subject {sid} — A/P + MM uptake ({hysteresis})  RMSE={RMSE:.2f}  NRMSE={NRMSE:.2f}", y=0.99)
    fig.tight_layout()
    fig.savefig(outdir / "fits" / f"subject_{sid}_fit_plot_APMM.png", dpi=200)
    plt.close(fig)

    row = {"subject_id": sid, "success": best.success, "cost": best.cost, "RMSE": RMSE, "NRMSE": NRMSE}
    row.update({k:v for k,v in zip(names, x_hat)})
    row["p_gap"] = x_hat[8] - x_hat[7]
    row["H_band"] = x_hat[10] - x_hat[9]
    return row

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--hysteresis", choices=["relay","smooth"], default="relay",
                    help="Use 'relay' (banded switch) or 'smooth' (logistic) hysteresis.")
    ap.add_argument("--n_starts", type=int, default=7)
    ap.add_argument("--min_points", type=int, default=4)
    ap.add_argument("--min_band", type=float, default=0.05)
    ap.add_argument("--min_gap", type=float, default=0.2)
    ap.add_argument("--prior_alpha", type=float, default=0.2)
    ap.add_argument("--prior_betaA", type=float, default=0.1)
    ap.add_argument("--prior_betaP", type=float, default=0.1)
    args = ap.parse_args()

    outdir = Path(args.outdir); (outdir / "fits").mkdir(parents=True, exist_ok=True)
    tidy = load_and_tidy(Path(args.input))
    tidy.to_csv(outdir / "prebiotic_scfa_timeseries_APMM_basis.csv", index=False)

    rows = []
    for sid, g in tidy.groupby("subject_id"):
        g = g.dropna(subset=["time_hr","butyrate"])
        if len(g) < max(2, args.min_points):
            print(f"[skip] subject {sid}: only {len(g)} usable points")
            continue
        row = fit_subject_APMM(
            g, outdir,
            hysteresis=args.hysteresis,
            n_starts=args.n_starts,
            min_band=args.min_band,
            min_gap=args.min_gap,
            prior_alpha=args.prior_alpha,
            prior_betaA=args.prior_betaA,
            prior_betaP=args.prior_betaP
        )
        rows.append(row)
        print(f"[fit] subject {sid}: NRMSE={row['NRMSE']:.3f}  p_gap={row['p_gap']:.2f}  H_band={row['H_band']:.2f}  alpha={row.get('alpha',np.nan):.2f}")

    if rows:
        summary = pd.DataFrame(rows).sort_values(["success","NRMSE","cost"], ascending=[False,True,True])
        summary.to_csv(outdir / "fits" / "fit_summary_all_subjects_APMM.csv", index=False)

        # Shortlist (same thresholds as before; tweak as you like)
        keep = summary[
            (summary["success"]==True) &
            (summary["NRMSE"]<=0.40) &
            (summary["p_gap"]>=1.0) &
            (summary["H_band"]>=0.10)
        ]
        keep.to_csv(outdir / "fits" / "shortlist_subjects_APMM.csv", index=False)

        print(f"[write] {outdir / 'fits' / 'fit_summary_all_subjects_APMM.csv'}")
        print(f"[write] {outdir / 'fits' / 'shortlist_subjects_APMM.csv'}")
        print(f"[info] kept {len(keep)}/{len(summary)} subjects (A/P + MM, {args.hysteresis})")
    else:
        print("[warn] No subjects were fitted (insufficient points or parsing issue).")

if __name__ == "__main__":
    main()
