#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refit prebiotic crossover data with Michaelis–Menten host uptake and TWO metabolites:
Acetate (A) and Butyrate (B).

Data: 40168_2022_1307_MOESM8_ESM.csv (Microbiome 2022 supplement)
Outputs: per-subject parameters, simulated trajectories, plots, and group summaries.

Usage:
  python refit_MM_acetate_butyrate.py \
    --input 40168_2022_1307_MOESM8_ESM.csv \
    --outdir out_MM_AB \
    --n_starts 9 \
    --min_points 4
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# --------------------------- data helpers ---------------------------

def _pick(cols, candidates):
    for c in candidates:
        if c in cols: return c
    return None

def load_and_tidy(inp: Path):
    df = pd.read_csv(inp)
    df.columns = [c.strip().replace("#", "_id") for c in df.columns]
    cols = set(df.columns)

    sid  = _pick(cols, ["ID_id","ID","Subject","Participant","participant_id","subject_id"])
    week = _pick(cols, ["Week","week","Study week","study_week"])
    preb = _pick(cols, ["Prebiotic","prebiotic","Arm","arm","Diet","diet"])
    trt  = _pick(cols, ["Treatment","treatment","Group","group"])

    ac   = _pick(cols, ["acetate","Acetate","Acetate_mM","acetate_mM"])
    bu   = _pick(cols, ["butyrate","Butyrate","Butyrate_mM","butyrate_mM"])

    if not (sid and week and ac and bu):
        raise SystemExit("Required columns missing. Need subject id, Week, acetate, butyrate.")

    df = df.rename(columns={sid:"subject_id", week:"week", ac:"acetate", bu:"butyrate"})
    if preb: df = df.rename(columns={preb:"prebiotic"})
    if trt:  df = df.rename(columns={trt:"treatment"})

    for c in ["week","acetate","butyrate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # time axis in hours from each subject's first week
    first_week = df.groupby("subject_id")["week"].transform("min")
    df["week_index"] = df["week"] - first_week
    df["time_hr"] = df["week_index"] * 7.0 * 24.0

    # binary forcing U(t); mark typical prebiotic arms
    if "prebiotic" in df.columns:
        df["U"] = (df["prebiotic"].astype(str).str.lower().isin(
            ["inulin","gos","dextrin","arabinoxylan","resistant starch","prebiotic","fiber","fibre"]
        )).astype(float)
    else:
        df["U"] = 0.0

    tidy_cols = ["subject_id","week","week_index","time_hr","prebiotic","treatment","U","acetate","butyrate"]
    tidy_cols = [c for c in tidy_cols if c in df.columns]
    tidy = df[tidy_cols].dropna(subset=["time_hr","acetate","butyrate"]).sort_values(["subject_id","time_hr"])
    return tidy

def make_U_fun(times, Us):
    times = np.asarray(times, float)
    Us    = np.asarray(Us, float)
    order = np.argsort(times)
    t_sorted = times[order]
    u_sorted = Us[order]
    def U_of_t(t):
        idx = np.searchsorted(t_sorted, t, side="right") - 1
        if idx < 0: idx = 0
        if idx >= len(u_sorted): idx = len(u_sorted) - 1
        return float(u_sorted[idx])
    return U_of_t

# --------------------------- ODE model ---------------------------

def ode_rhs_MM_AB(t, y, pars, U_fun):
    """
    States: y = [M, H, A, B, q]
    Parameters:
      r_max, K_M, c, d, g_A, g_B,
      u_A, K_A, u_B, K_B,
      pA_low, pA_high, pB_low, pB_high,
      H_on, H_off, tau_q,
      alpha_A, alpha_B

    Uptake (Michaelis–Menten): v_A = u_A * H * A / (K_A + A), v_B = u_B * H * B / (K_B + B)
    Production: p_A = pA_low + (pA_high - pA_low)*q + alpha_A * U(t)
                p_B = pB_low + (pB_high - pB_low)*q + alpha_B * U(t)
    """
    (r_max, K_M, c, d, g_A, g_B,
     u_A, K_A, u_B, K_B,
     pA_low, pA_high, pB_low, pB_high,
     H_on, H_off, tau_q,
     alpha_A, alpha_B) = pars

    M, H, A, B, q = y
    U = U_fun(t)

    # Switching policy for production
    p_A = pA_low + (pA_high - pA_low) * np.clip(q, 0, 1) + alpha_A * U
    p_B = pB_low + (pB_high - pB_low) * np.clip(q, 0, 1) + alpha_B * U
    p_A = max(0.0, p_A)
    p_B = max(0.0, p_B)

    # Host uptake (MM)
    v_A = u_A * H * A / (K_A + max(1e-9, A))
    v_B = u_B * H * B / (K_B + max(1e-9, B))

    # ODEs
    dMdt = (r_max - c*(p_A + p_B)) * M * (1 - M / K_M)
    dHdt = (g_A * A + g_B * B) * (1 - H) - d * H
    dAdt = p_A * M - v_A
    dBdt = p_B * M - v_B

    # hysteresis target for q
    if H < H_on:
        q_inf = 1.0
    elif H > H_off:
        q_inf = 0.0
    else:
        q_inf = q
    dqdt = (q_inf - q) / tau_q

    return [dMdt, dHdt, dAdt, dBdt, dqdt]

# --------------------------- fitting ---------------------------

def fit_subject_MM_AB(gsub, outdir, n_starts=9, min_band=0.05, min_gap=0.2):
    sid = gsub["subject_id"].iloc[0]
    t_obs = gsub["time_hr"].values.astype(float)
    A_obs = gsub["acetate"].values.astype(float)
    B_obs = gsub["butyrate"].values.astype(float)
    U_obs = gsub["U"].values.astype(float)

    t0, t1 = float(t_obs.min()), float(t_obs.max())
    U_fun = make_U_fun(t_obs, U_obs)

    # Initial conditions (heuristics)
    A0 = max(A_obs[0], 1e-3)
    B0 = max(B_obs[0], 1e-3)
    H0 = 0.6 if (B0 >= np.median(B_obs)) else 0.5
    M0 = 0.2
    def make_y0(x):
        H_on = x[14]
        q0 = 1.0 if H0 < H_on else 0.0
        return np.array([M0, H0, A0, B0, q0], float)

    # Parameters (19 total)
    #  r_max, K_M, c, d, g_A, g_B,
    #  u_A, K_A, u_B, K_B,
    #  pA_low, pA_high, pB_low, pB_high,
    #  H_on, H_off, tau_q,
    #  alpha_A, alpha_B
    x0 = np.array([0.30, 1.0, 0.08, 0.12, 0.10, 0.45,
                   0.70, 10.0, 0.60, 8.0,
                   0.30, 3.00, 0.10, 2.50,
                   0.55, 0.72, 6.0,
                   0.4, 0.8])

    lb = np.array([0.05, 0.3, 0.02, 0.08, 0.02, 0.10,
                   0.40,  1.0, 0.40, 1.0,
                   0.00, 0.60, 0.00, 0.60,
                   0.30, 0.45, 0.5,
                   0.0, 0.0])

    ub = np.array([0.80, 2.0, 0.25, 0.20, 0.80, 1.20,
                   1.20, 30.0, 1.20, 30.0,
                   1.50, 5.00, 0.80, 4.00,
                   0.75, 0.95, 24.0,
                   2.0, 2.0])

    t_eval = np.linspace(t0, t1, max(50, len(t_obs)*10))

    def simulate(x):
        y0 = make_y0(x)
        sol = solve_ivp(ode_rhs_MM_AB, (t0, t1), y0, args=(x, U_fun),
                        method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.5)
        return sol

    # Residuals: concatenate A and B errors (option to scale weights)
    wA = 1.0
    wB = 1.0
    def residuals(x):
        # hysteresis margins
        band  = x[15] - x[14]
        gap_A = x[11] - x[10]
        gap_B = x[13] - x[12]
        pen_band = max(0.0, min_band - band)
        pen_gapA = max(0.0, min_gap  - gap_A)
        pen_gapB = max(0.0, min_gap  - gap_B)

        if pen_band > 0.05 or pen_gapA > 0.1 or pen_gapB > 0.1:
            return np.ones(len(t_obs)*2) * 1e3

        sol = simulate(x)
        if not sol.success:
            return np.ones(len(t_obs)*2) * 1e3

        A_mod = np.interp(t_obs, sol.t, sol.y[2])
        B_mod = np.interp(t_obs, sol.t, sol.y[3])

        resA = wA * (A_mod - A_obs)
        resB = wB * (B_mod - B_obs)

        # small regularization terms to preserve margins
        reg = np.array([40.0*pen_band, 20.0*pen_gapA, 20.0*pen_gapB])
        return np.concatenate([resA, resB, reg])

    # Multi-start around x0
    rng = np.random.default_rng(123)
    starts = [x0]
    for _ in range(max(1, n_starts-1)):
        jitter = rng.normal(0, 0.08, size=len(x0))
        xj = np.clip(x0*(1.0 + jitter), lb, ub)
        # Ensure margins
        if xj[15] <= xj[14] + min_band:
            xj[15] = min(ub[15], xj[14] + (min_band + 0.05))
        if xj[11] <= xj[10] + min_gap:
            xj[11] = min(ub[11], xj[10] + (min_gap + 0.2))
        if xj[13] <= xj[12] + min_gap:
            xj[13] = min(ub[13], xj[12] + (min_gap + 0.2))
        starts.append(xj)

    best = None
    for init in starts:
        fit = least_squares(residuals, init, bounds=(lb, ub), max_nfev=800, verbose=0)
        if (best is None) or (fit.cost < best.cost):
            best = fit

    x_hat = best.x
    sol   = simulate(x_hat)

    # Errors per metabolite
    N = len(t_obs)
    # cost = 0.5*sum(res^2) over all residuals (A & B + regs). Recompute RMSEs cleanly:
    A_mod = np.interp(t_obs, sol.t, sol.y[2])
    B_mod = np.interp(t_obs, sol.t, sol.y[3])

    RMSE_A = np.sqrt(np.mean((A_mod - A_obs)**2))
    RMSE_B = np.sqrt(np.mean((B_mod - B_obs)**2))
    rngA = np.nanmax(A_obs) - np.nanmin(A_obs)
    rngB = np.nanmax(B_obs) - np.nanmin(B_obs)
    NRMSE_A = RMSE_A / (rngA if rngA > 0 else np.nan)
    NRMSE_B = RMSE_B / (rngB if rngB > 0 else np.nan)
    # simple combined scores
    RMSE_mean = 0.5*(RMSE_A + RMSE_B)
    NRMSE_mean = np.nanmean([NRMSE_A, NRMSE_B])

    # Save params
    names = ["r_max","K_M","c","d","g_A","g_B",
             "u_A","K_A","u_B","K_B",
             "pA_low","pA_high","pB_low","pB_high",
             "H_on","H_off","tau_q","alpha_A","alpha_B"]
    params_df = pd.Series(x_hat, index=names).to_frame(name="value")
    params_df["subject_id"] = sid

    outdir = Path(outdir)
    (outdir / "fits").mkdir(parents=True, exist_ok=True)
    params_df.to_csv(outdir / "fits" / f"subject_{sid}_MMAB_params.csv")

    # Save simulation
    sim_df = pd.DataFrame({
        "t_hr": sol.t,
        "M": sol.y[0],
        "H": sol.y[1],
        "Acetate_model": sol.y[2],
        "Butyrate_model": sol.y[3],
        "q": sol.y[4],
        "U": [float(make_U_fun(t_obs, U_obs)(t)) for t in sol.t],
    })
    sim_df.to_csv(outdir / "fits" / f"subject_{sid}_MMAB_sim.csv", index=False)

    # Plot
    fig, axs = plt.subplots(3,1, figsize=(8,8), sharex=True)
    # A & B
    axs[0].plot(sol.t, sol.y[2], lw=2, label="Acetate (model)")
    axs[0].plot(sol.t, sol.y[3], lw=2, label="Butyrate (model)")
    axs[0].scatter(t_obs, A_obs, c="C0", s=24, marker="o", label="Acetate obs")
    axs[0].scatter(t_obs, B_obs, c="C1", s=24, marker="s", label="Butyrate obs")
    axs[0].set_ylabel("mM"); axs[0].legend(frameon=False); axs[0].grid(ls=":", alpha=0.6)
    # H & thresholds
    axs[1].plot(sol.t, sol.y[1], lw=2, label="H (latent)")
    axs[1].axhline(x_hat[14], ls=":", c="gray", label="H_on")
    axs[1].axhline(x_hat[15], ls="--", c="gray", label="H_off")
    axs[1].set_ylabel("Host H"); axs[1].legend(frameon=False); axs[1].grid(ls=":", alpha=0.6)
    # U(t)
    U_line = [make_U_fun(t_obs, U_obs)(t) for t in sol.t]
    axs[2].step(sol.t, U_line, where="post")
    axs[2].set_xlabel("Time (hr)"); axs[2].set_ylabel("U(t)")
    axs[2].grid(ls=":", alpha=0.6)

    fig.suptitle(f"Subject {sid} — MM uptake; A+B fit  (NRMSE_A={NRMSE_A:.2f}, NRMSE_B={NRMSE_B:.2f})", y=0.98)
    fig.tight_layout()
    fig.savefig(outdir / "fits" / f"subject_{sid}_MMAB_plot.png", dpi=200)
    plt.close(fig)

    row = {
        "subject_id": sid, "success": best.success, "cost": best.cost,
        "RMSE_A": RMSE_A, "NRMSE_A": NRMSE_A,
        "RMSE_B": RMSE_B, "NRMSE_B": NRMSE_B,
        "RMSE_mean": RMSE_mean, "NRMSE_mean": NRMSE_mean
    }
    row.update({k:v for k,v in zip(names, x_hat)})
    # hysteresis metrics
    row["pA_gap"] = x_hat[11] - x_hat[10]
    row["pB_gap"] = x_hat[13] - x_hat[12]
    row["H_band"] = x_hat[15] - x_hat[14]
    return row

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_starts", type=int, default=9)
    ap.add_argument("--min_points", type=int, default=4)
    ap.add_argument("--min_band", type=float, default=0.05)
    ap.add_argument("--min_gap", type=float, default=0.2)
    args = ap.parse_args()

    outdir = Path(args.outdir); (outdir / "fits").mkdir(parents=True, exist_ok=True)
    tidy = load_and_tidy(Path(args.input))
    tidy.to_csv(outdir / "MMAB_timeseries_basis.csv", index=False)

    rows = []
    for sid, g in tidy.groupby("subject_id"):
        g = g.dropna(subset=["time_hr","acetate","butyrate"])
        if len(g) < max(2, args.min_points):
            print(f"[skip] subject {sid}: only {len(g)} usable points")
            continue
        row = fit_subject_MM_AB(
            g, outdir, n_starts=args.n_starts,
            min_band=args.min_band, min_gap=args.min_gap
        )
        rows.append(row)
        print(f"[fit] subject {sid}: "
              f"NRMSE_A={row['NRMSE_A']:.3f} NRMSE_B={row['NRMSE_B']:.3f} "
              f"H_band={row['H_band']:.2f} pA_gap={row['pA_gap']:.2f} pB_gap={row['pB_gap']:.2f}")

    if rows:
        summary = pd.DataFrame(rows).sort_values(["success","NRMSE_mean","cost"], ascending=[False,True,True])
        summary.to_csv(outdir / "fits" / "fit_summary_MMAB.csv", index=False)

        # quick shortlist with reasonable criteria
        keep = summary[
            (summary.get("success", True) == True) &
            (summary["NRMSE_A"] <= 0.45) &
            (summary["NRMSE_B"] <= 0.45) &
            (summary["H_band"]  >= 0.08) &
            (summary["pA_gap"]  >= 0.5) &
            (summary["pB_gap"]  >= 0.5)
        ]
        keep.to_csv(outdir / "fits" / "shortlist_MMAB.csv", index=False)

        print(f"[write] {outdir / 'fits' / 'fit_summary_MMAB.csv'}")
        print(f"[write] {outdir / 'fits' / 'shortlist_MMAB.csv'}")
        print(f"[info] kept {len(keep)}/{len(summary)} subjects")
    else:
        print("[warn] No subjects were fitted (insufficient points or parsing issue).")

if __name__ == "__main__":
    main()
