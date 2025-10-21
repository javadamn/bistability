#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refine the persistence boundary (magnitude–duration) for inulin-like interventions.

Inputs (produced by your pipeline):
- /path/to/summary_per_subject.csv
- (optional but preferred) /path/to/out/fit_params_<Subject>.csv  # per-subject full fit

Outputs (under --outdir):
- refine_<Subject>_minU_vs_duration.csv
- refine_<Subject>_minDuration_vs_U.csv
- refine_<Subject>_boundary.png  # overlayed curves

Usage example:
python refine_transition_boundary.py \
  --summary /mnt/data/summary_per_subject.csv \
  --subject S1 \
  --outdir /mnt/data/refine_S1 \
  --durations 48 72 96 120 144 168 192 216 240 264 288 312 336 \
  --Ugrid 0.1 0.15 0.2 0.25 0.3 0.35 0.4 \
  --U-bracket 0.0 1.2 \
  --D-bracket 12 504
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------- ODE (same model as your pipeline) --------------------
def rhs(t, y, pars, schedule_fn, U_amp):
    M, H, B, q = y
    r_max, K_M, c = pars["r_max"], pars["K_M"], pars["c"]
    d, g, u = pars["d"], pars["g"], pars["u"]
    p_low, p_high = pars["p_low"], pars["p_high"]
    H_on, H_off, tau_q = pars["H_on"], pars["H_off"], pars["tau_q"]

    I = schedule_fn(t)  # 1 during intervention, else 0
    q_clamped = 0.0 if q < 0 else (1.0 if q > 1.0 else q)
    p_base = p_low + (p_high - p_low) * q_clamped
    p_B = p_base + U_amp * I

    dMdt = (r_max - c*p_B) * M * (1 - M/K_M)
    dHdt = g * B * (1 - H) - d * H
    dBdt = p_B * M - u * H * B

    if H < H_on: q_inf = 1.0
    elif H > H_off: q_inf = 0.0
    else: q_inf = q
    dqdt = (q_inf - q) / tau_q
    return [dMdt, dHdt, dBdt, dqdt]

def baseline_state(pars, tmax=600.0):
    y0 = [0.2, 0.6, 0.1, 0.0]
    def sched0(t): return 0.0
    sol = solve_ivp(lambda t,y: rhs(t,y,pars,sched0,0.0),
                    (0, tmax), y0, t_eval=[tmax],
                    rtol=3e-6, atol=1e-8, max_step=1.0)
    return sol.y[:, -1]

def simulate_intervention(pars, y_init, U_amp, duration_hr, post_hr=300.0):
    # on for duration_hr, then off
    def sched(t): return 1.0 if t <= duration_hr else 0.0
    t_end = duration_hr + post_hr
    sol = solve_ivp(lambda t,y: rhs(t,y,pars,sched,U_amp),
                    (0, t_end), y_init, t_eval=[t_end],
                    rtol=3e-6, atol=1e-8, max_step=1.0)
    return sol.y[:, -1]

def persists_after_removal(pars, y_end, eps=1e-2):
    return bool(y_end[1] > pars["H_off"] + eps)

# -------------------- Bracketing + bisection --------------------
def bisection_min_U(pars, duration_hr, U_lo, U_hi, tol=1e-3, max_iter=40):
    """Smallest U in [lo,hi] that yields persistence at given duration."""
    y0 = baseline_state(pars)
    # ensure we have a bracket: lo fails, hi succeeds (or swap if needed)
    y_lo = simulate_intervention(pars, y0, U_lo, duration_hr); s_lo = persists_after_removal(pars, y_lo)
    y_hi = simulate_intervention(pars, y0, U_hi, duration_hr); s_hi = persists_after_removal(pars, y_hi)
    if s_lo and s_hi:
        return U_lo  # already succeeds at the low bound
    if (not s_lo) and (not s_hi):
        return np.nan  # no success even at hi
    # ensure ordering
    if s_lo and (not s_hi):
        U_lo, U_hi = U_hi, U_lo
        s_lo, s_hi = s_hi, s_lo
    # bisection
    for _ in range(max_iter):
        mid = 0.5*(U_lo + U_hi)
        y_mid = simulate_intervention(pars, y0, mid, duration_hr)
        s_mid = persists_after_removal(pars, y_mid)
        if s_mid:
            U_hi = mid
        else:
            U_lo = mid
        if abs(U_hi - U_lo) < tol:
            break
    return U_hi

def bisection_min_duration(pars, U_amp, D_lo, D_hi, tol=0.5, max_iter=40):
    """Smallest duration (hr) in [lo,hi] that yields persistence at given U."""
    y0 = baseline_state(pars)
    y_lo = simulate_intervention(pars, y0, U_amp, D_lo); s_lo = persists_after_removal(pars, y_lo)
    y_hi = simulate_intervention(pars, y0, U_amp, D_hi); s_hi = persists_after_removal(pars, y_hi)
    if s_lo and s_hi:
        return D_lo
    if (not s_lo) and (not s_hi):
        return np.nan
    if s_lo and (not s_hi):
        D_lo, D_hi = D_hi, D_lo
        s_lo, s_hi = s_hi, s_lo
    for _ in range(max_iter):
        mid = 0.5*(D_lo + D_hi)
        y_mid = simulate_intervention(pars, y0, U_amp, mid)
        s_mid = persists_after_removal(pars, y_mid)
        if s_mid:
            D_hi = mid
        else:
            D_lo = mid
        if abs(D_hi - D_lo) < tol:
            break
    return D_hi

# -------------------- IO helpers --------------------
def load_subject_params(summary_csv, subject, fit_dir=None):
    summary = pd.read_csv(summary_csv)
    row = summary[summary["Subject"].astype(str).str.upper()==str(subject).upper()]
    if row.empty:
        raise ValueError(f"Subject {subject} not found in {summary_csv}")
    r = row.iloc[0]
    pars = dict(
        d=float(r["d"]), g=float(r["g"]), u=float(r["u"]),
        p_low=float(r["p_low"]), p_high=float(r["p_high"]),
        H_on=float(r["H_on"]), H_off=float(r["H_off"]),
        tau_q=4.0, r_max=0.32, K_M=1.0, c=0.10
    )
    # try to refine from fit_params_<Subject>.csv if available
    if fit_dir:
        pfile = Path(fit_dir) / f"fit_params_{subject}.csv"
        if pfile.exists():
            fp = pd.read_csv(pfile, header=None, names=["name","value"])
            for k,v in zip(fp["name"], fp["value"]):
                pars[str(k)] = float(v)
    return pars

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary_per_subject.csv path")
    ap.add_argument("--subject", required=True, help="Subject ID (e.g., S1)")
    ap.add_argument("--fitdir", default="out", help="Directory with fit_params_<Subject>.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--durations", nargs="+", type=float, default=[48,72,96,120,144,168,192,216,240,264,288,312,336],
                    help="Durations (hr) for min-U search")
    ap.add_argument("--Ugrid", nargs="+", type=float, default=[0.1,0.15,0.2,0.25,0.3,0.35,0.4],
                    help="Magnitudes U for min-duration search")
    ap.add_argument("--U-bracket", nargs=2, type=float, default=[0.0, 1.2], help="U lower/upper for bisection")
    ap.add_argument("--D-bracket", nargs=2, type=float, default=[12, 504], help="Duration hr lower/upper for bisection")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pars = load_subject_params(args.summary, args.subject, args.fitdir)

    # 1) Minimal U for a set of durations
    rows_U = []
    for D in args.durations:
        Umin = bisection_min_U(pars, duration_hr=D, U_lo=args.U_bracket[0], U_hi=args.U_bracket[1], tol=1e-3)
        rows_U.append({"duration_hr": D, "min_U": Umin})
    dfU = pd.DataFrame(rows_U)
    dfU.to_csv(outdir / f"refine_{args.subject}_minU_vs_duration.csv", index=False)

    # 2) Minimal duration for a set of U magnitudes
    rows_D = []
    for U in args.Ugrid:
        Dmin = bisection_min_duration(pars, U_amp=U, D_lo=args.D_bracket[0], D_hi=args.D_bracket[1], tol=0.5)
        rows_D.append({"U": U, "min_duration_hr": Dmin})
    dfD = pd.DataFrame(rows_D)
    dfD.to_csv(outdir / f"refine_{args.subject}_minDuration_vs_U.csv", index=False)

    # 3) Plot boundary (both parameterizations)
    plt.figure(figsize=(8,5))
    # U as a function of duration (filter nan)
    d_ok = dfU.dropna()
    if len(d_ok):
        plt.plot(d_ok["duration_hr"]/24.0, d_ok["min_U"], "-o", label="min U for duration")
    # duration as a function of U
    u_ok = dfD.dropna()
    if len(u_ok):
        plt.plot(u_ok["min_duration_hr"]/24.0, u_ok["U"], "-o", label="min duration for U")
    plt.xlabel("Duration (days)")
    plt.ylabel("U (magnitude)")
    plt.title(f"Persistence boundary — {args.subject}")
    plt.grid(True, ls=":", alpha=0.6)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"refine_{args.subject}_boundary.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
