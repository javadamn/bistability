# run_ewi_and_minimal_push.py
# ------------------------------------------------------------
# A) Early-warning indicators (variance, lag-1 AC) per subject
#    computed on the observed H proxy series from your CSV.
# B) Minimal intervention search to flip the hysteretic model
#    back to the healthy branch using fitted global parameters.
#
# Inputs:
#   - /mnt/data/combined_scfas_table_scored.csv  (must have subject_id, sample_id, H_proxy_meta_smooth or H_proxy_meta)
#   - /mnt/data/fitted_global_params.csv         (from your previous fit)
#
# Outputs (in ./mw_actions_out):
#   - ewi_summary.csv
#   - ewi_<subject>.png (per-subject EWI plots)
#   - minimal_push_summary.csv
#   - minimal_push_<mode>.png (time-course plots for each successful mode)
#
# Reqs: numpy, pandas, scipy, matplotlib
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.integrate import solve_ivp

# ----------------- Config -----------------
DATA_CSV = "timeseries/combined_scfas_table_scored.csv"
PARAMS_CSV = "mw_fit_out/fitted_global_params.csv"
OUTDIR = "mw_actions_out"
os.makedirs(OUTDIR, exist_ok=True)

H_COL_CANDIDATES = ["H_proxy_meta_smooth", "H_proxy_meta"]
TIME_COL = None            # if you have a real time col, put its name here; else we use within-subject index
MIN_SERIES = 8             # minimum points for EWI on a subject
ROLL_FRAC = 0.25           # rolling window as fraction of series length (>= 6)
EPS = 1e-6

# ----------------- Load data -----------------
df = pd.read_csv(DATA_CSV)
h_col = None
for c in H_COL_CANDIDATES:
    if c in df.columns:
        h_col = c
        break
if h_col is None:
    raise ValueError("No H proxy column found (looked for H_proxy_meta_smooth or H_proxy_meta).")

if TIME_COL and TIME_COL in df.columns:
    df = df.dropna(subset=["subject_id", "sample_id", TIME_COL]).copy()
else:
    df = df.dropna(subset=["subject_id", "sample_id"]).copy()
    df["t_idx"] = df.groupby("subject_id").cumcount().astype(float)
    TIME_COL = "t_idx"

# clip H into [0,1]
df["H_obs"] = df[h_col].clip(0, 1)
df = df.sort_values(["subject_id", TIME_COL])

# ----------------- A) Early-warning indicators -----------------
def rolling_ac1(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(~np.isfinite(x)):
        return np.nan
    x0 = x[:-1]; x1 = x[1:]
    mask = np.isfinite(x0) & np.isfinite(x1)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x0[mask], x1[mask])[0, 1])

ewi_rows = []
for sid, sub in df.groupby("subject_id"):
    t = sub[TIME_COL].values.astype(float)
    H = sub["H_obs"].values.astype(float)
    # require enough finite points
    mask_fin = np.isfinite(H)
    if mask_fin.sum() < MIN_SERIES:
        continue

    # rolling window
    w = max(6, int(np.ceil(len(H) * ROLL_FRAC)))
    # variance
    H_var = pd.Series(H).rolling(window=w, min_periods=max(4, w//2)).var().values
    # lag-1 autocorrelation (computed per window)
    H_ac1 = pd.Series(H).rolling(window=w, min_periods=max(4, w//2)).apply(rolling_ac1, raw=False).values

    # trend tests (Kendall's tau) on available points
    def tau_trend(series):
        m = np.isfinite(series) & np.isfinite(t)
        if m.sum() < 6:  # need enough for a meaningful rank test
            return np.nan, np.nan
        tau, p = kendalltau(t[m], series[m])
        return float(tau), float(p)

    tau_var, p_var = tau_trend(H_var)
    tau_ac1, p_ac1 = tau_trend(H_ac1)

    # save per-subject plot
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(t, H, lw=1.8); ax[0].set_ylabel("H proxy")
    ax[0].grid(True, ls=":", alpha=0.6)
    ax[1].plot(t, H_var, lw=1.8); ax[1].set_ylabel("Var(H)")
    ax[1].set_title(f"{sid} | τ_var={tau_var:.2f} (p={p_var:.3g})")
    ax[1].grid(True, ls=":", alpha=0.6)
    ax[2].plot(t, H_ac1, lw=1.8); ax[2].set_ylabel("AC1(H)"); ax[2].set_xlabel("time")
    ax[2].set_title(f"τ_ac1={tau_ac1:.2f} (p={p_ac1:.3g})")
    ax[2].grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"ewi_{sid}.png"), dpi=180)
    plt.close()

    ewi_rows.append({
        "subject_id": sid, "n_points": int(mask_fin.sum()),
        "window": int(w),
        "tau_var": tau_var, "p_var": p_var,
        "tau_ac1": tau_ac1, "p_ac1": p_ac1
    })

ewi_df = pd.DataFrame(ewi_rows).sort_values(["p_var", "p_ac1"])
ewi_df.to_csv(os.path.join(OUTDIR, "ewi_summary.csv"), index=False)

# ----------------- B) Minimal intervention search -----------------
# Load fitted global parameters
g = pd.read_csv(PARAMS_CSV, index_col=0).squeeze("columns")
# Ordered parameter vector expected by rhs: [r_max,K_M,c,d,g,u,p_low,p_high,H_on,H_off,tau_q]
pars = [
    float(g.get("r_max", 0.32)),
    float(g.get("K_M", 1.0)),
    float(g.get("c", 0.10)),
    float(g.get("d", 0.12)),     # baseline d for intervention tests
    float(g.get("g", 0.5)),
    float(g.get("u", 0.6)),
    float(g.get("p_low", 0.1)),
    float(g.get("p_high", 2.5)),
    float(g.get("H_on", 0.55)),
    float(g.get("H_off", 0.70)),
    float(g.get("tau_q", 4.0)),
]

# Model with intervention
def rhs_mem(t, y, p, U=0.0, T=0.0, mode="butyrate"):
    M, H, B, q = y
    r_max, K_M, c, d, gH, u, pL, pH, H_on, H_off, tau = p
    pB = pL + (pH - pL) * np.clip(q, 0, 1)

    # base dynamics
    r_eff = r_max
    inp_B = 0.0
    pB_aug = 0.0

    if t <= T and U > 0:
        if mode == "butyrate":           # direct butyrate input (e.g., releasing formulation)
            inp_B = U
        elif mode == "prebiotic":        # transiently augments production rate
            pB_aug = U
        elif mode == "engineered":       # transient boost to growth (seeding/engineered producer)
            r_eff = r_max + U

    dM = (r_eff - c * pB) * M * (1 - M / K_M)
    dH = gH * B * (1 - H) - d * H
    dB = (pB + pB_aug) * M - u * H * B + inp_B

    if H < H_on:
        q_inf = 1.0
    elif H > H_off:
        q_inf = 0.0
    else:
        q_inf = q
    dq = (q_inf - q) / tau
    return [dM, dH, dB, dq]

def integrate(p, y0, U=0.0, T=0.0, mode="butyrate", T_end=220.0):
    ts = np.linspace(0, T_end, 900)
    sol = solve_ivp(lambda t,y: rhs_mem(t, y, p, U=U, T=T, mode=mode),
                    (0, T_end), y0, t_eval=ts, rtol=1e-6, atol=1e-8, max_step=0.5)
    return sol

# Get a "bad-branch" steady state y_bad at baseline d by relaxing from low-H
def relax_to_branch(p, H_init=0.55, q_init=1.0, T_relax=200.0):
    y0 = np.array([0.2, H_init, 0.1, q_init], float)
    sol = solve_ivp(lambda t,y: rhs_mem(t,y,p), (0, T_relax), y0,
                    t_eval=np.linspace(0, T_relax, 600),
                    rtol=1e-6, atol=1e-8, max_step=0.5)
    return sol.y[:, -1], sol

y_bad, _ = relax_to_branch(pars, H_init=min(0.6, pars[9]-0.05), q_init=1.0, T_relax=220.0)

def success(sol, H_off, eps=0.02):
    # success = ends above H_off with q ~ off
    H_end = float(np.mean(sol.y[1, -40:]))
    q_end = float(np.mean(sol.y[3, -40:]))
    return (H_end > H_off + eps) and (q_end < 0.2)

modes = ["butyrate", "prebiotic", "engineered"]
U_grid = np.linspace(0.02, 1.2, 25)     # magnitude grid
T_grid = np.linspace(2.0, 60.0, 20)     # duration grid

summary = []
for mode in modes:
    found = None
    for U in U_grid:
        for T in T_grid:
            sol = integrate(pars, y_bad, U=U, T=T, mode=mode, T_end=240.0)
            if success(sol, pars[9]):
                found = (U, T, sol)
                break
        if found:
            break
    if found:
        U*, T*, sol* = found
        summary.append({"mode": mode, "U": U*, "T": T*, "success": True})
        # plot time course
        plt.figure(figsize=(8, 6))
        plt.plot(sol*.t, sol*.y[1], lw=2, label="H")
        plt.plot(sol*.t, sol*.y[3], lw=1.5, label="q")
        plt.axhline(pars[8], ls=":", c="gray", label="H_on")
        plt.axhline(pars[9], ls="--", c="gray", label="H_off")
        plt.axvline(T*, ls="-.", c="k", alpha=0.6, label="end of intervention")
        plt.title(f"{mode}: minimal-ish U={U*:.2f}, T={T*:.1f} h")
        plt.xlabel("time (h)"); plt.ylabel("H, q")
        plt.legend(); plt.grid(True, ls=":", alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"minimal_push_{mode}.png"), dpi=180)
        plt.close()
    else:
        summary.append({"mode": mode, "U": np.nan, "T": np.nan, "success": False})

pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "minimal_push_summary.csv"), index=False)

print("✅ Done.")
print(f"  EWI summary: {os.path.join(OUTDIR,'ewi_summary.csv')}")
print(f"  Minimal push summary: {os.path.join(OUTDIR,'minimal_push_summary.csv')}")
