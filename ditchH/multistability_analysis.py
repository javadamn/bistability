#!/usr/bin/env python3
import argparse, os, json, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def subjcol(df):
    for c in df.columns:
        if isinstance(c,str) and re.search(r"^subject", c, flags=re.IGNORECASE): return c
    return df.columns[0]

def zspace_stats(B):
    z = np.log1p(B.astype(float))
    mu = float(np.nanmean(z)); sd = float(np.nanstd(z)+1e-8)
    return mu, sd

def e_update(zB, E, center=0.5, gap=0.2):
    up = center + gap/2.0; dn = center - gap/2.0
    if E < 0.5 and zB >= up: return 1.0
    if E > 0.5 and zB <= dn: return 0.0
    return E

def simulate(B0, F, A, L, dt, p0, aF, aA, lam, kL, h_amp=0.2, z_mu=0.0, z_sd=1.0, center=0.5, gap=0.4):
    n = len(F)
    B = np.zeros(n+1); E = np.zeros(n+1)
    B[0] = B0; E[0] = 0.0
    for t in range(n):
        zB = (np.log1p(max(B[t], 1e-12)) - z_mu)/z_sd
        E[t] = e_update(zB, E[t], center=center, gap=gap)
        dBdt = p0 + aF*F[t] - aA*A[t] + kL*L[t] - lam*B[t] + h_amp*(E[t]-0.5)
        B[t+1] = max(0.0, B[t] + dt[t]*dBdt)
        E[t+1] = E[t]
    return B[1:], E[1:]

def nullcline_B(F, A, L, p0, aF, aA, lam, kL, h_amp, E):
    return (p0 + aF*F - aA*A + kL*L + h_amp*(E-0.5)) / (lam + 1e-12)

def sweep_hysteresis_F(F_min, F_max, A0, L0, p0, aF, aA, lam, kL, h_amp, z_mu, z_sd, center=0.5, gap=0.4, steps=50):
    F_vals = np.linspace(F_min, F_max, steps)
    B_up = []; curB = nullcline_B(F_vals[0], A0, L0, p0, aF, aA, lam, kL, h_amp, 0.0)
    for F in F_vals:
        dt = np.ones(20)*1.0
        B, _ = simulate(curB, np.full_like(dt, F), np.full_like(dt, A0), np.full_like(dt, L0),
                        dt, p0, aF, aA, lam, kL, h_amp, z_mu, z_sd, center, gap)
        curB = B[-1]; B_up.append(curB)
    F_vals2 = np.linspace(F_max, F_min, steps)
    B_dn = []; curB = B_up[-1]
    for F in F_vals2:
        dt = np.ones(20)*1.0
        B, _ = simulate(curB, np.full_like(dt, F), np.full_like(dt, A0), np.full_like(dt, L0),
                        dt, p0, aF, aA, lam, kL, h_amp, z_mu, z_sd, center, gap)
        curB = B[-1]; B_dn.append(curB)
    return F_vals, np.array(B_up), F_vals2, np.array(B_dn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_csv", required=True)
    ap.add_argument("--pred_v4", required=True)
    ap.add_argument("--global_json", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--center", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--h_amp", type=float, default=0.2)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load per-subject params
    params = pd.read_csv(args.params_csv)
    sc = subjcol(params); params = params.rename(columns={sc:"subject"})
    row = params[params["subject"].astype(str)==str(args.subject)]
    if row.empty: raise SystemExit("Subject not found in params.")
    p0=float(row["p0"].iloc[0]); aF=float(row["alpha_F"].iloc[0]); aA=float(row["alpha_A"].iloc[0])

    # global constants
    with open(args.global_json,"r") as f: gj=json.load(f)
    lam = float(gj.get("lambda_best", gj.get("lambda_B", 0.1)))
    kL  = float(gj.get("k_LB_best", gj.get("k_LB", 0.0)))

    # aligned predictions table (already aligned to target step)
    pred = pd.read_csv(args.pred_v4)
    scp = subjcol(pred); pred = pred.rename(columns={scp:"subject"})
    gp = pred[pred["subject"].astype(str)==str(args.subject)].copy()
    if gp.empty: raise SystemExit("No predictions rows for subject.")

    if "aligned_index" in gp.columns:
        gp = gp.sort_values("aligned_index")

    # pull aligned series and hard-align lengths
    B_obs = gp["B_obs"].astype(float).to_numpy()
    F = gp["F_in"].astype(float).to_numpy() if "F_in" in gp.columns else np.zeros_like(B_obs)
    A = gp["A_in"].astype(float).to_numpy() if "A_in" in gp.columns else np.zeros_like(B_obs)
    L = gp["L_in"].astype(float).to_numpy() if "L_in" in gp.columns else np.zeros_like(B_obs)

    m = min(len(B_obs), len(F), len(A), len(L))
    B_obs = B_obs[:m]; F = F[:m]; A = A[:m]; L = L[:m]
    if m < 2: raise SystemExit("Not enough aligned points to simulate.")

    dt = np.ones(m, float)
    z_mu, z_sd = zspace_stats(B_obs)

    # simulate hysteresis-driven trajectory on the aligned window
    Bhat, E = simulate(B0=float(B_obs[0]), F=F, A=A, L=L, dt=dt,
                       p0=p0, aF=aF, aA=aA, lam=lam, kL=kL,
                       h_amp=args.h_amp, z_mu=z_mu, z_sd=z_sd,
                       center=args.center, gap=args.gap)

    # ensure equal lengths for plotting
    k = min(len(Bhat), len(B_obs))
    t = np.arange(k)
    plt.figure(figsize=(7,4))
    plt.plot(t, B_obs[:k], label="B_obs")
    plt.plot(t, Bhat[:k], label="B_sim(hysteresis)")
    plt.xlabel("Aligned step"); plt.ylabel("Butyrate (arb.)")
    plt.title(f"Subject {args.subject}: trajectory with hysteresis")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.subject}_traj.png"), dpi=200); plt.close()

    # drivers
    plt.figure(figsize=(7,4))
    plt.plot(t, F[:k], label="F")
    plt.plot(t, A[:k], label="A")
    plt.plot(t, L[:k], label="L")
    plt.xlabel("Aligned step"); plt.ylabel("Drivers (scaled)")
    plt.title(f"Subject {args.subject}: drivers")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.subject}_drivers.png"), dpi=200); plt.close()

    # hysteresis sweep vs F (hold A0,L0 at medians of aligned window)
    A0 = float(np.median(A[:k])) if k>0 else 0.0
    L0 = float(np.median(L[:k])) if k>0 else 0.0
    if np.all(~np.isfinite(F[:k])) or np.nanstd(F[:k]) < 1e-12:
        Fmin, Fmax = 0.0, 1.0
    else:
        Fmin, Fmax = float(np.nanpercentile(F[:k],5)), float(np.nanpercentile(F[:k],95))
        if not np.isfinite(Fmin) or not np.isfinite(Fmax) or Fmax<=Fmin:
            Fmin, Fmax = 0.0, 1.0

    F_up, B_up, F_dn, B_dn = sweep_hysteresis_F(Fmin, Fmax, A0, L0, p0, aF, aA, lam, kL, args.h_amp, z_mu, z_sd, args.center, args.gap, steps=50)

    plt.figure(figsize=(6,4))
    plt.plot(F_up, B_up, label="up-sweep")
    plt.plot(F_dn, B_dn, label="down-sweep")
    plt.xlabel("F"); plt.ylabel("B* (steady)"); plt.title(f"Subject {args.subject}: Hysteresis vs F")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.subject}_hysteresis_F.png"), dpi=200); plt.close()

    # E=0/1 nullclines at (A0,L0)
    F_axis = np.linspace(Fmin, Fmax, 200)
    B0 = nullcline_B(F_axis, A0, L0, p0, aF, aA, lam, kL, args.h_amp, 0.0)
    B1 = nullcline_B(F_axis, A0, L0, p0, aF, aA, lam, kL, args.h_amp, 1.0)
    plt.figure(figsize=(6,4))
    plt.plot(F_axis, B0, label="B-nullcline (E=0)")
    plt.plot(F_axis, B1, label="B-nullcline (E=1)")
    plt.xlabel("F"); plt.ylabel("B-nullcline"); plt.title(f"Subject {args.subject}: B-nullclines at (A0,L0)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.subject}_nullclines.png"), dpi=200); plt.close()

    print("Saved multistability figures to", args.outdir)

if __name__ == "__main__":
    main()
