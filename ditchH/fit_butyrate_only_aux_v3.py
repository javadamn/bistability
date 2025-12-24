#!/usr/bin/env python3
import argparse, os, re, json, numpy as np, pandas as pd

def guess_col(df, pats, default=None):
    for p in pats:
        for c in df.columns:
            if isinstance(c, str) and re.search(p, c, flags=re.IGNORECASE):
                return c
    return default if default else df.columns[0]

def r2_score(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = np.nansum((y - yhat)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1.0 - (ss_res/(ss_tot + 1e-12))

def simulate_forward(B0, F, A, dt, p0, aF, aA, lam):
    n = len(F); B = np.zeros(n+1, float); B[0] = B0
    for t in range(n):
        dBdt = p0 + aF*F[t] - aA*A[t] - lam*B[t]
        B[t+1] = max(0.0, B[t] + dt[t]*dBdt)
    return B[1:]

def huber_residuals(e, delta=0.5):
    e = np.asarray(e, float)
    mask = np.abs(e) <= delta
    out = np.empty_like(e)
    out[mask] = 0.5*e[mask]**2
    out[~mask] = delta*(np.abs(e[~mask]) - 0.5*delta)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lambda_grid", default="0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3,0.45,0.6")
    ap.add_argument("--min_obs", type=int, default=6)
    ap.add_argument("--min_sd_zB", type=float, default=0.2)
    ap.add_argument("--use_smoothed", action="store_true")
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--lag_F", type=int, default=0, help="Use F(t-lag_F)")
    ap.add_argument("--lag_A", type=int, default=0, help="Use A(t-lag_A)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    subj_col = guess_col(df, [r"^subject", r"participant", r"^id$"])
    date_col = guess_col(df, [r"^date of receipt$", r"^date$", r"collection", r"sample.*date", r"timestamp", r"interval sequence"])
    B_col = None
    for c in df.columns:
        if isinstance(c,str) and re.search(r"butyrate|c4\\b|butyric", c, flags=re.IGNORECASE):
            B_col = c; break
    if B_col is None: raise ValueError("No butyrate column found.")
    if re.search("date", date_col, flags=re.IGNORECASE):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df[date_col] = pd.to_numeric(df[date_col], errors="coerce")

    Fname = "F_met_s" if (args.use_smoothed and "F_met_s" in df.columns) else "F_met"
    Aname = "A_met_s" if (args.use_smoothed and "A_met_s" in df.columns) else "A_met"
    if Fname not in df.columns or Aname not in df.columns:
        raise ValueError("Missing F/A inputs.")

    # QC on zB
    zB = np.log1p(df[B_col].astype(float))
    zB = (zB - np.nanmean(zB))/ (np.nanstd(zB)+1e-8)
    df["__zB__"] = zB

    lam_grid = [float(x) for x in args.lambda_grid.split(",") if x.strip()!=""]
    os.makedirs(args.outdir, exist_ok=True)

    def build_aligned(g):
        # Returns aligned arrays for derivative regression with lags
        B = g[B_col].astype(float).values
        if len(B) < max(args.min_obs, 4): return None
        t = g[date_col].values
        if np.issubdtype(g[date_col].dtype, np.datetime64):
            tt = t.astype("datetime64[ns]").astype("int64")/1e9
            dt = np.diff(tt)/86400.0
        else:
            tt = t.astype(float); dt = np.diff(tt).astype(float)
        if len(dt)==0 or np.any(~np.isfinite(dt)) or np.any(dt<=0): return None

        F = g[Fname].values.astype(float)
        A = g[Aname].values.astype(float)
        # indices for derivative at t: 0..m-2
        m = len(B)
        t_idx = np.arange(0, m-1, dtype=int)
        # apply lags: need t - lag >= 0
        keep = (t_idx - args.lag_F >= 0) & (t_idx - args.lag_A >= 0)
        if keep.sum() < args.min_obs-1: return None
        t_idx = t_idx[keep]

        By = B[t_idx]                     # B(t)
        ydot = (B[t_idx+1] - B[t_idx]) / dt[t_idx]  # dB/dt over [t,t+1]
        F_use = F[t_idx - args.lag_F]
        A_use = A[t_idx - args.lag_A]
        dt_use = dt[t_idx]

        return B, F, A, dt, By, ydot, F_use, A_use, dt_use, t_idx

    # Build subject bundles
    bundles = []
    for sid, grp in df.groupby(subj_col):
        g = grp.sort_values(date_col).copy()
        if len(g) < args.min_obs: 
            continue
        if np.nanstd(g["__zB__"].values) < args.min_sd_zB:
            continue
        aligned = build_aligned(g)
        if aligned is None: 
            continue
        B, F, A, dt, By, ydot, F_use, A_use, dt_use, t_idx = aligned
        bundles.append((sid, g, B, F, A, dt, By, ydot, F_use, A_use, dt_use, t_idx))

    if not bundles:
        raise RuntimeError("No subjects passed QC and alignment constraints.")

    # Grid search lambda (global) minimizing robust loss in log-space
    sse_by_lam = []
    for lam in lam_grid:
        total = 0.0
        for (sid, g, B, F, A, dt, By, ydot, F_use, A_use, dt_use, t_idx) in bundles:
            X = np.column_stack([np.ones_like(F_use), F_use, A_use])
            yprime = ydot + lam*By
            beta, *_ = np.linalg.lstsq(X, yprime, rcond=None)
            p0, aF, aA = beta.tolist()

            # simulate only over aligned indices window
            # Start from B[t0]; simulate sequentially using dt_use and inputs
            t0 = t_idx.min()
            B0 = B[t0]
            # Need contiguous steps from t0..t_last inclusive
            # Build sequences in that contiguous window
            # For simplicity, we'll just simulate per-step using the aligned slice order
            Bhat = np.zeros_like(By)
            cur = B0
            for i in range(len(By)):
                dBdt = p0 + aF*F_use[i] - aA*A_use[i] - lam*cur
                cur = max(0.0, cur + dt_use[i]*dBdt)
                Bhat[i] = cur
            z_obs = np.log1p(B[t_idx+1])
            z_hat_raw = np.log1p(Bhat)

            Xo = np.column_stack([np.ones_like(z_hat_raw), z_hat_raw])
            theta, *_ = np.linalg.lstsq(Xo, z_obs, rcond=None)
            z_hat = Xo.dot(theta)

            e = z_obs - z_hat
            total += float(np.nansum(huber_residuals(e, delta=args.delta)))
        sse_by_lam.append((lam, total))

    lam_best, _ = min(sse_by_lam, key=lambda x: x[1])

    # Refit per subject with lam_best and write outputs
    rows_params, rows_pred, rows_summary = [], [], []
    for (sid, g, B, F, A, dt, By, ydot, F_use, A_use, dt_use, t_idx) in bundles:
        X = np.column_stack([np.ones_like(F_use), F_use, A_use])
        yprime = ydot + lam_best*By
        beta, *_ = np.linalg.lstsq(X, yprime, rcond=None)
        p0, aF, aA = beta.tolist()

        # simulate on aligned window
        t0 = t_idx.min()
        B0 = B[t0]
        Bhat = np.zeros_like(By)
        cur = B0
        for i in range(len(By)):
            dBdt = p0 + aF*F_use[i] - aA*A_use[i] - lam_best*cur
            cur = max(0.0, cur + dt_use[i]*dBdt)
            Bhat[i] = cur

        z_obs = np.log1p(B[t_idx+1])
        z_hat_raw = np.log1p(Bhat)
        Xo = np.column_stack([np.ones_like(z_hat_raw), z_hat_raw])
        theta, *_ = np.linalg.lstsq(Xo, z_obs, rcond=None)
        z_hat = Xo.dot(theta)
        R2 = r2_score(z_obs, z_hat)

        rows_params.append({
            "subject": sid, "p0": p0, "alpha_F": aF, "alpha_A": aA, "lambda_B": lam_best,
            "alpha_obs": theta[0], "beta_obs": theta[1], "n_obs": int(len(g)),
            "lag_F": args.lag_F, "lag_A": args.lag_A
        })
        # write predictions for aligned rows only
        for i in range(len(By)):
            rows_pred.append({
                "subject": sid,
                "aligned_index": int(t_idx[i]+1),
                "B_obs": float(B[t_idx[i]+1]),
                "B_hat": float(Bhat[i]),
                "z_obs": float(z_obs[i]),
                "z_hat": float(z_hat[i]),
                "F_in": float(F_use[i]),
                "A_in": float(A_use[i]),
                "lag_F": args.lag_F, "lag_A": args.lag_A
            })
        rows_summary.append({"subject": sid, "R2_logspace": R2, "n": int(len(g)), "lag_F": args.lag_F, "lag_A": args.lag_A})

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(rows_params).to_csv(os.path.join(args.outdir, "params_v3.csv"), index=False)
    pd.DataFrame(rows_pred).to_csv(os.path.join(args.outdir, "predictions_v3.csv"), index=False)
    pd.DataFrame(rows_summary).to_csv(os.path.join(args.outdir, "fit_summary_v3.csv"), index=False)
    with open(os.path.join(args.outdir, "global_fit_v3.json"), "w") as f:
        json.dump({"lambda_grid": lam_grid, "robust_sse_by_lambda": sse_by_lam, "lambda_best": lam_best,
                   "lag_F": args.lag_F, "lag_A": args.lag_A}, f, indent=2)
    print("v3 done. Outputs ->", args.outdir)

if __name__ == "__main__":
    main()
