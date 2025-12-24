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
    ap.add_argument("--use_smoothed", action="store_true", help="Use F_met_s/A_met_s if present")
    ap.add_argument("--delta", type=float, default=0.5, help="Huber delta")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    subj_col = guess_col(df, [r"^subject", r"participant", r"^id$"])
    date_col = guess_col(df, [r"^date of receipt$", r"^date$", r"collection", r"sample.*date", r"timestamp", r"interval sequence"])
    B_col = None
    for c in df.columns:
        if isinstance(c,str) and re.search(r"butyrate|c4\\b|butyric", c, flags=re.IGNORECASE):
            B_col = c; break
    if B_col is None: raise ValueError("No butyrate column found.")

    # time
    if re.search("date", date_col, flags=re.IGNORECASE):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df[date_col] = pd.to_numeric(df[date_col], errors="coerce")

    # choose inputs
    Fname = "F_met_s" if (args.use_smoothed and "F_met_s" in df.columns) else "F_met"
    Aname = "A_met_s" if (args.use_smoothed and "A_met_s" in df.columns) else "A_met"
    if Fname not in df.columns or Aname not in df.columns:
        raise ValueError("Missing F/A inputs. Run preprocess steps first.")

    # compute zB variability to filter subjects
    zB = np.log1p(df[B_col].astype(float))
    zB = (zB - np.nanmean(zB))/ (np.nanstd(zB)+1e-8)
    df["__zB__"] = zB

    lam_grid = [float(x) for x in args.lambda_grid.split(",") if x.strip()!=""]
    os.makedirs(args.outdir, exist_ok=True)

    # Build subject bundles
    subj_bundles = []
    for sid, grp in df.groupby(subj_col):
        g = grp.sort_values(date_col).copy()
        if len(g) < args.min_obs: 
            continue
        if np.nanstd(g["__zB__"].values) < args.min_sd_zB:
            continue
        B = g[B_col].astype(float).values
        t = g[date_col].values
        if np.issubdtype(g[date_col].dtype, np.datetime64):
            tt = t.astype("datetime64[ns]").astype("int64")/1e9
            dt = np.diff(tt)/86400.0
        else:
            tt = t.astype(float); dt = np.diff(tt).astype(float)
        if len(dt)==0 or np.any(~np.isfinite(dt)) or np.any(dt<=0):
            continue
        F = g[Fname].values.astype(float); A = g[Aname].values.astype(float)
        subj_bundles.append((sid, g, B, F, A, dt))

    if not subj_bundles:
        raise RuntimeError("No subjects passed QC thresholds.")

    # Evaluate lambda grid by robust loss in log space with obs link (alpha,beta)
    sse_by_lam = []
    for lam in lam_grid:
        total = 0.0
        for sid, g, B, F, A, dt in subj_bundles:
            # Derivative regression to get initial (p0,aF,aA)
            By = B[:-1]; ydot = (B[1:] - B[:-1]) / dt
            X = np.column_stack([np.ones_like(F[:-1]), F[:-1], A[:-1]])
            yprime = ydot + lam*By
            beta, *_ = np.linalg.lstsq(X, yprime, rcond=None)
            p0, aF, aA = beta.tolist()

            # Simulate forward
            Bhat = simulate_forward(B0=B[0], F=F[:-1], A=A[:-1], dt=dt, p0=p0, aF=aF, aA=aA, lam=lam)
            z_obs = np.log1p(B[1:]); z_hat_raw = np.log1p(Bhat)

            # Observation linear link alpha+beta*z_hat_raw ~= z_obs
            Xo = np.column_stack([np.ones_like(z_hat_raw), z_hat_raw])
            theta, *_ = np.linalg.lstsq(Xo, z_obs, rcond=None)  # [alpha, beta]
            z_hat = Xo.dot(theta)

            # Robust loss
            total += float(np.nansum(huber_residuals(z_obs - z_hat, delta=args.delta)))
        sse_by_lam.append((lam, total))

    lam_best, sse_best = min(sse_by_lam, key=lambda x: x[1])

    rows_params, rows_pred, rows_summary = [], [], []
    for sid, g, B, F, A, dt in subj_bundles:
        By = B[:-1]; ydot = (B[1:] - B[:-1]) / dt
        X = np.column_stack([np.ones_like(F[:-1]), F[:-1], A[:-1]])
        yprime = ydot + lam_best*By
        beta, *_ = np.linalg.lstsq(X, yprime, rcond=None)
        p0, aF, aA = beta.tolist()

        Bhat = simulate_forward(B0=B[0], F=F[:-1], A=A[:-1], dt=dt, p0=p0, aF=aF, aA=aA, lam=lam_best)
        z_obs = np.log1p(B[1:]); z_hat_raw = np.log1p(Bhat)
        Xo = np.column_stack([np.ones_like(z_hat_raw), z_hat_raw])
        theta, *_ = np.linalg.lstsq(Xo, z_obs, rcond=None)
        z_hat = Xo.dot(theta)

        R2 = r2_score(z_obs, z_hat)
        rows_params.append({"subject": sid, "p0": p0, "alpha_F": aF, "alpha_A": aA, "lambda_B": lam_best, "alpha_obs": theta[0], "beta_obs": theta[1], "n_obs": len(g)})
        for i in range(1, len(g)):
            rows_pred.append({
                "subject": sid,
                "date": g.iloc[i][guess_col(g, [r'^date of receipt$', r'^date$', r'interval sequence'])],
                "B_obs": float(B[i]),
                "B_hat": float(Bhat[i-1]),
                "z_obs": float(z_obs[i-1]),
                "z_hat": float(z_hat[i-1]),
                "F_in": float(F[i-1]),
                "A_in": float(A[i-1])
            })
        span_days = None
        if np.issubdtype(g[guess_col(g,[r'^date of receipt$', r'^date$', r'interval sequence'])].dtype, np.datetime64):
            span_days = (g[guess_col(g,[r'^date of receipt$', r'^date$', r'interval sequence'])].iloc[-1] - g[guess_col(g,[r'^date of receipt$', r'^date$', r'interval sequence'])].iloc[0]).days
        rows_summary.append({"subject": sid, "R2_logspace": R2, "n": len(g), "time_span_days": span_days})

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(rows_params).to_csv(os.path.join(args.outdir, "params_v2.csv"), index=False)
    pd.DataFrame(rows_pred).to_csv(os.path.join(args.outdir, "predictions_v2.csv"), index=False)
    pd.DataFrame(rows_summary).to_csv(os.path.join(args.outdir, "fit_summary_v2.csv"), index=False)
    with open(os.path.join(args.outdir, "global_fit_v2.json"), "w") as f:
        json.dump({"lambda_grid": lam_grid, "robust_sse_by_lambda": sse_by_lam, "lambda_best": lam_best}, f, indent=2)

    print("v2 done. Outputs ->", args.outdir)

if __name__ == "__main__":
    main()
