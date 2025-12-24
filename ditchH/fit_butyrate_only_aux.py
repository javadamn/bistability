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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lambda_grid", default="0.05,0.1,0.15,0.2,0.25,0.3")
    ap.add_argument("--min_obs", type=int, default=5)
    ap.add_argument("--aux_mets", default="")   # e.g., "aux_propionate_z,aux_lactate_z"
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    subj_col = guess_col(df, [r"^subject", r"participant", r"^id$"])
    date_col = guess_col(df, [r"^date of receipt$", r"^date$", r"collection", r"sample.*date", r"timestamp", r"interval sequence"])
    B_col = None
    for c in df.columns:
        if isinstance(c,str) and re.search(r"butyrate|c4\b|butyric", c, flags=re.IGNORECASE):
            B_col = c; break
    if B_col is None:
        raise ValueError("Could not locate a butyrate column by name.")

    # parse datetimes or numeric order
    if re.search("date", date_col, flags=re.IGNORECASE):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df[date_col] = pd.to_numeric(df[date_col], errors="coerce")

    need_cols = ["F_met", "A_met", B_col, subj_col, date_col, "Sample_ID" if "Sample_ID" in df.columns else "sample_id"]
    for c in need_cols:
        if c not in df.columns:
            # allow either Sample_ID or sample_id
            if c in ("Sample_ID", "sample_id"):
                continue
            raise ValueError(f"Missing required column: {c}")
    sidcol = "Sample_ID" if "Sample_ID" in df.columns else "sample_id"

    df = df.dropna(subset=[B_col, "F_met", "A_met", subj_col, date_col]).copy()

    os.makedirs(args.outdir, exist_ok=True)
    lam_grid = [float(x) for x in args.lambda_grid.split(",") if x.strip()!=""]
    sse_by_lam = []

    subjects = []
    for sid, grp in df.groupby(subj_col):
        g = grp.sort_values(date_col).copy()
        if len(g) < args.min_obs: 
            continue
        B = g[B_col].astype(float).values
        t = g[date_col].values
        if np.issubdtype(g[date_col].dtype, np.datetime64):
            t = t.astype("datetime64[ns]").astype("int64")/1e9
        t = t.astype(float)
        dt = np.diff(t); 
        if np.issubdtype(g[date_col].dtype, np.datetime64):
            dt = dt/86400.0  # seconds to days
        if len(dt)==0 or np.any(~np.isfinite(dt)) or np.any(dt<=0):
            continue
        F = g["F_met"].values[:-1].astype(float)
        A = g["A_met"].values[:-1].astype(float)
        By = B[:-1]
        ydot = (B[1:] - B[:-1]) / dt
        subjects.append((sid, By, F, A, ydot, dt, g))

    if not subjects:
        raise RuntimeError("No subjects have sufficient observations to fit.")

    for lam in lam_grid:
        SSE = 0.0
        for sid, By, F, A, ydot, dt, g in subjects:
            yprime = ydot + lam*By
            X = np.column_stack([np.ones_like(F), F, A])
            beta, *_ = np.linalg.lstsq(X, yprime, rcond=None)
            resid = yprime - X.dot(beta)
            SSE += float(np.nansum(resid**2))
        sse_by_lam.append((lam, SSE))

    lam_best, sse_best = min(sse_by_lam, key=lambda x: x[1])

    rows_params, rows_pred, rows_summary = [], [], []
    for sid, By, F, A, ydot, dt, g in subjects:
        yprime = ydot + lam_best*By
        X = np.column_stack([np.ones_like(F), F, A])
        beta, *_ = np.linalg.lstsq(X, yprime, rcond=None)
        p0, aF, aA = beta.tolist()
        B = g[B_col].astype(float).values
        Ft = g["F_met"].values.astype(float)
        At = g["A_met"].values.astype(float)
        tt = g[date_col].values
        if np.issubdtype(g[date_col].dtype, np.datetime64):
            tt = tt.astype("datetime64[ns]").astype("int64")/1e9
            dt_all = np.diff(tt)/86400.0
        else:
            dt_all = np.diff(tt).astype(float)
        Bhat = simulate_forward(B0=B[0], F=Ft[:-1], A=At[:-1], dt=dt_all, p0=p0, aF=aF, aA=aA, lam=lam_best)
        r2 = r2_score(B[1:], Bhat)

        rows_params.append({"subject": sid, "p0": p0, "alpha_F": aF, "alpha_A": aA, "lambda_B": lam_best, "n_obs": len(g)})
        for i in range(1, len(g)):
            rows_pred.append({
                "subject": sid,
                "Sample_ID": g.iloc[i][sidcol],
                "date": g.iloc[i][date_col],
                "B_obs": float(B[i]),
                "B_hat": float(Bhat[i-1]),
                "F_met": float(Ft[i-1]),
                "A_met": float(At[i-1])
            })
        span_days = None
        if np.issubdtype(g[date_col].dtype, np.datetime64):
            span_days = (g[date_col].iloc[-1] - g[date_col].iloc[0]).days
        rows_summary.append({"subject": sid, "R2": r2, "n": len(g), "time_span_days": span_days})

    pd.DataFrame(rows_params).to_csv(os.path.join(args.outdir, "params.csv"), index=False)
    pd.DataFrame(rows_pred).to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)
    pd.DataFrame(rows_summary).to_csv(os.path.join(args.outdir, "fit_summary.csv"), index=False)

    with open(os.path.join(args.outdir, "global_fit.json"), "w") as f:
        json.dump({"lambda_grid": lam_grid, "sse_by_lambda": sse_by_lam, "lambda_best": lam_best}, f, indent=2)

    with open(os.path.join(args.outdir, "log.txt"), "w") as f:
        f.write("Evaluated lambda grid (lambda, SSE):\n")
        for lam, sse in sse_by_lam:
            f.write(f"{lam:.6g}\t{sse:.6g}\n")
        f.write(f"\nChosen lambda: {lam_best:.6g}\n")

    print("Done. Outputs written to", args.outdir)

if __name__ == "__main__":
    main()
