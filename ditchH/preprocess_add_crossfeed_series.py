#!/usr/bin/env python3
import argparse, re, numpy as np, pandas as pd

def minmax01(a):
    a = np.asarray(a, float); lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: return np.zeros_like(a)
    return (a - lo) / (hi - lo)

def smooth_series(x, window=3):
    s = pd.Series(x); return s.rolling(window=window, center=True, min_periods=1).median().values

def stepify(y, eps=0.1):
    out = y.copy()
    for i in range(1, len(out)):
        if np.isfinite(out[i]) and np.isfinite(out[i-1]) and abs(out[i] - out[i-1]) < eps:
            out[i] = out[i-1]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True)
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    subj_col = None
    for c in df.columns:
        if re.search(r"^subject", c, flags=re.IGNORECASE): subj_col=c; break
    if subj_col is None:
        for c in df.columns:
            if re.search(r"subject|participant|id", str(c), flags=re.IGNORECASE):
                subj_col=c; break

    lactate_col = None
    if "aux_lactate_z" in df.columns:
        lactate_col = "aux_lactate_z"
    else:
        for c in df.columns:
            if isinstance(c,str) and re.search(r"lactate", c, flags=re.IGNORECASE):
                lactate_col = c; break
    if lactate_col is None: raise ValueError("No lactate column found.")

    out = df.copy(); out["L_cf"] = np.nan; out["L_cf_s"] = np.nan
    for sid, g in df.groupby(subj_col):
        idx = g.index; x = g[lactate_col].astype(float).values
        if lactate_col != "aux_lactate_z":
            xz = np.log1p(x); mu, sd = np.nanmean(xz), np.nanstd(xz)
            xz = (xz - mu)/(sd+1e-8) if np.isfinite(sd) and sd >= 1e-8 else np.zeros_like(xz)
        else:
            xz = x.copy()
        L = minmax01(xz); Ls = stepify(smooth_series(L, window=args.window), eps=args.epsilon)
        out.loc[idx, "L_cf"] = L; out.loc[idx, "L_cf_s"] = Ls

    out.to_csv(args.csv_out, index=False); print("Wrote", args.csv_out)

if __name__ == "__main__":
    main()
