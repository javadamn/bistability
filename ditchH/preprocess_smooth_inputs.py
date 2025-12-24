#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, re

def smooth_series(x, window=3):
    s = pd.Series(x)
    y = s.rolling(window=window, center=True, min_periods=1).median().values
    return y

def stepify(y, eps=0.1):
    out = y.copy()
    for i in range(1, len(out)):
        if np.isfinite(out[i]) and np.isfinite(out[i-1]):
            if abs(out[i] - out[i-1]) < eps:
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
    # detect subject and time columns
    subj_col = None
    for c in df.columns:
        if re.search(r"^subject", c, flags=re.IGNORECASE): subj_col=c; break
    if subj_col is None:
        subj_col = "subject_id" if "subject_id" in df.columns else df.columns[0]

    date_col = None
    for cand in ["Date of Receipt","Date","date","collection_date","sample_date","timestamp","Interval Sequence"]:
        if cand in df.columns: date_col = cand; break
    if date_col is None:
        for c in df.columns:
            if re.search(r"date|collection|time|interval", str(c), flags=re.IGNORECASE):
                date_col = c; break

    # ensure order
    if date_col in df.columns:
        if re.search("date", date_col, flags=re.IGNORECASE):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df[date_col] = pd.to_numeric(df[date_col], errors="coerce")
        df = df.sort_values([subj_col, date_col])

    out = df.copy()
    out["F_met_s"] = np.nan
    out["A_met_s"] = np.nan

    for sid, g in df.groupby(subj_col):
        idx = g.index
        if "F_met" in g.columns:
            F = g["F_met"].astype(float).values
            Fs = stepify(smooth_series(F, window=args.window), eps=args.epsilon)
            out.loc[idx, "F_met_s"] = Fs
        if "A_met" in g.columns:
            A = g["A_met"].astype(float).values
            As = stepify(smooth_series(A, window=args.window), eps=args.epsilon)
            out.loc[idx, "A_met_s"] = As

    out.to_csv(args.csv_out, index=False)
    print("Wrote", args.csv_out)

if __name__ == "__main__":
    main()
