#!/usr/bin/env python3
"""
Derive cross-feed series L_cf and smoothed L_cf_s from lactate.

Per subject:
- If aux_lactate_z exists: min-max to [0,1]
- Else: first lactate-like column -> log1p -> per-subject z-score -> min-max to [0,1]
- Rolling median smoothing (centered) with window (default 3)
- Light "stepify": if adjacent values differ < epsilon, make them equal
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def detect_subject_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in cols:
        if re.search(r"^subject", str(c), flags=re.IGNORECASE):
            return c
    for c in cols:
        if re.search(r"(subject|participant|patient).*id|(^|_)id($|_)", str(c), flags=re.IGNORECASE):
            return c
    for c in cols:
        if re.search(r"(subject|participant|patient)", str(c), flags=re.IGNORECASE):
            return c
    raise SystemExit("ERROR: Could not detect subject id column.")


def find_lactate_col(df: pd.DataFrame) -> str:
    # Priority: aux_lactate_z
    for c in df.columns:
        if re.search(r"^aux_lactate_z$", str(c), flags=re.IGNORECASE):
            return c
    for c in df.columns:
        if re.search(r"aux_lactate_z", str(c), flags=re.IGNORECASE):
            return c
    # Else: any lactate-like
    for c in df.columns:
        if re.search(r"lactate", str(c), flags=re.IGNORECASE):
            return c
    raise SystemExit("ERROR: Could not find lactate column (aux_lactate_z or /lactate/i).")


def minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    m = np.nanmin(x)
    M = np.nanmax(x)
    if not np.isfinite(m) or not np.isfinite(M) or (M - m) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - m) / (M - m)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(mu) or not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def stepify(x: np.ndarray, eps: float) -> np.ndarray:
    y = x.copy().astype(float)
    if len(y) <= 1:
        return y
    for i in range(1, len(y)):
        if np.isfinite(y[i]) and np.isfinite(y[i - 1]) and abs(y[i] - y[i - 1]) < eps:
            y[i] = y[i - 1]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True)
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    subj_col = detect_subject_col(df)
    lac_col = find_lactate_col(df)

    use_aux = bool(re.search(r"aux_lactate_z", lac_col, flags=re.IGNORECASE))

    L_cf_all = []
    L_cf_s_all = []

    for sid, g in df.groupby(subj_col, sort=False):
        x = pd.to_numeric(g[lac_col], errors="coerce").to_numpy(dtype=float)

        if use_aux:
            L = minmax01(x)
        else:
            # log1p -> zscore -> minmax
            x2 = np.log1p(np.clip(x, 0, None))
            x2 = zscore(x2)
            L = minmax01(x2)

        L_s = rolling_median(L, window=max(1, int(args.window)))
        L_s = stepify(L_s, eps=float(args.epsilon))

        L_cf_all.append(pd.Series(L, index=g.index))
        L_cf_s_all.append(pd.Series(L_s, index=g.index))

    df["L_cf"] = pd.concat(L_cf_all).sort_index()
    df["L_cf_s"] = pd.concat(L_cf_s_all).sort_index()

    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    print(f"Wrote: {args.csv_out}")
    print("Added columns: L_cf, L_cf_s")


if __name__ == "__main__":
    main()
