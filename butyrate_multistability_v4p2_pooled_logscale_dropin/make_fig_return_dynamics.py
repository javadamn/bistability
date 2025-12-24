#!/usr/bin/env python3
# Python 3.10; deps: pandas, numpy, matplotlib
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_col(cols, patterns):
    for p in patterns:
        rx = re.compile(p, re.I)
        for c in cols:
            if rx.search(str(c)):
                return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="e.g., combined_scfas_table_scored_plus_met_cf.csv")
    ap.add_argument("--subject", default=None, help="subject id; if omitted, use the first subject found")
    ap.add_argument("--out_png", default="fig_return_dynamics.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    subj_col = find_col(df.columns, [r"^subject", r"subject|participant|id"])
    date_col = find_col(df.columns, [r"^date$|^date of receipt$|collection|sample.*date|timestamp|interval sequence"])
    b_col = find_col(df.columns, [r"butyrate|butyric|(^|[^A-Za-z])c4([^A-Za-z]|$)"])
    if subj_col is None or date_col is None or b_col is None:
        raise ValueError(f"Missing required columns. Found subject={subj_col}, date={date_col}, butyrate={b_col}")

    # Parse date if possible; if not numeric, try datetime
    date_series = df[date_col]
    if np.issubdtype(date_series.dtype, np.number):
        df["_t"] = pd.to_numeric(date_series, errors="coerce")
    else:
        dt = pd.to_datetime(date_series, errors="coerce")
        if dt.notna().sum() > 0:
            df["_t"] = dt.view("int64") / (1e9 * 86400.0)  # days
        else:
            df["_t"] = pd.to_numeric(date_series, errors="coerce")

    df = df.dropna(subset=[subj_col, "_t", b_col]).copy()

    subject = args.subject
    if subject is None:
        subject = str(df[subj_col].iloc[0])

    d = df[df[subj_col].astype(str) == str(subject)].copy()
    if d.shape[0] < 3:
        raise ValueError(f"Not enough points for subject {subject}")

    d = d.sort_values("_t")
    B = pd.to_numeric(d[b_col], errors="coerce").to_numpy()
    z = np.log1p(np.clip(B, a_min=0, a_max=None))

    # transitions
    z_t = z[:-1]
    dz = z[1:] - z[:-1]

    # Fit dz = a + b z (OLS)
    X = np.column_stack([np.ones_like(z_t), z_t])
    coef, _, _, _ = np.linalg.lstsq(X, dz, rcond=None)
    a_hat, b_hat = coef[0], coef[1]

    # Plot
    fig = plt.figure(figsize=(6.8, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(z_t, dz, s=28)
    zz = np.linspace(np.nanmin(z_t), np.nanmax(z_t), 200)
    ax.plot(zz, a_hat + b_hat * zz, linewidth=2)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel(r"$z_t=\log(1+B_t)$")
    ax.set_ylabel(r"$\Delta z_t = z_{t+1}-z_t$")
    ax.set_title(f"Return dynamics (subject {subject}): slope b={b_hat:.3g}")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Wrote: {args.out_png}")

if __name__ == "__main__":
    main()
