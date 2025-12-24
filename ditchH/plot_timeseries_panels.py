#!/usr/bin/env python3
import argparse, os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def subjcol(df):
    for c in df.columns:
        if isinstance(c,str) and re.search(r"^subject", c, flags=re.IGNORECASE): return c
    return df.columns[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_v4", required=True)
    ap.add_argument("--subjects_csv", required=True, help="CSV with column 'subject'")
    ap.add_argument("--max_plots", type=int, default=6)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pred = pd.read_csv(args.pred_v4)
    sc = subjcol(pred); pred = pred.rename(columns={sc:"subject"})
    subs = pd.read_csv(args.subjects_csv)
    if "subject" not in subs.columns:
        ss = subjcol(subs); subs = subs.rename(columns={ss:"subject"})
    chosen = subs["subject"].astype(str).head(args.max_plots).tolist()

    for sid in chosen:
        g = pred[pred["subject"].astype(str)==str(sid)].copy()
        if g.empty: continue
        idx = np.arange(g.shape[0])
        plt.figure(figsize=(7,4))
        plt.plot(idx, g["B_obs"].astype(float).values, label="B_obs")
        plt.plot(idx, g["B_hat"].astype(float).values, label="B_hat")
        plt.xlabel("Aligned step"); plt.ylabel("Butyrate (arb.)"); plt.title(f"Subject {sid}: B_obs vs B_hat")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"subject_{sid}_butyrate.png"), dpi=200); plt.close()

        plt.figure(figsize=(7,4))
        if "F_in" in g.columns: plt.plot(idx, g["F_in"].astype(float).values, label="F_in")
        if "A_in" in g.columns: plt.plot(idx, g["A_in"].astype(float).values, label="A_in")
        if "L_in" in g.columns: plt.plot(idx, g["L_in"].astype(float).values, label="L_in")
        plt.xlabel("Aligned step"); plt.ylabel("Inputs (scaled)"); plt.title(f"Subject {sid}: Drivers")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"subject_{sid}_drivers.png"), dpi=200); plt.close()

    print("Wrote subject panels to", args.outdir)

if __name__ == "__main__":
    main()
