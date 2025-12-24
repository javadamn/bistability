#!/usr/bin/env python3
import argparse, os, re, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def r2_stats(series):
    s = pd.Series(series).astype(float).dropna()
    if s.empty:
        return {"n":0,"mean":np.nan,"median":np.nan,"p75":np.nan,"p90":np.nan,"ge_0_30":0,"ge_0_40":0,"ge_0_50":0}
    return {
        "n": int(s.shape[0]),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "p75": float(np.percentile(s, 75)),
        "p90": float(np.percentile(s, 90)),
        "ge_0_30": int(np.sum(s >= 0.30)),
        "ge_0_40": int(np.sum(s >= 0.40)),
        "ge_0_50": int(np.sum(s >= 0.50)),
    }

def subjcol(df):
    for c in df.columns:
        if isinstance(c,str) and re.search(r"^subject", c, flags=re.IGNORECASE): return c
    return df.columns[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_v4", required=True)
    ap.add_argument("--fit_v3", default="")
    ap.add_argument("--global_json", default="")
    ap.add_argument("--r2_thresh", type=float, default=0.40)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    v4 = pd.read_csv(args.fit_v4)
    sc = subjcol(v4); v4 = v4.rename(columns={sc:"subject"})
    v4_stats = pd.DataFrame([r2_stats(v4["R2_logspace"])]).assign(model="v4")

    if args.fit_v3 and os.path.exists(args.fit_v3):
        v3 = pd.read_csv(args.fit_v3)
        sc3 = subjcol(v3); v3 = v3.rename(columns={sc3:"subject"})
        v3_stats = pd.DataFrame([r2_stats(v3["R2_logspace"])]).assign(model="v3")
        merged = v3[["subject","R2_logspace"]].merge(v4[["subject","R2_logspace"]], on="subject", suffixes=("_v3","_v4"))
        merged["delta"] = merged["R2_logspace_v4"] - merged["R2_logspace_v3"]
        merged.to_csv(os.path.join(args.outdir,"r2_comparison_v3_v4.csv"), index=False)
    else:
        v3_stats = pd.DataFrame(columns=["n","mean","median","p75","p90","ge_0_30","ge_0_40","ge_0_50","model"])
        merged = None

    pd.concat([v3_stats, v4_stats], ignore_index=True).to_csv(os.path.join(args.outdir,"r2_summary.csv"), index=False)

    cohort = v4[v4["R2_logspace"] >= args.r2_thresh][["subject","R2_logspace","n"]].sort_values("R2_logspace", ascending=False)
    cohort.to_csv(os.path.join(args.outdir, "cohort_subjects.csv"), index=False)

    plt.figure(figsize=(6,4))
    s = v4["R2_logspace"].dropna().values
    plt.hist(s, bins=20)
    plt.xlabel("R² (log-space)"); plt.ylabel("Count"); plt.title("v4 R² distribution"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,"r2_hist_v4.png"), dpi=200); plt.close()

    if 'merged' in locals() and merged is not None:
        plt.figure(figsize=(6,4))
        d = merged["delta"].dropna().values
        plt.hist(d, bins=20)
        plt.xlabel("ΔR² (v4 - v3)"); plt.ylabel("Count"); plt.title("Change in R² after cross-feed"); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir,"delta_hist_v4_minus_v3.png"), dpi=200); plt.close()

    if args.global_json and os.path.exists(args.global_json):
        import json
        with open(args.global_json, "r") as f:
            gj = json.load(f)
        with open(os.path.join(args.outdir,"global_params_echo.json"), "w") as f:
            json.dump(gj, f, indent=2)

    print("Wrote outputs to", args.outdir)

if __name__ == "__main__":
    main()
