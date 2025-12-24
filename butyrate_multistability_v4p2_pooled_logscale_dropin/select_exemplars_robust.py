#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_summary", required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--r2_thresh", type=float, default=0.40)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    fs = pd.read_csv(args.fit_summary)
    fs["subject"] = fs["subject"].astype(str)
    fs["R2_logspace"] = pd.to_numeric(fs["R2_logspace"], errors="coerce")
    fs["n_aligned"] = pd.to_numeric(fs["n_aligned"], errors="coerce")

    # 1) try R2 threshold
    passed = fs.dropna(subset=["R2_logspace"]).query("R2_logspace >= @args.r2_thresh").copy()
    if len(passed) > 0:
        out = passed.sort_values(["R2_logspace", "n_aligned"], ascending=False).head(args.k)[["subject"]]
        out.to_csv(args.out_csv, index=False)
        print(f"Wrote exemplars (by R2>=thresh): {args.out_csv}")
        print(out)
        return

    # 2) fallback: top by n_aligned
    if fs["n_aligned"].notna().any():
        out = fs.sort_values(["n_aligned", "R2_logspace"], ascending=False).head(args.k)[["subject"]]
        if len(out) > 0:
            out.to_csv(args.out_csv, index=False)
            print(f"Wrote exemplars (by n_aligned): {args.out_csv}")
            print(out)
            return

    # 3) fallback: compute variance from predictions
    P = pd.read_csv(args.pred_csv)
    P["subject"] = P["subject"].astype(str)
    if "z_obs" not in P.columns:
        raise SystemExit("ERROR: predictions CSV missing z_obs column.")

    var_df = (P.groupby("subject")["z_obs"]
                .apply(lambda s: float(np.nanvar(pd.to_numeric(s, errors="coerce"))))
                .reset_index()
                .rename(columns={"z_obs": "var_z_obs"}))
    out = var_df.sort_values("var_z_obs", ascending=False).head(args.k)[["subject"]]
    if len(out) == 0:
        raise SystemExit("ERROR: Could not select any exemplars (empty predictions?).")

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote exemplars (by var z_obs): {args.out_csv}")
    print(out)


if __name__ == "__main__":
    main()
