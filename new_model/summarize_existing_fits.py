#!/usr/bin/env python3
# summarize_existing_fits.py
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--save", default="existing_fit_summary.csv")
    args = ap.parse_args()

    rows = []
    for fp in glob.glob(str(Path(args.outdir)/"fit_*.json")):
        j = json.load(open(fp))
        sid = j.get("subject")
        metr = j.get("metrics", {})
        row = {"subject_id": sid,
               "cost": j.get("cost"),
               "nfev": j.get("nfev"),
               "status": j.get("status"),
               "use_logB": j.get("use_logB", False),
               "NRMSE_B": metr.get("NRMSE_B"), "R2_B": metr.get("R2_B"),
               "NRMSE_H": metr.get("NRMSE_H"), "R2_H": metr.get("R2_H")}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("subject_id")
    df["fit_ok_rule"] = ( (df["NRMSE_B"] <= 0.7) | (df["R2_B"] >= 0.3) )
    out = Path(args.outdir)/args.save
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
