#!/usr/bin/env python3
# batch_multistability_and_sensitivity.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from analyze_multistability_slim import basin_map, hysteresis_curve
from sensitivity_sweep import run_sensitivity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--fits_dir", default="outputs_slim_tuned")
    ap.add_argument("--outdir", default="outputs_slim_tuned")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--lever", default="p_max", choices=["r_M","gamma","sigma_F","sigma_A","p_max"])
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--hold", type=float, default=50.0)
    ap.add_argument("--n_inits", type=int, default=120)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Determine subject list from fit JSONs (or --subjects)
    jsons = sorted(Path(args.fits_dir).glob("fit_*.json"))
    ids = [p.stem.split("_",1)[1] for p in jsons]
    if args.subjects:
        ids = [s for s in ids if s in args.subjects]
    if not ids:
        raise SystemExit("No fitted subjects found.")

    rows = []
    for sid in ids:
        fit_path = Path(args.fits_dir)/f"fit_{sid}.json"
        fit = json.loads(fit_path.read_text())
        theta = np.array(fit["theta_subj"], dtype=float)
        fixed = fit["globals"]

        bm = basin_map(sid, args.csv, theta, fixed, n_inits=args.n_inits, t_pad=35.0, seed=1)
        hs = hysteresis_curve(sid, args.csv, theta, fixed, F_min=0.0, F_max=1.0, steps=args.steps, hold=args.hold)

        # Save raw artifacts
        Path(outdir/f"basin_{sid}.json").write_text(json.dumps(bm, indent=2))
        Path(outdir/f"hysteresis_{sid}.json").write_text(json.dumps(hs, indent=2))

        # Simple hysteresis metric: ΔH at mid F (up vs down)
        ups = np.array([x for _,x in hs["ups"]]); upsF = np.array([v for v,_ in hs["ups"]])
        dns = np.array([x for _,x in hs["downs"]]); dnsF = np.array([v for v,_ in hs["downs"]])
        mid = 0.5
        ui = np.argmin(np.abs(upsF - mid)); di = np.argmin(np.abs(dnsF - mid))
        dH = float(ups[ui,3] - dns[di,3]) if (len(ups)>0 and len(dns)>0) else np.nan

        # Sensitivity (one lever)
        sens = run_sensitivity(sid, args.csv, theta, fixed, lever=args.lever, hold=args.hold)
        Path(outdir/f"sensitivity_{sid}_{args.lever}.json").write_text(json.dumps(sens, indent=2))

        rows.append({"subject_id": sid, "basins": int(bm["n_states"]), "deltaH_midF": dH})

    pd.DataFrame(rows).to_csv(outdir/"multistability_summary.csv", index=False)
    print(f"Wrote {outdir/'multistability_summary.csv'} and per-subject JSONs.")

if __name__ == "__main__":
    main()
