#!/usr/bin/env python3
# sensitivity_sweep.py
import json
from pathlib import Path
import numpy as np
from scfa_model import simulate, piecewise_constant_from_samples
from prepare_inputs import load_subject_data
from fit_subject_slim import build_params

LEVER_NAMES = ["r_M","gamma","sigma_F","sigma_A","p_max"]

def run_sensitivity(subject, csv_path, theta_subj, fixed, lever="p_max", factors=(0.8, 1.0, 1.2), F_levels=(0.0, 0.5, 1.0), hold=60.0):
    assert lever in LEVER_NAMES
    base = theta_subj.copy()
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]; A_fun = piecewise_constant_from_samples(t_obs, data["A"])
    def F_const(v): return (lambda t: float(v))
    idx = LEVER_NAMES.index(lever) if lever in LEVER_NAMES[:-1] else 4
    out = []
    for f in factors:
        th = base.copy()
        th[idx] = th[idx] * f
        p = build_params(th, fixed)
        for Fv in F_levels:
            x = np.array([0.1,0.5,0.1,0.2,0.2])
            _, X = simulate(0.0, hold, x, p, F_const(Fv), A_fun, dense=False)
            out.append({"lever": lever, "factor": float(f), "F": float(Fv), "M": float(X[-1,0]), "C": float(X[-1,1]), "B": float(X[-1,2]), "H": float(X[-1,3])})
    return out

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--fit_json", required=True)
    ap.add_argument("--lever", choices=LEVER_NAMES, default="p_max")
    ap.add_argument("--outdir", default="outputs_slim")
    args = ap.parse_args()

    fit = json.loads(Path(args.fit_json).read_text())
    theta = np.array(fit["theta_subj"], dtype=float)
    fixed = fit["globals"]

    res = run_sensitivity(args.subject, args.csv, theta, fixed, lever=args.lever)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir)/f"sensitivity_{args.subject}_{args.lever}.json".write_text(json.dumps(res, indent=2))
    print(f"Saved sensitivity results for {args.subject} on {args.lever}.")
