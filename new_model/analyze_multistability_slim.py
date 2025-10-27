#!/usr/bin/env python3
# analyze_multistability_slim.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scfa_model import Params, simulate, piecewise_constant_from_samples
from prepare_inputs import load_subject_data
from fit_subject_slim import build_params

def basin_map(subject, csv_path, theta_subj, fixed, n_inits=150, t_pad=30.0, seed=0):
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]
    F_fun = piecewise_constant_from_samples(t_obs, data["F"])
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])
    p = build_params(theta_subj, fixed)
    rng = np.random.default_rng(seed)
    T = float(t_obs.max() + t_pad)
    endpoints = []
    for x0 in rng.uniform(low=[0,0,0,0,0], high=[1,1,1,1,1], size=(n_inits,5)):
        _, X = simulate(0.0, T, x0, p, F_fun, A_fun, dense=False)
        endpoints.append(X[-1,:])
    Xend = np.array(endpoints)
    # simple clustering by rounding (no sklearn dependency here)
    rounded = np.round(Xend, 3)
    unique = np.unique(rounded, axis=0)
    return {"n_states": int(len(unique)), "endpoints": Xend.tolist()}

def hysteresis_curve(subject, csv_path, theta_subj, fixed, F_min=0.0, F_max=1.0, steps=24, hold=50.0):
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])
    def F_const(v): return (lambda t: float(v))
    p = build_params(theta_subj, fixed)
    x = np.array([0.1,0.5,0.1,0.2,0.2])
    ups, downs = [], []
    for v in np.linspace(F_min, F_max, steps):
        _, X = simulate(0.0, hold, x, p, F_const(v), A_fun, dense=False)
        x = X[-1,:]; ups.append((float(v), x.copy()))
    for v in np.linspace(F_max, F_min, steps):
        _, X = simulate(0.0, hold, x, p, F_const(v), A_fun, dense=False)
        x = X[-1,:]; downs.append((float(v), x.copy()))
    return {"ups": [(v, x.tolist()) for v,x in ups], "downs": [(v, x.tolist()) for v,x in downs]}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--fit_json", required=True, help="the per-subject JSON from outputs_slim")
    ap.add_argument("--outdir", default="outputs_slim")
    args = ap.parse_args()

    fit = json.loads(Path(args.fit_json).read_text())
    theta = np.array(fit["theta_subj"], dtype=float)
    fixed = fit["globals"]

    bm = basin_map(args.subject, args.csv, theta, fixed, n_inits=120, t_pad=35.0, seed=1)
    hs = hysteresis_curve(args.subject, args.csv, theta, fixed, F_min=0.0, F_max=1.0, steps=24, hold=50.0)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    Path(outdir/f"basin_{args.subject}.json").write_text(json.dumps(bm, indent=2))
    Path(outdir/f"hysteresis_{args.subject}.json").write_text(json.dumps(hs, indent=2))
    print(json.dumps({"basins": bm["n_states"], "saved": str(outdir)}, indent=2))
