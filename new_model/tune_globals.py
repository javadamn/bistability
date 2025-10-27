#!/usr/bin/env python3
# tune_globals.py
import json, copy
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fit_subject_slim import fit_subject_slim

TUNE_KEYS = ["u","k_B","g","K_B","d0","eta"]  # always tuned
# optional "H_center" is handled specially (converted to H_on/off with fixed gap)

def load_subjects(csv_path: str, min_obs_b: int, subset: List[str] | None):
    dfm = pd.read_csv(csv_path, parse_dates=["date"])
    if subset:
        dfm = dfm[dfm["subject_id"].astype(str).isin([str(s) for s in subset])]
    counts = dfm.dropna(subset=["butyrate"]).groupby("subject_id").size()
    subjects = [str(s) for s, n in counts.items() if n >= min_obs_b]
    return sorted(subjects)

def globals_to_vector(glob: Dict, gap_width: float, tune_h_center: bool):
    x = [glob["fixed"].get(k) for k in TUNE_KEYS]
    if tune_h_center:
        H_on  = glob["fixed"].get("H_on", 0.3)
        H_off = glob["fixed"].get("H_off", 0.7)
        center = 0.5*(H_on + H_off)
        x.append(center)
    return np.array(x, dtype=float)

def vector_to_globals(x: np.ndarray, base: Dict, gap_width: float, tune_h_center: bool):
    g = copy.deepcopy(base)
    for i,k in enumerate(TUNE_KEYS):
        g["fixed"][k] = float(x[i])
    if tune_h_center:
        center = float(x[len(TUNE_KEYS)])
        H_on  = center - gap_width/2
        H_off = center + gap_width/2
        # clamp into (0,1)
        H_on  = float(np.clip(H_on,  0.05, 0.95))
        H_off = float(np.clip(H_off, 0.05, 0.95))
        if H_on >= H_off:  # ensure order
            H_on, H_off = H_off-1e-3, H_off
        g["fixed"]["H_on"]  = H_on
        g["fixed"]["H_off"] = H_off
    return g

def penalty(x, tune_h_center: bool):
    pen = 0.0
    # positivity / reasonable magnitudes
    # u, k_B, g, K_B, d0, eta
    lb = np.array([1e-4, 1e-4, 0.05, 0.05, 0.01, 1.0])
    ub = np.array([2.0,   2.0,   4.0,  3.0,  0.5,  20.0])
    z = x[:len(lb)]
    pen += float(np.sum(np.maximum(0, lb - z)**2 + np.maximum(0, z - ub)**2))
    if tune_h_center:
        c = x[len(lb)]
        pen += 10.0 * float(np.maximum(0, 0.05 - c)**2 + np.maximum(0, c - 0.95)**2)
    return pen

def main():
    import argparse, tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--init_globals", default="globals_cohort.json")
    ap.add_argument("--out_json", default="globals_cohort_tuned.json")
    ap.add_argument("--subjects", nargs="*", default=None, help="subset to tune on; defaults to best-covered subjects")
    ap.add_argument("--min_obs_b", type=int, default=5)
    ap.add_argument("--n_starts", type=int, default=8)
    ap.add_argument("--max_nfev_subj", type=int, default=250)
    ap.add_argument("--logbase", choices=["e","10"], default="10")
    ap.add_argument("--rawB", action="store_true")
    ap.add_argument("--gap_width", type=float, default=0.4)
    ap.add_argument("--tune_h_center", action="store_true")
    ap.add_argument("--outer_loops", type=int, default=1, help="alternations; 1 is usually enough")
    args = ap.parse_args()

    use_logB = (not args.rawB)
    base = json.loads(Path(args.init_globals).read_text())
    subjects = load_subjects(args.csv, args.min_obs_b, args.subjects)
    if not subjects:
        raise SystemExit("No subjects eligible for tuning (check --min_obs_b and --subjects).")

    x0 = globals_to_vector(base, args.gap_width, args.tune_h_center)

    def obj(x):
        # penalty to keep in plausible region
        pen = penalty(x, args.tune_h_center)
        g_trial = vector_to_globals(x, base, args.gap_width, args.tune_h_center)
        total_cost = 0.0
        for sid in subjects:
            try:
                fit = fit_subject_slim(
                    sid, args.csv, globals_json=None,  # pass dict directly below
                    use_logB=use_logB, logbase=args.logbase,
                    max_nfev=args.max_nfev_subj, n_starts=args.n_starts
                )
            except TypeError:
                # older signature: pass path; create temp file
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                    json.dump(g_trial, tf)
                    tf.flush()
                    fit = fit_subject_slim(
                        sid, args.csv, tf.name,
                        use_logB=use_logB, logbase=args.logbase,
                        max_nfev=args.max_nfev_subj, n_starts=args.n_starts
                    )
            # Re-run with g_trial by overriding inside fit_subject_slim call (newer versions can accept dict)
            # If your local fit_subject_slim only takes a path, the tempfile route above covers it.

            # We need to re-fit using g_trial. If the function already used base globals, re-call:
            fit = fit_subject_slim(
                sid, args.csv, globals_json=json.dumps(g_trial),
                use_logB=use_logB, logbase=args.logbase,
                max_nfev=args.max_nfev_subj, n_starts=args.n_starts
            )
            total_cost += float(fit["cost"])
        return total_cost + 1000.0*pen

    # Run outer loops (usually 1 is enough since obj() re-fits per subject every evaluation)
    x = x0.copy()
    for _ in range(args.outer_loops):
        res = minimize(obj, x, method="Nelder-Mead", options={"maxiter": 120, "xatol":1e-3, "fatol":1e-2})
        x = res.x

    tuned = vector_to_globals(x, base, args.gap_width, args.tune_h_center)
    Path(args.out_json).write_text(json.dumps(tuned, indent=2))
    print(json.dumps({"tuned_globals": tuned["fixed"]}, indent=2))

if __name__ == "__main__":
    main()
