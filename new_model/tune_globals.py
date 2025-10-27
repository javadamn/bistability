#!/usr/bin/env python3
# tune_globals.py  — fixed to always pass a real JSON filepath to fit_subject_slim
import json, copy, tempfile, os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fit_subject_slim import fit_subject_slim

TUNE_KEYS = ["u","k_B","g","K_B","d0","eta"]  # tuned globals

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
        # enforce in (0,1) and keep gap width
        center = float(np.clip(center, 0.05, 0.95))
        H_on  = center - gap_width/2
        H_off = center + gap_width/2
        # clamp and ensure order
        H_on  = float(np.clip(H_on,  0.01, 0.99))
        H_off = float(np.clip(H_off, 0.01, 0.99))
        if H_on >= H_off:
            H_on  = max(0.01, H_off - 1e-3)
        g["fixed"]["H_on"]  = H_on
        g["fixed"]["H_off"] = H_off
    return g

def penalty(x, tune_h_center: bool):
    # soft bounds; keeps optimizer in a plausible region
    pen = 0.0
    lb = np.array([1e-4, 1e-4, 0.05, 0.05, 0.01, 1.0])   # u,kB,g,KB,d0,eta
    ub = np.array([2.0,   2.0,   4.0,  3.0,  0.5,  20.0])
    z = x[:len(lb)]
    pen += float(np.sum(np.maximum(0, lb - z)**2 + np.maximum(0, z - ub)**2))
    if tune_h_center:
        c = x[len(lb)]
        pen += 10.0 * float(np.maximum(0, 0.05 - c)**2 + np.maximum(0, c - 0.95)**2)
    return pen

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--init_globals", default="globals_cohort.json")
    ap.add_argument("--out_json", default="globals_cohort_tuned.json")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--min_obs_b", type=int, default=5)
    ap.add_argument("--n_starts", type=int, default=8, help="per-subject starts during tuning (use smaller to speed up)")
    ap.add_argument("--max_nfev_subj", type=int, default=250)
    ap.add_argument("--logbase", choices=["e","10"], default="10")
    ap.add_argument("--rawB", action="store_true")
    ap.add_argument("--gap_width", type=float, default=0.4)
    ap.add_argument("--tune_h_center", action="store_true")
    ap.add_argument("--outer_loops", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=120)
    args = ap.parse_args()

    use_logB = (not args.rawB)
    base = json.loads(Path(args.init_globals).read_text())
    subjects = load_subjects(args.csv, args.min_obs_b, args.subjects)
    if not subjects:
        raise SystemExit("No subjects eligible for tuning (check --min_obs_b / --subjects).")

    x0 = globals_to_vector(base, args.gap_width, args.tune_h_center)

    def obj(x):
        # Build candidate globals dict
        g_trial = vector_to_globals(x, base, args.gap_width, args.tune_h_center)
        # Write to a temp file and pass the path to fit_subject_slim
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(g_trial, tf)
            tf.flush()
            tmp_path = tf.name
        total_cost = 0.0
        try:
            for sid in subjects:
                fit = fit_subject_slim(
                    sid,
                    args.csv,
                    globals_json=tmp_path,
                    use_logB=use_logB,
                    logbase=args.logbase,
                    max_nfev=args.max_nfev_subj,
                    n_starts=args.n_starts
                )
                total_cost += float(fit["cost"])
        finally:
            # clean up the temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        # soft penalty keeps search stable
        return total_cost + 1000.0*penalty(x, args.tune_h_center)

    # Outer loop: typically 1 is enough since obj() re-fits subjects each evaluation
    x = x0.copy()
    for _ in range(args.outer_loops):
        res = minimize(
            obj, x, method="Nelder-Mead",
            options={"maxiter": args.maxiter, "xatol": 1e-3, "fatol": 1e-2}
        )
        x = res.x

    tuned = vector_to_globals(x, base, args.gap_width, args.tune_h_center)
    Path(args.out_json).write_text(json.dumps(tuned, indent=2))
    print(json.dumps({"tuned_globals": tuned["fixed"]}, indent=2))

if __name__ == "__main__":
    main()
