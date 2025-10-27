#!/usr/bin/env python3
# fit_subject_slim.py
import json, math
from pathlib import Path
from typing import Dict, Any, Callable
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from scfa_model import Params, simulate, piecewise_constant_from_samples, obs_health
from prepare_inputs import load_subject_data

SUBJECT_PARAM_NAMES = ["r_M","gamma","sigma_F","sigma_A","p_max","s_B","a_B"]

def build_params(theta_subj: np.ndarray, fixed: Dict[str, float]) -> Params:
    return Params(
        r_M   = float(theta_subj[0]),
        r_C   = fixed.get("r_C", 0.7),
        K_M   = fixed.get("K_M", 1.0),
        K_C   = fixed.get("K_C", 1.0),
        gamma = float(theta_subj[1]),
        alpha = fixed.get("alpha", 0.4),
        beta  = fixed.get("beta", 0.4),

        p_min = fixed.get("p_min", 0.2),
        p_max = float(theta_subj[4]),
        u     = fixed.get("u", 0.05),
        k_B   = fixed.get("k_B", 0.08),

        g     = fixed.get("g", 1.0),
        K_B   = fixed.get("K_B", 0.6),
        n     = fixed.get("n", 2.0),
        d0    = fixed.get("d0", 0.08),
        chi   = fixed.get("chi", 0.4),
        H_I   = fixed.get("H_I", 0.5),
        eta   = fixed.get("eta", 6.0),

        H_on  = fixed.get("H_on", 0.3),
        H_off = fixed.get("H_off", 0.7),
        eps   = fixed.get("eps", 0.05),
        k_on  = fixed.get("k_on", 0.2),
        k_off = fixed.get("k_off", 0.2),

        sigma_F = float(theta_subj[2]),
        sigma_A = float(theta_subj[3]),

        s_B   = float(theta_subj[5]),
        s_H   = 1.0  # keep linear scale; H may be unused in loss
    )

def make_cost(subject: str, csv_path: str, globals_json: str,
              use_logB: bool = True, logbase: str = "10", min_H_points: int = 3):
    fixed = json.loads(Path(globals_json).read_text())["fixed"]
    data = load_subject_data(csv_path, subject)

    t_obs = data["t_obs"]
    yB_obs = data["y_B"].astype(float)
    yH_obs = data["y_H"].astype(float)

    # Inputs (piecewise-constant)
    F_fun = piecewise_constant_from_samples(t_obs, data["F"])
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])

    # Initial conditions
    yB0 = yB_obs[np.isfinite(yB_obs)]
    B0 = float(np.median(yB0)) if len(yB0) else 1.0
    x0 = np.array([0.2, 0.2, max(1e-6, B0), 0.5, 0.5], dtype=float)

    # Log-B transform in fit space
    if use_logB:
        yB_fit = np.log1p(yB_obs) if logbase == "e" else np.log10(1.0 + yB_obs)
    else:
        yB_fit = yB_obs
    B_sd = np.nanstd(yB_fit); B_sd = B_sd if (B_sd and np.isfinite(B_sd)) else 1.0

    use_H = np.isfinite(yH_obs).sum() >= min_H_points
    H_sd = (np.nanstd(yH_obs) if use_H else 1.0) or 1.0

    # Subject parameter vector: [r_M, gamma, sigma_F, sigma_A, p_max, s_B, a_B]
    theta0 = np.array([0.6, 0.4, 0.4, 0.4, 1.5, 1.0, 0.0], dtype=float)
    lb = np.array([0.05, 0.0, 0.0, 0.0, 0.3, 0.05, -5.0])
    ub = np.array([2.0,  2.0, 1.5, 1.5, 5.0, 20.0,  5.0])

    def resid(theta: np.ndarray) -> np.ndarray:
        p = build_params(theta, fixed)
        t0, t1 = float(t_obs.min()), float(t_obs.max()+7.0)
        t_sim, X = simulate(t0, t1, x0, p, F_fun, A_fun, dense=True)
        idx = np.searchsorted(t_sim, t_obs, side="left"); idx = np.clip(idx, 0, len(t_sim)-1)
        B_sim = X[idx,2]; H_sim = X[idx,3]

        # Butyrate in fit space
        if use_logB:
            yB_hat = theta[6] + p.s_B * (np.log1p(B_sim) if logbase=="e" else np.log10(1.0+B_sim))
        else:
            yB_hat = p.s_B * B_sim
        rB = (yB_hat - yB_fit) / B_sd
        rB = rB[np.isfinite(yB_fit)]

        # Health residuals (optional)
        if use_H:
            yH_hat = obs_health(H_sim, p)
            rH = (yH_hat - yH_obs) / H_sd
            rH = rH[np.isfinite(yH_obs)]
            return np.concatenate([rB, rH])
        else:
            return rB

    return resid, theta0, lb, ub, fixed

def fit_subject_slim(subject: str, csv_path: str, globals_json: str,
                     use_logB: bool = True, logbase: str = "10",
                     min_H_points: int = 3, max_nfev: int = 250,
                     n_starts: int = 8, seed: int = 42) -> Dict[str, Any]:
    resid, theta0, lb, ub, fixed = make_cost(
        subject, csv_path, globals_json, use_logB=use_logB, logbase=logbase, min_H_points=min_H_points
    )
    rng = np.random.default_rng(seed)

    def random_start(th0):
        noise = rng.normal(0, 0.35, size=th0.shape)
        cand = th0 * np.exp(noise)
        return np.clip(cand, lb, ub)

    best = None
    for k in range(n_starts):
        th = theta0 if k==0 else random_start(theta0)
        try:
            res = least_squares(resid, th, bounds=(lb, ub), max_nfev=max_nfev, verbose=0)
            cost = 0.5*float(np.sum(res.fun**2))
            cand = {"theta": res.x.copy(), "cost": cost, "nfev": int(res.nfev), "status": int(res.status), "message": res.message}
            if (best is None) or (cand["cost"] < best["cost"]):
                best = cand
        except Exception:
            continue

    if best is None:
        raise RuntimeError(f"All starts failed for subject={subject}")

    out = {
        "subject": subject,
        "theta_subj_names": SUBJECT_PARAM_NAMES,
        "theta_subj": best["theta"].tolist(),
        "cost": best["cost"],
        "nfev": best["nfev"],
        "status": best["status"],
        "message": best["message"],
        "use_logB": use_logB,
        "logbase": logbase,
        "min_H_points": min_H_points,
        "globals": fixed
    }
    return out

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--globals", default="globals_template.json")
    ap.add_argument("--logbase", choices=["e","10"], default="10")
    ap.add_argument("--rawB", action="store_true")
    ap.add_argument("--min_H_points", type=int, default=3)
    ap.add_argument("--max_nfev", type=int, default=250)
    ap.add_argument("--n_starts", type=int, default=8)
    ap.add_argument("--outdir", default="outputs_slim")
    args = ap.parse_args()

    out = fit_subject_slim(args.subject, args.csv, args.globals,
                           use_logB=(not args.rawB), logbase=args.logbase,
                           min_H_points=args.min_H_points, max_nfev=args.max_nfev, n_starts=args.n_starts)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir)/f"fit_{args.subject}.json","w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
