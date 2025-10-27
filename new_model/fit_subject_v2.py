#!/usr/bin/env python3
# fit_subject_v2.py
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from scipy.optimize import least_squares

from scfa_model import Params, simulate, piecewise_constant_from_samples, obs_health
from prepare_inputs import load_subject_data

def make_cost(subject: str, csv_path: str, t_pad_days: float = 7.0, use_logB: bool = True):
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]; yB_obs = data["y_B"].astype(float); yH_obs = data["y_H"].astype(float)
    F_fun = piecewise_constant_from_samples(t_obs, data["F"])
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])
    t0, t1 = float(t_obs.min()), float(t_obs.max() + t_pad_days)

    # Initial state
    yB0 = yB_obs[np.isfinite(yB_obs)]
    B0 = float(np.median(yB0)) if len(yB0) else 1.0
    x0 = np.array([0.2, 0.2, max(1e-6, B0), 0.5, 0.5], dtype=float)

    # Scale for residuals (in the space we fit)
    if use_logB:
        yB_fit = np.log1p(yB_obs)
        B_sd = np.nanstd(yB_fit); B_sd = B_sd if (B_sd and np.isfinite(B_sd)) else 1.0
    else:
        yB_fit = yB_obs
        B_sd = np.nanstd(yB_obs); B_sd = B_sd if (B_sd and np.isfinite(B_sd)) else 1.0

    H_sd = np.nanstd(yH_obs); H_sd = H_sd if (H_sd and np.isfinite(H_sd)) else 1.0

    # Parameter vector (adds aB intercept for logB)
    # [theta(24)..., a_B, s_H] but s_H already inside theta; we keep a_B after the 24th param
    def unpack_theta(theta):
        p = Params(
            r_M=theta[0], r_C=theta[1], K_M=1.0, K_C=1.0, gamma=theta[2],
            alpha=theta[3], beta=theta[4],
            p_min=theta[5], p_max=theta[6], u=theta[7], k_B=theta[8],
            g=theta[9], K_B=theta[10], n=2.0, d0=theta[11],
            chi=theta[12], H_I=theta[13], eta=theta[14],
            H_on=theta[15], H_off=theta[16], eps=theta[17],
            k_on=theta[18], k_off=theta[19],
            sigma_F=theta[20], sigma_A=theta[21],
            s_B=theta[22], s_H=theta[23]
        )
        a_B = theta[24] if use_logB else 0.0
        return p, a_B

    theta0 = np.array([
        0.6,0.7,0.6,0.6,0.6,
        0.2,1.5,0.05,0.1,
        1.0,0.6,0.1,
        0.6,0.5,6.0,
        0.3,0.7,0.05,
        0.2,0.2,
        0.4,0.4,
        1.0,1.0,
        0.0  # a_B (intercept) in log space
    ], dtype=float)

    lb = np.array([
        0.05,0.05,0.0,0.0,0.0,
        0.0,0.3, 0.001,0.001,
        0.1,0.05,0.01,
        0.0,0.1, 1.0,
        0.1,0.3, 0.01,
        0.01,0.01,
        0.0,0.0,
        0.05,0.05,
        -5.0  # a_B
    ])

    ub = np.array([
        2.0,2.0,2.0,2.0,2.0,
        2.0,5.0, 1.0, 1.0,
        3.0,3.0,1.0,
        2.0,1.0,20.0,
        0.8,0.95,0.2,
        1.5,1.5,
        1.5,1.5,
        20.0,10.0,
        5.0   # a_B
    ])

    def resid(theta):
        p, a_B = unpack_theta(theta)
        t_sim, X = simulate(t0, t1, x0, p, F_fun, A_fun, dense=True)
        idx = np.searchsorted(t_sim, t_obs, side="left"); idx = np.clip(idx, 0, len(t_sim)-1)
        B_sim = X[idx,2]; H_sim = X[idx,3]
        if use_logB:
            yB_hat = a_B + p.s_B * np.log1p(B_sim)
            rB = (yB_hat - np.log1p(yB_obs)) / B_sd
        else:
            yB_hat = p.s_B * B_sim
            rB = (yB_hat - yB_obs) / B_sd

        yH_hat = obs_health(H_sim, p)
        rH = (yH_hat - yH_obs) / H_sd
        rH = rH[~np.isnan(yH_obs)]
        return np.concatenate([rB[~np.isnan(yB_obs)], rH])

    return resid, theta0, lb, ub

def fit_subject_v2(subject: str, csv_path: str, use_logB: bool = True, max_nfev: int = 250) -> Dict[str, Any]:
    resid, theta0, lb, ub = make_cost(subject, csv_path, use_logB=use_logB)
    res = least_squares(resid, theta0, bounds=(lb, ub), max_nfev=max_nfev, verbose=0)
    out = {"subject": subject, "theta": res.x.tolist(),
           "cost": float(0.5*np.sum(res.fun**2)),
           "nfev": int(res.nfev), "status": int(res.status), "message": res.message,
           "use_logB": use_logB}
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--rawB", action="store_true", help="fit on raw B instead of log1p(B)")
    args = ap.parse_args()
    out = fit_subject_v2(args.subject, args.csv, use_logB=not args.rawB)
    Path("outputs_v2").mkdir(exist_ok=True, parents=True)
    with open(Path("outputs_v2")/f"fit_{args.subject}.json","w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
