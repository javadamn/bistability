
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from dataclasses import replace
from scipy.optimize import least_squares
from scfa_model import Params, simulate, piecewise_constant_from_samples, obs_butyrate, obs_health
from prepare_inputs import load_subject_data

def make_cost(subject: str, csv_path: str, t_max_pad: float = 7.0):
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]
    yB_obs = data["y_B"]
    yH_obs = data["y_H"]
    F_fun = piecewise_constant_from_samples(t_obs, data["F"])
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])
    t0, t1 = float(t_obs.min()), float(t_obs.max() + t_max_pad)
    # initial state guess
    x0 = np.array([0.2, 0.2, max(1e-3, np.nanmedian(yB_obs)), 0.5, 0.5], dtype=float)
    # weights (z-score observations for balance)
    B_mu, B_sd = np.nanmean(yB_obs), np.nanstd(yB_obs) if np.nanstd(yB_obs)>0 else 1.0
    H_mu, H_sd = np.nanmean(yH_obs), np.nanstd(yH_obs) if np.nanstd(yH_obs)>0 else 1.0
    
    def resid(theta: np.ndarray) -> np.ndarray:
        # unpack a subset of parameters to fit; others fixed to defaults
        p = Params(
            r_M=theta[0], r_C=theta[1], K_M=1.0, K_C=1.0, gamma=theta[2],
            alpha=theta[3], beta=theta[4], p_min=theta[5], p_max=theta[6],
            u=theta[7], k_B=theta[8], g=theta[9], K_B=theta[10], n=2.0, d0=theta[11],
            chi=theta[12], H_I=theta[13], eta=theta[14], H_on=theta[15], H_off=theta[16],
            eps=theta[17], k_on=theta[18], k_off=theta[19], sigma_F=theta[20], sigma_A=theta[21],
            s_B=theta[22], s_H=theta[23]
        )
        t_sim, X = simulate(t0, t1, x0, p, F_fun, A_fun, dense=True)
        # sample at observation times by nearest neighbor
        def sample_at(ts, t_grid, arr):
            idx = np.searchsorted(t_grid, ts, side="left")
            idx = np.clip(idx, 0, len(t_grid)-1)
            return arr[idx]
        B_sim = sample_at(t_obs, t_sim, X[:,2])
        H_sim = sample_at(t_obs, t_sim, X[:,3])
        yB_hat = obs_butyrate(B_sim, p)
        yH_hat = obs_health(H_sim, p)
        rB = (yB_hat - yB_obs) / (B_sd if B_sd>0 else 1.0)
        rH = (yH_hat - yH_obs) / (H_sd if H_sd>0 else 1.0)
        # handle NaNs in yH_obs
        rH = rH[~np.isnan(yH_obs)]
        return np.concatenate([rB, rH])
    
    # initial theta
    theta0 = np.array([0.6, 0.7, 0.6, 0.6, 0.6,
                       0.2, 1.0, 0.15, 0.15, 0.8, 0.5, 0.1, 0.6, 0.5, 6.0,
                       0.3, 0.7, 0.05, 0.2, 0.2, 0.2, 0.6, 1.0, 1.0], dtype=float)
    lb = np.array([0.05, 0.05, 0.0, 0.0, 0.0,
                   0.0,  0.2, 0.01, 0.01, 0.1, 0.05, 0.01, 0.0, 0.1, 1.0,
                   0.1, 0.3, 0.01, 0.01, 0.01, 0.0, 0.0,  0.1,  0.1])
    ub = np.array([2.0, 2.0, 2.0, 2.0, 2.0,
                   1.0, 3.0, 1.0,  1.0,  3.0, 3.0, 1.0,  2.0, 1.0, 20.0,
                   0.8, 0.95,0.2,  1.5,  1.5, 1.0, 2.0,  10.0, 10.0])
    return resid, theta0, lb, ub, (data, t0, t1, x0)

def fit_subject(subject: str, csv_path: str, out_dir: str = "outputs") -> Dict[str, Any]:
    resid, theta0, lb, ub, aux = make_cost(subject, csv_path)
    res = least_squares(resid, theta0, bounds=(lb, ub), max_nfev=200, verbose=1)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = {
        "subject": subject,
        "theta": res.x.tolist(),
        "cost": float(0.5*np.sum(res.fun**2)),
        "nfev": int(res.nfev),
        "status": int(res.status),
        "message": res.message
    }
    with open(Path(out_dir)/f"fit_{subject}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="subject_id to fit")
    ap.add_argument("--csv", default="modeling_table_with_indicators.csv")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    result = fit_subject(args.subject, args.csv, args.outdir)
    print(json.dumps(result, indent=2))
