
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from scfa_model import Params, simulate, piecewise_constant_from_samples
from prepare_inputs import load_subject_data

def basin_map(subject: str, csv_path: str, theta: np.ndarray,
              n_inits: int = 200, t_pad: float = 30.0, seed: int = 0):
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]
    F_fun = piecewise_constant_from_samples(t_obs, data["F"])
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])
    p = Params(
        r_M=theta[0], r_C=theta[1], K_M=1.0, K_C=1.0, gamma=theta[2],
        alpha=theta[3], beta=theta[4], p_min=theta[5], p_max=theta[6],
        u=theta[7], k_B=theta[8], g=theta[9], K_B=theta[10], n=2.0, d0=theta[11],
        chi=theta[12], H_I=theta[13], eta=theta[14], H_on=theta[15], H_off=theta[16],
        eps=theta[17], k_on=theta[18], k_off=theta[19], sigma_F=theta[20], sigma_A=theta[21],
        s_B=theta[22], s_H=theta[23]
    )
    rng = np.random.default_rng(seed)
    T = float(t_obs.max() + t_pad)
    # Random initial conditions
    X0s = rng.uniform(low=[0,0,0,0,0], high=[1.0,1.0,1.0,1.0,1.0], size=(n_inits,5))
    endpoints = []
    for x0 in X0s:
        _, X = simulate(0.0, T, x0, p, F_fun, A_fun, dense=False)
        endpoints.append(X[-1,:])
    Xend = np.array(endpoints)
    # Cluster endpoints to count attractors (tolerance 1e-2)
    clustering = DBSCAN(eps=1e-2, min_samples=3).fit(Xend)
    labels = clustering.labels_
    n_states = len(set(labels)) - (1 if -1 in labels else 0)
    return {"n_states": int(n_states), "labels": labels.tolist(), "endpoints": Xend.tolist()}

def hysteresis_curve(subject: str, csv_path: str, theta: np.ndarray,
                     F_min: float=0.0, F_max: float=1.0, steps: int=30, hold: float=60.0):
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])
    # Construct F as an externally controlled constant sweep
    def make_F_const(val):
        return lambda t: float(val)
    p = Params(
        r_M=theta[0], r_C=theta[1], K_M=1.0, K_C=1.0, gamma=theta[2],
        alpha=theta[3], beta=theta[4], p_min=theta[5], p_max=theta[6],
        u=theta[7], k_B=theta[8], g=theta[9], K_B=theta[10], n=2.0, d0=theta[11],
        chi=theta[12], H_I=theta[13], eta=theta[14], H_on=theta[15], H_off=theta[16],
        eps=theta[17], k_on=theta[18], k_off=theta[19], sigma_F=theta[20], sigma_A=theta[21],
        s_B=theta[22], s_H=theta[23]
    )
    # Start from low-F equilibrium
    x = np.array([0.1,0.5,0.1,0.2,0.2])
    ups = []
    for val in np.linspace(F_min, F_max, steps):
        _, X = simulate(0.0, hold, x, p, make_F_const(val), A_fun, dense=False)
        x = X[-1,:]
        ups.append((val, x.copy()))
    # Then sweep down
    downs = []
    for val in np.linspace(F_max, F_min, steps):
        _, X = simulate(0.0, hold, x, p, make_F_const(val), A_fun, dense=False)
        x = X[-1,:]
        downs.append((val, x.copy()))
    return {"ups": [(float(v), xs.tolist()) for v,xs in ups],
            "downs": [(float(v), xs.tolist()) for v,xs in downs]}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--csv", default="modeling_table_with_indicators.csv")
    ap.add_argument("--theta_json", required=True, help="path to fit_{subject}.json")
    args = ap.parse_args()
    info = json.load(open(args.theta_json))
    theta = np.array(info["theta"], dtype=float)
    bm = basin_map(args.subject, args.csv, theta)
    print(json.dumps({"basin_map": {"n_states": bm["n_states"]}}, indent=2))
