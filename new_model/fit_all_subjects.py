#!/usr/bin/env python3
# fit_all_subjects.py
import argparse, json, sys, math, zipfile, io
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Local imports from previous drop-ins
sys.path.append(".")
from scfa_model import Params, simulate, piecewise_constant_from_samples, obs_butyrate, obs_health
from prepare_inputs import load_subject_data
import fit_subject as singlefit  # uses make_cost() and fit_subject()

# Parameter name order (must match fit_subject.py)
THETA_NAMES = [
    "r_M","r_C","gamma","alpha","beta",
    "p_min","p_max","u","k_B",
    "g","K_B","d0",
    "chi","H_I","eta",
    "H_on","H_off","eps",
    "k_on","k_off",
    "sigma_F","sigma_A",
    "s_B","s_H"
]

def evaluate_metrics(subject: str, csv_path: str, theta: np.ndarray, t_pad_days: float = 7.0) -> Dict[str, Any]:
    """Simulate with fitted params and compute metrics on observed times."""
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]; yB_obs = data["y_B"]; yH_obs = data["y_H"]
    F_fun = piecewise_constant_from_samples(t_obs, data["F"])
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])

    # Recreate Params in same ordering as fit
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

    t0, t1 = float(t_obs.min()), float(t_obs.max() + t_pad_days)
    x0 = np.array([0.2, 0.2, max(1e-3, np.nanmedian(yB_obs)), 0.5, 0.5], dtype=float)
    t_sim, X = simulate(t0, t1, x0, p, F_fun, A_fun, dense=True)

    # nearest-left sampling at obs times
    idx = np.searchsorted(t_sim, t_obs, side="left")
    idx = np.clip(idx, 0, len(t_sim)-1)

    B_sim = X[idx,2]; H_sim = X[idx,3]
    yB_hat = obs_butyrate(B_sim, p)
    yH_hat = obs_health(H_sim, p)

    # Metrics for B
    maskB = np.isfinite(yB_obs) & np.isfinite(yB_hat)
    nB = int(maskB.sum())
    sseB = float(np.sum((yB_hat[maskB] - yB_obs[maskB])**2)) if nB > 0 else np.nan
    rmseB = math.sqrt(sseB/nB) if nB > 0 else np.nan
    sdB = float(np.std(yB_obs[maskB])) if nB > 1 else np.nan
    nrmseB = (rmseB/sdB) if (nB > 1 and sdB and np.isfinite(sdB) and sdB > 0) else np.nan
    sstB = float(np.sum((yB_obs[maskB] - np.mean(yB_obs[maskB]))**2)) if nB > 1 else np.nan
    r2B = (1 - sseB/sstB) if (nB > 1 and sstB and np.isfinite(sstB) and sstB > 0) else np.nan

    # Metrics for H
    maskH = np.isfinite(yH_obs) & np.isfinite(yH_hat)
    nH = int(maskH.sum())
    sseH = float(np.sum((yH_hat[maskH] - yH_obs[maskH])**2)) if nH > 0 else np.nan
    rmseH = math.sqrt(sseH/nH) if nH > 0 else np.nan
    sdH = float(np.std(yH_obs[maskH])) if nH > 1 else np.nan
    nrmseH = (rmseH/sdH) if (nH > 1 and sdH and np.isfinite(sdH) and sdH > 0) else np.nan
    sstH = float(np.sum((yH_obs[maskH] - np.mean(yH_obs[maskH]))**2)) if nH > 1 else np.nan
    r2H = (1 - sseH/sstH) if (nH > 1 and sstH and np.isfinite(sstH) and sstH > 0) else np.nan

    return {
        "n_obs_B": nB, "RMSE_B": rmseB, "NRMSE_B": nrmseB, "R2_B": r2B,
        "n_obs_H": nH, "RMSE_H": rmseH, "NRMSE_H": nrmseH, "R2_H": r2H,
        "yB_hat": yB_hat, "yH_hat": yH_hat,
        "t_obs": t_obs, "yB_obs": yB_obs, "yH_obs": yH_obs
    }

def random_start(theta0: np.ndarray, lb: np.ndarray, ub: np.ndarray, scale: float = 0.3, rng: np.random.Generator = None):
    """Generate a random start near theta0 within bounds."""
    if rng is None:
        rng = np.random.default_rng()
    # log-normal-ish multiplicative perturbation then clip to [lb,ub]
    noise = rng.normal(loc=0.0, scale=scale, size=theta0.shape)
    cand = theta0 * np.exp(noise)
    return np.clip(cand, lb, ub)

def multi_start_fit(subject: str, csv_path: str, n_starts: int, max_nfev: int) -> Dict[str, Any]:
    """Run multiple starts; return best result (lowest cost)."""
    make = singlefit.make_cost
    resid, theta0, lb, ub, aux = make(subject, csv_path)
    rng = np.random.default_rng(12345)

    best = None
    for k in range(n_starts):
        if k == 0:
            th = theta0.copy()
        else:
            th = random_start(theta0, lb, ub, scale=0.35, rng=rng)

        try:
            res = least_squares(resid, th, bounds=(lb, ub), max_nfev=max_nfev, verbose=0)
            cost = 0.5*float(np.sum(res.fun**2))
            cand = {
                "theta": res.x.copy(),
                "cost": cost,
                "nfev": int(res.nfev),
                "status": int(res.status),
                "message": res.message
            }
            if (best is None) or (cand["cost"] < best["cost"]):
                best = cand
        except Exception as e:
            # keep going even if a start fails
            continue

    if best is None:
        raise RuntimeError(f"All {n_starts} starts failed for subject={subject}")

    best["theta"] = best["theta"].tolist()
    return best

def any_at_bounds(theta: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol: float = 1e-4) -> int:
    return int(np.sum((np.abs(theta - lb) <= tol) | (np.abs(theta - ub) <= tol)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--min_obs_b", type=int, default=4, help="Min non-null butyrate obs to fit a subject")
    ap.add_argument("--n_starts", type=int, default=5, help="Random restart count per subject")
    ap.add_argument("--max_nfev", type=int, default=200)
    ap.add_argument("--save_preds", action="store_true", help="Write per-visit predictions for each subject")
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional subset of subject_ids to fit")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    dfm = pd.read_csv(args.csv, parse_dates=["date"])
    if args.subjects:
        dfm = dfm[dfm["subject_id"].astype(str).isin([str(s) for s in args.subjects])]

    # choose subjects with enough butyrate observations
    ok = dfm.dropna(subset=["butyrate"]).groupby("subject_id").size()
    subjects = [str(s) for s, n in ok.items() if n >= args.min_obs_b]
    subjects = sorted(subjects)

    if not subjects:
        raise SystemExit("No subjects with sufficient butyrate observations.")

    # get lb/ub from make_cost once
    resid0, theta0, lb, ub, _ = singlefit.make_cost(subjects[0], args.csv)

    rows = []
    for sid in subjects:
        print(f"\n=== Fitting subject {sid} ===")
        try:
            # multi-start fit
            best = multi_start_fit(sid, args.csv, n_starts=args.n_starts, max_nfev=args.max_nfev)
            theta = np.array(best["theta"], dtype=float)

            # metrics
            metr = evaluate_metrics(sid, args.csv, theta)
            bound_hits = any_at_bounds(theta, lb, ub, tol=1e-4)

            # save per-subject JSON
            j = {
                "subject": sid,
                "theta": best["theta"],
                "theta_names": THETA_NAMES,
                "cost": best["cost"],
                "nfev": best["nfev"],
                "status": best["status"],
                "message": best["message"],
                "metrics": {k: float(v) if np.isscalar(v) and np.isfinite(v) else v
                            for k, v in metr.items() if k not in ("yB_hat","yH_hat","t_obs","yB_obs","yH_obs")}
            }
            with open(outdir / f"fit_{sid}.json", "w") as f:
                json.dump(j, f, indent=2)

            # optional per-visit predictions
            if args.save_preds:
                df_pred = pd.DataFrame({
                    "t_days": metr["t_obs"],
                    "yB_obs": metr["yB_obs"],
                    "yB_hat": metr["yB_hat"],
                    "yH_obs": metr["yH_obs"],
                    "yH_hat": metr["yH_hat"]
                })
                df_pred.to_csv(outdir / f"preds_{sid}.csv", index=False)

            # summary row
            row = {
                "subject_id": sid,
                "cost": best["cost"],
                "nfev": best["nfev"],
                "status": best["status"],
                "bound_hits": bound_hits,
                "n_obs_B": metr["n_obs_B"], "RMSE_B": metr["RMSE_B"], "NRMSE_B": metr["NRMSE_B"], "R2_B": metr["R2_B"],
                "n_obs_H": metr["n_obs_H"], "RMSE_H": metr["RMSE_H"], "NRMSE_H": metr["NRMSE_H"], "R2_H": metr["R2_H"],
            }
            # add parameters as columns
            for name, val in zip(THETA_NAMES, theta):
                row[name] = float(val)
            rows.append(row)

        except Exception as e:
            # record failure
            rows.append({
                "subject_id": sid, "cost": np.nan, "nfev": 0, "status": -1, "bound_hits": np.nan,
                "n_obs_B": np.nan, "RMSE_B": np.nan, "NRMSE_B": np.nan, "R2_B": np.nan,
                "n_obs_H": np.nan, "RMSE_H": np.nan, "NRMSE_H": np.nan, "R2_H": np.nan,
                "error": str(e)
            })
            continue

    # write summary CSV
    df_summary = pd.DataFrame(rows)
    summary_path = outdir / "fit_summary_all_subjects.csv"
    df_summary.to_csv(summary_path, index=False)

    # bundle outputs for easy sharing
    bundle_path = outdir / "all_fits_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_path, arcname="fit_summary_all_subjects.csv")
        for sid in subjects:
            j = outdir / f"fit_{sid}.json"
            if j.exists(): zf.write(j, arcname=j.name)
            p = outdir / f"preds_{sid}.csv"
            if args.save_preds and p.exists(): zf.write(p, arcname=p.name)

    print(f"\nWrote summary: {summary_path}")
    print(f"Bundle ready:  {bundle_path}")

if __name__ == "__main__":
    main()
