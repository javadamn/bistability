#!/usr/bin/env python3
# fit_all_subjects_v2.py
import argparse, json, math, zipfile
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from scfa_model import Params, simulate, piecewise_constant_from_samples, obs_health
from prepare_inputs import load_subject_data
from fit_subject_v2 import make_cost

THETA_NAMES = [
    "r_M","r_C","gamma","alpha","beta",
    "p_min","p_max","u","k_B",
    "g","K_B","d0",
    "chi","H_I","eta",
    "H_on","H_off","eps",
    "k_on","k_off",
    "sigma_F","sigma_A",
    "s_B","s_H",
    "a_B"  # intercept (only used if logB)
]

def random_start(theta0, lb, ub, scale=0.35, rng=None):
    rng = rng or np.random.default_rng()
    th = theta0 * np.exp(rng.normal(0, scale, size=theta0.shape))
    return np.clip(th, lb, ub)

def evaluate_metrics(subject: str, csv_path: str, theta: np.ndarray, use_logB: bool) -> Dict[str, Any]:
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]; yB_obs = data["y_B"].astype(float); yH_obs = data["y_H"].astype(float)
    F_fun = piecewise_constant_from_samples(t_obs, data["F"])
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])

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

    t0, t1 = float(t_obs.min()), float(t_obs.max()+7.0)
    x0 = np.array([0.2,0.2, max(1e-6, np.nanmedian(yB_obs[np.isfinite(yB_obs)])) ,0.5,0.5], dtype=float)
    t_sim, X = simulate(t0, t1, x0, p, F_fun, A_fun, dense=True)
    idx = np.searchsorted(t_sim, t_obs, side="left"); idx = np.clip(idx, 0, len(t_sim)-1)

    B_sim = X[idx,2]; H_sim = X[idx,3]
    if use_logB:
        yB_hat = a_B + p.s_B * np.log1p(B_sim)
        yB_obs_fit = np.log1p(yB_obs)
    else:
        yB_hat = p.s_B * B_sim
        yB_obs_fit = yB_obs

    # Metrics in fit space for B
    maskB = np.isfinite(yB_obs_fit) & np.isfinite(yB_hat)
    nB = int(maskB.sum())
    sseB = float(np.sum((yB_hat[maskB] - yB_obs_fit[maskB])**2)) if nB>0 else np.nan
    rmseB = math.sqrt(sseB/nB) if nB>0 else np.nan
    sdB = float(np.std(yB_obs_fit[maskB])) if nB>1 else np.nan
    nrmseB = (rmseB/sdB) if (nB>1 and sdB and np.isfinite(sdB) and sdB>0) else np.nan
    sstB = float(np.sum((yB_obs_fit[maskB]-np.mean(yB_obs_fit[maskB]))**2)) if nB>1 else np.nan
    r2B = (1 - sseB/sstB) if (nB>1 and sstB and np.isfinite(sstB) and sstB>0) else np.nan

    # Metrics for H
    yH_hat = obs_health(H_sim, p)
    maskH = np.isfinite(yH_obs) & np.isfinite(yH_hat)
    nH = int(maskH.sum())
    sseH = float(np.sum((yH_hat[maskH] - yH_obs[maskH])**2)) if nH>0 else np.nan
    rmseH = math.sqrt(sseH/nH) if nH>0 else np.nan
    sdH = float(np.std(yH_obs[maskH])) if nH>1 else np.nan
    nrmseH = (rmseH/sdH) if (nH>1 and sdH and np.isfinite(sdH) and sdH>0) else np.nan
    sstH = float(np.sum((yH_obs[maskH]-np.mean(yH_obs[maskH]))**2)) if nH>1 else np.nan
    r2H = (1 - sseH/sstH) if (nH>1 and sstH and np.isfinite(sstH) and sstH>0) else np.nan

    return {
        "n_obs_B": nB, "RMSE_B": rmseB, "NRMSE_B": nrmseB, "R2_B": r2B,
        "n_obs_H": nH, "RMSE_H": rmseH, "NRMSE_H": nrmseH, "R2_H": r2H
    }

def any_at_bounds(theta, lb, ub, tol=1e-4): 
    return int(np.sum((np.abs(theta-lb)<=tol) | (np.abs(theta-ub)<=tol)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--outdir", default="outputs_v2")
    ap.add_argument("--min_obs_b", type=int, default=4)
    ap.add_argument("--n_starts", type=int, default=8)
    ap.add_argument("--max_nfev", type=int, default=300)
    ap.add_argument("--rawB", action="store_true", help="fit on raw B (default: log1p(B))")
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    use_logB = (not args.rawB)

    dfm = pd.read_csv(args.csv, parse_dates=["date"])
    if args.subjects:
        dfm = dfm[dfm["subject_id"].astype(str).isin([str(s) for s in args.subjects])]

    counts = dfm.dropna(subset=["butyrate"]).groupby("subject_id").size()
    subjects = [str(s) for s, n in counts.items() if n >= args.min_obs_b]
    subjects = sorted(subjects)
    if not subjects:
        raise SystemExit("No subjects with sufficient butyrate observations.")

    # Get bounds from make_cost for the first subject (same for all)
    resid0, th0, lb, ub = make_cost(subjects[0], args.csv, use_logB=use_logB)
    rows = []
    rng = np.random.default_rng(123)

    for sid in subjects:
        print(f"\n=== Fitting {sid} (logB={use_logB}) ===")
        resid, theta0, lb, ub = make_cost(sid, args.csv, use_logB=use_logB)
        best = None
        for k in range(args.n_starts):
            th = theta0 if k==0 else random_start(theta0, lb, ub, scale=0.35, rng=rng)
            try:
                res = least_squares(resid, th, bounds=(lb, ub), max_nfev=args.max_nfev, verbose=0)
                cost = 0.5*float(np.sum(res.fun**2))
                cand = {"theta": res.x.copy(), "cost": cost, "nfev": int(res.nfev),
                        "status": int(res.status), "message": res.message}
                if (best is None) or (cand["cost"] < best["cost"]):
                    best = cand
            except Exception:
                continue

        if best is None:
            rows.append({"subject_id": sid, "status": -1, "error": "all starts failed"})
            continue

        theta = best["theta"].copy()
        metr = evaluate_metrics(sid, args.csv, theta, use_logB=use_logB)
        bh = any_at_bounds(theta, lb, ub)

        # Fit quality flag (tune thresholds as you like)
        fit_ok = (
            (metr["NRMSE_B"] is not None and np.isfinite(metr["NRMSE_B"]) and metr["NRMSE_B"] <= 0.7)
            or (metr["R2_B"] is not None and np.isfinite(metr["R2_B"]) and metr["R2_B"] >= 0.3)
        )

        # Save per-subject JSON & optional preds
        j = {
            "subject": sid,
            "theta": theta.tolist(),
            "theta_names": THETA_NAMES,
            "cost": best["cost"],
            "nfev": best["nfev"],
            "status": best["status"],
            "message": best["message"],
            "use_logB": use_logB,
            "metrics": {k: (float(v) if (v is not None and np.isscalar(v) and np.isfinite(v)) else v) for k,v in metr.items()},
            "fit_ok": bool(fit_ok),
            "bound_hits": int(bh)
        }
        with open(outdir/f"fit_{sid}.json","w") as f: json.dump(j, f, indent=2)

        # Optional predictions CSV (in fit space for B)
        if args.save_preds:
            data = load_subject_data(args.csv, sid)
            t_obs = data["t_obs"]; yB_obs = data["y_B"].astype(float); yH_obs = data["y_H"].astype(float)
            F_fun = piecewise_constant_from_samples(t_obs, data["F"])
            A_fun = piecewise_constant_from_samples(t_obs, data["A"])
            p = Params(
                r_M=theta[0], r_C=theta[1], K_M=1.0, K_C=1.0, gamma=theta[2],
                alpha=theta[3], beta=theta[4], p_min=theta[5], p_max=theta[6],
                u=theta[7], k_B=theta[8], g=theta[9], K_B=theta[10], n=2.0, d0=theta[11],
                chi=theta[12], H_I=theta[13], eta=theta[14], H_on=theta[15], H_off=theta[16], eps=theta[17],
                k_on=theta[18], k_off=theta[19], sigma_F=theta[20], sigma_A=theta[21], s_B=theta[22], s_H=theta[23]
            )
            a_B = theta[24] if use_logB else 0.0
            x0 = np.array([0.2,0.2,max(1e-6, np.nanmedian(yB_obs[np.isfinite(yB_obs)])),0.5,0.5])
            t0, t1 = float(t_obs.min()), float(t_obs.max()+7.0)
            t_sim, X = simulate(t0, t1, x0, p, F_fun, A_fun, dense=True)
            idx = np.searchsorted(t_sim, t_obs, side="left"); idx = np.clip(idx, 0, len(t_sim)-1)
            B_sim, H_sim = X[idx,2], X[idx,3]
            yB_hat_fit = (a_B + p.s_B*np.log1p(B_sim)) if use_logB else (p.s_B*B_sim)
            yB_obs_fit = np.log1p(yB_obs) if use_logB else yB_obs
            yH_hat = obs_health(H_sim, p)
            pd.DataFrame({
                "t_days": t_obs,
                "yB_obs_fitspace": yB_obs_fit,
                "yB_hat_fitspace": yB_hat_fit,
                "yH_obs": yH_obs,
                "yH_hat": yH_hat
            }).to_csv(outdir/f"preds_{sid}.csv", index=False)

        # Row for summary
        row = {"subject_id": sid, "fit_ok": fit_ok, "bound_hits": bh, "cost": best["cost"], "nfev": best["nfev"], "status": best["status"]}
        row.update(metr)
        for name, val in zip(THETA_NAMES, theta):
            row[name] = float(val)
        rows.append(row)

    df = pd.DataFrame(rows)
    summ = outdir / "fit_summary_all_subjects.csv"
    df.to_csv(summ, index=False)

    # Bundle for upload
    bundle = outdir / "all_fits_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(summ, arcname=summ.name)
        for sid in subjects:
            j = outdir / f"fit_{sid}.json"
            if j.exists(): zf.write(j, arcname=j.name)
            p = outdir / f"preds_{sid}.csv"
            if args.save_preds and p.exists(): zf.write(p, arcname=p.name)
    print(f"Wrote {summ}")
    print(f"Wrote {bundle}")

if __name__ == "__main__":
    main()
