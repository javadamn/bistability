#!/usr/bin/env python3
# fit_all_subjects_slim.py
import json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from scfa_model import Params, simulate, piecewise_constant_from_samples, obs_health
from prepare_inputs import load_subject_data
from fit_subject_slim import fit_subject_slim, build_params, SUBJECT_PARAM_NAMES

def evaluate_metrics(subject, csv_path, theta_subj, fixed, use_logB=True, logbase="10"):
    data = load_subject_data(csv_path, subject)
    t_obs = data["t_obs"]; yB_obs = data["y_B"].astype(float); yH_obs = data["y_H"].astype(float)

    F_fun = piecewise_constant_from_samples(t_obs, data["F"])
    A_fun = piecewise_constant_from_samples(t_obs, data["A"])

    p = build_params(theta_subj, fixed)
    a_B = float(theta_subj[6])  # intercept for logB fit
    t0, t1 = float(t_obs.min()), float(t_obs.max()+7.0)
    x0 = np.array([0.2,0.2,max(1e-6, np.nanmedian(yB_obs[np.isfinite(yB_obs)])),0.5,0.5])
    t_sim, X = simulate(t0, t1, x0, p, F_fun, A_fun, dense=True)
    idx = np.searchsorted(t_sim, t_obs, side="left"); idx = np.clip(idx, 0, len(t_sim)-1)

    B_sim, H_sim = X[idx,2], X[idx,3]
    if use_logB:
        Bhat_fit = a_B + p.s_B*(np.log1p(B_sim) if logbase=="e" else np.log10(1.0+B_sim))
        Bobserved_fit = np.log1p(yB_obs) if logbase=="e" else np.log10(1.0+yB_obs)
    else:
        Bhat_fit = p.s_B * B_sim
        Bobserved_fit = yB_obs

    maskB = np.isfinite(Bobserved_fit) & np.isfinite(Bhat_fit)
    nB = int(maskB.sum())
    if nB>0:
        sseB = float(np.sum((Bhat_fit[maskB]-Bobserved_fit[maskB])**2))
        rmseB = math.sqrt(sseB/nB)
        sdB = float(np.std(Bobserved_fit[maskB])) if nB>1 else np.nan
        nrmseB = (rmseB/sdB) if (nB>1 and sdB and np.isfinite(sdB) and sdB>0) else np.nan
        sstB = float(np.sum((Bobserved_fit[maskB]-np.mean(Bobserved_fit[maskB]))**2)) if nB>1 else np.nan
        r2B = (1 - sseB/sstB) if (nB>1 and sstB and np.isfinite(sstB) and sstB>0) else np.nan
    else:
        rmseB = nrmseB = r2B = np.nan

    # Health (report only; may be sparse)
    yH_hat = obs_health(H_sim, p)
    maskH = np.isfinite(yH_obs) & np.isfinite(yH_hat)
    nH = int(maskH.sum())
    if nH>0:
        sseH = float(np.sum((yH_hat[maskH]-yH_obs[maskH])**2))
        rmseH = math.sqrt(sseH/nH)
        sdH = float(np.std(yH_obs[maskH])) if nH>1 else np.nan
        nrmseH = (rmseH/sdH) if (nH>1 and sdH and np.isfinite(sdH) and sdH>0) else np.nan
        sstH = float(np.sum((yH_obs[maskH]-np.mean(yH_obs[maskH]))**2)) if nH>1 else np.nan
        r2H = (1 - sseH/sstH) if (nH>1 and sstH and np.isfinite(sstH) and sstH>0) else np.nan
    else:
        rmseH = nrmseH = r2H = np.nan

    return {"n_obs_B": nB, "RMSE_B": rmseB, "NRMSE_B": nrmseB, "R2_B": r2B,
            "n_obs_H": nH, "RMSE_H": rmseH, "NRMSE_H": nrmseH, "R2_H": r2H}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--globals", default="globals_template.json")
    ap.add_argument("--outdir", default="outputs_slim")
    ap.add_argument("--min_obs_b", type=int, default=4)
    ap.add_argument("--n_starts", type=int, default=8)
    ap.add_argument("--max_nfev", type=int, default=250)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--logbase", choices=["e","10"], default="10")
    ap.add_argument("--rawB", action="store_true")
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    use_logB = (not args.rawB)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    dfm = pd.read_csv(args.csv, parse_dates=["date"])
    if args.subjects:
        dfm = dfm[dfm["subject_id"].astype(str).isin([str(s) for s in args.subjects])]
    counts = dfm.dropna(subset=["butyrate"]).groupby("subject_id").size()
    subjects = [str(s) for s, n in counts.items() if n >= args.min_obs_b]
    subjects = sorted(subjects)
    if not subjects:
        raise SystemExit("No subjects with sufficient butyrate observations.")

    rows = []
    for sid in subjects:
        print(f"\n=== Fitting {sid} (reduced model) ===")
        out = fit_subject_slim(sid, args.csv, args.globals, use_logB=use_logB,
                               logbase=args.logbase, max_nfev=args.max_nfev, n_starts=args.n_starts)
        theta_subj = np.array(out["theta_subj"], dtype=float)
        fixed = out["globals"]

        # Metrics
        metr = evaluate_metrics(sid, args.csv, theta_subj, fixed, use_logB=use_logB, logbase=args.logbase)

        # Save JSON
        with open(outdir/f"fit_{sid}.json","w") as f:
            json.dump({**out, "metrics": {k: (float(v) if (v is not None and np.isscalar(v) and np.isfinite(v)) else v) for k,v in metr.items()}}, f, indent=2)

        # Optional predictions (fit space for B)
        if args.save_preds:
            data = load_subject_data(args.csv, sid)
            t_obs = data["t_obs"]; yB_obs = data["y_B"].astype(float); yH_obs = data["y_H"].astype(float)
            F_fun = piecewise_constant_from_samples(t_obs, data["F"])
            A_fun = piecewise_constant_from_samples(t_obs, data["A"])
            p = build_params(theta_subj, fixed)
            a_B = theta_subj[6]
            t0, t1 = float(t_obs.min()), float(t_obs.max()+7.0)
            x0 = np.array([0.2,0.2,max(1e-6, np.nanmedian(yB_obs[np.isfinite(yB_obs)])),0.5,0.5])
            t_sim, X = simulate(t0, t1, x0, p, F_fun, A_fun, dense=True)
            idx = np.searchsorted(t_sim, t_obs, side="left"); idx = np.clip(idx, 0, len(t_sim)-1)
            B_sim, H_sim = X[idx,2], X[idx,3]
            Bhat_fit = (a_B + p.s_B*(np.log1p(B_sim) if args.logbase=="e" else np.log10(1.0+B_sim))) if use_logB else (p.s_B*B_sim)
            Bobserved_fit = (np.log1p(yB_obs) if args.logbase=="e" else np.log10(1.0+yB_obs)) if use_logB else yB_obs
            yH_hat = obs_health(H_sim, p)
            pd.DataFrame({
                "t_days": t_obs,
                "yB_obs_fitspace": Bobserved_fit,
                "yB_hat_fitspace": Bhat_fit,
                "yH_obs": yH_obs,
                "yH_hat": yH_hat
            }).to_csv(outdir/f"preds_{sid}.csv", index=False)

        # Summary row
        row = {"subject_id": sid}
        row.update({k: float(v) if (v is not None and np.isscalar(v) and np.isfinite(v)) else v for k,v in metr.items()})
        for name, val in zip(SUBJECT_PARAM_NAMES, theta_subj):
            row[name] = float(val)
        rows.append(row)

    df = pd.DataFrame(rows)
    summ = outdir / "fit_summary_all_subjects_slim.csv"
    df.to_csv(summ, index=False)

    # Bundle for upload
    bundle = outdir / "all_fits_bundle_slim.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(summ, arcname=summ.name)
        for sid in subjects:
            j = outdir / f"fit_{sid}.json"
            if j.exists(): zf.write(j, arcname=j.name)
            p = outdir / f"preds_{sid}.csv"
            if args.save_preds and p.exists(): zf.write(p, arcname=p.name)

    print(f"\nWrote {summ}")
    print(f"Wrote {bundle}")

if __name__ == "__main__":
    main()
