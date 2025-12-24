#!/usr/bin/env python3
"""
Fit a minimal pooled, lagged linear ODE in log-space for butyrate.

State:
  z(t) = log1p(B(t))

Model (Euler on z):
  z_next = z_prev + dt * ( p0_s
                           + alpha_F * F(t-lagF) - alpha_A * A(t-lagA)
                           + k_LB * L(t-lagL)
                           - lambda_B * z_prev )

Global: alpha_F, alpha_A, k_LB, lambda_B
Per-subject: p0_s only

Key improvement (point 2):
  Within-subject centering of drivers (F/A/L) BEFORE global robust scaling.
  This removes cross-subject baseline confounding and typically improves pooled fits.

Outputs:
  params_v4_cf.csv
  predictions_v4_cf.csv
  fit_summary_v4_cf.csv
  global_fit_v4_cf.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# ------------------------ detection helpers ------------------------

def detect_subject_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if re.search(r"^subject", str(c), re.I):
            return c
    for c in df.columns:
        if re.search(r"(subject|participant|patient).*id|(^|_)id($|_)", str(c), re.I):
            return c
    for c in df.columns:
        if re.search(r"(subject|participant|patient)", str(c), re.I):
            return c
    raise SystemExit("ERROR: Could not detect subject id column.")


def detect_time_col(df: pd.DataFrame) -> str:
    pats = [r"^date of receipt$", r"^date$", r"collection", r"sample.*date", r"timestamp", r"interval sequence"]
    for pat in pats:
        for c in df.columns:
            if re.search(pat, str(c), re.I):
                return c
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).head(30)
            t = pd.to_datetime(s, errors="coerce", utc=False)
            if t.notna().mean() >= 0.7:
                return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            return c
    raise SystemExit("ERROR: Could not detect time column.")


def parse_time_series(s: pd.Series) -> Tuple[np.ndarray, str]:
    if np.issubdtype(s.dtype, np.number):
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return x, "numeric"
    t = pd.to_datetime(s, errors="coerce", utc=False)
    if t.notna().mean() >= 0.7:
        t2 = t.view("int64") / 1e9  # seconds since epoch
        return t2.to_numpy(dtype=float), "datetime"
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return x, "numeric"


def detect_butyrate_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if re.search(r"butyrate|butyric|(^|[^A-Za-z])c4([^A-Za-z]|$)", str(c), re.I):
            return c
    raise SystemExit("ERROR: Could not detect butyrate column by /butyrate|butyric|c4/.")


def choose_driver_col(df: pd.DataFrame, base: str, use_smoothed: bool) -> str:
    if use_smoothed:
        cand = base + "_s"
        if cand in df.columns:
            return cand
    if base in df.columns:
        return base
    for c in df.columns:
        if str(c).lower() == base.lower():
            return c
    raise SystemExit(f"ERROR: Missing required driver column '{base}' (or '{base}_s' when smoothed).")


# ------------------------ helpers ------------------------

def safe_mkdir(p: str) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def huber_loss(res: np.ndarray, delta: float) -> float:
    r = np.asarray(res, dtype=float)
    a = np.abs(r)
    quad = a <= delta
    return float(np.sum(0.5 * (r[quad] ** 2)) + np.sum(delta * (a[~quad] - 0.5 * delta)))


def robust_scale_global(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    med = float(np.nanmedian(x))
    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr < 1e-12:
        iqr = 1.0
    return (x - med) / iqr, med, iqr


def ls_solve(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
    return coef


def fit_obs_calibration(z_obs: np.ndarray, z_hat_raw: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(z_obs) & np.isfinite(z_hat_raw)
    if m.sum() < 2:
        return 0.0, 1.0
    X = np.c_[np.ones(m.sum()), z_hat_raw[m]]
    beta = ls_solve(X, z_obs[m])
    return float(beta[0]), float(beta[1])


# ------------------------ alignment ------------------------

def build_aligned_rows(df: pd.DataFrame,
                       subj_col: str,
                       time_col: str,
                       B_col: str,
                       F_col: str,
                       A_col: str,
                       L_col: str,
                       lag_F: int,
                       lag_A: int,
                       lag_L: int,
                       min_obs: int,
                       min_sd_zB: float) -> pd.DataFrame:
    rows = []

    for sid, g0 in df.groupby(subj_col, sort=False):
        g = g0.copy()

        t_raw, t_kind = parse_time_series(g[time_col])
        g["_t_"] = t_raw
        g = g.sort_values("_t_")

        dt = np.diff(g["_t_"].to_numpy(dtype=float))
        dt_days = dt / (24 * 3600.0) if t_kind == "datetime" else dt.astype(float)
        if len(dt_days) == 0:
            continue
        if np.any(~np.isfinite(dt_days)) or np.any(dt_days <= 0):
            continue

        B = pd.to_numeric(g[B_col], errors="coerce").to_numpy(dtype=float)
        zB = np.log1p(np.clip(B, 0, None))
        if np.nanstd(zB) < min_sd_zB:
            continue

        F = pd.to_numeric(g[F_col], errors="coerce").to_numpy(dtype=float)
        A = pd.to_numeric(g[A_col], errors="coerce").to_numpy(dtype=float)
        L = pd.to_numeric(g[L_col], errors="coerce").to_numpy(dtype=float)

        max_lag = max(lag_F, lag_A, lag_L)
        aligned_count = 0

        for i in range(max_lag, len(g) - 1):
            j = i + 1
            Fi = F[i - lag_F]
            Ai = A[i - lag_A]
            Li = L[i - lag_L]
            if not (np.isfinite(Fi) and np.isfinite(Ai) and np.isfinite(Li)):
                continue
            if not (np.isfinite(zB[i]) and np.isfinite(zB[j]) and np.isfinite(dt_days[i])):
                continue

            rows.append({
                "subject": str(sid),
                "aligned_index": int(i),
                "dt_days": float(dt_days[i]),
                "B_prev": float(B[i]),
                "B_next": float(B[j]),
                "z_prev": float(zB[i]),
                "z_obs": float(zB[j]),
                "F_raw": float(Fi),
                "A_raw": float(Ai),
                "L_raw": float(Li),
            })
            aligned_count += 1

        if aligned_count < min_obs:
            rows = [r for r in rows if r["subject"] != str(sid)]

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("ERROR: No aligned rows after QC. Lower --min_obs or check timestamps/columns.")
    return out


# ------------------------ fitting ------------------------

def fit_for_lambda(aligned: pd.DataFrame,
                   lambda_B: float,
                   nonneg_k: bool,
                   n_iter: int = 3) -> Dict[str, object]:
    z_prev = aligned["z_prev"].to_numpy(dtype=float)
    z_obs = aligned["z_obs"].to_numpy(dtype=float)
    dt = aligned["dt_days"].to_numpy(dtype=float)

    F = aligned["F_in"].to_numpy(dtype=float)
    A = aligned["A_in"].to_numpy(dtype=float)
    L = aligned["L_in"].to_numpy(dtype=float)

    subjects = aligned["subject"].astype(str).to_numpy()
    uniq = sorted(pd.unique(subjects).tolist())
    s_to_idx = {s: i for i, s in enumerate(uniq)}
    S = np.zeros((len(subjects), len(uniq)), dtype=float)
    for r, s in enumerate(subjects):
        S[r, s_to_idx[s]] = 1.0

    z_dot = (z_obs - z_prev) / dt
    k = 0.0

    for _ in range(n_iter):
        y = z_dot + lambda_B * z_prev - k * L
        M = np.c_[F, -A, S]  # [alpha_F, alpha_A, p0_subject...]
        coef = ls_solve(M, y)
        alpha_F = float(coef[0])
        alpha_A = float(coef[1])
        p0_vec = coef[2:].astype(float)

        resid = y - (alpha_F * F - alpha_A * A + S @ p0_vec)
        denom = float(np.dot(L, L))
        k = 0.0 if denom < 1e-12 else float(np.dot(L, resid) / denom)
        if nonneg_k:
            k = max(0.0, k)

    z_hat_raw = z_prev + dt * (S @ p0_vec + alpha_F * F - alpha_A * A + k * L - lambda_B * z_prev)
    a_obs, b_obs = fit_obs_calibration(z_obs=z_obs, z_hat_raw=z_hat_raw)
    z_hat = a_obs + b_obs * z_hat_raw

    return {
        "lambda_B": float(lambda_B),
        "alpha_F": alpha_F,
        "alpha_A": alpha_A,
        "k_LB": float(k),
        "p0_vec": p0_vec,
        "subjects": uniq,
        "alpha_obs": float(a_obs),
        "beta_obs": float(b_obs),
        "z_hat_raw": z_hat_raw,
        "z_hat": z_hat,
    }


def per_subject_r2(aligned: pd.DataFrame, z_hat: np.ndarray) -> pd.DataFrame:
    subj = aligned["subject"].astype(str).to_numpy()
    z_obs = aligned["z_obs"].to_numpy(dtype=float)
    rows = []
    for s in pd.unique(subj):
        m = (subj == s) & np.isfinite(z_obs) & np.isfinite(z_hat)
        y = z_obs[m]
        yh = z_hat[m]
        if len(y) < 2:
            continue
        ss_res = float(np.sum((y - yh) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = np.nan if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        rows.append({"subject": str(s), "R2_logspace": float(r2), "n_aligned": int(len(y))})
    return pd.DataFrame(rows)


# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--use_smoothed", action="store_true")
    ap.add_argument("--min_obs", type=int, default=6)
    ap.add_argument("--min_sd_zB", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--lag_F", type=int, default=1)
    ap.add_argument("--lag_A", type=int, default=0)
    ap.add_argument("--lag_L", type=int, default=1)
    ap.add_argument("--L_col", type=str, default="L_cf_s")
    ap.add_argument("--lambda_grid", type=str, default="0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3,0.45,0.6")
    ap.add_argument("--nonneg_k", action="store_true")
    ap.add_argument("--model", type=str, default="full", choices=["full", "noL", "noA", "noF", "noAL"],
                help="Model variant: full uses F,A,L; noL removes L; noA removes A; noF removes F; noAL removes A and L.")


    # point (2) controls
    ap.add_argument("--within_subject_center", action="store_true", default=True,
                    help="Center F/A/L within subject before global scaling (recommended).")
    ap.add_argument("--no_within_subject_center", action="store_true",
                    help="Disable within-subject centering (for ablation comparison).")

    args = ap.parse_args()

    if args.no_within_subject_center:
        args.within_subject_center = False

    outdir = safe_mkdir(args.outdir)

    df = pd.read_csv(args.csv)
    subj_col = detect_subject_col(df)
    time_col = detect_time_col(df)
    B_col = detect_butyrate_col(df)
    F_col = choose_driver_col(df, "F_met", args.use_smoothed)
    A_col = choose_driver_col(df, "A_met", args.use_smoothed)

    L_col = args.L_col
    if L_col not in df.columns:
        raise SystemExit(f"ERROR: L_col '{L_col}' not found. Run preprocess to create L_cf/L_cf_s.")

    aligned = build_aligned_rows(
        df=df,
        subj_col=subj_col,
        time_col=time_col,
        B_col=B_col,
        F_col=F_col,
        A_col=A_col,
        L_col=L_col,
        lag_F=int(args.lag_F),
        lag_A=int(args.lag_A),
        lag_L=int(args.lag_L),
        min_obs=int(args.min_obs),
        min_sd_zB=float(args.min_sd_zB),
    )

    # ---------- point (2): within-subject centering ----------
    if args.within_subject_center:
        grp = aligned.groupby("subject", sort=False)
        aligned["F_c"] = grp["F_raw"].transform(lambda s: s - float(np.nanmean(s.to_numpy(dtype=float))))
        aligned["A_c"] = grp["A_raw"].transform(lambda s: s - float(np.nanmean(s.to_numpy(dtype=float))))
        aligned["L_c"] = grp["L_raw"].transform(lambda s: s - float(np.nanmean(s.to_numpy(dtype=float))))
    else:
        aligned["F_c"] = aligned["F_raw"]
        aligned["A_c"] = aligned["A_raw"]
        aligned["L_c"] = aligned["L_raw"]

    # global robust scaling AFTER centering
    F_scaled, F_med, F_iqr = robust_scale_global(aligned["F_c"].to_numpy(dtype=float))
    A_scaled, A_med, A_iqr = robust_scale_global(aligned["A_c"].to_numpy(dtype=float))
    L_scaled, L_med, L_iqr = robust_scale_global(aligned["L_c"].to_numpy(dtype=float))
    aligned["F_in"] = F_scaled
    aligned["A_in"] = A_scaled
    aligned["L_in"] = L_scaled

    # --- model ablations (still metabolite-only) ---
    if args.model == "noL":
        aligned["L_in"] = 0.0
    elif args.model == "noA":
        aligned["A_in"] = 0.0
    elif args.model == "noF":
        aligned["F_in"] = 0.0
    elif args.model == "noAL":
        aligned["A_in"] = 0.0
        aligned["L_in"] = 0.0

    lambda_grid = [float(x) for x in args.lambda_grid.split(",") if x.strip() != ""]
    best = None
    best_loss = None

    for lam in lambda_grid:
        fit = fit_for_lambda(aligned, float(lam), bool(args.nonneg_k), n_iter=3)
        res = aligned["z_obs"].to_numpy(dtype=float) - fit["z_hat"]
        loss = huber_loss(res, delta=float(args.delta))
        if best is None or loss < best_loss:
            best = fit
            best_loss = float(loss)

    assert best is not None

    aligned = aligned.copy()
    aligned["z_hat_raw"] = best["z_hat_raw"]
    aligned["z_hat"] = best["z_hat"]
    aligned["B_hat_next"] = np.expm1(np.clip(aligned["z_hat"], -50, 50))

    p0_map = {s: float(best["p0_vec"][i]) for i, s in enumerate(best["subjects"])}
    aligned["p0"] = aligned["subject"].map(p0_map).astype(float)

    fit_summary = per_subject_r2(aligned, aligned["z_hat"].to_numpy(dtype=float))
    fit_summary["lag_F"] = int(args.lag_F)
    fit_summary["lag_A"] = int(args.lag_A)
    fit_summary["lag_L"] = int(args.lag_L)

    params = fit_summary[["subject", "n_aligned", "R2_logspace"]].copy()
    params["p0"] = params["subject"].map(p0_map).astype(float)
    params["lambda_B"] = float(best["lambda_B"])
    params["alpha_F"] = float(best["alpha_F"])
    params["alpha_A"] = float(best["alpha_A"])
    params["k_LB"] = float(best["k_LB"])
    params["alpha_obs"] = float(best["alpha_obs"])
    params["beta_obs"] = float(best["beta_obs"])
    params["lag_F"] = int(args.lag_F)
    params["lag_A"] = int(args.lag_A)
    params["lag_L"] = int(args.lag_L)
    params["within_subject_center"] = bool(args.within_subject_center)

    gj = {
        "lambda_grid": lambda_grid,
        "best": {
            "lambda_B": float(best["lambda_B"]),
            "alpha_F": float(best["alpha_F"]),
            "alpha_A": float(best["alpha_A"]),
            "k_LB": float(best["k_LB"]),
            "alpha_obs": float(best["alpha_obs"]),
            "beta_obs": float(best["beta_obs"]),
            "loss_huber": float(best_loss),
        },
        "lags": {"F": int(args.lag_F), "A": int(args.lag_A), "L": int(args.lag_L)},
        "within_subject_center": bool(args.within_subject_center),
        "driver_centering_note": "Drivers were centered within subject (X - mean_subject) before global robust scaling." if args.within_subject_center else "No within-subject centering applied.",
        "driver_scaling_centered_values": {
            "F": {"median": float(F_med), "iqr": float(F_iqr)},
            "A": {"median": float(A_med), "iqr": float(A_iqr)},
            "L": {"median": float(L_med), "iqr": float(L_iqr)},
            "model": args.model,
        },
    }

    params.to_csv(outdir / "params_v4_cf.csv", index=False)

    pred_cols = [
        "subject", "aligned_index", "dt_days",
        "B_prev", "B_next", "B_hat_next",
        "z_prev", "z_obs", "z_hat_raw", "z_hat",
        "F_in", "A_in", "L_in",
        "F_raw", "A_raw", "L_raw",
        "F_c", "A_c", "L_c",
        "p0",
    ]
    pred_cols = [c for c in pred_cols if c in aligned.columns]
    aligned[pred_cols].to_csv(outdir / "predictions_v4_cf.csv", index=False)

    fit_summary.to_csv(outdir / "fit_summary_v4_cf.csv", index=False)

    with open(outdir / "global_fit_v4_cf.json", "w", encoding="utf-8") as f:
        json.dump(gj, f, indent=2)

    print(f"Wrote: {outdir/'params_v4_cf.csv'}")
    print(f"Wrote: {outdir/'predictions_v4_cf.csv'}")
    print(f"Wrote: {outdir/'fit_summary_v4_cf.csv'}")
    print(f"Wrote: {outdir/'global_fit_v4_cf.json'}")
    print("Best:", gj["best"])


if __name__ == "__main__":
    main()
