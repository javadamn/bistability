#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)

def detect_subject_col(df):
    for c in df.columns:
        if re.search(r"^subject", str(c), re.I): return c
    for c in df.columns:
        if re.search(r"(subject|participant|patient).*id|(^|_)id($|_)", str(c), re.I): return c
    raise SystemExit("ERROR: Could not detect subject column in meta_csv.")

def detect_abx_col(df):
    for c in df.columns:
        if re.search(r"(antibiotic|antibiotics|abx)", str(c), re.I):
            return c
    return None

def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")

def fit_linear_drift(z_prev, z_next):
    z_prev = np.asarray(z_prev, float)
    z_next = np.asarray(z_next, float)
    m = np.isfinite(z_prev) & np.isfinite(z_next)
    z0 = z_prev[m]
    dz = (z_next[m] - z0)
    if len(z0) < 4:
        return None
    X = np.c_[np.ones(len(z0)), z0]
    coef, _, _, _ = np.linalg.lstsq(X, dz, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b, len(z0)

def bootstrap_params(z_prev, z_next, boot=500, seed=0):
    rng = np.random.default_rng(seed)
    z_prev = np.asarray(z_prev, float)
    z_next = np.asarray(z_next, float)
    m = np.isfinite(z_prev) & np.isfinite(z_next)
    z0 = z_prev[m]
    z1 = z_next[m]
    n = len(z0)
    if n < 5:
        return None
    ab = []
    for _ in range(boot):
        idx = rng.integers(0, n, size=n)
        fit = fit_linear_drift(z0[idx], z1[idx])
        if fit is None:
            continue
        a, b, _ = fit
        ab.append((a, b))
    if len(ab) < max(50, int(0.2 * boot)):
        return None
    ab = np.array(ab, float)
    return ab[:, 0], ab[:, 1]

def summarize_subject(sid, g, boot, seed):
    # Prefer z columns if present
    if ("z_prev" in g.columns) and ("z_obs" in g.columns):
        z_prev = to_numeric_safe(g["z_prev"]).to_numpy()
        z_next = to_numeric_safe(g["z_obs"]).to_numpy()
    else:
        if not (("B_prev" in g.columns) and ("B_next" in g.columns)):
            raise SystemExit("ERROR: predictions must have either (z_prev,z_obs) or (B_prev,B_next).")
        Bp = np.clip(to_numeric_safe(g["B_prev"]).to_numpy(), 0, None)
        Bn = np.clip(to_numeric_safe(g["B_next"]).to_numpy(), 0, None)
        z_prev = np.log1p(Bp)
        z_next = np.log1p(Bn)

    fit = fit_linear_drift(z_prev, z_next)
    if fit is None:
        return {"subject": sid, "n_pairs": 0, "a": np.nan, "b": np.nan,
                "z_star": np.nan, "kappa": np.nan,
                "kappa_lo": np.nan, "kappa_hi": np.nan,
                "kappa_norm": np.nan, "kappa_norm_lo": np.nan, "kappa_norm_hi": np.nan,
                "stable_b_lt_0": False, "note": "too_few_pairs"}

    a, b, n = fit
    z_star = (-a / b) if np.isfinite(b) and abs(b) > 1e-12 else np.nan
    kappa = (-b) if np.isfinite(b) else np.nan
    stable = bool(np.isfinite(b) and (b < 0))

    # normalized curvature (helps compare across baselines)
    denom = (1.0 + abs(z_star)) if np.isfinite(z_star) else np.nan
    kappa_norm = (kappa / denom) if np.isfinite(kappa) and np.isfinite(denom) and denom > 0 else np.nan

    boots = bootstrap_params(z_prev, z_next, boot=boot, seed=seed)
    if boots is None:
        return {"subject": sid, "n_pairs": n, "a": a, "b": b,
                "z_star": z_star, "kappa": kappa,
                "kappa_lo": np.nan, "kappa_hi": np.nan,
                "kappa_norm": kappa_norm, "kappa_norm_lo": np.nan, "kappa_norm_hi": np.nan,
                "stable_b_lt_0": stable, "note": "bootstrap_insufficient"}

    a_s, b_s = boots
    k_s = -b_s
    k_lo, k_hi = np.nanpercentile(k_s, [2.5, 97.5])

    z_star_s = np.full_like(a_s, np.nan, dtype=float)
    m = np.isfinite(a_s) & np.isfinite(b_s) & (np.abs(b_s) > 1e-12)
    z_star_s[m] = -a_s[m] / b_s[m]
    denom_s = 1.0 + np.abs(z_star_s)
    k_norm_s = np.full_like(k_s, np.nan, dtype=float)
    mm = np.isfinite(k_s) & np.isfinite(denom_s) & (denom_s > 0)
    k_norm_s[mm] = k_s[mm] / denom_s[mm]
    kn_lo, kn_hi = np.nanpercentile(k_norm_s[np.isfinite(k_norm_s)], [2.5, 97.5]) if np.isfinite(k_norm_s).any() else (np.nan, np.nan)

    return {"subject": sid, "n_pairs": n, "a": a, "b": b,
            "z_star": z_star, "kappa": kappa,
            "kappa_lo": float(k_lo), "kappa_hi": float(k_hi),
            "kappa_norm": float(kappa_norm) if np.isfinite(kappa_norm) else np.nan,
            "kappa_norm_lo": float(kn_lo), "kappa_norm_hi": float(kn_hi),
            "stable_b_lt_0": stable, "note": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--subjects_csv", default=None)
    ap.add_argument("--meta_csv", default=None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    P = pd.read_csv(args.pred_csv)
    if "subject" not in P.columns:
        raise SystemExit("ERROR: predictions CSV must contain 'subject'.")
    P["subject"] = P["subject"].astype(str).str.strip()

    if args.subjects_csv:
        S = pd.read_csv(args.subjects_csv)
        subj = S.iloc[:, 0].astype(str).str.strip().tolist()
        subj = [s for s in subj if s]
    else:
        subj = sorted(P["subject"].unique().tolist())

    rows = []
    for sid in subj:
        g = P[P["subject"] == sid]
        rows.append(summarize_subject(sid, g, boot=args.boot, seed=args.seed))

    out = pd.DataFrame(rows)
    out.to_csv(Path(args.outdir) / "resilience_curvature_subjects.csv", index=False)

    if args.meta_csv:
        M = pd.read_csv(args.meta_csv)
        s_col = detect_subject_col(M)
        abx_col = detect_abx_col(M)
        if abx_col is None:
            print("NOTE: Could not detect ABX column in meta_csv; skipping ABX merge.")
        else:
            M[s_col] = M[s_col].astype(str).str.strip()
            abx = pd.to_numeric(M[abx_col], errors="coerce")
            abx_flag = (pd.DataFrame({"subject": M[s_col], "abx_val": abx})
                        .groupby("subject")["abx_val"]
                        .apply(lambda x: int(np.nanmax(x.fillna(0).to_numpy()) >= 1))
                        .reset_index()
                        .rename(columns={"abx_val": "abx_any"}))
            out2 = out.merge(abx_flag, on="subject", how="left")
            out2["abx_any"] = out2["abx_any"].fillna(0).astype(int)
            out2.to_csv(Path(args.outdir) / "resilience_curvature_subjects_with_abx.csv", index=False)

    print("Wrote:", Path(args.outdir) / "resilience_curvature_subjects.csv")


if __name__ == "__main__":
    main()
