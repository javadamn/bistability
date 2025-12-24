#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)

def detect_subject_col(df):
    for c in df.columns:
        if re.search(r"^subject", str(c), re.I): return c
    for c in df.columns:
        if re.search(r"(subject|participant|patient).*id|(^|_)id($|_)", str(c), re.I): return c
    for c in df.columns:
        if re.search(r"(subject|participant|patient)", str(c), re.I): return c
    raise SystemExit("ERROR: Could not detect subject id column.")

def detect_time_col(df):
    pats = [r"^date of receipt$", r"^date$", r"collection", r"sample.*date", r"timestamp", r"interval sequence"]
    for pat in pats:
        for c in df.columns:
            if re.search(pat, str(c), re.I):
                return c
    # fallback: try parse-able object col
    for c in df.columns:
        if df[c].dtype == object:
            t = pd.to_datetime(df[c].astype(str).head(50), errors="coerce")
            if t.notna().mean() >= 0.7:
                return c
    raise SystemExit("ERROR: Could not detect time column.")

def detect_butyrate_col(df):
    for c in df.columns:
        if re.search(r"butyrate|butyric|(^|[^A-Za-z])c4([^A-Za-z]|$)", str(c), re.I):
            return c
    raise SystemExit("ERROR: Could not detect butyrate column.")

def parse_time(s):
    t = pd.to_datetime(s, errors="coerce")
    if t.notna().mean() >= 0.7:
        return (t.view("int64") / 1e9).astype(float), "datetime"
    return pd.to_numeric(s, errors="coerce").astype(float), "numeric"

def huber_loss(r, delta=0.5):
    r = np.asarray(r, float)
    a = np.abs(r)
    q = a <= delta
    return float(np.sum(0.5 * r[q] ** 2) + np.sum(delta * (a[~q] - 0.5 * delta)))

def robust_scale(x):
    x = np.asarray(x, float)
    med = float(np.nanmedian(x))
    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr < 1e-12:
        iqr = 1.0
    return (x - med) / iqr

def build_transitions(df, subj_col, time_col, B_col, x_col, min_pairs=5):
    rows = []
    for sid, g0 in df.groupby(subj_col, sort=False):
        g = g0.copy()
        t_raw, t_kind = parse_time(g[time_col])
        g["_t_"] = t_raw
        g = g.sort_values("_t_")
        tt = g["_t_"].to_numpy(float)
        dt = np.diff(tt)
        if len(dt) == 0:
            continue
        dt_days = dt / (24 * 3600.0) if t_kind == "datetime" else dt
        if np.any(~np.isfinite(dt_days)) or np.any(dt_days <= 0):
            continue

        B = pd.to_numeric(g[B_col], errors="coerce").to_numpy(float)
        z = np.log1p(np.clip(B, 0, None))

        X = pd.to_numeric(g[x_col], errors="coerce").to_numpy(float)

        for i in range(0, len(g) - 1):
            if not (np.isfinite(z[i]) and np.isfinite(z[i+1]) and np.isfinite(X[i]) and np.isfinite(dt_days[i])):
                continue
            rows.append({"subject": str(sid), "z_prev": float(z[i]), "dz": float(z[i+1] - z[i]), "X": float(X[i])})

    out = pd.DataFrame(rows)
    # drop subjects with too few
    if out.empty:
        return out
    cnt = out.groupby("subject").size()
    keep = cnt[cnt >= min_pairs].index.astype(str).tolist()
    return out[out["subject"].isin(keep)].copy()

def fit_fixed_effects_dz(dataset, use_X: bool):
    # dz = b z_prev + gamma X + subject intercepts
    z_prev = dataset["z_prev"].to_numpy(float)
    dz = dataset["dz"].to_numpy(float)
    subj = dataset["subject"].astype(str).to_numpy()
    uniq = sorted(pd.unique(subj).tolist())
    s_to_i = {s:i for i,s in enumerate(uniq)}
    S = np.zeros((len(subj), len(uniq)), float)
    for r, s in enumerate(subj):
        S[r, s_to_i[s]] = 1.0

    if use_X:
        X = dataset["X"].to_numpy(float)
        M = np.c_[z_prev, X, S]
    else:
        M = np.c_[z_prev, S]

    coef, _, _, _ = np.linalg.lstsq(M, dz, rcond=None)
    pred = M @ coef
    resid = dz - pred
    return coef, resid, uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metabolites", type=str, default="cholate,glycocholate,imidazolelactate,indole-3-propionate,lactate,lithocholate,propionate_y,succinate",
                    help="Comma-separated metabolite columns to screen.")
    ap.add_argument("--min_pairs", type=int, default=6)
    ap.add_argument("--delta", type=float, default=0.5)
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    df = pd.read_csv(args.csv)
    subj_col = detect_subject_col(df)
    time_col = detect_time_col(df)
    B_col = detect_butyrate_col(df)

    mets = [m.strip() for m in args.metabolites.split(",") if m.strip()]
    missing = [m for m in mets if m not in df.columns]
    if missing:
        print("NOTE: Missing columns (skipped):", missing)
    mets = [m for m in mets if m in df.columns]
    if not mets:
        raise SystemExit("ERROR: No metabolite columns found to screen.")

    results = []
    for m in mets:
        D = build_transitions(df, subj_col, time_col, B_col, m, min_pairs=args.min_pairs)
        if D.empty:
            results.append({"metabolite": m, "n_rows": 0, "n_subjects": 0,
                            "loss_base": np.nan, "loss_plus": np.nan, "delta_loss": np.nan,
                            "gamma": np.nan})
            continue

        # within-subject center X to avoid baseline confounding
        D["X"] = D["X"] - D.groupby("subject")["X"].transform("mean")
        # robust scale X
        D["X"] = robust_scale(D["X"].to_numpy(float))

        coef0, resid0, _ = fit_fixed_effects_dz(D, use_X=False)
        coef1, resid1, _ = fit_fixed_effects_dz(D, use_X=True)

        loss0 = huber_loss(resid0, delta=args.delta)
        loss1 = huber_loss(resid1, delta=args.delta)

        gamma = float(coef1[1])  # [b, gamma, subject...]
        results.append({
            "metabolite": m,
            "n_rows": int(len(D)),
            "n_subjects": int(D["subject"].nunique()),
            "loss_base": float(loss0),
            "loss_plus": float(loss1),
            "delta_loss": float(loss0 - loss1),  # positive means improvement
            "gamma": gamma,
        })

    R = pd.DataFrame(results).sort_values("delta_loss", ascending=False)
    R.to_csv(Path(args.outdir) / "metabolite_driver_screen.csv", index=False)

    # plot delta_loss
    # --- NEW: top-K focused plots for paper ---
    topK = 5
    R_top = R.head(topK).copy()
    R_top.to_csv(Path(args.outdir) / "metabolite_driver_screen_top3.csv", index=False)

    plt.figure(figsize=(5.2, 3.6))
    plt.bar(np.arange(len(R_top)), R_top["delta_loss"].to_numpy(float))
    plt.xticks(np.arange(len(R_top)), R_top["metabolite"].tolist(), rotation=30, ha="right")
    plt.ylabel("Delta Huber loss (base - plus metabolite)")
    plt.title("Top 3 candidate metabolite modulators")
    plt.tight_layout()
    plt.savefig(Path(args.outdir) / "metabolite_top3_delta_loss.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5.2, 3.6))
    plt.bar(np.arange(len(R_top)), R_top["gamma"].to_numpy(float))
    plt.xticks(np.arange(len(R_top)), R_top["metabolite"].tolist(), rotation=30, ha="right")
    plt.ylabel("gamma (effect on dz)")
    plt.title("Top 3 metabolite effects on butyrate drift")
    plt.tight_layout()
    plt.savefig(Path(args.outdir) / "metabolite_top3_gamma.png", dpi=220)
    plt.close()


    print("Wrote:", Path(args.outdir) / "metabolite_driver_screen.csv")
    print("Wrote plots in:", args.outdir)


if __name__ == "__main__":
    main()
