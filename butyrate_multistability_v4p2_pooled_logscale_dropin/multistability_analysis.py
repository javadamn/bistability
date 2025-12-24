#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p: str) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def detect_subject_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if re.search(r"^subject", str(c), re.I):
            return c
    for c in df.columns:
        if re.search(r"(subject|participant|patient)", str(c), re.I):
            return c
    raise SystemExit("ERROR: subject column not found")


def simulate_latent_switch(
    B0: float,
    dt: np.ndarray,
    F: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    p0: float,
    alpha_F: float,
    alpha_A: float,
    k_LB: float,
    lambda_B: float,
    center: float,
    gap: float,
    h_amp: float,
    mu_z: float,
    sd_z: float,
):
    B = np.zeros(len(dt) + 1, dtype=float)
    E = np.zeros(len(dt) + 1, dtype=int)
    B[0] = max(0.0, float(B0))
    E[0] = 0
    up_th = center + gap / 2.0
    dn_th = center - gap / 2.0

    for i in range(len(dt)):
        z = (np.log1p(max(B[i], 0.0)) - mu_z) / (sd_z if sd_z > 1e-12 else 1.0)
        if E[i] == 0 and z >= up_th:
            E[i + 1] = 1
        elif E[i] == 1 and z <= dn_th:
            E[i + 1] = 0
        else:
            E[i + 1] = E[i]

        h_term = h_amp * (E[i] - 0.5)
        dB = p0 + alpha_F * F[i] - alpha_A * A[i] + k_LB * L[i] + h_term - lambda_B * B[i]
        B[i + 1] = max(0.0, B[i] + dt[i] * dB)

    return B, E


def hysteresis_sweep_F(p0, alpha_F, alpha_A, k_LB, lambda_B, A0, L0, mu_z, sd_z, center, gap, h_amp,
                       F_min=0.0, F_max=1.0, n=80):
    Fs = np.linspace(F_min, F_max, n)
    up_th = center + gap / 2.0
    dn_th = center - gap / 2.0

    # sweep up
    E = 0
    B_prev = 0.0
    up_B = []
    for F in Fs:
        B = B_prev
        for _ in range(25):
            z = (np.log1p(max(B, 0.0)) - mu_z) / (sd_z if sd_z > 1e-12 else 1.0)
            if E == 0 and z >= up_th:
                E2 = 1
            elif E == 1 and z <= dn_th:
                E2 = 0
            else:
                E2 = E
            h_term = h_amp * (E2 - 0.5)
            B2 = (p0 + alpha_F * F - alpha_A * A0 + k_LB * L0 + h_term) / max(lambda_B, 1e-6)
            B2 = max(0.0, B2)
            if abs(B2 - B) < 1e-6 and E2 == E:
                B, E = B2, E2
                break
            B, E = B2, E2
        up_B.append(B)
        B_prev = B

    # sweep down
    E = 1
    B_prev = up_B[-1]
    down_B = []
    for F in Fs[::-1]:
        B = B_prev
        for _ in range(25):
            z = (np.log1p(max(B, 0.0)) - mu_z) / (sd_z if sd_z > 1e-12 else 1.0)
            if E == 0 and z >= up_th:
                E2 = 1
            elif E == 1 and z <= dn_th:
                E2 = 0
            else:
                E2 = E
            h_term = h_amp * (E2 - 0.5)
            B2 = (p0 + alpha_F * F - alpha_A * A0 + k_LB * L0 + h_term) / max(lambda_B, 1e-6)
            B2 = max(0.0, B2)
            if abs(B2 - B) < 1e-6 and E2 == E:
                B, E = B2, E2
                break
            B, E = B2, E2
        down_B.append(B)
        B_prev = B

    return pd.DataFrame({"F": Fs, "B_up": up_B, "B_down": list(reversed(down_B))})


def nullclines_F(p0, alpha_F, alpha_A, k_LB, lambda_B, A0, L0, h_amp, F_min=0.0, F_max=1.0, n=80):
    Fs = np.linspace(F_min, F_max, n)
    B_E0 = (p0 + alpha_F * Fs - alpha_A * A0 + k_LB * L0 + h_amp * (0 - 0.5)) / max(lambda_B, 1e-6)
    B_E1 = (p0 + alpha_F * Fs - alpha_A * A0 + k_LB * L0 + h_amp * (1 - 0.5)) / max(lambda_B, 1e-6)
    return pd.DataFrame({"F": Fs, "B_null_E0": np.clip(B_E0, 0, None), "B_null_E1": np.clip(B_E1, 0, None)})


def gap_sweep_metrics(hs: pd.DataFrame) -> Dict[str, float]:
    area = float(np.trapz(np.abs(hs["B_up"].to_numpy() - hs["B_down"].to_numpy()), hs["F"].to_numpy()))
    max_sep = float(np.max(np.abs(hs["B_up"].to_numpy() - hs["B_down"].to_numpy())))
    return {"hysteresis_area": area, "max_branch_separation": max_sep}


def run_one(sid: str, params: pd.DataFrame, pred: pd.DataFrame, best: dict,
            center: float, gap: float, h_amp: float, gaps_sweep: Optional[List[float]], outdir: Path):
    subj_p = detect_subject_col(params)
    subj = detect_subject_col(pred)

    prow = params[params[subj_p].astype(str) == sid]
    if prow.empty:
        raise SystemExit(f"ERROR: subject {sid} not in params_csv.")
    p0 = float(prow["p0"].iloc[0])

    alpha_F = float(best["alpha_F"])
    alpha_A = float(best["alpha_A"])
    k_LB = float(best["k_LB"])
    lambda_B = float(best["lambda_B"])

    g = pred[pred[subj].astype(str) == sid].copy()
    if g.empty:
        raise SystemExit(f"ERROR: subject {sid} not in predictions.")
    if "aligned_index" in g.columns:
        g = g.sort_values("aligned_index")

    # Expect columns from earlier pipeline
    dt = pd.to_numeric(g.get("dt_days", np.ones(len(g))), errors="coerce").to_numpy(dtype=float)
    F = pd.to_numeric(g["F_in"], errors="coerce").to_numpy(dtype=float)
    A = pd.to_numeric(g["A_in"], errors="coerce").to_numpy(dtype=float)
    L = pd.to_numeric(g["L_in"], errors="coerce").to_numpy(dtype=float)
    B_next = pd.to_numeric(g.get("B_next", np.nan), errors="coerce").to_numpy(dtype=float)
    B_prev = pd.to_numeric(g.get("B_prev", np.nan), errors="coerce").to_numpy(dtype=float)

    # truncate to common length
    n = int(np.min([len(dt), len(F), len(A), len(L), len(B_next), len(B_prev)]))
    dt, F, A, L, B_next, B_prev = dt[:n], F[:n], A[:n], L[:n], B_next[:n], B_prev[:n]

    z_all = np.log1p(np.clip(np.concatenate([B_prev[:1], B_next]), 0, None))
    mu_z = float(np.mean(z_all))
    sd_z = float(np.std(z_all)) if float(np.std(z_all)) > 1e-12 else 1.0

    B_sim, _ = simulate_latent_switch(float(B_prev[0]), dt, F, A, L, p0, alpha_F, alpha_A, k_LB, lambda_B,
                                      center, gap, h_amp, mu_z, sd_z)

    x = np.arange(n)
    B_sim_next = B_sim[1:n + 1]
    m = min(len(x), len(B_next), len(B_sim_next))
    x, B_next, B_sim_next = x[:m], B_next[:m], B_sim_next[:m]

    plt.figure(figsize=(8, 4))
    plt.plot(x, B_next, label="B_obs")
    plt.plot(x, B_sim_next, label="B_sim (latent E)")
    plt.xlabel("aligned step")
    plt.ylabel("butyrate")
    plt.title(f"{sid} — Observed vs simulated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{sid}_traj.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, F[:m], label="F_in")
    plt.plot(x, A[:m], label="A_in")
    plt.plot(x, L[:m], label="L_in")
    plt.xlabel("aligned step")
    plt.ylabel("driver")
    plt.title(f"{sid} — Drivers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{sid}_drivers.png", dpi=200)
    plt.close()

    A0 = float(np.nanmedian(A))
    L0 = float(np.nanmedian(L))
    F_min = float(np.nanmin(F))
    F_max = float(np.nanmax(F))
    if not (np.isfinite(F_min) and np.isfinite(F_max)) or F_max <= F_min:
        F_min, F_max = 0.0, 1.0

    hs = hysteresis_sweep_F(p0, alpha_F, alpha_A, k_LB, lambda_B, A0, L0, mu_z, sd_z, center, gap, h_amp,
                            F_min=F_min, F_max=F_max, n=80)
    plt.figure(figsize=(7, 5))
    plt.plot(hs["F"], hs["B_up"], label="sweep up")
    plt.plot(hs["F"], hs["B_down"], label="sweep down")
    plt.xlabel("F (fixed A0,L0)")
    plt.ylabel("B*")
    plt.title(f"{sid} — Hysteresis vs F (gap={gap:g})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{sid}_hysteresis_F.png", dpi=200)
    plt.close()

    nc = nullclines_F(p0, alpha_F, alpha_A, k_LB, lambda_B, A0, L0, h_amp, F_min=F_min, F_max=F_max, n=80)
    plt.figure(figsize=(7, 5))
    plt.plot(nc["F"], nc["B_null_E0"], label="E=0")
    plt.plot(nc["F"], nc["B_null_E1"], label="E=1")
    plt.xlabel("F (fixed A0,L0)")
    plt.ylabel("B*")
    plt.title(f"{sid} — Nullclines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{sid}_nullclines.png", dpi=200)
    plt.close()

    # gap sweep (optional)
    if gaps_sweep:
        metrics = []
        for gg in gaps_sweep:
            hs2 = hysteresis_sweep_F(p0, alpha_F, alpha_A, k_LB, lambda_B, A0, L0, mu_z, sd_z, center, gg, h_amp,
                                     F_min=F_min, F_max=F_max, n=80)
            ms = gap_sweep_metrics(hs2)
            ms["gap"] = float(gg)
            metrics.append(ms)
        mdf = pd.DataFrame(metrics)
        mdf.to_csv(outdir / f"{sid}_gap_sweep_metrics.csv", index=False)

        plt.figure(figsize=(7, 4))
        plt.plot(mdf["gap"], mdf["hysteresis_area"], label="area")
        plt.plot(mdf["gap"], mdf["max_branch_separation"], label="max_sep")
        plt.xlabel("gap")
        plt.ylabel("metric")
        plt.title(f"{sid} — gap sensitivity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{sid}_gap_sweep.png", dpi=200)
        plt.close()

    return {"subject": sid, "p0": p0, "A0": A0, "L0": L0, "F_min": F_min, "F_max": F_max}


def main():
    ap = argparse.ArgumentParser(description="Latent hysteresis multistability analysis (batch + gap sweep).")
    ap.add_argument("--params_csv", required=True)
    ap.add_argument("--pred_v4", required=True)
    ap.add_argument("--global_json", required=True)  # now supported
    ap.add_argument("--subject", default=None)
    ap.add_argument("--subjects_csv", default=None)
    ap.add_argument("--center", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--h_amp", type=float, default=0.2)
    ap.add_argument("--gap_sweep", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = safe_mkdir(args.outdir)

    params = pd.read_csv(args.params_csv)
    pred = pd.read_csv(args.pred_v4)
    with open(args.global_json, "r", encoding="utf-8") as f:
        gj = json.load(f)
    best = gj.get("best", gj)

    # subject list
    subjects = []
    if args.subject:
        subjects = [str(args.subject)]
    elif args.subjects_csv:
        s = pd.read_csv(args.subjects_csv)
        sc = detect_subject_col(s)
        subjects = s[sc].astype(str).tolist()
    else:
        raise SystemExit("ERROR: provide --subject or --subjects_csv")

    gaps_sweep = None
    if args.gap_sweep:
        gaps_sweep = [float(x) for x in args.gap_sweep.split(",") if x.strip()]

    summary = []
    for sid in subjects:
        summary.append(
            run_one(
                sid=str(sid),
                params=params,
                pred=pred,
                best=best,
                center=float(args.center),
                gap=float(args.gap),
                h_amp=float(args.h_amp),
                gaps_sweep=gaps_sweep,
                outdir=outdir,
            )
        )

    pd.DataFrame(summary).to_csv(outdir / "multistability_run_summary.csv", index=False)
    print(f"Wrote multistability outputs to: {outdir}")


if __name__ == "__main__":
    main()
