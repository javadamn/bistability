#!/usr/bin/env python3
import argparse, re
from pathlib import Path
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


def detect_abx_col(df: pd.DataFrame):
    for c in df.columns:
        if re.search(r"antibiotic", str(c), re.I):
            return c
    return None


def is_yes(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in ["yes", "y", "true", "1", "t"]


def main():
    ap = argparse.ArgumentParser(description="Plot B_obs vs B_hat and drivers; optionally overlay antibiotic events.")
    ap.add_argument("--pred_v4", required=True)
    ap.add_argument("--subjects_csv", required=True)
    ap.add_argument("--max_plots", type=int, default=6)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--meta_csv", default=None)
    ap.add_argument("--abx_col", default=None)
    args = ap.parse_args()

    outdir = safe_mkdir(args.outdir)
    pred = pd.read_csv(args.pred_v4)
    subj_col = detect_subject_col(pred)

    # which x?
    xcol = "aligned_index" if "aligned_index" in pred.columns else None

    # read subjects
    sub = pd.read_csv(args.subjects_csv)
    s_col = detect_subject_col(sub)
    subjects = sub[s_col].astype(str).tolist()[: args.max_plots]

    # build ABX map: subject -> set(positions in sorted original time series)
    abx_map = {}
    if args.meta_csv:
        meta = pd.read_csv(args.meta_csv)
        meta_subj = detect_subject_col(meta)
        abx_col = args.abx_col if (args.abx_col and args.abx_col in meta.columns) else detect_abx_col(meta)
        if abx_col:
            # sort within subject by any reasonable "date-like" col if exists; else keep file order
            time_col = None
            for c in meta.columns:
                if re.search(r"date|time|timestamp|collection", str(c), re.I):
                    time_col = c
                    break
            for sid, g in meta.groupby(meta_subj, sort=False):
                if time_col:
                    gg = g.copy()
                    gg["_t"] = pd.to_datetime(gg[time_col], errors="coerce")
                    gg = gg.sort_values("_t")
                else:
                    gg = g
                flags = gg[abx_col].map(is_yes).to_numpy()
                idxs = set(np.where(flags)[0].tolist())
                if idxs:
                    abx_map[str(sid)] = idxs

    for sid in subjects:
        g = pred[pred[subj_col].astype(str) == str(sid)].copy()
        if g.empty:
            continue
        if xcol:
            g = g.sort_values(xcol)
            x = pd.to_numeric(g[xcol], errors="coerce").to_numpy()
        else:
            x = np.arange(len(g))

        B_obs = pd.to_numeric(g.get("B_next", np.nan), errors="coerce").to_numpy()
        B_hat = pd.to_numeric(g.get("B_hat_next", g.get("B_hat", np.nan)), errors="coerce").to_numpy()

        F = pd.to_numeric(g.get("F_in", np.nan), errors="coerce").to_numpy()
        A = pd.to_numeric(g.get("A_in", np.nan), errors="coerce").to_numpy()
        L = pd.to_numeric(g.get("L_in", np.nan), errors="coerce").to_numpy()

        n = int(np.min([len(x), len(B_obs), len(B_hat)]))
        x1, B_obs1, B_hat1 = x[:n], B_obs[:n], B_hat[:n]

        # ABX markers: if aligned_index exists, align ABX with i or i+1 from original series
        abx_lines = []
        if str(sid) in abx_map and xcol:
            abx_idxs = abx_map[str(sid)]
            ai = pd.to_numeric(g["aligned_index"], errors="coerce").to_numpy()[:n]
            for j, aij in enumerate(ai):
                if not np.isfinite(aij):
                    continue
                ii = int(aij)
                if ii in abx_idxs or (ii + 1) in abx_idxs:
                    abx_lines.append(x1[j])

        plt.figure(figsize=(8, 4))
        plt.plot(x1, B_obs1, label="B_obs")
        plt.plot(x1, B_hat1, label="B_hat")
        for xv in abx_lines:
            plt.axvline(x=xv, linestyle="--", linewidth=1)
        plt.xlabel("aligned_index" if xcol else "index")
        plt.ylabel("butyrate")
        plt.title(f"{sid} — Butyrate fit" + (" (ABX marked)" if abx_lines else ""))
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{sid}_butyrate_fit.png", dpi=200)
        plt.close()

        n2 = int(np.min([len(x), len(F), len(A), len(L)]))
        xx = x[:n2]
        abx_lines2 = abx_lines  # reuse
        plt.figure(figsize=(8, 4))
        plt.plot(xx, F[:n2], label="F_in")
        plt.plot(xx, A[:n2], label="A_in")
        plt.plot(xx, L[:n2], label="L_in")
        for xv in abx_lines2:
            plt.axvline(x=xv, linestyle="--", linewidth=1)
        plt.xlabel("aligned_index" if xcol else "index")
        plt.ylabel("driver")
        plt.title(f"{sid} — Drivers" + (" (ABX marked)" if abx_lines2 else ""))
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{sid}_drivers.png", dpi=200)
        plt.close()

    print(f"Wrote panels to: {outdir}")


if __name__ == "__main__":
    main()
