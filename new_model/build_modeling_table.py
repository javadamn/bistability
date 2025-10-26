
"""
Utility to build a modeling table from the raw IBD SCFA time-series CSV.

Usage:
    python build_modeling_table.py --src timeseries/combined_scfas_table_raw.csv --out outputs/modeling_table.csv
"""
import argparse, re, os
from pathlib import Path
import numpy as np
import pandas as pd

def pick_best_column(candidates, frame):
    cols = [c for c in frame.columns for q in candidates if q.lower() in c.lower()]
    cols = sorted(set(cols), key=lambda c: -frame[c].notna().sum())
    return cols[0] if cols else None

def first_nonnull_columns(cols, frame):
    if not cols:
        return pd.Series([np.nan]*len(frame), index=frame.index)
    out = pd.Series([np.nan]*len(frame), index=frame.index, dtype="float64")
    for c in cols:
        if c in frame.columns:
            s = frame[c]
            mask = out.isna() & s.notna()
            out.loc[mask] = pd.to_numeric(s[mask], errors="coerce")
    return out

def to_datetime_best(series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

DIET_MAP = {'never': 0.0, 'rarely': 0.1, 'seldom': 0.1, 'once per week': 0.2, '1 time per week': 0.2, 'weekly': 0.3, 'few times per month': 0.3, 'several times per week': 0.6, '3-4 times per week': 0.6, 'daily': 1.0, 'almost daily': 0.9, 'every day': 1.0, '2+ times per day': 1.0}

def map_choice_to01(val):
    import numpy as np
    import pandas as pd
    if pd.isna(val):
        return np.nan
    low = str(val).strip().lower()
    for k, v in DIET_MAP.items():
        if k in low:
            return v
    if any(w in low for w in ["never", "none"]):
        return 0.0
    if any(w in low for w in ["rare", "seldom"]):
        return 0.1
    if any(w in low for w in ["once", "1 time"]):
        return 0.2
    if any(w in low for w in ["week"]):
        return 0.5
    if any(w in low for w in ["daily", "every day", "day"]):
        return 1.0
    return np.nan

def normalize01(s):
    import numpy as np
    x = s.astype(float).copy()
    lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) if np.nanmax(x) != np.nanmin(x) else x*0+0.5
    y = (x - lo) / (hi - lo)
    return y.clip(0, 1)

def zscore_by_subject(s, subjects):
    import numpy as np
    out = s.astype(float).copy()
    for sid, idx in subjects.groupby(subjects).groups.items():
        sub = out.loc[idx]
        mu = np.nanmean(sub)
        sd = np.nanstd(sub)
        if not np.isfinite(sd) or sd == 0:
            out.loc[idx] = sub - (mu if np.isfinite(mu) else 0.0)
        else:
            out.loc[idx] = (sub - mu) / sd
    return out

def any_yes_like(val):
    import pandas as pd
    if pd.isna(val):
        return False
    low = str(val).lower()
    return any(w in low for w in ["yes", "y", "true", "on", "present"])

def any_antibiotic_like(val):
    import pandas as pd
    if pd.isna(val):
        return False
    low = str(val).lower()
    return any(w in low for w in ["antibiot", "amoxic", "cipro", "metronid", "flagyl", "augmentin",
                                  "doxy", "azith", "rifax", "rifaximin", "levoflox", "clarith",
                                  "clindamy", "penicillin", "macrolide", "quinolone"])

def build_table(src, out):
    df = pd.read_csv(src)
    subject_col = pick_best_column(["subject_id", "participant", "patient"], df) or "subject_id"
    sample_col  = pick_best_column(["sample_id", "sample"], df) or "sample_id"
    date_col    = pick_best_column(["date of receipt", "collection date", "date"], df)

    butyrate_col = pick_best_column(["butyr", "butyrate", "c4"], df)

    calp_cols = [c for c in df.columns if "calprotectin" in c.lower()]
    hbi_col   = pick_best_column(["hbi"], df)
    sccai_col = pick_best_column(["sccai"], df)

    diet_cols = [c for c in df.columns if any(k in c.lower() for k in [
        "whole grains", "vegetables", "fruits", "yogurt", "kefir", "sauerkraut", "fermented"
    ])]
    abx_cols = [c for c in df.columns if "antibi" in c.lower() or "abx" in c.lower() or "medication" in c.lower()]

    mdl = pd.DataFrame(index=df.index)
    mdl["subject_id"] = df.get(subject_col)
    mdl["sample_id"]  = df.get(sample_col)
    dates = to_datetime_best(df.get(date_col)) if date_col in df.columns else pd.NaT
    mdl["date"] = dates

    mdl["t_days"] = np.nan
    if mdl["subject_id"].notna().any() and mdl["date"].notna().any():
        for sid, idx in mdl.groupby("subject_id").groups.items():
            t = mdl.loc[idx, "date"]
            if t.notna().any():
                t0 = t.min()
                mdl.loc[idx, "t_days"] = (t - t0).dt.days

    mdl["butyrate_raw"] = pd.to_numeric(df[butyrate_col], errors="coerce") if butyrate_col else np.nan

    calp_series = first_nonnull_columns(calp_cols, df) if calp_cols else pd.Series([np.nan]*len(df))
    hbi_series   = pd.to_numeric(df[hbi_col], errors="coerce") if hbi_col else pd.Series([np.nan]*len(df))
    sccai_series = pd.to_numeric(df[sccai_col], errors="coerce") if sccai_col else pd.Series([np.nan]*len(df))

    inflam = calp_series.copy()
    if inflam.isna().all():
        parts = []
        if not hbi_series.isna().all():
            parts.append((hbi_series - hbi_series.mean(skipna=True)) / (hbi_series.std(skipna=True) or 1))
        if not sccai_series.isna().all():
            parts.append((sccai_series - sccai_series.mean(skipna=True)) / (sccai_series.std(skipna=True) or 1))
        if parts:
            inflam = pd.concat(parts, axis=1).mean(axis=1)
        else:
            inflam = pd.Series([np.nan]*len(df))

    inflam_z = zscore_by_subject(inflam, mdl["subject_id"]) if mdl["subject_id"].notna().any() else inflam
    mdl["H_proxy"] = -inflam_z

    if diet_cols:
        diet_matrix = pd.DataFrame({c: [map_choice_to01(v) for v in df[c]] for c in diet_cols})
        mdl["F_raw"] = diet_matrix.mean(axis=1, skipna=True)
        mdl["F"] = np.nan
        for sid, idx in mdl.groupby("subject_id").groups.items():
            mdl.loc[idx, "F"] = normalize01(mdl.loc[idx, "F_raw"])
    else:
        mdl["F_raw"] = np.nan
        mdl["F"] = np.nan

    def abx_indicator_row(row) -> int:
        for c in abx_cols:
            val = row.get(c, np.nan)
            if any_yes_like(val) or any_antibiotic_like(val):
                return 1
        return 0

    if abx_cols:
        mdl["A"] = df[abx_cols].apply(lambda row: abx_indicator_row(row), axis=1)
    else:
        mdl["A"] = 0

    mdl["butyrate"] = mdl["butyrate_raw"]

    keep_cols = ["subject_id","sample_id","date","t_days","butyrate","A","F","H_proxy","butyrate_raw","F_raw"]
    modeling_table = mdl[keep_cols].copy().sort_values(["subject_id","date"])

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    modeling_table.to_csv(out, index=False)
    return modeling_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="timeseries/combined_scfas_table_raw.csv")
    ap.add_argument("--out", type=str, default="outputs/modeling_table.csv")
    args = ap.parse_args()
    modeling_table = build_table(args.src, args.out)
    print(f"Wrote: {{args.out}}  (rows={{len(modeling_table)}}, cols={{modeling_table.shape[1]}})")

if __name__ == "__main__":
    main()
