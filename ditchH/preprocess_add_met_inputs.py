#!/usr/bin/env python3
# (see previous cell's content in the assistant's plan) -- full code embedded below

import argparse, re, os, numpy as np, pandas as pd

def svd_pc1_scores(X):
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    Xc = np.where(np.isnan(Xc), 0.0, Xc)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    s = U[:,0]*S[0]
    return (s - np.nanmean(s)) / (np.nanstd(s)+1e-8)

def minmax01(a):
    a = np.asarray(a, float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)

def add_aux_z(df, name_pat, out_name):
    cols = [c for c in df.columns if isinstance(c,str) and re.search(name_pat, c, flags=re.IGNORECASE)]
    if not cols:
        df[out_name] = np.nan; return df
    x = np.log1p(df[cols[0]].astype(float).values)
    mu, sd = np.nanmean(x), np.nanstd(x)
    df[out_name] = (x-mu)/(sd+1e-8)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True)
    ap.add_argument("--metabolomics", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    combined = pd.read_csv(args.combined)
    met = pd.read_csv(args.metabolomics)

    sample_col = "Sample_ID" if "Sample_ID" in combined.columns else ("sample_id" if "sample_id" in combined.columns else None)
    if sample_col is None:
        for c in combined.columns:
            if re.search(r"^sample[_\s-]*id$|^sample$|^id$", str(c), flags=re.IGNORECASE):
                sample_col = c; break
    combined[sample_col] = combined[sample_col].astype(str).str.strip()

    subject_col = None
    for cand in ["Subject","subject","subject_id","Subject_ID","participant","Participant"]:
        if cand in combined.columns: subject_col = cand; break
    if subject_col is None:
        for c in combined.columns:
            if re.search(r"^subject|participant|^id$", str(c), flags=re.IGNORECASE):
                subject_col = c; break

    date_col = None
    for cand in ["Date of Receipt","Date","date","collection_date","sample_date","timestamp","Timepoint","timepoint","Interval Sequence"]:
        if cand in combined.columns: date_col = cand; break
    if date_col is None:
        for c in combined.columns:
            if re.search(r"date|collection|time|interval", str(c), flags=re.IGNORECASE):
                date_col = c; break

    if date_col in combined.columns:
        if "date" in date_col.lower():
            combined[date_col] = pd.to_datetime(combined[date_col], errors="coerce")
        else:
            combined[date_col] = pd.to_numeric(combined[date_col], errors="coerce")

    metab_col = met.columns[0]
    met_long = met.melt(id_vars=[metab_col], var_name=sample_col, value_name="intensity")
    met_long.rename(columns={metab_col:"metabolite"}, inplace=True)
    met_long[sample_col] = met_long[sample_col].astype(str).str.strip()

    PAT_F = {"lactate": r"lactate", "succinate": r"succinate", "propionate": r"\bprop(?!yl)"}
    PAT_A = {"i3p": r"indole.?3.?prop", "lithocholate": r"lithocholate|lca\b", "cholate": r"\bcholate(?!.*glyco)", "glycocholate": r"glycocholate"}
    use_pats = list(PAT_F.values()) + list(PAT_A.values())

    def matches_any(name, patterns):
        return any(re.search(p, name, flags=re.IGNORECASE) for p in patterns)

    met_needed = met_long[met_long["metabolite"].astype(str).apply(lambda s: matches_any(s, use_pats))].copy()
    wide = met_needed.pivot_table(index=sample_col, columns="metabolite", values="intensity", aggfunc="mean").reset_index()
    comb2 = combined.merge(wide, on=sample_col, how="left")

    def zcol(x):
        x = np.log1p(x.astype(float))
        mu, sd = np.nanmean(x), np.nanstd(x)
        if not np.isfinite(sd) or sd < 1e-8:
            return np.zeros_like(x)
        return (x - mu) / (sd + 1e-8)

    def compute_inputs_for_subject(df_subj):
        cols_F = []
        for pat in PAT_F.values():
            cols_F += [c for c in df_subj.columns if isinstance(c,str) and re.search(pat, c, flags=re.IGNORECASE)]
        cols_F = sorted(set(cols_F))
        if cols_F:
            XF = np.column_stack([zcol(df_subj[c].values) for c in cols_F])
            F_pc1 = svd_pc1_scores(XF)
            F_01 = minmax01(F_pc1)
        else:
            F_01 = np.zeros(len(df_subj))

        A_feats = []
        i3p_cols = [c for c in df_subj.columns if re.search(PAT_A["i3p"], str(c), flags=re.IGNORECASE)]
        if i3p_cols:
            A_feats.append(zcol(df_subj[i3p_cols[0]].values))
        lca_cols = [c for c in df_subj.columns if re.search(PAT_A["lithocholate"], str(c), flags=re.IGNORECASE)]
        chol_cols = [c for c in df_subj.columns if re.search(PAT_A["cholate"], str(c), flags=re.IGNORECASE)]
        if lca_cols and chol_cols:
            ratio = np.log1p(df_subj[lca_cols[0]].astype(float).values) - np.log1p(df_subj[chol_cols[0]].astype(float).values + 1e-8)
            ratio = (ratio - np.nanmean(ratio)) / (np.nanstd(ratio)+1e-8)
            A_feats.append(ratio)
        glyco_cols = [c for c in df_subj.columns if re.search(PAT_A["glycocholate"], str(c), flags=re.IGNORECASE)]
        if glyco_cols:
            invg = 1.0 / (np.log1p(df_subj[glyco_cols[0]].astype(float).values) + 1e-6)
            invg = (invg - np.nanmean(invg)) / (np.nanstd(invg)+1e-8)
            A_feats.append(invg)
        if A_feats:
            XA = np.column_stack(A_feats) if len(A_feats)>1 else np.array(A_feats[0])[:,None]
            A_pc1 = svd_pc1_scores(XA)
            A_01 = minmax01(np.maximum(A_pc1, 0.0))
        else:
            A_01 = np.zeros(len(df_subj))

        return pd.DataFrame({"F_met": F_01, "A_met": A_01}, index=df_subj.index)

    if date_col in comb2.columns:
        comb2 = comb2.sort_values([subject_col, date_col, sample_col])
    else:
        comb2 = comb2.sort_values([subject_col, sample_col])

    inputs = []
    for sid, df_s in comb2.groupby(subject_col, sort=False):
        inputs.append(compute_inputs_for_subject(df_s))
    inputs = pd.concat(inputs).sort_index()

    aug = comb2.copy()
    aug["F_met"] = inputs["F_met"].values
    aug["A_met"] = inputs["A_met"].values

    aug = add_aux_z(aug, r"\bprop(?!yl)", "aux_propionate_z")
    aug = add_aux_z(aug, r"succinate", "aux_succinate_z")
    aug = add_aux_z(aug, r"lactate", "aux_lactate_z")

    aug.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
