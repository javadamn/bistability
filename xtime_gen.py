#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# -----------------------
# Helpers
# -----------------------
def read_text(p):
    return Path(p).read_text(errors="ignore")

def norm_key(k):
    k = str(k).strip().strip(":=").strip()
    k = re.sub(r"\s+", "_", k)
    return k

def parse_pairs_pipe(s):
    out = {}
    if not isinstance(s, str): return out
    for piece in s.split("|"):
        piece = piece.strip()
        if not piece: continue
        if ":" in piece:
            k,v = piece.split(":", 1)
            out[norm_key(k)] = v.strip()
    return out

def parse_pairs_semicolon(s):
    out = {}
    if not isinstance(s, str): return out
    for piece in s.split(";"):
        piece = piece.strip()
        if not piece: continue
        if "=" in piece:
            k,v = piece.split("=", 1)
            out[norm_key(k)] = v.strip()
    return out

def parse_date(x):
    x = str(x).strip()
    for fmt in ("%m/%d/%Y","%Y-%m-%d","%d-%b-%Y","%m/%d/%y"):
        try: return datetime.strptime(x, fmt)
        except: pass
    return pd.NaT

# -----------------------
# 1) Parse SUBJECT_SAMPLE_FACTORS → subject–sample map + time
# -----------------------
def parse_subject_sample_factors(txt):
    m = re.search(r"#SUBJECT_SAMPLE_FACTORS(.*?)(?:\n#|\Z)", txt, flags=re.S)
    if not m:
        raise RuntimeError("Could not find #SUBJECT_SAMPLE_FACTORS section.")
    block = m.group(1)

    # First non-empty line with tabs is header
    lines = [ln for ln in block.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if "\t" in ln and "SUBJECT" in ln and "SAMPLE" in ln:
            header_idx = i; break
    if header_idx is None:
        raise RuntimeError("Could not find header row in SUBJECT_SAMPLE_FACTORS.")

    header = [c.strip() for c in lines[header_idx].split("\t")]
    rows = []
    for ln in lines[header_idx+1:]:
        parts = [c.strip() for c in ln.split("\t")]
        if len(parts) < len(header):
            parts += [""]*(len(header)-len(parts))
        rows.append(parts[:len(header)])
    df = pd.DataFrame(rows, columns=header)

    # Identify columns
    # (Some files include a leading label column; we just find the columns by name)
    subj_col = next((c for c in df.columns if c.lower().startswith("subject")), None)
    samp_col = next((c for c in df.columns if c.lower().startswith("sample")), None)
    fac_col  = next((c for c in df.columns if "factors" in c.lower()), None)
    add_col  = next((c for c in df.columns if "additional sample data" in c.lower()), None)
    if not (subj_col and samp_col and fac_col and add_col):
        raise RuntimeError("Missing expected columns in SUBJECT_SAMPLE_FACTORS block.")

    # Expand rows
    recs = []
    for _, r in df.iterrows():
        subj = str(r[subj_col]).strip()
        smid = str(r[samp_col]).strip()
        facs = parse_pairs_pipe(r[fac_col])
        adds = parse_pairs_semicolon(r[add_col])
        rec = {"subject_id": subj, "local_sample_id": smid}
        rec.update(facs); rec.update(adds)
        recs.append(rec)
    meta = pd.DataFrame(recs).drop_duplicates(subset=["local_sample_id"], keep="first")

    # Build time_hr
    time_hr = np.full(len(meta), np.nan)
    if "Interval_days" in meta.columns:
        v = pd.to_numeric(meta["Interval_days"], errors="coerce")
        time_hr = np.where(~np.isnan(v), v*24.0, np.nan)

    if np.isnan(time_hr).all() and "Week_Number" in meta.columns:
        v = pd.to_numeric(meta["Week_Number"], errors="coerce")
        time_hr = np.where(~np.isnan(v), v*7*24.0, np.nan)

    if "Date_of_Receipt" in meta.columns and np.isnan(time_hr).all():
        meta["_date"] = meta["Date_of_Receipt"].map(parse_date)
        for sid, idx in meta.groupby("subject_id").groups.items():
            s = meta.loc[idx]
            if s["_date"].notna().any():
                t0 = s["_date"].min()
                time_hr[idx] = (s["_date"] - t0).dt.total_seconds().to_numpy()/3600.0
        meta.drop(columns=["_date"], inplace=True, errors="ignore")

    if np.isnan(time_hr).all():
        meta = meta.sort_values(["subject_id", "local_sample_id"])
        meta["time_hr"] = meta.groupby("subject_id").cumcount().astype(float)
    else:
        meta["time_hr"] = time_hr

    return meta

# -----------------------
# 2) Parse “Samples … SM-… SM-… …” inline matrix → long table
# -----------------------
def parse_inline_samples_matrix(txt):
    """
    Assumes a single header line that begins with 'Samples ' and then
    lists SM-IDs separated by spaces, possibly followed by 'Factors …' text.
    Each subsequent metabolite line starts with the metabolite name
    followed by one number per sample (same order as header).
    """
    # Find the header line starting with 'Samples '
    hdr = re.search(r"(?m)^Samples\s+(.+)$", txt)
    if not hdr:
        raise RuntimeError("Could not find 'Samples ...' header line.")
    hdr_line = hdr.group(0)

    # Extract ordered sample IDs from the header
    tokens = hdr_line.split()
    # start after the literal "Samples"
    toks = tokens[1:]
    sm_ids = []
    for t in toks:
        if t.startswith("SM-"):
            sm_ids.append(t)
        else:
            # we've reached 'Factors' or other text; stop
            break
    if not sm_ids:
        raise RuntimeError("No SM- sample IDs parsed from the 'Samples' header.")

    # The rest of the block with metabolite rows:
    # Heuristic: it’s the text from the next line after header until the end of the block
    start = hdr.end()
    tail = txt[start:].strip()

    # Each metabolite row: begins with a name (can include hyphens/digits), then numbers
    # We’ll split lines and parse the first token as name, then read N numbers afterward.
    rows = []
    # Many files wrap lines; so collect numeric runs of length >= len(sm_ids)
    # Strategy: find occurrences: <name> <n1> <n2> ... <nK>, K>=len(sm_ids)
    # We’ll do a two-pass: split by newline, then try to harvest numbers.
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    i = 0
    N = len(sm_ids)
    while i < len(lines):
        ln = lines[i].strip()
        # Find the first float in the line; everything before is the metabolite name
        # Replace commas to be safe; allow scientific notation.
        parts = ln.split()
        # If the line begins with 'Factors', skip and move on (some files echo factors again)
        if parts[0].lower().startswith("factors"):
            i += 1; continue

        # Identify where numbers start
        num_start = None
        for j, tok in enumerate(parts):
            if re.match(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$", tok):
                num_start = j; break
        if num_start is None:
            # Might be a metamultiline (name on this line, numbers begin next line)
            # Treat whole line as a name and try to read numbers from subsequent lines
            name = ln.strip()
            nums = []
            k = i+1
            while k < len(lines) and len(nums) < N:
                for tok in lines[k].split():
                    if re.match(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$", tok):
                        nums.append(float(tok))
                k += 1
            if len(nums) >= N:
                rows.append((name, nums[:N]))
                i = k
                continue
            else:
                i += 1
                continue

        # Normal case: name followed by numbers on same (and possibly following) lines
        name = " ".join(parts[:num_start]).strip()
        nums = []
        # collect numbers on this line from num_start
        for tok in parts[num_start:]:
            if re.match(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$", tok):
                nums.append(float(tok))
        # keep consuming subsequent lines until we have at least N numbers
        k = i+1
        while len(nums) < N and k < len(lines):
            for tok in lines[k].split():
                if re.match(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$", tok):
                    nums.append(float(tok))
            k += 1

        if len(nums) >= N and name:
            rows.append((name, nums[:N]))
            i = k
        else:
            i += 1

    # Build long dataframe
    recs = []
    for name, nums in rows:
        for sm, val in zip(sm_ids, nums):
            recs.append({"metabolite_name": name, "local_sample_id": sm, "intensity": val})
    long = pd.DataFrame(recs)
    return long, sm_ids

# -----------------------
# 3) Simple feature engineering (optional)
# -----------------------
def build_features(meta):
    freq_map = {
        "No, I did not consume these products in the last 7 days": 0,
        "Within the past 4 to 7 days": 1,
        "Within the past 2 to 3 days": 2,
        "Yesterday, 1 to 2 times": 3,
        "Yesterday, 3 or more times": 4,
    }
    def score(col):
        return meta[col].map(freq_map).fillna(0)

    fiber_cols = [
        "Whole_grains_(wheat,_oats,_brown_rice,_rye,_quinoa,_wheat_bread,_wheat_pasta)",
        "Beans_(tofu,_soy,_soy_burgers,_lentils,_Mexican_beans,_lima_beans_etc)",
        "Fruits_(no_juice)_(Apples,_raisins,_bananas,_oranges,_strawberries,_blueberries",
        "Vegetables_(salad,_tomatoes,_onions,_greens,_carrots,_peppers,_green_beans,_etc)",
    ]
    meta["fiber_score"] = sum(score(c) for c in fiber_cols if c in meta.columns)

    fer_cols = ["Yogurt_or_other_foods_containing_active_bacterial_cultures_(kefir,_sauerkraut)"]
    if "Probiotic" in meta.columns:
        probiotic_score = meta["Probiotic"].map(freq_map).fillna(0)
    else:
        probiotic_score = 0
    meta["fermented_score"] = sum(score(c) for c in fer_cols if c in meta.columns) + probiotic_score

    neg_cols = [
        "Processed_meat_(other_red_or_white_meat_such_as_lunch_meat,_ham,_salami,_bologna",
        "Red_meat_(beef,_hamburger,_pork,_lamb)"
    ]
    meta["procmeat_score"] = sum(score(c) for c in neg_cols if c in meta.columns)

    bin_cols = [
        "Antibiotics",
        "Immunosuppressants_(e.g._oral_corticosteroids)",
        "Chemotherapy",
        "In_the_past_2_weeks,_have_you_been_hospitalized?",
        "In_the_past_2_weeks,_have_you_used_an_oral_contrast?",
        "In_the_past_2_weeks,_have_you_undergone_a_colonoscopy_or_other_procedure",
    ]
    for k in bin_cols:
        if k in meta.columns:
            meta[k+"_bin"] = meta[k].str.strip().str.lower().map({"yes":1,"no":0}).fillna(0).astype(int)

    # HBI proxy (very rough)
    hbi_map_wellbeing = {"Very well":0,"Well":1,"Slightly below par":2,"Poor":3,"Terrible":4}
    if "General_wellbeing" in meta.columns:
        meta["HBI_gw"] = meta["General_wellbeing"].map(hbi_map_wellbeing)
    hbi_map_pain = {"None":0,"Mild":1,"Moderate":2,"Severe":3}
    if "Abdominal_pain" in meta.columns:
        meta["HBI_pain"] = meta["Abdominal_pain"].map(hbi_map_pain)
    if "Number_of_liquid_or_very_soft_stools_in_the_past_24_hours:" in meta.columns:
        meta["HBI_stools"] = pd.to_numeric(meta["Number_of_liquid_or_very_soft_stools_in_the_past_24_hours:"], errors="coerce")

    h_cols = [c for c in ["HBI_gw","HBI_pain","HBI_stools"] if c in meta.columns]
    if h_cols:
        h_raw = meta[h_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        meta["H_proxy"] = 1 - (h_raw - h_raw.min())/(h_raw.max()-h_raw.min() + 1e-9)

    keep = ["subject_id","local_sample_id","time_hr","fiber_score","fermented_score","procmeat_score","H_proxy"] + \
           [c for c in meta.columns if c.endswith("_bin")]
    return meta[[c for c in keep if c in meta.columns]].copy()

# -----------------------
# 4) Main
# -----------------------
def run(input_txt, out_prefix="MW_parsed"):
    txt = read_text(input_txt)

    # subject–sample map
    meta = parse_subject_sample_factors(txt)
    meta.to_csv("subject_sample_map.csv", index=False)

    # inline samples matrix → long
    long, sm_order = parse_inline_samples_matrix(txt)
    # join meta
    if "Diagnosis" in meta.columns:
        join_cols = ["local_sample_id","subject_id","time_hr","Diagnosis"]
    else:
        join_cols = ["local_sample_id","subject_id","time_hr"]
    long = long.merge(meta[join_cols], on="local_sample_id", how="left")
    long.to_csv("metabolite_long.csv", index=False)

    # try butyrate (may not be present in this analysis)
    def norm(s): return str(s).strip().lower().replace(" ","").replace("-","")
    mask = long["metabolite_name"].map(lambda x: norm(x) in {"butyrate","butyricacid","butanoate","butanoicacid"})
    if mask.any():
        long.loc[mask, ["subject_id","local_sample_id","time_hr","intensity"]]\
            .rename(columns={"intensity":"butyrate_intensity"})\
            .to_csv("butyrate_timeseries.csv", index=False)

    # features
    feats = build_features(meta)
    feats.to_csv("features_for_model.csv", index=False)

    print("Wrote: subject_sample_map.csv, metabolite_long.csv, features_for_model.csv")
    if mask.any():
        print("Wrote: butyrate_timeseries.csv")
    else:
        print("Butyrate not present here — use AN001514 for butyrate and join by local_sample_id.")

if __name__ == "__main__":
    # Example:
    run("ST000923_AN001515.txt")
    # pass
