#!/usr/bin/env python3
# tune_globals_fast.py — speedy coordinate grid tuner for cohort globals (with live progress + ETA)
import json, os, tempfile, math, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from fit_subject_slim import fit_subject_slim

TUNE_KEYS = ["u","k_B","g","K_B","d0","eta"]  # always tuned

# Reasonable bounds to keep search stable
BOUNDS = {
    "u":  (1e-4, 1.0),
    "k_B":(1e-4, 1.0),
    "g":  (0.05, 4.0),
    "K_B":(0.05, 3.0),
    "d0": (0.01, 0.5),
    "eta":(1.0, 20.0),
}

# ---------- utilities ----------
def now_s():
    return time.time()

def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h:d}h {m:02d}m {s:02d}s"
    if m: return f"{m:d}m {s:02d}s"
    return f"{s:d}s"

def progress_bar(done: int, total: int, width: int = 40) -> str:
    total = max(total, 1)
    frac = min(max(done / total, 0.0), 1.0)
    filled = int(frac * width)
    return "[" + "#"*filled + "-"*(width-filled) + f"] {int(frac*100):3d}%"

def log_jsonl(fp, payload: dict):
    if not fp: return
    fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    fp.flush()

def load_subjects(csv_path: str, min_obs_b: int, subset: Optional[List[str]]):
    dfm = pd.read_csv(csv_path, parse_dates=["date"])
    if subset:
        dfm = dfm[dfm["subject_id"].astype(str).isin([str(s) for s in subset])]
    counts = dfm.dropna(subset=["butyrate"]).groupby("subject_id").size()
    subjects = [str(s) for s, n in counts.items() if n >= min_obs_b]
    return sorted(subjects)

def eval_cost_for_globals(globals_dict: Dict, subjects: List[str], csv: str,
                          n_starts=6, max_nfev=200, logbase="10", use_logB=True,
                          per_subject_cb=None) -> float:
    """
    per_subject_cb(sid, cost, idx, n_subjects) is called after each subject fit (if provided)
    """
    # write globals to a temp file and pass its path to the fitter
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(globals_dict, tf)
        tmp = tf.name
    total = 0.0
    try:
        for i, sid in enumerate(subjects, 1):
            try:
                fit = fit_subject_slim(
                    sid, csv, globals_json=tmp,
                    use_logB=use_logB, logbase=logbase,
                    max_nfev=max_nfev, n_starts=n_starts
                )
                c = float(fit["cost"])
            except Exception as e:
                # Fail-soft: count a large penalty so search can continue
                c = 1e9
            total += c
            if per_subject_cb:
                per_subject_cb(sid, c, i, len(subjects))
    finally:
        try: os.remove(tmp)
        except OSError: pass
    return total

def clamp(val: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, val)))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/modeling_table_with_indicators.csv")
    ap.add_argument("--init_globals", default="globals_cohort.json")
    ap.add_argument("--out_json", default="globals_cohort_tuned_fast.json")
    ap.add_argument("--subjects", nargs="*", default=None, help="subset for tuning (3–6 good subjects)")
    ap.add_argument("--min_obs_b", type=int, default=5)
    ap.add_argument("--logbase", choices=["e","10"], default="10")
    ap.add_argument("--rawB", action="store_true")
    ap.add_argument("--tune_h_center", action="store_true")
    ap.add_argument("--gap_width", type=float, default=0.4)
    ap.add_argument("--passes", type=int, default=3, help="coarse-to-fine passes")
    ap.add_argument("--grid", type=int, default=5, help="grid points per param per pass")
    ap.add_argument("--step0", type=float, default=0.35, help="initial multiplicative step (e.g., 0.35 ~ ±35%)")
    ap.add_argument("--shrink", type=float, default=0.5, help="step shrink factor each pass")
    ap.add_argument("--n_starts", type=int, default=6)
    ap.add_argument("--max_nfev", type=int, default=200)
    ap.add_argument("--max_evals", type=int, default=200, help="early stop eval budget (in eval-subjects)")
    ap.add_argument("--verbose", type=int, default=1, help="0=minimal, 1=default, 2=chatty per-eval")
    ap.add_argument("--jsonl_log", default="", help="optional path to write streaming JSONL progress")
    args = ap.parse_args()

    jsonl_fp = open(args.jsonl_log, "a", encoding="utf-8") if args.jsonl_log else None
    start = now_s()

    use_logB = (not args.rawB)
    base = json.loads(Path(args.init_globals).read_text())
    subjects = load_subjects(args.csv, args.min_obs_b, args.subjects)
    if not subjects:
        raise SystemExit("No subjects eligible (check --min_obs_b / --subjects).")

    # Current globals
    g = base
    evals = 0  # counts eval-subjects

    # Optionally represent H center instead of separate on/off
    if args.tune_h_center:
        H_on  = g["fixed"].get("H_on", 0.3)
        H_off = g["fixed"].get("H_off", 0.7)
        center = 0.5*(H_on + H_off)
        center = float(np.clip(center, 0.05, 0.95))
    else:
        center = None

    # Evaluate baseline cost
    if args.verbose:
        print(f"▶ Baseline evaluation on {len(subjects)} subjects...", flush=True)

    def baseline_cb(sid, cost, i, n):
        nonlocal evals
        evals += 1
        if args.verbose >= 2:
            print(f"    · [{i:02d}/{n}] sid={sid}  cost={cost:.4f}", flush=True)
        log_jsonl(jsonl_fp, {
            "phase": "baseline",
            "subject": sid,
            "cost": cost,
            "evals": evals,
            "t": time.time()
        })

    best_cost = eval_cost_for_globals(
        g, subjects, args.csv, args.n_starts, args.max_nfev, args.logbase, use_logB,
        per_subject_cb=baseline_cb
    )
    print(f"[pass 0] baseline cost = {best_cost:.6f}", flush=True)

    # Build tune list
    tune_list = list(TUNE_KEYS)
    if args.tune_h_center:
        tune_list.append("H_center")

    # Rough estimate of total work (for the bar) — upper bound; early stop may end sooner.
    est_total_evals = evals + args.passes * len(tune_list) * args.grid * len(subjects)
    if args.max_evals:
        est_total_evals = min(est_total_evals, args.max_evals)

    def update_bar():
        done = min(evals, est_total_evals)
        bar = progress_bar(done, est_total_evals)
        elapsed = now_s() - start
        rate = done / max(elapsed, 1e-9)
        remain = (est_total_evals - done) / max(rate, 1e-12)
        sys.stdout.write("\r" + bar + f"  evals={done}/{est_total_evals}  elapsed={fmt_hms(elapsed)}  ETA={fmt_hms(remain)}")
        sys.stdout.flush()

    step = args.step0
    update_bar()

    for p in range(1, args.passes+1):
        print(f"\n\n=== PASS {p} (step ~ ±{int(step*100)}%) ===", flush=True)

        for key in tune_list:
            # current value and bounds
            if key == "H_center":
                cur = center
                lo, hi = 0.05, 0.95
            else:
                cur = float(g["fixed"].get(key))
                lo, hi = BOUNDS[key]

            # grid around multiplicative neighbourhood (for H_center use additive)
            if key == "H_center":
                half = step * 0.5  # keep small on "center"
                vals = np.linspace(cur - half, cur + half, args.grid)
            else:
                factors = np.linspace(1.0 - step, 1.0 + step, args.grid)
                vals = factors * cur

            # clamp to bounds
            vals = [clamp(float(v), lo, hi) for v in vals]
            vals = sorted(set(vals))

            best_local_val = cur
            best_local_cost = best_cost

            if args.verbose:
                rng_str = f"[{vals[0]:.4g} … {vals[-1]:.4g}]"
                print(f"  → Tuning {key}: cur={cur:.4g}  grid={len(vals)}  range={rng_str}", flush=True)

            for j, v in enumerate(vals, 1):
                g_try = json.loads(json.dumps(g))  # deep copy
                if key == "H_center":
                    c = float(v)
                    H_on  = clamp(c - args.gap_width/2, 0.01, 0.99)
                    H_off = clamp(c + args.gap_width/2, 0.01, 0.99)
                    if H_on >= H_off:
                        H_on = max(0.01, H_off - 1e-3)
                    g_try["fixed"]["H_on"]  = H_on
                    g_try["fixed"]["H_off"] = H_off
                else:
                    g_try["fixed"][key] = float(v)

                # per-eval subject callback for progress
                trial_cost_accum = {"sum": 0.0}
                def eval_cb(sid, cst, i_subj, n_subj):
                    nonlocal evals
                    evals += 1
                    trial_cost_accum["sum"] += cst
                    if args.verbose >= 2:
                        print(f"      · [{i_subj:02d}/{n_subj}] sid={sid} val={v:.4g} cost={cst:.4f}", flush=True)
                    update_bar()
                    log_jsonl(jsonl_fp, {
                        "phase": "grid",
                        "pass": p,
                        "param": key,
                        "value": v,
                        "subject": sid,
                        "subject_idx": i_subj,
                        "n_subjects": n_subj,
                        "partial_cost_sum": trial_cost_accum["sum"],
                        "evals": evals,
                        "t": time.time()
                    })

                cost = eval_cost_for_globals(
                    g_try, subjects, args.csv,
                    n_starts=args.n_starts, max_nfev=args.max_nfev,
                    logbase=args.logbase, use_logB=use_logB,
                    per_subject_cb=eval_cb
                )

                if cost < best_local_cost - 1e-6:
                    best_local_cost = cost
                    best_local_val = v

                if args.max_evals and evals >= args.max_evals:
                    # early stop: write best so far
                    if key == "H_center":
                        center = best_local_val
                        H_on  = clamp(center - args.gap_width/2, 0.01, 0.99)
                        H_off = clamp(center + args.gap_width/2, 0.01, 0.99)
                        if H_on >= H_off:
                            H_on = max(0.01, H_off - 1e-3)
                        g["fixed"]["H_on"]  = H_on
                        g["fixed"]["H_off"] = H_off
                    else:
                        g["fixed"][key] = float(best_local_val)
                    Path(args.out_json).write_text(json.dumps(g, indent=2))
                    elapsed = now_s() - start
                    print(f"\n[early stop @ {evals} eval-subjects | {fmt_hms(elapsed)}] "
                          f"cost={best_local_cost:.6f} saved to {args.out_json}", flush=True)
                    log_jsonl(jsonl_fp, {
                        "event": "early_stop",
                        "evals": evals,
                        "best_cost": best_local_cost,
                        "out_json": args.out_json,
                        "t": time.time()
                    })
                    if jsonl_fp: jsonl_fp.close()
                    return

            # commit best for this coordinate
            if key == "H_center":
                center = best_local_val
                H_on  = clamp(center - args.gap_width/2, 0.01, 0.99)
                H_off = clamp(center + args.gap_width/2, 0.01, 0.99)
                if H_on >= H_off:
                    H_on = max(0.01, H_off - 1e-3)
                g["fixed"]["H_on"]  = H_on
                g["fixed"]["H_off"] = H_off
            else:
                g["fixed"][key] = float(best_local_val)

            best_cost = best_local_cost
            elapsed = now_s() - start
            print(f"  ✓ tuned {key}: new={best_local_val:.4g}  cost={best_cost:.6f}  elapsed={fmt_hms(elapsed)}", flush=True)
            log_jsonl(jsonl_fp, {
                "event": "param_commit",
                "pass": p,
                "param": key,
                "value": best_local_val,
                "best_cost": best_cost,
                "evals": evals,
                "t": time.time()
            })

        step *= args.shrink

    Path(args.out_json).write_text(json.dumps(g, indent=2))
    elapsed = now_s() - start
    print(f"\nDone. Total eval-subjects ~ {evals}. Tuned globals saved to {args.out_json}  ({fmt_hms(elapsed)})", flush=True)
    print(json.dumps({"tuned_globals": g["fixed"], "final_cost": best_cost}, indent=2))
    log_jsonl(jsonl_fp, {
        "event": "done",
        "evals": evals,
        "final_cost": best_cost,
        "out_json": args.out_json,
        "elapsed_s": elapsed,
        "t": time.time()
    })
    if jsonl_fp: jsonl_fp.close()

if __name__ == "__main__":
    # Ensure immediate flush on some environments
    try:
        import os
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
    except Exception:
        pass
    main()
