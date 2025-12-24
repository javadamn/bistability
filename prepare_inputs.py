
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

def load_subject_data(csv_path: str, subject_id: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    sub = df[df["subject_id"].astype(str) == str(subject_id)].copy().sort_values("date")
    if sub.empty:
        raise ValueError(f"No rows for subject_id={subject_id}")
    # Build time axis in days since first sample
    t = (sub["date"] - sub["date"].min()).dt.days.astype(float).values
    y_B = sub["butyrate"].astype(float).values
    y_H = sub["H_proxy"].astype(float).values if "H_proxy" in sub.columns else np.full_like(y_B, np.nan)
    # Inputs: F_fiber (fallback to F_fermented if missing), A_recent
    F = sub["F_fiber"].astype(float).fillna(sub.get("F_fermented", np.nan)).fillna(0.0).values
    A = sub["A_recent"].astype(float).fillna(0.0).values
    return {
        "t_obs": t,
        "y_B": y_B,
        "y_H": y_H,
        "F": F,
        "A": A,
        "rows": sub
    }
