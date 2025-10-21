# mw_model_constants.py
# Single source of truth for model evaluation.
# Make sure FIT_PATH points to the file produced by calibrate_guild_hard_targets.py
# that you used in the "Bistable = YES" run.

FIT_PATH  = "mw_fit_out_guild_hard_targets/fitted_global_params.csv"

# Curvature / memory settings that produced YES in your bifurcation script:
N_HILL    = 4
KQ        = 100

# If your "YES" run used a specific d override, put that float here; otherwise None.
D_OVERRIDE = None  # e.g., 0.9*baseline_d
