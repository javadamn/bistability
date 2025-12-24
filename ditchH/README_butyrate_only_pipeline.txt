
Butyrate-only pipeline (no H proxies) — Quick Start
==================================================

1) Create metabolite-derived inputs (F_met, A_met):
   python preprocess_add_met_inputs.py        --combined /mnt/data/combined_scfas_table_raw.csv        --metabolomics /mnt/data/metabolomics_samples.csv        --out_csv /mnt/data/combined_scfas_table_scored_plus_met.csv

   Notes:
   - F_met = PC1 over {lactate, succinate, propionate} (per subject, z-scored, min-max to [0,1]).
   - A_met = PC1 over {I3P, LCA/Cholate, 1/Glycocholate} (per subject, rectified, min-max to [0,1]).
   - Adds aux_propionate_z, aux_succinate_z, aux_lactate_z (diagnostic only).

2) Fit butyrate-only discrete model with global clearance:
   python fit_butyrate_only_aux.py        --csv /mnt/data/combined_scfas_table_scored_plus_met.csv        --outdir /mnt/data/outputs_b_fit        --lambda_grid 0.05,0.1,0.15,0.2,0.25,0.3        --min_obs 5

   Outputs in /mnt/data/outputs_b_fit:
     - params.csv        (per-subject p0, alpha_F, alpha_A, lambda, n_obs)
     - predictions.csv   (row-level B_obs, B_hat, F_met, A_met)
     - fit_summary.csv   (per-subject R2, n, time_span_days)
     - global_fit.json   (grid search of lambda and best choice)
     - log.txt

To integrate with your existing ODE fits later:
- Use the augmented CSV as input and point the scripts to columns F_met and A_met for drivers.
- Disable H coupling (e.g., set g=0, u=0, s_B=1) and pool a single clearance parameter.
- Once fit quality is acceptable, re-enable hysteresis with a fixed gap and only tune the center.
