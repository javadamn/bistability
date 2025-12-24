
Multistability Wrap-up: Quick Commands
======================================

1) Select cohort and summary plots
----------------------------------
python /mnt/data/scripts_bundle/analyze_v4_and_select_cohort.py   --fit_v4 /mnt/data/fit_summary_v4_cf.csv   --outdir /mnt/data/wrapup_outputs/cohort   --r2_thresh 0.40

# Optional (if available)
#  --fit_v3 /mnt/data/fit_summary_v3.csv #  --global_json /mnt/data/outputs_b_fit_v4_cf/global_fit_v4_cf.json

2) Subject panels
-----------------
python /mnt/data/scripts_bundle/plot_timeseries_panels.py   --pred_v4 /mnt/data/predictions_v4_cf.csv   --subjects_csv /mnt/data/wrapup_outputs/cohort/cohort_subjects.csv   --max_plots 6   --outdir /mnt/data/wrapup_outputs/panels

3) Multistability analysis for one subject
------------------------------------------
python /mnt/data/scripts_bundle/multistability_analysis.py   --params_csv /mnt/data/outputs_b_fit_v4_cf/params_v4_cf.csv   --pred_v4    /mnt/data/predictions_v4_cf.csv   --global_json /mnt/data/outputs_b_fit_v4_cf/global_fit_v4_cf.json   --subject <SUBJECT_ID>   --center 0.5 --gap 0.4 --h_amp 0.2   --outdir /mnt/data/wrapup_outputs/multistability

(If you don't have outputs_b_fit_v4_cf, but have those files in /mnt/data/, just adjust the paths.)
