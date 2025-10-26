
# 5-variable smooth model: data → calibration → multistability

## Files
- `scfa_model.py`: model ODEs, simulation, observation mapping.
- `prepare_inputs.py`: loads `modeling_table_with_indicators.csv` and prepares per-subject time series (`t_obs`, `y_B`, `y_H`, inputs `F`, `A`).
- `fit_subject.py`: least-squares fit for one subject (multi-parameter subset), writes `outputs/fit_{subject}.json`.
- `analyze_multistability.py`: basin mapping from random initial conditions; hysteresis sweep by varying fiber `F`.

## Usage

1. **Have inputs ready**  
   Place your merged data at `modeling_table_with_indicators.csv` (as built earlier).

2. **Fit a subject**  
```bash
python fit_subject.py --subject C3001 --csv modeling_table_with_indicators.csv --outdir outputs
```

3. **Check multistability**  
```bash
python analyze_multistability.py --subject C3001 --csv modeling_table_with_indicators.csv --theta_json outputs/fit_C3001.json
```

This will print the number of endpoint clusters (approximate attractor count) and you can add plotting as needed.
