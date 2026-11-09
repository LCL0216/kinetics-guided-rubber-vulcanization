# Kinetics-Guided Machine Learning for Polymer Vulcanization

Core code, experimental curve data, and original frozen models for *Kinetics-Guided Machine Learning for Interpretable Prediction of Polymer Vulcanization*.

## Data and frozen models

- `model_ready_data.csv.gz`: 88 modeling curves (IDs 1–88), including the seven process inputs, `Time`, and `torque`. This is the original CSV compressed without numerical changes; pandas reads it directly.
- `independent_validation_process_6.csv`: process conditions for six independent curves (IDs 1–6).
- `independent_validation_curves_6.csv`: measured time–torque observations for those six curves.
- `Stage9_Models_88.zip`: the original five frozen models trained on all 88 curves, plus their original model manifest. Stage9 automatically restores them into `Stage9_Models_88/` on first use.

The process inputs are `mixing_time`, `mixing_temp`, `curing_temp`, `curing_pressure`, `rotor_speed_1`, `rotor_speed_2`, and `curing_time_offset`. `Time` is in seconds and torque in kgf·cm; process variables retain the source-record units described in the manuscript. Training and validation IDs are local to separate datasets. Validation measurements are used for evaluation, not training.

## Environment

Tested with Python 3.12. Install the scientific dependencies:

```bash
python -m pip install numpy==2.4.2 pandas==3.0.1 scipy==1.17.1 matplotlib==3.10.8 scikit-learn==1.8.0 xgboost==3.2.0 shap==0.50.0 joblib==1.5.3
```

## Run

Run commands from the repository root. To reproduce the six-curve predictions using the supplied frozen models, run:

```bash
python Stage9.py --project-root . --mode predict
```

This requires no Stage1–8 outputs and performs no training. The following commands run the analyses in dependency order; each stage creates its own output directory.

```bash
python Stage1.py --project-root . --no-plots
python Stage2.py --project-root . --no-plots
python Stage3.py --project-root .
python Stage4B.py --project-root .
python Stage5.py --project-root . --no-plots
python Stage6.py --project-root .
python Stage7A.py --project-root .
python Stage7B.py --project-root .
python Stage7C.py --project-root .
python Stage8.py --project-root .
python Stage5B_alpha0_ablation.py --project-root . --no-plots
python Stage9.py --project-root . --mode predict
python Stage10.py --project-root .
```

| Stage | Analysis |
| --- | --- |
| 1–3 | Characteristic extraction, kinetic fitting, and curve-level dataset construction |
| 4B | SVR, RF, XGBoost, and MLP parameter-regressor comparison |
| 5 / 5B | Curve reconstruction / induction-boundary ablation; run 5B after Stage8 |
| 6 | Assisted prediction of maximum torque |
| 7A / 7B / 7C | Correlations / feature ablation / SHAP and permutation importance |
| 8 | Direct-MLP, Direct-XGB, and PG-ML curve comparisons |
| 9 | Load the original frozen final models and predict the six independent curves |
| 10 | Free-exponent versus fixed-exponent kinetics, with the fixed model's rate refitted |

Stage9 `predict` is the default and loads the supplied frozen models without fitting or reading the 88-curve training table. Its outputs reproduce the original saved six-curve results in the tested environment: mean R² 0.955570, RMSE 1.856420 and MAE 1.571318 (kgf·cm for errors). For an optional new training run, select a separate directory, for example `python Stage9.py --project-root . --mode fit --model-dir retrained_models`; refitting may produce different numerical predictions and should not replace the frozen models used for the paper.

The frozen model manifest records Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0 and XGBoost 3.1.3. Loading the original models was also verified in the Python 3.12 environment above. Stage10 requires outputs from Stages1–3 and Stage8. Intermediate tables and figures are generated locally; remove `--no-plots` where those individual plots are needed. Full cross-validation and interpretation take substantially longer than six-curve inference; reduced folds/repeats are debugging options only.

The original analysis is retained. Stage3 identifies 33 distinct time–torque curve groups among the 88 modeling records; 67 records belong to repeated groups. The archived Stage4/8 evaluation splits by curve ID, rather than `curve_group_id`, so its scores do not establish generalization to completely unseen curve groups.
