# Kinetics-Guided Prediction of Rubber Vulcanization Curves

Core code and data for *Kinetics-Guided Machine Learning for Interpretable Prediction of Rubber Vulcanization Curves*.

## Data

- `model_ready_data.csv.gz`: 88 modeling curves (IDs 1–88), including the seven process inputs, `Time`, and `torque`. This is the original CSV compressed without numerical changes; pandas reads it directly.
- `independent_validation_process_6.csv`: process conditions for six independent curves (IDs 1–6).
- `independent_validation_curves_6.csv`: measured time–torque observations for those six curves.

The process inputs are `mixing_time`, `mixing_temp`, `curing_temp`, `curing_pressure`, `rotor_speed_1`, `rotor_speed_2`, and `curing_time_offset`. `Time` is in seconds and torque in kgf·cm; process variables retain the source-record units described in the manuscript. Training and validation IDs are local to separate datasets. Validation measurements are used for evaluation, not training.

## Environment

Tested with Python 3.12. Install the scientific dependencies:

```bash
python -m pip install numpy==2.4.2 pandas==3.0.1 scipy==1.17.1 matplotlib==3.10.8 scikit-learn==1.8.0 xgboost==3.2.0 shap==0.50.0 joblib==1.5.3
```

## Run

Run commands from the repository root. Each stage creates its own output directory.

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
python Stage9.py --project-root . --mode fit
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
| 9 | Fit final models on all 88 curves, then predict the six independent curves |
| 10 | Free-exponent versus fixed-exponent kinetics, with the fixed model's rate refitted |

Stage9 `fit` generates and freezes the five models locally. Subsequent `predict` runs load them without retraining; `fit` refuses to replace existing models unless `--overwrite-models` is explicitly supplied. Stage10 requires the outputs of Stages1–3 and Stage8. Saved models, intermediate results, and image collections are regenerated locally and are not included here. Remove `--no-plots` from Stages1, 2, 5, or 5B when their individual figures are needed. Full cross-validation and interpretation runs can take substantially longer than six-curve inference; reduced folds/repeats are debugging options only.

The original frozen six-curve models recorded Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0, and XGBoost 3.1.3. This minimal repository refits Stage9 instead of distributing those frozen models. Fresh training in the tested Python 3.12 environment gives different numerical predictions from the archived frozen models; exact reproduction of their reported scores is not established by this release.

The original analysis is retained. Stage3 identifies 33 distinct time–torque curve groups among the 88 modeling records; 67 records belong to repeated groups. The archived Stage4/8 evaluation splits by curve ID, rather than `curve_group_id`, so its scores do not establish generalization to completely unseen curve groups.
