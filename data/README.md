# Scientific inputs

`source_manifest.json` records the original relative filename, row count, columns, curve count, and SHA-256 of every input. `training_curves.csv.gz` contains the original training CSV compressed without numerical changes. The three training parameter tables are preserved as originally saved.

Training consists of 88 curves numbered 1–88. Independent validation consists of six separately numbered curves 1–6. Their overlapping IDs do not indicate shared observations. Validation measurements are used to evaluate predictions and are never training targets.

The seven process features are `mixing_time`, `mixing_temp`, `curing_temp`, `curing_pressure`, `rotor_speed_1`, `rotor_speed_2`, and `curing_time_offset`. Curve observations use `No`, `Time`, and `torque`. Follow the manuscript for measurement units and experimental definitions. `ML`, `MH`, `ti`, `k1`, and `n1` are the five learned targets; `k1` is learned in log10 space.

`frozen_models.zip` preserves the five models trained on all 88 modeling curves and their original model manifest. That manifest records the original serialization environment. The RF files require a compatible scikit-learn installation; the release requirements use scikit-learn 1.8.0. Stage 09 prediction loads these models and does not refit them.

Run `python prepare_workspace.py` to verify and unpack inputs into the ignored `work/` directory. CSVs in `reference/` are saved numerical results for comparison; they are not substituted for newly computed results. New intermediate tables, trained models and figures stay in the selected work directory.
