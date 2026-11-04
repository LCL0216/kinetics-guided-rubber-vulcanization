# Kinetics Guided Prediction of Rubber Vulcanization Curves

Research code for **Kinetics-Guided Machine Learning for Interpretable Prediction of Rubber Vulcanization Curves**.

[中文说明](README.zh-CN.md)

The workflow learns five interpretable curve parameters from seven process features, then reconstructs a continuous torque–time response with an induction–curing representation. It contains characteristic extraction, kinetic fitting, regressor comparison, curve reconstruction, ablations, interpretation, direct-model comparisons, independent validation, and manuscript/SI plotting.

## Quick start

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_pipeline.py --stages 09 --no-plots
```

This verifies and unpacks the supplied scientific inputs into `work/`, loads models already fitted on the 88 modeling curves, and predicts the six independent curves. It does **not** retrain the models or use independent measurements as training targets. Predictions and metrics appear in `work/Stage9_一键预测/Stage9_Output/`.

To generate the six-curve plots from these predictions:

```bash
python run_pipeline.py --stages 09
```

## Reproduce analyses

```bash
# Extract characteristics, fit kinetics and build the 88-curve table
python run_pipeline.py --stages 01 02 03 --no-plots

# All manuscript analyses and figure scripts, in dependency order
python run_pipeline.py --all

# Inspect the full execution plan without running it
python run_pipeline.py --all --dry-run

# Verify input/model hashes and frozen-model prediction against saved metrics
python checks/verify_release.py
```

The full pipeline includes repeated/nested cross-validation, permutation importance, and many high-resolution figures; it takes substantially longer than the six-curve inference example. Reduced folds/repeats in individual script options are for debugging and do not replace the paper's evaluation. Separate workspaces can be selected with `--work-dir work-new`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `scripts/` | 17 manuscript analysis and plotting scripts with descriptive English names |
| `exploratory/` | Two retained exploratory neural-network/amplitude-target analyses |
| `data/` | Six input tables, frozen models, and data provenance |
| `reference/` | 23 original numerical result tables for comparison |
| `prepare_workspace.py` | Hash verification and isolated input unpacking |
| `run_pipeline.py` | Stage selection and ordered execution |
| `checks/` | Release checks and saved verification results |
| `docs/` | Original-to-release filename mapping and code provenance |

Stage numbers retain the original analysis order. Output directory names retain the scientific scripts' existing conventions, including the Stage 09 subdirectory, so saved-source relationships remain clear. See [stage mapping](docs/stage_mapping.csv) and [input documentation](data/README.md).

## Scientific scope and reproducibility

- The 88 modeling curves and six independent validation curves are separate datasets. Their curve numbers are local to each dataset.
- The learned targets are `ML`, `MH`, `ti`, `k1`, and `n1`. `k1` is learned in log10 space. Forward prediction uses an induction-boundary value estimated from training data, rather than the held-out curve's observed value.
- Main curve comparisons use curve-level five-fold splits. Measured-amplitude comparisons are diagnostics and must not be reported as forward prediction.
- The dataset-building step identifies 33 distinct time–torque curve groups among 88 modeling records; 67 records belong to groups with repeated curves. The archived Stage 04/08 splits are based on curve IDs, rather than these groups. Those archived scores therefore do not establish generalization to entirely unseen curve groups. The original warning and results are preserved; a group-based re-evaluation is separate scientific work.
- The fixed-exponent comparison refits the rate constant with `n1=1`; it does not merely replace the predicted exponent.
- Saved reference metrics are provided for comparison. Software-version differences may affect refitting and attribution results. Scripts retain their existing numerical consistency checks.
- The original scientific functions, model settings, and dataset contents were preserved. Release changes concern filenames, local imports, help text, portable cache paths, and plotting workspace selection; these are recorded in `docs/code_provenance.json`.
- The frozen-model manifest records its original serialization environment. The pinned requirements describe the environment checked for this code release. Full cross-validation and all figure regeneration were not rerun during release preparation; the verification report records the checks actually completed.

Manuscript DOCX files, citation-manager databases, personal workstation paths, historical backups, generated image collections, and local caches are outside this repository.
