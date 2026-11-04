#!/usr/bin/env python3
"""Controlled n1 ablation using only modeling curves 1--88.

Run with the existing scientific environment, for example:
  python Stage10.py --project-root .

The restricted model fixes n1=1 and REFITS its rate constant from each
training target curve. It is not obtained by replacing predicted n1 with 1.
Stage8 fold membership, process inputs, ML/MH/ti predictions, alpha0 medians,
XGBoost settings, evaluation times and curve-level aggregation are retained.
No independent-validation curves are opened. Existing outputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon
import sklearn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost
from xgboost import XGBRegressor

PROCESS = ["mixing_time", "mixing_temp", "curing_temp", "curing_pressure",
           "rotor_speed_1", "rotor_speed_2", "curing_time_offset"]
SETTINGS = dict(objective="reg:squarederror", n_estimators=300, max_depth=3,
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=2, reg_alpha=0.01, reg_lambda=1.0,
                random_state=42, n_jobs=-1)
FREE, FIXED = "Free_n1", "Fixed_n1_1_refit_k"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    return dict(R2=float(r2_score(y, p)),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)))


def kinetic(t, k, n, ti, a0):
    dt = np.maximum(np.asarray(t, float) - ti, 0)
    return a0 + (1 - a0) * (1 - np.exp(-k * dt ** n))


def alpha(t, ti, k, n, a0):
    # The same safeguards and induction branch as Stage8.curing_alpha.
    ti, k, n = max(float(ti), 1e-12), max(float(k), 1e-12), max(float(n), .1)
    a0 = float(np.clip(a0, 0, 1))
    t = np.asarray(t, float)
    a = np.empty_like(t)
    early = t < ti
    a[early] = a0 * np.clip(t[early] / ti, 0, 1)
    a[~early] = kinetic(t[~early], k, n, ti, a0)
    return np.clip(a, 0, 1)


def paired(rows, keys, mode):
    subset = rows.loc[rows.Evaluation == mode]
    a = subset.loc[subset.Representation == FREE].set_index(keys).sort_index()
    b = subset.loc[subset.Representation == FIXED].set_index(keys).sort_index()
    assert a.index.equals(b.index)
    records = []
    for m in ["R2", "RMSE", "MAE"]:
        d = b[m].to_numpy() - a[m].to_numpy()
        nonzero = d != 0
        p = float(wilcoxon(d, alternative="two-sided", method="auto").pvalue) if nonzero.any() else 1.
        records.append(dict(Evaluation=mode, Metric=m, N=len(d),
                            Mean_fixed_minus_free=float(d.mean()),
                            Median_fixed_minus_free=float(np.median(d)),
                            Fixed_better=int((d > 0 if m == "R2" else d < 0).sum()),
                            Free_better=int((d < 0 if m == "R2" else d > 0).sum()),
                            Ties=int((d == 0).sum()), Wilcoxon_two_sided_p=p))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir or root / "Stage10_n1_Ablation_Output"
    out.mkdir(parents=True, exist_ok=True)
    sources = dict(raw=root / "model_ready_data.csv.gz",
                   stage1=root / "Stage1_Output/kinetic_parameters_stage1.csv",
                   stage2=root / "Stage2_Output/kinetic_parameters_stage2.csv",
                   dataset=root / "Stage3_Output/pgmlp_curve_dataset.csv",
                   membership=root / "Stage8_Output/fold_membership.csv",
                   saved_parameters=root / "Stage8_Output/pg_parameter_predictions.csv",
                   saved_pointwise=root / "Stage8_Output/pointwise_predictions.csv",
                   saved_metrics=root / "Stage8_Output/curve_metrics_all_methods.csv")
    initial_hashes = {key: sha(path) for key, path in sources.items()}
    raw, s1, s2, dataset, members, params, points, old_metrics = [
        pd.read_csv(sources[k]) for k in sources]
    ids = set(range(1, 89))
    for name, frame in [("stage1", s1), ("stage2", s2), ("dataset", dataset), ("parameters", params)]:
        assert len(frame) == 88 and set(frame.No) == ids and frame.No.is_unique, name
    assert set(raw.No) == ids
    s1, s2, dataset = [f.set_index("No").sort_index() for f in (s1, s2, dataset)]
    assert np.allclose(s1[["ML", "MH", "ti", "tp", "alpha0"]],
                       dataset[["ML", "MH", "ti", "tp", "alpha0"]])
    fit_records, fixed_rates, free_refit_error = [], {}, []
    for no, row in s1.iterrows():
        g = raw.loc[raw.No == no, ["Time", "torque"]].replace([np.inf, -np.inf], np.nan).dropna()
        g = g.groupby("Time", as_index=False).torque.mean().sort_values("Time")
        t, torque = g.Time.to_numpy(float), g.torque.to_numpy(float)
        observed = np.clip((torque - row.ML) / (row.MH - row.ML), 0, 1)
        mask = (t >= row.ti) & (t <= row.tp)
        tf, af = t[mask], observed[mask]
        assert len(tf) >= 10
        pfree, _ = curve_fit(lambda tt, k, n: kinetic(tt, k, n, row.ti, row.alpha0),
                            tf, af, p0=[5e-4, 1.2], bounds=([1e-8, .2], [1., 5.]), maxfev=50000)
        assert np.allclose(pfree, s2.loc[no, ["k1", "n1"]].to_numpy(float), rtol=1e-6, atol=1e-10), no
        free_refit_error.append(float(np.max(np.abs(pfree - s2.loc[no, ["k1", "n1"]].to_numpy(float)))))
        pfixed, _ = curve_fit(lambda tt, k: kinetic(tt, k, 1., row.ti, row.alpha0),
                             tf, af, p0=[5e-4], bounds=([1e-8], [1.]), maxfev=50000)
        kfixed = float(pfixed[0])
        fixed_rates[no] = kfixed
        for rep, k, n in [(FREE, float(s2.loc[no, "k1"]), float(s2.loc[no, "n1"])),
                          (FIXED, kfixed, 1.)]:
            fit_records.append(dict(No=int(no), Representation=rep, Evaluation="Kinetic_fit",
                                    k=k, n=n, Fit_points=len(tf),
                                    k_at_bound=bool(np.isclose(k, 1e-8, rtol=0, atol=1e-10) or
                                                    np.isclose(k, 1, rtol=0, atol=1e-6)),
                                    **metrics(af, kinetic(tf, k, n, row.ti, row.alpha0))))
    fits = pd.DataFrame(fit_records)
    assert (fits.pivot(index="No", columns="Representation", values="RMSE")[FREE] <=
            fits.pivot(index="No", columns="Representation", values="RMSE")[FIXED] + 1e-8).all()
    print("88 constrained fits completed; original free-exponent fits reproduced.", flush=True)
    rows, prediction_rows, point_frames = [], [], []
    refit_prediction_errors, original_curve_errors = [], []
    for fold in sorted(members.Fold.unique()):
        train_ids = members.loc[(members.Fold == fold) & (members.Subset == "Train"), "No"].astype(int).tolist()
        test_ids = members.loc[(members.Fold == fold) & (members.Subset == "Test"), "No"].astype(int).tolist()
        assert set(train_ids).isdisjoint(test_ids) and set(train_ids) | set(test_ids) == ids
        # Preserve dataset row ordering used in Stage8.
        tr, te = dataset.loc[dataset.index.isin(train_ids)], dataset.loc[dataset.index.isin(test_ids)]
        xtr, xte = tr[PROCESS].to_numpy(float), te[PROCESS].to_numpy(float)
        saved = params.loc[params.Fold == fold].set_index("No").loc[te.index]
        a0 = float(np.median(tr.alpha0))
        assert np.allclose(a0, saved.alpha0_train_median, atol=1e-14)
        for target in ["k1", "n1"]:
            reg = XGBRegressor(**SETTINGS)
            target_values = tr[target].to_numpy(float)
            reg.fit(xtr, np.log10(target_values) if target == "k1" else target_values)
            pred = reg.predict(xte)
            if target == "k1": pred = 10 ** pred
            original = saved[target + "_pred"].to_numpy(float)
            assert np.allclose(pred, original, rtol=1e-6, atol=1e-10), (fold, target, np.max(abs(pred-original)))
            refit_prediction_errors.append(float(np.max(abs(pred - original))))
        reg_fixed = XGBRegressor(**SETTINGS)
        reg_fixed.fit(xtr, np.log10(np.array([fixed_rates[no] for no in tr.index])))
        fixed_pred = 10 ** reg_fixed.predict(xte)
        for j, no in enumerate(te.index):
            p = saved.loc[no]
            pts = points.loc[(points.Fold == fold) & (points.No == no)].sort_values("Time")
            t, observed = pts.Time.to_numpy(float), pts.Experimental.to_numpy(float)
            assert len(t) > 1 and np.all(np.diff(t) > 0)
            for rep, k, n in [(FREE, p.k1_pred, p.n1_pred), (FIXED, float(fixed_pred[j]), 1.)]:
                shape = alpha(t, p.ti_pred, k, n, a0)
                for evaluation, ml, mh in [("Full_forward", p.ML_pred, p.MH_pred),
                                            ("Measured_amplitude_diagnostic", p.ML_true, p.MH_true)]:
                    mh = max(mh, ml + 1e-6)
                    torque = ml + (mh - ml) * shape
                    vals = metrics(observed, torque)
                    rows.append(dict(Fold=int(fold), No=int(no), Representation=rep,
                                     Evaluation=evaluation, N_points=len(t), **vals))
                    if evaluation == "Full_forward":
                        point_frames.append(pd.DataFrame(dict(Fold=fold, No=no, Time=t,
                            Experimental=observed, Representation=rep, Predicted=torque)))
                        if rep == FREE:
                            err = float(np.max(abs(torque - pts.PG_ML.to_numpy(float))))
                            original_curve_errors.append(err)
                            assert err < 1e-9
                            saved_m = old_metrics.loc[(old_metrics.Method == "PG-ML") & (old_metrics.No == no)].iloc[0]
                            assert all(np.isclose(vals[key], saved_m[key], rtol=1e-6, atol=2e-7) for key in vals)
                prediction_rows.append(dict(Fold=int(fold), No=int(no), Representation=rep,
                    ML_pred=p.ML_pred, MH_pred=p.MH_pred, ti_pred=p.ti_pred,
                    k_pred=k, n_pred=n, alpha0_train_median=a0,
                    k_true=float(dataset.loc[no, "k1"]) if rep == FREE else fixed_rates[no]))
        print(f"Fold {fold}/5 completed; saved free-exponent predictions reproduced.", flush=True)
    result = pd.DataFrame(rows)
    summaries = []
    for evaluation, frame in [("Kinetic_fit", fits), ("Full_forward", result),
                               ("Measured_amplitude_diagnostic", result)]:
        frame = frame.loc[frame.Evaluation == evaluation]
        for rep, g in frame.groupby("Representation", sort=False):
            summaries.append(dict(Evaluation=evaluation, Representation=rep, N_curves=len(g),
                **{f"{agg}_{m}": float(getattr(g[m], agg)())
                   for agg in ["mean", "median", "std"] for m in ["R2", "RMSE", "MAE"]}))
    summary = pd.DataFrame(summaries)
    pair_records = paired(fits, ["No"], "Kinetic_fit")
    for mode in ["Full_forward", "Measured_amplitude_diagnostic"]:
        pair_records.extend(paired(result, ["Fold", "No"], mode))
    pairs = pd.DataFrame(pair_records)
    for name, frame in [("kinetic_fit_by_curve.csv", fits), ("curve_metrics.csv", result),
                        ("parameter_predictions.csv", pd.DataFrame(prediction_rows)),
                        ("summary.csv", summary), ("paired_comparisons.csv", pairs),
                        ("fold_membership.csv", members),
                        ("pointwise_forward_predictions.csv", pd.concat(point_frames, ignore_index=True))]:
        frame.to_csv(out / name, index=False, encoding="utf-8-sig")
    # A single figure separates fit flexibility, forward prediction and a
    # diagnostic holding measured amplitudes common between representations.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), layout="constrained")
    order = [FREE, FIXED]
    a = fits.pivot(index="No", columns="Representation", values="RMSE")
    for no, r in a.iterrows(): axes[0].plot([0, 1], r[order], color="#8798A2", alpha=.18, lw=.6)
    axes[0].boxplot([a[c] for c in order], positions=[0, 1], widths=.4, showfliers=False,
                    patch_artist=True, boxprops={"facecolor": "#E6EDF2", "edgecolor": "#587787"},
                    medianprops={"color": "#193D50", "lw": 1.5})
    axes[0].scatter([0, 1], [a[c].mean() for c in order], marker="D", color="#D3854A", s=28, zorder=4, label="Mean")
    axes[0].set(xticks=[0, 1], xticklabels=["Free n1", "n1 = 1\nrefitted rate"], ylabel="Fit RMSE (dimensionless)", title="(a) Observed-curve fit")
    axes[0].legend(frameon=False, fontsize=8)
    for ax, mode, title in [(axes[1], "Full_forward", "(b) Forward prediction"),
                            (axes[2], "Measured_amplitude_diagnostic", "(c) Amplitude diagnostic")]:
        a = result.loc[result.Evaluation == mode].pivot(index="No", columns="Representation", values="RMSE")
        lim = float(a.max().max()) * 1.06
        ax.plot([0, lim], [0, lim], ls="--", lw=1, color="#8D969C")
        ax.scatter(a[FREE], a[FIXED], s=18, color="#387D91", alpha=.65, edgecolors="none")
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Free n1 RMSE (kgf·cm)",
               ylabel="n1 = 1 RMSE (kgf·cm)", title=title)
        ax.text(.04, .96, "Above line: free n1 better", transform=ax.transAxes,
                va="top", fontsize=6.7)
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"Figure_n1_representation_comparison.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)
    assert {key: sha(path) for key, path in sources.items()} == initial_hashes
    manifest = dict(analysis="Controlled free n1 versus n1=1 with refitted rate", date="2026-10-29",
        python=sys.executable, python_version=platform.python_version(),
        versions={m.__name__: m.__version__ for m in [np, pd, scipy, sklearn, xgboost, matplotlib]},
        inputs={k: {"path": str(v), "sha256": initial_hashes[k]} for k, v in sources.items()},
        script_sha256=sha(__file__), model_settings=SETTINGS,
        fitted_curves=88, completed_folds=5, n_forward_points=int(sum(len(g) for g in point_frames[::2])),
        modeling_curve_ids="1-88", additional_six_curves_read=False,
        fixed_model="alpha0+(1-alpha0)*(1-exp(-k_fixed*(t-ti))); n1=1; k_fixed has units s^-1",
        fixed_rate_fit=dict(initial=.0005, lower=1e-8, upper=1., maxfev=50000,
                            interval="ti <= t <= tp", normalized_response="clip((M-ML)/(MH-ML),0,1)"),
        shared_components=["saved Stage8 fold membership", "seven process inputs", "ML/MH/ti fold predictions",
                           "training-fold median alpha0", "original test times", "XGB settings and log10(rate) training"],
        max_free_parameter_refit_abs_error=max(free_refit_error),
        max_free_OOF_parameter_retrain_abs_error=max(refit_prediction_errors),
        max_free_pointwise_reconstruction_abs_error=max(original_curve_errors),
        inference_note="Paired tests condition on one saved five-fold split. No equivalence margin is specified; nonsignificance is not equivalence.",
        diagnostic_note="Measured amplitudes use held-out curves and are diagnostic only, never forward-prediction scores.",
        scope="Controlled parameterization comparison under the retained XGB settings; no new algorithm search or tuning.",
        original_inputs_unchanged=True)
    # Each curve contributes two representations, hence divide total by two.
    manifest["n_forward_points"] = int(sum(len(g) for g in point_frames) // 2)
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pairs.to_string(index=False), flush=True)
    print(f"Saved to {out}", flush=True)


if __name__ == "__main__":
    main()
