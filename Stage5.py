# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
import xgboost
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

SEED = 42
N_SPLITS = 5
EXPECTED_NOS = set(range(1, 89))
X_COLS = ["mixing_time", "mixing_temp", "curing_temp", "curing_pressure",
          "rotor_speed_1", "rotor_speed_2", "curing_time_offset"]
TARGETS = ["ML", "MH", "ti", "k1", "n1"]


def locate_files(root_arg: str | None, curve_arg: str | None, stage3_arg: str | None):
    """Find training data; do not read the separately numbered 6 validation curves."""
    here = Path(__file__).resolve().parent
    roots = ([Path(root_arg).expanduser().resolve()] if root_arg else
             [here, here / "硫化曲线", here.parent / "硫化曲线",
              Path.cwd(), Path.cwd() / "硫化曲线"])
    suffixes = [Path("."), Path("建模集_1-88"), Path("建模集"),
                Path("Training_88"), Path("Modeling_88")]
    tried = []
    for root in roots:
        for suffix in suffixes:
            base = root / suffix
            curve = (Path(curve_arg).expanduser().resolve() if curve_arg else
                     base / "model_ready_data.csv.gz")
            candidates = ([Path(stage3_arg).expanduser().resolve()] if stage3_arg else
                          [base / "Stage3_Output" / "pgmlp_curve_dataset.csv",
                           base / "pgmlp_curve_dataset.csv"])
            for s3 in candidates:
                tried.append(f"  曲线: {curve}\n  Stage3: {s3}")
                if curve.is_file() and s3.is_file():
                    return base.resolve(), curve, s3
    raise FileNotFoundError(
        "无法找到建模集的两个输入CSV。请在 --project-root 指定『硫化曲线』实际路径，"
        "或用 --curve-input 和 --stage3-input 指定两个文件。\n"
        + "\n".join(tried[:18]))


def get_model(target: str):
    # Exactly the same hyperparameters as the user-provided Stage 5 / Stage 8.
    if target in ("ML", "MH"):
        return RandomForestRegressor(
            n_estimators=300, max_features="sqrt", min_samples_leaf=1,
            random_state=SEED, n_jobs=-1)
    return XGBRegressor(
        objective="reg:squarederror", n_estimators=300, max_depth=3,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=0.01, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1)


def alpha_curve(t, ti, k1, n1, alpha0):
    """The ORIGINAL model: empirical induction interpolation + curing kinetics."""
    t = np.asarray(t, dtype=float)
    ti = max(float(ti), 1e-8)
    k1 = max(float(k1), 1e-10)
    n1 = max(float(n1), 0.1)
    alpha0 = float(np.clip(alpha0, 0.0, 1.0))
    pre = alpha0 * np.clip(t / ti, 0.0, 1.0)
    dt = np.maximum(t - ti, 0.0)
    post = alpha0 + (1 - alpha0) * (1 - np.exp(-k1 * dt ** n1))
    return np.clip(np.where(t < ti, pre, post), 0.0, 1.0)


def curve_metrics(true, pred):
    return {"R2": float(r2_score(true, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
            "MAE": float(mean_absolute_error(true, pred))}


def plot_curve(outpath, no, t, truth, a, b, met_a, met_b, save_pdf):
    # Style and content are retained from the source Stage 5.
    fig, ax = plt.subplots(figsize=(8.1, 5.25))
    ax.plot(t, truth, color="#20252B", lw=3.2, label="Experimental", zorder=4)
    ax.plot(t, a, color="#2374AB", lw=2.6, ls="--", label="Version A", zorder=3)
    ax.plot(t, b, color="#D55641", lw=2.6, ls="-.", label="Version B", zorder=2)
    info = (f"A: $R^2$={met_a['R2']:.3f}, RMSE={met_a['RMSE']:.3f}\n"
            f"B: $R^2$={met_b['R2']:.3f}, RMSE={met_b['RMSE']:.3f}")
    ax.text(0.025, 0.975, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.4, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#B0B0B0", lw=0.65, alpha=0.82), zorder=10)
    ax.set_xlabel("Time (s)", fontsize=12, labelpad=7)
    ax.set_ylabel("Torque (kgf·cm)", fontsize=12, labelpad=7)
    ax.set_title(f"Curve {no}", fontsize=14, weight="semibold", pad=10)
    ax.tick_params(which="major", direction="out", labelsize=10.5, width=1.2, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    ax.grid(ls="--", lw=0.65, alpha=0.15)
    ax.set_axisbelow(True)
    ax.margins(x=0.012, y=0.055)
    legend = ax.legend(loc="lower right", fontsize=8.3, frameon=True,
                       fancybox=True, framealpha=0.86, borderpad=0.3,
                       labelspacing=0.3, handlelength=2.3, handletextpad=0.5)
    legend.get_frame().set_edgecolor("#C8C8C8")
    legend.get_frame().set_linewidth(0.65)
    fig.tight_layout(pad=1.2)
    fig.savefig(outpath, dpi=400, facecolor="white", bbox_inches="tight")
    if save_pdf:
        fig.savefig(outpath.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def normalize_no(df, source):
    values = pd.to_numeric(df["No"], errors="coerce")
    if values.isna().any() or not np.all(values % 1 == 0):
        raise ValueError(f"{source} 的 No 含缺失或非整数编号")
    df["No"] = values.astype(int)
    return df


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Stage5: 88 curves, RF300/XGB300, A/B reconstruction")
    parser.add_argument("--project-root", default=None, help="建模数据所在目录，如 ~/Desktop/硫化曲线")
    parser.add_argument("--curve-input", default=None, help="可选: model_ready_data.csv.gz 的绝对路径")
    parser.add_argument("--stage3-input", default=None, help="可选: pgmlp_curve_dataset.csv 的绝对路径")
    parser.add_argument("--output-dir", default=None, help="可选: 指定输出目录")
    parser.add_argument("--no-plots", action="store_true", help="只计算CSV，不生成88张图")
    parser.add_argument("--save-pdf", action="store_true", help="另保存88张PDF矢量图")
    args = parser.parse_args()

    base, curve_path, dataset_path = locate_files(
        args.project_root, args.curve_input, args.stage3_input)
    out_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
               base / "Stage5_Output_alpha0_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "Curves"
    if not args.no_plots:
        plot_dir.mkdir(exist_ok=True)

    required = ["No"] + X_COLS + TARGETS + ["alpha0"]
    data = pd.read_csv(dataset_path, low_memory=False)
    curves = pd.read_csv(curve_path, low_memory=False)
    missing = [col for col in required if col not in data.columns]
    missing_curves = [col for col in ("No", "Time", "torque") if col not in curves.columns]
    if missing or missing_curves:
        raise ValueError(f"Stage3缺失列: {missing}; 原始曲线缺失列: {missing_curves}")

    data = normalize_no(data, "Stage3")
    curves = normalize_no(curves, "model_ready_data")
    if data["No"].duplicated().any():
        raise ValueError("Stage3 每个 No 应只出现一行，检测到重复编号")
    if set(data["No"]) != EXPECTED_NOS or set(curves["No"]) != EXPECTED_NOS:
        raise ValueError("Stage5只接受建模集 No=1..88，检查两个输入文件是否误用了6条验证曲线")

    for col in X_COLS + TARGETS + ["alpha0"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    invalid = data[X_COLS + TARGETS + ["alpha0"]].isna().any(axis=1)
    if invalid.any() or not np.isfinite(data[X_COLS + TARGETS + ["alpha0"]].to_numpy()).all():
        raise ValueError(f"Stage3中有 {int(invalid.sum())} 条缺失参数或非有限数值，需先检查")
    if (data["k1"] <= 0).any():
        raise ValueError("训练数据k1必须大于0，才能使用log10变换")

    curves["Time"] = pd.to_numeric(curves["Time"], errors="coerce")
    curves["torque"] = pd.to_numeric(curves["torque"], errors="coerce")
    # Stage3 order is retained, because KFold uses its row order.
    data = data[required].reset_index(drop=True)
    print(f"项目目录: {base}\n原始曲线: {curve_path}\nStage3参数: {dataset_path}")
    print(f"曲线数量: {len(data)}; No=1–88\n输出目录: {out_dir}")

    X = data[X_COLS].to_numpy(dtype=float)
    oof = {target: np.full(len(data), np.nan) for target in TARGETS}
    oof_alpha0 = np.full(len(data), np.nan)
    fold_ids = np.zeros(len(data), dtype=int)
    fold_membership = []
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        alpha0_train_median = float(np.median(data.iloc[train_idx]["alpha0"]))
        oof_alpha0[test_idx] = alpha0_train_median
        fold_ids[test_idx] = fold
        for subset, idxs in (("Train", train_idx), ("Test", test_idx)):
            fold_membership.extend({"Fold": fold, "Subset": subset,
                                    "No": int(data.iloc[i]["No"])} for i in idxs)
        for target in TARGETS:
            y = data[target].to_numpy(dtype=float)
            model = get_model(target)
            if target == "k1":
                model.fit(X[train_idx], np.log10(y[train_idx]))
                oof[target][test_idx] = 10 ** model.predict(X[test_idx])
            else:
                model.fit(X[train_idx], y[train_idx])
                oof[target][test_idx] = model.predict(X[test_idx])
        print(f"Fold {fold}/{N_SPLITS}: train={len(train_idx)}, test={len(test_idx)},"
              f" alpha0(train median)={alpha0_train_median:.6f}")

    pd.DataFrame(fold_membership).to_csv(
        out_dir / "fold_membership.csv", index=False, encoding="utf-8-sig")
    pred_df = pd.DataFrame({"No": data["No"], "fold": fold_ids,
                            "alpha0_true_for_diagnostics_only": data["alpha0"],
                            "alpha0_used_from_train_fold": oof_alpha0})
    for target in TARGETS:
        pred_df[f"{target}_true"] = data[target].to_numpy()
        pred_df[f"{target}_pred"] = oof[target]
    pred_df.to_csv(out_dir / "oof_parameter_predictions.csv", index=False,
                   encoding="utf-8-sig")

    no_groups = {no: sub.dropna(subset=["Time", "torque"]).sort_values("Time")
                 for no, sub in curves.groupby("No")}
    metrics_records, point_records = [], []
    for i, row in data.iterrows():
        no = int(row["No"])
        sub = no_groups.get(no)
        if sub is None or len(sub) < 5:
            raise ValueError(f"曲线 {no} 的有效时间/扭矩点少于5个")
        t = sub["Time"].to_numpy(dtype=float)
        truth = sub["torque"].to_numpy(dtype=float)
        true_ml, true_mh = float(row["ML"]), float(row["MH"])
        pred_ml, pred_mh = float(oof["ML"][i]), float(oof["MH"][i])
        pred_mh = max(pred_mh, pred_ml + 1e-6)
        ti, k1, n1 = (float(oof[x][i]) for x in ("ti", "k1", "n1"))
        curves_for_plot = None
        for mode, a0 in (("zero", 0.0),
                         ("train_fold_median", float(oof_alpha0[i]))):
            alpha = alpha_curve(t, ti, k1, n1, a0)
            version_a = true_ml + alpha * (true_mh - true_ml)
            version_b = pred_ml + alpha * (pred_mh - pred_ml)
            for version, pred in (("A_true_ML_MH", version_a),
                                  ("B_pred_ML_MH", version_b)):
                metrics_records.append({"No": no, "fold": int(fold_ids[i]),
                                        "alpha0_mode": mode, "alpha0_used": a0,
                                        "Version": version, "n_points": len(t),
                                        **curve_metrics(truth, pred)})
            if mode == "train_fold_median":
                curves_for_plot = (version_a, version_b,
                                   curve_metrics(truth, version_a),
                                   curve_metrics(truth, version_b))
                # Facilitates exact point-by-point comparison with Stage 8.
                point_records.append(pd.DataFrame({
                    "No": no, "fold": int(fold_ids[i]), "Time": t,
                    "Experimental": truth, "Version_A": version_a,
                    "Version_B": version_b}))
        if not args.no_plots:
            a, b, met_a, met_b = curves_for_plot
            plot_curve(plot_dir / f"Curve_{no}_reconstruction.png", no, t, truth,
                       a, b, met_a, met_b, args.save_pdf)
        if no % 20 == 0 or no == 88:
            print(f"重构完成 {no}/88")

    metric_df = pd.DataFrame(metrics_records)
    if len(metric_df) != 88 * 4:
        raise RuntimeError(f"预期88×4行曲线指标，实际{len(metric_df)}行")
    metric_df.to_csv(out_dir / "curve_metrics_all_alpha0_modes.csv", index=False,
                     encoding="utf-8-sig")
    for version, filename in (("A_true_ML_MH", "curve_metrics_A.csv"),
                              ("B_pred_ML_MH", "curve_metrics_B.csv")):
        metric_df[(metric_df["Version"] == version) &
                  (metric_df["alpha0_mode"] == "train_fold_median")].to_csv(
                      out_dir / filename, index=False, encoding="utf-8-sig")
    pd.concat(point_records, ignore_index=True).to_csv(
        out_dir / "pointwise_AB_predictions.csv", index=False, encoding="utf-8-sig")

    summary = (metric_df.groupby(["alpha0_mode", "Version"])
               .agg(N=("No", "count"), Mean_R2=("R2", "mean"),
                    Median_R2=("R2", "median"), Std_R2=("R2", "std"),
                    Mean_RMSE=("RMSE", "mean"), Median_RMSE=("RMSE", "median"),
                    Mean_MAE=("MAE", "mean"), Median_MAE=("MAE", "median"))
               .reset_index())
    summary.to_csv(out_dir / "alpha0_ablation_summary.csv", index=False,
                   encoding="utf-8-sig")
    summary[summary["alpha0_mode"] == "train_fold_median"].to_csv(
        out_dir / "curve_metrics_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {"stage": 5, "input_curve": str(curve_path), "input_stage3": str(dataset_path),
                "sha256_input_curve": sha256(curve_path),
                "sha256_input_stage3": sha256(dataset_path), "n_curves": 88,
                "random_state": SEED, "n_splits": N_SPLITS,
                "rf_n_estimators": 300, "xgb_n_estimators": 300,
                "alpha0": "training_fold_median", "cv": "KFold(shuffle=True)",
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__}
    (out_dir / "stage5_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nStage 5 | alpha0 modes and Version A/B")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\n全部88条已完成。CSV与图像位置：{out_dir}")
    

if __name__ == "__main__":
    main()
