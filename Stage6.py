# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RANDOM_STATE = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 5
INNER_SPLITS = 5
EXPECTED_NOS = set(range(1, 89))
X_COLS = [
    "mixing_time", "mixing_temp", "curing_temp", "curing_pressure",
    "rotor_speed_1", "rotor_speed_2", "curing_time_offset",
]
KINETIC_COLS = ["ti", "k1", "n1"]


def find_stage3(root_arg: str | None, input_arg: str | None):
    """Resolve only the model-training Stage3 dataset, never external No 1–6."""
    here = Path(__file__).resolve().parent
    roots = ([Path(root_arg).expanduser().resolve()] if root_arg else [
        here, here / "硫化曲线", here.parent / "硫化曲线",
        Path.cwd(), Path.cwd() / "硫化曲线",
    ])
    variants = [Path("."), Path("建模集_1-88"), Path("建模集"),
                Path("Training_88"), Path("Modeling_88")]
    attempted = []
    if input_arg:
        candidate = Path(input_arg).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"指定的 --input 文件不存在: {candidate}")
        return (Path(root_arg).expanduser().resolve() if root_arg else candidate.parent), candidate
    for root in roots:
        for variant in variants:
            folder = root / variant
            for candidate in (folder / "Stage3_Output" / "pgmlp_curve_dataset.csv",
                              folder / "pgmlp_curve_dataset.csv"):
                attempted.append(str(candidate))
                if candidate.is_file():
                    return folder.resolve(), candidate.resolve()
    raise FileNotFoundError(
        "未找到 Stage3 的 pgmlp_curve_dataset.csv，建议使用 --project-root 指定『硫化曲线』目录。\n"
        + "\n".join(attempted[:16]))


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    required = ["No"] + X_COLS + KINETIC_COLS + ["MH"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Stage3 缺少必需字段: {missing}")
    nos = pd.to_numeric(data["No"], errors="coerce")
    if nos.isna().any() or not np.all(nos % 1 == 0):
        raise ValueError("Stage3 的 No 必须是有效整数")
    data["No"] = nos.astype(int)
    if len(data) != 88 or data["No"].duplicated().any() or set(data["No"]) != EXPECTED_NOS:
        raise ValueError("此程序只读取建模集编号 1–88 的88条曲线；请检查是否误用独立验证集")
    # Original row order is retained to match the original RepeatedKFold split.
    for col in X_COLS + KINETIC_COLS + ["MH"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    arr = data[X_COLS + KINETIC_COLS + ["MH"]].to_numpy(float)
    if not np.isfinite(arr).all():
        raise ValueError("存在缺失/非有限值；不自动丢弃建模曲线，需先修复 Stage3 数据")
    if (data["k1"] <= 0).any():
        raise ValueError("k1 必须为正数，因为原始 Stage6 模型对 k1 做 log10 变换")
    return data


def kinetic_model(n_jobs: int):
    # Exact original XGBoost settings.
    return XGBRegressor(
        objective="reg:squarederror", n_estimators=300,
        max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=2,
        reg_alpha=0.01, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=n_jobs,
    )


def mh_models(n_jobs: int):
    """Exact candidate models / settings from the user's original Stage 6."""
    return {
        "RF": RandomForestRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=n_jobs,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=n_jobs,
        ),
        "XGB": XGBRegressor(
            objective="reg:squarederror", n_estimators=300,
            max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2,
            reg_alpha=0.01, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=n_jobs,
        ),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
        ]),
    }


def metrics(true, pred):
    return (float(r2_score(true, pred)),
            float(np.sqrt(mean_squared_error(true, pred))),
            float(mean_absolute_error(true, pred)))


def plot_comparison(summary: pd.DataFrame, out_path: Path):
    model_order = ["RF", "ExtraTrees", "XGB", "SVR"]
    baseline = [summary.loc[(summary.Model == name) &
                            (summary.Method == "Baseline_X")].iloc[0]
                for name in model_order]
    assisted = [summary.loc[(summary.Model == name) &
                            (summary.Method == "Kinetics_Assisted")].iloc[0]
                for name in model_order]
    x = np.arange(len(model_order))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, [r.R2_mean for r in baseline], width,
            yerr=[r.R2_std if pd.notna(r.R2_std) else 0 for r in baseline],
            capsize=5, label="X only")
    plt.bar(x + width / 2, [r.R2_mean for r in assisted], width,
            yerr=[r.R2_std if pd.notna(r.R2_std) else 0 for r in assisted],
            capsize=5, label="X + predicted kinetics")
    plt.axhline(0, lw=1)
    plt.xticks(x, model_order)
    plt.ylabel("Mean R²")
    plt.xlabel("MH Model")
    plt.title("Kinetics-Assisted MH Prediction")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: MH baseline vs nested-OOF kinetics-assisted MH, No 1–88")
    parser.add_argument("--project-root", default=None, help="模型项目根目录，如 ~/Desktop/硫化曲线")
    parser.add_argument("--input", default=None, help="Stage3 pgmlp_curve_dataset.csv 的绝对路径")
    parser.add_argument("--output-dir", default=None, help="结果文件夹，可选")
    parser.add_argument("--outer-splits", type=int, default=OUTER_SPLITS)
    parser.add_argument("--outer-repeats", type=int, default=OUTER_REPEATS)
    parser.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--max-outer-folds", type=int, default=None,
                        help="只运行前N个外层折，用于测试；不能作为完整交叉验证结果")
    parser.add_argument("--quick-check", action="store_true", help="仅计算第1个外层折的快速自检")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="RF/XGB并行线程数，默认-1与原代码一致")
    parser.add_argument("--no-plots", action="store_true", help="跳过比较图生成")
    args = parser.parse_args()
    if not 2 <= args.outer_splits <= 88:
        parser.error("--outer-splits 必须位于2–88")
    if not 2 <= args.inner_splits <= 88:
        parser.error("--inner-splits 必须大于等于2")
    if args.outer_repeats < 1 or (args.max_outer_folds is not None and args.max_outer_folds < 1):
        parser.error("--outer-repeats 和 --max-outer-folds 必须是正整数")

    base, input_path = find_stage3(args.project_root, args.input)
    out_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
               base / "Stage6_Output")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(input_path)
    X = data[X_COLS].to_numpy(float)
    MH = data["MH"].to_numpy(float)
    nos = data["No"].to_numpy(int)
    target_arrays = {c: data[c].to_numpy(float) for c in KINETIC_COLS}
    print("=" * 72, flush=True)
    print("Stage 6：动力学参数辅助 MH 预测（88条建模曲线）", flush=True)
    print(f"Stage3 输入: {input_path}\n数据: {len(data)}条 No=1–88", flush=True)
    print(f"Outer CV: {args.outer_splits}折 × {args.outer_repeats}次；"
          f"Inner CV: {args.inner_splits}折", flush=True)
    print(f"结果目录: {out_dir}", flush=True)
    print("=" * 72, flush=True)

    outer = RepeatedKFold(n_splits=args.outer_splits,
                          n_repeats=args.outer_repeats,
                          random_state=RANDOM_STATE)
    cap = 1 if args.quick_check else args.max_outer_folds
    all_results = []
    all_predictions = []
    membership = []
    outer_completed = 0
    for fold, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
        if cap is not None and fold > cap:
            break
        outer_completed += 1
        repeat_id = (fold - 1) // args.outer_splits + 1
        fold_in_repeat = (fold - 1) % args.outer_splits + 1
        x_tr, x_te = X[train_idx], X[test_idx]
        mh_tr, mh_te = MH[train_idx], MH[test_idx]
        if len(train_idx) < args.inner_splits:
            raise ValueError("外层训练集小于inner-splits，请调整交叉验证设置")
        for subset, ids in (("Train", train_idx), ("Test", test_idx)):
            membership.extend({"Fold": fold, "Repeat": repeat_id,
                               "Fold_in_repeat": fold_in_repeat,
                               "Subset": subset, "No": int(nos[index])}
                              for index in ids)

        kinetic_train_oof = np.full((len(train_idx), len(KINETIC_COLS)), np.nan)
        kinetic_test = np.full((len(test_idx), len(KINETIC_COLS)), np.nan)
        inner_cv = KFold(n_splits=args.inner_splits, shuffle=True,
                         random_state=RANDOM_STATE + fold)
        inner_splits = list(inner_cv.split(x_tr))
        for j, target in enumerate(KINETIC_COLS):
            y_tr = target_arrays[target][train_idx]
            y_train_model = np.log10(y_tr) if target == "k1" else y_tr
            for inner_train, inner_val in inner_splits:
                mdl = kinetic_model(args.n_jobs)
                mdl.fit(x_tr[inner_train], y_train_model[inner_train])
                pred = mdl.predict(x_tr[inner_val])
                if target == "k1":
                    pred = 10 ** pred
                kinetic_train_oof[inner_val, j] = pred
            mdl = kinetic_model(args.n_jobs)
            mdl.fit(x_tr, y_train_model)
            test_pred = mdl.predict(x_te)
            if target == "k1":
                test_pred = 10 ** test_pred
            kinetic_test[:, j] = test_pred
        assert np.isfinite(kinetic_train_oof).all() and np.isfinite(kinetic_test).all()
        x_train_aug = np.column_stack([x_tr, kinetic_train_oof])
        x_test_aug = np.column_stack([x_te, kinetic_test])

        for name in ["RF", "ExtraTrees", "XGB", "SVR"]:
            baseline = mh_models(args.n_jobs)[name]
            baseline.fit(x_tr, mh_tr)
            pred_base = baseline.predict(x_te)
            assisted = mh_models(args.n_jobs)[name]
            assisted.fit(x_train_aug, mh_tr)
            pred_assisted = assisted.predict(x_test_aug)
            scores_base = metrics(mh_te, pred_base)
            scores_assist = metrics(mh_te, pred_assisted)
            for method, scores in (("Baseline_X", scores_base),
                                   ("Kinetics_Assisted", scores_assist)):
                all_results.append({"Fold": fold, "Repeat": repeat_id,
                                    "Fold_in_repeat": fold_in_repeat,
                                    "Model": name, "Method": method,
                                    "R2": scores[0], "RMSE": scores[1],
                                    "MAE": scores[2],
                                    "Train_N": len(train_idx), "Test_N": len(test_idx),
                                    "Test_Nos": ",".join(map(str, nos[test_idx]))})
            for local_i, global_i in enumerate(test_idx):
                all_predictions.append({
                    "Fold": fold, "Repeat": repeat_id,
                    "Fold_in_repeat": fold_in_repeat,
                    "No": int(nos[global_i]), "Model": name,
                    "MH_true": float(mh_te[local_i]),
                    "MH_baseline": float(pred_base[local_i]),
                    "MH_assisted": float(pred_assisted[local_i]),
                    "ti_pred": float(kinetic_test[local_i, 0]),
                    "k1_pred": float(kinetic_test[local_i, 1]),
                    "n1_pred": float(kinetic_test[local_i, 2]),
                })
            print(f"Outer {fold:02d}/{args.outer_splits * args.outer_repeats}"
                  f" | {name:10s} | baseline R²={scores_base[0]: .4f}"
                  f" | assisted R²={scores_assist[0]: .4f}", flush=True)

    results = pd.DataFrame(all_results)
    predictions = pd.DataFrame(all_predictions)
    summary = (results.groupby(["Model", "Method"], as_index=False)
               .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std")))
    improvement = []
    for name in ["RF", "ExtraTrees", "XGB", "SVR"]:
        base_row = summary.loc[(summary.Model == name) &
                               (summary.Method == "Baseline_X")].iloc[0]
        assist_row = summary.loc[(summary.Model == name) &
                                 (summary.Method == "Kinetics_Assisted")].iloc[0]
        improvement.append({
            "Model": name,
            "Baseline_R2": float(base_row.R2_mean),
            "Assisted_R2": float(assist_row.R2_mean),
            "Delta_R2": float(assist_row.R2_mean - base_row.R2_mean),
            "Baseline_RMSE": float(base_row.RMSE_mean),
            "Assisted_RMSE": float(assist_row.RMSE_mean),
            "Delta_RMSE": float(assist_row.RMSE_mean - base_row.RMSE_mean),
        })
    improvement = pd.DataFrame(improvement)
    artifacts = {
        "MH_assisted_all_folds.csv": results,
        "MH_assisted_summary.csv": summary,
        "MH_assisted_predictions.csv": predictions,
        "MH_improvement.csv": improvement,
        "fold_membership.csv": pd.DataFrame(membership),
    }
    for filename, frame in artifacts.items():
        frame.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")
    if not args.no_plots:
        plot_comparison(summary, out_dir / "comparison_MH_assisted.png")
    manifest = {
        "stage": "Stage6", "source": str(input_path), "source_sha256": sha256(input_path),
        "rows": len(data), "No_min": int(nos.min()), "No_max": int(nos.max()),
        "outer_splits": args.outer_splits, "outer_repeats": args.outer_repeats,
        "inner_splits": args.inner_splits, "completed_outer_folds": outer_completed,
        "planned_outer_folds": args.outer_splits * args.outer_repeats,
        "is_full_cv": outer_completed == args.outer_splits * args.outer_repeats,
        "random_state": RANDOM_STATE,
        "n_jobs": args.n_jobs,
        "note": "Only full CV is suitable for reporting. Kinetic test inputs were predicted from outer training folds only.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
    }
    (out_dir / "stage6_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 72, flush=True)
    print("Stage6 完成。各模型平均指标（辅助 - 基线）:", flush=True)
    print(improvement.to_string(index=False, float_format=lambda v: f"{v:.6f}"), flush=True)
    if not manifest["is_full_cv"]:
        print("注意：本次只完成部分外层折，结果用于代码检查，不作为正式交叉验证结果。", flush=True)
    print(f"输出目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
