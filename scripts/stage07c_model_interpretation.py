
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import RepeatedKFold
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import shap



import argparse
import hashlib
import json


def resolve_stage3_file(project_root=None, explicit_input=None):
    script_dir = Path(__file__).resolve().parent
    if explicit_input:
        file_path = Path(explicit_input).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"指定的 Stage3 文件不存在：{file_path}")
        root = file_path.parent.parent if file_path.parent.name == "Stage3_Output" else file_path.parent
        return root, file_path
    if project_root:
        roots = [Path(project_root).expanduser().resolve()]
    else:
        roots = [script_dir, script_dir.parent, Path.cwd(),
                 Path.home()/"Desktop"/"硫化曲线"]
    checked = []
    for root in roots:
        for file_path in (root/"Stage3_Output"/"pgmlp_curve_dataset.csv",
                          root/"pgmlp_curve_dataset.csv",
                          root/"硫化曲线"/"Stage3_Output"/"pgmlp_curve_dataset.csv",
                          root/"建模集_1-88"/"Stage3_Output"/"pgmlp_curve_dataset.csv"):
            checked.append(file_path)
            if file_path.is_file():
                base = file_path.parent.parent if file_path.parent.name == "Stage3_Output" else file_path.parent
                return base, file_path
    raise FileNotFoundError(
        "找不到 Stage3 的 pgmlp_curve_dataset.csv。请检查 --project-root 或 --input。\n"
        + "\n".join(str(p) for p in checked))


def validate_88(df, required_columns):
    if "No" not in df.columns:
        raise ValueError("数据缺少 No 字段")
    nos = pd.to_numeric(df["No"], errors="coerce")
    if (len(df) != 88 or nos.isna().any() or
        not np.array_equal(np.sort(nos.to_numpy()), np.arange(1, 89))):
        raise ValueError("Stage7仅使用建模集：No须为1–88且每个编号恰好出现一次。")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Stage3 数据缺少以下字段：{missing}")
    for c in required_columns:
        val = pd.to_numeric(df[c], errors="coerce")
        if val.isna().any() or not np.isfinite(val.to_numpy(float)).all():
            raise ValueError(f"Stage3 的 {c} 存在非数值、缺失或非有限值。")
    print("数据检查：88条建模曲线，编号1–88，所需字段完整；未加载外部验证6条。")


def write_manifest(out_dir, input_file, name, settings):
    payload = {
        "script": name, "source_stage3": str(input_file),
        "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
        "dataset": "modeling No 1-88", "external_validation_included": False,
        "settings": settings,
    }
    (out_dir / "stage7_run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


parser = argparse.ArgumentParser(description="Stage 7C: SHAP与置换重要性")
parser.add_argument("--project-root", default=None, help="硫化曲线项目文件夹")
parser.add_argument("--input", default=None, help="明确指定 Stage3 数据文件")
parser.add_argument("--cv-repeats", type=int, default=int(os.getenv("STAGE_CV_REPEATS", "5")),
                    help="正式默认5，快速测试可设1")
parser.add_argument("--permutation-repeats", type=int,
                    default=int(os.getenv("STAGE_PERM_REPEATS", "30")),
                    help="正式默认30，快速测试可设2")
parser.add_argument("--jobs", type=int, default=1,
                    help="执行线程数；小样本推荐1（不改变模型算法）")
args = parser.parse_args()
if args.cv_repeats < 1 or args.permutation_repeats < 1:
    parser.error("重复次数必须>=1")

SEED = 42

N_SPLITS = 5
N_REPEATS = args.cv_repeats
N_PERM = args.permutation_repeats

X_COLS = [
    "mixing_time",
    "mixing_temp",
    "curing_temp",
    "curing_pressure",
    "rotor_speed_1",
    "rotor_speed_2",
    "curing_time_offset"
]

TARGETS = ["ti", "n1", "MH"]

RPM_INDEX = [
    X_COLS.index("rotor_speed_1"),
    X_COLS.index("rotor_speed_2")
]



def locate_input():
    return resolve_stage3_file(args.project_root, args.input)


def save(frame, name):

    frame.to_csv(
        OUT / name,
        index=False,
        encoding="utf-8-sig"
    )



def model_for(target):

    if target == "MH":

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=args.jobs
        )

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=args.jobs
    )



def metrics(actual, pred):

    return {
        "R2": r2_score(actual, pred),
        "RMSE": np.sqrt(
            mean_squared_error(actual, pred)
        ),
        "MAE": mean_absolute_error(
            actual, pred
        )
    }



def barplot(summary, target):

    sub = summary[
        summary["Target"] == target
    ].sort_values("Importance_mean")

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.barh(
        sub["Feature"],
        sub["Importance_mean"],
        xerr=sub["Importance_std"],
        color="#437D9B",
        capsize=3
    )

    ax.axvline(
        0,
        color="#555555",
        lw=1
    )

    ax.set_xlabel(
        "CV decrease in $R^2$ after permutation"
    )

    ax.set_title(
        f"Permutation Importance: {target}"
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        OUT / f"permutation_importance_{target}.png",
        dpi=350,
        bbox_inches="tight"
    )

    plt.close(fig)



ROOT, INPUT = locate_input()

OUT = ROOT / "Stage7C_Output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
validate_88(df, X_COLS + ["ML", "MH", "ti", "k1", "n1"])
write_manifest(OUT, INPUT, "Stage7C.py", {
    "cv": "RepeatedKFold", "folds": N_SPLITS, "repeats": N_REPEATS,
    "permutation_repeats": N_PERM, "seed": SEED, "parallel_jobs": args.jobs,
    "targets": TARGETS, "models": "RF300 / XGB300 as Stage4B",
    "split_unit": "No (original Stage4B RepeatedKFold)",
    "shap": "TreeExplainer on model fitted to all 88 data"
})

all_targets = ["ML", "MH", "ti", "k1", "n1"]

required = X_COLS + all_targets

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"缺少字段：{missing}"
    )

for c in required:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )

data = df.dropna(
    subset=required
).reset_index(drop=True)

X_df = data[X_COLS].copy()
X = X_df.to_numpy(dtype=float)

print("=" * 80)
print("Stage 7C：Permutation + SHAP")
print(f"输入文件：{INPUT}")
print(f"样本数：{len(data)}")
print(f"CV：{N_SPLITS} x {N_REPEATS}")
print(f"Permutation repeats：{N_PERM}")
print("=" * 80)



r1 = data["rotor_speed_1"]
r2 = data["rotor_speed_2"]

if r1.nunique() > 1 and r2.nunique() > 1:

    pr, pp = pearsonr(r1, r2)
    sr, sp = spearmanr(r1, r2)

else:

    pr = pp = sr = sp = np.nan

rpm_corr = pd.DataFrame([{
    "Variable_1": "rotor_speed_1",
    "Variable_2": "rotor_speed_2",
    "Pearson_r": pr,
    "Pearson_p": pp,
    "Spearman_rho": sr,
    "Spearman_p": sp
}])

save(
    rpm_corr,
    "rotor_speed_correlation.csv"
)

rotor_pairs = (
    data.groupby([
        "rotor_speed_1",
        "rotor_speed_2"
    ])
    .size()
    .reset_index(name="Count")
)

save(
    rotor_pairs,
    "rotor_speed_combinations.csv"
)



splits = list(
    RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED
    ).split(X)
)

fold_info = []

for fold, (tr, te) in enumerate(splits, 1):

    for subset, indices in (
        ("Train", tr),
        ("Test", te)
    ):

        for idx in indices:

            no = (
                int(data.iloc[idx]["No"])
                if "No" in data
                else idx
            )

            fold_info.append({
                "Fold": fold,
                "Subset": subset,
                "No": no
            })

save(
    pd.DataFrame(fold_info),
    "cv_fold_membership.csv"
)



cv_records = []
perm_raw = []
perm_fold = []

joint_raw = []
joint_fold = []

shap_records = []



for target in TARGETS:

    print(f"\n正在计算：{target}")

    y = data[target].to_numpy(float)


    for fold, (tr, te) in enumerate(
        splits, 1
    ):

        model = model_for(target)

        model.fit(
            X[tr],
            y[tr]
        )

        pred = model.predict(X[te])

        base_r2 = r2_score(
            y[te],
            pred
        )

        cv_records.append({
            "Target": target,
            "Fold": fold,
            **metrics(y[te], pred)
        })


        perm = permutation_importance(
            model,
            X[te],
            y[te],
            scoring="r2",
            n_repeats=N_PERM,
            random_state=SEED + fold,
            n_jobs=args.jobs
        )

        for idx, name in enumerate(X_COLS):

            importance = perm.importances[idx]

            for repeat, score in enumerate(
                importance, 1
            ):

                perm_raw.append({
                    "Target": target,
                    "Fold": fold,
                    "Feature": name,
                    "Repeat": repeat,
                    "Importance": float(score)
                })

            perm_fold.append({
                "Target": target,
                "Fold": fold,
                "Feature": name,
                "Importance_fold_mean": float(
                    np.mean(importance)
                )
            })


        rng = np.random.default_rng(
            SEED + fold
        )

        deltas = []

        for repeat in range(
            1, N_PERM + 1
        ):

            shuffled = X[te].copy()

            reindex = rng.permutation(
                len(te)
            )

            shuffled[:, RPM_INDEX] = (
                shuffled[reindex][:, RPM_INDEX]
            )

            score = (
                base_r2
                -
                r2_score(
                    y[te],
                    model.predict(shuffled)
                )
            )

            deltas.append(float(score))

            joint_raw.append({
                "Target": target,
                "Fold": fold,
                "Repeat": repeat,
                "Joint_Importance": float(score)
            })

        joint_fold.append({
            "Target": target,
            "Fold": fold,
            "Joint_Importance_fold_mean": float(
                np.mean(deltas)
            )
        })

    print(
        f"Permutation 完成：{len(splits)} folds"
    )


    full_model = model_for(target)

    full_model.fit(
        X_df,
        y
    )

    explainer = shap.TreeExplainer(
        full_model
    )

    shap_arr = np.asarray(
        explainer.shap_values(X_df)
    )

    if shap_arr.shape != X_df.shape:

        raise RuntimeError(
            f"{target} SHAP维度不符："
            f"{shap_arr.shape}"
        )

    mean_abs_shap = (
        np.abs(shap_arr).mean(axis=0)
    )

    for feature, value in zip(
        X_COLS,
        mean_abs_shap
    ):

        shap_records.append({
            "Target": target,
            "Feature": feature,
            "Mean_abs_SHAP": float(value)
        })


    for kind in ("bar", "beeswarm"):

        plt.figure(
            figsize=(8.5, 5.7)
        )

        shap.summary_plot(
            shap_arr,
            X_df,
            plot_type=(
                "bar"
                if kind == "bar"
                else "dot"
            ),
            max_display=len(X_COLS),
            show=False
        )

        plt.title(
            f"SHAP {kind.title()} for {target}",
            fontsize=13
        )

        plt.tight_layout()

        plt.savefig(
            OUT / f"SHAP_{kind}_{target}.png",
            dpi=350,
            bbox_inches="tight"
        )

        plt.close("all")

    # ========================================================
    # 10.4 SHAP Dependence
    # ========================================================

    order = np.argsort(
        mean_abs_shap
    )[::-1]

    for rank, idx in enumerate(
        order[:2], 1
    ):

        feature = X_COLS[idx]

        plt.figure(
            figsize=(7, 5)
        )

        shap.dependence_plot(
            feature,
            shap_arr,
            X_df,
            interaction_index=None,
            show=False
        )

        plt.title(
            f"{target} - {feature}"
        )

        plt.tight_layout()

        plt.savefig(
            OUT /
            f"SHAP_dependence_{target}_{rank}_{feature}.png",
            dpi=350,
            bbox_inches="tight"
        )

        plt.close("all")



cv = pd.DataFrame(cv_records)

save(
    cv,
    "cv_metrics_all_folds.csv"
)

cv_summary = (
    cv.groupby(
        "Target",
        as_index=False
    )
    .agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std")
    )
)

save(
    cv_summary,
    "cv_metrics_summary.csv"
)



save(
    pd.DataFrame(perm_raw),
    "permutation_importance_all.csv"
)

save(
    pd.DataFrame(perm_fold),
    "permutation_importance_by_fold.csv"
)

summary = (
    pd.DataFrame(perm_fold)
    .groupby(
        ["Target", "Feature"],
        as_index=False
    )
    .agg(
        Importance_mean=(
            "Importance_fold_mean",
            "mean"
        ),
        Importance_std=(
            "Importance_fold_mean",
            "std"
        )
    )
)

summary["Permutation_rank"] = (
    summary.groupby("Target")[
        "Importance_mean"
    ].rank(
        ascending=False,
        method="min"
    )
)

save(
    summary,
    "permutation_importance_summary.csv"
)

for target in TARGETS:
    barplot(summary, target)



shap_summary = pd.DataFrame(
    shap_records
)

shap_summary["SHAP_rank"] = (
    shap_summary.groupby("Target")[
        "Mean_abs_SHAP"
    ].rank(
        ascending=False,
        method="min"
    )
)

save(
    shap_summary,
    "SHAP_importance_summary.csv"
)



combined = summary.merge(
    shap_summary,
    on=["Target", "Feature"],
    how="left"
)

combined["Mean_rank"] = (
    combined["Permutation_rank"]
    +
    combined["SHAP_rank"]
) / 2

combined = combined.sort_values(
    ["Target", "Mean_rank"]
)

save(
    combined,
    "combined_feature_importance_ranking.csv"
)



save(
    pd.DataFrame(joint_raw),
    "rotor_speed_joint_permutation_all.csv"
)

joint = (
    pd.DataFrame(joint_fold)
    .groupby(
        "Target",
        as_index=False
    )
    .agg(
        Joint_Importance_mean=(
            "Joint_Importance_fold_mean",
            "mean"
        ),
        Joint_Importance_std=(
            "Joint_Importance_fold_mean",
            "std"
        )
    )
)

save(
    joint,
    "rotor_speed_joint_importance.csv"
)



print("\nRepeated CV Performance:")
print(
    cv_summary.to_string(index=False)
)

print("\nJoint Rotor Speed Importance:")
print(
    joint.to_string(index=False)
)

print("\nStage 7C 完成！")
print(f"输出目录：{OUT}")
