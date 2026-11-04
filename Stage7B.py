
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    pearsonr, spearmanr, rankdata,
    t as t_distribution
)

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



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


parser = argparse.ArgumentParser(description="Stage 7B: 偏相关与特征消融")
parser.add_argument("--project-root", default=None, help="硫化曲线项目文件夹")
parser.add_argument("--input", default=None, help="明确指定 Stage3 数据文件")
parser.add_argument("--cv-repeats", type=int, default=int(os.getenv("STAGE_CV_REPEATS", "5")),
                    help="正式默认5，快速检查可指定1")
args = parser.parse_args()
if args.cv_repeats < 1:
    parser.error("--cv-repeats 必须>=1")

SEED = 42
N_SPLITS = 5
N_REPEATS = args.cv_repeats

X_COLS = [
    "mixing_time",
    "mixing_temp",
    "curing_temp",
    "curing_pressure",
    "rotor_speed_1",
    "rotor_speed_2",
    "curing_time_offset"
]

TARGETS = ["ML", "MH", "ti", "k1", "n1"]

PARAMS = [
    "ML", "MH", "ti", "tp",
    "alpha0", "b", "k1", "n1"
]

CONTROL = "curing_temp"

FEATURE_SETS = {
    "CuringTemp_Only": ["curing_temp"],
    "All_Process": X_COLS,
    "Without_CuringTemp": [
        c for c in X_COLS if c != "curing_temp"
    ]
}



def locate_input():
    return resolve_stage3_file(args.project_root, args.input)


def save(df, name):
    df.to_csv(
        OUT / name,
        index=False,
        encoding="utf-8-sig"
    )



def bh_fdr(p):
    arr = np.asarray(p, dtype=float)
    result = np.full(arr.shape, np.nan)

    good = np.where(np.isfinite(arr))[0]

    if len(good) == 0:
        return result

    order = good[np.argsort(arr[good])]

    adjusted = (
        arr[order] * len(order)
        / np.arange(1, len(order) + 1)
    )

    result[order] = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1].clip(0, 1)

    return result



def residualize(y, control):
    mat = np.column_stack([
        np.ones(len(y)),
        control
    ])

    beta = np.linalg.lstsq(
        mat, y, rcond=None
    )[0]

    return np.asarray(y) - mat @ beta


def partial_corr(x, y, z, method="spearman"):

    x, y, z = map(
        lambda a: np.asarray(a, dtype=float),
        (x, y, z)
    )

    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
        z = rankdata(z)

    rx = residualize(x, z)
    ry = residualize(y, z)

    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan, np.nan

    r = float(np.clip(
        np.corrcoef(rx, ry)[0, 1],
        -1, 1
    ))

    dfree = len(x) - 3

    if dfree <= 0:
        return r, np.nan

    tval = abs(r) * np.sqrt(
        dfree / max(1 - r * r, 1e-14)
    )

    p = float(
        2 * t_distribution.sf(tval, dfree)
    )

    return r, p



def get_model(target):

    if target in ("ML", "MH"):

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
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
        n_jobs=-1
    )


def evaluate(actual, pred):
    return {
        "R2": r2_score(actual, pred),
        "RMSE": np.sqrt(
            mean_squared_error(actual, pred)
        ),
        "MAE": mean_absolute_error(actual, pred)
    }



ROOT, INPUT = locate_input()

OUT = ROOT / "Stage7B_Output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
validate_88(df, X_COLS + PARAMS)
if (pd.to_numeric(df["k1"]) <= 0).any():
    raise ValueError("k1 必须>0")
write_manifest(OUT, INPUT, "Stage7B.py", {
    "cv": "RepeatedKFold", "folds": N_SPLITS, "repeats": N_REPEATS,
    "seed": SEED, "models": "RF300 / XGB300 as Stage4B", "control": CONTROL,
    "targets": TARGETS, "features": FEATURE_SETS,
    "split_unit": "No (original Stage4B RepeatedKFold)"
})

needed = list(dict.fromkeys(
    ["No"] + X_COLS + PARAMS
))

missing = [
    c for c in needed if c not in df.columns
]

if missing:
    raise ValueError(f"数据缺少字段：{missing}")

for c in X_COLS + PARAMS:
    df[c] = pd.to_numeric(
        df[c], errors="coerce"
    )

model_df = df.dropna(
    subset=X_COLS + TARGETS
).reset_index(drop=True)

if (model_df["k1"] <= 0).any():
    raise ValueError(
        "k1 必须大于0，才能进行log10训练"
    )

print("=" * 80)
print("Stage 7B：偏相关 + 特征消融")
print(f"输入文件：{INPUT}")
print(f"有效样本：{len(model_df)}")
print(f"交叉验证：{N_SPLITS} x {N_REPEATS}")
print("=" * 80)



param_records = []

for i, a in enumerate(PARAMS):

    for b in PARAMS[i + 1:]:

        sub = df[
            [a, b, CONTROL]
        ].dropna()

        if len(sub) < 4:
            continue

        x, y, z = (
            sub[c].to_numpy(float)
            for c in (a, b, CONTROL)
        )

        raw_p = pearsonr(x, y)
        raw_s = spearmanr(x, y)

        pr, pp = partial_corr(
            x, y, z, "pearson"
        )

        sr, sp = partial_corr(
            x, y, z, "spearman"
        )

        param_records.append({
            "Variable_1": a,
            "Variable_2": b,
            "N": len(sub),
            "Pearson_r": raw_p.statistic,
            "Spearman_rho": raw_s.statistic,
            "Partial_Pearson_r": pr,
            "Partial_Pearson_p": pp,
            "Partial_Spearman_rho": sr,
            "Partial_Spearman_p": sp
        })

parameter = pd.DataFrame(param_records)

parameter["Partial_Spearman_FDR_q"] = bh_fdr(
    parameter["Partial_Spearman_p"]
)

parameter["Abs_Partial_Spearman"] = (
    parameter["Partial_Spearman_rho"].abs()
)

parameter = parameter.sort_values(
    "Abs_Partial_Spearman",
    ascending=False
)

save(
    parameter,
    "parameter_partial_correlations_control_curing_temp.csv"
)



process_records = []

for a in X_COLS:

    if a == CONTROL:
        continue

    for b in PARAMS:

        sub = df[
            [a, b, CONTROL]
        ].dropna()

        if len(sub) < 4:
            continue

        x, y, z = (
            sub[c].to_numpy(float)
            for c in (a, b, CONTROL)
        )

        raw = spearmanr(x, y)

        r, p = partial_corr(
            x, y, z, "spearman"
        )

        process_records.append({
            "Process_variable": a,
            "Parameter": b,
            "N": len(sub),
            "Raw_Spearman_rho": raw.statistic,
            "Raw_Spearman_p": raw.pvalue,
            "Partial_Spearman_rho": r,
            "Partial_Spearman_p": p
        })

process = pd.DataFrame(process_records)

process["Partial_Spearman_FDR_q"] = bh_fdr(
    process["Partial_Spearman_p"]
)

process["Abs_Partial_Spearman"] = (
    process["Partial_Spearman_rho"].abs()
)

process = process.sort_values(
    "Abs_Partial_Spearman",
    ascending=False
)

save(
    process,
    "process_parameter_partial_correlations_control_curing_temp.csv"
)



mat = pd.DataFrame(
    np.eye(len(PARAMS)),
    index=PARAMS,
    columns=PARAMS
)

for row in parameter.itertuples():
    mat.loc[
        row.Variable_1, row.Variable_2
    ] = row.Partial_Spearman_rho

    mat.loc[
        row.Variable_2, row.Variable_1
    ] = row.Partial_Spearman_rho

mat.to_csv(
    OUT / "partial_spearman_parameter_matrix_control_curing_temp.csv",
    encoding="utf-8-sig"
)

fig, ax = plt.subplots(figsize=(8.3, 6.8))

im = ax.imshow(
    mat.to_numpy(),
    vmin=-1,
    vmax=1,
    cmap="coolwarm"
)

ax.set_xticks(
    range(len(PARAMS)),
    PARAMS,
    rotation=45,
    ha="right"
)

ax.set_yticks(
    range(len(PARAMS)),
    PARAMS
)

for i in range(len(PARAMS)):
    for j in range(len(PARAMS)):

        v = mat.iloc[i, j]

        label = (
            f"{v:.2f}"
            if np.isfinite(v)
            else "–"
        )

        ax.text(
            j, i, label,
            ha="center",
            va="center",
            fontsize=9
        )

fig.colorbar(
    im,
    ax=ax,
    label="Partial Spearman correlation"
)

ax.set_title(
    "Partial Spearman Correlation\n"
    "Controlling for Curing Temperature"
)

fig.tight_layout()

fig.savefig(
    OUT / "heatmap_partial_Spearman_parameters_control_curing_temp.png",
    dpi=300
)

plt.close(fig)



splits = list(
    RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED
    ).split(model_df)
)

fold_rows = []

for f, (tr, te) in enumerate(splits, 1):

    for idx in tr:
        fold_rows.append({
            "Fold": f,
            "Subset": "Train",
            "No": model_df.loc[idx, "No"]
        })

    for idx in te:
        fold_rows.append({
            "Fold": f,
            "Subset": "Test",
            "No": model_df.loc[idx, "No"]
        })

save(
    pd.DataFrame(fold_rows),
    "cv_fold_membership.csv"
)



ablation_rows = []
pred_rows = []

for target in TARGETS:

    y = model_df[target].to_numpy(dtype=float)

    for set_name, features in FEATURE_SETS.items():

        X = model_df[
            features
        ].to_numpy(dtype=float)

        for fold, (tr, te) in enumerate(
            splits, 1
        ):

            model = get_model(target)

            if target == "k1":
                train_y = np.log10(y[tr])
            else:
                train_y = y[tr]

            model.fit(
                X[tr],
                train_y
            )

            pred = model.predict(X[te])

            if target == "k1":
                pred = 10 ** pred

            results = evaluate(
                y[te],
                pred
            )

            ablation_rows.append({
                "Target": target,
                "Feature_Set": set_name,
                "Fold": fold,
                "N_features": len(features),
                **results
            })

            for idx, actual, estimated in zip(
                te, y[te], pred
            ):

                pred_rows.append({
                    "Target": target,
                    "Feature_Set": set_name,
                    "Fold": fold,
                    "No": model_df.loc[idx, "No"],
                    "y_true": actual,
                    "y_pred": estimated
                })

        print(
            f"{target:>3} | "
            f"{set_name:<19} | "
            f"完成 {len(splits)} folds"
        )



ab = pd.DataFrame(ablation_rows)

save(
    ab,
    "ablation_all_folds.csv"
)

save(
    pd.DataFrame(pred_rows),
    "ablation_OOF_predictions.csv"
)

summary = (
    ab.groupby(
        ["Target", "Feature_Set"],
        as_index=False
    )
    .agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        R2_median=("R2", "median"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std")
    )
)

save(
    summary,
    "ablation_summary.csv"
)

key = (
    summary.pivot(
        index="Target",
        columns="Feature_Set",
        values="R2_mean"
    )
    .reset_index()
)

key = key.rename(columns={
    "CuringTemp_Only": "R2_CuringTemp_Only",
    "All_Process": "R2_All_Process",
    "Without_CuringTemp": "R2_Without_CuringTemp"
})

key["Delta_All_minus_TempOnly"] = (
    key["R2_All_Process"]
    - key["R2_CuringTemp_Only"]
)

save(
    key,
    "ablation_key_comparison.csv"
)



for target in TARGETS:

    sub = summary[
        summary["Target"] == target
    ].set_index("Feature_Set")

    sub = sub.loc[
        list(FEATURE_SETS)
    ]

    fig, ax = plt.subplots(
        figsize=(7.6, 5)
    )

    ax.bar(
        range(3),
        sub["R2_mean"],
        yerr=sub["R2_std"],
        capsize=4,
        color=[
            "#77A6C3",
            "#3E7695",
            "#B6C2C9"
        ]
    )

    ax.set_xticks(
        range(3),
        [
            "Curing temp\nonly",
            "All process\nvariables",
            "Without\ncuring temp"
        ]
    )

    ax.axhline(
        0,
        lw=1,
        color="#444444"
    )

    ax.set_ylabel("Mean CV $R^2$")
    ax.set_title(f"Feature Ablation for {target}")

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        OUT / f"ablation_{target}.png",
        dpi=350,
        bbox_inches="tight"
    )

    plt.close(fig)



print("\n消融实验结果：")
print(key.to_string(index=False))

print("\nStage 7B 完成！")
print(f"输出文件夹：{OUT}")
