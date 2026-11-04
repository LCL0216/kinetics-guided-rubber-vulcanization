
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import argparse
import json
import hashlib


def resolve_dataset():
    parser = argparse.ArgumentParser(description="PG-ML Stage 4: 编号1–88")
    parser.add_argument("--project-root", default=None,
                        help="硫化曲线文件夹；可包含 Stage3_Output/ 子目录")
    parser.add_argument("--input", default=None,
                        help="明确指定 pgmlp_curve_dataset.csv 文件的完整路径")
    parser.add_argument("--repeats", type=int, default=5,
                        help="仅Stage4B/4C有效，重复5折的次数；正式默认5")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    candidates = []
    if args.input:
        fp = Path(args.input).expanduser().resolve()
        if not fp.is_file():
            raise FileNotFoundError(f"指定输入文件不存在：{fp}")
        return fp, fp.parent.parent if fp.parent.name == "Stage3_Output" else fp.parent, args
    bases = ([Path(args.project_root).expanduser().resolve()] if args.project_root else
             [script_dir, script_dir.parent, Path.cwd(), Path.home()/"Desktop"/"硫化曲线"])
    for base in bases:
        for fp in (base/"Stage3_Output"/"pgmlp_curve_dataset.csv",
                   base/"pgmlp_curve_dataset.csv",
                   base/"硫化曲线"/"Stage3_Output"/"pgmlp_curve_dataset.csv",
                   base/"建模集_1-88"/"Stage3_Output"/"pgmlp_curve_dataset.csv"):
            candidates.append(fp)
            if fp.is_file():
                root = fp.parent.parent if fp.parent.name == "Stage3_Output" else fp.parent
                return fp, root, args
    raise FileNotFoundError("找不到 Stage 3 数据集 pgmlp_curve_dataset.csv；已检查：\n" +
                            "\n".join(str(p) for p in candidates) +
                            "\n可以通过 --project-root 或 --input 指定路径")


def validate_model_dataset(df):
    required = ["No", "mixing_time", "mixing_temp", "curing_temp", "curing_pressure",
                "rotor_speed_1", "rotor_speed_2", "curing_time_offset",
                "ML", "MH", "ti", "k1", "n1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stage3数据缺少字段：{missing}")
    ids = pd.to_numeric(df["No"], errors="coerce")
    if ids.isna().any() or (ids != ids.round()).any() or len(ids) != 88 or sorted(ids.astype(int).tolist()) != list(range(1, 89)):
        raise ValueError("本阶段必须使用建模集88条曲线，No应为1–88各出现一次；请检查输入文件。")
    if df[required].isna().any().any():
        raise ValueError("Stage3核心字段存在缺失值，请先检查Stage3输出")
    print("已核对：建模集 No=1–88，88条样本，7个工艺输入与5个预测目标完整。")


def save_run_manifest(out_dir, input_file, script_name, settings):
    content = input_file.read_bytes()
    manifest = {"script": script_name, "input": str(input_file),
                "input_sha256": hashlib.sha256(content).hexdigest(),
                "output": str(out_dir), "dataset": "建模集_No1-88",
                "validation_set_included": False, "settings": settings}
    (out_dir/"stage4_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")



from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor



RANDOM_STATE = 42

file_path, PROJECT_ROOT, ARGS = resolve_dataset()
out_dir = PROJECT_ROOT / "Stage4C_Output"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"读取：{file_path}\n输出：{out_dir}")



df = pd.read_csv(
    file_path
)
validate_model_dataset(df)
save_run_manifest(out_dir, file_path, "Stage4C.py", {"cv": "RepeatedKFold", "n_splits": 5, "n_repeats": ARGS.repeats, "seed": 42, "targets": ["MH", "deltaM"], "RF_n_estimators": 500})

print("=" * 75)
print("Stage 4C：MH vs ΔM 预测对比")
print("=" * 75)

print(
    f"样本数量：{len(df)}"
)



X_cols = [

    "mixing_time",
    "mixing_temp",
    "curing_temp",
    "curing_pressure",
    "rotor_speed_1",
    "rotor_speed_2",
    "curing_time_offset"
]


required = (
    X_cols
    +
    ["ML", "MH"]
)


missing = [

    col

    for col in required

    if col not in df.columns
]


if missing:

    raise ValueError(
        f"缺少字段：{missing}"
    )


data = (
    df[
        required
    ]
    .dropna()
    .copy()
)



data["deltaM"] = (
    data["MH"]
    -
    data["ML"]
)


print("\n目标统计：")

print(
    data[
        [
            "ML",
            "MH",
            "deltaM"
        ]
    ].describe()
)


# ============================================================
# 5. X
# ============================================================

X = data[
    X_cols
].to_numpy(
    dtype=float
)


# ============================================================
# 6. CV
# ============================================================

cv = RepeatedKFold(

    n_splits=5,

    n_repeats=ARGS.repeats,

    random_state=RANDOM_STATE
)



def get_models():

    return {

        "RF":
            RandomForestRegressor(

                n_estimators=500,

                max_features="sqrt",

                min_samples_leaf=1,

                random_state=RANDOM_STATE,

                n_jobs=-1
            ),

        "XGB":
            XGBRegressor(

                objective="reg:squarederror",

                n_estimators=300,

                max_depth=3,

                learning_rate=0.03,

                subsample=0.8,

                colsample_bytree=0.8,

                min_child_weight=2,

                reg_alpha=0.01,

                reg_lambda=1.0,

                random_state=RANDOM_STATE,

                n_jobs=-1
            ),

        "SVR":
            Pipeline([

                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    SVR(

                        kernel="rbf",

                        C=10,

                        epsilon=0.1,

                        gamma="scale"
                    )
                )

            ])
    }



def metrics(
    y_true,
    y_pred
):

    r2 = r2_score(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return r2, rmse, mae



targets = [
    "MH",
    "deltaM"
]

records = []


for target in targets:

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        f"目标：{target}"
    )

    print(
        "=" * 75
    )


    y = data[
        target
    ].to_numpy(
        dtype=float
    )


    models = get_models()


    for model_name, model in models.items():

        print(
            f"\n模型：{model_name}"
        )

        fold_id = 0


        for train_idx, test_idx in cv.split(X):

            fold_id += 1

            X_train = X[
                train_idx
            ]

            X_test = X[
                test_idx
            ]

            y_train = y[
                train_idx
            ]

            y_test = y[
                test_idx
            ]


            model.fit(
                X_train,
                y_train
            )


            pred = model.predict(
                X_test
            )


            r2, rmse, mae = metrics(
                y_test,
                pred
            )


            records.append({

                "Target":
                    target,

                "Model":
                    model_name,

                "Fold":
                    fold_id,

                "R2":
                    r2,

                "RMSE":
                    rmse,

                "MAE":
                    mae
            })



all_df = pd.DataFrame(
    records
)


all_path = (
    out_dir
    /
    "MH_deltaM_all_folds.csv"
)


all_df.to_csv(

    all_path,

    index=False,

    encoding="utf-8-sig"
)



summary = (

    all_df

    .groupby(
        [
            "Target",
            "Model"
        ]
    )

    .agg(

        R2_mean=(
            "R2",
            "mean"
        ),

        R2_std=(
            "R2",
            "std"
        ),

        RMSE_mean=(
            "RMSE",
            "mean"
        ),

        RMSE_std=(
            "RMSE",
            "std"
        ),

        MAE_mean=(
            "MAE",
            "mean"
        ),

        MAE_std=(
            "MAE",
            "std"
        )
    )

    .reset_index()
)


summary_path = (
    out_dir
    /
    "MH_deltaM_summary.csv"
)


summary.to_csv(

    summary_path,

    index=False,

    encoding="utf-8-sig"
)



print(
    "\n"
    +
    "=" * 75
)

print(
    "MH vs ΔM 结果"
)

print(
    "=" * 75
)


print(

    summary

    .sort_values(
        [
            "Target",
            "R2_mean"
        ],

        ascending=[
            True,
            False
        ]
    )

    .to_string(
        index=False
    )
)



print(
    "\n最佳模型："
)


best_rows = []


for target in targets:

    temp = summary[
        summary["Target"]
        ==
        target
    ]

    idx = (
        temp[
            "R2_mean"
        ]
        .idxmax()
    )

    row = summary.loc[
        idx
    ]

    best_rows.append(
        row
    )

    print(
        f"{target}: "
        f"{row['Model']} | "
        f"R² = "
        f"{row['R2_mean']:.4f}"
        f" ± "
        f"{row['R2_std']:.4f} | "
        f"RMSE = "
        f"{row['RMSE_mean']:.4f}"
    )



plot_df = summary.copy()


models = [
    "RF",
    "SVR",
    "XGB"
]


x = np.arange(
    len(models)
)


width = 0.35


mh_values = []

mh_errors = []

dm_values = []

dm_errors = []


for model in models:

    mh_row = plot_df[
        (
            plot_df["Target"]
            ==
            "MH"
        )
        &
        (
            plot_df["Model"]
            ==
            model
        )
    ].iloc[0]


    dm_row = plot_df[
        (
            plot_df["Target"]
            ==
            "deltaM"
        )
        &
        (
            plot_df["Model"]
            ==
            model
        )
    ].iloc[0]


    mh_values.append(
        mh_row["R2_mean"]
    )

    mh_errors.append(
        mh_row["R2_std"]
    )

    dm_values.append(
        dm_row["R2_mean"]
    )

    dm_errors.append(
        dm_row["R2_std"]
    )


plt.figure(
    figsize=(7, 4.8)
)


plt.bar(

    x - width/2,

    mh_values,

    width,

    yerr=mh_errors,

    capsize=5,

    label="MH"
)


plt.bar(

    x + width/2,

    dm_values,

    width,

    yerr=dm_errors,

    capsize=5,

    label="ΔM = MH - ML"
)


plt.axhline(
    0,
    lw=1
)


plt.xticks(
    x,
    models
)


plt.ylabel(
    "Mean R²"
)


plt.xlabel(
    "Model"
)


plt.title(
    "Prediction of MH vs ΔM"
)


plt.legend(
    frameon=False
)


plt.tight_layout()


plt.savefig(

    out_dir
    /
    "comparison_MH_deltaM.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close()



print(
    "\n输出文件："
)

print(
    f"全部fold："
    f"{all_path}"
)

print(
    f"汇总："
    f"{summary_path}"
)

print(
    f"图片："
    f"{out_dir / 'comparison_MH_deltaM.png'}"
)

print("=" * 75)