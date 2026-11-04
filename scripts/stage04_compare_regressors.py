
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
    parser.add_argument("--only-targets", default=None,
                        help="快速检查时指定目标，如 ML,ti，默认全部5个")
    parser.add_argument("--only-models", default=None,
                        help="快速检查时指定模型，如 RF,XGB，默认4种模型全部运行")
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
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor



RANDOM_STATE = 42

np.random.seed(
    RANDOM_STATE
)



file_path, PROJECT_ROOT, ARGS = resolve_dataset()
out_dir = PROJECT_ROOT / "Stage4B_Output"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"读取：{file_path}\n输出：{out_dir}")



df = pd.read_csv(
    file_path
)
validate_model_dataset(df)
save_run_manifest(out_dir, file_path, "Stage4B.py", {"cv": "RepeatedKFold", "n_splits": 5, "n_repeats": ARGS.repeats, "seed": 42, "models": ["SVR", "RF", "XGB", "MLP"], "only_targets": ARGS.only_targets, "only_models": ARGS.only_models})


print("=" * 80)
print("PG-MLP Stage 4B：多模型动力学参数预测")
print("=" * 80)

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


Y_cols = [

    "ML",

    "MH",

    "ti",

    "k1",

    "n1"
]


if ARGS.only_targets:
    selected = [c.strip() for c in ARGS.only_targets.split(",") if c.strip()]
    invalid = [c for c in selected if c not in Y_cols]
    if invalid:
        raise ValueError(f"未知预测目标: {invalid}; 可选 {Y_cols}")
    Y_cols = selected


required_cols = (
    X_cols
    +
    Y_cols
)


missing_cols = [

    col

    for col in required_cols

    if col not in df.columns
]


if len(
    missing_cols
) > 0:

    raise ValueError(
        f"缺少字段：{missing_cols}"
    )


model_df = (
    df[
        required_cols
    ]
    .dropna()
    .copy()
)


print(
    f"有效样本数量："
    f"{len(model_df)}"
)


X = model_df[
    X_cols
].to_numpy(
    dtype=float
)



cv = RepeatedKFold(

    n_splits=5,

    n_repeats=ARGS.repeats,

    random_state=RANDOM_STATE
)



def get_models():

    models = {}


    models["SVR"] = Pipeline([

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


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models["RF"] = RandomForestRegressor(

        n_estimators=300,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    models["XGB"] = XGBRegressor(

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
    )


    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    models["MLP"] = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            MLPRegressor(

                hidden_layer_sizes=(
                    32,
                    16
                ),

                activation="relu",

                solver="adam",

                alpha=1e-3,

                learning_rate_init=0.001,

                max_iter=5000,

                early_stopping=True,

                validation_fraction=0.15,

                n_iter_no_change=100,

                random_state=RANDOM_STATE
            )
        )

    ])

    return models



def calc_metrics(
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

    return (
        r2,
        rmse,
        mae
    )



all_results = []


for target in Y_cols:

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        f"目标参数：{target}"
    )

    print(
        "=" * 80
    )


    y = model_df[
        target
    ].to_numpy(
        dtype=float
    )


    models = get_models()
    if ARGS.only_models:
        selected_models = [c.strip() for c in ARGS.only_models.split(",") if c.strip()]
        invalid = [m for m in selected_models if m not in models]
        if invalid:
            raise ValueError(f"未知模型: {invalid}; 可选 {list(models)}")
        models = {m: models[m] for m in selected_models}


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



            use_log = (
                target == "k1"
            )


            if use_log:

                y_train_model = np.log10(
                    y_train
                )

            else:

                y_train_model = y_train



            model.fit(

                X_train,

                y_train_model
            )



            pred = model.predict(
                X_test
            )



            if use_log:

                pred = (
                    10 ** pred
                )



            r2, rmse, mae = (
                calc_metrics(

                    y_test,

                    pred
                )
            )


            all_results.append({

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


            print(
                f"  Fold "
                f"{fold_id:02d} | "
                f"R²={r2:7.4f} | "
                f"RMSE={rmse:.6g}"
            )



results_df = pd.DataFrame(
    all_results
)


all_path = (
    out_dir
    /
    "model_comparison_all_folds.csv"
)


results_df.to_csv(

    all_path,

    index=False,

    encoding="utf-8-sig"
)



summary_df = (

    results_df

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
    "model_comparison_summary.csv"
)


summary_df.to_csv(

    summary_path,

    index=False,

    encoding="utf-8-sig"
)



best_rows = []


for target in Y_cols:

    temp = summary_df[
        summary_df[
            "Target"
        ]
        ==
        target
    ].copy()


    best_idx = (
        temp[
            "R2_mean"
        ]
        .idxmax()
    )


    best_rows.append(
        summary_df.loc[
            best_idx
        ]
    )


best_df = pd.DataFrame(
    best_rows
)


best_path = (

    out_dir
    /
    "best_model_per_target.csv"
)


best_df.to_csv(

    best_path,

    index=False,

    encoding="utf-8-sig"
)



print(
    "\n"
    +
    "=" * 80
)

print(
    "交叉验证完成"
)

print(
    "=" * 80
)


for target in Y_cols:

    print(
        f"\n【{target}】"
    )


    temp = (

        summary_df[
            summary_df[
                "Target"
            ]
            ==
            target
        ]

        .sort_values(
            "R2_mean",
            ascending=False
        )
    )


    print(

        temp[
            [
                "Model",

                "R2_mean",
                "R2_std",

                "RMSE_mean",
                "MAE_mean"
            ]
        ]
        .to_string(
            index=False
        )
    )



for target in Y_cols:

    temp = (

        summary_df[
            summary_df[
                "Target"
            ]
            ==
            target
        ]
        .copy()
    )


    plt.figure(
        figsize=(
            6.5,
            4.5
        )
    )


    plt.bar(

        temp[
            "Model"
        ],

        temp[
            "R2_mean"
        ],

        yerr=
            temp[
                "R2_std"
            ],

        capsize=5
    )


    plt.axhline(
        0,
        lw=1
    )


    plt.ylabel(
        "Mean R²"
    )


    plt.xlabel(
        "Model"
    )


    plt.title(
        f"{target}: 5-fold × 5 Repeated CV"
    )


    plt.tight_layout()


    plt.savefig(

        out_dir
        /
        f"comparison_{target}.png",

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()



print(
    "\n"
    +
    "=" * 80
)

print(
    "每个目标平均R²最高的模型"
)

print(
    "=" * 80
)


print(

    best_df[
        [
            "Target",

            "Model",

            "R2_mean",
            "R2_std",

            "RMSE_mean",
            "MAE_mean"
        ]
    ]
    .to_string(
        index=False
    )
)



print(
    "\n输出文件："
)

print(
    f"全部Fold结果："
    f"{all_path}"
)

print(
    f"统计汇总："
    f"{summary_path}"
)

print(
    f"最佳模型："
    f"{best_path}"
)

print(
    f"模型比较图片："
    f"{out_dir}"
)

print("=" * 80)