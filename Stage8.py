# -*- coding: utf-8 -*-



from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import TransformedTargetRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor



warnings.filterwarnings("ignore")

RANDOM_STATE = 42

N_SPLITS = 5

DIRECT_TRAIN_POINTS_PER_CURVE = 300



PROCESS_COLS = [

    "mixing_time",

    "mixing_temp",

    "curing_temp",

    "curing_pressure",

    "rotor_speed_1",

    "rotor_speed_2",

    "curing_time_offset"
]



PG_TARGETS = [

    "ML",

    "MH",

    "ti",

    "k1",

    "n1"
]



METHODS = [

    "Direct-MLP",

    "Direct-XGB",

    "PG-ML"
]



COLORS = {

    "Experimental":
        "#1F77B4",

    "Direct-MLP":
        "#2CA02C",

    "Direct-XGB":
        "#FF7F0E",

    "PG-ML":
        "#D62728"

}



PARSER = argparse.ArgumentParser(description="Stage 8: Direct-MLP, Direct-XGB 与 PG-ML 公平对照（No.1–88）")
PARSER.add_argument("--project-root", default=None, help="包含 model_ready_data.csv.gz 和 Stage3_Output 的项目目录")
PARSER.add_argument("--output-dir", default=None, help="可选：自定义输出目录；默认项目目录/Stage8_Output")
PARSER.add_argument("--fold-limit", type=int, default=N_SPLITS,
                    help="仅作程序快速测试：运行前几个折；正式科研评价必须使用默认的5折")
ARGS = PARSER.parse_args()
if not (1 <= ARGS.fold_limit <= N_SPLITS):
    PARSER.error(f"--fold-limit 必须为1到{N_SPLITS}之间的整数")


def locate_project():
    script_dir = Path(__file__).resolve().parent
    if ARGS.project_root:
        candidates = [Path(ARGS.project_root).expanduser().resolve()]
    else:
        candidates = [script_dir, script_dir.parent, script_dir.parent.parent, Path.cwd()]
    tried = []
    for root in candidates:
        curve_file = root / "model_ready_data.csv.gz"
        for stage3_file in [root / "Stage3_Output" / "pgmlp_curve_dataset.csv",
                            root / "pgmlp_curve_dataset.csv"]:
            tried.append(f"{curve_file} + {stage3_file}")
            if curve_file.is_file() and stage3_file.is_file():
                return root, curve_file, stage3_file
    raise FileNotFoundError("没有找到建模数据：\n" + "\n".join(tried) +
                            "\n请设置 --project-root，例如 /path/to/work")



(
    ROOT,
    CURVE_FILE,
    STAGE3_FILE

) = locate_project()


OUT = (Path(ARGS.output_dir).expanduser().resolve() if ARGS.output_dir
       else ROOT / "Stage8_Output")


OUT.mkdir(

    parents=True,

    exist_ok=True

)



def save_csv(
    df,
    filename
):

    df.to_csv(

        OUT
        /
        filename,

        index=False,

        encoding="utf-8-sig"

    )



def canonical_no(
    series
):

    return (

        series

        .astype(str)

        .str.strip()

        .str.replace(
            r"\.0$",
            "",
            regex=True
        )

    )



def calc_metrics(
    y_true,
    y_pred
):

    return {

        "R2":
            r2_score(
                y_true,
                y_pred
            ),

        "RMSE":
            np.sqrt(

                mean_squared_error(
                    y_true,
                    y_pred
                )

            ),

        "MAE":
            mean_absolute_error(
                y_true,
                y_pred
            )

    }



def clean_curve(
    group
):

    g = group[
        [
            "Time",
            "torque"
        ]
    ].copy()


    g["Time"] = pd.to_numeric(

        g["Time"],

        errors="coerce"

    )


    g["torque"] = pd.to_numeric(

        g["torque"],

        errors="coerce"

    )


    g = (

        g

        .dropna()

        .sort_values(
            "Time"
        )

    )



    g = (

        g

        .groupby(
            "Time",
            as_index=False
        )["torque"]

        .mean()

        .sort_values(
            "Time"
        )

        .reset_index(
            drop=True
        )

    )


    return g



def resample_curve(

    group,

    n_points=500

):

    g = clean_curve(
        group
    )


    if len(g) < 2:

        raise ValueError(
            "曲线有效数据点少于2个。"
        )


    t = g[
        "Time"
    ].to_numpy(
        dtype=float
    )


    y = g[
        "torque"
    ].to_numpy(
        dtype=float
    )


    if np.isclose(
        t[0],
        t[-1]
    ):

        raise ValueError(
            "曲线时间范围为0。"
        )


    t_new = np.linspace(

        t[0],

        t[-1],

        n_points

    )


    y_new = np.interp(

        t_new,

        t,

        y

    )


    return (
        t_new,
        y_new
    )



def curing_alpha(

    t,

    ti,

    k1,

    n1,

    alpha0

):

    t = np.asarray(
        t,
        dtype=float
    )


    ti = max(
        float(ti),
        1e-12
    )


    k1 = max(
        float(k1),
        1e-12
    )


    n1 = max(
        float(n1),
        0.1
    )


    alpha0 = float(

        np.clip(
            alpha0,
            0.0,
            1.0
        )

    )


    alpha = np.zeros_like(

        t,

        dtype=float

    )


    early = (
        t
        <
        ti
    )


    late = ~early



    alpha[
        early
    ] = (

        alpha0

        *

        np.clip(

            t[early]
            /
            ti,

            0.0,

            1.0

        )

    )



    dt = np.maximum(

        t[late]
        -
        ti,

        0.0

    )


    alpha[
        late
    ] = (

        alpha0

        +

        (
            1.0
            -
            alpha0
        )

        *

        (

            1.0

            -

            np.exp(

                -k1

                *

                dt ** n1

            )

        )

    )


    return np.clip(

        alpha,

        0.0,

        1.0

    )



def make_direct_mlp():

    regressor = Pipeline([

        (

            "scaler",

            StandardScaler()

        ),

        (

            "mlp",

            MLPRegressor(

                hidden_layer_sizes=(
                    64,
                    64,
                    32
                ),

                activation="relu",

                solver="adam",

                alpha=1e-3,

                learning_rate_init=1e-3,

                max_iter=3000,

                early_stopping=True,

                validation_fraction=0.15,

                n_iter_no_change=80,

                random_state=RANDOM_STATE

            )

        )

    ])



    model = TransformedTargetRegressor(

        regressor=
            regressor,

        transformer=
            StandardScaler()

    )


    return model


# ============================================================
# 10. Direct-XGB
# ============================================================

def make_direct_xgb():

    return XGBRegressor(

        objective=
            "reg:squarederror",

        n_estimators=
            600,

        max_depth=
            4,

        learning_rate=
            0.03,

        subsample=
            0.8,

        colsample_bytree=
            0.9,

        min_child_weight=
            2,

        reg_alpha=
            0.01,

        reg_lambda=
            1.0,

        random_state=
            RANDOM_STATE,

        n_jobs=
            -1

    )



def make_pg_model(
    target
):

    if target in [
        "ML",
        "MH"
    ]:

        return RandomForestRegressor(

            n_estimators=300,

            max_depth=None,

            min_samples_split=2,

            min_samples_leaf=1,

            max_features="sqrt",

            random_state=
                RANDOM_STATE,

            n_jobs=-1

        )


    return XGBRegressor(

        objective=
            "reg:squarederror",

        n_estimators=
            300,

        max_depth=
            3,

        learning_rate=
            0.03,

        subsample=
            0.8,

        colsample_bytree=
            0.8,

        min_child_weight=
            2,

        reg_alpha=
            0.01,

        reg_lambda=
            1.0,

        random_state=
            RANDOM_STATE,

        n_jobs=
            -1

    )



def build_direct_train(

    train_df,

    curve_df

):

    X_blocks = []

    y_blocks = []


    for _, row in train_df.iterrows():

        no_key = row[
            "_NoKey"
        ]


        if no_key not in RESAMPLED_CURVES:
            raise KeyError(f"训练曲线 No={no_key} 不存在于原始数据")
        t, y = RESAMPLED_CURVES[no_key]

        process_values = (

            row[
                PROCESS_COLS
            ]

            .to_numpy(
                dtype=float
            )

        )


        process_matrix = np.repeat(

            process_values.reshape(
                1,
                -1
            ),

            len(t),

            axis=0

        )


        X_block = np.column_stack([

            process_matrix,

            t

        ])


        X_blocks.append(
            X_block
        )


        y_blocks.append(
            y
        )


    return (

        np.vstack(
            X_blocks
        ),

        np.concatenate(
            y_blocks
        )

    )



def build_direct_test(

    row,

    t

):

    process_values = (

        row[
            PROCESS_COLS
        ]

        .to_numpy(
            dtype=float
        )

    )


    process_matrix = np.repeat(

        process_values.reshape(
            1,
            -1
        ),

        len(t),

        axis=0

    )


    return np.column_stack([

        process_matrix,

        t

    ])



def predict_pg_parameters(

    train_df,

    test_df

):

    X_train = (

        train_df[
            PROCESS_COLS
        ]

        .to_numpy(
            dtype=float
        )

    )


    X_test = (

        test_df[
            PROCESS_COLS
        ]

        .to_numpy(
            dtype=float
        )

    )


    predictions = {}


    for target in PG_TARGETS:

        model = make_pg_model(
            target
        )


        y_train = (

            train_df[
                target
            ]

            .to_numpy(
                dtype=float
            )

        )



        if target == "k1":

            if np.any(
                y_train <= 0
            ):

                raise ValueError(

                    "k1中存在非正数，"
                    "无法进行log10。"

                )


            model.fit(

                X_train,

                np.log10(
                    y_train
                )

            )


            predictions[
                target
            ] = (

                10
                **
                model.predict(
                    X_test
                )

            )


        else:

            model.fit(

                X_train,

                y_train

            )


            predictions[
                target
            ] = model.predict(
                X_test
            )


    return predictions



curve_df = pd.read_csv(
    CURVE_FILE
)


stage3 = pd.read_csv(
    STAGE3_FILE
)



required_curve = [

    "No",

    "Time",

    "torque"

]


required_stage3 = [

    "No",

    *PROCESS_COLS,

    "ML",

    "MH",

    "ti",

    "alpha0",

    "k1",

    "n1"

]


missing_curve = [

    c for c in required_curve

    if c not in curve_df.columns

]


missing_stage3 = [

    c for c in required_stage3

    if c not in stage3.columns

]


if missing_curve:

    raise ValueError(

        f"model_ready_data.csv.gz缺少："
        f"{missing_curve}"

    )


if missing_stage3:

    raise ValueError(

        f"pgmlp_curve_dataset.csv缺少："
        f"{missing_stage3}"

    )



curve_df[
    "_NoKey"
] = canonical_no(

    curve_df[
        "No"
    ]

)


stage3[
    "_NoKey"
] = canonical_no(

    stage3[
        "No"
    ]

)



for col in [

    "Time",

    "torque"

]:

    curve_df[col] = pd.to_numeric(

        curve_df[col],

        errors="coerce"

    )


for col in (

    PROCESS_COLS

    +

    [
        "ML",
        "MH",
        "ti",
        "alpha0",
        "k1",
        "n1"
    ]

):

    stage3[col] = pd.to_numeric(

        stage3[col],

        errors="coerce"

    )



stage3 = (

    stage3

    .dropna(

        subset=

        PROCESS_COLS

        +

        [
            "ML",
            "MH",
            "ti",
            "alpha0",
            "k1",
            "n1"
        ]

    )

    .reset_index(
        drop=True
    )

)



available_No = set(

    curve_df[
        "_NoKey"
    ]

    .dropna()

    .unique()

)


stage3 = (

    stage3[

        stage3[
            "_NoKey"
        ]

        .isin(
            available_No
        )

    ]

    .reset_index(
        drop=True
    )

)


EXPECTED_NOS = set(range(1, 89))
for name, frame in (("原始曲线", curve_df), ("Stage3曲线级数据", stage3)):
    numeric_nos = pd.to_numeric(frame["_NoKey"], errors="coerce")
    if numeric_nos.isna().any() or not np.all(np.isfinite(numeric_nos)):
        raise ValueError(f"{name}存在非数值No")
    if not np.all(numeric_nos.astype(int).eq(numeric_nos)):
        raise ValueError(f"{name}含非整数No")
    actual = set(numeric_nos.astype(int).tolist())
    if actual != EXPECTED_NOS:
        raise ValueError(f"{name}编号不符合1–88：缺少={sorted(EXPECTED_NOS-actual)}，"
                         f"多出={sorted(actual-EXPECTED_NOS)}")
if len(stage3) != 88 or stage3["_NoKey"].duplicated().any():
    raise ValueError("Stage3必须恰好包含88条曲线级记录且No无重复")
if (stage3["k1"] <= 0).any() or not np.isfinite(
    stage3[PROCESS_COLS + ["ML", "MH", "ti", "alpha0", "k1", "n1"]].to_numpy(float)
).all():
    raise ValueError("Stage3包含非法或缺失的动力学参数")

CURVE_GROUPS = {str(k): v for k, v in curve_df.groupby("_NoKey", sort=False)}
RESAMPLED_CURVES = {
    key: resample_curve(group, DIRECT_TRAIN_POINTS_PER_CURVE)
    for key, group in CURVE_GROUPS.items()
}
print(f"验证建模集 No.1–88：Stage3={len(stage3)}条，原始曲线={len(CURVE_GROUPS)}条")


print(
    "=" * 90
)

print(
    "Stage 8："
    "Direct ML vs Physics-Guided ML"
)

print(
    "=" * 90
)

print(
    f"项目目录：{ROOT}"
)

print(
    f"有效曲线数：{len(stage3)}"
)

print(
    f"交叉验证：{N_SPLITS}-fold"
)

print(
    f"Direct训练采样："
    f"{DIRECT_TRAIN_POINTS_PER_CURVE} "
    f"points/curve"
)

print(
    "=" * 90
)



save_csv(

    stage3,

    "frozen_stage8_dataset.csv"

)



kf = KFold(

    n_splits=
        N_SPLITS,

    shuffle=
        True,

    random_state=
        RANDOM_STATE

)


splits = list(kf.split(stage3))[:ARGS.fold_limit]
if ARGS.fold_limit != N_SPLITS:
    print(f"警告：快速检查模式（仅{ARGS.fold_limit}/{N_SPLITS}折），不能作为正式研究结论")



fold_records = []


for fold, (
    train_idx,
    test_idx
) in enumerate(
    splits,
    start=1
):

    for subset, indices in [

        (
            "Train",
            train_idx
        ),

        (
            "Test",
            test_idx
        )

    ]:

        for idx in indices:

            fold_records.append({

                "Fold":
                    fold,

                "Subset":
                    subset,

                "No":
                    stage3.loc[
                        idx,
                        "No"
                    ]

            })


save_csv(

    pd.DataFrame(
        fold_records
    ),

    "fold_membership.csv"

)



metric_records = []

point_records = []

pg_parameter_records = []



for fold, (
    train_idx,
    test_idx
) in enumerate(
    splits,
    start=1
):

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        f"Fold {fold}/{N_SPLITS}"
    )

    print(
        "=" * 90
    )


    train_df = (

        stage3

        .iloc[
            train_idx
        ]

        .reset_index(
            drop=True
        )

    )


    test_df = (

        stage3

        .iloc[
            test_idx
        ]

        .reset_index(
            drop=True
        )

    )



    (
        X_train_direct,
        y_train_direct

    ) = build_direct_train(

        train_df,

        curve_df

    )


    print(

        f"Direct训练点："
        f"{len(y_train_direct):,}"

    )


    # ========================================================
    # Direct-MLP
    # ========================================================

    print(
        "Training Direct-MLP..."
    )

    direct_mlp = make_direct_mlp()

    direct_mlp.fit(

        X_train_direct,

        y_train_direct

    )


    # ========================================================
    # Direct-XGB
    # ========================================================

    print(
        "Training Direct-XGB..."
    )

    direct_xgb = make_direct_xgb()

    direct_xgb.fit(

        X_train_direct,

        y_train_direct

    )



    print(
        "Training Physics-Guided models..."
    )

    pg_predictions = predict_pg_parameters(

        train_df,

        test_df

    )



    alpha0_fold = float(

        train_df[
            "alpha0"
        ]

        .median()

    )


    print(

        f"alpha0(train median) = "
        f"{alpha0_fold:.6f}"

    )



    for i, row in test_df.iterrows():

        no = row[
            "No"
        ]


        no_key = row[
            "_NoKey"
        ]


        group = CURVE_GROUPS.get(no_key, pd.DataFrame(columns=["Time", "torque"]))

        group = clean_curve(
            group
        )


        if len(group) < 2:

            continue



        t = (

            group[
                "Time"
            ]

            .to_numpy(
                dtype=float
            )

        )


        torque_true = (

            group[
                "torque"
            ]

            .to_numpy(
                dtype=float
            )

        )



        X_test_direct = build_direct_test(

            row,

            t

        )



        torque_mlp = direct_mlp.predict(

            X_test_direct

        )


        torque_xgb = direct_xgb.predict(

            X_test_direct

        )



        ML_pred = float(

            pg_predictions[
                "ML"
            ][i]

        )


        MH_pred = float(

            pg_predictions[
                "MH"
            ][i]

        )


        ti_pred = max(

            float(

                pg_predictions[
                    "ti"
                ][i]

            ),

            1e-12

        )


        k1_pred = max(

            float(

                pg_predictions[
                    "k1"
                ][i]

            ),

            1e-12

        )


        n1_pred = max(

            float(

                pg_predictions[
                    "n1"
                ][i]

            ),

            0.1

        )



        if MH_pred <= ML_pred:

            MH_pred = (

                ML_pred
                +
                1e-6

            )


        # ====================================================
        # alpha(t)
        # ====================================================

        alpha_pg = curing_alpha(

            t=
                t,

            ti=
                ti_pred,

            k1=
                k1_pred,

            n1=
                n1_pred,

            alpha0=
                alpha0_fold

        )



        torque_pg = (

            ML_pred

            +

            alpha_pg

            *

            (

                MH_pred
                -
                ML_pred

            )

        )



        pg_parameter_records.append({

            "Fold":
                fold,

            "No":
                no,

            "alpha0_train_median":
                alpha0_fold,

            "ML_true":
                row["ML"],

            "ML_pred":
                ML_pred,

            "MH_true":
                row["MH"],

            "MH_pred":
                MH_pred,

            "ti_true":
                row["ti"],

            "ti_pred":
                ti_pred,

            "k1_true":
                row["k1"],

            "k1_pred":
                k1_pred,

            "n1_true":
                row["n1"],

            "n1_pred":
                n1_pred

        })



        predictions = {

            "Direct-MLP":
                torque_mlp,

            "Direct-XGB":
                torque_xgb,

            "PG-ML":
                torque_pg

        }



        for method, pred in predictions.items():

            metric = calc_metrics(

                torque_true,

                pred

            )


            metric_records.append({

                "Fold":
                    fold,

                "No":
                    no,

                "Method":
                    method,

                "N_points":
                    len(t),

                "R2":
                    metric["R2"],

                "RMSE":
                    metric["RMSE"],

                "MAE":
                    metric["MAE"]

            })



        for j in range(
            len(t)
        ):

            point_records.append({

                "Fold":
                    fold,

                "No":
                    no,

                "Time":
                    t[j],

                "Experimental":
                    torque_true[j],

                "Direct_MLP":
                    torque_mlp[j],

                "Direct_XGB":
                    torque_xgb[j],

                "PG_ML":
                    torque_pg[j]

            })



metrics_df = pd.DataFrame(
    metric_records
)


points_df = pd.DataFrame(
    point_records
)


pg_params_df = pd.DataFrame(
    pg_parameter_records
)


save_csv(

    metrics_df,

    "curve_metrics_all_methods.csv"

)


save_csv(

    points_df,

    "pointwise_predictions.csv"

)


save_csv(

    pg_params_df,

    "pg_parameter_predictions.csv"

)


count_by_no = metrics_df.groupby("No")["Method"].nunique()
if not (count_by_no == len(METHODS)).all():
    raise RuntimeError("部分测试曲线未获得全部三种方法的预测，请检查输出")
if metrics_df[["R2", "RMSE", "MAE"]].isna().any().any():
    raise RuntimeError("部分曲线级评价指标缺失，停止生成结论")
print(f"曲线比较记录：{len(metrics_df)} = {len(count_by_no)}条曲线 × {len(METHODS)}种方法")


summary_df = (

    metrics_df

    .groupby(
        "Method",
        as_index=False
    )

    .agg(

        N_curves=(
            "No",
            "count"
        ),

        Mean_R2=(
            "R2",
            "mean"
        ),

        Median_R2=(
            "R2",
            "median"
        ),

        Std_R2=(
            "R2",
            "std"
        ),

        Mean_RMSE=(
            "RMSE",
            "mean"
        ),

        Median_RMSE=(
            "RMSE",
            "median"
        ),

        Std_RMSE=(
            "RMSE",
            "std"
        ),

        Mean_MAE=(
            "MAE",
            "mean"
        ),

        Median_MAE=(
            "MAE",
            "median"
        ),

        Std_MAE=(
            "MAE",
            "std"
        )

    )

)


summary_df[
    "Method"
] = pd.Categorical(

    summary_df[
        "Method"
    ],

    categories=
        METHODS,

    ordered=
        True

)


summary_df = (

    summary_df

    .sort_values(
        "Method"
    )

    .reset_index(
        drop=True
    )

)


summary_df[
    "Method"
] = (

    summary_df[
        "Method"
    ]

    .astype(str)

)


save_csv(

    summary_df,

    "method_summary.csv"

)



distribution_records = []


for method in METHODS:

    values = (

        metrics_df.loc[

            metrics_df[
                "Method"
            ]
            ==
            method,

            "RMSE"

        ]

        .dropna()

        .to_numpy(
            dtype=float
        )

    )


    distribution_records.append({

        "Method":
            method,

        "N_curves":
            len(values),

        "Mean_RMSE":
            np.mean(values),

        "Median_RMSE":
            np.median(values),

        "Std_RMSE":
            np.std(
                values,
                ddof=1
            ),

        "Q1_RMSE":
            np.percentile(
                values,
                25
            ),

        "Q3_RMSE":
            np.percentile(
                values,
                75
            ),

        "IQR_RMSE":
            (

                np.percentile(
                    values,
                    75
                )

                -

                np.percentile(
                    values,
                    25
                )

            ),

        "P90_RMSE":
            np.percentile(
                values,
                90
            ),

        "P95_RMSE":
            np.percentile(
                values,
                95
            ),

        "Max_RMSE":
            np.max(values),

        "Min_RMSE":
            np.min(values)

    })


distribution_df = pd.DataFrame(
    distribution_records
)


save_csv(

    distribution_df,

    "rmse_distribution_summary.csv"

)



def make_wide(
    metric
):

    return (

        metrics_df

        .pivot(

            index=[
                "Fold",
                "No"
            ],

            columns=
                "Method",

            values=
                metric

        )

        .reset_index()

    )


wide_rmse = make_wide(
    "RMSE"
)


wide_mae = make_wide(
    "MAE"
)


wide_r2 = make_wide(
    "R2"
)



paired_records = []

win_records = []


for direct_method in [

    "Direct-MLP",

    "Direct-XGB"

]:

    temp = wide_rmse[

        [
            "Fold",
            "No",
            direct_method,
            "PG-ML"
        ]

    ].dropna().copy()


    temp[
        "Delta_RMSE_Direct_minus_PG"
    ] = (

        temp[
            direct_method
        ]

        -

        temp[
            "PG-ML"
        ]

    )


    temp[
        "PG_lower_RMSE"
    ] = (

        temp[
            "Delta_RMSE_Direct_minus_PG"
        ]
        >
        0

    )



    win_records.append({

        "Direct_Method":
            direct_method,

        "N_curves":
            len(temp),

        "PG_lower_RMSE_count":
            int(

                temp[
                    "PG_lower_RMSE"
                ]

                .sum()

            ),

        "PG_lower_RMSE_fraction":
            float(

                temp[
                    "PG_lower_RMSE"
                ]

                .mean()

            ),

        "Mean_Delta_RMSE":
            float(

                temp[
                    "Delta_RMSE_Direct_minus_PG"
                ]

                .mean()

            ),

        "Median_Delta_RMSE":
            float(

                temp[
                    "Delta_RMSE_Direct_minus_PG"
                ]

                .median()

            )

    })


win_df = pd.DataFrame(
    win_records
)


save_csv(

    win_df,

    "paired_win_summary.csv"

)



pairwise_rows = []


for direct_method in [

    "Direct-MLP",

    "Direct-XGB"

]:

    temp = wide_rmse[

        [
            "Fold",
            "No",
            direct_method,
            "PG-ML"
        ]

    ].dropna()


    for _, row in temp.iterrows():

        delta = (

            row[
                direct_method
            ]

            -

            row[
                "PG-ML"
            ]

        )


        pairwise_rows.append({

            "Fold":
                row[
                    "Fold"
                ],

            "No":
                row[
                    "No"
                ],

            "Direct_Method":
                direct_method,

            "Direct_RMSE":
                row[
                    direct_method
                ],

            "PG_RMSE":
                row[
                    "PG-ML"
                ],

            "Delta_RMSE_Direct_minus_PG":
                delta,

            "PG_Better":
                delta > 0

        })


pairwise_df = pd.DataFrame(
    pairwise_rows
)


save_csv(

    pairwise_df,

    "rmse_pairwise_improvement.csv"

)



wilcoxon_records = []


for direct_method in [

    "Direct-MLP",

    "Direct-XGB"

]:

    for metric_name, wide in [

        (
            "R2",
            wide_r2
        ),

        (
            "RMSE",
            wide_rmse
        ),

        (
            "MAE",
            wide_mae
        )

    ]:

        direct_values = (

            wide[
                direct_method
            ]

            .to_numpy(
                dtype=float
            )

        )


        pg_values = (

            wide[
                "PG-ML"
            ]

            .to_numpy(
                dtype=float
            )

        )


        mask = (

            np.isfinite(
                direct_values
            )

            &

            np.isfinite(
                pg_values
            )

        )


        direct_values = (
            direct_values[
                mask
            ]
        )


        pg_values = (
            pg_values[
                mask
            ]
        )


        if len(
            direct_values
        ) == 0:

            statistic = np.nan
            p_value = np.nan


        elif np.allclose(

            direct_values,

            pg_values

        ):

            statistic = 0.0
            p_value = 1.0


        else:

            (
                statistic,
                p_value

            ) = wilcoxon(

                direct_values,

                pg_values,

                alternative=
                    "two-sided",

                zero_method=
                    "wilcox"

            )


        wilcoxon_records.append({

            "Direct_Method":
                direct_method,

            "Metric":
                metric_name,

            "N_pairs":
                len(
                    direct_values
                ),

            "Direct_mean":
                np.mean(
                    direct_values
                ),

            "PG_mean":
                np.mean(
                    pg_values
                ),

            "Wilcoxon_statistic":
                statistic,

            "p_value_two_sided":
                p_value

        })


wilcoxon_df = pd.DataFrame(
    wilcoxon_records
)


save_csv(

    wilcoxon_df,

    "paired_wilcoxon_tests.csv"

)



worst_records = []


for method in METHODS:

    temp = (

        metrics_df[

            metrics_df[
                "Method"
            ]
            ==
            method

        ]

        .sort_values(

            "RMSE",

            ascending=False

        )

        .head(
            10
        )

    )


    for rank, row in enumerate(

        temp.itertuples(),

        start=1

    ):

        worst_records.append({

            "Method":
                method,

            "Worst_Rank":
                rank,

            "No":
                row.No,

            "R2":
                row.R2,

            "RMSE":
                row.RMSE,

            "MAE":
                row.MAE

        })


worst_df = pd.DataFrame(
    worst_records
)


save_csv(

    worst_df,

    "worst_curves_by_method.csv"

)



plot_data = [

    metrics_df.loc[
        metrics_df[
            "Method"
        ]
        ==
        method,
        "RMSE"
    ]
    .dropna()
    .to_numpy(
        dtype=float
    )

    for method in METHODS

]


fig, ax = plt.subplots(

    figsize=(
        8.2,
        6.0
    )

)


box = ax.boxplot(

    plot_data,

    positions=[
        1,
        2,
        3
    ],

    widths=0.52,

    patch_artist=True,

    showmeans=True,

    showfliers=False,

    meanprops=dict(

        marker="D",

        markerfacecolor="white",

        markeredgecolor="#222222",

        markersize=6

    ),

    medianprops=dict(

        color="#222222",

        linewidth=2.0

    ),

    whiskerprops=dict(
        linewidth=1.3
    ),

    capprops=dict(
        linewidth=1.3
    )

)


for patch, method in zip(

    box[
        "boxes"
    ],

    METHODS

):

    patch.set_facecolor(

        COLORS[
            method
        ]

    )

    patch.set_alpha(
        0.28
    )

    patch.set_edgecolor(

        COLORS[
            method
        ]

    )

    patch.set_linewidth(
        1.6
    )



rng = np.random.default_rng(
    RANDOM_STATE
)


for position, method in enumerate(

    METHODS,

    start=1

):

    values = (

        metrics_df.loc[

            metrics_df[
                "Method"
            ]
            ==
            method,

            "RMSE"

        ]

        .dropna()

        .to_numpy(
            dtype=float
        )

    )


    jitter = rng.normal(

        loc=0,

        scale=0.055,

        size=len(values)

    )


    ax.scatter(

        np.full(
            len(values),
            position
        )
        +
        jitter,

        values,

        s=30,

        alpha=0.60,

        color=
            COLORS[
                method
            ],

        edgecolor="white",

        linewidth=0.35,

        zorder=3

    )



global_max = max(

    np.max(
        x
    )

    for x in plot_data

)


for position, method in enumerate(

    METHODS,

    start=1

):

    row = distribution_df[

        distribution_df[
            "Method"
        ]
        ==
        method

    ].iloc[0]


    ax.text(

        position,

        global_max
        *
        1.025,

        (
            f"Mean={row['Mean_RMSE']:.3f}\n"
            f"Median={row['Median_RMSE']:.3f}"
        ),

        ha="center",

        va="bottom",

        fontsize=8.5

    )


ax.set_xticks(
    [
        1,
        2,
        3
    ]
)


ax.set_xticklabels(

    METHODS,

    fontsize=11

)


ax.set_ylabel(

    "Curve-level RMSE",

    fontsize=12

)


ax.set_xlabel(

    "Prediction method",

    fontsize=12

)


ax.set_title(

    "Distribution of Curve-level Prediction Errors",

    fontsize=14,

    fontweight="bold",

    pad=14

)


ax.grid(

    axis="y",

    linestyle="--",

    linewidth=0.7,

    alpha=0.20

)


ax.spines[
    [
        "top",
        "right"
    ]
].set_visible(
    False
)


ax.set_ylim(

    0,

    global_max
    *
    1.18

)


ax.text(

    0.985,

    0.97,

    "Each point = one curing curve\n"
    "◆ = Mean",

    transform=
        ax.transAxes,

    ha="right",

    va="top",

    fontsize=8.5,

    bbox=dict(

        boxstyle=
            "round,pad=0.30",

        facecolor=
            "white",

        edgecolor=
            "#CCCCCC",

        alpha=
            0.88

    )

)


fig.tight_layout()


fig.savefig(

    OUT
    /
    "RMSE_distribution_box_scatter.png",

    dpi=400,

    bbox_inches="tight",

    facecolor="white"

)


fig.savefig(

    OUT
    /
    "RMSE_distribution_box_scatter.pdf",

    bbox_inches="tight",

    facecolor="white"

)


plt.close(
    fig
)



for direct_method in [

    "Direct-MLP",

    "Direct-XGB"

]:

    temp = wide_rmse[

        [
            direct_method,
            "PG-ML"
        ]

    ].dropna()


    x = temp[
        direct_method
    ].to_numpy(
        dtype=float
    )


    y = temp[
        "PG-ML"
    ].to_numpy(
        dtype=float
    )


    low = min(

        np.min(x),

        np.min(y)

    )


    high = max(

        np.max(x),

        np.max(y)

    )


    pad = max(

        (
            high
            -
            low
        )
        *
        0.05,

        0.1

    )


    fig, ax = plt.subplots(

        figsize=(
            6.2,
            5.8
        )

    )


    ax.scatter(

        x,

        y,

        s=44,

        alpha=0.75,

        color="#4C78A8",

        edgecolor="white",

        linewidth=0.4

    )


    ax.plot(

        [
            low - pad,
            high + pad
        ],

        [
            low - pad,
            high + pad
        ],

        "--",

        color="#555555",

        lw=1.5

    )


    ax.set_xlim(

        low - pad,

        high + pad

    )


    ax.set_ylim(

        low - pad,

        high + pad

    )


    ax.set_xlabel(

        f"{direct_method} RMSE",

        fontsize=11

    )


    ax.set_ylabel(

        "PG-ML RMSE",

        fontsize=11

    )


    ax.set_title(

        f"{direct_method} vs PG-ML",

        fontsize=13,

        fontweight="bold"

    )


    ax.text(

        0.04,

        0.96,

        "Below diagonal: PG-ML better",

        transform=
            ax.transAxes,

        va="top",

        fontsize=8.5

    )


    ax.spines[
        [
            "top",
            "right"
        ]
    ].set_visible(
        False
    )


    ax.grid(

        linestyle="--",

        alpha=0.15

    )


    fig.tight_layout()


    fig.savefig(

        OUT
        /
        (
            "RMSE_"
            +
            direct_method.replace(
                "-",
                "_"
            )
            +
            "_vs_PGML.png"
        ),

        dpi=400,

        bbox_inches="tight"

    )


    plt.close(
        fig
    )



fig, axes = plt.subplots(

    1,

    3,

    figsize=(
        13.2,
        4.5
    )

)


for i, (
    metric,
    ylabel
) in enumerate([

    (
        "Mean_R2",
        "Mean curve $R^2$"
    ),

    (
        "Mean_RMSE",
        "Mean curve RMSE"
    ),

    (
        "Mean_MAE",
        "Mean curve MAE"
    )

]):

    values = [

        float(

            summary_df.loc[

                summary_df[
                    "Method"
                ]
                ==
                method,

                metric

            ].iloc[0]

        )

        for method in METHODS

    ]


    colors = [

        COLORS[
            method
        ]

        for method in METHODS

    ]


    bars = axes[i].bar(

        METHODS,

        values,

        color=colors,

        alpha=0.82

    )


    axes[i].set_ylabel(
        ylabel
    )


    axes[i].spines[
        [
            "top",
            "right"
        ]
    ].set_visible(
        False
    )


    axes[i].tick_params(

        axis="x",

        rotation=20

    )


    axes[i].grid(

        axis="y",

        linestyle="--",

        alpha=0.15

    )


    for bar, value in zip(

        bars,

        values

    ):

        axes[i].text(

            bar.get_x()
            +
            bar.get_width()
            /
            2,

            bar.get_height(),

            f"{value:.3f}",

            ha="center",

            va="bottom",

            fontsize=8.5

        )


fig.suptitle(

    "Direct ML vs Physics-Guided ML",

    fontsize=14,

    fontweight="bold"

)


fig.tight_layout(

    rect=[
        0,
        0,
        1,
        0.94
    ]

)


fig.savefig(

    OUT
    /
    "method_mean_metrics.png",

    dpi=400,

    bbox_inches="tight"

)


plt.close(
    fig
)



pg_rank = (

    metrics_df[

        metrics_df[
            "Method"
        ]
        ==
        "PG-ML"

    ]

    .sort_values(
        "RMSE"
    )

    .reset_index(
        drop=True
    )

)


representatives = {

    "low_error":
        pg_rank.iloc[
            0
        ],

    "median_error":
        pg_rank.iloc[
            len(pg_rank)
            //
            2
        ],

    "high_error":
        pg_rank.iloc[
            -1
        ]

}



for label, selected in representatives.items():

    no = selected[
        "No"
    ]


    fold = selected[
        "Fold"
    ]


    sub = points_df[

        (
            points_df[
                "No"
            ]
            ==
            no
        )

        &

        (
            points_df[
                "Fold"
            ]
            ==
            fold
        )

    ].sort_values(
        "Time"
    )


    curve_metric = (

        metrics_df[

            (
                metrics_df[
                    "No"
                ]
                ==
                no
            )

            &

            (
                metrics_df[
                    "Fold"
                ]
                ==
                fold
            )

        ]

        .set_index(
            "Method"
        )

    )


    fig, ax = plt.subplots(

        figsize=(
            8.3,
            5.4
        )

    )


    # ========================================================
    # Experimental
    # ========================================================

    ax.plot(

        sub[
            "Time"
        ],

        sub[
            "Experimental"
        ],

        color=
            COLORS[
                "Experimental"
            ],

        lw=3.3,

        label=
            "Experimental"

    )


    # ========================================================
    # Direct-XGB
    # ========================================================

    ax.plot(

        sub[
            "Time"
        ],

        sub[
            "Direct_XGB"
        ],

        color=
            COLORS[
                "Direct-XGB"
            ],

        lw=2.6,

        ls="--",

        label=
            "Direct-XGB"

    )


    # ========================================================
    # Direct-MLP
    # ========================================================

    ax.plot(

        sub[
            "Time"
        ],

        sub[
            "Direct_MLP"
        ],

        color=
            COLORS[
                "Direct-MLP"
            ],

        lw=2.4,

        ls=":",

        label=
            "Direct-MLP"

    )


    # ========================================================
    # PG-ML
    # ========================================================

    ax.plot(

        sub[
            "Time"
        ],

        sub[
            "PG_ML"
        ],

        color=
            COLORS[
                "PG-ML"
            ],

        lw=2.9,

        ls="-.",

        label=
            "PG-ML"

    )



    info_lines = []


    for method in METHODS:

        info_lines.append(

            f"{method}: "
            f"R²="
            f"{curve_metric.loc[method, 'R2']:.3f}, "
            f"RMSE="
            f"{curve_metric.loc[method, 'RMSE']:.3f}"

        )


    ax.text(

        0.025,

        0.975,

        "\n".join(
            info_lines
        ),

        transform=
            ax.transAxes,

        va=
            "top",

        ha=
            "left",

        fontsize=
            8.5,

        linespacing=
            1.20,

        bbox=dict(

            boxstyle=
                "round,pad=0.28",

            facecolor=
                "white",

            edgecolor=
                "#AAAAAA",

            linewidth=
                0.7,

            alpha=
                0.86

        )

    )



    title_map = {

        "low_error":
            "Low-error Case",

        "median_error":
            "Median-error Case",

        "high_error":
            "High-error Case"

    }


    ax.set_title(

        (
            f"{title_map[label]} "
            f"- No. {no}"
        ),

        fontsize=14,

        fontweight="bold"

    )


    ax.set_xlabel(

        "Time (s)",

        fontsize=12

    )


    ax.set_ylabel(

        "Torque",

        fontsize=12

    )


    ax.legend(

        loc=
            "lower right",

        fontsize=
            8.5,

        frameon=
            True,

        framealpha=
            0.90,

        borderpad=
            0.4,

        labelspacing=
            0.35,

        handlelength=
            2.4

    )


    ax.grid(

        linestyle=
            "--",

        linewidth=
            0.7,

        alpha=
            0.16

    )


    ax.spines[
        [
            "top",
            "right"
        ]
    ].set_visible(
        False
    )


    fig.tight_layout()


    fig.savefig(

        OUT
        /
        f"representative_{label}.png",

        dpi=400,

        bbox_inches="tight",

        facecolor="white"

    )


    fig.savefig(

        OUT
        /
        f"representative_{label}.pdf",

        bbox_inches="tight",

        facecolor="white"

    )


    plt.close(
        fig
    )



print(
    "\n"
    +
    "=" * 90
)

print(
    "Overall Curve-level Performance"
)

print(
    "=" * 90
)

print(

    summary_df.to_string(
        index=False
    )

)


print(
    "\n"
    +
    "=" * 90
)

print(
    "RMSE Distribution"
)

print(
    "=" * 90
)

print(

    distribution_df.to_string(
        index=False
    )

)


print(
    "\n"
    +
    "=" * 90
)

print(
    "PG-ML Win Rate"
)

print(
    "=" * 90
)

print(

    win_df.to_string(
        index=False
    )

)


print(
    "\n"
    +
    "=" * 90
)

print(
    "Wilcoxon Paired Tests"
)

print(
    "=" * 90
)

print(

    wilcoxon_df.to_string(
        index=False
    )

)


print(
    "\n"
    +
    "=" * 90
)

print(
    "Stage 8 completed"
)

print(
    "=" * 90
)

print(
    f"\n输出目录：\n{OUT}"
)


print(
    """

主要结果文件：

1. method_summary.csv
   三种方法总体R² / RMSE / MAE

2. curve_metrics_all_methods.csv
   每条曲线三种方法的R² / RMSE / MAE

3. pointwise_predictions.csv
   每个测试时间点的预测结果

4. pg_parameter_predictions.csv
   PG-ML参数预测结果

5. paired_win_summary.csv
   PG-ML逐曲线胜率

6. paired_wilcoxon_tests.csv
   Wilcoxon配对显著性检验

7. rmse_distribution_summary.csv
   Mean / Median / Std / P90 / P95 / Max RMSE

8. RMSE_distribution_box_scatter.png
   RMSE箱线图 + 全部曲线散点

9. RMSE_Direct_MLP_vs_PGML.png
   Direct-MLP vs PG-ML逐曲线RMSE

10. RMSE_Direct_XGB_vs_PGML.png
    Direct-XGB vs PG-ML逐曲线RMSE

11. method_mean_metrics.png
    三种方法总体平均性能

12. representative_low_error.png
    PG-ML低误差代表曲线

13. representative_median_error.png
    PG-ML中位误差代表曲线

14. representative_high_error.png
    PG-ML高误差代表曲线

15. worst_curves_by_method.csv
    各方法RMSE最大的10条曲线

"""
)

print(
    "=" * 90
)

def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

manifest = {
    "stage": "Stage8",
    "analysis": "Direct MLP vs Direct XGB vs Physics-Guided ML",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "project_root": str(ROOT),
    "input_curve_file": str(CURVE_FILE),
    "input_curve_sha256": _sha256(CURVE_FILE),
    "input_dataset_file": str(STAGE3_FILE),
    "input_dataset_sha256": _sha256(STAGE3_FILE),
    "n_modeling_curves": len(stage3),
    "n_cv_folds_completed": len(splits),
    "n_test_curves_evaluated": len(count_by_no),
    "n_curve_method_records": len(metrics_df),
    "cv": f"KFold(n_splits={N_SPLITS},shuffle=True,random_state={RANDOM_STATE})",
    "direct_resampled_points_per_training_curve": DIRECT_TRAIN_POINTS_PER_CURVE,
    "PG_alpha0": "training-fold median",
    "PG_targets": PG_TARGETS,
    "validation_curves_used_for_training": False,
    "methods": METHODS,
    "note": "fold-limit小于5时仅为快速测试，不能作为正式科研结果"
}
(OUT / "stage8_run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
