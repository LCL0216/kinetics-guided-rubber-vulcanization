
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression



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


parser = argparse.ArgumentParser(description="Stage 7A: 88条建模曲线相关性分析")
parser.add_argument("--project-root", default=None, help="Mac上硫化曲线项目目录")
parser.add_argument("--input", default=None, help="指定 Stage3 数据集完整路径")
args = parser.parse_args()
BASE_DIR, INPUT_FILE = resolve_stage3_file(args.project_root, args.input)
OUTPUT_DIR = BASE_DIR / "Stage7A_Output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TOP_N_SCATTER = 8

ALPHA = 0.05



X_cols = [
    "mixing_time",
    "mixing_temp",
    "curing_temp",
    "curing_pressure",
    "rotor_speed_1",
    "rotor_speed_2",
    "curing_time_offset"
]

parameter_cols = [
    "ML",
    "MH",
    "ti",
    "tp",
    "alpha0",
    "b",
    "k1",
    "n1"
]

all_numeric_cols = X_cols + parameter_cols



df = pd.read_csv(INPUT_FILE)
validate_88(df, all_numeric_cols)
write_manifest(OUTPUT_DIR, INPUT_FILE, "Stage7A.py", {
    "analysis": "Pearson and Spearman with BH-FDR", "top_scatter": TOP_N_SCATTER,
    "fdr_alpha": ALPHA, "No_range": "1-88"
})

print("=" * 80)
print("Stage 7A：描述性统计与相关性分析")
print("=" * 80)
print(f"原始样本数：{len(df)}")


missing_cols = [
    col for col in all_numeric_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"数据中缺少以下字段：{missing_cols}"
    )


data = df[all_numeric_cols].copy()

for col in all_numeric_cols:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


print(f"分析变量数：{len(all_numeric_cols)}")



stats_records = []

for col in all_numeric_cols:

    x = data[col].dropna()

    n = len(x)
    missing = data[col].isna().sum()

    mean = x.mean()
    std = x.std(ddof=1)
    median = x.median()

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)

    min_val = x.min()
    max_val = x.max()

    if mean > 0 and abs(mean) > 1e-12:
        cv = std / mean * 100
    else:
        cv = np.nan

    stats_records.append({
        "Variable": col,
        "N": n,
        "Missing": missing,
        "Mean": mean,
        "SD": std,
        "Median": median,
        "Q1": q1,
        "Q3": q3,
        "Min": min_val,
        "Max": max_val,
        "Range": max_val - min_val,
        "CV_percent": cv
    })


stats_df = pd.DataFrame(stats_records)

stats_path = OUTPUT_DIR / "descriptive_statistics.csv"

stats_df.to_csv(
    stats_path,
    index=False,
    encoding="utf-8-sig"
)


print("\n描述性统计：")

print(
    stats_df.to_string(
        index=False
    )
)


# ============================================================
# 5. Benjamini-Hochberg FDR
# ============================================================

def benjamini_hochberg(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(p_values)

    order = np.argsort(p_values)

    ranked_p = p_values[order]

    q_values = np.empty(n)

    previous_q = 1.0

    for i in range(
        n - 1,
        -1,
        -1
    ):

        rank = i + 1

        q = ranked_p[i] * n / rank

        q = min(
            q,
            previous_q,
            1.0
        )

        q_values[
            order[i]
        ] = q

        previous_q = q

    return q_values



def cross_correlation_analysis(
    df,
    row_cols,
    col_cols,
    method="pearson"
):

    corr_matrix = pd.DataFrame(
        index=row_cols,
        columns=col_cols,
        dtype=float
    )

    p_matrix = pd.DataFrame(
        index=row_cols,
        columns=col_cols,
        dtype=float
    )

    records = []

    for x_col in row_cols:

        for y_col in col_cols:

            temp = df[
                [x_col, y_col]
            ].dropna()

            x = temp[x_col].to_numpy()
            y = temp[y_col].to_numpy()

            if len(temp) < 3:

                corr = np.nan
                p = np.nan

            elif (
                np.std(x) == 0
                or
                np.std(y) == 0
            ):

                corr = np.nan
                p = np.nan

            else:

                if method == "pearson":

                    corr, p = pearsonr(
                        x,
                        y
                    )

                elif method == "spearman":

                    corr, p = spearmanr(
                        x,
                        y
                    )

                else:

                    raise ValueError(
                        "method必须为pearson或spearman"
                    )

            corr_matrix.loc[
                x_col,
                y_col
            ] = corr

            p_matrix.loc[
                x_col,
                y_col
            ] = p

            records.append({
                "X": x_col,
                "Y": y_col,
                "Correlation": corr,
                "P_value": p,
                "N": len(temp)
            })


    result_df = pd.DataFrame(
        records
    )


    valid_mask = (
        result_df["P_value"]
        .notna()
    )


    result_df[
        "FDR_q"
    ] = np.nan


    if valid_mask.sum() > 0:

        result_df.loc[
            valid_mask,
            "FDR_q"
        ] = benjamini_hochberg(
            result_df.loc[
                valid_mask,
                "P_value"
            ].to_numpy()
        )


    result_df[
        "FDR_significant"
    ] = (
        result_df["FDR_q"]
        <
        ALPHA
    )


    q_matrix = pd.DataFrame(
        index=row_cols,
        columns=col_cols,
        dtype=float
    )


    for _, row in result_df.iterrows():

        q_matrix.loc[
            row["X"],
            row["Y"]
        ] = row["FDR_q"]


    return (
        corr_matrix,
        p_matrix,
        q_matrix,
        result_df
    )


# ============================================================
# 7. X vs Parameters：Pearson
# ============================================================

(
    pearson_x_param,
    pearson_x_param_p,
    pearson_x_param_q,
    pearson_x_param_long
) = cross_correlation_analysis(
    data,
    X_cols,
    parameter_cols,
    method="pearson"
)


pearson_x_param.to_csv(
    OUTPUT_DIR /
    "pearson_X_vs_parameters.csv",
    encoding="utf-8-sig"
)

pearson_x_param_p.to_csv(
    OUTPUT_DIR /
    "pearson_X_vs_parameters_pvalues.csv",
    encoding="utf-8-sig"
)

pearson_x_param_q.to_csv(
    OUTPUT_DIR /
    "pearson_X_vs_parameters_FDR.csv",
    encoding="utf-8-sig"
)

pearson_x_param_long.to_csv(
    OUTPUT_DIR /
    "pearson_X_vs_parameters_long.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. X vs Parameters：Spearman
# ============================================================

(
    spearman_x_param,
    spearman_x_param_p,
    spearman_x_param_q,
    spearman_x_param_long
) = cross_correlation_analysis(
    data,
    X_cols,
    parameter_cols,
    method="spearman"
)


spearman_x_param.to_csv(
    OUTPUT_DIR /
    "spearman_X_vs_parameters.csv",
    encoding="utf-8-sig"
)

spearman_x_param_p.to_csv(
    OUTPUT_DIR /
    "spearman_X_vs_parameters_pvalues.csv",
    encoding="utf-8-sig"
)

spearman_x_param_q.to_csv(
    OUTPUT_DIR /
    "spearman_X_vs_parameters_FDR.csv",
    encoding="utf-8-sig"
)

spearman_x_param_long.to_csv(
    OUTPUT_DIR /
    "spearman_X_vs_parameters_long.csv",
    index=False,
    encoding="utf-8-sig"
)



pearson_param_matrix = (
    data[
        parameter_cols
    ]
    .corr(
        method="pearson"
    )
)

spearman_param_matrix = (
    data[
        parameter_cols
    ]
    .corr(
        method="spearman"
    )
)


pearson_param_matrix.to_csv(
    OUTPUT_DIR /
    "pearson_parameter_matrix.csv",
    encoding="utf-8-sig"
)

spearman_param_matrix.to_csv(
    OUTPUT_DIR /
    "spearman_parameter_matrix.csv",
    encoding="utf-8-sig"
)



parameter_pair_records = []

for i in range(
    len(parameter_cols)
):

    for j in range(
        i + 1,
        len(parameter_cols)
    ):

        x_col = parameter_cols[i]
        y_col = parameter_cols[j]

        temp = data[
            [x_col, y_col]
        ].dropna()

        x = temp[x_col]
        y = temp[y_col]

        if (
            len(temp) >= 3
            and
            x.std() > 0
            and
            y.std() > 0
        ):

            pr, pp = pearsonr(
                x,
                y
            )

            sr, sp = spearmanr(
                x,
                y
            )

        else:

            pr = pp = np.nan
            sr = sp = np.nan


        parameter_pair_records.append({
            "Variable_1": x_col,
            "Variable_2": y_col,
            "Pearson_r": pr,
            "Pearson_p": pp,
            "Spearman_rho": sr,
            "Spearman_p": sp,
            "N": len(temp)
        })


parameter_pair_df = pd.DataFrame(
    parameter_pair_records
)


valid = parameter_pair_df[
    "Pearson_p"
].notna()

parameter_pair_df[
    "Pearson_FDR_q"
] = np.nan

parameter_pair_df.loc[
    valid,
    "Pearson_FDR_q"
] = benjamini_hochberg(
    parameter_pair_df.loc[
        valid,
        "Pearson_p"
    ]
)


valid = parameter_pair_df[
    "Spearman_p"
].notna()

parameter_pair_df[
    "Spearman_FDR_q"
] = np.nan

parameter_pair_df.loc[
    valid,
    "Spearman_FDR_q"
] = benjamini_hochberg(
    parameter_pair_df.loc[
        valid,
        "Spearman_p"
    ]
)


parameter_pair_df[
    "Abs_Spearman"
] = (
    parameter_pair_df[
        "Spearman_rho"
    ].abs()
)


parameter_pair_df = (
    parameter_pair_df
    .sort_values(
        "Abs_Spearman",
        ascending=False
    )
)


parameter_pair_df.to_csv(
    OUTPUT_DIR /
    "parameter_pair_correlations.csv",
    index=False,
    encoding="utf-8-sig"
)



def plot_heatmap(
    matrix,
    title,
    output_path,
    figsize=(11, 6)
):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    values = matrix.to_numpy(
        dtype=float
    )

    im = ax.imshow(
        values,
        vmin=-1,
        vmax=1,
        aspect="auto",
        cmap="coolwarm"
    )


    ax.set_xticks(
        np.arange(
            len(matrix.columns)
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right"
    )


    ax.set_yticks(
        np.arange(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        matrix.index
    )


    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            value = values[i, j]

            if np.isnan(value):
                text = ""
            else:
                text = f"{value:.2f}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8
            )


    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Correlation coefficient"
    )


    ax.set_title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



plot_heatmap(
    pearson_x_param,
    "Pearson Correlation: Process Variables vs Model Parameters",
    OUTPUT_DIR /
    "heatmap_Pearson_X_vs_parameters.png",
    figsize=(11, 6)
)


plot_heatmap(
    spearman_x_param,
    "Spearman Correlation: Process Variables vs Model Parameters",
    OUTPUT_DIR /
    "heatmap_Spearman_X_vs_parameters.png",
    figsize=(11, 6)
)



plot_heatmap(
    pearson_param_matrix,
    "Pearson Correlation among Model Parameters",
    OUTPUT_DIR /
    "heatmap_Pearson_parameters.png",
    figsize=(8, 7)
)


plot_heatmap(
    spearman_param_matrix,
    "Spearman Correlation among Model Parameters",
    OUTPUT_DIR /
    "heatmap_Spearman_parameters.png",
    figsize=(8, 7)
)



key_df = (
    spearman_x_param_long
    .copy()
)


key_df = key_df.rename(
    columns={
        "Correlation":
            "Spearman_rho",
        "P_value":
            "Spearman_p",
        "FDR_q":
            "Spearman_FDR_q",
        "FDR_significant":
            "Spearman_significant"
    }
)


pearson_lookup = (
    pearson_x_param_long[
        [
            "X",
            "Y",
            "Correlation",
            "P_value",
            "FDR_q"
        ]
    ]
    .rename(
        columns={
            "Correlation":
                "Pearson_r",
            "P_value":
                "Pearson_p",
            "FDR_q":
                "Pearson_FDR_q"
        }
    )
)


key_df = key_df.merge(
    pearson_lookup,
    on=["X", "Y"],
    how="left"
)


key_df[
    "Abs_Spearman"
] = (
    key_df[
        "Spearman_rho"
    ].abs()
)


key_df = (
    key_df
    .sort_values(
        "Abs_Spearman",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


key_df.to_csv(
    OUTPUT_DIR /
    "ranked_process_parameter_correlations.csv",
    index=False,
    encoding="utf-8-sig"
)



significant_df = key_df[
    key_df[
        "Spearman_FDR_q"
    ] < ALPHA
].copy()


significant_df.to_csv(
    OUTPUT_DIR /
    "significant_process_parameter_correlations.csv",
    index=False,
    encoding="utf-8-sig"
)



def plot_scatter_relationship(
    df,
    x_col,
    y_col,
    pearson_r,
    pearson_p,
    spearman_rho,
    spearman_q,
    output_path
):

    temp = df[
        [x_col, y_col]
    ].dropna()

    x = temp[
        x_col
    ].to_numpy()

    y = temp[
        y_col
    ].to_numpy()


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    ax.scatter(
        x,
        y,
        s=45,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4
    )



    if (
        len(x) >= 3
        and
        np.std(x) > 0
    ):

        model = LinearRegression()

        model.fit(
            x.reshape(-1, 1),
            y
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            200
        )

        y_line = model.predict(
            x_line.reshape(-1, 1)
        )

        ax.plot(
            x_line,
            y_line,
            linewidth=2
        )


    ax.set_xlabel(
        x_col
    )

    ax.set_ylabel(
        y_col
    )


    ax.set_title(
        f"{x_col} vs {y_col}"
    )


    text = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"Spearman ρ = {spearman_rho:.3f}\n"
        f"FDR q = {spearman_q:.3g}"
    )


    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            alpha=0.15
        )
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



scatter_dir = (
    OUTPUT_DIR /
    "Key_Scatter_Plots"
)

scatter_dir.mkdir(
    exist_ok=True
)


top_scatter_df = key_df.head(
    TOP_N_SCATTER
).copy()


for rank, row in top_scatter_df.iterrows():

    x_col = row["X"]
    y_col = row["Y"]

    filename = (
        f"{rank + 1:02d}_"
        f"{x_col}_vs_{y_col}.png"
    )

    plot_scatter_relationship(
        data,
        x_col,
        y_col,
        row["Pearson_r"],
        row["Pearson_p"],
        row["Spearman_rho"],
        row["Spearman_FDR_q"],
        scatter_dir /
        filename
    )



manual_pairs = [
    ("ML", "MH"),
    ("ti", "tp"),
    ("b", "k1"),
    ("ti", "k1"),
    ("tp", "k1"),
    ("tp", "n1")
]


manual_dir = (
    OUTPUT_DIR /
    "Parameter_Scatter_Plots"
)

manual_dir.mkdir(
    exist_ok=True
)


for x_col, y_col in manual_pairs:

    if (
        x_col not in data.columns
        or
        y_col not in data.columns
    ):
        continue


    temp = data[
        [x_col, y_col]
    ].dropna()


    if (
        len(temp) < 3
        or
        temp[x_col].std() == 0
        or
        temp[y_col].std() == 0
    ):
        continue


    pr, pp = pearsonr(
        temp[x_col],
        temp[y_col]
    )

    sr, sp = spearmanr(
        temp[x_col],
        temp[y_col]
    )


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    ax.scatter(
        temp[x_col],
        temp[y_col],
        s=45,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4
    )


    model = LinearRegression()

    model.fit(
        temp[[x_col]],
        temp[y_col]
    )


    x_line = np.linspace(
        temp[x_col].min(),
        temp[x_col].max(),
        200
    )

    y_line = model.predict(
        pd.DataFrame(
            {
                x_col: x_line
            }
        )
    )


    ax.plot(
        x_line,
        y_line,
        linewidth=2
    )


    ax.set_xlabel(
        x_col
    )

    ax.set_ylabel(
        y_col
    )

    ax.set_title(
        f"{x_col} vs {y_col}"
    )


    ax.text(
        0.04,
        0.96,
        (
            f"Pearson r = {pr:.3f}\n"
            f"Spearman ρ = {sr:.3f}\n"
            f"p = {sp:.3g}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            alpha=0.15
        )
    )


    plt.tight_layout()

    plt.savefig(
        manual_dir /
        f"{x_col}_vs_{y_col}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



print("\n" + "=" * 80)
print("工艺参数 → 模型参数：Spearman绝对值最高的关系")
print("=" * 80)

display_cols = [
    "X",
    "Y",
    "Pearson_r",
    "Spearman_rho",
    "Spearman_FDR_q"
]

print(
    key_df[
        display_cols
    ]
    .head(20)
    .to_string(
        index=False
    )
)



print("\n" + "=" * 80)
print(f"FDR q < {ALPHA} 的显著关系")
print("=" * 80)

if len(significant_df) == 0:

    print(
        "没有发现FDR校正后显著关系。"
    )

else:

    print(
        significant_df[
            display_cols
        ]
        .to_string(
            index=False
        )
    )



print("\n" + "=" * 80)
print("模型参数之间Spearman相关性最高的关系")
print("=" * 80)

print(
    parameter_pair_df[
        [
            "Variable_1",
            "Variable_2",
            "Pearson_r",
            "Spearman_rho",
            "Spearman_FDR_q"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)



print("\n" + "=" * 80)
print("Stage 7A 完成")
print("=" * 80)

print(
    """
    )

主要文件：

1. descriptive_statistics.csv
   → 所有工艺变量和动力学参数描述性统计

2. pearson_X_vs_parameters.csv
   → 工艺参数与模型参数 Pearson 相关系数

3. spearman_X_vs_parameters.csv
   → 工艺参数与模型参数 Spearman 相关系数

4. pearson_X_vs_parameters_FDR.csv
   → Pearson多重检验FDR校正

5. spearman_X_vs_parameters_FDR.csv
   → Spearman多重检验FDR校正

6. ranked_process_parameter_correlations.csv
   → 所有工艺-参数关系按照|Spearman|排序

7. significant_process_parameter_correlations.csv
   → FDR q < 0.05的显著关系

8. parameter_pair_correlations.csv
   → ML/MH/ti/tp/alpha0/b/k1/n1之间的相关性

9. heatmap_Pearson_X_vs_parameters.png

10. heatmap_Spearman_X_vs_parameters.png

11. heatmap_Pearson_parameters.png

12. heatmap_Spearman_parameters.png

13. Key_Scatter_Plots/
    → 自动选出的最强工艺-参数关系散点图

14. Parameter_Scatter_Plots/
    → ML-MH、ti-tp、b-k1等关键参数关系
"""
)