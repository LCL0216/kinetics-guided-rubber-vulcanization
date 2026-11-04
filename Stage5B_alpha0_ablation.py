#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXPECTED_COUNT = 88
EXPECTED_MEAN_RMSE_B = 1.595846


def parse_args():
    p = argparse.ArgumentParser(description="Standalone alpha0=0 vs fold-median ablation using saved Stage8 predictions")
    p.add_argument("--project-root", type=Path, default=Path.cwd(), help="Working folder containing Stage8 output")
    p.add_argument("--stage8-source", type=Path, default=None, help="Stage8_Output directory or ZIP containing output files")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: project-root/Stage5_alpha0_Ablation")
    p.add_argument("--expected-rmse", type=float, default=EXPECTED_MEAN_RMSE_B, help="Version check against selected final Stage8 run")
    p.add_argument("--allow-other-version", action="store_true", help="Allow different final Stage8 version after explicit acknowledgement")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true", help="Export CSV only")
    p.add_argument("--no-pdf", action="store_true", help="Do not also save PDF plots")
    return p.parse_args()


def select_source(args):
    if args.stage8_source is not None:
        source = args.stage8_source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"所指定的Stage 8文件不存在：{source}")
        return source
    root = args.project_root.expanduser().resolve()
    for cand in (root / "Stage8_Output(1).zip", root / "Stage8_Output(1)",
                 root / "Stage8_Output", root / "Stage8_Output.zip"):
        if cand.exists():
            return cand
    raise FileNotFoundError("未找到Stage8输出，请通过 --stage8-source 指定Stage8_Output(1).zip或文件夹")


class InputFiles:
    def __init__(self, path):
        self.path = path
        self.zf = ZipFile(path) if path.is_file() else None
        if self.zf:
            all_names = self.zf.namelist()
            self.members = {Path(name).name: name for name in all_names
                            if name.endswith(".csv") and not name.startswith("__MACOSX/")
                            and not Path(name).name.startswith("._")}
        else:
            self.members = {}
            for f in path.rglob("*.csv"):
                if not f.name.startswith("._"):
                    self.members.setdefault(f.name, f)

    def bytes(self, basename, required=True):
        member = self.members.get(basename)
        if member is None:
            if required:
                raise FileNotFoundError(f"Stage 8输出缺少 {basename}（所在输入：{self.path}）")
            return None
        if self.zf:
            return self.zf.read(member)
        return Path(member).read_bytes()

    def read_csv(self, basename, required=True):
        raw = self.bytes(basename, required=required)
        if raw is None:
            return None
        return pd.read_csv(pd.io.common.BytesIO(raw))

    def close(self):
        if self.zf:
            self.zf.close()


def alpha_response(t, ti, k1, n1, alpha0):
    t = np.asarray(t, dtype=float)
    alpha = np.empty_like(t)
    mask = t <= ti
    alpha[mask] = alpha0 * np.clip(t[mask] / max(ti, 1e-12), 0.0, 1.0)
    tau = np.maximum(t[~mask] - ti, 0.0)
    alpha[~mask] = alpha0 + (1.0 - alpha0) * (1.0 - np.exp(-k1 * np.power(tau, n1)))
    return np.clip(alpha, 0.0, 1.0)


def scores(observed, prediction):
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    resid = pred - obs
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def wilcoxon_p(paired_delta):
    delta = np.asarray(paired_delta, dtype=float)
    if np.allclose(delta, 0.0):
        return 1.0
    try:
        # Two-sided paired test, as described in the manuscript and report.
        return float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def write_plot(fig, stem, output, args):
    png = output / f"{stem}.png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax):
    """Publication-oriented, lightweight axes styling."""
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(axis="both", colors="#374151", labelsize=10, pad=5)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.33, linewidth=0.75)


def _box_scatter(ax, data_zero, data_med):
    """Show the distribution and individual curves; median lines are from raw data."""
    bp = ax.boxplot(
        [data_zero, data_med], positions=(1, 2), widths=0.45,
        patch_artist=True, showfliers=False, whis=1.5,
        medianprops={"color": "#182D43", "linewidth": 2.0},
        boxprops={"edgecolor": "#7D8796", "linewidth": 1.0},
        whiskerprops={"color": "#8492A6", "linewidth": 1.0},
        capprops={"color": "#8492A6", "linewidth": 1.0},
    )
    for patch, col in zip(bp["boxes"], ("#EFC78F", "#9AD0DF")):
        patch.set_facecolor(col)
        patch.set_alpha(0.63)
    rng = np.random.default_rng(42)
    for pos, values, col in ((1, data_zero, "#C88B42"),
                              (2, data_med, "#287FA4")):
        x = rng.normal(pos, 0.054, size=len(values))
        ax.scatter(x, values, s=22, alpha=0.70, color=col,
                   linewidths=0.4, edgecolors="white", zorder=3)
    ax.set_xlim(0.55, 2.45)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([r"$\alpha_0 = 0$", "Training-fold\nmedian " + r"$\alpha_0$"])


def create_plots(paired, examples, output, args):
    """Create 3 reproducible figures; preserve all numerical predictions."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    sub = paired.loc[paired["Version"] == "B_pred_ML_MH"].copy()
    rmse_zero = sub["RMSE_zero"].to_numpy(float)
    rmse_median = sub["RMSE_median"].to_numpy(float)
    reductions = sub["RMSE_reduction"].to_numpy(float)
    n_improved = int(np.count_nonzero(reductions > 0))
    change_pct = 100 * (rmse_zero.mean() - rmse_median.mean()) / rmse_zero.mean()

    # FIGURE 1: RMSE distribution and paired improvement histogram.
    # Use a fixed top/bottom margin instead of constrained_layout + y > 1 suptitle.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.205,
                        top=0.805, wspace=0.29)
    fig.suptitle(r"Effect of $\alpha_0$ treatment on fully forward Version B",
                 fontsize=16, weight="bold", color="#1D2F43", y=0.965)

    ax = axes[0]
    _style_axis(ax)
    _box_scatter(ax, rmse_zero, rmse_median)
    ax.set_title("(a)  Curve-level RMSE distribution", fontsize=12.5,
                 weight="semibold", color="#1D2F43", pad=15)
    ax.set_ylabel(r"RMSE (kgf$\cdot$cm)", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(rmse_zero.max(), rmse_median.max()) * 1.10)

    ax = axes[1]
    _style_axis(ax)
    # Same 88 paired curves, original reductions unchanged.
    ax.hist(reductions, bins=17, color="#40A59A", edgecolor="white",
            linewidth=1.2, alpha=0.9)
    ax.axvline(0, color="#C24D4D", linestyle="--", linewidth=1.6,
               label="No change")
    ax.set_title("(b)  Per-curve RMSE reduction", fontsize=12.5,
                 weight="semibold", color="#1D2F43", pad=15)
    ax.set_xlabel(r"RMSE($\alpha_0=0$) $-$ RMSE(median $\alpha_0$)",
                  fontsize=10.5, labelpad=9)
    ax.set_ylabel("Number of curves", fontsize=11, labelpad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    footer = (f"Mean RMSE: {rmse_zero.mean():.3f}  \u2192  {rmse_median.mean():.3f}  "
              f"({change_pct:.2f}% lower)    |    "
              f"Improved: {n_improved}/{len(sub)} curves    |    "
              "Positive reduction favors median $\\alpha_0$")
    fig.text(0.5, 0.045, footer, ha="center", va="center",
             fontsize=10.4, color="#334155")
    write_plot(fig, "Fig_Alpha0_Ablation_Intuitive", output, args)

    # FIGURE 2: Exactly the three requested averaged metrics.
    # Separate panel titles, value labels, and explanatory footer.
    metrics = [
        ("(a)  Mean R$^2$", "Mean R$^2$", float(sub["R2_zero"].mean()),
         float(sub["R2_median"].mean())),
        ("(b)  Mean RMSE", r"RMSE (kgf$\cdot$cm)",
         float(rmse_zero.mean()), float(rmse_median.mean())),
        ("(c)  Mean MAE", r"MAE (kgf$\cdot$cm)",
         float(sub["MAE_zero"].mean()), float(sub["MAE_median"].mean())),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 5.2))
    fig.subplots_adjust(left=0.073, right=0.978, bottom=0.205,
                        top=0.795, wspace=0.31)
    fig.suptitle(r"Comparison of prediction metrics under two $\alpha_0$ settings",
                 fontsize=15.5, weight="bold", color="#1D2F43", y=0.970)

    for ax, (panel_title, ylabel, zero_val, median_val) in zip(axes, metrics):
        _style_axis(ax)
        vals = [zero_val, median_val]
        bars = ax.bar([0, 1], vals, width=0.56,
                      color=["#E5B16F", "#5FA8C6"],
                      edgecolor=["#CB9257", "#3986A4"],
                      linewidth=1.0, zorder=3)
        ax.set_xlim(-0.65, 1.65)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$\alpha_0 = 0$", "Training-fold\nmedian " + r"$\alpha_0$"])
        ax.set_ylabel(ylabel, fontsize=10.8, labelpad=9)
        ax.set_title(panel_title, fontsize=12.5, color="#1D2F43",
                     weight="semibold", pad=15)
        # R2 difference is small; use 4 decimals to avoid hiding the change.
        y_top = max(vals) * 1.21
        if "Mean R$^2$" in panel_title:
            y_top = max(1.10, y_top)
        ax.set_ylim(0, y_top)
        fmt = ".4f" if "Mean R$^2$" in panel_title else ".3f"
        for bar, val in zip(bars, vals):
            ax.annotate(format(val, fmt),
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 8), textcoords="offset points",
                        fontsize=10.2, weight="semibold", color="#22364B",
                        ha="center", va="bottom")

    fig.text(0.5, 0.046,
             f"Version B (88 curves)  |  RMSE reduced by {change_pct:.2f}%  "
             f"|  R$^2$ increased by {metrics[0][3]-metrics[0][2]:.4f}",
             fontsize=10.5, ha="center", va="center", color="#334155")
    write_plot(fig, "Fig_Alpha0_Metric_Bars", output, args)

    # FIGURE 3: experimental and reconstructed curves, same selection as v2.
    median_delta = float(sub["RMSE_reduction"].median())
    typical = int(sub.loc[(sub["RMSE_reduction"] - median_delta).abs().idxmin(), "No"])
    most_improved = int(sub.loc[sub["RMSE_reduction"].idxmax(), "No"])
    show = list(dict.fromkeys((typical, most_improved)))
    fig, axes = plt.subplots(1, len(show), figsize=(6.3 * len(show), 4.7))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.16,
                        top=0.88, wspace=0.20)
    if len(show) == 1:
        axes = [axes]
    for ax, no in zip(axes, show):
        _style_axis(ax)
        obj = examples[no]
        ax.plot(obj["t"], obj["y"], color="#24313D", lw=2.2,
                label="Experimental")
        ax.plot(obj["t"], obj["median"], color="#3189BB", ls="--", lw=1.9,
                label=r"Training-fold median $\alpha_0$")
        ax.plot(obj["t"], obj["zero"], color="#CD7853", ls=":", lw=2.0,
                label=r"$\alpha_0=0$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Torque (kgf$\cdot$cm)")
        ax.set_title(f"Curve {no}", fontsize=12.5, pad=12)
        ax.legend(loc="lower right", frameon=True, fontsize=8)
    write_plot(fig, "Fig_Alpha0_Representative_Curves", output, args)

    return {"median_delta_curve_No": typical,
            "most_improved_curve_No": most_improved}


def main():
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    source = select_source(args)
    out = args.output_dir.expanduser().resolve() if args.output_dir else root / "Stage5_alpha0_Ablation"

    files = InputFiles(source)
    try:
        parameters = files.read_csv("pg_parameter_predictions.csv")
        points = files.read_csv("pointwise_predictions.csv")
        archived_summary = files.read_csv("method_summary.csv", required=False)
        hash_prm = hashlib.sha256(files.bytes("pg_parameter_predictions.csv")).hexdigest()
        hash_pts = hashlib.sha256(files.bytes("pointwise_predictions.csv")).hexdigest()
    finally:
        files.close()

    req_param = ["No", "Fold", "alpha0_train_median", "ML_true", "ML_pred", "MH_true", "MH_pred",
                 "ti_pred", "k1_pred", "n1_pred"]
    req_pts = ["No", "Fold", "Time", "Experimental", "PG_ML"]
    for label, df, columns in (("pg_parameter_predictions.csv", parameters, req_param),
                               ("pointwise_predictions.csv", points, req_pts)):
        miss = [name for name in columns if name not in df.columns]
        if miss:
            raise ValueError(f"{label}缺少列：{miss}")
        if df[columns].isna().any().any():
            raise ValueError(f"{label}存在缺失值")

    nos = parameters.No.astype(int)
    if len(parameters) != EXPECTED_COUNT or nos.nunique() != EXPECTED_COUNT or sorted(nos) != list(range(1, 89)):
        raise ValueError("该实验限定88条建模曲线，编号需为1–88且无重复")
    if points.No.nunique() != EXPECTED_COUNT or points.duplicated(["No", "Time"]).any():
        raise ValueError("逐点预测结果的编号数或时间点存在异常")

    pgroup = {int(no): grp.sort_values("Time") for no, grp in points.groupby("No")}
    if set(pgroup) != set(range(1, 89)):
        raise ValueError("参数表和逐点预测文件的曲线编号无法完整对应")

    rows = []
    shown = {}
    largest_pointwise_diff = 0.0

    for row in parameters.itertuples(index=False):
        prm = row._asdict()
        no = int(prm["No"])
        data = pgroup[no]
        t = data.Time.to_numpy(float)
        actual = data.Experimental.to_numpy(float)

        if data.Fold.nunique() != 1 or int(data.Fold.iloc[0]) != int(prm["Fold"]):
            raise ValueError(f"No {no} 的交叉验证折不匹配")
        if len(t) < 5 or not np.isfinite(t).all() or not np.isfinite(actual).all():
            raise ValueError(f"No {no} 的实验数据不完整")

        ti, k1, n1 = (float(prm[key]) for key in ("ti_pred", "k1_pred", "n1_pred"))
        ml_pred = float(prm["ML_pred"])
        mh_pred = max(float(prm["MH_pred"]), ml_pred + 1e-6)
        ml_true, mh_true = (float(prm[key]) for key in ("ML_true", "MH_true"))
        a0 = float(prm["alpha0_train_median"])
        if not 0 <= a0 <= 1:
            raise ValueError(f"No {no} 的训练折 alpha0 不在0–1内")

        alpha0_zero = alpha_response(t, ti, k1, n1, 0.0)
        alpha0_median = alpha_response(t, ti, k1, n1, a0)

        b_zero = ml_pred + (mh_pred - ml_pred) * alpha0_zero
        b_median = ml_pred + (mh_pred - ml_pred) * alpha0_median

        diff = float(np.max(np.abs(b_median - data.PG_ML.to_numpy(float))))
        largest_pointwise_diff = max(largest_pointwise_diff, diff)
        if diff > 1e-6:
            raise ValueError(f"No {no} 的重构结果与当前Stage8原有PG_ML不一致，差值 {diff:.5g}")

        a_zero = ml_true + (mh_true - ml_true) * alpha0_zero
        a_median = ml_true + (mh_true - ml_true) * alpha0_median

        for mode, a0used, A, B in (("zero", 0.0, a_zero, b_zero),
                                   ("train_fold_median", a0, a_median, b_median)):
            for version, curve in (("A_true_ML_MH", A), ("B_pred_ML_MH", B)):
                rows.append({
                    "No": no, "Fold": int(prm["Fold"]), "alpha0_mode": mode,
                    "alpha0_used": a0used, "Version": version, "n_points": len(t),
                    **scores(actual, curve)
                })
        shown[no] = {"t": t, "y": actual, "median": b_median, "zero": b_zero}

    ablation = pd.DataFrame(rows).sort_values(["No", "Version", "alpha0_mode"]).reset_index(drop=True)
    summary = (ablation.groupby(["alpha0_mode", "Version"], as_index=False)
               .agg(N=("No", "size"),
                    Mean_R2=("R2", "mean"), Median_R2=("R2", "median"), Std_R2=("R2", "std"),
                    Mean_RMSE=("RMSE", "mean"), Median_RMSE=("RMSE", "median"),
                    Mean_MAE=("MAE", "mean"), Median_MAE=("MAE", "median")))

    b_mean = float(summary[(summary.Version == "B_pred_ML_MH") &
                           (summary.alpha0_mode == "train_fold_median")].Mean_RMSE.iloc[0])
    if archived_summary is not None:
        pg = archived_summary[archived_summary.Method == "PG-ML"]
        if len(pg) != 1 or abs(b_mean - float(pg.Mean_RMSE.iloc[0])) > 1e-8:
            raise ValueError("当前保存的method_summary与动力学参数预测文件不一致")
    if abs(b_mean - args.expected_rmse) > 0.002 and not args.allow_other_version:
        raise ValueError(f"检测到Stage8版本不符：Version B mean RMSE={b_mean:.6f}，"
                         f"预期约{args.expected_rmse:.6f}。请使用最新的Stage8_Output(1).zip；"
                         "不要混用旧版数据。")

    pivot = ablation.pivot(index=["No", "Fold", "Version"], columns="alpha0_mode", values=["R2", "RMSE", "MAE"])
    if pivot.isna().any().any():
        raise ValueError("存在不能配对的曲线结果")
    pivot.columns = [f"{metric}_{'median' if mode == 'train_fold_median' else 'zero'}" for metric, mode in pivot.columns]
    paired = pivot.reset_index()
    paired["RMSE_reduction"] = paired["RMSE_zero"] - paired["RMSE_median"]
    paired["R2_gain"] = paired["R2_median"] - paired["R2_zero"]

    paired_sum = []
    for ver, df in paired.groupby("Version"):
        avg0 = float(df.RMSE_zero.mean())
        avgm = float(df.RMSE_median.mean())
        paired_sum.append({
            "Version": ver, "N": len(df),
            "Mean_RMSE_zero": avg0,
            "Mean_RMSE_median": avgm,
            "Mean_RMSE_reduction": avg0 - avgm,
            "RMSE_reduction_pct": 100 * (avg0 - avgm) / avg0,
            "Improved_curves": int((df.RMSE_reduction > 0).sum()),
            "Worsened_curves": int((df.RMSE_reduction < 0).sum()),
            "Wilcoxon_p_RMSE": wilcoxon_p(df.RMSE_reduction)
        })
    paired_sum = pd.DataFrame(paired_sum)

    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("alpha0_ablation_by_curve.csv", ablation),
                        ("alpha0_ablation_summary.csv", summary),
                        ("alpha0_paired_comparison.csv", paired),
                        ("alpha0_paired_summary.csv", paired_sum)):
        table.to_csv(out / name, index=False, encoding="utf-8-sig", float_format="%.12g")

    examples = {}
    if not args.no_plots:
        examples = create_plots(paired, shown, out, args)

    b = paired_sum[paired_sum.Version == "B_pred_ML_MH"].iloc[0]
    manifest = {
        "stage8_source": str(source),
        "N": EXPECTED_COUNT,
        "stage8_pgml_version_B_mean_RMSE": b_mean,
        "max_difference_from_saved_PG_ML": largest_pointwise_diff,
        "source_sha256": {
            "pg_parameter_predictions.csv": hash_prm,
            "pointwise_predictions.csv": hash_pts
        },
        "no_training_performed": True,
        "tested_variables": "alpha0 only: zero vs training-fold median",
        "all_other_oof_parameters_unchanged": True,
        "uses_external_validation_curves": False,
        "figure_examples": examples
    }
    (out / "alpha0_ablation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = (
        f"alpha0 ablation (Stage 5/8) — 88 held-out curves; no retraining\n"
        f"Source: {source}\n"
        f"Version B, alpha0 = 0: Mean RMSE = {b.Mean_RMSE_zero:.6f} kgf cm\n"
        f"Version B, alpha0 = fold median: Mean RMSE = {b.Mean_RMSE_median:.6f} kgf cm\n"
        f"Reduction = {b.RMSE_reduction_pct:.2f}%\n"
        f"Curves improved = {int(b.Improved_curves)}/{int(b.N)}\n"
        f"Two-sided paired Wilcoxon p = {b.Wilcoxon_p_RMSE:.6g}\n\n"
        f"The five out-of-fold model predictions were held fixed. "
        f"The test curve's alpha0 was never used.\n"
        f"Version A is diagnostic only, since it uses test curves' measured ML/MH.\n"
    )
    (out / "alpha0_ablation_readme.txt").write_text(notes, encoding="utf-8")
    print(notes)
    print(f"结果文件夹：{out}")


if __name__ == "__main__":
    main()
