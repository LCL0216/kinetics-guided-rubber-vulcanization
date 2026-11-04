# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


P0 = [5e-4, 1.2]
LOWER_BOUNDS = [1e-8, 0.2]
UPPER_BOUNDS = [1.0, 5.0]
MAXFEV = 50_000
EXPECTED_NOS = set(range(1, 89))

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})

OUTPUT_COLUMNS = [
    "No", "ML", "MH", "ti", "tp", "alpha0", "b", "k1", "n1",
    "R2", "RMSE", "n_points", "k1_boundary_warning",
    "n1_boundary_warning", "success"
]


def _unique_paths(paths):
    seen = set()
    for p in paths:
        p = Path(p).expanduser().resolve()
        if str(p) not in seen:
            seen.add(str(p))
            yield p


def locate_inputs(project_root: str | None, input_file: str | None,
                  stage1_file: str | None):
    here = Path(__file__).resolve().parent
    desktop = Path.home() / "Desktop"

    if project_root:
        bases = list(_unique_paths([project_root]))
    else:
        bases = list(_unique_paths([
            here, Path.cwd(), here / "硫化曲线", Path.cwd() / "硫化曲线",
            here.parent / "硫化曲线", desktop / "硫化曲线",
            desktop / "新数据" / "硫化曲线",
        ]))

    raw_relatives = [
        "model_ready_data.csv.gz", "建模集_1-88/model_ready_data.csv.gz",
        "建模集/model_ready_data.csv.gz", "Training_88/model_ready_data.csv.gz",
        "Modeling_88/model_ready_data.csv.gz", "硫化曲线/model_ready_data.csv.gz",
        "硫化曲线/建模集_1-88/model_ready_data.csv.gz",
    ]
    checked_raw = []
    if input_file:
        raw_candidates = list(_unique_paths([input_file]))
    else:
        raw_candidates = list(_unique_paths(
            base / relative for base in bases for relative in raw_relatives
        ))
    checked_raw = [str(p) for p in raw_candidates]
    raw = next((p for p in raw_candidates if p.is_file()), None)
    if raw is None:
        raise FileNotFoundError(
            "没有找到88条建模曲线 model_ready_data.csv.gz。\n"
            "建议：python Stage2.py --project-root \"/path/to/work\"\n"
            "或传入 --input 指定完整CSV路径。\n已检查：\n" +
            "\n".join(checked_raw[:16])
        )

    if stage1_file:
        stage1_candidates = list(_unique_paths([stage1_file]))
    else:
        stage1_candidates = list(_unique_paths([
            raw.parent / "Stage1_Output" / "kinetic_parameters_stage1.csv",
            raw.parent.parent / "Stage1_Output" / "kinetic_parameters_stage1.csv",
            *[base / "Stage1_Output" / "kinetic_parameters_stage1.csv" for base in bases],
            here / "Stage1_Output" / "kinetic_parameters_stage1.csv",
        ]))
    stage1 = next((p for p in stage1_candidates if p.is_file()), None)
    if stage1 is None:
        raise FileNotFoundError(
            "已找到 model_ready_data.csv.gz，但找不到 Stage1_Output/kinetic_parameters_stage1.csv。\n"
            "请将 Stage 1 的输出文件夹放到建模数据旁边，或用 --stage1-input 指定文件。\n"
            "已检查：\n" + "\n".join(map(str, stage1_candidates[:16]))
        )

    output_root = (Path(project_root).expanduser().resolve()
                   if project_root else stage1.parent.parent)
    return raw, stage1, output_root


def check_no_column(frame: pd.DataFrame, name: str):
    if "No" not in frame.columns:
        raise ValueError(f"{name} 缺少 No 列。")
    values = pd.to_numeric(frame["No"], errors="coerce")
    if values.isna().any() or (values != np.floor(values)).any():
        raise ValueError(f"{name} 中的No有非整数或空值。")
    frame["No"] = values.astype(int)
    found = set(frame["No"].unique().tolist())
    if found != EXPECTED_NOS:
        raise ValueError(
            f"{name} 必须包含完整建模编号1–88，实际有{len(found)}条。\n"
            f"缺少: {sorted(EXPECTED_NOS-found)}；多出: {sorted(found-EXPECTED_NOS)}。\n"
            "不要把独立验证数据1–6误当作建模数据。"
        )
    return frame


def curing_model(t, k1, n1, ti, alpha0):
    dt = np.maximum(np.asarray(t, dtype=float) - ti, 0.0)
    return alpha0 + (1.0 - alpha0) * (1.0 - np.exp(-k1 * (dt ** n1)))


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(ss_tot, 0):
        return np.nan
    return float(1 - ss_res / ss_tot)


def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def sha256sum(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plot_fit(filename: Path, no: int, t_fit, alpha_fit, ti, tp, alpha0,
             k1, n1, r2, rmse, save_pdf: bool):
    t_dense = np.linspace(float(t_fit.min()), float(t_fit.max()), 500)
    alpha_dense = curing_model(t_dense, k1, n1, ti, alpha0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(t_fit, alpha_fit, s=10, alpha=0.45, label="Experimental DOC")
    ax.plot(t_dense, alpha_dense, lw=2.3, label="Kinetic model")
    ax.axvline(ti, ls="--", lw=1.2, label="ti")
    ax.axvline(tp, ls="--", lw=1.2, label="tp")
    ax.text(0.05, 0.95,
            f"k1 = {k1:.6g}\nn1 = {n1:.4f}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.7))
    ax.set(xlabel="Time (s)", ylabel="Degree of Cure", title=f"Curve {no}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(filename.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: 新编号1–88固化动力学参数拟合")
    parser.add_argument("--project-root", default=None,
                        help="项目根目录（通常为桌面的「硫化曲线」文件夹）")
    parser.add_argument("--input", default=None, help="88条曲线model_ready_data.csv.gz的绝对路径")
    parser.add_argument("--stage1-input", default=None,
                        help="Stage1_Output/kinetic_parameters_stage1.csv的绝对路径")
    parser.add_argument("--output-dir", default=None, help="自定义Stage2输出文件夹")
    parser.add_argument("--no-plots", action="store_true", help="仅提取参数，不生成88张曲线图")
    parser.add_argument("--save-pdf", action="store_true", help="保存图片时同时输出PDF")
    args = parser.parse_args()

    raw_path, stage1_path, root = locate_inputs(
        args.project_root, args.input, args.stage1_input
    )
    output_dir = (Path(args.output_dir).expanduser().resolve()
                  if args.output_dir else root / "Stage2_Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "Curves"
    if not args.no_plots:
        curves_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path, low_memory=False)
    stage1_df = pd.read_csv(stage1_path, low_memory=False)
    required_raw = ["No", "Time", "torque"]
    required_s1 = ["No", "ML", "MH", "ti", "tp", "alpha0", "b"]
    missing_raw = [x for x in required_raw if x not in raw_df.columns]
    missing_s1 = [x for x in required_s1 if x not in stage1_df.columns]
    if missing_raw or missing_s1:
        raise ValueError(f"建模曲线缺失字段：{missing_raw}；Stage1参数缺失字段：{missing_s1}")
    raw_df = check_no_column(raw_df, "建模集model_ready_data.csv.gz")
    stage1_df = check_no_column(stage1_df, "Stage1参数表")
    if stage1_df["No"].duplicated().any():
        raise ValueError("Stage1参数表中No有重复编号。")
    if "success" in stage1_df.columns:
        unsuccessful = stage1_df.loc[stage1_df["success"] != 1, "No"].tolist()
        if unsuccessful:
            raise ValueError(f"Stage1存在尚未成功提取的曲线：{unsuccessful}")
    for col in ["Time", "torque"]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    for col in required_s1[1:]:
        stage1_df[col] = pd.to_numeric(stage1_df[col], errors="coerce")
    if not np.isfinite(stage1_df[required_s1[1:]].to_numpy(float)).all():
        raise ValueError("Stage1参数表有NaN/Inf，无法进行Stage2拟合。")

    print("=" * 75)
    print("Stage 2 V3.3：建模曲线1–88固化动力学参数辨识")
    print(f"建模曲线：{raw_path}")
    print(f"Stage1参数：{stage1_path}")
    print(f"输出目录：{output_dir}")
    print(f"建模曲线数量：{raw_df['No'].nunique()}；Stage1成功参数：{len(stage1_df)}")
    print("拟合方程：alpha0+(1-alpha0)*(1-exp(-k1*(t-ti)^n1))")
    print("=" * 75)

    groups = {no: g for no, g in raw_df.groupby("No", sort=True)}
    results = []
    failed = []

    for _, row in stage1_df.sort_values("No").iterrows():
        no = int(row["No"])
        ML, MH, ti, tp, alpha0, b = (float(row[c]) for c in required_s1[1:])
        group = (groups[no][["Time", "torque"]].replace([np.inf, -np.inf], np.nan)
                 .dropna(subset=["Time", "torque"])
                 .sort_values("Time"))
        t = group["Time"].to_numpy(float)
        torque = group["torque"].to_numpy(float)
        reason = None
        if len(t) < 7:
            reason = "原始有效数据不足7个"
        else:
            if np.any(np.diff(t) <= 0):
                tmp = (pd.DataFrame({"Time": t, "torque": torque})
                       .groupby("Time", as_index=False)["torque"].mean()
                       .sort_values("Time"))
                t, torque = tmp["Time"].to_numpy(float), tmp["torque"].to_numpy(float)
            if np.isclose(MH, ML):
                reason = "MH≈ML"
            elif len(t) < 7:
                reason = "去除重复时间后有效数据不足7个"
            elif tp <= ti:
                reason = "tp<=ti，固化区间无效"
        if reason is not None:
            print(f"Curve {no:02d}: FAILED {reason}")
            failed.append({"No": no, "reason": reason})
            continue

        alpha = np.clip((torque - ML) / (MH - ML), 0.0, 1.0)
        fit_mask = (t >= ti) & (t <= tp)
        t_fit, alpha_fit = t[fit_mask], alpha[fit_mask]
        if len(t_fit) < 10:
            reason = "ti~tp拟合点不足10个"
            print(f"Curve {no:02d}: FAILED {reason}")
            failed.append({"No": no, "reason": reason})
            continue

        try:
            popt, pcov = curve_fit(
                lambda tt, k1, n1: curing_model(tt, k1, n1, ti, alpha0),
                t_fit, alpha_fit, p0=P0, bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                maxfev=MAXFEV,
            )
            k1, n1 = map(float, popt)
            alpha_pred = curing_model(t_fit, k1, n1, ti, alpha0)
            r2 = calc_r2(alpha_fit, alpha_pred)
            rmse = calc_rmse(alpha_fit, alpha_pred)
            k1_boundary = int(np.isclose(k1, LOWER_BOUNDS[0], rtol=0, atol=1e-10) or
                              np.isclose(k1, UPPER_BOUNDS[0], rtol=0, atol=1e-6))
            n1_boundary = int(np.isclose(n1, LOWER_BOUNDS[1], atol=1e-4) or
                              np.isclose(n1, UPPER_BOUNDS[1], atol=1e-4))
            results.append({
                "No": no, "ML": ML, "MH": MH, "ti": ti, "tp": tp,
                "alpha0": alpha0, "b": b, "k1": k1, "n1": n1,
                "R2": r2, "RMSE": rmse, "n_points": len(t_fit),
                "k1_boundary_warning": k1_boundary,
                "n1_boundary_warning": n1_boundary, "success": 1
            })
            print(f"Curve {no:02d}: k1={k1:.8g} n1={n1:.6f} "
                  f"DOC_R2={r2:.6f} DOC_RMSE={rmse:.6f}")
            if not args.no_plots:
                plot_fit(curves_dir / f"Fit_{no}.png", no, t_fit, alpha_fit,
                         ti, tp, alpha0, k1, n1, r2, rmse, args.save_pdf)
        except Exception as exc:
            reason = f"curve_fit/绘图失败：{exc}"
            print(f"Curve {no:02d}: FAILED {reason}")
            failed.append({"No": no, "reason": reason})

    result_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    failed_df = pd.DataFrame(failed, columns=["No", "reason"])
    result_path = output_dir / "kinetic_parameters_stage2.csv"
    failed_path = output_dir / "failed_stage2.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")

    manifest = {
        "stage": "Stage 2", "modeling_ids": "1-88", "validation_ids_excluded": "1-6",
        "n_stage1": int(len(stage1_df)), "n_success": int(len(result_df)),
        "n_failed": int(len(failed_df)),
        "raw_input_file": str(raw_path), "raw_input_sha256": sha256sum(raw_path),
        "stage1_input_file": str(stage1_path), "stage1_input_sha256": sha256sum(stage1_path),
        "curing_model": "alpha0+(1-alpha0)*(1-exp(-k1*(t-ti)**n1))",
        "fit_range": "ti <= t <= tp", "p0": P0,
        "lower_bounds": LOWER_BOUNDS, "upper_bounds": UPPER_BOUNDS,
        "maxfev": MAXFEV, "no_plots": args.no_plots,
    }
    (output_dir / "stage2_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print(f"Stage 2: 88条中成功 {len(result_df)} 条，失败 {len(failed_df)} 条")
    if not result_df.empty:
        print(f"Mean DOC R² = {result_df['R2'].mean():.6f}")
        print(f"Mean DOC RMSE = {result_df['RMSE'].mean():.6f}")
        print(f"k1撞边界: {int(result_df['k1_boundary_warning'].sum())} 条")
        print(f"n1撞边界: {int(result_df['n1_boundary_warning'].sum())} 条")
        print("R²最低的5条：")
        print(result_df.nsmallest(5, "R2")[["No", "R2", "RMSE", "k1", "n1"]].to_string(index=False))
    print(f"参数CSV: {result_path}")
    print(f"失败CSV: {failed_path}")
    if not args.no_plots:
        print(f"拟合图片: {curves_dir}")
    print("注：这里的R²/RMSE评价的是在实测曲线上拟合的DOC，不是独立预测性能。")
    print("=" * 75)
    if len(result_df) != 88:
        raise SystemExit("警告：并非88条均成功，请检查 failed_stage2.csv")


if __name__ == "__main__":
    main()
