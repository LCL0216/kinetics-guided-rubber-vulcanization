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
from scipy.signal import savgol_filter

MAX_SG_WINDOW = 21
BASELINE_RATIO = 0.20
MIN_BASELINE_POINTS = 5
B_HALF_WINDOW = 2
EXPECTED_NOS = set(range(1, 89))

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


class CurveError(ValueError):
    pass


def locate_dataset(project_root: str | None, explicit_input: str | None):
    script_dir = Path(__file__).resolve().parent
    if explicit_input:
        p = Path(explicit_input).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"指定的建模数据不存在：{p}")
        return p, p.parent

    if project_root:
        bases = [Path(project_root).expanduser().resolve()]
    else:
        bases = [script_dir, script_dir / "硫化曲线", Path.cwd(), Path.cwd() / "硫化曲线"]
    relatives = [
        "model_ready_data.csv",
        "建模集_1-88/model_ready_data.csv",
        "建模集_1-88/Stage1/model_ready_data.csv",
        "建模集_1-88/Stage1_Output/model_ready_data.csv",
        "Training_88/model_ready_data.csv",
        "Modeling_88/model_ready_data.csv",
        "建模集/model_ready_data.csv",
        "硫化曲线/model_ready_data.csv",
        "硫化曲线/建模集_1-88/model_ready_data.csv",
    ]
    checked = []
    for base in bases:
        for relative in relatives:
            p = base / relative
            checked.append(str(p))
            if p.is_file():
                return p.resolve(), base.resolve()
    raise FileNotFoundError(
        "未找到88条建模数据 model_ready_data.csv。\n"
        "请将脚本与文件放在『硫化曲线』同一个文件夹中，"
        "或使用 --project-root、--input 指定路径。\n"
        + "\n".join(checked[:12])
    )


def load_training_data(path: Path):
    df = pd.read_csv(path, low_memory=False)
    required = ["No", "Time", "torque"]
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise ValueError(f"{path.name} 缺少必要列 {absent}；要求 No, Time, torque。")
    numeric_ids = pd.to_numeric(df["No"], errors="coerce")
    if numeric_ids.isna().any() or not np.all(np.equal(numeric_ids, np.floor(numeric_ids))):
        raise ValueError("建模集 No 列有空值、非数值或非整数编号，请先核对重编号文件。")
    df["No"] = numeric_ids.astype(int)
    actual = set(df["No"].unique().tolist())
    if actual != EXPECTED_NOS:
        missing = sorted(EXPECTED_NOS - actual)
        extra = sorted(actual - EXPECTED_NOS)
        raise ValueError(
            f"建模集应恰好包含 No=1..88 共88条；当前 {len(actual)} 条。\n"
            f"缺少：{missing}\n不应存在：{extra}\n"
            "请不要把6条验证曲线合并到这里。"
        )
    for col in ["Time", "torque"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def extract_one_curve(no: int, frame: pd.DataFrame):
    g = frame.dropna(subset=["Time", "torque"]).sort_values("Time")
    t = g["Time"].to_numpy(float)
    y = g["torque"].to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    if len(t) < 7:
        raise CurveError("数据点不足7个")

    if np.any(np.diff(t) <= 0):
        tmp = (pd.DataFrame({"Time": t, "torque": y})
               .groupby("Time", as_index=False)["torque"].mean()
               .sort_values("Time"))
        t, y = tmp["Time"].to_numpy(float), tmp["torque"].to_numpy(float)
    if len(t) < 7:
        raise CurveError("去除重复时间后数据点不足7个")

    ML, MH = float(np.min(y)), float(np.max(y))
    if np.isclose(MH, ML):
        raise CurveError("MH≈ML，曲线无有效固化变化")
    alpha = np.clip((y - ML) / (MH - ML), 0.0, 1.0)
    window = min(MAX_SG_WINDOW, len(alpha))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        raise CurveError("无法建立有效SG平滑窗口")
    poly = min(3, window - 2)
    alpha_smooth = np.clip(savgol_filter(alpha, window_length=window,
                                        polyorder=poly, mode="interp"), 0.0, 1.0)
    try:
        d_alpha = np.gradient(alpha_smooth, t)
    except Exception as exc:
        raise CurveError(f"导数计算失败：{exc}") from exc
    if not np.all(np.isfinite(d_alpha)):
        raise CurveError("导数存在NaN或Inf")

    max_rate_idx = int(np.argmax(d_alpha))
    max_rate = float(d_alpha[max_rate_idx])
    if max_rate <= 0:
        raise CurveError("最大固化速率<=0")
    tm = float(t[max_rate_idx])
    alpha_m = float(alpha_smooth[max_rate_idx])

    baseline_end_idx = int(max_rate_idx * BASELINE_RATIO)
    baseline_end_idx = max(baseline_end_idx, MIN_BASELINE_POINTS)
    baseline_end_idx = min(baseline_end_idx, max_rate_idx)
    if baseline_end_idx < MIN_BASELINE_POINTS:
        raise CurveError("最大速率点过早，诱导平台数据不足")
    baseline = alpha_smooth[:baseline_end_idx]
    alpha_baseline = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))

    ti_raw = tm + (alpha_baseline - alpha_m) / max_rate
    if not np.isfinite(ti_raw):
        raise CurveError("ti计算结果无效")
    ti = float(np.clip(ti_raw, t.min(), tm))
    ti_clip_warning = int(not np.isclose(ti, ti_raw))
    ti_idx = int(np.argmin(np.abs(t - ti)))
    ti_sample = float(t[ti_idx])
    alpha0 = float(np.clip(alpha[ti_idx], 0.0, 1.0))

    tp_idx = int(np.argmax(alpha_smooth))
    tp = float(t[tp_idx])
    if tp_idx <= ti_idx:
        raise CurveError("tp<=ti，无法形成有效固化区间")

    left = max(0, ti_idx - B_HALF_WINDOW)
    right = min(len(d_alpha), ti_idx + B_HALF_WINDOW + 1)
    b = float(np.mean(d_alpha[left:right]))
    if b < 0 and abs(b) < 1e-10:
        b = 0.0
    b_warning = int(b < 0)
    curing_duration = tp - ti
    dt_median = float(np.median(np.diff(t)))
    ti_warning = int(ti <= (t.min() + 2 * dt_median))

    result = {
        "No": no, "success": 1, "failure_reason": "",
        "ML": round(ML, 6), "MH": round(MH, 6),
        "ti": round(ti, 6), "ti_raw": round(float(ti_raw), 6),
        "ti_sample": round(ti_sample, 6), "tp": round(tp, 6),
        "curing_duration": round(curing_duration, 6),
        "alpha0": round(alpha0, 8), "b": round(b, 10),
        "tm": round(tm, 6), "max_rate": round(max_rate, 10),
        "alpha_baseline": round(alpha_baseline, 8),
        "baseline_std": round(baseline_std, 8),
        "ti_warning": ti_warning, "ti_clip_warning": ti_clip_warning,
        "b_warning": b_warning, "n_points": len(t),
    }
    plot_data = (t, alpha, alpha_smooth, ti, tm, tp, alpha_baseline,
                 alpha_m, max_rate, tp_idx, alpha0, b)
    return result, plot_data


def plot_curve(no: int, plot_data, output: Path, *, save_vector: bool = False):
    """Journal figure with a full response and a separate early-stage detail.

    This function changes presentation only. Every response value, boundary,
    slope and tangent is supplied by the unchanged extract_one_curve routine.
    The palette follows the manuscript's navy/teal figures. A shared legend
    and parameter line sit outside both axes, leaving the data unobstructed.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator

    t, alpha, smooth, ti, tm, tp, baseline, alpha_m, rate, tp_idx, alpha0, b = plot_data
    colors = {
        "navy": "#203B4D", "blue": "#4D86B6", "raw": "#4D86B6",
        "teal": "#2D8F92", "orange": "#E77A5A", "muted": "#657A86",
        "induction": "#DCEAF5", "curing": "#E0F0E9",
        "line": "#D9E3E8", "grid": "#E5EBEE",
    }
    style = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9.5,
        "axes.labelcolor": colors["navy"], "text.color": colors["navy"],
        "xtick.color": colors["muted"], "ytick.color": colors["muted"],
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "axes.edgecolor": colors["muted"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False, "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.2, 3.8), facecolor="white")
        grid = fig.add_gridspec(1, 2, width_ratios=[1.48, 1],
                               left=.09, right=.985, bottom=.25,
                               top=.81, wspace=.26)
        full, detail = [fig.add_subplot(grid[0, i]) for i in range(2)]
        # The detail interval is determined only by the existing ti and tm.
        span = max(tm - ti, float(np.median(np.diff(t))))
        detail_end = min(float(t.max()), tm + 1.8 * span)
        detail_start = min(0., float(t.min()))
        detail_ymax = min(1.04, max(.30, float(np.interp(detail_end, t, smooth)) + .09))
        ti_sample = float(t[int(np.argmin(abs(t - ti)))])
        for ax in (full, detail):
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=colors["grid"], lw=.55)
            ax.plot(t, alpha, color=colors["raw"], lw=.8, alpha=.85,
                    marker="o", markevery=max(1, len(t) // 32), markersize=2.6,
                    markerfacecolor="white", markeredgewidth=.65, zorder=2)
            ax.plot(t, smooth, color=colors["teal"], lw=1.65, zorder=3)
            ax.axhline(baseline, color=colors["muted"], lw=.85,
                       ls=(0, (1.4, 2.4)), zorder=2)
            ax.tick_params(direction="out", length=3, width=.65, pad=3)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="minor", length=1.7, width=.5)
            ax.set_xlabel("Time (s)", labelpad=5)
        full.set(xlim=(min(0., float(t.min())), float(t.max()) * 1.025), ylim=(-.025, 1.065))
        full.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        full.yaxis.set_major_locator(MultipleLocator(.2))
        full.set_ylabel(r"Normalized torque response, $\alpha$", labelpad=5)
        full.set_title(f"(a) Curve {no}: full response", loc="left", fontweight="bold", pad=10)
        full.axvspan(min(0., float(t.min())), ti, color=colors["induction"], alpha=.65, zorder=0)
        full.axvspan(ti, tp, color=colors["curing"], alpha=.65, zorder=0)
        for time, symbol, color in [(ti, r"$t_i$", colors["blue"]), (tp, r"$t_p$", colors["navy"])]:
            full.axvline(time, color=color, lw=.85,
                         ls=(0, (4, 3)), zorder=1)
            full.annotate(symbol, xy=(time, 1.035), xytext=(4, 0),
                          textcoords="offset points", ha="left", va="center", fontsize=9, color=color)
        full.scatter([tp], [smooth[tp_idx]], s=21, color=colors["navy"], zorder=5)
        full.text(.66, .12, "Curing interval", transform=full.transAxes,
                  ha="center", va="center", fontsize=8, color=colors["teal"])
        full.add_patch(Rectangle((detail_start, 0), detail_end - detail_start,
                                 detail_ymax, fill=False, edgecolor=colors["blue"],
                                 linewidth=.85, linestyle=(0, (2, 2)), zorder=4))
        detail.set(xlim=(detail_start, detail_end), ylim=(-.025, detail_ymax))
        detail.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        detail.set_title("(b) Early-stage detail", loc="left", fontweight="bold", pad=10)
        detail.set_ylabel(r"$\alpha$", labelpad=3)
        detail.axvspan(detail_start, ti, color=colors["induction"], alpha=.65, zorder=0)
        detail.axvspan(ti, min(tp, detail_end), color=colors["curing"], alpha=.65, zorder=0)
        tangent_t = np.linspace(ti, detail_end, 200)
        detail.plot(tangent_t, alpha_m + rate * (tangent_t - tm),
                    color=colors["orange"], lw=1.35, ls=(0, (5, 2)), zorder=4)
        for time, symbol, color in [(ti, r"$t_i$", colors["blue"]), (tm, r"$t_m$", colors["orange"])]:
            detail.axvline(time, color=color, lw=.85,
                           ls=(0, (4, 3)), zorder=1)
            detail.annotate(symbol, xy=(time, detail_ymax), xytext=(4, -6),
                            textcoords="offset points", ha="left", va="top", fontsize=9, color=color)
        detail.scatter([ti], [baseline], s=19, color=colors["blue"], zorder=5)
        detail.scatter([tm], [alpha_m], s=21, color=colors["orange"], zorder=5)
        detail.scatter([ti_sample], [alpha0], s=23, marker="D", facecolors="white",
                       edgecolors=colors["orange"], linewidths=1.1, zorder=6)
        detail.annotate(r"$\alpha_0$", xy=(ti_sample, alpha0), xycoords="data",
                        xytext=(.10, .30), textcoords="axes fraction", fontsize=9,
                        ha="left", va="center", color=colors["orange"],
                        arrowprops=dict(arrowstyle="-", color=colors["orange"],
                                        lw=.8, shrinkA=3, shrinkB=4))
        handles = [
            Line2D([], [], color=colors["raw"], lw=.8, marker="o", markersize=3,
                   markerfacecolor="white", markeredgewidth=.65, label="Raw response"),
            Line2D([], [], color=colors["teal"], lw=1.65, label="Smoothed response"),
            Line2D([], [], color=colors["orange"], lw=1.35, ls=(0, (5, 2)), label="Maximum-rate tangent"),
            Line2D([], [], color=colors["muted"], lw=.85, ls=(0, (1.4, 2.4)), label="Induction baseline"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .99),
                   ncol=2, frameon=False, fontsize=8, handlelength=2.6,
                   columnspacing=2.8, handletextpad=.65, labelspacing=.65)
        # Numeric annotations use a dedicated band rather than covering data.
        for x, label, color in [(.28, rf"$t_i={ti:.1f}$ s", colors["blue"]),
                                (.53, rf"$t_m={tm:.1f}$ s", colors["orange"]),
                                (.78, rf"$t_p={tp:.1f}$ s", colors["navy"])]:
            fig.text(x, .100, label, ha="center", va="center", fontsize=8.5, color=color)
        fig.text(.53, .045,
                 rf"$\alpha_0={alpha0:.4f}$     $b={b:.6f}$ s$^{{-1}}$",
                 ha="center", va="center", fontsize=8.5, color=colors["muted"])
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=.04)
        if save_vector:
            for suffix in (".pdf", ".svg"):
                fig.savefig(output.with_suffix(suffix), bbox_inches="tight", pad_inches=.04)
        plt.close(fig)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Stage1 V3.3：重新提取88条编号1–88的硫化曲线参数")
    parser.add_argument("--project-root", type=str, default=None,
                        help="硫化曲线项目文件夹的位置")
    parser.add_argument("--input", type=str, default=None,
                        help="明确指定建模集 model_ready_data.csv 的路径")
    parser.add_argument("--no-plots", action="store_true", help="只生成CSV，不画88张图")
    args = parser.parse_args()

    data_path, base = locate_dataset(args.project_root, args.input)
    df = load_training_data(data_path)
    out_dir = base / "Stage1_Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "Curves"
    if not args.no_plots:
        image_dir.mkdir(exist_ok=True)
    print("=" * 76)
    print("PG-ML Stage 1 V3.3 | 88条建模曲线 | 新编号 1–88")
    print(f"输入：{data_path}\n总记录：{len(df):,}\n输出：{out_dir}")
    print("注意：验证集1–6不参与此阶段")
    print("=" * 76)

    records = []
    for no, group in df.groupby("No", sort=True):
        try:
            record, plot_data = extract_one_curve(int(no), group)
            records.append(record)
            if not args.no_plots:
                plot_curve(int(no), plot_data, image_dir / f"Curve_{int(no)}.png")
            print(f"[{no:02d}/88] OK  ML={record['ML']:.3f} MH={record['MH']:.3f} "
                  f"ti={record['ti']:.3f}s alpha0={record['alpha0']:.5f}")
        except CurveError as exc:
            records.append({"No": int(no), "success": 0, "failure_reason": str(exc)})
            print(f"[{no:02d}/88] FAILED: {exc}")

    all_df = pd.DataFrame(records).sort_values("No").reset_index(drop=True)
    ok = all_df.loc[all_df["success"].eq(1)].copy()
    failed = all_df.loc[all_df["success"].ne(1)].copy()
    all_df.to_csv(out_dir / "stage1_all_records.csv", index=False, encoding="utf-8-sig")
    ok.to_csv(out_dir / "kinetic_parameters_stage1.csv", index=False, encoding="utf-8-sig")
    failed.to_csv(out_dir / "failed_curves.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "stage": 1, "algorithm": "V3.3 same definitions as uploaded Stage1_Output.py",
        "dataset_role": "training_88_only", "dataset_path": str(data_path),
        "dataset_sha256": sha256_file(data_path), "numbering": "1-88",
        "source_rows": int(len(df)), "n_curves": 88,
        "n_success": int(len(ok)), "n_failed": int(len(failed)),
        "n_curve_plots": (0 if args.no_plots else int(len(ok))),
        "validation_6_used": False,
        "configuration": {
            "MAX_SG_WINDOW": MAX_SG_WINDOW, "BASELINE_RATIO": BASELINE_RATIO,
            "MIN_BASELINE_POINTS": MIN_BASELINE_POINTS,
            "B_HALF_WINDOW": B_HALF_WINDOW,
        },
    }
    (out_dir / "stage1_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 76)
    print(f"Stage 1 完成：成功 {len(ok)}/88，失败 {len(failed)}/88")
    if len(ok):
        print("\n主要参数描述统计：")
        print(ok[["ML", "MH", "ti", "tp", "alpha0", "b"]].describe().round(6).to_string())
    if len(failed):
        print("\n失败编号及原因：")
        print(failed[["No", "failure_reason"]].to_string(index=False))
    print(f"\nCSV保存于：{out_dir}")
    if not args.no_plots:
        print(f"图片保存于：{image_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
