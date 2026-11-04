# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_NOS = set(range(1, 89))
PROCESS_COLS = [
    "mixing_time", "mixing_temp", "curing_temp", "curing_pressure",
    "rotor_speed_1", "rotor_speed_2", "curing_time_offset",
]
CORE_TARGETS = ["ML", "MH", "ti", "k1", "n1"]
KINETIC_TARGETS = ["ML", "MH", "ti", "tp", "alpha0", "b", "k1", "n1"]
ROUND_TIME = 6
ROUND_TORQUE = 8


def read_args():
    p = argparse.ArgumentParser(description="Stage3：88条曲线的工艺-动力学数据集构建")
    p.add_argument("--project-root", type=str, default=None,
                   help="项目根目录，如 /path/to/work")
    p.add_argument("--raw", type=str, default=None,
                   help="指定 model_ready_data.csv 完整路径")
    p.add_argument("--stage2", type=str, default=None,
                   help="指定 kinetic_parameters_stage2.csv 完整路径")
    return p.parse_args()


def candidate_roots(root_arg: str | None) -> list[Path]:
    here = Path(__file__).resolve().parent
    bases = [Path(root_arg).expanduser().resolve()] if root_arg else [
        here, Path.cwd().resolve(), here.parent,
        Path.home() / "Desktop" / "硫化曲线",
        Path.home() / "Desktop" / "新数据" / "硫化曲线",
    ]
    roots: list[Path] = []
    for base in bases:
        for folder in (base, base / "建模集_1-88", base / "硫化曲线",
                       base / "硫化曲线" / "建模集_1-88"):
            if folder not in roots:
                roots.append(folder)
    return roots


def find_paths(args):
    roots = candidate_roots(args.project_root)
    raw_explicit = Path(args.raw).expanduser().resolve() if args.raw else None
    stage2_explicit = Path(args.stage2).expanduser().resolve() if args.stage2 else None
    tests = []
    for root in roots:
        raw_options = [raw_explicit] if raw_explicit else [
            root / "model_ready_data.csv",
            root / "建模集_1-88" / "model_ready_data.csv",
        ]
        stage2_options = [stage2_explicit] if stage2_explicit else [
            root / "Stage2_Output" / "kinetic_parameters_stage2.csv",
            root / "建模集_1-88" / "Stage2_Output" / "kinetic_parameters_stage2.csv",
            root.parent / "Stage2_Output" / "kinetic_parameters_stage2.csv",
        ]
        for raw in raw_options:
            for stage2 in stage2_options:
                if raw is None or stage2 is None:
                    continue
                tests.append((raw, stage2))
                if raw.is_file() and stage2.is_file():
                    # explicit project root has priority for the output directory
                    output_root = (Path(args.project_root).expanduser().resolve()
                                   if args.project_root else
                                   raw.parent if raw.parent.name != "建模集_1-88" else raw.parent.parent)
                    return raw, stage2, output_root
    detail = "\n".join(f"  Raw: {a}\n  Stage2: {b}" for a, b in tests[:14])
    raise FileNotFoundError(
        "未找到 Stage 3 需要的两份文件。请核对 model_ready_data.csv 和 "
        "Stage2_Output/kinetic_parameters_stage2.csv 是否都已经放好。\n"
        "建议添加 --project-root /path/to/work。\n"
        f"已检查的部分位置：\n{detail}"
    )


def enforce_no_1_to_88(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "No" not in frame.columns:
        raise ValueError(f"{label} 缺少 No 列")
    tmp = pd.to_numeric(frame["No"], errors="coerce")
    if tmp.isna().any() or not np.allclose(tmp, np.round(tmp)):
        raise ValueError(f"{label} 的 No 含空值或非整数编号，请检查")
    frame = frame.copy()
    frame["No"] = tmp.astype(int)
    actual = set(frame["No"].unique().tolist())
    if actual != EXPECTED_NOS:
        raise ValueError(
            f"{label} 编号必须是 1–88。缺少：{sorted(EXPECTED_NOS - actual)}；"
            f"额外：{sorted(actual - EXPECTED_NOS)}。"
        )
    return frame


def generate_curve_hash(group: pd.DataFrame) -> str:
    valid = (group.dropna(subset=["Time", "torque"])
             .sort_values("Time", kind="mergesort"))
    t = np.round(valid["Time"].to_numpy(dtype=np.float64), ROUND_TIME)
    y = np.round(valid["torque"].to_numpy(dtype=np.float64), ROUND_TORQUE)
    combined = np.column_stack([t, y])
    return hashlib.sha256(combined.tobytes()).hexdigest()


def main():
    args = read_args()
    raw_file, stage2_file, root = find_paths(args)
    out_dir = root / "Stage3_Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Stage 3：工艺-动力学曲线级数据集 / 重复曲线检查")
    print(f"原始数据：{raw_file}")
    print(f"Stage 2参数：{stage2_file}")
    print(f"输出目录：{out_dir}")
    print("=" * 70)

    raw_df = pd.read_csv(raw_file, low_memory=False)
    kinetic_df = pd.read_csv(stage2_file)
    for c in ["Time", "torque", *PROCESS_COLS]:
        if c not in raw_df.columns:
            raise ValueError(f"model_ready_data.csv 缺少必要字段 {c}")
    for c in KINETIC_TARGETS:
        if c not in kinetic_df.columns:
            raise ValueError(f"Stage 2 参数表缺少必要字段 {c}")

    raw_df = enforce_no_1_to_88(raw_df, "建模原始曲线")
    kinetic_df = enforce_no_1_to_88(kinetic_df, "Stage 2 动力学参数")
    if kinetic_df["No"].duplicated().any():
        raise ValueError("Stage 2 参数表中有重复 No，不能进行一对一合并")
    if "success" in kinetic_df.columns and not (kinetic_df["success"] == 1).all():
        raise ValueError("Stage 2 参数表包含未拟合成功的记录")

    for c in ["Time", "torque", *PROCESS_COLS]:
        raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")
    for c in KINETIC_TARGETS:
        kinetic_df[c] = pd.to_numeric(kinetic_df[c], errors="coerce")
    if kinetic_df[KINETIC_TARGETS].isna().any().any():
        raise ValueError("Stage 2 动力学参数存在缺失值")
    if (kinetic_df["k1"] <= 0).any():
        raise ValueError("Stage 2 的 k1 含非正数，后续XGBoost不能进行log10训练")

    variation = raw_df.groupby("No", sort=True)[PROCESS_COLS].nunique(dropna=False)
    unstable = [(int(no), col) for no, row in variation.iterrows()
                for col in PROCESS_COLS if row[col] != 1]
    if unstable:
        raise ValueError(f"工艺参数在同一条曲线内不恒定，前15项：{unstable[:15]}")
    process_df = (raw_df.sort_values(["No", "Time"], kind="mergesort")
                  .groupby("No", as_index=False, sort=True)[PROCESS_COLS].first())
    if process_df[PROCESS_COLS].isna().any().any():
        raise ValueError("工艺参数表存在缺失，请核对对应曲线")

    groups = raw_df.groupby("No", sort=True)
    hashes = [
        {"No": int(no), "curve_hash": generate_curve_hash(group),
         "curve_n_points": len(group)}
        for no, group in groups
    ]
    hash_df = pd.DataFrame(hashes).sort_values("No").reset_index(drop=True)
    unique_hashes = hash_df["curve_hash"].drop_duplicates().tolist()
    hash_to_group = {h: i + 1 for i, h in enumerate(unique_hashes)}
    hash_df["curve_group_id"] = hash_df["curve_hash"].map(hash_to_group)
    counts = hash_df.groupby("curve_group_id").size().rename("group_size")
    hash_df = hash_df.join(counts, on="curve_group_id")
    hash_df["is_duplicate_curve"] = (hash_df["group_size"] > 1).astype(int)
    duplicate_df = hash_df.loc[hash_df["is_duplicate_curve"] == 1].copy()
    group_summary = (hash_df.groupby("curve_group_id", as_index=False)
                     .agg(group_size=("No", "size"),
                          No_list=("No", lambda v: ",".join(map(str, sorted(v)))),
                          curve_n_points=("curve_n_points", "first")))
    group_summary["is_duplicate_group"] = (group_summary["group_size"] > 1).astype(int)

    dataset = (process_df.merge(kinetic_df, on="No", how="inner", validate="one_to_one")
               .merge(hash_df[["No", "curve_group_id", "group_size",
                               "is_duplicate_curve", "curve_n_points"]],
                      on="No", how="inner", validate="one_to_one")
               .sort_values("No").reset_index(drop=True))
    if len(dataset) != 88 or set(dataset["No"]) != EXPECTED_NOS:
        raise RuntimeError("Stage 3合并后样本不是完整的88条，已停止保存")
    if dataset[[*PROCESS_COLS, *CORE_TARGETS, "alpha0"]].isna().any().any():
        raise RuntimeError("合并后的核心输入/目标中存在缺失值")

    dataset.to_csv(out_dir / "pgmlp_curve_dataset.csv", index=False, encoding="utf-8-sig")
    duplicate_df.to_csv(out_dir / "duplicate_curve_groups.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(out_dir / "curve_group_summary.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "recommended_features.txt").open("w", encoding="utf-8") as fh:
        fh.write("PG-ML Stage 3 特征建议（建模集No 1–88）\n" + "=" * 58 + "\n\n")
        fh.write("工艺输入 X（7项）：\n" + "".join(f"  - {x}\n" for x in PROCESS_COLS))
        fh.write("\n推荐预测目标 Y（5项）：\n" + "".join(f"  - {x}\n" for x in CORE_TARGETS))
        fh.write("\n保留供诊断的参数：\n" + "".join(f"  - {x}\n" for x in KINETIC_TARGETS))
        fh.write("\n分组说明：按 Time-torque 指纹识别完全重复曲线；"
                 "若存在重复，则相同curve_group_id不得跨训练/测试集。\n")
        fh.write("六条新实验（No 1–6，另一套独立编号）不参与本阶段。\n")

    manifest = {
        "stage": 3, "created_at": datetime.now().astimezone().isoformat(),
        "raw_file": str(raw_file), "stage2_file": str(stage2_file),
        "output_file": str(out_dir / "pgmlp_curve_dataset.csv"),
        "n_curves": int(len(dataset)), "no_min": 1, "no_max": 88,
        "n_curve_groups": int(group_summary.shape[0]),
        "n_duplicate_curve_records": int(duplicate_df.shape[0]),
        "process_cols": PROCESS_COLS, "predicted_targets": CORE_TARGETS,
        "round_time": ROUND_TIME, "round_torque": ROUND_TORQUE,
        "note": "Stage3仅使用建模集88条，独立验证6条不参与。"
    }
    with (out_dir / "stage3_run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"\n成功生成曲线级数据集：{dataset.shape[0]} 条，{dataset.shape[1]} 列")
    print(f"唯一曲线组：{len(group_summary)}；重复曲线记录：{len(duplicate_df)}")
    if len(duplicate_df):
        print("⚠ 重要：同一 curve_group_id 下的完全相同曲线不能分散到训练集和测试集！")
        print("⚠ Stage 4/8 需要采用按 curve_group_id 分组的交叉验证；")
        print("  原来仅按 No 使用 KFold 的结果不能视为严格防泄漏的评估。")
    print(f"核心输入：{PROCESS_COLS}")
    print(f"预测目标：{CORE_TARGETS}")
    print(f"输出：{out_dir / 'pgmlp_curve_dataset.csv'}")
    print("Stage 3 完成。进入 Stage 4 前请先修改为按 curve_group_id 分组划分。")


if __name__ == "__main__":
    main()
