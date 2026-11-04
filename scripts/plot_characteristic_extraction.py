#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
import scipy

from stage01_extract_characteristics import extract_one_curve, plot_curve


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extraction_ast(path):
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "extract_one_curve")
    return ast.dump(node, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir or root / "Figure2_Publication"
    out.mkdir(parents=True, exist_ok=True)
    curve_file = root / "model_ready_data.csv"
    parameter_file = root / "Stage1_Output/kinetic_parameters_stage1.csv"
    manifest_file = root / "Stage1_Output/stage1_run_manifest.json"
    inputs = {str(p): sha(p) for p in [curve_file, parameter_file, manifest_file]}
    frame = pd.read_csv(curve_file, low_memory=False)
    frame = frame.loc[pd.to_numeric(frame.No) == 1].copy()
    record, plot_data = extract_one_curve(1, frame)
    expected = pd.read_csv(parameter_file).set_index("No").loc[1]
    differences = {}
    for key, value in record.items():
        if key in {"No", "failure_reason"}: continue
        differences[key] = abs(float(value) - float(expected[key]))
        assert np.isclose(value, expected[key], rtol=1e-10, atol=1e-10), (key, value, expected[key])
    backup = root / "Figure2_Layout_Backup_20261029"
    backup.mkdir(exist_ok=True)
    existing_png = root / "Stage1_Output/Curves/Curve_1.png"
    if existing_png.exists() and not (backup / existing_png.name).exists():
        shutil.copy2(existing_png, backup / existing_png.name)
    if (backup / "Stage1.py").exists():
        assert extraction_ast(Path(__file__).with_name("stage01_extract_characteristics.py")) == extraction_ast(backup / "Stage1.py")
    output = out / "Figure2_characteristic_extraction.png"
    plot_curve(1, plot_data, output, save_vector=True)
    existing_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, existing_png)
    assert {p: sha(p) for p in inputs} == inputs
    manifest = dict(
        purpose="Figure 2 layout and palette revision only", date="2026-10-29",
        modeling_curve=1, validation_curves_read=False,
        input_sha256=inputs, script_sha256=sha(__file__), stage1_code_sha256=sha(Path(__file__).with_name("stage01_extract_characteristics.py")),
        extraction_algorithm_unchanged=True, checked_parameter_count=len(differences),
        max_saved_parameter_difference=max(differences.values()), parameters=record,
        layout="Full response plus separate early-stage detail; legend and parameters outside axes",
        palette=dict(navy="#203B4D", blue="#4D86B6", teal="#2D8F92", orange="#E77A5A"),
        color_roles=dict(raw_response="#4D86B6", smoothed_response="#2D8F92",
                         maximum_rate_tangent="#E77A5A", induction_fill="#DCEAF5",
                         curing_fill="#E0F0E9"),
        versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        png_dpi=600, vector_formats=["pdf", "svg"],
        outputs={str(p): sha(p) for p in [output, output.with_suffix(".pdf"), output.with_suffix(".svg")]},
    )
    (out / "figure2_layout_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Figure 2 generated: {output}")
    print(f"Checked {len(differences)} saved parameters; maximum difference = {max(differences.values()):.3g}")
    print("Parameter CSV, model-ready data and Stage1 run manifest remain unchanged.")


if __name__ == "__main__":
    main()
