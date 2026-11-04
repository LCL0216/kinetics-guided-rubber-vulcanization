"""Check input hashes, script syntax and six-curve frozen-model inference."""
from pathlib import Path
import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_workspace import prepare
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    for path in [*ROOT.glob('*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('exploratory/*.py'), *ROOT.glob('checks/*.py')]:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    with tempfile.TemporaryDirectory(prefix='vulcanization-check-') as folder:
        work = prepare(Path(folder) / 'work')
        with (work / 'model_ready_data.csv').open(encoding='utf-8-sig', newline='') as handle:
            curves = {int(row['No']) for row in csv.DictReader(handle)}
        assert curves == set(range(1, 89))
        train = pd.read_csv(work / 'Stage3_Output/pgmlp_curve_dataset.csv')
        validation = pd.read_csv(work / 'independent_validation_process_6.csv')
        assert len(train) == 88 and train.No.nunique() == 88
        assert len(validation) == 6 and set(validation.No) == set(range(1, 7))
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        env['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
        subprocess.run([sys.executable, str(ROOT / 'run_pipeline.py'), '--stages', '09',
                        '--work-dir', str(work), '--no-plots'], env=env, check=True)
        output = work / 'Stage9_一键预测/Stage9_Output'
        reference = ROOT / 'reference/Stage9_一键预测/Stage9_Output'
        actual = pd.read_csv(output / 'stage9_curve_metrics.csv').sort_values('No')
        expected = pd.read_csv(reference / 'stage9_curve_metrics.csv').sort_values('No')
        assert np.array_equal(actual.No, expected.No)
        columns = ['R2', 'RMSE', 'MAE']
        np.testing.assert_allclose(actual[columns], expected[columns], rtol=1e-6, atol=1e-7)
        predicted = pd.read_csv(output / 'stage9_pointwise_predictions.csv')
        assert set(predicted.No) == set(range(1, 7))
        report = {'script_syntax': 'passed', 'input_and_model_hashes': 'passed',
                  'modeling_curves': 88, 'independent_validation_curves': 6,
                  'frozen_model_prediction': 'passed', 'metrics_match_saved_reference': True,
                  'max_absolute_metric_difference': float(abs(actual[columns].to_numpy()-expected[columns].to_numpy()).max()),
                  'validation_mean_metrics': actual[columns].mean().to_dict(),
                  'full_cross_validation_pipeline_rerun': False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
