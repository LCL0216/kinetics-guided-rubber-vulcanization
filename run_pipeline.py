"""Run selected manuscript analyses in a separate work directory."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys
from prepare_workspace import prepare

ROOT = Path(__file__).resolve().parent
STAGES = {
    '01': 'stage01_extract_characteristics.py',
    '02': 'stage02_fit_kinetics.py',
    '03': 'stage03_build_dataset.py',
    '04': 'stage04_compare_regressors.py',
    '05': 'stage05_reconstruct_curves.py',
    '06': 'stage06_assist_high_torque.py',
    '07a': 'stage07a_correlation_analysis.py',
    '07b': 'stage07b_feature_ablation.py',
    '07c': 'stage07c_model_interpretation.py',
    '08': 'stage08_compare_curve_models.py',
    'alpha0': 'ablate_induction_boundary.py',
    '09': 'stage09_predict_external_curves.py',
    '10': 'stage10_compare_kinetic_representations.py',
    'figure2': 'plot_characteristic_extraction.py',
    'figures': 'plot_publication_figures.py',
    'merged': 'plot_merged_figures.py',
    'si': 'plot_supplementary_figures.py',
}


def command(stage, work, no_plots):
    cmd = [sys.executable, str(ROOT / 'scripts' / STAGES[stage])]
    if stage not in {'figures', 'merged', 'si'}:
        project = work / 'Stage9_一键预测' if stage == '09' else work
        cmd += ['--project-root', str(project)]
    if stage == '09':
        cmd += ['--mode', 'predict', '--external-features', str(work / 'independent_validation_process_6.csv'),
                '--external-curves', str(work / 'independent_validation_curves_6.csv')]
    if no_plots and stage in {'01', '02', '05', '06', '09', 'alpha0'}:
        cmd.append('--no-plots')
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--stages', nargs='+', choices=STAGES)
    selection.add_argument('--all', action='store_true', help='Run all analyses and plotting; includes repeated/nested cross-validation')
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'work')
    parser.add_argument('--no-plots', action='store_true', help='Suppress per-curve plots where supported; stages 07/08 still export their summary plots')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without preparing data or running analyses')
    args = parser.parse_args()
    stages = list(STAGES) if args.all else (args.stages or ['09'])
    work = args.work_dir.expanduser().resolve()
    if not args.dry_run and not (work / '.prepared.json').exists():
        prepare(work)
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(ROOT / '.cache' / 'matplotlib')
    env['VULCANIZATION_PROJECT_ROOT'] = str(work)
    for stage in stages:
        cmd = command(stage, work, args.no_plots)
        print(f'\n[{stage}] {shlex.join(cmd)}', flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=work, env=env, check=True)


if __name__ == '__main__':
    main()
