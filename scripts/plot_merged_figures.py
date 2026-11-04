#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge manuscript figures 5/6, 7/8 and 13/14 from saved results.

No fitting, prediction or numerical source file is changed. The selected
examples, original method colors and numerical definitions are retained.
Run: python scripts/plot_merged_figures.py
Outputs: new Figures 5, 6 and 11 in 600-dpi PNG and vector PDF/SVG.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("VULCANIZATION_PROJECT_ROOT", str(Path(__file__).resolve().parents[1] / "work"))).resolve()
OUT = ROOT / 'Merged_Figures'
INPUTS = {}
INK, GRID = '#213247', '#E5EAF0'
BLUE, RED = '#1F77B4', '#E5533D'
ALPHA_ORANGE, ALPHA_BLUE, GREEN = '#E5B16D', '#5CA4BD', '#50A99F'
DIRECT = {'Direct-MLP': '#2CA02C', 'Direct-XGB': '#FF7F0E', 'PG-ML': '#D62728'}


def read(rel):
    path = ROOT / rel
    INPUTS[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.4,
        'mathtext.fontset': 'dejavusans', 'axes.labelsize': 9.4,
        'axes.titlesize': 10, 'axes.titleweight': 'semibold',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#8292A2', 'axes.linewidth': .75,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': '#425466', 'ytick.color': '#425466',
        'xtick.labelsize': 8.4, 'ytick.labelsize': 8.4,
        'legend.fontsize': 8.5, 'legend.frameon': False,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })


def axis(ax, grid='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, lw=.65, alpha=.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))


def title(ax, letter, text):
    ax.set_title(f'({letter})  {text}', loc='left', pad=9, color=INK)


def save(fig, stem):
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{fmt}', dpi=600)
    plt.close(fig)


def meanbars(ax, values, cols, labels, fmt='.3f'):
    axis(ax)
    bars = ax.bar([0, 1], values, color=cols, width=.55,
                  edgecolor=[colors.to_rgba(c, .9) for c in cols], linewidth=.7)
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-.7, 1.7)
    ax.set_ylim(0, max(values)*1.24)
    for bar, val in zip(bars, values):
        ax.annotate(format(val, fmt), (bar.get_x()+bar.get_width()/2, val),
                    xytext=(0, 5), textcoords='offset points', ha='center',
                    va='bottom', fontsize=9, color=INK)


def boxpoints(ax, groups, cols, labels, seed=42, means=False):
    pos = np.arange(1, len(groups)+1)
    bp = ax.boxplot(groups, positions=pos, widths=.48, showfliers=False,
                    patch_artist=True, whis=1.5,
                    medianprops={'color': INK, 'lw': 1.4},
                    whiskerprops={'color': '#8292A2', 'lw': .9},
                    capprops={'color': '#8292A2', 'lw': .9})
    rng = np.random.default_rng(seed)
    for i, group, col, patch in zip(pos, groups, cols, bp['boxes']):
        patch.set(facecolor=colors.to_rgba(col, .25), edgecolor=col, lw=1)
        ax.scatter(rng.normal(i, .053, len(group)), group,
                   s=12, c=col, alpha=.65, edgecolors='white', lw=.3, zorder=3)
        if means:
            ax.scatter(i, np.mean(group), marker='D', s=28, facecolor='white',
                       edgecolor=INK, lw=1.1, zorder=5)
    ax.set_xticks(pos, labels)
    ax.set_xlim(.5, len(groups)+.5)
    axis(ax)
    ax.set_xticks(pos, labels)


def reconstruction():
    d = read('Stage5_Output_alpha0_comparison/pointwise_AB_predictions.csv')
    ma = read('Stage5_Output_alpha0_comparison/curve_metrics_A.csv').set_index('No')
    mb = read('Stage5_Output_alpha0_comparison/curve_metrics_B.csv').set_index('No')
    assert d.No.nunique() == len(ma) == len(mb) == 88
    fig = plt.figure(figsize=(8.4, 7.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, .76],
                          left=.085, right=.985, bottom=.07, top=.935,
                          hspace=.66, wspace=.30)
    legend = [Line2D([], [], color='#212529', lw=1.8, label='Experimental'),
              Line2D([], [], color=BLUE, ls='--', lw=1.7, label='Version A'),
              Line2D([], [], color=RED, ls='-.', lw=1.7, label='Version B')]
    fig.legend(handles=legend, ncol=3, loc='upper center', bbox_to_anchor=(.53, .998),
               columnspacing=2.4, handlelength=2.7)
    for j, no in enumerate([50, 62, 71, 88]):
        ax = fig.add_subplot(gs[j//2, j%2])
        s = d[d.No == no].sort_values('Time')
        ax.plot(s.Time, s.Experimental, color='#212529', lw=1.9, zorder=3)
        ax.plot(s.Time, s.Version_A, color=BLUE, ls='--', lw=1.6, zorder=4)
        ax.plot(s.Time, s.Version_B, color=RED, ls='-.', lw=1.6, zorder=4)
        title(ax, 'abcd'[j], f'Curve {no}')
        ax.set(xlabel='Time (s)', ylabel='Torque (kgf·cm)')
        axis(ax)
        ax.margins(x=.015, y=.09)
        ax.text(.98, .06,
                f'A: $R^2$ = {ma.loc[no,"R2"]:.3f}, RMSE = {ma.loc[no,"RMSE"]:.3f}\n'
                f'B: $R^2$ = {mb.loc[no,"R2"]:.3f}, RMSE = {mb.loc[no,"RMSE"]:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7.8,
                linespacing=1.6)
    for j, (key, label) in enumerate([('R2', r'Mean curve-level $R^2$'),
                                     ('RMSE', 'Mean curve-level RMSE')]):
        ax = fig.add_subplot(gs[2, j])
        meanbars(ax, [ma[key].mean(), mb[key].mean()], [BLUE, RED],
                 ['Version A', 'Version B'], fmt='.4f')
        title(ax, 'ef'[j], label)
        ax.set_ylabel(r'$R^2$' if key == 'R2' else 'RMSE (kgf·cm)')
    save(fig, 'Figure_05_reconstruction_and_accuracy')


def alpha_ablation():
    d = read('Stage5_alpha0_Ablation/alpha0_paired_comparison.csv')
    d = d[d.Version == 'B_pred_ML_MH'].sort_values('No')
    assert len(d) == d.No.nunique() == 88
    delta = d.RMSE_zero - d.RMSE_median
    assert np.allclose(delta, d.RMSE_reduction)
    fig = plt.figure(figsize=(8.4, 6.1))
    gs = fig.add_gridspec(2, 6, height_ratios=[.9, 1.2],
                          left=.085, right=.985, bottom=.105, top=.94,
                          hspace=.62, wspace=1.40)
    labs = [r'$\alpha_0=0$', 'Training-fold\nmedian '+r'$\alpha_0$']
    for j, (key, label) in enumerate([('R2', r'Mean $R^2$'), ('RMSE', 'Mean RMSE'),
                                     ('MAE', 'Mean MAE')]):
        ax = fig.add_subplot(gs[0, 2*j:2*j+2])
        meanbars(ax, [d[f'{key}_zero'].mean(), d[f'{key}_median'].mean()],
                 [ALPHA_ORANGE, ALPHA_BLUE], labs,
                 fmt='.4f' if key == 'R2' else '.3f')
        ax.tick_params(axis='x', labelsize=8)
        title(ax, 'abc'[j], label)
        ax.set_ylabel(r'$R^2$' if key == 'R2' else f'{key} (kgf·cm)')
    ax = fig.add_subplot(gs[1, :3])
    boxpoints(ax, [d.RMSE_zero, d.RMSE_median], [ALPHA_ORANGE, ALPHA_BLUE], labs)
    title(ax, 'd', 'Curve-level RMSE distribution')
    ax.set(ylabel='RMSE (kgf·cm)', ylim=(0, max(d.RMSE_zero.max(), d.RMSE_median.max())*1.12))
    ax = fig.add_subplot(gs[1, 3:])
    axis(ax)
    ax.hist(delta, bins=18, color=GREEN, edgecolor='white', linewidth=.8)
    ax.axvline(0, ls='--', color='#C94C4C', lw=1.4, label='No change')
    title(ax, 'e', 'Per-curve RMSE reduction')
    ax.set(xlabel='RMSE reduction (kgf·cm)', ylabel='Number of curves')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    ax.legend(loc='upper left', fontsize=8)
    fig.text(.535, .016, 'Version B  ·  88 held-out curves  ·  64/88 curves improved  ·  Mean RMSE reduced by 7.39%',
             ha='center', color='#425466', fontsize=8.5)
    save(fig, 'Figure_06_induction_boundary_ablation')


def direct_comparison():
    d = read('Stage8_Output/pointwise_predictions.csv')
    m = read('Stage8_Output/curve_metrics_all_methods.csv')
    assert d.No.nunique() == 88 and len(m) == 264
    fig = plt.figure(figsize=(8.4, 6.65))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.17],
                          left=.085, right=.985, bottom=.075, top=.925,
                          hspace=.55, wspace=.30)
    legend = [Line2D([], [], color=BLUE, lw=1.8, label='Experimental')]
    for method, ls in [('Direct-MLP', ':'), ('Direct-XGB', '--'), ('PG-ML', '-.')]:
        legend.append(Line2D([], [], color=DIRECT[method], ls=ls, lw=1.8, label=method))
    fig.legend(handles=legend, ncol=4, loc='upper center', bbox_to_anchor=(.535, .998),
               columnspacing=1.7, handlelength=2.5)
    for j, (no, case) in enumerate([(51, 'Low-error example'), (10, 'Median-error example')]):
        ax = fig.add_subplot(gs[0, j])
        s = d[d.No == no].sort_values('Time')
        ax.plot(s.Time, s.Experimental, color=BLUE, lw=1.8)
        for method, col, ls in [('Direct-MLP', 'Direct_MLP', ':'),
                                ('Direct-XGB', 'Direct_XGB', '--'), ('PG-ML', 'PG_ML', '-.')]:
            ax.plot(s.Time, s[col], color=DIRECT[method], ls=ls, lw=1.7)
        title(ax, 'ab'[j], f'{case}: curve {no}')
        ax.set(xlabel='Time (s)', ylabel='Torque (kgf·cm)')
        axis(ax)
        ax.margins(x=.015, y=.09)
        lines = []
        for method in ['Direct-MLP', 'Direct-XGB', 'PG-ML']:
            r = m[(m.No == no) & (m.Method == method)].iloc[0]
            lines.append(f'{method}: $R^2$ = {r.R2:.3f}, RMSE = {r.RMSE:.3f}')
        ax.text(.98, .045, '\n'.join(lines), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7.1, linespacing=1.55)
    ax = fig.add_subplot(gs[1, :])
    methods = ['Direct-MLP', 'Direct-XGB', 'PG-ML']
    groups = [m.loc[m.Method == k, 'RMSE'].to_numpy() for k in methods]
    boxpoints(ax, groups, [DIRECT[k] for k in methods], methods, means=True)
    title(ax, 'c', 'RMSE distribution across 88 held-out curves')
    ax.set(ylabel='RMSE (kgf·cm)', ylim=(-.6, 25))
    for i, g in enumerate(groups, 1):
        ax.text(i, 22.6, f'Mean = {np.mean(g):.3f}\nMedian = {np.median(g):.3f}',
                ha='center', va='center', fontsize=8.4, linespacing=1.5)
    ax.text(.99, .48, 'Each point: one curve\n◇ Mean', transform=ax.transAxes,
            ha='right', va='center', fontsize=8.1, linespacing=1.6)
    save(fig, 'Figure_11_direct_and_kinetics_guided_predictions')


def main():
    OUT.mkdir(exist_ok=True)
    style()
    reconstruction()
    alpha_ablation()
    direct_comparison()
    for rel, digest in INPUTS.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
    (OUT/'source_manifest.json').write_text(json.dumps({
        'old_to_new_figures': {'5+6': 5, '7+8': 6, '13+14': 11},
        'source_sha256': INPUTS, 'dpi': 600,
        'models_retrained': False, 'source_data_modified': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved three merged figures; source hashes unchanged.')


if __name__ == '__main__':
    main()
