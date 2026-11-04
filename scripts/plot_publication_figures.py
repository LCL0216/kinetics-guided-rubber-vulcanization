#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw the 16 manuscript figures from saved results using one journal style.

Run with the existing scientific environment:
  python scripts/plot_publication_figures.py

No prediction model or numerical result is overwritten. The two descriptive
SHAP explainers are reconstructed with the original Stage7C settings, and their
mean absolute attributions must match the saved original summary. All other
plots read saved measurements, predictions and metrics. Source hashes are
checked again before completion. Output: 600 dpi PNG and editable PDF/SVG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("VULCANIZATION_PROJECT_ROOT", str(Path(__file__).resolve().parents[1] / "work"))).resolve()
NAVY, BLUE, TEAL, ORANGE = '#203B4D', '#4D86B6', '#2D8F92', '#E77A5A'
GRAY, GRID, PALE_BLUE, PALE_TEAL = '#657A86', '#E5EBEE', '#DCEAF5', '#E0F0E9'
METHOD_COLORS = {'Direct-MLP': BLUE, 'Direct-XGB': ORANGE, 'PG-ML': TEAL}
METHOD_LINES = {'Direct-MLP': ':', 'Direct-XGB': '-.', 'PG-ML': '--'}
FEATURES = ['mixing_time', 'mixing_temp', 'curing_temp', 'curing_pressure',
            'rotor_speed_1', 'rotor_speed_2', 'curing_time_offset']
FEATURE_LABELS = {'mixing_time': 'Mixing time', 'mixing_temp': 'Mixing temperature',
                  'curing_temp': 'Curing temperature', 'curing_pressure': 'Curing pressure',
                  'rotor_speed_1': 'Rotor speed 1', 'rotor_speed_2': 'Rotor speed 2',
                  'curing_time_offset': 'Curing-time offset'}
PARAMS = {'ML': r'$M_L$', 'MH': r'$M_H$', 'ti': r'$t_i$', 'tp': r'$t_p$',
          'alpha0': r'$\alpha_0$', 'b': r'$b$', 'k1': r'$k_1$', 'n1': r'$n_1$'}
INPUTS, CHECKS, OUTPUTS = {}, {}, []
OUT, DPI = None, 600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    p = ROOT / path
    INPUTS[str(p.relative_to(ROOT))] = sha(p)
    return p


def read(path, **kwargs):
    return pd.read_csv(source(path), **kwargs)


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'mathtext.fontset': 'dejavusans', 'axes.labelsize': 9,
        'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'text.color': NAVY, 'axes.labelcolor': NAVY, 'axes.edgecolor': GRAY,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'xtick.color': GRAY, 'ytick.color': GRAY,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'legend.fontsize': 8, 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'lines.solid_capstyle': 'round',
    })


def axis(ax, grid='y'):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.55)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))


def title(ax, letter, text):
    ax.set_title(f'({letter})  {text}', loc='left', pad=11)


def save(fig, no, name):
    stem = OUT / f'Figure_{no:02d}_{name}'
    fig.canvas.draw()
    for fmt in ['png', 'pdf', 'svg']:
        p = stem.with_suffix('.' + fmt)
        fig.savefig(p, dpi=DPI, bbox_inches='tight', pad_inches=.055)
    OUTPUTS.append({'figure': no, 'stem': stem.name,
                    'png_sha256': sha(stem.with_suffix('.png'))})
    plt.close(fig)
    print(f'Figure {no:02d} complete: {stem.name}', flush=True)


def handles(names, colors_, styles=None):
    return [Line2D([], [], color=c, lw=1.7, ls=(styles or ['-'] * len(names))[i], label=n)
            for i, (n, c) in enumerate(zip(names, colors_))]


def meanbars(ax, vals, cols, labels, fmt='.4f', errors=None):
    pos = np.arange(len(vals))
    ax.bar(pos, vals, width=.57, color=cols, alpha=.9, edgecolor='white', lw=.7,
           yerr=errors, error_kw=dict(ecolor=NAVY, capsize=3, elinewidth=.85, capthick=.85))
    ax.set_xticks(pos, labels)
    ax.yaxis.grid(True, color=GRID, lw=.55)
    ax.set_axisbelow(True)
    for i, val in enumerate(vals):
        offset = float(errors[i]) if errors is not None else 0
        ax.annotate(format(val, fmt), (i, val + offset), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=8.2, color=NAVY)


def figure1():
    """A code-native seven-stage schematic; mini-curves use saved observations."""
    fig = plt.figure(figsize=(7.2, 4.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 7.2), ylim=(0, 4.9)); ax.axis('off')
    w, h, top_y, bottom_y = 1.54, 1.74, 2.83, .49
    xs = [.08, 1.92, 3.76, 5.60]
    nodes = {
        1: (xs[0], top_y, BLUE, 'Experimental data', '88 modeling curves\n7 process variables'),
        2: (xs[1], top_y, BLUE, 'Preprocessing', 'Savitzky–Golay filter\nInduction / curing'),
        3: (xs[2], top_y, TEAL, 'Kinetic descriptors', 'Compact, interpretable\nparameter representation'),
        4: (xs[3], top_y, TEAL, 'Parameter prediction', 'RF / XGBoost\nFive predicted targets'),
        5: (5.60, bottom_y, TEAL, 'Curve reconstruction', 'Continuous prediction\nInduction + curing'),
        6: (2.84, bottom_y, BLUE, 'Internal assessment', 'Metrics and ablation\nDirect-ML comparison'),
        7: (.08, bottom_y, ORANGE, 'Additional validation', '88-curve frozen models\n6 evaluation-only curves'),
    }
    for n, (x, y, col, heading, body) in nodes.items():
        ax.add_patch(Rectangle((x, y), w, h, fc=colors.to_rgba(col, .055), ec=colors.to_rgba(col, .45), lw=.7))
        ax.text(x + .10, y + h - .18, f'{n:02d}', fontsize=8.5, fontweight='bold', color=col)
        ax.text(x + .1, y + h - .48, heading, fontsize=8.1, fontweight='bold')
        ax.text(x + .1, y + h - .76, body, fontsize=7.5, va='top', linespacing=1.5)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i] + w + .025, top_y + h / 2), (xs[i+1] - .025, top_y + h / 2),
                                    arrowstyle='-|>', mutation_scale=10, lw=.9, color=GRAY))
    ax.add_patch(FancyArrowPatch((6.37, top_y - .035), (6.37, bottom_y + h + .03),
                                arrowstyle='-|>', mutation_scale=11, lw=.9, color=GRAY))
    for xr, xl in [(5.60, 2.84), (2.84, .08)]:
        ax.add_patch(FancyArrowPatch((xr - .04, bottom_y + h / 2), (xl + w + .04, bottom_y + h / 2),
                                    arrowstyle='-|>', mutation_scale=11, lw=.9, color=GRAY))
    ax.text(3.86, 3.05, r'$M_L,\ M_H,\ t_i,\ \alpha_0$' + '\n' + r'$k_1,\ n_1$', fontsize=9.8, linespacing=1.6)
    ax.text(5.71, 3.19, r'$M_L,\ M_H,\ t_i,\ k_1,\ n_1$' + '\n' + r'$\alpha_0$: training-fold median', fontsize=7.4, linespacing=1.8)
    ax.text(2.94, .78, r'$R^2$ · RMSE · MAE' + '\n' + 'SHAP / permutation', fontsize=7.2, linespacing=1.7)
    ax.text(.1, .17, 'Kinetic guidance enters through parameterization and explicit reconstruction equations.', fontsize=8.2)
    p8 = read('Stage8_Output/pointwise_predictions.csv')
    p9 = read('Stage9_一键预测/Stage9_Output/stage9_pointwise_predictions.csv')
    for n, frame, ids, pred in [(1, p8, [2, 10, 50], None), (2, p8, [1], None),
                                 (5, p8, [50], 'PG_ML'), (7, p9, [2, 5], 'PG_ML_pred')]:
        x, y, *_ = nodes[n]
        mini = fig.add_axes([(x+.12)/7.2, (y+.13)/4.9, (w-.25)/7.2, .55/4.9])
        for j, no in enumerate(ids):
            g = frame.loc[frame.No == no]
            mini.plot(g.Time / g.Time.max(), g.Experimental, color=NAVY if pred else [BLUE, TEAL, ORANGE][j], lw=.85)
            if pred:
                mini.plot(g.Time / g.Time.max(), g[pred], color=TEAL, ls='--', lw=.85)
        mini.set(xticks=[], yticks=[])
        mini.spines[['left', 'bottom']].set_color('#C5D2DA')
    save(fig, 1, 'workflow')


def figure2():
    from stage01_extract_characteristics import extract_one_curve, plot_curve
    raw = read('model_ready_data.csv', low_memory=False)
    row, data = extract_one_curve(1, raw.loc[pd.to_numeric(raw.No) == 1].copy())
    expected = read('Stage1_Output/kinetic_parameters_stage1.csv').set_index('No').loc[1]
    for key, value in row.items():
        if key in ['No', 'failure_reason']: continue
        assert np.isclose(value, expected[key], rtol=1e-10, atol=1e-10), key
    CHECKS['figure2_parameters'] = 'All saved extraction parameters reproduced'
    p = OUT / 'Figure_02_characteristic_extraction.png'
    plot_curve(1, data, p, save_vector=True)
    OUTPUTS.append({'figure': 2, 'stem': p.stem, 'png_sha256': sha(p)})
    print('Figure 02 complete: characteristic extraction', flush=True)


def figure3():
    d = read('Stage2_Output/kinetic_parameters_stage2.csv')
    assert len(d) == 88
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.85))
    fig.subplots_adjust(left=.085, right=.98, top=.85, bottom=.20, wspace=.33)
    for ax, key, col, label, letter in zip(axs, ['R2', 'RMSE'], [BLUE, TEAL],
            [r'Kinetic-fit $R^2$', r'Fit RMSE in normalized $\alpha$'], 'ab'):
        axis(ax)
        counts, _, _ = ax.hist(d[key], bins=18, color=col, edgecolor='white', lw=.65, alpha=.87)
        ax.set_ylim(0, max(counts)*1.25)  # keep the mean legend above every bar
        ax.axvline(d[key].mean(), color=ORANGE, ls='--', lw=1.2, label=f'Mean = {d[key].mean():.4f}')
        ax.set(xlabel=label, ylabel='Number of curves')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        title(ax, letter, 'Fitting accuracy' if key == 'R2' else 'Fitting error')
        ax.legend(loc='upper right', fontsize=7.5)
    save(fig, 3, 'kinetic_fit_distribution')


def figure4():
    d = read('Stage4B_Output/best_model_per_target.csv').set_index('Target').loc[['ML', 'MH', 'ti', 'k1', 'n1']]
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    fig.subplots_adjust(left=.10, right=.985, bottom=.23, top=.90)
    axis(ax)
    meanbars(ax, d.R2_mean.to_numpy(), [BLUE, BLUE, TEAL, TEAL, TEAL],
             [PARAMS[t] + '\n' + d.loc[t, 'Model'] for t in d.index], fmt='.3f', errors=d.R2_std.to_numpy())
    ax.axhline(0, color=GRAY, lw=.7)
    ax.set(ylim=(-.5, 1.18), ylabel=r'Mean held-out $R^2$ ± SD')
    ax.text(.99, .04, '5-fold CV × 5 repetitions', ha='right', transform=ax.transAxes, fontsize=8, color=GRAY)
    save(fig, 4, 'selected_parameter_models')


def figure5():
    d = read('Stage5_Output_alpha0_comparison/pointwise_AB_predictions.csv')
    ma = read('Stage5_Output_alpha0_comparison/curve_metrics_A.csv').set_index('No')
    mb = read('Stage5_Output_alpha0_comparison/curve_metrics_B.csv').set_index('No')
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.95))
    fig.subplots_adjust(left=.085, right=.98, top=.84, bottom=.10, hspace=.61, wspace=.28)
    fig.legend(handles=handles(['Experimental', 'Version A', 'Version B'], [NAVY, BLUE, TEAL], ['-', ':', '--']),
               ncol=3, loc='upper center', bbox_to_anchor=(.52, .995), columnspacing=2.0)
    for ax, no, letter in zip(axs.flat, [50, 62, 71, 88], 'abcd'):
        g = d.loc[d.No == no].sort_values('Time'); axis(ax)
        for key, color, ls in [('Experimental', NAVY, '-'), ('Version_A', BLUE, ':'), ('Version_B', TEAL, '--')]:
            ax.plot(g.Time, g[key], color=color, ls=ls, lw=1.45)
        ax.set(xlabel='Time (s)', ylabel='Torque (kgf·cm)')
        title(ax, letter, f'Curve {no}')
        ax.text(.98, .09, f'A: $R^2$ = {ma.loc[no,"R2"]:.3f}; RMSE = {ma.loc[no,"RMSE"]:.3f}\n'
                f'B: $R^2$ = {mb.loc[no,"R2"]:.3f}; RMSE = {mb.loc[no,"RMSE"]:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7.6, linespacing=1.5)
    save(fig, 5, 'curve_reconstruction_examples')


def figure6():
    d = read('Stage5_Output_alpha0_comparison/curve_metrics_summary.csv').set_index('Version').loc[['A_true_ML_MH', 'B_pred_ML_MH']]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.85))
    fig.subplots_adjust(left=.085, right=.98, bottom=.19, top=.83, wspace=.32)
    for ax, key, lab, letter in zip(axs, ['Mean_R2', 'Mean_RMSE'], [r'Mean curve-level $R^2$', 'Mean RMSE (kgf·cm)'], 'ab'):
        axis(ax)
        vals = d[key].to_numpy()
        meanbars(ax, vals, [BLUE, TEAL], ['Version A', 'Version B'])
        ax.set(ylim=(0, max(vals)*1.2), ylabel=lab)
        title(ax, letter, 'Reconstruction accuracy' if key == 'Mean_R2' else 'Reconstruction error')
    save(fig, 6, 'version_A_B_metrics')


def alpha_data():
    d = read('Stage5_alpha0_Ablation/alpha0_paired_comparison.csv')
    return d.loc[d.Version == 'B_pred_ML_MH'].sort_values('No')


def figure7():
    d = alpha_data()
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.95))
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.84, wspace=.40)
    for ax, key, lab, letter in zip(axs, ['R2', 'RMSE', 'MAE'], [r'Mean $R^2$', 'RMSE (kgf·cm)', 'MAE (kgf·cm)'], 'abc'):
        vals = [d[f'{key}_zero'].mean(), d[f'{key}_median'].mean()]
        axis(ax)
        meanbars(ax, vals, [ORANGE, TEAL], [r'$\alpha_0 = 0$', 'Training-fold\nmedian '+r'$\alpha_0$'], fmt='.4f' if key == 'R2' else '.3f')
        ax.set(ylim=(0, max(vals)*1.22), ylabel=lab)
        title(ax, letter, r'Mean $R^2$' if key == 'R2' else 'Mean '+key)
        ax.tick_params(axis='x', labelsize=7.4)
    save(fig, 7, 'alpha0_mean_metrics')


def boxpoints(ax, groups, cols, labels, seed=42, paired=False):
    rng = np.random.default_rng(seed)
    pos = np.arange(len(groups))
    if paired:
        for row in np.asarray(groups).T:
            ax.plot(pos, row, color='#B8C5CC', lw=.5, alpha=.4, zorder=1)
    bp = ax.boxplot(groups, positions=pos, widths=.43, patch_artist=True, showfliers=False,
                    medianprops=dict(color=NAVY, lw=1.35),
                    whiskerprops=dict(color=GRAY, lw=.8), capprops=dict(color=GRAY, lw=.8))
    for patch, col in zip(bp['boxes'], cols):
        patch.set(facecolor=colors.to_rgba(col, .20), edgecolor=col, lw=1)
    for i, (vals, col) in enumerate(zip(groups, cols)):
        ax.scatter(rng.uniform(i-.15, i+.15, len(vals)), vals, s=10, color=col,
                   alpha=.65, edgecolors='white', linewidths=.25, zorder=3)
        ax.scatter(i, np.mean(vals), s=31, marker='D', color=NAVY, edgecolors='white', linewidths=.6, zorder=5)
    ax.set_xticks(pos, labels)
    ax.set_xlim(-.55, len(groups)-.45)


def figure8():
    d = alpha_data()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(left=.085, right=.98, bottom=.23, top=.84, wspace=.33)
    axis(axs[0]); axis(axs[1])
    boxpoints(axs[0], [d.RMSE_zero, d.RMSE_median], [ORANGE, TEAL],
              [r'$\alpha_0 = 0$', 'Training-fold\nmedian '+r'$\alpha_0$'])
    axs[0].set_ylabel('Curve RMSE (kgf·cm)'); axs[0].set_ylim(bottom=0)
    title(axs[0], 'a', 'Curve-level error distribution')
    extent = float(abs(d.RMSE_reduction).max())
    bins = np.linspace(-extent, extent, 19)  # zero is an exact bin boundary
    counts, edges = np.histogram(d.RMSE_reduction, bins=bins)
    mids = (edges[1:] + edges[:-1])/2
    axs[1].bar(mids, counts, width=np.diff(edges)*.95,
                color=[TEAL if x >= 0 else ORANGE for x in mids], edgecolor='white', lw=.4)
    axs[1].axvline(0, color=NAVY, ls='--', lw=1)
    axs[1].set(xlabel='RMSE reduction (kgf·cm)', ylabel='Number of curves')
    axs[1].set_ylim(0, max(counts)*1.25)
    title(axs[1], 'b', 'Paired improvement')
    axs[1].text(.97, .96, f'{int((d.RMSE_reduction > 0).sum())}/88 improved',
                transform=axs[1].transAxes, ha='right', va='top', color=TEAL, fontsize=8.5)
    save(fig, 8, 'alpha0_paired_distribution')


def figure9():
    d = read('Stage7A_Output/spearman_X_vs_parameters.csv', index_col=0)
    cmap = colors.LinearSegmentedColormap.from_list('blue_white_orange', [BLUE, '#F8FAFC', ORANGE])
    fig, ax = plt.subplots(figsize=(7.2, 3.95))
    fig.subplots_adjust(left=.255, right=.885, top=.96, bottom=.15)
    im = ax.imshow(d.to_numpy(), vmin=-1, vmax=1, cmap=cmap, aspect='auto')
    ax.set_xticks(np.arange(len(d.columns)), [PARAMS[c] for c in d.columns])
    ax.set_yticks(np.arange(len(d.index)), [FEATURE_LABELS[c] for c in d.index])
    ax.tick_params(length=0, pad=8)
    for i in range(len(d)):
        for j in range(len(d.columns)):
            ax.text(j, i, f'{d.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(d.iloc[i,j])>.73 else NAVY)
    ax.set_xticks(np.arange(-.5, len(d.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(d.index), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0); ax.tick_params(which='minor', length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cax = fig.add_axes([.91, .18, .018, .75])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -.5, 0, .5, 1]); cb.set_label('Spearman '+r'$\rho$', labelpad=8)
    cb.outline.set_visible(False)
    save(fig, 9, 'process_descriptor_correlations')


def figure10():
    d = read('Stage7B_Output/ablation_summary.csv')
    order = ['CuringTemp_Only', 'All_Process', 'Without_CuringTemp']
    # Saved identifiers are checked rather than inferred from plotted pixels.
    if 'Without_CuringTemp' not in set(d.Feature_Set):
        order[-1] = next(s for s in d.Feature_Set.unique() if 'Without' in s)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.35))
    fig.subplots_adjust(left=.085, right=.98, top=.84, bottom=.24, wspace=.33)
    for ax, target, letter in zip(axs, ['ti', 'k1'], 'ab'):
        g = d.loc[d.Target == target].set_index('Feature_Set').loc[order]
        axis(ax)
        meanbars(ax, g.R2_mean.to_numpy(), [BLUE, TEAL, '#9EAEB9'],
                  ['Curing temperature\nonly', 'All process\nvariables', 'Without curing\ntemperature'],
                  fmt='.3f', errors=g.R2_std.to_numpy())
        ax.axhline(0, color=GRAY, lw=.8)
        low = min(0, float((g.R2_mean-g.R2_std).min()))
        high = max(1.05, float((g.R2_mean+g.R2_std).max())+.25)
        ax.set(ylim=(low-.15, high), ylabel=r'Mean held-out $R^2$ ± SD')
        ax.tick_params(axis='x', labelsize=7.1)
        title(ax, letter, PARAMS[target]+' feature ablation')
    save(fig, 10, 'curing_temperature_ablation')


def shap_values():
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    d = read('Stage3_Output/pgmlp_curve_dataset.csv')
    expected = read('Stage7C_Output/SHAP_importance_summary.csv')
    X = d[FEATURES].copy()
    records = {}
    for target in ['ti', 'MH']:
        model = (RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=1)
                 if target == 'MH' else XGBRegressor(objective='reg:squarederror', n_estimators=300,
                    max_depth=3, learning_rate=.03, subsample=.8, colsample_bytree=.8,
                    min_child_weight=2, reg_alpha=.01, reg_lambda=1., random_state=42, n_jobs=1))
        model.fit(X, d[target].to_numpy(float))
        values = np.asarray(shap.TreeExplainer(model).shap_values(X))
        exp = expected.loc[expected.Target == target].set_index('Feature').loc[FEATURES, 'Mean_abs_SHAP'].to_numpy()
        delta = float(np.max(abs(np.abs(values).mean(axis=0)-exp)))
        assert delta < 1e-6, (target, delta)
        CHECKS['SHAP_'+target+'_max_summary_difference'] = delta
        records[target] = values
    return X, records


def bee_offsets(values):
    """Deterministic symmetric stacking; x coordinates remain exact SHAP values."""
    span = max(np.ptp(values), 1e-12)
    bins = np.floor((values-values.min())/span*38).astype(int)
    out = np.zeros(len(values))
    for b in np.unique(bins):
        indices = np.flatnonzero(bins == b)
        for k, i in enumerate(indices):
            out[i] = (0 if k == 0 else ((k+1)//2)*(1 if k%2 else -1))*.065
    mx = max(abs(out).max(), .35)
    return out*min(1, .33/mx)


def figure11():
    X, records = shap_values()
    cmap = colors.LinearSegmentedColormap.from_list('feature_value', [BLUE, '#AEC7CD', ORANGE])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.65))
    fig.subplots_adjust(left=.235, right=.92, top=.85, bottom=.19, wspace=1.30)
    for ax, target, letter in zip(axs, ['ti', 'MH'], 'ab'):
        v = records[target]
        order = np.argsort(np.abs(v).mean(axis=0))[::-1]
        ax.axvline(0, color='#BCC9D0', lw=.8)
        for row, feature in enumerate(order):
            raw = X.iloc[:, feature].to_numpy(float)
            norm = (raw-raw.min()) / max(np.ptp(raw), 1e-12)
            ax.axhline(row, lw=.5, color=GRID, ls=':', zorder=0)
            ax.scatter(v[:, feature], row + bee_offsets(v[:, feature]), c=norm, cmap=cmap,
                        vmin=0, vmax=1, s=10.5, alpha=.88, edgecolors='none', zorder=3)
        ax.set_yticks(range(len(order)), [FEATURE_LABELS[FEATURES[i]] for i in order])
        ax.set_ylim(len(order)-.55, -.55)
        ax.tick_params(axis='y', length=0, labelsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('SHAP value'+(' (s)' if target=='ti' else ' (kgf·cm)'), fontsize=8)
        title(ax, letter, PARAMS[target]+' attribution')
    cbax = fig.add_axes([.952, .20, .012, .62])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0,1), cmap=cmap), cax=cbax, ticks=[0,1])
    cb.ax.set_yticklabels(['Low', 'High']); cb.ax.tick_params(length=0, labelsize=7.5)
    cb.set_label('Feature value', fontsize=8, labelpad=4); cb.outline.set_visible(False)
    save(fig, 11, 'SHAP_attribution')


def figure12():
    d = read('Stage7C_Output/permutation_importance_summary.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.55))
    fig.subplots_adjust(left=.235, right=.98, top=.85, bottom=.20, wspace=1.18)
    for ax, target, letter in zip(axs, ['ti', 'MH'], 'ab'):
        g = d.loc[d.Target == target].sort_values('Importance_mean', ascending=False)
        yy = np.arange(len(g)); axis(ax, 'x')
        ax.barh(yy, g.Importance_mean, height=.65, color=[TEAL]+[BLUE]*6, alpha=.9,
                 xerr=g.Importance_std, error_kw=dict(ecolor=NAVY, capsize=2, elinewidth=.7, capthick=.7))
        ax.set_yticks(yy, [FEATURE_LABELS[f] for f in g.Feature]); ax.invert_yaxis()
        ax.tick_params(axis='y', length=0, labelsize=8)
        ax.axvline(0, color=GRAY, lw=.75); ax.spines['left'].set_visible(False)
        ax.set_xlabel(r'Decrease in held-out $R^2$', fontsize=8)
        title(ax, letter, 'Induction time, '+PARAMS[target] if target=='ti' else 'Maximum torque, '+PARAMS[target])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    save(fig, 12, 'permutation_importance')


def figure13():
    d = read('Stage8_Output/pointwise_predictions.csv')
    m = read('Stage8_Output/curve_metrics_all_methods.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=.085, right=.98, top=.78, bottom=.18, wspace=.28)
    names = ['Experimental', 'Direct-MLP', 'Direct-XGB', 'PG-ML']
    fig.legend(handles=handles(names, [NAVY, BLUE, ORANGE, TEAL], ['-', ':', '-.', '--']), ncol=4,
               loc='upper center', bbox_to_anchor=(.53, 1), columnspacing=1.3, fontsize=7.8)
    for ax, no, letter, name in zip(axs, [51,10], 'ab', ['Low-error case', 'Median-error case']):
        axis(ax);g=d.loc[d.No==no].sort_values('Time')
        for key, color, ls in [('Experimental', NAVY, '-'), ('Direct_MLP', BLUE, ':'), ('Direct_XGB', ORANGE, '-.'), ('PG_ML', TEAL, '--')]:
            ax.plot(g.Time, g[key], color=color, ls=ls, lw=1.5)
        ax.set(xlabel='Time (s)', ylabel='Torque (kgf·cm)')
        title(ax, letter, f'{name}: curve {no}')
        texts=[]
        for meth in ['Direct-MLP','Direct-XGB','PG-ML']:
            r=m.loc[(m.No==no)&(m.Method==meth)].iloc[0]
            texts.append(f'{meth}: $R^2$ {r.R2:.3f}; RMSE {r.RMSE:.3f}')
        ax.text(.98,.08,'\n'.join(texts),transform=ax.transAxes,ha='right',fontsize=7.5,linespacing=1.45)
    save(fig, 13, 'direct_and_kinetic_examples')


def figure14():
    d = read('Stage8_Output/curve_metrics_all_methods.csv')
    summary = read('Stage8_Output/method_summary.csv').set_index('Method')
    names = ['Direct-MLP','Direct-XGB','PG-ML']
    groups = [d.loc[d.Method==n].sort_values('No').RMSE.to_numpy() for n in names]
    for n,g in zip(names,groups):
        assert len(g)==88 and np.isclose(np.mean(g),summary.loc[n,'Mean_RMSE'])
    fig, ax=plt.subplots(figsize=(7.2,4.3))
    fig.subplots_adjust(left=.095,right=.98,bottom=.17,top=.82)
    axis(ax);boxpoints(ax,groups,[METHOD_COLORS[n] for n in names],names)
    ax.set(ylabel='Curve RMSE (kgf·cm)',ylim=(0,max(map(max,groups))*1.12))
    for i,n in enumerate(names):
        s=summary.loc[n]
        ax.text(i,1.025,f'Mean {s.Mean_RMSE:.3f}\nMedian {s.Median_RMSE:.3f}',
                transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8,linespacing=1.4)
    ax.legend(handles=[Line2D([],[],marker='D',color=NAVY,ls='none',markersize=4,label='Mean'),
                        Line2D([],[],color=NAVY,lw=1.4,label='Median')],
              ncol=2,loc='upper right',bbox_to_anchor=(1,1.23),fontsize=7.5)
    save(fig,14,'held_out_error_distribution')


def figure15():
    d=read('Stage9_一键预测/Stage9_Output/stage9_pointwise_predictions.csv')
    m=read('Stage9_一键预测/Stage9_Output/stage9_curve_metrics.csv').sort_values('No')
    assert len(m)==6
    for no in range(1,7):
        g=d.loc[d.No==no];r=m.loc[m.No==no].iloc[0]
        rmse=float(np.sqrt(np.mean((g.PG_ML_pred-g.Experimental)**2)))
        assert np.isclose(rmse,r.RMSE,rtol=1e-10)
    fig=plt.figure(figsize=(7.4,7.05))
    outer=fig.add_gridspec(3,3,left=.08,right=.98,bottom=.085,top=.91,
                          height_ratios=[2.3,2.3,1.65],hspace=.55,wspace=.36)
    fig.legend(handles=handles(['Experimental','PG-ML prediction'],[NAVY,TEAL],['-','--']),
               ncol=2,loc='upper center',bbox_to_anchor=(.53,.995),fontsize=8.5,columnspacing=2.5)
    for i,no in enumerate(range(1,7)):
        inner=outer[i//3,i%3].subgridspec(2,1,height_ratios=[3.4,1],hspace=.10)
        ax=fig.add_subplot(inner[0]);ar=fig.add_subplot(inner[1],sharex=ax)
        g=d.loc[d.No==no].sort_values('Time');r=m.loc[m.No==no].iloc[0]
        axis(ax);axis(ar,None)
        ax.plot(g.Time,g.Experimental,color=NAVY,lw=1.25)
        ax.plot(g.Time,g.PG_ML_pred,color=TEAL,lw=1.35,ls='--')
        ax.tick_params(labelbottom=False,labelsize=7)
        ax.set_ylim(bottom=min(g.Experimental.min(),g.PG_ML_pred.min())-.7)
        title(ax,chr(97+i),f'Curve {no}')
        ax.title.set_fontsize(8.5)
        ax.text(.98,.08,f'$R^2$ {r.R2:.3f}\nRMSE {r.RMSE:.2f}',transform=ax.transAxes,
                ha='right',va='bottom',fontsize=7,linespacing=1.4)
        residual=g.PG_ML_pred-g.Experimental
        ar.plot(g.Time,residual,color=BLUE,lw=.8)
        ar.fill_between(g.Time,residual,0,color=BLUE,alpha=.16)
        ar.axhline(0,color=GRAY,lw=.65,ls='--')
        # Cover every residual, rather than truncating extreme observations.
        lim=max(float(abs(residual).max())*1.1,.5);ar.set_ylim(-lim,lim)
        ar.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ar.tick_params(labelsize=7,length=2)
        ar.set_xlabel('Time (s)',fontsize=7.2,labelpad=2)
        if i%3==0:
            ax.set_ylabel('Torque (kgf·cm)',fontsize=7.5,labelpad=3)
            ar.set_ylabel('Residual',fontsize=7.2,labelpad=3)
    for j,(metric,col,label) in enumerate([('R2',BLUE,r'$R^2$'),('RMSE',ORANGE,'RMSE'),('MAE',TEAL,'MAE')]):
        ax=fig.add_subplot(outer[2,j]);axis(ax,'x')
        ax.barh(np.arange(6),m[metric],height=.6,color=col,alpha=.88)
        mean=m[metric].mean();ax.axvline(mean,color=NAVY,ls='--',lw=.9)
        ax.set_yticks(np.arange(6),[str(i) for i in range(1,7)]);ax.invert_yaxis()
        ax.tick_params(labelsize=7,length=2)
        xmax = float(m[metric].max())
        ax.set_xlim(0,1.27 if metric=='R2' else xmax*1.35)
        if metric=='R2':ax.set_xticks([0,.3,.6,.9])
        else:ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        # A dedicated value column keeps every label clear of the mean line.
        label_x = 1.035 if metric=='R2' else xmax*1.045
        for k,v in enumerate(m[metric]):ax.text(label_x,k,f'{v:.3f}',fontsize=7.2,va='center')
        ax.set_title(label+f'   mean {mean:.3f}',loc='left',fontsize=8.3,pad=9)
        ax.set_xlabel(label if metric=='R2' else 'kgf·cm',fontsize=7.6,labelpad=3)
        ax.spines['left'].set_visible(False)
        if j==0:ax.set_ylabel('Curve',fontsize=8)
    save(fig,15,'six_additional_curves')


def figure16():
    fits=read('Stage10_n1_Ablation_Output/kinetic_fit_by_curve.csv')
    result=read('Stage10_n1_Ablation_Output/curve_metrics.csv')
    free,fixed='Free_n1','Fixed_n1_1_refit_k'
    fig,axs=plt.subplots(1,3,figsize=(7.2,3.2))
    fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.81,wspace=.55)
    a=fits.pivot(index='No',columns='Representation',values='RMSE')
    axis(axs[0]);boxpoints(axs[0],[a[free],a[fixed]],[TEAL,BLUE],
                         [r'Free $n_1$',r'$n_1=1$'+'\nrefitted rate'],paired=True)
    axs[0].set_ylabel('Fit RMSE (dimensionless)',fontsize=8)
    title(axs[0],'a','Observed-curve fit')
    for ax,mode,letter,name in [(axs[1],'Full_forward','b','Forward prediction'),
                               (axs[2],'Measured_amplitude_diagnostic','c','Amplitude diagnostic')]:
        a=result.loc[result.Evaluation==mode].pivot(index='No',columns='Representation',values='RMSE')
        lim=float(a.max().max())*1.08
        axis(ax);ax.plot([0,lim],[0,lim],color=GRAY,lw=.9,ls='--')
        ax.scatter(a[free],a[fixed],s=16,color=TEAL,alpha=.7,edgecolors='white',linewidths=.3)
        ax.set(xlim=(0,lim),ylim=(0,lim))
        ax.set_xlabel(r'Free $n_1$ RMSE (kgf·cm)',fontsize=8)
        ax.set_ylabel(r'$n_1=1$ RMSE (kgf·cm)',fontsize=8,labelpad=3)
        title(ax,letter,name)
    for ax in axs:ax.title.set_fontsize(8.6)
    save(fig,16,'free_and_fixed_exponents')


def main():
    global OUT,DPI
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'Publication_Figures')
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--figures',nargs='+',type=int,default=list(range(1,17)))
    args=p.parse_args();OUT=args.output_dir.resolve();DPI=args.dpi
    OUT.mkdir(parents=True,exist_ok=True);style()
    for n in args.figures:
        if n not in range(1,17):p.error('Figure numbers must be 1–16')
        globals()[f'figure{n}']()
    assert {k:sha(ROOT/k) for k in INPUTS}==INPUTS, 'A numerical source changed'
    manifest=dict(purpose='Publication figure restyling from original scientific results',
        script_sha256=sha(__file__),sources=INPUTS,checks=CHECKS,
        palette=dict(navy=NAVY,blue=BLUE,teal=TEAL,orange=ORANGE),
        method_colors=METHOD_COLORS,outputs=OUTPUTS,dpi=DPI,
        matplotlib_version=matplotlib.__version__,source_results_unchanged=True,
        note='SHAP explanatory models reproduced in memory only; saved prediction models were not changed.')
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
