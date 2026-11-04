#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost
from xgboost import XGBRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
INPUTS = [
    'mixing_time', 'mixing_temp', 'curing_temp', 'curing_pressure',
    'rotor_speed_1', 'rotor_speed_2', 'curing_time_offset',
]
TARGETS = ['ML', 'MH', 'ti', 'k1', 'n1']
EXPECTED_TRAIN = set(range(1, 89))
EXPECTED_VALIDATION = set(range(1, 7))


def arg_parser():
    ap = argparse.ArgumentParser(description='Frozen 88-trained PG-ML -> 6 experimental curves')
    ap.add_argument('--project-root', default=None, help='e.g. /path/to/work')
    ap.add_argument('--mode', choices=['fit', 'predict', 'plot'], default='predict',
                    help='predict (default) loads saved 88-trained models, predicts only 6; plot draws from saved six-curve results; fit is optional one-time training')
    ap.add_argument('--train-stage3', default=None, help='only used for --mode fit')
    ap.add_argument('--model-dir', default=None, help='default: project-root/Stage9_Models_88')
    ap.add_argument('--external-features', default=None, help='6-row process CSV')
    ap.add_argument('--external-curves', default=None, help='6 curves Time/torque CSV')
    ap.add_argument('--output-dir', default=None, help='default: project-root/Stage9_Output')
    ap.add_argument('--no-plots', action='store_true', help='Save CSV only (no figures)')
    ap.add_argument('--fig-output-dir', default=None, help='Defaults to project-root/Stage9_Visual_Figures')
    ap.add_argument('--dpi', type=int, default=300, help='Figure DPI')
    ap.add_argument('--no-pdf', action='store_true', help='Disable vector PDF exports')
    ap.add_argument('--overwrite-models', action='store_true',
                    help='allow replacing already-saved 88-trained models in --mode fit')
    return ap.parse_args()


def to_no(value):
    try:
        z = float(str(value).strip())
        if not np.isfinite(z) or not z.is_integer():
            raise ValueError
        return int(z)
    except (TypeError, ValueError):
        raise ValueError(f'No must be integer; invalid value: {value!r}')


def read_csv(path, required, label):
    df = pd.read_csv(path, encoding='utf-8-sig')
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{label} missing columns: {missing}')
    df = df.copy()
    df['No'] = df['No'].map(to_no)
    for col in required:
        if col == 'No':
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise ValueError(f'{label} contains NA/Inf in {col}')
    return df


def resolve_existing(explicit, choices, label):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'{label}: {p}')
        return p
    for p in choices:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f'{label} not found; specify explicit path. Checked:\n' +
                            '\n'.join(str(p) for p in choices))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def make_model(target):
    if target in ('ML', 'MH'):
        return RandomForestRegressor(n_estimators=300, max_depth=None,
                                     min_samples_split=2, min_samples_leaf=1,
                                     max_features='sqrt', random_state=SEED, n_jobs=-1)
    return XGBRegressor(objective='reg:squarederror', n_estimators=300,
                        max_depth=3, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=2,
                        reg_alpha=0.01, reg_lambda=1.0,
                        random_state=SEED, n_jobs=-1)


def fit_once(root, args, model_dir):
    train_path = resolve_existing(args.train_stage3, [
        root/'Stage3_Output'/'pgmlp_curve_dataset.csv',
        root/'pgmlp_curve_dataset.csv',
        root/'建模集_1-88'/'Stage3_Output'/'pgmlp_curve_dataset.csv',
    ], 'Stage3 88-training table')
    data = read_csv(train_path, ['No', *INPUTS, *TARGETS, 'alpha0'], 'train-88')
    if len(data) != 88 or set(data.No) != EXPECTED_TRAIN or data.No.duplicated().any():
        raise ValueError('Train table must contain exactly No=1..88, once each')
    if (data.k1 <= 0).any():
        raise ValueError('k1 must be positive for log10 training')
    if not data.alpha0.between(0, 1).all():
        raise ValueError('alpha0 must be within [0,1]')

    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir/'model_manifest.json'
    if manifest_path.exists() and not args.overwrite_models:
        raise FileExistsError('Existing final models found. --mode fit must not overwrite them '
                              'unless --overwrite-models is explicitly supplied. '
                              'For validation, use --mode predict instead.')

    X = data[INPUTS].to_numpy(float)
    for target in TARGETS:
        model = make_model(target)
        y = data[target].to_numpy(float)
        model.fit(X, np.log10(y) if target == 'k1' else y)
        if target in ('ML', 'MH'):
            joblib.dump(model, model_dir/f'{target}_RF.joblib', compress=3)
        else:
            model.save_model(str(model_dir/f'{target}_XGB.json'))
        print(f'  Trained & saved: {target}')

    manifest = {
        'description': 'Frozen full-88 final PG-ML (not any 5-fold model)',
        'training_no': '1-88',
        'training_n': 88,
        'training_stage3_sha256': sha256(train_path),
        'training_csv_name': train_path.name,
        'process_features': INPUTS,
        'targets': TARGETS,
        'k1_transform': 'log10 train, power10 inverse',
        'alpha0_train_median': float(data.alpha0.median()),
        'feature_ranges': {c: {'min': float(data[c].min()), 'max': float(data[c].max())} for c in INPUTS},
        'models': {'ML': 'RF300', 'MH': 'RF300', 'ti': 'XGB300',
                   'k1': 'XGB300_log10', 'n1': 'XGB300'},
        'random_state': SEED,
        'versions': {'python': platform.python_version(), 'sklearn': SKLEARN_VERSION,
                     'xgboost': xgboost.__version__, 'numpy': np.__version__},
        'fit_utc': datetime.now(timezone.utc).isoformat(),
        'train_curve_predictions_computed': False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nModels saved: {model_dir}')
    print('Only the five models were fitted on 88 records. No predictions for 88 curves were computed.')
    print('Future independent experiments: --mode predict (will not open Stage3).')


def curing_alpha(time, ti, k1, n1, alpha0):
    """Same Stage 5 / 8 piecewise induction interpolation and curing law."""
    t = np.asarray(time, dtype=float)
    ti = max(float(ti), 1e-12)
    k1 = max(float(k1), 1e-12)
    n1 = max(float(n1), 0.1)
    a0 = float(np.clip(alpha0, 0, 1))
    pre = a0 * np.clip(t / ti, 0, 1)  # empirical induction interpolation
    dt = np.maximum(t - ti, 0)
    post = a0 + (1 - a0) * (1 - np.exp(-k1 * dt ** n1))
    return np.clip(np.where(t < ti, pre, post), 0, 1)


def load_models(model_dir):
    manifest_path = model_dir/'model_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Frozen models absent at {model_dir}. '
                                'Run --mode fit ONCE (only if you lack saved 88-trained models), '
                                'or completely extract the accompanying ZIP package.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('training_n') != 88 or manifest.get('process_features') != INPUTS:
        raise ValueError('Model metadata does not match 88 curves or expected process fields')
    models = {}
    for target in TARGETS:
        if target in ('ML', 'MH'):
            p = model_dir/f'{target}_RF.joblib'
            if not p.is_file():
                raise FileNotFoundError(p)
            models[target] = joblib.load(p)  # Load only trusted locally-generated files.
        else:
            p = model_dir/f'{target}_XGB.json'
            if not p.is_file():
                raise FileNotFoundError(p)
            model = XGBRegressor()
            model.load_model(str(p))
            models[target] = model
    return models, manifest


def metrics(y, yh):
    return {'R2': float(r2_score(y, yh)),
            'RMSE': float(np.sqrt(mean_squared_error(y, yh))),
            'MAE': float(mean_absolute_error(y, yh))}


def plot_single(path, no, time, actual, predicted, met):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, actual, color='#252A34', label='Experiment', lw=2.7)
    ax.plot(time, predicted, color='#D45A40', label='PG-ML forward (frozen 88)', lw=2.5, ls='--')
    ax.set(title=f'Independent curve {no}', xlabel='Time (s)', ylabel='Torque (kgf cm)')
    ax.text(0.03, 0.97, f"R² = {met['R2']:.3f}\nRMSE = {met['RMSE']:.3f}",
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='#BBBBBB', alpha=.9))
    ax.grid(alpha=.15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def predict_only(root, args, model_dir):
    # IMPORTANT: no Stage3 table is accessed in this function; no .fit() calls.
    models, manifest = load_models(model_dir)
    feature_path = resolve_existing(args.external_features, [
        root/'independent_validation_process_6.csv',
        root/'验证集_1-6'/'independent_validation_process_6.csv',
        root/'PGML_重新编号_建模88_验证6'/'验证集_1-6'/'independent_validation_process_6.csv',
    ], '6 external process inputs')
    curve_path = resolve_existing(args.external_curves, [
        root/'independent_validation_curves_6.csv',
        root/'验证集_1-6'/'independent_validation_curves_6.csv',
        root/'PGML_重新编号_建模88_验证6'/'验证集_1-6'/'independent_validation_curves_6.csv',
    ], '6 experimental Time/torque curves')
    ext = read_csv(feature_path, ['No', *INPUTS], 'external process-6')
    curves = read_csv(curve_path, ['No', 'Time', 'torque'], 'external curves-6')
    if len(ext) != 6 or set(ext.No) != EXPECTED_VALIDATION or ext.No.duplicated().any():
        raise ValueError('6-row external process data must contain No=1..6 exactly once')
    if set(curves.No) != EXPECTED_VALIDATION:
        raise ValueError('External curve data must contain exactly No=1..6')
    if (curves.Time < 0).any():
        raise ValueError('Negative times found in experimental curves')
    if ext[INPUTS].isna().any().any():
        raise ValueError('External process inputs have missing values')

    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root/'Stage9_Output'
    out.mkdir(parents=True, exist_ok=True)
    X = ext[INPUTS].to_numpy(float)
    estimates = {}
    for target in TARGETS:
        preds = models[target].predict(X)
        estimates[target] = np.power(10.0, preds) if target == 'k1' else np.asarray(preds, float)

    ptab = ext.copy()
    ptab['alpha0_used'] = float(manifest['alpha0_train_median'])
    for target in TARGETS:
        ptab[f'{target}_pred'] = estimates[target]
    ptab['MH_pred_unclipped'] = ptab['MH_pred'].copy()
    ptab['MH_pred'] = np.maximum(ptab.MH_pred, ptab.ML_pred+1e-6)
    ptab['ti_pred'] = np.maximum(ptab.ti_pred, 1e-12)
    ptab['k1_pred'] = np.maximum(ptab.k1_pred, 1e-12)
    ptab['n1_pred'] = np.maximum(ptab.n1_pred, 0.1)
    ranges = manifest['feature_ranges']
    ptab['features_outside_training_range'] = [
        ','.join(c for c in INPUTS if row[c] < ranges[c]['min'] or row[c] > ranges[c]['max'])
        for _, row in ptab.iterrows()
    ]
    ptab['outside_training_range'] = ptab.features_outside_training_range.ne('')

    point_tables, metric_rows = [], []
    fig_dir = out/'Curves'
    if not args.no_plots:
        fig_dir.mkdir(exist_ok=True)
    for row in ptab.itertuples(index=False):
        g = curves.loc[curves.No == row.No, ['Time', 'torque']].copy()
        g = g.groupby('Time', as_index=False)['torque'].mean().sort_values('Time')
        if len(g) < 5:
            raise ValueError(f'No={row.No} has fewer than 5 valid observation points')
        t = g.Time.to_numpy(float)
        observed = g.torque.to_numpy(float)
        alpha = curing_alpha(t, row.ti_pred, row.k1_pred, row.n1_pred, row.alpha0_used)
        prediction = row.ML_pred + (row.MH_pred - row.ML_pred)*alpha
        met = metrics(observed, prediction)  # only evaluation reads measured torque
        met.update({'No': int(row.No), 'N_points': int(len(t)),
                    'outside_training_range': bool(row.outside_training_range)})
        metric_rows.append(met)
        point_tables.append(pd.DataFrame({'No': int(row.No), 'Time': t,
                                          'Experimental': observed,
                                          'PG_ML_pred': prediction,
                                          'alpha_pred': alpha}))
        if not args.no_plots:
            plot_single(fig_dir/f'Curve_{int(row.No)}.png', row.No, t, observed, prediction, met)

    point_df = pd.concat(point_tables, ignore_index=True)
    met_df = pd.DataFrame(metric_rows).sort_values('No')
    summary = pd.DataFrame([{
        'N': len(met_df),
        'Mean_R2': float(met_df.R2.mean()),
        'Median_R2': float(met_df.R2.median()),
        'Mean_RMSE': float(met_df.RMSE.mean()),
        'Median_RMSE': float(met_df.RMSE.median()),
        'Mean_MAE': float(met_df.MAE.mean()),
    }])
    for filename, frame in [
        ('stage9_predicted_parameters.csv', ptab),
        ('stage9_pointwise_predictions.csv', point_df),
        ('stage9_curve_metrics.csv', met_df),
        ('stage9_summary.csv', summary),
    ]:
        frame.to_csv(out/filename, index=False, encoding='utf-8-sig')
    run = {
        'mode': 'predict_only; NO model.fit, NO Stage3 88-curve access, NO predictions for 88',
        'model_dir': str(model_dir),
        'training_n_from_frozen_model_metadata': manifest['training_n'],
        'trained_model_input_sha256': manifest['training_stage3_sha256'],
        'validation_n': 6,
        'validation_process_sha256': sha256(feature_path),
        'validation_curves_sha256': sha256(curve_path),
        'external_torques_used_for': 'evaluation and plots only',
        'evaluation_utc': datetime.now(timezone.utc).isoformat(),
        'software_versions': {'sklearn': SKLEARN_VERSION, 'xgboost': xgboost.__version__},
    }
    (out/'stage9_run_manifest.json').write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nStage 9 — FROZEN model / six-curve prediction (no training or predicting 88):')
    print(met_df.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print('\nSummary:\n' + summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'\nSaved to {out}')



from io import BytesIO
from zipfile import ZipFile
from matplotlib import gridspec, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# Restrained journal-friendly palette; consistent throughout both figures.
NAVY = '#203B4D'
CHARCOAL = '#293B47'
ORANGE = '#E77A5A'
TEAL = '#2D8F92'
BLUE = '#4D86B6'
GOLD = '#D69A43'
PALE = '#E8F0F3'
LIGHT = '#D9E3E8'
MID = '#657A86'
GRID = '#E5EBEE'
WHITE = '#FFFFFF'



def load_data(source):
    names=('stage9_pointwise_predictions.csv','stage9_curve_metrics.csv','stage9_summary.csv')
    if source.is_dir():
        data={n:pd.read_csv(next(iter(sorted(source.rglob(n))))) for n in names}
    else:
        if source.suffix.lower()!='.zip':
            raise ValueError(f'--stage9-source should be Stage9_Output folder or ZIP, not {source}')
        with ZipFile(source) as z:
            data={}
            for n in names:
                members=[m for m in z.namelist() if m.endswith('/'+n) or m==n]
                members=[m for m in members if '__MACOSX' not in m]
                if not members: raise FileNotFoundError(f'ZIP missing {n}')
                data[n]=pd.read_csv(BytesIO(z.read(members[0])))
    return data[names[0]],data[names[1]],data[names[2]]


def validate(point, met, summary):
    needed_points={'No','Time','Experimental','PG_ML_pred'}
    needed_metrics={'No','R2','RMSE','MAE'}
    if needed_points-set(point.columns): raise ValueError(f'Pointwise CSV missing {needed_points-set(point.columns)}')
    if needed_metrics-set(met.columns): raise ValueError(f'Metrics CSV missing {needed_metrics-set(met.columns)}')
    if set(point.No)!=set(range(1,7)) or set(met.No)!=set(range(1,7)) or len(met)!=6:
        raise ValueError('Expected exactly six independent validation curves, No 1..6')
    if point[['Time','Experimental','PG_ML_pred']].isna().any().any():
        raise ValueError('Missing points in predictions')
    if point.duplicated(['No','Time']).any(): raise ValueError('Duplicate (No,Time) records')
    for no,g in point.groupby('No'):
        row=met.loc[met.No==no].iloc[0]
        residual=g.PG_ML_pred.to_numpy()-g.Experimental.to_numpy()
        rmse=float(np.sqrt(np.mean(residual**2)))
        mae=float(np.mean(np.abs(residual)))
        obs=g.Experimental.to_numpy()
        r2=1-float((residual**2).sum()/((obs-obs.mean())**2).sum())
        for metric,val in [('RMSE',rmse),('MAE',mae),('R2',r2)]:
            if abs(row[metric]-val)>1e-6: raise ValueError(f'Curve {no} {metric} mismatches saved curve metrics')
    for n,c in [('Mean_R2','R2'),('Mean_RMSE','RMSE'),('Mean_MAE','MAE')]:
        if not np.isclose(float(summary[n].iloc[0]),met[c].mean(),rtol=0,atol=1e-6):
            raise ValueError(f'Summary {n} does not match curve-level CSV')
    return met.sort_values('No').reset_index(drop=True)


def style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':10,'text.color':NAVY,
        'axes.labelcolor':NAVY,'axes.edgecolor':LIGHT,'axes.titlecolor':NAVY,
        'xtick.color':MID,'ytick.color':MID,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':WHITE,'savefig.facecolor':WHITE,
    })


def save(fig,stem,out,dpi,pdf):
    fig.savefig(out/(stem+'.png'),dpi=dpi,facecolor=WHITE,bbox_inches='tight',pad_inches=.12)
    if pdf:
        fig.savefig(out/(stem+'.pdf'),facecolor=WHITE,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def metric_ax(ax,met,metric,metric_color,kind):
    vals=met[metric].to_numpy(float)
    nos=met.No.to_numpy(int)
    y=np.arange(len(nos))[::-1]
    high=max(vals.max()*1.15,0.01)
    mean=float(vals.mean())
    ax.set_axisbelow(True)
    ax.grid(axis='x',color=GRID,lw=.8)
    ax.set_xlim(0,high)
    ax.set_ylim(-.6,5.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{i:02d}' for i in nos],fontsize=8.5)
    ax.tick_params(axis='y',length=0)
    ax.tick_params(axis='x',labelsize=8)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(LIGHT)
    ax.barh(y,vals,color=metric_color,height=.52,alpha=.90,edgecolor='none')
    ax.axvline(mean,ls='--',lw=1.4,color=NAVY,alpha=.7)
    for yy,v in zip(y,vals):
        ax.text(v+high*.025,yy,f'{v:.3f}',va='center',ha='left',fontsize=7.9,color=NAVY)
    ax.set_title(f'{kind}   |   mean {mean:.3f}',loc='left',fontsize=10.7,fontweight='bold',pad=13)
    if metric=='R2':
        ax.set_xlim(0,1.1)
        ax.set_xticks([0,.25,.5,.75,1.0])
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_xlabel('kgf·cm' if metric!='R2' else 'R²',fontsize=9,labelpad=3)


def draw_six_curve_atlas(point,met,summary,out,dpi,pdf):
    fig=plt.figure(figsize=(17.0,15.1),facecolor=WHITE)
    outer=gridspec.GridSpec(3,3,figure=fig,height_ratios=[3.8,3.8,2.65],
                            left=.074,right=.965,bottom=.065,top=.855,hspace=.42,wspace=.30)

    # Header with a quiet editorial hierarchy.
    fig.text(.074,.985,'SIX-CURVE  |  PREDICTION ATLAS',fontsize=11.3,fontweight='bold',color=TEAL,va='top')
    fig.text(.074,.961,'Frozen-model PG-ML prediction of six experimental curing curves',
             fontsize=21,fontweight='bold',va='top',color=NAVY)
    fig.text(.074,.935,'88-curve training model  ·  Six experimental curves evaluated at their measured time points',
             fontsize=10.6,color=MID,va='top')
    fig.add_artist(Line2D([.074,.965],[.893,.893],transform=fig.transFigure,color=LIGHT,lw=1.0))
    handles=[Line2D([0],[0],color=CHARCOAL,lw=2.5,label='Experimental torque'),
             Line2D([0],[0],color=ORANGE,lw=2.4,ls='--',label='PG-ML prediction')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.968,.923),frameon=False,ncol=2,
               fontsize=10,columnspacing=1.5,handlelength=2.8)
    # Top two rows: each mini-panel includes original curve AND residual timeline.
    all_span=float(point.Experimental.max()-point.Experimental.min())
    for i,no in enumerate(range(1,7)):
        slot=outer[i//3,i%3]
        gs=gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=slot,
                                            height_ratios=[4.5,1.05],hspace=.10)
        ax=fig.add_subplot(gs[0])
        axr=fig.add_subplot(gs[1],sharex=ax)
        g=point[point.No==no].sort_values('Time')
        t=g.Time.to_numpy(float)
        y=g.Experimental.to_numpy(float)
        pred=g.PG_ML_pred.to_numpy(float)
        resid=pred-y
        m=met.loc[met.No==no].iloc[0]
        ax.plot(t,y,color=CHARCOAL,lw=2.15,solid_capstyle='round',rasterized=False)
        ax.plot(t,pred,color=ORANGE,lw=2.0,ls='--',solid_capstyle='round')
        span=max(float(y.max()-y.min()),1.0)
        ax.set_ylim(min(y.min(),pred.min())-.06*span,max(y.max(),pred.max())+.22*span)
        ax.grid(color=GRID,lw=.7,alpha=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both',labelsize=8.7,length=3)
        ax.tick_params(axis='x',labelbottom=False)
        ax.text(.02,.98,f'({chr(97+i)})  CURVE {no:02d}',transform=ax.transAxes,fontsize=11,
                va='top',fontweight='bold',color=NAVY)
        ax.text(.98,.98,f'R²  {m.R2:.3f}\nRMSE  {m.RMSE:.2f}',transform=ax.transAxes,
                fontsize=8.8,va='top',ha='right',color=NAVY,
                bbox=dict(boxstyle='round,pad=.38',facecolor=WHITE,alpha=.95,edgecolor=LIGHT))
        if i%3==0:
            ax.set_ylabel('Torque (kgf·cm)',fontsize=9.5)
        # Residual track: pred minus experimental, no claim of uncertainty.
        axr.axhline(0,color=MID,lw=.9,ls='--')
        axr.plot(t,resid,lw=1.15,color=TEAL)
        axr.fill_between(t,resid,0,color=TEAL,alpha=.18)
        lim=max(np.quantile(np.abs(resid),.98)*1.15,.7)
        axr.set_ylim(-lim,lim)
        axr.tick_params(axis='both',labelsize=7.8,length=2)
        axr.grid(axis='x',color=GRID,lw=.55)
        axr.yaxis.set_major_locator(ticker.MaxNLocator(3))
        axr.set_xlabel('Time (s)',fontsize=9,labelpad=1)
        if i%3==0:
            axr.set_ylabel('Error',fontsize=8)
    # Bottom dashboard: per-curve R², RMSE, MAE with mean markers.
    for idx,(metric,name,color) in enumerate((('R2','R²',BLUE),('RMSE','RMSE',ORANGE),('MAE','MAE',TEAL))):
        ax=fig.add_subplot(outer[2,idx])
        metric_ax(ax,met,metric,color,name)
        if idx==0:
            ax.set_ylabel('Validation curve No.',fontsize=9)
        elif idx>0:
            ax.set_ylabel('')
    # No extrapolation beyond the six supplied outcomes.
    fig.text(.075,.016,'Predictions from frozen 88-curve models. Residual = predicted − experimental torque. '
             'Dashed vertical metric markers = six-curve means.',fontsize=9,color=MID)
    save(fig,'Figure_Stage9_SixCurve_Atlas',out,dpi,pdf)


def draw_metrics_only(met,summary,out,dpi,pdf):
    fig,axes=plt.subplots(1,3,figsize=(14.5,5.1))
    fig.subplots_adjust(left=.085,right=.975,bottom=.17,top=.75,wspace=.32)
    fig.text(.085,.965,'CURVE-LEVEL PREDICTION PERFORMANCE',fontsize=18.5,
             fontweight='bold',color=NAVY,va='top')
    fig.text(.085,.87,'Six additional experimental curves  |  frozen 88-curve PG-ML',
             fontsize=10.5,color=MID,va='top')
    for idx,(metric,name,color) in enumerate((('R2','R²',BLUE),('RMSE','RMSE',ORANGE),('MAE','MAE',TEAL))):
        metric_ax(axes[idx],met,metric,color,name)
        if idx==0: axes[idx].set_ylabel('Curve No.',fontsize=10)
    fig.text(.085,.065,'Bars = individual-curve metric  ·  Dashed line = mean of six curves  ·  '
             'RMSE/MAE in kgf·cm',fontsize=9.5,color=MID)
    save(fig,'Figure_Stage9_Metric_Overview',out,dpi,pdf)




def main():
    args = arg_parser()
    root = Path(args.project_root).expanduser().resolve() if args.project_root else Path(__file__).resolve().parent
    if not root.is_dir():
        raise FileNotFoundError(f"Project root directory does not exist: {root}")
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else root/'Stage9_Models_88'
    if args.mode == 'fit':
        fit_once(root, args, model_dir)
        print('训练模型已保存。下一步请运行：python ' + repr(Path(__file__).name) + ' --mode predict')
        return

    # --mode plot NEVER opens models or performs predictions.
    if args.mode == 'predict':
        predict_only(root, args, model_dir)

    saved = Path(args.output_dir).expanduser().resolve() if args.output_dir else root/'Stage9_Output'
    if not saved.exists():
        raise FileNotFoundError(f"缺少{saved}。预测模式请检查模型和6条曲线数据；仅绘图模式请指定已有的Stage9_Output。")
    if args.no_plots:
        print('CSV预测结果已生成。已按 --no-plots 跳过绘图。')
        return

    figout = (Path(args.fig_output_dir).expanduser().resolve()
              if args.fig_output_dir else root/'Stage9_Visual_Figures')
    figout.mkdir(parents=True, exist_ok=True)
    point, met, summary = load_data(saved)
    met = validate(point, met, summary)
    style()
    draw_six_curve_atlas(point, met, summary, figout, args.dpi, not args.no_pdf)
    draw_metrics_only(met, summary, figout, args.dpi, not args.no_pdf)
    print('\n图表生成完成：', figout)
    print('① Figure_Stage9_SixCurve_Atlas.png/.pdf  六条曲线+残差+综合指标')
    print('② Figure_Stage9_Metric_Overview.png/.pdf  6条曲线的R²/RMSE/MAE')
    print('Mean R²={:.6f}, RMSE={:.6f}, MAE={:.6f}'.format(met.R2.mean(),met.RMSE.mean(),met.MAE.mean()))
    print('本次不会重新训练88条曲线，也不会重新预测88条建模曲线。')

if __name__ == '__main__':
    main()
