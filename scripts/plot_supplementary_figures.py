#!/usr/bin/env python3
"""Prepare seven supplementary figures from saved scientific results.

Use the scientific dependencies in requirements.txt. Saved scientific result
tables are read without modification.
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
from scipy.stats import pearsonr, rankdata
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
import plot_publication_figures as pub

ROOT = Path(os.environ.get("VULCANIZATION_PROJECT_ROOT", str(Path(__file__).resolve().parents[1] / "work"))).resolve()
OUT = ROOT / 'SI_Figures_20261103'
OUT.mkdir(exist_ok=True)
INPUTS, CHECKS, OUTPUTS = {}, {}, []
P, F = pub.PARAMS, pub.FEATURE_LABELS
NAVY, BLUE, TEAL, ORANGE, GRAY, GRID = pub.NAVY, pub.BLUE, pub.TEAL, pub.ORANGE, pub.GRAY, pub.GRID
CMAP = colors.LinearSegmentedColormap.from_list('si_correlations', [BLUE, '#F8FAFC', ORANGE])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path, **kwargs):
    INPUTS[path] = sha(ROOT/path)
    return pd.read_csv(ROOT/path, **kwargs)

def save(fig, no, name, original_images):
    stem = OUT / f'Figure_S{no}_{name}'
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(stem.with_suffix('.'+fmt), dpi=600, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)
    OUTPUTS.append(dict(figure=no, stem=stem.name, original_images=original_images,
                        png_sha256=sha(stem.with_suffix('.png'))))
    print('Completed',stem.name,flush=True)

def heatmap(ax, d, size=8):
    im=ax.imshow(d.to_numpy(),vmin=-1,vmax=1,cmap=CMAP)
    ax.set_xticks(range(len(d.columns)),[P[x] for x in d.columns])
    ax.set_yticks(range(len(d.index)),[P[x] for x in d.index])
    ax.tick_params(length=0,pad=6)
    for i in range(len(d)):
        for j in range(len(d.columns)):
            v=d.iloc[i,j]
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=size,
                    color='white' if abs(v)>.73 else NAVY)
    ax.set_xticks(np.arange(-.5,len(d.columns),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(d.index),1),minor=True)
    ax.grid(which='minor',color='white',lw=.8)
    ax.tick_params(which='minor',length=0)
    for s in ax.spines.values():s.set_visible(False)
    return im

pub.style()
data=read('Stage3_Output/pgmlp_curve_dataset.csv')
assert len(data)==88 and set(data.No)==set(range(1,89))

# S1: supplementary descriptor-to-descriptor matrices, absent from the main paper.
pe=read('Stage7A_Output/pearson_parameter_matrix.csv',index_col=0)
sp=read('Stage7A_Output/spearman_parameter_matrix.csv',index_col=0)
cols=list(sp.columns)
for method,d in [('pearson',pe),('spearman',sp)]:
    computed=data[cols].corr(method=method)
    delta=float(np.max(abs(computed.to_numpy()-d.to_numpy())))
    assert delta<1e-10,(method,delta)
    CHECKS[method+'_matrix_max_difference']=delta
fig,axs=plt.subplots(1,2,figsize=(7.2,3.8))
fig.subplots_adjust(left=.055,right=.925,top=.86,bottom=.12,wspace=.21)
for ax,d,letter,title in zip(axs,[pe,sp],'ab',['Pearson correlations','Spearman correlations']):
    im=heatmap(ax,d,size=7.5);pub.title(ax,letter,title)
cb=fig.colorbar(im,cax=fig.add_axes([.95,.18,.016,.62]),ticks=[-1,-.5,0,.5,1]);cb.outline.set_visible(False)
save(fig,1,'descriptor_correlations',['Stage7A_Output/heatmap_Pearson_parameters.png','Stage7A_Output/heatmap_Spearman_parameters.png'])

# S2: exactly reproduce the saved rank-residual partial correlation.
partial=read('Stage7B_Output/partial_spearman_parameter_matrix_control_curing_temp.csv',index_col=0)
z=rankdata(data.curing_temp.to_numpy(float));design=np.column_stack([np.ones(88),z])
ranked=np.column_stack([rankdata(data[x].to_numpy(float)) for x in cols])
resid=ranked-design@np.linalg.lstsq(design,ranked,rcond=None)[0]
delta=float(np.max(abs(np.corrcoef(resid,rowvar=False)-partial.to_numpy())))
assert delta<1e-10,delta
CHECKS['partial_spearman_max_difference']=delta
fig,ax=plt.subplots(figsize=(6.8,5.6))
fig.subplots_adjust(left=.12,right=.85,top=.91,bottom=.11)
im=heatmap(ax,partial,size=9)
ax.set_title('Partial Spearman correlations',loc='left',pad=15)
cb=fig.colorbar(im,cax=fig.add_axes([.88,.14,.022,.72]),ticks=[-1,-.5,0,.5,1]);cb.set_label(r'Partial $\rho$');cb.outline.set_visible(False)
save(fig,2,'temperature_controlled_correlations',['Stage7B_Output/heatmap_partial_Spearman_parameters_control_curing_temp.png'])

# S3: extend the main paper's ti/k1 ablation to the remaining three targets.
ab=read('Stage7B_Output/ablation_summary.csv')
raw_ab=read('Stage7B_Output/ablation_all_folds.csv')
assert raw_ab.groupby(['Target','Feature_Set']).size().eq(25).all()
order=['CuringTemp_Only','All_Process','Without_CuringTemp']
labels=['Curing temperature only','All process variables','Without curing temperature']
fig,axs=plt.subplots(3,1,figsize=(7.2,5.5),sharex=True)
fig.subplots_adjust(left=.31,right=.96,top=.92,bottom=.11,hspace=.6)
for ax,target,letter in zip(axs,['ML','MH','n1'],'abc'):
    g=ab.loc[ab.Target==target].set_index('Feature_Set').loc[order]
    yy=np.arange(3)
    ax.barh(yy,g.R2_mean,xerr=g.R2_std,height=.58,color=[BLUE,TEAL,'#9EAEB9'],
            error_kw=dict(ecolor=NAVY,capsize=2,elinewidth=.8))
    ax.set_yticks(yy,labels);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=.8)
    ax.set_xlim(-1.4,1.08);pub.axis(ax,'x');ax.spines['left'].set_visible(False)
    ax.set_yticks(yy,labels);ax.tick_params(axis='y',length=0)
    pub.title(ax,letter,P[target]+' feature ablation')
    for y,row in enumerate(g.itertuples()):
        ax.text(1.065,y,f'{row.R2_mean:.3f}',ha='right',va='center',fontsize=8)
axs[-1].set_xlabel(r'Mean held-out $R^2$ ± SD')
save(fig,3,'additional_target_ablation',[f'Stage7B_Output/ablation_{t}.png' for t in ['ML','MH','n1']])

# Reproduce full-data descriptive explainers only; no CV models are rerun.
X=data[pub.FEATURES].copy()
expected=read('Stage7C_Output/SHAP_importance_summary.csv')
records={}
for target in ['ti','n1','MH']:
    model=(RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_split=2,
            min_samples_leaf=1,max_features='sqrt',random_state=42,n_jobs=1)
           if target=='MH' else XGBRegressor(objective='reg:squarederror',n_estimators=300,
            max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,
            min_child_weight=2,reg_alpha=.01,reg_lambda=1.,random_state=42,n_jobs=1))
    model.fit(X,data[target].to_numpy(float))
    v=np.asarray(shap.TreeExplainer(model).shap_values(X))
    exp=expected.loc[expected.Target==target].set_index('Feature').loc[pub.FEATURES,'Mean_abs_SHAP'].to_numpy()
    delta=float(np.max(abs(abs(v).mean(axis=0)-exp)))
    assert delta<1e-6,(target,delta)
    CHECKS['SHAP_'+target+'_max_mean_absolute_difference']=delta
    records[target]=v

# S4: n1, omitted from the main attribution panels.
perm=read('Stage7C_Output/permutation_importance_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(7.2,4.1))
fig.subplots_adjust(left=.235,right=.98,top=.84,bottom=.27,wspace=1.24)
ax=axs[0];v=records['n1'];order=np.argsort(abs(v).mean(axis=0))[::-1]
cmap=colors.LinearSegmentedColormap.from_list('si_feature_value',[BLUE,'#AEC7CD',ORANGE])
ax.axvline(0,color=GRAY,lw=.7)
for row,feature in enumerate(order):
    raw=X.iloc[:,feature].to_numpy(float);norm=(raw-raw.min())/max(np.ptp(raw),1e-12)
    ax.axhline(row,color=GRID,lw=.5,ls=':')
    ax.scatter(v[:,feature],row+pub.bee_offsets(v[:,feature]),c=norm,cmap=cmap,
               vmin=0,vmax=1,s=11,alpha=.9,edgecolors='none',zorder=3)
ax.set_yticks(range(7),[F[pub.FEATURES[i]] for i in order]);ax.set_ylim(6.55,-.55)
ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
ax.set_xlabel(r'SHAP value for $n_1$');pub.title(ax,'a',r'$n_1$ SHAP attribution')
cb=fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0,1),cmap=cmap),
    cax=fig.add_axes([.252,.07,.26,.017]),orientation='horizontal',ticks=[0,1])
cb.ax.set_xticklabels(['Low feature value','High feature value']);cb.ax.tick_params(length=0,labelsize=6.5);cb.outline.set_visible(False)
ax=axs[1];g=perm.loc[perm.Target=='n1'].sort_values('Importance_mean',ascending=False)
ax.barh(range(7),g.Importance_mean,xerr=g.Importance_std,height=.65,color=[TEAL]+[BLUE]*6,
    error_kw=dict(ecolor=NAVY,capsize=2,elinewidth=.7))
ax.set_yticks(range(7),[F[f] for f in g.Feature]);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=.7)
pub.axis(ax,'x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
ax.set_yticks(range(7),[F[f] for f in g.Feature])
ax.set_xlabel(r'Decrease in held-out $R^2$');pub.title(ax,'b',r'$n_1$ permutation importance')
save(fig,4,'exponent_attribution',['Stage7C_Output/SHAP_beeswarm_n1.png','Stage7C_Output/permutation_importance_n1.png'])

# S5: the two most influential inputs in each full-data explainer.
fig,axs=plt.subplots(3,2,figsize=(7.2,6.5))
fig.subplots_adjust(left=.105,right=.975,top=.95,bottom=.08,hspace=.9,wspace=.34)
original=[];k=0
unit={'curing_temp':' (°C)','mixing_temp':' (°C)','rotor_speed_1':' (rpm)'}
for i,target in enumerate(['ti','n1','MH']):
    v=records[target];order=np.argsort(abs(v).mean(axis=0))[::-1][:2]
    for j,idx in enumerate(order):
        ax=axs[i,j];feature=pub.FEATURES[idx]
        ax.scatter(X[feature],v[:,idx],s=15,color=BLUE,alpha=.75,edgecolors='white',linewidths=.2)
        ax.axhline(0,color=GRAY,lw=.65,ls='--');pub.axis(ax)
        ax.set_xlabel(F[feature]+unit.get(feature,''),fontsize=8)
        ax.set_ylabel('SHAP value'+({'ti':' (s)','MH':' (kgf·cm)'}.get(target,'')),fontsize=8)
        pub.title(ax,'abcdef'[k],P[target]+' dependence');k+=1
        original.append(f'Stage7C_Output/SHAP_dependence_{target}_{j+1}_{feature}.png')
        assert (ROOT/original[-1]).exists()
save(fig,5,'SHAP_dependence',original)

# S6: paired held-out curve RMSE, same 300-point training-resampling run.
metrics=read('Stage8_Output/curve_metrics_all_methods.csv')
wide=metrics.pivot(index='No',columns='Method',values='RMSE').sort_index()
assert len(wide)==88 and wide.notna().all().all()
fig,axs=plt.subplots(1,2,figsize=(7.2,3.6))
fig.subplots_adjust(left=.09,right=.98,top=.84,bottom=.19,wspace=.28)
for ax,meth,letter,col in zip(axs,['Direct-MLP','Direct-XGB'],'ab',[BLUE,ORANGE]):
    x,y=wide[meth],wide['PG-ML'];lim=max(x.max(),y.max())*1.06
    ax.scatter(x,y,s=21,color=col,alpha=.78,edgecolors='white',linewidth=.3,zorder=3)
    ax.plot([0,lim],[0,lim],color=GRAY,lw=.9,ls='--')
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel=meth+' RMSE (kgf·cm)',ylabel='PG-ML RMSE (kgf·cm)')
    ax.set_aspect('equal');pub.axis(ax)
    n=int((y<x).sum());CHECKS[meth+'_PG_lower_RMSE_count']=n
    ax.text(.04,.96,f'PG-ML lower RMSE: {n}/88',transform=ax.transAxes,va='top',fontsize=8)
    pub.title(ax,letter,meth+' versus PG-ML')
assert CHECKS['Direct-MLP_PG_lower_RMSE_count']==59 and CHECKS['Direct-XGB_PG_lower_RMSE_count']==27
save(fig,6,'paired_curve_errors',['Stage8_Output/RMSE_Direct_MLP_vs_PGML.png','Stage8_Output/RMSE_Direct_XGB_vs_PGML.png'])

# S7: the original high-error selection is the maximum PG-ML curve RMSE.
points=read('Stage8_Output/pointwise_predictions.csv')
no=int(wide['PG-ML'].idxmax());assert no==15
g=points.loc[points.No==no].sort_values('Time')
for method,key in [('Direct-MLP','Direct_MLP'),('Direct-XGB','Direct_XGB'),('PG-ML','PG_ML')]:
    rmse=float(np.sqrt(np.mean((g[key]-g.Experimental)**2)))
    delta=abs(rmse-wide.loc[no,method])
    # Direct-XGB's saved metrics retain float32 prediction arithmetic;
    # CSV reloading promotes those predictions to float64.
    assert delta<1e-6,(method,delta)
    CHECKS['high_error_'+method+'_CSV_RMSE_difference']=float(delta)
CHECKS['high_error_curve']=no
CHECKS['high_error_metrics']=metrics.loc[metrics.No==no].to_dict(orient='records')
fig,axs=plt.subplots(2,1,figsize=(7.2,4.7),sharex=True,gridspec_kw={'height_ratios':[2.2,1]})
fig.subplots_adjust(left=.1,right=.98,top=.86,bottom=.12,hspace=.4)
names=['Experimental','Direct-MLP','Direct-XGB','PG-ML']
fig.legend(handles=pub.handles(names,[NAVY,BLUE,ORANGE,TEAL],['-',':','-.','--']),
           loc='upper center',ncol=4,bbox_to_anchor=(.54,1),fontsize=8)
for method,key,col,ls in [('Direct-MLP','Direct_MLP',BLUE,':'),('Direct-XGB','Direct_XGB',ORANGE,'-.'),('PG-ML','PG_ML',TEAL,'--')]:
    axs[0].plot(g.Time,g[key],color=col,ls=ls,lw=1.5)
    axs[1].plot(g.Time,g[key]-g.Experimental,color=col,ls=ls,lw=1.3)
axs[0].plot(g.Time,g.Experimental,color=NAVY,lw=1.6)
axs[0].set_ylabel('Torque (kgf·cm)');pub.title(axs[0],'a','Highest PG-ML error: curve 15')
axs[1].set(xlabel='Time (s)',ylabel='Residual (kgf·cm)');axs[1].axhline(0,color=GRAY,lw=.7)
pub.title(axs[1],'b','Predicted minus measured torque')
for ax in axs:pub.axis(ax)
save(fig,7,'high_error_case',['Stage8_Output/representative_high_error.png'])

for p,h in INPUTS.items():assert sha(ROOT/p)==h,p
payload=dict(purpose='Supplementary figures from saved results omitted from the main article',
    inputs_sha256=INPUTS,checks=CHECKS,outputs=OUTPUTS,
    notes='Only full-data descriptive SHAP models were refitted; their summaries were verified. Saved CV predictions and scientific results were not overwritten.')
(OUT/'source_manifest.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
print('All scientific source checks passed.',flush=True)
