#!/usr/bin/env python3
"""
performance_figures.py
======================
Classification-performance figures and tables (Fig 1-3, Supp Fig 1-7, 9).

Reads the benchmark summary (ml_summary_means.tsv), keeps the best model per
(study x condition x seed), and aggregates to per (study x feature x
rarefaction) cell means over the 10 seeds.

Metrics: AUROC (AUC), AUPRC, Sensitivity, Specificity, Balanced accuracy,
MCC (Matthews correlation coefficient).

Outputs -> output/
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

INPUT_TSV = '../../data/ml_summary_means.tsv'
OUT = 'output'
os.makedirs(OUT, exist_ok=True)

STUDY_LABELS = {1:"IBD",2:"HCC",6:"VAP",8:"Breast Cancer",9:"Parkinson's",
                10:"Gout",11:"CRC Polyps",12:"Periodontitis",13:"S-ECC",
                15:"HIV-1 (a)",16:"HIV-1 (b)",17:"Bacteriuria"}
BODY_SITES = {1:"Gut",2:"Liver",6:"Respiratory",8:"Blood",9:"Gut",10:"Gut",
              11:"Gut",12:"Oral",13:"Oral",15:"Gut",16:"Gut",17:"Urinary"}

METRICS = ['AUC','AUPRC','Sensitivity','Specificity','Balanced_Accuracy','MCC']
MLABEL  = {'AUC':'AUROC','AUPRC':'AUPRC','Sensitivity':'Sensitivity',
           'Specificity':'Specificity','Balanced_Accuracy':'Balanced accuracy',
           'MCC':'MCC'}

# Shared plot style for all performance figures.
mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':13,'axes.labelsize':14,'axes.titlesize':14,'axes.titleweight':'bold',
    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
    'legend.title_fontsize':12.5,'figure.titlesize':16,
    'axes.linewidth':1.0,'xtick.major.width':1.0,'ytick.major.width':1.0,
    'xtick.major.size':4,'ytick.major.size':4,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
    'savefig.dpi':400,'figure.dpi':110,'pdf.fonttype':42,'ps.fonttype':42,
})
from matplotlib.patches import Patch
PAL = {'Non-rarefied':'#0F7B8A','Rarefied':'#E07B39','ASV':'#3B6AA0','Taxa':'#C36F2A'}

def clean_spines(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

def panel_label(ax, label, x=-0.18, y=1.06, fontsize=26):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='left')

def sig_stars(p):
    if pd.isna(p): return 'ns'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

# Larger-font context applied to Fig 2, Fig 3 and Supp Figs 2-7, 9 so labels,
# ticks, titles and legends read clearly at print size. Fig 1 and Supp Fig 1
# keep the base sizing and are drawn outside this context.
BIG = {'font.size': 17, 'axes.labelsize': 20, 'axes.titlesize': 20,
       'axes.titleweight': 'bold', 'xtick.labelsize': 16, 'ytick.labelsize': 16,
       'legend.fontsize': 16}


def slabel(sid):
    return f"S{int(sid)} {STUDY_LABELS.get(int(sid), '')}".strip()

def savepng(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=400, bbox_inches='tight', facecolor='white')
    print('Saved', p)
    plt.close(fig)

def panel(ax, letter, x=-0.16, y=1.06, fs=26):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fs,
            fontweight='bold', va='top', ha='left')

def pstars(p):
    if pd.isna(p): return 'ns'
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'


# --------------------------------------------------------------------------
def load():
    df = pd.read_csv(INPUT_TSV, sep='\t')
    df = df[df['Is_Best'] == True].copy()
    df[['Rarefaction','Feature']] = df['Dataset'].str.split('_', expand=True)
    df['Rarefaction'] = df['Rarefaction'].map({'nonrarefied':'Non-rarefied','rarefied':'Rarefied'})
    df['Feature'] = df['Feature'].map({'asv':'ASV','taxa':'Taxa'})
    df['Site'] = df['ML_Number'].map(BODY_SITES)
    # per (study,feature,rarefaction) mean over seeds
    cell = (df.groupby(['ML_Number','Site','Feature','Rarefaction'])[METRICS]
              .mean().reset_index())
    return df, cell


# --------------------------------------------------------------------------
# STATISTICS (also drives tables)
# --------------------------------------------------------------------------
def pooled_stats(cell):
    piv = cell.pivot_table(index=['ML_Number','Feature'],
                           columns='Rarefaction', values=METRICS)
    rows, pvals = [], []
    for m in METRICS:
        nr = piv[(m,'Non-rarefied')].values; r = piv[(m,'Rarefied')].values
        d = r - nr
        W, p = stats.wilcoxon(r, nr)
        n = len(d)
        rb = 1 - (4*min(W, n*(n+1)/2 - W)) / (n*(n+1))  # rank-biserial approx
        rows.append(dict(Metric=MLABEL[m], NR=nr.mean(), R=r.mean(),
                         mean_delta=d.mean(), median_delta=np.median(d),
                         rank_biserial=rb, p=p))
        pvals.append(p)
    fdr = multipletests(pvals, method='fdr_bh')[1]
    out = pd.DataFrame(rows)
    out['p_FDR'] = fdr
    out['Significant'] = np.where(out['p_FDR'] < 0.05, 'yes', 'no')
    return out, piv

def byfeature_stats(piv):
    rows = []
    for feat in ['ASV','Taxa']:
        pv = piv.xs(feat, level='Feature')
        for m in METRICS:
            nr = pv[(m,'Non-rarefied')].values; r = pv[(m,'Rarefied')].values
            W, p = stats.wilcoxon(r, nr)
            rows.append(dict(Feature=feat, Metric=MLABEL[m], NR=nr.mean(),
                             R=r.mean(), mean_delta=(r-nr).mean(), p_uncorrected=p))
    return pd.DataFrame(rows)

def win_equiv_loss(piv, band=0.01):
    rows = []
    for m in METRICS:
        for feat in ['ASV','Taxa']:
            pv = piv.xs(feat, level='Feature')
            d = pv[(m,'Rarefied')].values - pv[(m,'Non-rarefied')].values
            rows.append(dict(Metric=MLABEL[m], Feature=feat, N_studies=len(d),
                             Wins=int((d>band).sum()),
                             Equivalent=int((np.abs(d)<=band).sum()),
                             Losses=int((d<-band).sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# MAIN FIG 1 - overview heatmaps
# --------------------------------------------------------------------------
def fig1(cell):
    """Four-panel overview heatmaps.
    a, b: AUROC / Balanced accuracy per study (rows in study-ID order)
    c, d: AUROC / Balanced accuracy per body site (rows by sample size).
    """
    agg = cell.copy()
    agg['Body_Site'] = agg['ML_Number'].map(BODY_SITES)
    site_order = (pd.Series(BODY_SITES).value_counts()
                  .sort_values(ascending=False).index.tolist())
    site_n = pd.Series(BODY_SITES).value_counts().to_dict()
    cond_order = ['ASV | NR', 'ASV | R', 'Taxa | NR', 'Taxa | R']

    def build_study_heatmap(metric):
        g = (agg.groupby(['ML_Number', 'Feature', 'Rarefaction'])[metric]
             .mean().reset_index())
        g['cond'] = g['Feature'] + ' | ' + g['Rarefaction'].map(
            {'Non-rarefied': 'NR', 'Rarefied': 'R'})
        pivot = g.pivot(index='ML_Number', columns='cond', values=metric)[cond_order]
        pivot.index = [slabel(i) for i in pivot.index]
        return pivot

    def build_site_heatmap(metric):
        g = (agg.groupby(['Body_Site', 'Feature', 'Rarefaction'])[metric]
             .mean().reset_index())
        g['cond'] = g['Feature'] + ' | ' + g['Rarefaction'].map(
            {'Non-rarefied': 'NR', 'Rarefied': 'R'})
        pivot = g.pivot(index='Body_Site', columns='cond', values=metric)[cond_order]
        pivot = pivot.reindex(site_order)
        pivot.index = [f'{s} (n={site_n[s]})' for s in pivot.index]
        return pivot

    METRIC_LIMITS = {'AUC': (0.6, 1.0), 'Balanced_Accuracy': (0.5, 1.0)}
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 6],
                          left=0.18, right=0.95, top=0.95, bottom=0.06,
                          wspace=0.55, hspace=0.18)
    metrics = [('AUC', 'AUROC'), ('Balanced_Accuracy', 'Balanced accuracy')]

    for col, (metric, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col])
        p = build_study_heatmap(metric)
        vmin, vmax = METRIC_LIMITS[metric]
        im = ax.imshow(p.values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns, rotation=0, fontsize=14)
        ax.set_yticks(range(p.shape[0])); ax.set_yticklabels(p.index, fontsize=13.5)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=13,
                        color='white' if v < (vmin + vmax) / 2 else 'black')
        ax.set_title(f'{label} per study and condition', fontsize=16, fontweight='bold', pad=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(label, fontsize=14); cbar.ax.tick_params(labelsize=13)
        if metric == 'AUC':
            cbar.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0])
            cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        clean_spines(ax); panel_label(ax, 'ab'[col], x=-0.32, y=1.04, fontsize=26)

    for col, (metric, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, col])
        p = build_site_heatmap(metric)
        vmin, vmax = METRIC_LIMITS[metric]
        im = ax.imshow(p.values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns, rotation=0, fontsize=14)
        ax.set_yticks(range(p.shape[0])); ax.set_yticklabels(p.index, fontsize=13.5)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=13,
                        color='white' if v < (vmin + vmax) / 2 else 'black')
        ax.set_title(f'{label} per body site and condition', fontsize=16, fontweight='bold', pad=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(label, fontsize=14); cbar.ax.tick_params(labelsize=13)
        if metric == 'AUC':
            cbar.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0])
            cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        clean_spines(ax); panel_label(ax, 'cd'[col], x=-0.32, y=1.10, fontsize=26)

    savepng(fig, 'MainFig1_overview_heatmaps.png')


# --------------------------------------------------------------------------
# MAIN FIG 2 - overall metric comparison (6 metrics)
# --------------------------------------------------------------------------
def fig2(cell, byfeat):
    """Overall metric comparison across six metrics (AUROC, AUPRC,
    Sensitivity, Specificity, Balanced accuracy and MCC). Each plotted point
    is a per-study best-model mean; p-values are per-study paired Wilcoxon.
    """
    METRIC_ORDER = [('AUC', 'AUROC'), ('AUPRC', 'AUPRC'),
                    ('Sensitivity', 'Sensitivity'), ('Specificity', 'Specificity'),
                    ('Balanced_Accuracy', 'Balanced accuracy'), ('MCC', 'MCC')]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))
    axes = axes.flatten()
    for idx, (mcol, mlabel) in enumerate(METRIC_ORDER):
        ax = axes[idx]
        groups = [('ASV', 'Non-rarefied'), ('ASV', 'Rarefied'),
                  ('Taxa', 'Non-rarefied'), ('Taxa', 'Rarefied')]
        positions = [1, 2, 3.5, 4.5]
        colors = [PAL['Non-rarefied'], PAL['Rarefied'],
                  PAL['Non-rarefied'], PAL['Rarefied']]
        data_lists = [cell.loc[(cell['Feature'] == f) & (cell['Rarefaction'] == r), mcol].values
                      for f, r in groups]
        bp = ax.boxplot(data_lists, positions=positions, widths=0.75,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.8),
                        whiskerprops=dict(color='black', linewidth=1.0),
                        capprops=dict(color='black', linewidth=1.0),
                        boxprops=dict(linewidth=1.0))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        rng = np.random.default_rng(0)
        for pos, d, c in zip(positions, data_lists, colors):
            jitter = rng.normal(0, 0.07, size=len(d))
            ax.scatter(pos + jitter, d, s=18, color=c,
                       edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3)
        dmin = min(np.concatenate(data_lists))
        for feat, x_left, x_right in [('ASV', 1, 2), ('Taxa', 3.5, 4.5)]:
            p = byfeat[(byfeat.Feature == feat) & (byfeat.Metric == mlabel)]['p_uncorrected'].values[0]
            ymax = max(max(data_lists[positions.index(x_left)]),
                       max(data_lists[positions.index(x_right)]))
            y_line = ymax + 0.02
            ax.plot([x_left, x_right], [y_line, y_line], color='black', lw=1.0)
            sig = sig_stars(p)
            txt = (f"{sig}\nP = {p:.2g}" if (not pd.isna(p) and p < 0.05)
                   else f"{sig} (P = {p:.2g})")
            ax.text((x_left + x_right) / 2, y_line + 0.008, txt,
                    ha='center', va='bottom', fontsize=14)
        ax.set_xticks([1.5, 4.0]); ax.set_xticklabels(['ASV', 'Taxonomic'])
        ax.set_ylabel(mlabel)
        lo = min(0, dmin - 0.05)
        ax.set_ylim(lo, 1.18)
        clean_spines(ax); panel_label(ax, 'abcdef'[idx], fontsize=26)
    handles = [Patch(facecolor=PAL['Non-rarefied'], edgecolor='black', alpha=0.7, label='Non-rarefied'),
               Patch(facecolor=PAL['Rarefied'], edgecolor='black', alpha=0.7, label='Rarefied')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.10,
                        wspace=0.26, hspace=0.24)
    savepng(fig, 'MainFig2_overall_metric_comparison.png')


# --------------------------------------------------------------------------
# MAIN FIG 3 - model selection frequency & stability across rarefaction
# --------------------------------------------------------------------------
def fig3(best):
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    disp = lambda s: s.replace('_',' ')
    order = best.Model.value_counts().index.tolist()
    # (a) overall selection frequency
    ax=axes[0]
    counts = best.Model.value_counts().reindex(order)
    ax.barh([disp(m) for m in order][::-1], counts.values[::-1], color='#4C72B0', alpha=0.85)
    ax.set_xlabel('Times selected as best model (of 480)')
    panel(ax,'a', x=-0.44, y=1.03, fs=26)
    # (b) selection by rarefaction
    ax=axes[1]
    best['Rar'] = best.Dataset.str.split('_').str[0].map({'nonrarefied':'Non-rarefied','rarefied':'Rarefied'})
    tab = best.groupby(['Model','Rar']).size().unstack(fill_value=0).reindex(order)
    y=np.arange(len(order)); h=0.38
    ax.barh(y+h/2, tab['Non-rarefied'].values[::-1], height=h, color=PAL['Non-rarefied'], alpha=0.8, label='Non-rarefied')
    ax.barh(y-h/2, tab['Rarefied'].values[::-1], height=h, color=PAL['Rarefied'], alpha=0.8, label='Rarefied')
    ax.set_yticks(y); ax.set_yticklabels([disp(m) for m in order][::-1], fontsize=16)
    ax.set_xlabel('Times selected as best model')
    ax.legend(frameon=False, fontsize=16)
    panel(ax,'b', x=-0.44, y=1.03, fs=26)
    # (c) AUROC of selected best model by condition
    ax=axes[2]
    order2=[('ASV','Non-rarefied'),('ASV','Rarefied'),('Taxa','Non-rarefied'),('Taxa','Rarefied')]
    best['Feat']=best.Dataset.str.split('_').str[1].map({'asv':'ASV','taxa':'Taxa'})
    data=[best[(best.Feat==f)&(best.Rar==r)]['AUC'].values for f,r in order2]
    xpos=[0,1,2.4,3.4]; cols=[PAL['Non-rarefied'],PAL['Rarefied']]*2
    bp=ax.boxplot(data,positions=xpos,widths=0.7,patch_artist=True,showfliers=False,
                  medianprops=dict(color='black',lw=1.5))
    for patch,c in zip(bp['boxes'],cols): patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticks([0.5,2.9]); ax.set_xticklabels(['ASV','Taxa'])
    ax.set_ylabel('AUROC')
    panel(ax,'c', x=-0.24, y=1.03, fs=26)
    fig.tight_layout()
    # bring panel c closer to panel b (keep a safe distance)
    p2 = axes[2].get_position()
    axes[2].set_position([p2.x0 - 0.055, p2.y0, p2.width, p2.height])
    savepng(fig, 'MainFig3_model_selection.png')


# --------------------------------------------------------------------------
# SUPP FIG 1 - per-study run-level AUROC
# --------------------------------------------------------------------------
def figS1(best):
    studies=sorted(best.ML_Number.unique(),key=lambda s:best[best.ML_Number==s]['AUC'].mean())
    fig,ax=plt.subplots(figsize=(10,7))
    data=[best[best.ML_Number==s]['AUC'].values for s in studies]
    bp=ax.boxplot(data,vert=False,patch_artist=True,showfliers=True,
                  medianprops=dict(color='black',lw=1.5),
                  flierprops=dict(marker='o',ms=3,alpha=0.4))
    for patch in bp['boxes']: patch.set_facecolor('#4C72B0'); patch.set_alpha=0.6
    ax.set_yticklabels([slabel(s) for s in studies])
    ax.set_xlabel('AUROC (per seed x condition, best model)')
    fig.tight_layout(); savepng(fig,'SuppFig1_per_study_run_level_AUC.png')


# --------------------------------------------------------------------------
# SUPP FIG 2 - per-study deltas (forest) for 6 metrics
# --------------------------------------------------------------------------
def figS2(cell):
    fig,axes=plt.subplots(2,3,figsize=(16,11))
    studies=sorted(cell.ML_Number.unique())
    for k,m in enumerate(METRICS):
        ax=axes[k//3,k%3]
        ys=[]; labels=[]
        for i,s in enumerate(studies):
            sub=cell[cell.ML_Number==s]
            nr=sub[sub.Rarefaction=='Non-rarefied'][m].mean()
            r=sub[sub.Rarefaction=='Rarefied'][m].mean()
            ys.append(r-nr); labels.append(slabel(s))
        yv=np.arange(len(studies))
        colors=['#E07B39' if v>0 else '#0F7B8A' for v in ys]
        ax.barh(yv,ys,color=colors,alpha=0.8)
        ax.axvline(0,color='k',lw=0.8)
        ax.set_yticks(yv); ax.set_yticklabels(labels,fontsize=13)
        ax.set_xlabel(f'Δ {MLABEL[m]}')
        ax.set_title(MLABEL[m]); panel(ax,chr(ord('a')+k), fs=26)
    fig.tight_layout(); savepng(fig,'SuppFig2_per_study_deltas_all_metrics.png')


# --------------------------------------------------------------------------
# SUPP FIG 3 - statistical summary
# --------------------------------------------------------------------------
def figS3(pooled, wel):
    fig,axes=plt.subplots(1,3,figsize=(17.5,5.8))
    # (a) mean delta with FDR annotation
    ax=axes[0]
    yv=np.arange(len(pooled))
    ax.barh(yv, pooled['mean_delta'].values, color='#7E57C2', alpha=0.8)
    ax.axvline(0,color='k',lw=0.8)
    ax.set_yticks(yv); ax.set_yticklabels(pooled['Metric']); ax.invert_yaxis()
    ax.set_xlabel('Mean Δ (Rarefied − Non-rarefied)')
    for i,(d,f) in enumerate(zip(pooled['mean_delta'],pooled['p_FDR'])):
        ax.text(d+0.001, i, f'FDR={f:.2f}', va='center', fontsize=13)
    panel(ax,'a', x=-0.30, y=1.04, fs=26)
    # (b) rank-biserial effect size
    ax=axes[1]
    ax.barh(yv, pooled['rank_biserial'].values, color='#2C7B5C', alpha=0.8)
    ax.axvline(0,color='k',lw=0.8)
    ax.set_yticks(yv); ax.set_yticklabels(pooled['Metric']); ax.invert_yaxis()
    ax.set_xlabel('Rank-biserial effect size')
    panel(ax,'b', x=-0.30, y=1.04, fs=26)
    # (c) win/equiv/loss stacked (pooled over feature)
    ax=axes[2]
    agg=wel.groupby('Metric')[['Wins','Equivalent','Losses']].sum().reindex([MLABEL[m] for m in METRICS])
    yv2=np.arange(len(agg))
    ax.barh(yv2, agg['Wins'], color='#4DAF4A', label='Wins (R>NR)')
    ax.barh(yv2, agg['Equivalent'], left=agg['Wins'], color='#BBBBBB', label='Equivalent')
    ax.barh(yv2, agg['Losses'], left=agg['Wins']+agg['Equivalent'], color='#E64B35', label='Losses (R<NR)')
    ax.set_yticks(yv2); ax.set_yticklabels(agg.index); ax.invert_yaxis()
    ax.set_xlabel('Studies x feature cells (n=24)')
    ax.legend(frameon=False, fontsize=13, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    panel(ax,'c', x=-0.30, y=1.04, fs=26)
    fig.tight_layout(); savepng(fig,'SuppFig3_statistical_summary.png')


# --------------------------------------------------------------------------
# SUPP FIG 4 - decision synthesis (TOST + Bland-Altman)
# --------------------------------------------------------------------------
def figS4(piv):
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    # TOST for AUROC & BA
    for col,m in enumerate(['AUC','Balanced_Accuracy']):
        ax=axes[0,col]
        for i,feat in enumerate(['ASV','Taxa']):
            pv=piv.xs(feat,level='Feature'); d=pv[(m,'Rarefied')].values-pv[(m,'Non-rarefied')].values
            n=len(d); se=d.std(ddof=1)/np.sqrt(n); mean=d.mean()
            tc=stats.t.ppf(0.95,n-1); lo,hi=mean-tc*se,mean+tc*se
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',ms=9,capsize=5,
                        color=PAL[feat],lw=2)
        ax.axvspan(-0.02,0.02,color='green',alpha=0.12)
        ax.axvline(0,color='k',lw=0.8,ls='--')
        ax.set_yticks([0,1]); ax.set_yticklabels(['ASV','Taxa'])
        ax.set_xlabel(f'Δ {MLABEL[m]} (90% CI)')
        ax.set_xlim(-0.05,0.05); panel(ax,'a' if col==0 else 'b', x=-0.20, y=1.05, fs=26)
    # Bland-Altman for AUROC (ASV, Taxa)
    for col,feat in enumerate(['ASV','Taxa']):
        ax=axes[1,col]
        pv=piv.xs(feat,level='Feature')
        nr=pv[('AUC','Non-rarefied')].values; r=pv[('AUC','Rarefied')].values
        mean=(nr+r)/2; diff=r-nr
        ax.scatter(mean,diff,color=PAL[feat],s=45,alpha=0.75,zorder=3)
        bias=diff.mean(); sd=diff.std(ddof=1)
        ax.axhline(bias,color='k',lw=1.2); ax.axhline(bias+1.96*sd,color='r',ls='--',lw=1)
        ax.axhline(bias-1.96*sd,color='r',ls='--',lw=1)
        ax.set_xlabel('Mean AUROC'); ax.set_ylabel('Δ AUROC (R − NR)')
        panel(ax,'c' if col==0 else 'd', x=-0.20, y=1.05, fs=26)
    fig.tight_layout(); savepng(fig,'SuppFig4_decision_synthesis.png')


# --------------------------------------------------------------------------
# SUPP FIG 5 - delta heatmaps study x metric (ASV, Taxa)
# --------------------------------------------------------------------------
def figS5(cell):
    studies=sorted(cell.ML_Number.unique())
    fig,axes=plt.subplots(1,2,figsize=(14,8))
    for col,feat in enumerate(['ASV','Taxa']):
        M=np.zeros((len(studies),len(METRICS)))
        for i,s in enumerate(studies):
            sub=cell[(cell.ML_Number==s)&(cell.Feature==feat)]
            for j,m in enumerate(METRICS):
                nr=sub[sub.Rarefaction=='Non-rarefied'][m].mean()
                r=sub[sub.Rarefaction=='Rarefied'][m].mean()
                M[i,j]=r-nr
        ax=axes[col]
        im=ax.imshow(M,cmap='RdBu_r',aspect='auto',vmin=-0.15,vmax=0.15)
        ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([MLABEL[m] for m in METRICS],rotation=45,ha='right',fontsize=14)
        ax.set_yticks(range(len(studies))); ax.set_yticklabels([slabel(s) for s in studies],fontsize=13)
        for i in range(len(studies)):
            for j in range(len(METRICS)):
                ax.text(j,i,f"{M[i,j]:+.02f}",ha='center',va='center',fontsize=10,
                        color='white' if abs(M[i,j])>0.09 else 'black')
        panel(ax,'a' if col==0 else 'b', x=-0.20, y=1.05, fs=26)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.03)
    fig.tight_layout(); savepng(fig,'SuppFig5_delta_heatmaps.png')


# --------------------------------------------------------------------------
# SUPP FIG 6 - all metrics distribution by condition
# --------------------------------------------------------------------------
def figS6(cell):
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    order=[('ASV','Non-rarefied'),('ASV','Rarefied'),('Taxa','Non-rarefied'),('Taxa','Rarefied')]
    labs=['ASV NR','ASV R','Taxa NR','Taxa R']
    cols=[PAL['Non-rarefied'],PAL['Rarefied'],PAL['Non-rarefied'],PAL['Rarefied']]
    for k,m in enumerate(METRICS):
        ax=axes[k//3,k%3]
        data=[cell[(cell.Feature==f)&(cell.Rarefaction==r)][m].values for f,r in order]
        parts=ax.violinplot(data,positions=range(4),showmedians=True,widths=0.8)
        for pc,c in zip(parts['bodies'],cols): pc.set_facecolor(c); pc.set_alpha(0.6)
        ax.set_xticks(range(4)); ax.set_xticklabels(labs,fontsize=14,rotation=20)
        ax.set_ylabel(MLABEL[m]); ax.set_title(MLABEL[m]); panel(ax,chr(ord('a')+k), x=-0.24, y=1.05, fs=26)
    fig.tight_layout(); savepng(fig,'SuppFig6_all_metrics_per_condition.png')


# --------------------------------------------------------------------------
# SUPP FIG 7 - sensitivity-specificity trade-off
# --------------------------------------------------------------------------
def figS7(cell):
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for col,feat in enumerate(['ASV','Taxa']):
        ax=axes[col]
        for r,c in [('Non-rarefied',PAL['Non-rarefied']),('Rarefied',PAL['Rarefied'])]:
            sub=cell[(cell.Feature==feat)&(cell.Rarefaction==r)]
            ax.scatter(sub['Sensitivity'],sub['Specificity'],color=c,s=55,alpha=0.7,label=r,edgecolor='white',lw=0.5)
        ax.plot([0,1],[1,0],'k--',lw=0.6,alpha=0.4)
        ax.set_xlabel('Sensitivity'); ax.set_ylabel('Specificity')
        ax.set_xlim(0.3,1.02); ax.set_ylim(0.3,1.02)
        ax.legend(frameon=False); panel(ax,'a' if col==0 else 'b', x=-0.20, y=1.05, fs=26)
    fig.tight_layout(); savepng(fig,'SuppFig7_sens_spec_tradeoff.png')


# --------------------------------------------------------------------------
# SUPP FIG 9 - win/equiv/loss tally
# --------------------------------------------------------------------------
def figS9(wel):
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for col,feat in enumerate(['ASV','Taxa']):
        ax=axes[col]
        sub=wel[wel.Feature==feat].set_index('Metric').reindex([MLABEL[m] for m in METRICS])
        x=np.arange(len(sub))
        ax.bar(x,sub['Wins'],color='#4DAF4A',label='Rarefaction wins')
        ax.bar(x,sub['Equivalent'],bottom=sub['Wins'],color='#BBBBBB',label='Equivalent')
        ax.bar(x,sub['Losses'],bottom=sub['Wins']+sub['Equivalent'],color='#E64B35',label='Rarefaction loses')
        ax.set_xticks(x); ax.set_xticklabels(sub.index,rotation=45,ha='right',fontsize=14)
        ax.set_ylim(0,12)
        if col==0: ax.set_ylabel('Number of studies')
        panel(ax,'a' if col==0 else 'b', x=-0.20, y=1.05, fs=26)
    handles=[plt.Rectangle((0,0),1,1,color=c) for c in ['#4DAF4A','#BBBBBB','#E64B35']]
    fig.legend(handles,['Rarefaction wins (Δ > +0.01)','Equivalent (|Δ| ≤ 0.01)','Rarefaction loses (Δ < −0.01)'],
               loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.03))
    fig.tight_layout(rect=[0,0.04,1,1]); savepng(fig,'SuppFig9_win_equiv_loss_tally.png')


# --------------------------------------------------------------------------
# TABLES
# --------------------------------------------------------------------------
def tables(pooled, byfeat, wel, best):
    # TableS6: full statistics (pooled + by feature)
    p2 = pooled.rename(columns={'NR':'Non_rarefied_mean','R':'Rarefied_mean'}).copy()
    p2 = p2[['Metric','Non_rarefied_mean','Rarefied_mean','mean_delta','median_delta',
             'rank_biserial','p','p_FDR','Significant']].round(4)
    bf = byfeat.rename(columns={'NR':'Non_rarefied_mean','R':'Rarefied_mean'})
    bf = bf[['Feature','Metric','Non_rarefied_mean','Rarefied_mean','mean_delta','p_uncorrected']].round(4)
    with open(f'{OUT}/TableS6_full_statistics.csv','w') as fh:
        fh.write('# Pooled rarefaction effect (paired Wilcoxon over study x feature, n=24; BH-FDR over 6 metrics)\n')
        p2.to_csv(fh, index=False)
        fh.write('\n# By feature representation (paired Wilcoxon over 12 studies, uncorrected)\n')
        bf.to_csv(fh, index=False)
    print('Saved TableS6_full_statistics.csv')

    # Pooled performance summary (Is_Best)
    rows=[]
    for m in METRICS:
        q=best[m].quantile([.25,.5,.75])
        rows.append(dict(Metric=MLABEL[m], Mean=round(best[m].mean(),3),
                         Median=round(q[.5],3), Q1=round(q[.25],3), Q3=round(q[.75],3)))
    pd.DataFrame(rows).to_csv(f'{OUT}/TableS6b_pooled_performance_summary.csv',index=False)
    print('Saved TableS6b_pooled_performance_summary.csv')

    # TableS3: win/equiv/loss
    wel.rename(columns={'Wins':'Wins (R>NR)','Losses':'Losses (R<NR)'}).to_csv(
        f'{OUT}/TableS3_win_equiv_loss_tally.csv', index=False)
    print('Saved TableS3_win_equiv_loss_tally.csv')

    # Model selection frequency table
    ms = best.Model.value_counts().reset_index()
    ms.columns=['Model','Times_selected_best']
    ms['Percent']=(100*ms['Times_selected_best']/len(best)).round(1)
    ms.to_csv(f'{OUT}/TableS4_model_selection_frequency.csv',index=False)
    print('Saved TableS4_model_selection_frequency.csv')


def main():
    best, cell = load()
    pooled, piv = pooled_stats(cell)
    byfeat = byfeature_stats(piv)
    wel = win_equiv_loss(piv)
    print(pooled.to_string()); print(); print(wel.to_string())
    fig1(cell)
    with mpl.rc_context(BIG):
        fig2(cell, byfeat); fig3(best)
        figS2(cell); figS3(pooled, wel); figS4(piv)
        figS5(cell); figS6(cell); figS7(cell); figS9(wel)
    figS1(best)
    tables(pooled, byfeat, wel, best)
    print('\nDone.')

if __name__ == '__main__':
    main()
