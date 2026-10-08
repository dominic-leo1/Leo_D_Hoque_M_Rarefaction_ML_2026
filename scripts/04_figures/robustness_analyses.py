#!/usr/bin/env python3
"""
robustness_analyses.py
======================
Robustness and sensitivity checks on the SHAP biomarker tables:
  * Top-K Jaccard sensitivity at K = 10, 20, 50 and 100   -> Table S5, Supp Fig 10
  * SHAP magnitude-floor robustness of the top-20 overlap  -> Table S2

Reads ../../data/shap_bodysite_*.csv. Outputs -> output/.
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams.update({
    'figure.dpi': 100, 'savefig.dpi': 600,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
ASV_COLOR = '#377EB8'; TAXA_COLOR = '#E64B35'

SRC = '../../data'
OUT = 'output'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Top-K Jaccard sensitivity
# ============================================================
SHAP_FILES = {
    'Gut':         f'{SRC}/shap_bodysite_gut.csv',
    'Oral':        f'{SRC}/shap_bodysite_oral.csv',
    'Blood':       f'{SRC}/shap_bodysite_blood.csv',
    'Respiratory': f'{SRC}/shap_bodysite_respiratory.csv',
    'Liver':       f'{SRC}/shap_bodysite_liver.csv',
    'Urinary':     f'{SRC}/shap_bodysite_catheter_urinary.csv',
}
SITES = list(SHAP_FILES.keys())

K_VALUES = [10, 20, 50, 100]
jrows = []
for site, fp in SHAP_FILES.items():
    df = pd.read_csv(fp)
    for ft in ['ASV','Taxa']:
        nr = df[df['Group']==f'Non-rarefied {ft}']
        r  = df[df['Group']==f'Rarefied {ft}']
        # Convention from main script: take min Rank per Feature across studies
        nr_minrank = nr.groupby('Feature')['Rank'].min()
        r_minrank  = r.groupby('Feature')['Rank'].min()
        for K in K_VALUES:
            top_nr = set(nr_minrank[nr_minrank <= K].index)
            top_r  = set(r_minrank[r_minrank <= K].index)
            j = len(top_nr & top_r) / len(top_nr | top_r) if (top_nr | top_r) else float('nan')
            jrows.append({'Body_Site': site, 'Feature_type': ft, 'K': K,
                         'N_top_NR': len(top_nr), 'N_top_R': len(top_r),
                         'N_intersection': len(top_nr & top_r),
                         'N_union': len(top_nr | top_r),
                         'Jaccard': round(j, 3)})

jacc = pd.DataFrame(jrows)
jacc.to_csv(f'{OUT}/TableS5_jaccard_topK_sensitivity.csv', index=False)

# Figure: lines per body site, separate ASV/Taxa panels
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
markers = ['o','s','^','D','v','P']
for col, ft in enumerate(['ASV','Taxa']):
    ax = axes[col]
    for site, mk in zip(SITES, markers):
        sub = jacc[(jacc['Body_Site']==site) & (jacc['Feature_type']==ft)].sort_values('K')
        color = ASV_COLOR if ft=='ASV' else TAXA_COLOR
        ax.plot(sub['K'], sub['Jaccard'], '-'+mk, label=site, ms=8, lw=1.5, color=color, alpha=0.85)
    # Mean
    mean_per_k = jacc[jacc['Feature_type']==ft].groupby('K')['Jaccard'].mean()
    ax.plot(mean_per_k.index, mean_per_k.values, 'k--', lw=2, label='Mean', alpha=0.8)
    ax.set_xlabel('Top-K threshold')
    ax.set_ylabel('Jaccard (NR vs R)' if col==0 else '')
    ax.set_title(f'{ft} features', fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(K_VALUES); ax.set_xticklabels(K_VALUES)
    ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    if col == 1:
        ax.legend(loc='upper right', frameon=True, fontsize=9, ncol=2)
for i, ax in enumerate(axes):
    ax.text(-0.06, 1.05, chr(ord('a')+i), transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/SuppFig10_jaccard_topK_sensitivity.png', bbox_inches='tight', dpi=600)
plt.close(fig)
print(f"Table S5 + Supp Fig 10 written ({len(jacc)} rows)")

# Summary
print("\n=== Mean Jaccard by K ===")
print(jacc.pivot_table(index='K', columns='Feature_type', values='Jaccard', aggfunc='mean').round(3).to_string())

# ============================================================
# SHAP magnitude-floor robustness
# ============================================================
# Drop artefactual rank-1 SHAP magnitudes (|SHAP| > 1e6, clearly non-physical for SHAP values
# which should be in 0-1 range), and recompute top-20 Jaccard using mean |SHAP|-based ranking
# AND test with a magnitude floor (drop bottom 10% by |SHAP|).
rob_rows = []
for site, fp in SHAP_FILES.items():
    df = pd.read_csv(fp)
    df['absSHAP'] = df['SHAP'].abs()
    artefact_mask = df['absSHAP'] > 1e6
    n_art = int(artefact_mask.sum())
    dfc = df[~artefact_mask].copy()

    for ft in ['ASV','Taxa']:
        nr = df[df['Group']==f'Non-rarefied {ft}']
        r  = df[df['Group']==f'Rarefied {ft}']

        # (full) top-20 by min rank
        nr_rk = nr.groupby('Feature')['Rank'].min()
        r_rk  = r.groupby('Feature')['Rank'].min()
        tnr = set(nr_rk[nr_rk <= 20].index)
        tr  = set(r_rk[r_rk <= 20].index)
        j_full = len(tnr & tr) / len(tnr | tr) if (tnr | tr) else float('nan')

        # (drop artefact) top-20 by max |SHAP| pooled across studies
        ncl = dfc[dfc['Group']==f'Non-rarefied {ft}']
        rcl = dfc[dfc['Group']==f'Rarefied {ft}']
        nr_max = ncl.groupby('Feature')['absSHAP'].max().sort_values(ascending=False)
        r_max  = rcl.groupby('Feature')['absSHAP'].max().sort_values(ascending=False)
        tnr2 = set(nr_max.head(20).index)
        tr2  = set(r_max.head(20).index)
        j_noart = len(tnr2 & tr2) / len(tnr2 | tr2) if (tnr2 | tr2) else float('nan')

        # (magnitude floor) drop |SHAP| below 10th percentile of all non-artefact magnitudes,
        # then top-20 by min rank
        floor = np.percentile(dfc['absSHAP'][dfc['absSHAP']>0], 10)
        flo_nr = dfc[(dfc['Group']==f'Non-rarefied {ft}') & (dfc['absSHAP'] >= floor)]
        flo_r  = dfc[(dfc['Group']==f'Rarefied {ft}') & (dfc['absSHAP'] >= floor)]
        nf_rk = flo_nr.groupby('Feature')['Rank'].min()
        rf_rk = flo_r.groupby('Feature')['Rank'].min()
        tnr3 = set(nf_rk[nf_rk <= 20].index) if len(nf_rk)>=20 else set(nf_rk.index)
        tr3  = set(rf_rk[rf_rk <= 20].index) if len(rf_rk)>=20 else set(rf_rk.index)
        j_floor = len(tnr3 & tr3) / len(tnr3 | tr3) if (tnr3 | tr3) else float('nan')

        rob_rows.append({
            'Body_Site': site,
            'Feature_type': ft,
            'N_artefact_dropped': n_art,
            'Jaccard_full': round(j_full, 3),
            'Jaccard_no_artefact_magnitude': round(j_noart, 3),
            'Jaccard_magnitude_floor_top10pct': round(j_floor, 3),
        })

rob = pd.DataFrame(rob_rows)
rob.to_csv(f'{OUT}/TableS2_shap_magnitude_robustness.csv', index=False)

# Summary
print("\n=== SHAP magnitude robustness ===")
print(rob.to_string(index=False))
print(f"\nMean ASV: full={rob[rob.Feature_type=='ASV'].Jaccard_full.mean():.3f}, "
      f"floored={rob[rob.Feature_type=='ASV'].Jaccard_magnitude_floor_top10pct.mean():.3f}")
print(f"Mean Taxa: full={rob[rob.Feature_type=='Taxa'].Jaccard_full.mean():.3f}, "
      f"floored={rob[rob.Feature_type=='Taxa'].Jaccard_magnitude_floor_top10pct.mean():.3f}")

print("\nAll new analyses complete.")
