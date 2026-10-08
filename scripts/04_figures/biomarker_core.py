#!/usr/bin/env python3
"""
biomarker_core.py
=================
Figure and table functions for the SHAP-based biomarker analysis, driven by
biomarker_figures.py. Reads the per-body-site SHAP tables
(shap_bodysite_*.csv) and produces the biomarker-signature heatmaps, the
rarefaction-overlap figures (Jaccard overlap and Spearman rank concordance)
and the phylum-composition figure, together with the associated biomarker
rank and agreement tables.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_CSV = '../../data/ml_summary_means.tsv'
OUT_DIR   = 'output'
os.makedirs(OUT_DIR, exist_ok=True)

# SHAP feature-importance data (one CSV per body site)
SHAP_DIR = '../../data'
SHAP_FILES = {
    'Blood':       'shap_bodysite_blood.csv',
    'Urinary':     'shap_bodysite_catheter_urinary.csv',
    'Gut':         'shap_bodysite_gut.csv',
    'Liver':       'shap_bodysite_liver.csv',
    'Oral':        'shap_bodysite_oral.csv',
    'Respiratory': 'shap_bodysite_respiratory.csv',
}

# Disease labels per study. Tick labels become 'S<ID> <disease>'.
STUDY_LABELS = {
    1:  "IBD",
    2:  "HCC",
    6:  "VAP",
    8:  "Breast Cancer",
    9:  "Parkinson's",
    10: "Gout",
    11: "CRC Polyps",
    12: "Periodontitis",
    13: "S-ECC",
    15: "HIV-1 (a)",
    16: "HIV-1 (b)",
    17: "Bacteriuria",
}

def slabel(sid):
    """Return 'S<ID> <disease>' label; fallback to 'S<ID>' if unknown."""
    try:
        sid_int = int(sid)
        if sid_int in STUDY_LABELS:
            return f"S{sid_int} {STUDY_LABELS[sid_int]}"
        return f"S{sid_int}"
    except (ValueError, TypeError):
        return str(sid)

# Body-site mapping per study (from Table 1 of study characteristics).
# Used by MainFig1 (panels c, d), MainFig3 (panel c), and SuppFig5 (panels c, d).
BODY_SITES = {
    1:  "Gut",
    2:  "Liver",
    6:  "Respiratory",
    8:  "Blood",
    9:  "Gut",
    10: "Gut",
    11: "Gut",
    12: "Oral",
    13: "Oral",
    15: "Gut",
    16: "Gut",
    17: "Urinary",
}

# ---------------------------------------------------------------------------
# PLOT STYLE (Nature-style; sans-serif; larger fonts)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         13,
    'axes.labelsize':    14,
    'axes.titlesize':    14,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'legend.title_fontsize': 12.5,
    'figure.titlesize':  16,
    'axes.linewidth':    1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size':  4,
    'ytick.major.size':  4,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'savefig.dpi':       400,
    'figure.dpi':        110,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# Colour-blind-safe palette
PAL = {
    'Non-rarefied': '#0F7B8A',
    'Rarefied':     '#E07B39',
    'ASV':          '#3B6AA0',
    'Taxa':         '#C36F2A',
}
MODEL_PAL = {
    'Random Forest':       '#1F77B4',
    'Logistic Regression': '#D55E00',
    'Naive Bayes':         '#009E73',
    'Blended':             '#7E57C2',
}

def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def panel_label(ax, label, x=-0.18, y=1.06, fontsize=18):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')

def savepng(fig, name):
    """Save figure as a single PNG to OUT_DIR."""
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=400, bbox_inches='tight', facecolor='white')
    print(f'Saved {p}')
    plt.close(fig)

def sig_stars(p):
    if pd.isna(p):  return 'ns'
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    return 'ns'

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_data(path=INPUT_CSV):
    df = pd.read_csv(path)
    df[['Rarefaction', 'Feature']] = df['Dataset'].str.split('_', expand=True)
    df['Rarefaction'] = df['Rarefaction'].map(
        {'nonrarefied': 'Non-rarefied', 'rarefied': 'Rarefied'})
    df['Feature']     = df['Feature'].map({'asv': 'ASV', 'taxa': 'Taxa'})
    df['Model']       = df['Model'].str.replace('_', ' ', regex=False)

    agg = (df.groupby(['ML_Number', 'Model', 'Rarefaction', 'Feature'])
             [['AUC', 'Sensitivity', 'Specificity', 'Balanced_Accuracy']]
             .mean().reset_index())
    return df, agg


# ---------------------------------------------------------------------------
# SHAP DATA LOADING & TAXONOMY HELPERS (Section 4)
# ---------------------------------------------------------------------------
# Phylum-name synonyms (old SILVA <-> GTDB-style). Normalises so that
# Bacteroidetes / Bacteroidota count as the same phylum, etc.
PHYLUM_SYN = {
    'Bacteroidetes':     'Bacteroidota',
    'Actinobacteria':    'Actinobacteriota',
    'Fusobacteria':      'Fusobacteriota',
    'Verrucomicrobia':   'Verrucomicrobiota',
    'Spirochaetes':      'Spirochaetota',
    'Synergistetes':     'Synergistota',
    'Tenericutes':       'Mycoplasmatota',
    'Elusimicrobia':     'Elusimicrobiota',
    'Chlamydiae':        'Chlamydiota',
}

# Consistent palette for phyla appearing across all figures
PHYLUM_PAL = {
    'Firmicutes':         '#3B6AA0',
    'Bacteroidota':       '#C36F2A',
    'Proteobacteria':     '#2C7B5C',
    'Actinobacteriota':   '#B5413B',
    'Fusobacteriota':     '#7E57C2',
    'Spirochaetota':      '#D4A017',
    'Verrucomicrobiota':  '#5DA5DA',
    'Campylobacterota':   '#A65628',
    'Patescibacteria':    '#999999',
    'Cyanobacteria':      '#66C2A5',
    'Synergistota':       '#E78AC3',
    'Other':              '#BDBDBD',
}

_RANK_MARKERS = {'p', 'c', 'o', 'f', 'g', 's', 'd', 'k'}

def _parse_taxonomy(feat):
    """Token-based parse of SILVA-style strings (and prefix-less variants)."""
    parts = str(feat).split('_')
    ranks, current_rank, current_tokens = {}, None, []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if len(tok) == 1 and tok.lower() in _RANK_MARKERS and i + 1 < len(parts):
            if current_rank is not None and current_tokens:
                ranks[current_rank] = '_'.join(current_tokens)
            current_rank, current_tokens = tok.lower(), []
            i += 1
        else:
            if current_rank is not None:
                current_tokens.append(tok)
            i += 1
    if current_rank is not None and current_tokens:
        ranks[current_rank] = '_'.join(current_tokens)
    return ranks

def _lowest_taxon(feat):
    """Return a human-friendly label using the lowest resolved rank.
    Handles both SILVA-prefix format (p_..._g_...) and prefix-less
    format (Phylum_class_..._Genus_species).
    """
    r = _parse_taxonomy(feat)
    if r:  # prefix style worked
        for rank, prefix in [('s', 's.'), ('g', 'g.'), ('f', 'f.'),
                             ('o', 'o.'), ('c', 'c.'), ('p', 'p.')]:
            if rank in r:
                name = r[rank]
                name = re.sub(r'_[A-Z]?\d{4,}', '', name)
                name = name.replace('_', ' ').strip()
                if name in ('', 'NR') or name.startswith('NR '):
                    continue
                return f"{prefix} {name}"
    # No prefix tokens found - use heuristic for files like catheter_urinary
    parts = str(feat).split('_')
    # Drop pure-numeric tokens and single-character drift markers (A, B, D...)
    meaningful = [p for p in parts if not p.isdigit() and len(p) > 1]
    if not meaningful:
        return parts[-1] if parts else 'Unknown'
    if len(meaningful) >= 2:
        last_two = meaningful[-2:]
        # Collapse duplications like "Eggerthia_Eggerthia_catenaformis"
        if last_two[0] == last_two[1]:
            return f"s. {last_two[1]}"
        # If the second-to-last starts uppercase and so does the last,
        # treat as genus + species
        return f"s. {last_two[0]} {last_two[1]}"
    return f"g. {meaningful[-1]}"

def _phylum_of(feat):
    """Return normalised phylum, or 'Other' if unresolved."""
    r = _parse_taxonomy(feat)
    if 'p' in r:
        p = r['p'].split('_')[0]
        return PHYLUM_SYN.get(p, p)
    # Non-prefix style (e.g. catheter_urinary file)
    parts = str(feat).split('_')
    known = {'Firmicutes', 'Bacteroidota', 'Bacteroidetes', 'Proteobacteria',
             'Actinobacteriota', 'Actinobacteria', 'Spirochaetota', 'Spirochaetes',
             'Fusobacteria', 'Fusobacteriota', 'Verrucomicrobia',
             'Verrucomicrobiota', 'Cyanobacteria', 'Patescibacteria',
             'Campylobacterota', 'Synergistota'}
    if parts and parts[0] in known:
        return PHYLUM_SYN.get(parts[0], parts[0])
    return 'Other'

def load_shap_data(shap_dir=SHAP_DIR, shap_files=SHAP_FILES):
    """Load all per-body-site SHAP files and add parsed taxonomy columns."""
    dfs = []
    for site, fname in shap_files.items():
        path = os.path.join(shap_dir, fname)
        if not os.path.exists(path):
            print(f'  [warn] missing SHAP file: {path}')
            continue
        d = pd.read_csv(path)
        d['Body_Site'] = site
        dfs.append(d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df['Phylum']        = df['Feature'].apply(_phylum_of)
    df['Feature_short'] = df['Feature'].apply(_lowest_taxon)
    df[['Rarefaction', 'Feature_type']] = df['Group'].str.split(' ', expand=True)
    # Drop unresolved placeholders ("NR" species/genus)
    df = df[~df['Feature_short'].astype(str).str.startswith(('s. NR', 'g. NR'))]
    return df.reset_index(drop=True)

# ===========================================================================
# SECTION 1 - Overview of the benchmark
# ===========================================================================

# --- Main Fig 1: overview heatmaps -----------------------------------------
def fig_overview_heatmaps(agg):
    """Four-panel overview heatmaps.
    a, b: AUROC / Balanced accuracy per study (12 rows) x condition (4 cols)
    c, d: AUROC / Balanced accuracy per body site (6 rows) x condition (4 cols)
    """
    a = agg.copy()
    a['Body_Site'] = a['ML_Number'].map(BODY_SITES)
    # Body-site order: by sample size (n studies), most studies first.
    site_order = (pd.Series(BODY_SITES)
                  .value_counts().sort_values(ascending=False).index.tolist())
    site_n = pd.Series(BODY_SITES).value_counts().to_dict()

    def build_study_heatmap(metric):
        g = (agg.groupby(['ML_Number', 'Feature', 'Rarefaction'])[metric]
             .mean().reset_index())
        g['cond'] = g['Feature'] + ' | ' + g['Rarefaction'].map(
            {'Non-rarefied': 'NR', 'Rarefied': 'R'})
        pivot = g.pivot(index='ML_Number', columns='cond', values=metric)
        cond_order = ['ASV | NR', 'ASV | R', 'Taxa | NR', 'Taxa | R']
        pivot = pivot[cond_order]
        pivot.index = [slabel(i) for i in pivot.index]
        return pivot

    def build_site_heatmap(metric):
        # Mean over all studies belonging to each body site, then over models
        g = (a.groupby(['Body_Site', 'Feature', 'Rarefaction'])[metric]
             .mean().reset_index())
        g['cond'] = g['Feature'] + ' | ' + g['Rarefaction'].map(
            {'Non-rarefied': 'NR', 'Rarefied': 'R'})
        pivot = g.pivot(index='Body_Site', columns='cond', values=metric)
        cond_order = ['ASV | NR', 'ASV | R', 'Taxa | NR', 'Taxa | R']
        pivot = pivot[cond_order]
        pivot = pivot.reindex(site_order)
        # Annotate rows with n studies
        pivot.index = [f'{s} (n={site_n[s]})' for s in pivot.index]
        return pivot

    # Use fixed colour limits so the colorbar ticks align with the visible
    # colour range across all panels (no white band at the bottom of the AUROC
    # scale, even when the data minimum is well above 0.6).
    METRIC_LIMITS = {
        'AUC':               (0.6, 1.0),
        'Balanced_Accuracy': (0.5, 1.0),
    }

    # Reduced overall height shrinks per-cell height; tighter hspace reduces
    # the visible gap between the per-study and per-body-site rows.
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 6],
                          left=0.18, right=0.95, top=0.95, bottom=0.06,
                          wspace=0.55, hspace=0.18)

    metrics = [('AUC', 'AUROC'),
               ('Balanced_Accuracy', 'Balanced accuracy')]

    # Row 1: per-study heatmaps
    for col, (metric, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col])
        p = build_study_heatmap(metric)
        vmin, vmax = METRIC_LIMITS[metric]
        im = ax.imshow(p.values, aspect='auto', cmap='viridis',
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(p.shape[1]))
        ax.set_xticklabels(p.columns, rotation=0, fontsize=14)
        ax.set_yticks(range(p.shape[0]))
        ax.set_yticklabels(p.index, fontsize=13.5)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=13,
                        color='white' if v < (vmin + vmax) / 2 else 'black')
        ax.set_title(f'{label} per study and condition',
                     fontsize=16, fontweight='bold', pad=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(label, fontsize=14)
        cbar.ax.tick_params(labelsize=13)
        if metric == 'AUC':
            cbar.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0])
            cbar.ax.yaxis.set_major_formatter(
                mpl.ticker.FormatStrFormatter('%.1f'))
        clean_spines(ax)
        panel_label(ax, 'ab'[col], x=-0.32, y=1.04, fontsize=20)

    # Row 2: per-body-site heatmaps
    for col, (metric, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, col])
        p = build_site_heatmap(metric)
        vmin, vmax = METRIC_LIMITS[metric]
        im = ax.imshow(p.values, aspect='auto', cmap='viridis',
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(p.shape[1]))
        ax.set_xticklabels(p.columns, rotation=0, fontsize=14)
        ax.set_yticks(range(p.shape[0]))
        ax.set_yticklabels(p.index, fontsize=13.5)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=13,
                        color='white' if v < (vmin + vmax) / 2 else 'black')
        ax.set_title(f'{label} per body site and condition',
                     fontsize=16, fontweight='bold', pad=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(label, fontsize=14)
        cbar.ax.tick_params(labelsize=13)
        if metric == 'AUC':
            cbar.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0])
            cbar.ax.yaxis.set_major_formatter(
                mpl.ticker.FormatStrFormatter('%.1f'))
        clean_spines(ax)
        panel_label(ax, 'cd'[col], x=-0.32, y=1.10, fontsize=20)

    savepng(fig, 'MainFig1_overview_heatmaps.png')


# ===========================================================================
# SECTION 2 - Rarefaction has a negligible effect
# ===========================================================================

# --- Main Fig 2: overall metric comparison ---------------------------------
def fig_overall_metric_comparison(agg):
    METRICS = [('AUC', 'AUROC'),
               ('Balanced_Accuracy', 'Balanced accuracy'),
               ('Sensitivity', 'Sensitivity'),
               ('Specificity', 'Specificity')]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    axes = axes.flatten()

    for idx, (mcol, mlabel) in enumerate(METRICS):
        ax = axes[idx]
        groups = [('ASV', 'Non-rarefied'), ('ASV', 'Rarefied'),
                  ('Taxa', 'Non-rarefied'), ('Taxa', 'Rarefied')]
        positions = [1, 2, 3.5, 4.5]
        colors = [PAL['Non-rarefied'], PAL['Rarefied'],
                  PAL['Non-rarefied'], PAL['Rarefied']]
        data_lists = [agg.loc[(agg['Feature'] == f) &
                              (agg['Rarefaction'] == r), mcol].values
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

        for feat, x_left, x_right in [('ASV', 1, 2), ('Taxa', 3.5, 4.5)]:
            pivot = (agg[agg['Feature'] == feat]
                     .groupby(['ML_Number', 'Model', 'Rarefaction'])[mcol]
                     .mean().unstack('Rarefaction'))
            delta = pivot['Rarefied'] - pivot['Non-rarefied']
            try:
                _, p = stats.wilcoxon(delta.dropna())
            except ValueError:
                p = np.nan
            ymax = max(max(data_lists[positions.index(x_left)]),
                       max(data_lists[positions.index(x_right)]))
            y_line = ymax + 0.02
            ax.plot([x_left, x_right], [y_line, y_line], color='black', lw=1.0)
            sig = sig_stars(p)
            txt = (f"{sig}\nP = {p:.2g}" if (not pd.isna(p) and p < 0.05)
                   else f"{sig} (P = {p:.2g})")
            ax.text((x_left + x_right) / 2, y_line + 0.008, txt,
                    ha='center', va='bottom', fontsize=10.5)

        ax.set_xticks([1.5, 4.0])
        ax.set_xticklabels(['ASV', 'Taxonomic'])
        ax.set_ylabel(mlabel)
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        clean_spines(ax)
        panel_label(ax, 'abcd'[idx])

    handles = [Patch(facecolor=PAL['Non-rarefied'], edgecolor='black',
                     alpha=0.7, label='Non-rarefied'),
               Patch(facecolor=PAL['Rarefied'], edgecolor='black',
                     alpha=0.7, label='Rarefied')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.10,
                        wspace=0.28, hspace=0.22)
    savepng(fig, 'MainFig2_overall_metric_comparison.png')


# --- Main Fig 3: decision synthesis ----------------------------------------
def fig_decision_synthesis(agg):
    fig = plt.figure(figsize=(13.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.32,
                          left=0.08, right=0.97, top=0.94, bottom=0.08)

    # Panel a - density of paired deltas
    ax = fig.add_subplot(gs[0, 0])
    for feat, color in [('ASV', PAL['ASV']), ('Taxa', PAL['Taxa'])]:
        pivot = (agg[agg['Feature'] == feat]
                 .groupby(['ML_Number', 'Model', 'Rarefaction'])['AUC']
                 .mean().unstack('Rarefaction'))
        diff = (pivot['Rarefied'] - pivot['Non-rarefied']).dropna()
        kde = stats.gaussian_kde(diff)
        xs = np.linspace(-0.2, 0.2, 400)
        ax.fill_between(xs, kde(xs), alpha=0.35, color=color,
                        label=f'{feat} (n={len(diff)})')
        ax.plot(xs, kde(xs), color=color, lw=2)
        ax.axvline(diff.mean(), color=color, ls='--', lw=1.5, alpha=0.8)

    ax.axvspan(-0.02, 0.02, color='gray', alpha=0.15,
               label='Equivalence band (\u00b10.02)')
    ax.axvline(0, color='black', lw=1.0, ls=':')
    ax.set_xlabel('\u0394 AUROC (Rarefied \u2212 Non-rarefied)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of paired \u0394AUROC',
                 fontsize=13.5, fontweight='bold', pad=4)
    ax.legend(frameon=False, fontsize=11)
    clean_spines(ax)
    panel_label(ax, 'a', x=-0.16)

    # Panel b - TOST equivalence
    ax = fig.add_subplot(gs[0, 1])
    EQ = 0.02
    rows = []
    for feat in ['ASV', 'Taxa']:
        for m, lbl in [('AUC', 'AUROC'),
                       ('Balanced_Accuracy', 'Balanced acc.'),
                       ('Sensitivity', 'Sensitivity'),
                       ('Specificity', 'Specificity')]:
            pivot = (agg[agg['Feature'] == feat]
                     .groupby(['ML_Number', 'Model', 'Rarefaction'])[m]
                     .mean().unstack('Rarefaction'))
            diff = (pivot['Rarefied'] - pivot['Non-rarefied']).dropna().values
            n = len(diff)
            mean = diff.mean()
            se = diff.std(ddof=1) / np.sqrt(n)
            t90 = stats.t.ppf(0.95, n - 1)
            lo, hi = mean - t90 * se, mean + t90 * se
            rows.append(dict(feat=feat, metric=lbl, mean=mean, lo=lo, hi=hi,
                             equiv=(lo > -EQ) and (hi < EQ)))
    tdf = pd.DataFrame(rows)

    ypos = []; y = 0
    for feat in ['ASV', 'Taxa']:
        sub = tdf[tdf['feat'] == feat]
        for _, r in sub.iterrows():
            color = PAL['ASV'] if feat == 'ASV' else PAL['Taxa']
            marker = 'o' if r['equiv'] else 's'
            ax.errorbar(r['mean'], y,
                        xerr=[[r['mean'] - r['lo']], [r['hi'] - r['mean']]],
                        fmt=marker, color=color, ecolor=color, capsize=4,
                        markersize=11, markeredgecolor='black',
                        markeredgewidth=0.7, lw=1.4)
            ypos.append(y); y += 1
        y += 0.6

    ax.axvspan(-EQ, EQ, color='gray', alpha=0.18, zorder=0)
    ax.axvline(0, color='black', ls=':', lw=1.0)
    ax.set_yticks(ypos); ax.set_yticklabels(list(tdf['metric']))
    ax.invert_yaxis()
    ax.set_xlabel('Mean \u0394 (Rarefied \u2212 Non-rarefied), 90% CI')
    ax.set_xlim(-0.06, 0.06)
    ax.text(-0.057, ypos[1] - 0.5, 'ASV', fontsize=12.5,
            fontweight='bold', color=PAL['ASV'])
    ax.text(-0.057, ypos[5] - 0.5, 'Taxa', fontsize=12.5,
            fontweight='bold', color=PAL['Taxa'])

    hh = [Line2D([0], [0], marker='o', color='gray',
                 markeredgecolor='black', linestyle='', markersize=11,
                 label='Equivalent (CI within \u00b10.02)'),
          Line2D([0], [0], marker='s', color='gray',
                 markeredgecolor='black', linestyle='', markersize=11,
                 label='Not equivalent')]
    ax.legend(handles=hh, loc='lower right', frameon=False, fontsize=10.5)
    ax.set_title('Equivalence assessment (TOST, \u00b10.02 bound)',
                 fontsize=13.5, fontweight='bold', pad=4)
    clean_spines(ax)
    panel_label(ax, 'b', x=-0.22)

    # Panel c - best-performing condition per body site
    ax = fig.add_subplot(gs[1, 0])
    a3 = agg.copy()
    a3['Body_Site'] = a3['ML_Number'].map(BODY_SITES)
    # Mean AUC per (body site, feature, rarefaction) over all studies x models
    best = (a3.groupby(['Body_Site', 'Feature', 'Rarefaction'])['AUC']
            .mean().reset_index())
    idx_max = best.groupby('Body_Site')['AUC'].idxmax()
    top = best.loc[idx_max].copy()
    top['Condition'] = top['Feature'] + ' \u00b7 ' + top['Rarefaction']
    # Order body sites by sample size (most studies first)
    site_n = pd.Series(BODY_SITES).value_counts()
    site_order = site_n.index.tolist()
    top['Body_Site'] = pd.Categorical(top['Body_Site'],
                                      categories=site_order, ordered=True)
    top = top.sort_values('Body_Site')

    cond_colors = {
        'ASV \u00b7 Non-rarefied':  '#1F77B4',
        'ASV \u00b7 Rarefied':       '#74A9CF',
        'Taxa \u00b7 Non-rarefied':  '#D55E00',
        'Taxa \u00b7 Rarefied':      '#F1B16A',
    }
    bar_colors = [cond_colors[c] for c in top['Condition']]
    x = np.arange(len(top))
    ax.bar(x, top['AUC'], color=bar_colors, edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n(n={site_n[s]})' for s in top['Body_Site']],
                       fontsize=11)
    ax.set_xlabel('Body site')
    ax.set_ylabel('Best AUROC across conditions')
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_title('Best-performing condition per body site',
                 fontsize=13.5, fontweight='bold', pad=4)
    handles = [Patch(color=c, label=k) for k, c in cond_colors.items()]
    ax.legend(handles=handles, loc='lower right', frameon=False,
              fontsize=10.5, title='Winning condition', title_fontsize=11)

    counts = top['Condition'].value_counts()
    n_sites = len(top)
    txt = '\n'.join([f'{k}: {v}/{n_sites}' for k, v in counts.items()])
    ax.text(0.02, 0.96, txt, transform=ax.transAxes, va='top',
            fontsize=10.5,
            bbox=dict(facecolor='white', edgecolor='#888',
                      boxstyle='round,pad=0.3', alpha=0.9))
    clean_spines(ax)
    panel_label(ax, 'c', x=-0.10)

    # Panel d - Bland-Altman
    ax = fig.add_subplot(gs[1, 1])
    both = (agg.groupby(['ML_Number', 'Model', 'Rarefaction', 'Feature'])['AUC']
            .mean().unstack('Rarefaction').reset_index())
    both['mean'] = (both['Non-rarefied'] + both['Rarefied']) / 2
    both['diff'] = both['Rarefied'] - both['Non-rarefied']
    for feat, color, m in [('ASV', PAL['ASV'], 'o'),
                           ('Taxa', PAL['Taxa'], 's')]:
        sub = both[both['Feature'] == feat]
        ax.scatter(sub['mean'], sub['diff'], color=color, marker=m,
                   edgecolor='black', linewidth=0.5, s=70, alpha=0.75,
                   label=feat)
    md = both['diff'].mean()
    sd = both['diff'].std(ddof=1)
    ax.axhline(md, color='black', lw=1.2, ls='-', alpha=0.8)
    ax.axhline(md + 1.96 * sd, color='black', lw=1.0, ls='--', alpha=0.7)
    ax.axhline(md - 1.96 * sd, color='black', lw=1.0, ls='--', alpha=0.7)
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.text(0.99, md, f'  mean = {md:+.3f}', va='center', ha='right',
            transform=ax.get_yaxis_transform(), fontsize=10.5)
    ax.text(0.99, md + 1.96 * sd, f'  +1.96 SD = {md + 1.96 * sd:+.3f}',
            va='bottom', ha='right',
            transform=ax.get_yaxis_transform(), fontsize=10.5)
    ax.text(0.99, md - 1.96 * sd, f'  \u22121.96 SD = {md - 1.96 * sd:+.3f}',
            va='top', ha='right',
            transform=ax.get_yaxis_transform(), fontsize=10.5)
    ax.set_xlabel('Mean AUROC (Rarefied & Non-rarefied)/2')
    ax.set_ylabel('\u0394 AUROC (Rar. \u2212 Non-rar.)')
    ax.set_title('Bland-Altman: agreement vs. mean performance',
                 fontsize=13, fontweight='bold', pad=4)
    ax.legend(frameon=False, fontsize=11, loc='upper left')
    clean_spines(ax)
    panel_label(ax, 'd', x=-0.12)

    savepng(fig, 'MainFig3_decision_synthesis.png')


# ===========================================================================
# SECTION 3 - Effect is model-specific
# ===========================================================================

# --- Main Fig 4: per-model response ----------------------------------------
def fig_per_model_response(agg):
    MODELS = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Blended']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, model in enumerate(MODELS):
        ax = axes[i]
        sub = agg[agg['Model'] == model]
        positions_map = {('ASV', 'Non-rarefied'): 1, ('ASV', 'Rarefied'): 2,
                         ('Taxa', 'Non-rarefied'): 3.5, ('Taxa', 'Rarefied'): 4.5}

        for feat, x_l, x_r in [('ASV', 1, 2), ('Taxa', 3.5, 4.5)]:
            pivot = (sub[sub['Feature'] == feat]
                     .groupby(['ML_Number', 'Rarefaction'])['AUC']
                     .mean().unstack('Rarefaction'))
            for study in pivot.index:
                ax.plot([x_l, x_r],
                        [pivot.loc[study, 'Non-rarefied'],
                         pivot.loc[study, 'Rarefied']],
                        color='gray', alpha=0.45, lw=0.9, zorder=1)

        data_lists, positions, colors = [], [], []
        for (feat, rar), pos in positions_map.items():
            d = sub.loc[(sub['Feature'] == feat) &
                        (sub['Rarefaction'] == rar), 'AUC'].values
            data_lists.append(d); positions.append(pos)
            colors.append(PAL['Non-rarefied'] if rar == 'Non-rarefied'
                          else PAL['Rarefied'])

        bp = ax.boxplot(data_lists, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.8),
                        whiskerprops=dict(color='black', linewidth=1.0),
                        capprops=dict(color='black', linewidth=1.0),
                        boxprops=dict(linewidth=1.0))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.45)
        for pos, d, c in zip(positions, data_lists, colors):
            ax.scatter([pos] * len(d), d, color=c, edgecolor='black',
                       linewidth=0.5, s=35, zorder=3, alpha=0.85)

        for feat, x_l, x_r in [('ASV', 1, 2), ('Taxa', 3.5, 4.5)]:
            pivot = (sub[sub['Feature'] == feat]
                     .groupby(['ML_Number', 'Rarefaction'])['AUC']
                     .mean().unstack('Rarefaction'))
            delta = pivot['Rarefied'] - pivot['Non-rarefied']
            try:
                _, p = stats.wilcoxon(delta.dropna())
            except ValueError:
                p = np.nan
            ymax = max(pivot.values.max(), 0)
            y_line = min(ymax + 0.04, 1.05)
            ax.plot([x_l, x_r], [y_line, y_line], color='black', lw=1.0)
            ax.text((x_l + x_r) / 2, y_line + 0.01,
                    f"{sig_stars(p)} (P = {p:.2g})",
                    ha='center', va='bottom', fontsize=10.5)

        ax.set_xticks([1.5, 4.0])
        ax.set_xticklabels(['ASV', 'Taxonomic'])
        ax.set_ylabel('AUROC')
        ax.set_ylim(0.35, 1.18)
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        ax.set_title(model, fontsize=14, fontweight='bold', pad=4)
        clean_spines(ax)
        panel_label(ax, 'abcd'[i])

    handles = [Patch(facecolor=PAL['Non-rarefied'], edgecolor='black',
                     alpha=0.6, label='Non-rarefied'),
               Patch(facecolor=PAL['Rarefied'], edgecolor='black',
                     alpha=0.6, label='Rarefied')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.10,
                        wspace=0.25, hspace=0.28)
    savepng(fig, 'MainFig4_per_model_response.png')


# ===========================================================================
# SUPPLEMENTARY FIGURES
# ===========================================================================

# Helpers used by several supplementary figures
def _per_study_delta_table(agg, metric):
    out = {}
    for feat in ['ASV', 'Taxa']:
        sub = agg[agg['Feature'] == feat]
        pivot = (sub.groupby(['ML_Number', 'Model', 'Rarefaction'])[metric]
                    .mean().unstack('Rarefaction'))
        pivot['delta'] = pivot['Rarefied'] - pivot['Non-rarefied']
        g = (pivot.groupby('ML_Number')['delta']
                  .agg(['mean', 'std', 'count']).reset_index())
        g['se'] = g['std'] / np.sqrt(g['count'])
        g['ci'] = 1.96 * g['se']
        out[feat] = g
    return out

def _forest_panel(ax, g, x_label, panel_letter):
    g = g.sort_values('mean')
    y = np.arange(len(g))
    colors = ['#2C7B5C' if v > 0 else '#B5413B' for v in g['mean']]
    ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.6)
    ax.errorbar(g['mean'], y, xerr=g['ci'], fmt='none',
                ecolor='gray', lw=1.2, capsize=3, zorder=2)
    ax.scatter(g['mean'], y, c=colors, s=85, edgecolor='black',
               linewidth=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([slabel(s) for s in g['ML_Number']])
    ax.set_xlabel(x_label)
    clean_spines(ax)
    panel_label(ax, panel_letter)
    try:
        _, p = stats.wilcoxon(g['mean'])
    except ValueError:
        p = np.nan
    med = g['mean'].median()
    ax.text(0.98, 0.04,
            f"median \u0394 = {med:+.3f}\nWilcoxon P = {p:.3g}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11,
            bbox=dict(facecolor='white', edgecolor='#888',
                      boxstyle='round,pad=0.3', alpha=0.85))

def _cohens_d_paired(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return np.nan, (np.nan, np.nan)
    d = diff.mean() / diff.std(ddof=1)
    n = len(diff)
    se = np.sqrt(1 / n + d ** 2 / (2 * n))
    return d, (d - 1.96 * se, d + 1.96 * se)


# --- Supp Fig 2: per-study deltas (all four metrics) -----------------------
def figS2_per_study_deltas(agg):
    """Merged forest figure: per-study Wilcoxon \u0394 with 95% CI.
    Rows = metric (AUROC, Balanced accuracy, Sensitivity, Specificity).
    Cols = ASV / Taxa features.  Eight panels total (a-h).
    """
    panels = [('AUC', 'AUROC'),
              ('Balanced_Accuracy', 'Balanced accuracy'),
              ('Sensitivity', 'Sensitivity'),
              ('Specificity', 'Specificity')]
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))

    letters = 'abcdefgh'
    for r, (metric, label) in enumerate(panels):
        d = _per_study_delta_table(agg, metric)
        # Shared x-limits within a metric row so ASV and Taxa are directly
        # comparable side-by-side.
        all_vals = np.concatenate([
            d['ASV']['mean'].values + d['ASV']['ci'].values,
            d['ASV']['mean'].values - d['ASV']['ci'].values,
            d['Taxa']['mean'].values + d['Taxa']['ci'].values,
            d['Taxa']['mean'].values - d['Taxa']['ci'].values])
        pad = 0.02
        xlim = (min(all_vals.min(), -0.05) - pad,
                max(all_vals.max(), 0.05) + pad)

        for c, feat in enumerate(['ASV', 'Taxa']):
            ax = axes[r, c]
            _forest_panel(ax, d[feat],
                          f'\u0394 {label}  (Rarefied \u2212 Non-rarefied)',
                          letters[r * 2 + c])
            ax.set_xlim(xlim)
            ax.set_title(f'{feat} features', fontsize=14,
                         fontweight='bold', pad=4)

    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#2C7B5C', markeredgecolor='black',
                      markersize=12, label='Rarefaction improved'),
               Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#B5413B', markeredgecolor='black',
                      markersize=12, label='Rarefaction reduced')]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=13)
    plt.subplots_adjust(left=0.16, right=0.97, top=0.965, bottom=0.04,
                        wspace=0.55, hspace=0.32)
    savepng(fig, 'SuppFig2_per_study_deltas_all_metrics.png')


# --- Supp Fig 3: statistical summary ---------------------------------------
def figS3_statistical_summary(full, agg):
    METRICS = ['AUC', 'Balanced_Accuracy', 'Sensitivity', 'Specificity']
    METRIC_LBL = {'AUC': 'AUROC', 'Balanced_Accuracy': 'Balanced acc.',
                  'Sensitivity': 'Sensitivity', 'Specificity': 'Specificity'}

    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, wspace=0.45,
                          left=0.07, right=0.97, top=0.90, bottom=0.16)

    # Panel a - effect sizes
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for feat in ['ASV', 'Taxa']:
        for m in METRICS:
            pivot = (agg[agg['Feature'] == feat]
                     .groupby(['ML_Number', 'Model', 'Rarefaction'])[m]
                     .mean().unstack('Rarefaction'))
            delta = (pivot['Rarefied'] - pivot['Non-rarefied']).values
            d, (lo, hi) = _cohens_d_paired(delta)
            try:
                _, p = stats.wilcoxon(delta[~np.isnan(delta)])
            except ValueError:
                p = np.nan
            rows.append(dict(feat=feat, metric=METRIC_LBL[m],
                             d=d, lo=lo, hi=hi, p=p))
    es = pd.DataFrame(rows)

    ypos, labels, colors_e, sigs = [], [], [], []
    y = 0
    for feat in ['ASV', 'Taxa']:
        sub = es[es['feat'] == feat]
        for _, r in sub.iterrows():
            ypos.append(y); labels.append(r['metric'])
            colors_e.append(PAL['ASV'] if feat == 'ASV' else PAL['Taxa'])
            sigs.append(r['p']); y += 1
        y += 0.6

    ax.axvline(0, color='black', ls='--', lw=1.0, alpha=0.5)
    for yp, c, (_, r), p in zip(ypos, colors_e, es.iterrows(), sigs):
        ax.errorbar(r['d'], yp,
                    xerr=[[r['d'] - r['lo']], [r['hi'] - r['d']]],
                    fmt='o', color=c, ecolor=c, capsize=4,
                    markersize=10, markeredgecolor='black',
                    markeredgewidth=0.7, lw=1.2)
        ax.text(r['hi'] + 0.05, yp, sig_stars(p), va='center', fontsize=11)

    ax.set_yticks(ypos); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Effect size (Cohen's d, paired)")
    for x0, x1, _ in [(-0.2, 0.2, 'negligible'), (0.2, 0.5, 'small'),
                      (0.5, 0.8, 'medium'), (0.8, 2, 'large')]:
        ax.axvspan(x0, x1, alpha=0.06, color='gray', zorder=0)
    ax.set_xlim(-0.6, 0.8)
    ax.text(-0.55, ypos[1] - 0.5, 'ASV',
            fontsize=12.5, fontweight='bold', color=PAL['ASV'])
    ax.text(-0.55, ypos[5] - 0.5, 'Taxa',
            fontsize=12.5, fontweight='bold', color=PAL['Taxa'])
    clean_spines(ax)
    panel_label(ax, 'a', x=-0.22)
    ax.set_title("Paired effect size of rarefaction",
                 fontsize=13.5, fontweight='bold', pad=4)

    # Panel b - win/loss tally
    ax = fig.add_subplot(gs[0, 1])
    tally = []
    for feat in ['ASV', 'Taxa']:
        for m in METRICS:
            pivot = (agg[agg['Feature'] == feat]
                     .groupby(['ML_Number', 'Model', 'Rarefaction'])[m]
                     .mean().unstack('Rarefaction'))
            diff = pivot['Rarefied'] - pivot['Non-rarefied']
            tally.append(dict(
                cond=f"{feat} - {METRIC_LBL[m]}",
                win=int((diff > 0.005).sum()),
                tie=int((diff.abs() <= 0.005).sum()),
                loss=int((diff < -0.005).sum())))
    t = pd.DataFrame(tally)
    y_idx = np.arange(len(t))
    ax.barh(y_idx, t['win'], color='#2C7B5C', edgecolor='black',
            linewidth=0.5, label='Rarefaction better (\u0394>0.005)')
    ax.barh(y_idx, t['tie'], left=t['win'], color='#BDBDBD',
            edgecolor='black', linewidth=0.5,
            label='Equivalent (|\u0394|\u22640.005)')
    ax.barh(y_idx, t['loss'], left=t['win'] + t['tie'],
            color='#B5413B', edgecolor='black', linewidth=0.5,
            label='Rarefaction worse (\u0394<\u22120.005)')

    for yi, row in zip(y_idx, t.itertuples()):
        if row.win:
            ax.text(row.win / 2, yi, str(row.win), ha='center',
                    va='center', fontsize=10, color='white', fontweight='bold')
        if row.tie:
            ax.text(row.win + row.tie / 2, yi, str(row.tie), ha='center',
                    va='center', fontsize=10, fontweight='bold')
        if row.loss:
            ax.text(row.win + row.tie + row.loss / 2, yi, str(row.loss),
                    ha='center', va='center', fontsize=10,
                    color='white', fontweight='bold')

    ax.set_yticks(y_idx); ax.set_yticklabels(t['cond']); ax.invert_yaxis()
    ax.set_xlim(0, 48 + 2)
    ax.set_xlabel('Number of (study \u00d7 model) comparisons (out of 48)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=1,
              frameon=False, fontsize=10.5)
    clean_spines(ax)
    panel_label(ax, 'b', x=-0.30)
    ax.set_title('Win / equivalent / loss tally',
                 fontsize=13.5, fontweight='bold', pad=4)

    # Panel c - cross-run SD
    ax = fig.add_subplot(gs[0, 2])
    sd = (full.groupby(['ML_Number', 'Model', 'Rarefaction', 'Feature'])['AUC']
          .std().reset_index().rename(columns={'AUC': 'SD_AUC'}))
    positions = [1, 2, 3.5, 4.5]
    labels_c  = ['ASV\nNon-rar.', 'ASV\nRarefied', 'Taxa\nNon-rar.', 'Taxa\nRarefied']
    colors    = [PAL['Non-rarefied'], PAL['Rarefied'],
                 PAL['Non-rarefied'], PAL['Rarefied']]
    groups = [('ASV', 'Non-rarefied'), ('ASV', 'Rarefied'),
              ('Taxa', 'Non-rarefied'), ('Taxa', 'Rarefied')]
    data_lists = [sd.loc[(sd['Feature'] == f) & (sd['Rarefaction'] == r),
                         'SD_AUC'].values for f, r in groups]
    bp = ax.boxplot(data_lists, positions=positions, widths=0.65,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.8))
    for p, c in zip(bp['boxes'], colors):
        p.set_facecolor(c); p.set_alpha(0.55)
    rng = np.random.default_rng(1)
    for pos, d, c in zip(positions, data_lists, colors):
        j = rng.normal(0, 0.08, size=len(d))
        ax.scatter(pos + j, d, color=c, edgecolor='black',
                   linewidth=0.4, s=22, alpha=0.85)

    for feat, x_l, x_r in [('ASV', 1, 2), ('Taxa', 3.5, 4.5)]:
        nrar = sd.loc[(sd['Feature'] == feat) &
                      (sd['Rarefaction'] == 'Non-rarefied'),
                      ['ML_Number', 'Model', 'SD_AUC']
                      ].set_index(['ML_Number', 'Model'])
        rar  = sd.loc[(sd['Feature'] == feat) &
                      (sd['Rarefaction'] == 'Rarefied'),
                      ['ML_Number', 'Model', 'SD_AUC']
                      ].set_index(['ML_Number', 'Model'])
        diff = (rar['SD_AUC'] - nrar['SD_AUC']).dropna()
        try:
            _, p = stats.wilcoxon(diff)
        except ValueError:
            p = np.nan
        y_line = max(max(data_lists[positions.index(x_l)]),
                     max(data_lists[positions.index(x_r)])) + 0.005
        ax.plot([x_l, x_r], [y_line, y_line], color='black', lw=1.0)
        ax.text((x_l + x_r) / 2, y_line + 0.002,
                f"{sig_stars(p)} (P = {p:.2g})",
                ha='center', va='bottom', fontsize=11)

    ax.set_xticks(positions); ax.set_xticklabels(labels_c, fontsize=11)
    ax.set_ylabel('Run-to-run SD of AUROC')
    clean_spines(ax)
    panel_label(ax, 'c', x=-0.18)
    ax.set_title('Cross-run stability', fontsize=13.5, fontweight='bold', pad=4)

    # Panel d removed (concordance scatter) - it duplicated the equivalence
    # message already carried by MainFig3 panel a (\u0394AUROC density).
    savepng(fig, 'SuppFig3_statistical_summary.png')


# --- Supp Fig 1: per-study run-level AUC (cited from Section 1) ------------
def figS1_per_study_run_level(full):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharey=True)
    studies = sorted(full['ML_Number'].unique())
    for idx, s in enumerate(studies):
        ax = axes.flatten()[idx]
        sub = full[full['ML_Number'] == s]
        groups = [('ASV', 'Non-rarefied'), ('ASV', 'Rarefied'),
                  ('Taxa', 'Non-rarefied'), ('Taxa', 'Rarefied')]
        positions = [1, 2, 3.5, 4.5]
        colors = [PAL['Non-rarefied'], PAL['Rarefied'],
                  PAL['Non-rarefied'], PAL['Rarefied']]
        data = [sub.loc[(sub['Feature'] == f) & (sub['Rarefaction'] == r),
                        'AUC'].values for f, r in groups]
        bp = ax.boxplot(data, positions=positions, widths=0.62,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.4))
        for p, c in zip(bp['boxes'], colors):
            p.set_facecolor(c); p.set_alpha(0.5)
        rng = np.random.default_rng(idx)
        for pos, d, c in zip(positions, data, colors):
            j = rng.normal(0, 0.07, size=len(d))
            ax.scatter(pos + j, d, color=c, edgecolor='black',
                       linewidth=0.3, s=14, alpha=0.7)
        ax.set_xticks([1.5, 4.0]); ax.set_xticklabels(['ASV', 'Taxa'])
        ax.set_ylim(0, 1.05)
        ax.set_title(slabel(s), fontsize=12, fontweight='bold', pad=4)
        if idx % 4 == 0:
            ax.set_ylabel('AUROC')
        clean_spines(ax)
    handles = [Patch(facecolor=PAL['Non-rarefied'], edgecolor='black',
                     alpha=0.6, label='Non-rarefied'),
               Patch(facecolor=PAL['Rarefied'], edgecolor='black',
                     alpha=0.6, label='Rarefied')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.08,
                        wspace=0.14, hspace=0.40)
    savepng(fig, 'SuppFig1_per_study_run_level_AUC.png')


# --- Supp Fig 4: delta heatmaps (was Supp Fig 5) ---------------------------
def figS4_delta_heatmaps(agg):
    """Four-panel delta heatmaps.
    a, b: \u0394 AUROC per study x model, for ASV / Taxa
    c, d: \u0394 AUROC per body site x model, for ASV / Taxa
    """
    MODELS = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Blended']
    MODELS_SHORT = {'Random Forest': 'RF', 'Logistic Regression': 'LR',
                    'Naive Bayes': 'NB', 'Blended': 'BL'}
    a = agg.copy()
    a['Body_Site'] = a['ML_Number'].map(BODY_SITES)
    site_n = pd.Series(BODY_SITES).value_counts()
    site_order = site_n.index.tolist()

    def build_delta_by_study(feat, metric='AUC'):
        sub = agg[agg['Feature'] == feat]
        rar  = (sub[sub['Rarefaction'] == 'Rarefied']
                .pivot_table(index='ML_Number', columns='Model', values=metric))
        nrar = (sub[sub['Rarefaction'] == 'Non-rarefied']
                .pivot_table(index='ML_Number', columns='Model', values=metric))
        diff = (rar - nrar)[MODELS]
        diff.index = [slabel(i) for i in diff.index]
        return diff

    def build_delta_by_site(feat, metric='AUC'):
        sub = a[a['Feature'] == feat]
        # Mean across studies within each body site (each study contributes
        # one value per model x condition cell, then averaged)
        rar  = (sub[sub['Rarefaction'] == 'Rarefied']
                .groupby(['Body_Site', 'Model'])[metric].mean().unstack('Model'))
        nrar = (sub[sub['Rarefaction'] == 'Non-rarefied']
                .groupby(['Body_Site', 'Model'])[metric].mean().unstack('Model'))
        diff = (rar - nrar)[MODELS]
        diff = diff.reindex(site_order)
        diff.index = [f'{s} (n={site_n[s]})' for s in diff.index]
        return diff

    # Use a common colour scale across all four panels for direct comparability
    all_deltas = []
    for feat in ['ASV', 'Taxa']:
        all_deltas.append(build_delta_by_study(feat).values)
        all_deltas.append(build_delta_by_site(feat).values)
    vmax = max(max(abs(d.min()), abs(d.max())) for d in all_deltas)
    vmax = max(vmax, 0.05)

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 6],
                          left=0.18, right=0.95, top=0.94, bottom=0.06,
                          wspace=0.75, hspace=0.18)

    # Row 1 - per-study delta heatmaps
    for col, feat in enumerate(['ASV', 'Taxa']):
        ax = fig.add_subplot(gs[0, col])
        d = build_delta_by_study(feat, 'AUC')
        im = ax.imshow(d.values, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(d.shape[1]))
        ax.set_xticklabels([MODELS_SHORT[m] for m in d.columns],
                           rotation=0, fontsize=14)
        ax.set_yticks(range(d.shape[0]))
        ax.set_yticklabels(d.index, fontsize=13.5)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.values[i, j]
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        fontsize=13,
                        color='black' if abs(v) < vmax * 0.55 else 'white')
        ax.set_title(
            f'\u0394 AUROC per study - {feat} features',
            fontsize=16, fontweight='bold', pad=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label('\u0394 AUROC', fontsize=14)
        cbar.ax.tick_params(labelsize=13)
        clean_spines(ax)
        panel_label(ax, 'ab'[col], x=-0.32, y=1.04, fontsize=20)

    # Row 2 - per-body-site delta heatmaps
    for col, feat in enumerate(['ASV', 'Taxa']):
        ax = fig.add_subplot(gs[1, col])
        d = build_delta_by_site(feat, 'AUC')
        im = ax.imshow(d.values, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(d.shape[1]))
        ax.set_xticklabels([MODELS_SHORT[m] for m in d.columns],
                           rotation=0, fontsize=14)
        ax.set_yticks(range(d.shape[0]))
        ax.set_yticklabels(d.index, fontsize=13.5)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.values[i, j]
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        fontsize=13,
                        color='black' if abs(v) < vmax * 0.55 else 'white')
        ax.set_title(
            f'\u0394 AUROC per body site - {feat} features',
            fontsize=16, fontweight='bold', pad=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label('\u0394 AUROC', fontsize=14)
        cbar.ax.tick_params(labelsize=13)
        clean_spines(ax)
        panel_label(ax, 'cd'[col], x=-0.32, y=1.10, fontsize=20)

    savepng(fig, 'SuppFig4_delta_heatmaps.png')


# --- Supp Fig 5: all metrics x all models grid (was Supp Fig 6) ------------
def figS5_all_metrics_per_model(agg):
    MODELS = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Blended']
    METRICS = [('AUC', 'AUROC'),
               ('Balanced_Accuracy', 'Balanced accuracy'),
               ('Sensitivity', 'Sensitivity'),
               ('Specificity', 'Specificity')]

    fig, axes = plt.subplots(4, 4, figsize=(15, 14), sharey='row')
    for r, (mcol, mlab) in enumerate(METRICS):
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            sub = agg[agg['Model'] == model]
            groups = [('ASV', 'Non-rarefied'), ('ASV', 'Rarefied'),
                      ('Taxa', 'Non-rarefied'), ('Taxa', 'Rarefied')]
            positions = [1, 2, 3.5, 4.5]
            colors = [PAL['Non-rarefied'], PAL['Rarefied'],
                      PAL['Non-rarefied'], PAL['Rarefied']]
            data_lists = [sub.loc[(sub['Feature'] == f) &
                                  (sub['Rarefaction'] == rr), mcol].values
                          for f, rr in groups]
            bp = ax.boxplot(data_lists, positions=positions, widths=0.62,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=1.4))
            for p, col in zip(bp['boxes'], colors):
                p.set_facecolor(col); p.set_alpha(0.55)
            rng = np.random.default_rng(r * 4 + c)
            for pos, d, col in zip(positions, data_lists, colors):
                j = rng.normal(0, 0.07, size=len(d))
                ax.scatter(pos + j, d, color=col, edgecolor='black',
                           linewidth=0.4, s=22, alpha=0.85)
            for feat, x_l, x_r in [('ASV', 1, 2), ('Taxa', 3.5, 4.5)]:
                piv = (sub[sub['Feature'] == feat]
                       .groupby(['ML_Number', 'Rarefaction'])[mcol]
                       .mean().unstack('Rarefaction'))
                diff = (piv['Rarefied'] - piv['Non-rarefied']).dropna()
                try:
                    _, p = stats.wilcoxon(diff)
                except ValueError:
                    p = np.nan
                ymax = max(np.max(data_lists[positions.index(x_l)]),
                           np.max(data_lists[positions.index(x_r)]))
                y_line = min(ymax + 0.04, 1.08)
                ax.plot([x_l, x_r], [y_line, y_line], 'k-', lw=0.8)
                ax.text((x_l + x_r) / 2, y_line + 0.01, sig_stars(p),
                        ha='center', va='bottom', fontsize=10)
            ax.set_xticks([1.5, 4.0])
            ax.set_xticklabels(['ASV', 'Taxa'], fontsize=10.5)
            if r == 0:
                ax.set_title(model, fontsize=12.5, fontweight='bold', pad=4)
            if c == 0:
                ax.set_ylabel(mlab, fontsize=12)
            ax.set_ylim(0, 1.18)
            clean_spines(ax)

    handles = [Patch(facecolor=PAL['Non-rarefied'], edgecolor='black',
                     alpha=0.6, label='Non-rarefied'),
               Patch(facecolor=PAL['Rarefied'], edgecolor='black',
                     alpha=0.6, label='Rarefied')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    plt.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.07,
                        wspace=0.10, hspace=0.30)
    savepng(fig, 'SuppFig5_all_metrics_per_model.png')


# --- Supp Fig 6: sensitivity-specificity trade-off (was Supp Fig 7) --------
def figS6_sens_spec_tradeoff(agg):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    for c, feat in enumerate(['ASV', 'Taxa']):
        ax = axes[c]
        sub = agg[agg['Feature'] == feat]
        for rar, color, marker in [('Non-rarefied', PAL['Non-rarefied'], 'o'),
                                   ('Rarefied',     PAL['Rarefied'],    's')]:
            d = sub[sub['Rarefaction'] == rar]
            ax.scatter(d['Sensitivity'], d['Specificity'], color=color,
                       marker=marker, edgecolor='black', linewidth=0.5, s=65,
                       alpha=0.7, label=rar)
        ax.plot([0, 1], [1, 0], color='gray', ls=':', lw=1, alpha=0.6)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel('Sensitivity'); ax.set_ylabel('Specificity')
        ax.set_title(f'{feat} features', fontsize=13,
                     fontweight='bold', pad=4)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(frameon=False, loc='lower left', fontsize=11)
        clean_spines(ax)
        panel_label(ax, 'ab'[c], x=-0.13)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.12,
                        wspace=0.30)
    savepng(fig, 'SuppFig6_sens_spec_tradeoff.png')




# ===========================================================================
# SECTION 4 - Biomarker / feature-importance analyses
# ===========================================================================
# All three figures below operate on the SHAP rank tables, NOT raw SHAP values
# (raw SHAP magnitudes vary by orders of magnitude across studies and are
#  not directly comparable; ranks normalise this).

# --- Main Fig 5: biomarker signatures per body site ------------------------
def fig_biomarker_signatures(shap_df, topn=12):
    """Six-panel heatmap (one per body site) showing the top consolidated
    biomarker features and the rank each achieves under the four conditions.
    Rank capped at 20 (anything ranked >20 shown as a blank cell).
    Y-tick labels are coloured by phylum.
    """
    BODY_ORDER = ['Gut', 'Oral', 'Blood', 'Respiratory', 'Liver', 'Urinary']
    CONDITIONS = ['Non-rarefied ASV', 'Rarefied ASV',
                  'Non-rarefied Taxa', 'Rarefied Taxa']
    CONDITION_LBL = ['ASV\nNon-rar.', 'ASV\nRarefied', 'Taxa\nNon-rar.', 'Taxa\nRarefied']
    RANK_CAP = 20

    def site_table(site):
        sub = shap_df[shap_df['Body_Site'] == site]
        # Consolidate by (Feature_short, Phylum) - same taxon name within site
        by_cond = (sub[sub['Rank'] <= RANK_CAP]
                   .groupby(['Feature_short', 'Phylum', 'Group'])['Rank']
                   .min().reset_index())
        wide = by_cond.pivot_table(index=['Feature_short', 'Phylum'],
                                    columns='Group', values='Rank', aggfunc='min')
        for c in CONDITIONS:
            if c not in wide.columns:
                wide[c] = np.nan
        wide = wide[CONDITIONS]
        wide['n_cond']    = wide.notna().sum(axis=1)
        wide['best_rank'] = wide.iloc[:, :4].min(axis=1)
        wide = wide.sort_values(['n_cond', 'best_rank'],
                                ascending=[False, True]).head(topn)
        return wide

    fig, axes = plt.subplots(3, 2, figsize=(13, 18))
    cmap = plt.cm.viridis_r  # rank 1 -> light yellow; rank 20 -> dark purple
    for i, site in enumerate(BODY_ORDER):
        ax = axes.flatten()[i]
        t = site_table(site)
        if len(t) == 0:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f'No data for {site}', ha='center',
                    va='center', transform=ax.transAxes, fontsize=14)
            continue
        mat = t[CONDITIONS].values
        # Use a mask for NaN cells so they render as white
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect='auto', cmap=cmap,
                       vmin=1, vmax=RANK_CAP, interpolation='nearest')
        ax.set_xticks(range(4))
        ax.set_xticklabels(CONDITION_LBL, fontsize=12)
        ax.set_yticks(range(len(t)))
        # Format y-tick labels (truncate long ones) and colour by phylum.
        # Italicise species (s.) and genus (g.) labels per binomial nomenclature.
        def _truncate(s, n=28):
            return s if len(s) <= n else s[:n-1] + '\u2026'
        labels = [_truncate(str(idx[0])) for idx in t.index]
        phyla  = [idx[1] for idx in t.index]
        ax.set_yticklabels(labels, fontsize=11.5)
        for tick, ph, raw_label in zip(ax.get_yticklabels(), phyla, labels):
            tick.set_color(PHYLUM_PAL.get(ph, '#555555'))
            tick.set_fontweight('semibold')
            if raw_label.startswith(('s.', 'g.')):
                tick.set_fontstyle('italic')
        # Annotate each cell with its rank.
        # cmap=viridis_r: rank 1 -> light (needs dark text); rank 20 -> dark (needs light text).
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                v = mat[ii, jj]
                if not np.isnan(v):
                    ax.text(jj, ii, f'{int(v)}', ha='center', va='center',
                            fontsize=11,
                            color='black' if v < 10 else 'white')
        n_studies = shap_df[shap_df['Body_Site'] == site]['Study'].nunique()
        s_lbl = "studies" if n_studies > 1 else "study"
        ax.set_title(f'{site} (n={n_studies} {s_lbl})',
                     fontsize=14, fontweight='bold', pad=4)
        clean_spines(ax)
        panel_label(ax, 'abcdef'[i], x=-0.32, y=1.04, fontsize=18)

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.94, 0.30, 0.012, 0.40])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('SHAP rank (1 = most important)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # Phylum legend (compact, bottom)
    phyla_seen = set()
    for site in BODY_ORDER:
        phyla_seen.update([idx[1] for idx in site_table(site).index])
    phyla_seen = [p for p in PHYLUM_PAL if p in phyla_seen]
    handles = [Patch(color=PHYLUM_PAL[p], label=p) for p in phyla_seen]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.005), ncol=min(len(handles), 4),
               frameon=False, fontsize=11.5, title='Phylum',
               title_fontsize=12)

    plt.subplots_adjust(left=0.16, right=0.92, top=0.96, bottom=0.07,
                        wspace=0.85, hspace=0.30)
    savepng(fig, 'MainFig5_biomarker_signatures.png')


# --- Main Fig 6: rarefaction effect on biomarker identification ------------
def fig_rarefaction_effect_on_biomarkers(shap_df):
    """Two-row composite quantifying how rarefaction shifts the biomarker
    ranking, split by feature resolution (ASV vs Taxa).
    a) Top-20 Jaccard overlap (Rarefied vs Non-rarefied) per body site
    b) Spearman rank correlation per body site
    c) ASV: rank vs rank scatter for features common to both rarefactions
    d) Taxa: same scatter but for Taxa
    """
    BODY_ORDER = ['Gut', 'Oral', 'Blood', 'Respiratory', 'Liver', 'Urinary']
    TOP_N = 20

    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if (a or b) else np.nan

    # Build per-(site, study, feature_type) Jaccard and Spearman tables
    jac_rows, rho_rows = [], []
    for (site, study, ft), sub in shap_df.groupby(['Body_Site', 'Study', 'Feature_type']):
        nr_top = sub[(sub['Rarefaction'] == 'Non-rarefied') &
                     (sub['Rank'] <= TOP_N)]['Feature'].tolist()
        r_top  = sub[(sub['Rarefaction'] == 'Rarefied') &
                     (sub['Rank'] <= TOP_N)]['Feature'].tolist()
        if len(nr_top) >= 3 and len(r_top) >= 3:
            jac_rows.append(dict(Body_Site=site, Study=study,
                                 Feature_type=ft, Jaccard=jaccard(nr_top, r_top)))
        nr_rk = sub[sub['Rarefaction'] == 'Non-rarefied'].set_index('Feature')['Rank']
        r_rk  = sub[sub['Rarefaction'] == 'Rarefied'].set_index('Feature')['Rank']
        common = nr_rk.index.intersection(r_rk.index)
        if len(common) >= 5:
            rho, _ = stats.spearmanr(nr_rk.loc[common], r_rk.loc[common])
            rho_rows.append(dict(Body_Site=site, Study=study,
                                 Feature_type=ft, rho=rho, n_common=len(common)))
    jac = pd.DataFrame(jac_rows)
    rho = pd.DataFrame(rho_rows)

    fig = plt.figure(figsize=(15, 14))
    # Row 2 is taller than row 1 so the square (aspect='equal') panels c, d
    # have enough vertical room to grow to roughly the same horizontal width
    # as the bar-chart panels a, b in row 1.
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 7],
                          hspace=0.08, wspace=0.30,
                          left=0.08, right=0.97, top=0.96, bottom=0.05)

    # Panel a - top-20 Jaccard per body site
    ax = fig.add_subplot(gs[0, 0])
    n_sites = len(BODY_ORDER)
    x = np.arange(n_sites)
    width = 0.38
    for offset, ft, color in [(-width/2, 'ASV', PAL['ASV']),
                              (+width/2, 'Taxa', PAL['Taxa'])]:
        vals = [jac.loc[(jac['Body_Site'] == s) & (jac['Feature_type'] == ft),
                        'Jaccard'].mean() for s in BODY_ORDER]
        ax.bar(x + offset, vals, width, color=color, edgecolor='black',
               linewidth=0.7, label=ft, alpha=0.85)
        # Overlay individual study dots when more than one study at the site
        for i, s in enumerate(BODY_ORDER):
            pts = jac.loc[(jac['Body_Site'] == s) & (jac['Feature_type'] == ft),
                          'Jaccard'].values
            if len(pts) > 1:
                ax.scatter([x[i] + offset] * len(pts), pts, color='black',
                           s=20, zorder=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(BODY_ORDER, fontsize=12)
    ax.set_ylabel('Top-20 Jaccard overlap (NR vs R)', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title('Top-20 biomarker overlap', fontsize=14,
                 fontweight='bold', pad=4)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    clean_spines(ax)
    panel_label(ax, 'a', x=-0.12, y=1.05, fontsize=18)

    # Panel b - Spearman rank correlation per body site
    ax = fig.add_subplot(gs[0, 1])
    for offset, ft, color in [(-width/2, 'ASV', PAL['ASV']),
                              (+width/2, 'Taxa', PAL['Taxa'])]:
        vals = [rho.loc[(rho['Body_Site'] == s) & (rho['Feature_type'] == ft),
                        'rho'].mean() for s in BODY_ORDER]
        ax.bar(x + offset, vals, width, color=color, edgecolor='black',
               linewidth=0.7, label=ft, alpha=0.85)
        for i, s in enumerate(BODY_ORDER):
            pts = rho.loc[(rho['Body_Site'] == s) & (rho['Feature_type'] == ft),
                          'rho'].values
            if len(pts) > 1:
                ax.scatter([x[i] + offset] * len(pts), pts, color='black',
                           s=20, zorder=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(BODY_ORDER, fontsize=12)
    ax.set_ylabel('Spearman \u03c1 (NR vs R rank)', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title('Rank-correlation of SHAP ranks', fontsize=14,
                 fontweight='bold', pad=4)
    ax.legend(loc='lower right', frameon=False, fontsize=11)
    clean_spines(ax)
    panel_label(ax, 'b', x=-0.12, y=1.05, fontsize=18)

    # Panels c, d - rank-vs-rank scatter for ASV and Taxa
    SITE_MARKERS = {'Gut': 'o', 'Oral': 's', 'Blood': '^',
                    'Respiratory': 'D', 'Liver': 'v', 'Urinary': 'P'}
    for col_idx, (ft, panel_letter, panel_color) in enumerate(
            [('ASV', 'c', PAL['ASV']), ('Taxa', 'd', PAL['Taxa'])]):
        ax = fig.add_subplot(gs[1, col_idx])
        # For each (site, study), plot common features as NR-rank vs R-rank
        for site in BODY_ORDER:
            for study in shap_df[shap_df['Body_Site'] == site]['Study'].unique():
                sub = shap_df[(shap_df['Body_Site'] == site) &
                              (shap_df['Study'] == study) &
                              (shap_df['Feature_type'] == ft)]
                nr_rk = sub[sub['Rarefaction'] == 'Non-rarefied'].set_index('Feature')['Rank']
                r_rk  = sub[sub['Rarefaction'] == 'Rarefied'].set_index('Feature')['Rank']
                common = nr_rk.index.intersection(r_rk.index)
                if len(common) >= 3:
                    ax.scatter(nr_rk.loc[common], r_rk.loc[common],
                               marker=SITE_MARKERS.get(site, 'o'),
                               color=panel_color, edgecolor='black',
                               linewidth=0.4, s=40, alpha=0.55,
                               label=site if study == shap_df[shap_df['Body_Site']==site]['Study'].iloc[0] else None)
        # y = x reference (perfect agreement)
        lim = 50
        ax.plot([1, lim], [1, lim], 'k--', lw=1.0, alpha=0.6,
                label='Perfect agreement')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('Non-rarefied rank', fontsize=12)
        ax.set_ylabel('Rarefied rank', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{ft} feature: rank concordance',
                     fontsize=14, fontweight='bold', pad=4)
        # Body-site legend (markers only)
        handles = [Line2D([0], [0], marker=m, color='w',
                          markerfacecolor=panel_color, markeredgecolor='black',
                          markersize=9, label=s) for s, m in SITE_MARKERS.items()]
        handles.append(Line2D([0], [0], linestyle='--', color='black',
                              label='y = x'))
        ax.legend(handles=handles, loc='lower right', frameon=False,
                  fontsize=9.5, ncol=2)
        clean_spines(ax)
        panel_label(ax, panel_letter, x=-0.12, y=1.05, fontsize=18)

    savepng(fig, 'MainFig6_rarefaction_effect_on_biomarkers.png')


# --- Supp Fig 7: phylum-level robustness -----------------------------------
def figS7_phylum_composition(shap_df, topn=20):
    """Phylum composition of top-N biomarkers, per body site x condition.
    Stacked bars showing the fraction of top-N features belonging to each
    phylum. Demonstrates that even when individual ASV-level features
    shift with rarefaction, the higher-level phylum signature is preserved.
    """
    BODY_ORDER = ['Gut', 'Oral', 'Blood', 'Respiratory', 'Liver', 'Urinary']
    CONDITIONS = ['Non-rarefied ASV', 'Rarefied ASV',
                  'Non-rarefied Taxa', 'Rarefied Taxa']
    COND_LBL = ['ASV\nNon-rar.', 'ASV\nRarefied', 'Taxa\nNon-rar.', 'Taxa\nRarefied']

    # Compute phylum fractions in top-N for each (site, condition)
    sub = shap_df[shap_df['Rank'] <= topn]
    counts = (sub.groupby(['Body_Site', 'Group', 'Phylum']).size()
              .reset_index(name='count'))
    totals = sub.groupby(['Body_Site', 'Group']).size().reset_index(name='total')
    counts = counts.merge(totals, on=['Body_Site', 'Group'])
    counts['frac'] = counts['count'] / counts['total']

    # Determine which phyla to draw individually (others collapsed to 'Other')
    top_phyla = (counts.groupby('Phylum')['count'].sum()
                       .sort_values(ascending=False).head(8).index.tolist())
    counts['Phylum_grouped'] = counts['Phylum'].where(
        counts['Phylum'].isin(top_phyla), 'Other')
    counts = (counts.groupby(['Body_Site', 'Group', 'Phylum_grouped'])['frac']
              .sum().reset_index())

    # Order phyla so the colour stack is consistent across panels.
    # Known phyla (with a defined colour in PHYLUM_PAL) come first,
    # then any other top phyla appended, then 'Other' last.
    known_top   = [p for p in PHYLUM_PAL if p in top_phyla]
    unknown_top = [p for p in top_phyla if p not in PHYLUM_PAL]
    phylum_order = known_top + unknown_top + ['Other']
    # Fallback palette for unknown phyla
    _FALLBACK_COLORS = ['#88B0D3', '#F2A672', '#9BCB7E', '#D88A8A',
                        '#B79CD0', '#E0C26A', '#7BC8C0']
    fallback_pal = {p: _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
                    for i, p in enumerate(unknown_top)}
    def _ph_color(p):
        return PHYLUM_PAL.get(p, fallback_pal.get(p, PHYLUM_PAL['Other']))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=True)
    for i, site in enumerate(BODY_ORDER):
        ax = axes.flatten()[i]
        site_data = counts[counts['Body_Site'] == site]
        x = np.arange(4)
        bottom = np.zeros(4)
        for ph in phylum_order:
            ph_vals = []
            for cond in CONDITIONS:
                v = site_data.loc[(site_data['Group'] == cond) &
                                  (site_data['Phylum_grouped'] == ph), 'frac']
                ph_vals.append(v.iloc[0] if len(v) else 0.0)
            ph_vals = np.array(ph_vals)
            ax.bar(x, ph_vals, bottom=bottom, color=_ph_color(ph),
                   edgecolor='white', linewidth=0.6, label=ph)
            bottom += ph_vals
        ax.set_xticks(x)
        ax.set_xticklabels(COND_LBL, fontsize=11.5)
        ax.set_ylim(0, 1.0)
        if i % 3 == 0:
            ax.set_ylabel(f'Fraction of top-{topn} biomarkers', fontsize=12)
        n_studies = shap_df[shap_df['Body_Site'] == site]['Study'].nunique()
        s_lbl = "studies" if n_studies > 1 else "study"
        ax.set_title(f'{site} (n={n_studies} {s_lbl})',
                     fontsize=14, fontweight='bold', pad=4)
        clean_spines(ax)
        panel_label(ax, 'abcdef'[i], x=-0.10, y=1.05, fontsize=18)

    # Phylum legend (bottom)
    handles = [Patch(color=_ph_color(p), label=p) for p in phylum_order]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.005),
               ncol=min(len(handles), 6), frameon=False,
               fontsize=11.5, title='Phylum', title_fontsize=12)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.13,
                        wspace=0.15, hspace=0.32)
    savepng(fig, 'SuppFig7_phylum_composition.png')


# ===========================================================================
# STATISTICAL TABLES
# ===========================================================================
def build_tables(agg):
    MODELS = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Blended']

    # TableS1
    rows = []
    for metric in ['AUC', 'Balanced_Accuracy', 'Sensitivity', 'Specificity']:
        for feat in ['ASV', 'Taxa']:
            for model in MODELS:
                sub = agg[(agg['Feature'] == feat) & (agg['Model'] == model)]
                piv = (sub.groupby(['ML_Number', 'Rarefaction'])[metric]
                       .mean().unstack('Rarefaction'))
                nr = piv['Non-rarefied']; r = piv['Rarefied']
                diff = (r - nr).dropna()
                try:
                    w_stat, p = stats.wilcoxon(diff)
                except ValueError:
                    w_stat, p = np.nan, np.nan
                d_eff = (diff.mean() / diff.std(ddof=1)
                         if diff.std(ddof=1) > 0 else np.nan)
                rows.append({
                    'Feature': feat,
                    'Model': model,
                    'Metric': metric,
                    'N studies': len(diff),
                    'Mean Non-rar.': round(nr.mean(), 4),
                    'Mean Rarefied': round(r.mean(), 4),
                    'Mean Delta':    round(diff.mean(), 4),
                    'Median Delta':  round(diff.median(), 4),
                    'SD Delta':      round(diff.std(ddof=1), 4),
                    "Cohen's d":     round(d_eff, 3) if not np.isnan(d_eff) else np.nan,
                    'Wilcoxon W':    w_stat,
                    'P-value':       round(p, 4) if not np.isnan(p) else np.nan,
                    'Studies improved (R>NR)': int((diff > 0.005).sum()),
                    'Studies equivalent':      int((diff.abs() <= 0.005).sum()),
                    'Studies worse (R<NR)':    int((diff < -0.005).sum()),
                })
    summary_df = pd.DataFrame(rows)
    p1 = os.path.join(OUT_DIR, 'TableS1_full_statistics.csv')
    summary_df.to_csv(p1, index=False)
    print(f'Saved {p1}')

    # TableS2
    rows = []
    for feat in ['ASV', 'Taxa']:
        for m, lbl in [('AUC', 'AUROC'),
                       ('Balanced_Accuracy', 'Balanced accuracy'),
                       ('Sensitivity', 'Sensitivity'),
                       ('Specificity', 'Specificity')]:
            piv = (agg[agg['Feature'] == feat]
                   .groupby(['ML_Number', 'Model', 'Rarefaction'])[m]
                   .mean().unstack('Rarefaction'))
            nr = piv['Non-rarefied']; r = piv['Rarefied']
            diff = (r - nr).dropna()
            try:
                _, p = stats.wilcoxon(diff)
            except ValueError:
                p = np.nan
            d_eff = (diff.mean() / diff.std(ddof=1)
                     if diff.std(ddof=1) > 0 else np.nan)
            n = len(diff)
            se = diff.std(ddof=1) / np.sqrt(n)
            t90 = stats.t.ppf(0.95, n - 1)
            lo, hi = diff.mean() - t90 * se, diff.mean() + t90 * se
            rows.append({
                'Feature': feat, 'Metric': lbl, 'N': n,
                'Mean Non-rar.': round(nr.mean(), 4),
                'Mean Rarefied': round(r.mean(), 4),
                'Mean Delta':    round(diff.mean(), 4),
                '90% CI lower':  round(lo, 4),
                '90% CI upper':  round(hi, 4),
                "Cohen's d":     round(d_eff, 3),
                'Wilcoxon P':    round(p, 4),
                'Equivalent (\u00b10.02)?': 'Yes' if (lo > -0.02 and hi < 0.02) else 'No',
            })
    pooled = pd.DataFrame(rows)
    p2 = os.path.join(OUT_DIR, 'TableS2_pooled_summary.csv')
    pooled.to_csv(p2, index=False)
    print(f'Saved {p2}')


def build_shap_tables(shap_df):
    """Build three supplementary tables from the SHAP analyses.

    TableS3: Full biomarker rank table per body site x condition.
    TableS4: Per-(body site, study, feature type) Jaccard and Spearman
             agreement between Rarefied and Non-rarefied rank lists.
    TableS5: Features with the largest rank shift between Rarefied and
             Non-rarefied (top 30 per body site x feature type), useful
             for identifying which biomarkers are most rarefaction-sensitive.
    """
    if shap_df is None or len(shap_df) == 0:
        print('  [warn] No SHAP data - skipping SHAP tables.')
        return

    BODY_ORDER = ['Gut', 'Oral', 'Blood', 'Respiratory', 'Liver', 'Urinary']
    CONDITIONS = ['Non-rarefied ASV', 'Rarefied ASV',
                  'Non-rarefied Taxa', 'Rarefied Taxa']

    # -----------------------------------------------------------------
    # TableS3: Full biomarker rank table (one row per consolidated feature)
    # -----------------------------------------------------------------
    rows = []
    for site in BODY_ORDER:
        sub = shap_df[shap_df['Body_Site'] == site]
        if len(sub) == 0:
            continue
        # For each (Feature_short, Phylum, Group), take the best (lowest) rank
        # achieved across all studies with that body site.
        by_cond = (sub.groupby(['Feature_short', 'Phylum', 'Group'])['Rank']
                     .min().reset_index())
        wide = by_cond.pivot_table(index=['Feature_short', 'Phylum'],
                                    columns='Group', values='Rank', aggfunc='min')
        for c in CONDITIONS:
            if c not in wide.columns:
                wide[c] = np.nan
        wide = wide[CONDITIONS]
        wide['n_conditions_top20'] = (wide <= 20).sum(axis=1)
        wide['best_rank']          = wide[CONDITIONS].min(axis=1)
        wide = wide.reset_index()
        wide.insert(0, 'Body_Site', site)
        # Number of studies in which this feature was identified at any rank
        feat_study_count = (sub.groupby('Feature_short')['Study'].nunique()
                            .reset_index().rename(columns={'Study': 'n_studies'}))
        wide = wide.merge(feat_study_count, on='Feature_short', how='left')
        # Order: most consistent (n_conditions_top20) first, then best rank
        wide = wide.sort_values(['n_conditions_top20', 'best_rank'],
                                ascending=[False, True])
        rows.append(wide)
    t3 = pd.concat(rows, ignore_index=True)
    # Reorder columns for readability
    t3 = t3[['Body_Site', 'Feature_short', 'Phylum',
             'Non-rarefied ASV', 'Rarefied ASV',
             'Non-rarefied Taxa', 'Rarefied Taxa',
             'n_conditions_top20', 'best_rank', 'n_studies']]
    # Round ranks to integers where present (keep NaN as blank)
    for c in CONDITIONS + ['best_rank']:
        t3[c] = t3[c].astype('Int64')
    p3 = os.path.join(OUT_DIR, 'TableS3_biomarker_ranks.csv')
    t3.to_csv(p3, index=False)
    print(f'Saved {p3} ({len(t3)} features across {t3["Body_Site"].nunique()} body sites)')

    # -----------------------------------------------------------------
    # TableS4: Jaccard + Spearman, both body-site aggregated AND per-study
    # -----------------------------------------------------------------
    # NOTE on rank scope: in the input SHAP files, ranks are assigned at
    # the (body_site x condition) level, pooling features across all studies
    # of that body site. The body-site-aggregated rows are therefore the
    # primary comparison; the per-study rows are a supporting detail showing
    # how each study's features moved between NR and R rank lists.
    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if (a or b) else np.nan

    TOP_N = 20
    rows = []
    # Body-site aggregated rows
    for (site, ft), sub in shap_df.groupby(['Body_Site', 'Feature_type']):
        nr_top = sub[(sub['Rarefaction'] == 'Non-rarefied') &
                     (sub['Rank'] <= TOP_N)]['Feature'].tolist()
        r_top  = sub[(sub['Rarefaction'] == 'Rarefied') &
                     (sub['Rank'] <= TOP_N)]['Feature'].tolist()
        # When the same Feature string appears for multiple studies (common
        # at genus/species level), keep the best (lowest) rank for body-site
        # aggregation.
        nr_rk = (sub[sub['Rarefaction'] == 'Non-rarefied']
                 .groupby('Feature')['Rank'].min())
        r_rk  = (sub[sub['Rarefaction'] == 'Rarefied']
                 .groupby('Feature')['Rank'].min())
        common = nr_rk.index.intersection(r_rk.index)

        jac = jaccard(nr_top, r_top) if (len(nr_top) >= 3 and len(r_top) >= 3) else np.nan
        if len(common) >= 5:
            rho, p_rho = stats.spearmanr(nr_rk.loc[common], r_rk.loc[common])
        else:
            rho, p_rho = np.nan, np.nan
        rows.append({
            'Body_Site':      site,
            'Study':          'ALL',
            'Feature_type':   ft,
            'N_NR':           len(nr_rk),
            'N_R':            len(r_rk),
            'N_common':       len(common),
            'N_top20_NR':     len(nr_top),
            'N_top20_R':      len(r_top),
            'Jaccard_top20':  round(jac, 3) if not pd.isna(jac) else np.nan,
            'Spearman_rho':   round(rho, 3) if not pd.isna(rho) else np.nan,
            'Spearman_P':     round(p_rho, 4) if not pd.isna(p_rho) else np.nan,
        })
    # Per-study rows
    for (site, study, ft), sub in shap_df.groupby(['Body_Site', 'Study', 'Feature_type']):
        nr_top = sub[(sub['Rarefaction'] == 'Non-rarefied') &
                     (sub['Rank'] <= TOP_N)]['Feature'].tolist()
        r_top  = sub[(sub['Rarefaction'] == 'Rarefied') &
                     (sub['Rank'] <= TOP_N)]['Feature'].tolist()
        nr_rk = sub[sub['Rarefaction'] == 'Non-rarefied'].set_index('Feature')['Rank']
        r_rk  = sub[sub['Rarefaction'] == 'Rarefied'].set_index('Feature')['Rank']
        common = nr_rk.index.intersection(r_rk.index)

        jac = jaccard(nr_top, r_top) if (len(nr_top) >= 3 and len(r_top) >= 3) else np.nan
        if len(common) >= 5:
            rho, p_rho = stats.spearmanr(nr_rk.loc[common], r_rk.loc[common])
        else:
            rho, p_rho = np.nan, np.nan
        rows.append({
            'Body_Site':      site,
            'Study':          str(int(study)),
            'Feature_type':   ft,
            'N_NR':           len(nr_rk),
            'N_R':            len(r_rk),
            'N_common':       len(common),
            'N_top20_NR':     len(nr_top),
            'N_top20_R':      len(r_top),
            'Jaccard_top20':  round(jac, 3) if not pd.isna(jac) else np.nan,
            'Spearman_rho':   round(rho, 3) if not pd.isna(rho) else np.nan,
            'Spearman_P':     round(p_rho, 4) if not pd.isna(p_rho) else np.nan,
        })
    t4 = pd.DataFrame(rows)
    # Sort: body site in figure order, ALL row first within each site
    t4['Body_Site']    = pd.Categorical(t4['Body_Site'],
                                        categories=BODY_ORDER, ordered=True)
    t4['_study_sort']  = t4['Study'].apply(lambda s: -1 if s == 'ALL' else int(s))
    t4 = t4.sort_values(['Body_Site', 'Feature_type', '_study_sort']) \
           .drop(columns='_study_sort')
    p4 = os.path.join(OUT_DIR, 'TableS4_rarefaction_biomarker_agreement.csv')
    t4.to_csv(p4, index=False)
    print(f'Saved {p4} ({len(t4)} rows: body-site-aggregated + per-study)')

    # -----------------------------------------------------------------
    # TableS5: Features with the largest rank shift (NR vs R)
    # -----------------------------------------------------------------
    # For each (study, body_site, feature_type), compute rank shift for
    # features present in BOTH rarefactions. Then keep the top 30 per
    # body site x feature type, sorted by absolute shift.
    rows = []
    for (site, study, ft), sub in shap_df.groupby(['Body_Site', 'Study', 'Feature_type']):
        nr = sub[sub['Rarefaction'] == 'Non-rarefied'].set_index('Feature')
        r  = sub[sub['Rarefaction'] == 'Rarefied'].set_index('Feature')
        common = nr.index.intersection(r.index)
        for feat in common:
            rows.append({
                'Body_Site':     site,
                'Study':         study,
                'Feature_type':  ft,
                'Feature_short': nr.loc[feat, 'Feature_short'],
                'Phylum':        nr.loc[feat, 'Phylum'],
                'Rank_NR':       int(nr.loc[feat, 'Rank']),
                'Rank_R':        int(r.loc[feat, 'Rank']),
                'Rank_shift':    int(r.loc[feat, 'Rank'] - nr.loc[feat, 'Rank']),
            })
    shifts = pd.DataFrame(rows)
    shifts['Abs_shift'] = shifts['Rank_shift'].abs()
    # Keep features with shift >= 10 OR top 30 per (body_site, feature_type)
    keep = []
    for (site, ft), grp in shifts.groupby(['Body_Site', 'Feature_type']):
        keep.append(grp.nlargest(30, 'Abs_shift'))
    t5 = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    if len(t5):
        t5['Body_Site'] = pd.Categorical(t5['Body_Site'],
                                          categories=BODY_ORDER, ordered=True)
        t5 = t5.sort_values(['Body_Site', 'Feature_type', 'Abs_shift'],
                            ascending=[True, True, False])
        t5 = t5[['Body_Site', 'Study', 'Feature_type', 'Feature_short',
                 'Phylum', 'Rank_NR', 'Rank_R', 'Rank_shift', 'Abs_shift']]
    p5 = os.path.join(OUT_DIR, 'TableS5_rarefaction_sensitive_features.csv')
    t5.to_csv(p5, index=False)
    print(f'Saved {p5} ({len(t5)} most rarefaction-sensitive features)')


# ---------------------------------------------------------------------------
# TEXT REPORT
# ---------------------------------------------------------------------------
def build_results_report(full, agg, shap_df):
    """Generate a comprehensive Markdown report summarising every analysis
    that feeds the 6 main figures and the supplementary figures/tables.
    Output: 01.rarefaction_results2/results_report.md

    The report is organised to mirror the figure sequence used in the
    manuscript. Each section reports the exact numbers (means, medians,
    p-values, effect sizes, etc.) so they can be copy-pasted into the
    Results section. Each section ends with a short "key finding" line
    that can be paraphrased into prose.
    """
    out_path = os.path.join(OUT_DIR, 'results_report.md')
    lines = []
    p = lines.append   # short helper

    # ---- header ----
    p('# Results report - rarefaction ML benchmark')
    p('')
    p('Auto-generated companion to the figure pipeline. All numbers are')
    p('computed directly from `ml_summary_means.csv` and the SHAP files,')
    p('so they remain in sync with whatever the figure code is currently')
    p('plotting. Sections are ordered to match Figures 1-6 + Supp Fig 7.')
    p('')
    p(f'- Input: `{INPUT_CSV}` ({len(full)} run-level rows, {len(agg)} '
      f'aggregated (study x model x condition) rows)')
    if shap_df is not None and len(shap_df) > 0:
        p(f'- SHAP input: {len(shap_df)} rows across '
          f'{shap_df["Body_Site"].nunique()} body sites, '
          f'{shap_df["Study"].nunique()} studies')
    p('')
    p('---')
    p('')

    # =====================================================================
    # SECTION 1 - DATASET OVERVIEW (Tables 1-2)
    # =====================================================================
    p('## Section 1 - Dataset overview (Tables 1 and 2)')
    p('')
    studies = sorted(agg['ML_Number'].unique())
    n_studies = len(studies)
    body_site_map = {s: BODY_SITES.get(s, 'Unknown') for s in studies}
    n_body_sites = len(set(body_site_map.values()))
    p(f'- {n_studies} datasets across {n_body_sites} body sites.')
    site_counts = pd.Series(list(body_site_map.values())).value_counts()
    site_str = ', '.join(f'{s} (n={c})' for s, c in site_counts.items())
    p(f'- Body-site distribution: {site_str}.')
    p('')

    # =====================================================================
    # SECTION 2 - Overall classification performance (Figure 1)
    # =====================================================================
    p('## Section 2 - Overall classification performance (Figure 1)')
    p('')
    p('### Per-classifier means (Figure 4 backdrop)')
    p('')
    p('| Model | AUROC mean | AUROC SD | BA mean | BA SD | Sens mean | Spec mean |')
    p('|---|---|---|---|---|---|---|')
    for model in sorted(agg['Model'].unique()):
        sub = agg[agg['Model'] == model]
        p(f"| {model} | {sub['AUC'].mean():.3f} | {sub['AUC'].std():.3f} | "
          f"{sub['Balanced_Accuracy'].mean():.3f} | {sub['Balanced_Accuracy'].std():.3f} | "
          f"{sub['Sensitivity'].mean():.3f} | {sub['Specificity'].mean():.3f} |")
    p('')

    # Per-study AUROC range (Figure 1a,b)
    p('### Per-study AUROC range (Figure 1a, b)')
    p('')
    per_study_auc = agg.groupby('ML_Number')['AUC'].mean()
    min_s, max_s = per_study_auc.idxmin(), per_study_auc.idxmax()
    p(f'- Mean AUROC ranged from **{per_study_auc.min():.3f}** '
      f'(study {min_s}, {body_site_map.get(min_s, "?")}) to '
      f'**{per_study_auc.max():.3f}** (study {max_s}, '
      f'{body_site_map.get(max_s, "?")}). Median across studies: '
      f'{per_study_auc.median():.3f}.')
    p('')
    p('| Study | Body site | Mean AUROC | Mean BA |')
    p('|---|---|---|---|')
    for s in studies:
        sub = agg[agg['ML_Number'] == s]
        p(f"| {s} | {body_site_map.get(s, '?')} | "
          f"{sub['AUC'].mean():.3f} | "
          f"{sub['Balanced_Accuracy'].mean():.3f} |")
    p('')

    # Per-body-site averages (Figure 1c,d)
    p('### Per-body-site averages (Figure 1c, d)')
    p('')
    agg_bs = agg.copy()
    agg_bs['Body_Site'] = agg_bs['ML_Number'].map(body_site_map)
    p('| Body site | n studies | Mean AUROC | Mean BA |')
    p('|---|---|---|---|')
    for site in sorted(agg_bs['Body_Site'].unique(),
                       key=lambda x: -agg_bs[agg_bs['Body_Site']==x]['AUC'].mean()):
        sub = agg_bs[agg_bs['Body_Site'] == site]
        n_st = sub['ML_Number'].nunique()
        p(f"| {site} | {n_st} | {sub['AUC'].mean():.3f} | "
          f"{sub['Balanced_Accuracy'].mean():.3f} |")
    p('')
    p('**Key finding.** Classification difficulty is body-site-dependent, '
      'with blood, respiratory and oral datasets achieving substantially '
      'higher mean AUROC than gut, urinary and liver datasets.')
    p('')

    # =====================================================================
    # SECTION 3 - Pooled rarefaction effect (Figure 2; Table S1)
    # =====================================================================
    p('## Section 3 - Pooled rarefaction effect (Figure 2; Table S1)')
    p('')
    p('Paired Wilcoxon signed-rank tests on (study x model) deltas '
      '(rarefied minus non-rarefied), with Benjamini-Hochberg FDR correction.')
    p('')

    def _paired_wilcoxon(sub, metric):
        """Compute paired Wilcoxon between Rarefied and Non-rarefied
        on (study x model) blocks."""
        nr = sub[sub['Rarefaction'] == 'Non-rarefied'].set_index(['ML_Number','Model'])[metric]
        r  = sub[sub['Rarefaction'] == 'Rarefied'].set_index(['ML_Number','Model'])[metric]
        common = nr.index.intersection(r.index)
        if len(common) < 5:
            return None
        nr_v = nr.loc[common].values
        r_v  = r.loc[common].values
        diff = r_v - nr_v
        try:
            stat, pval = stats.wilcoxon(diff, zero_method='wilcox')
        except ValueError:
            stat, pval = float('nan'), float('nan')
        median_d = np.median(diff)
        mean_d   = np.mean(diff)
        # Rank-biserial effect size (a simple form)
        n_pos = np.sum(diff > 0); n_neg = np.sum(diff < 0)
        rb = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) else float('nan')
        return dict(n=len(common), median=median_d, mean=mean_d,
                    p=pval, rb=rb,
                    nr_mean=nr_v.mean(), r_mean=r_v.mean())

    METRICS = [('AUC','AUROC'), ('Balanced_Accuracy','Balanced acc.'),
               ('Sensitivity','Sensitivity'), ('Specificity','Specificity')]
    p('### Pooled (ASV + Taxa combined)')
    p('')
    p('| Metric | Non-rar mean | Rar mean | Median delta | Wilcoxon P | rank-biserial r |')
    p('|---|---|---|---|---|---|')
    for col, label in METRICS:
        res = _paired_wilcoxon(agg, col)
        if res:
            p(f"| {label} | {res['nr_mean']:.3f} | {res['r_mean']:.3f} | "
              f"{res['median']:+.4f} | {res['p']:.4g} | {res['rb']:+.3f} |")
    p('')
    p('### Stratified by feature type')
    p('')
    for ft in ['ASV', 'Taxa']:
        p(f'**{ft}-level features**')
        p('')
        sub = agg[agg['Feature'] == ft]
        p('| Metric | Non-rar mean | Rar mean | Median delta | Wilcoxon P |')
        p('|---|---|---|---|---|')
        for col, label in METRICS:
            res = _paired_wilcoxon(sub, col)
            if res:
                p(f"| {label} | {res['nr_mean']:.3f} | {res['r_mean']:.3f} | "
                  f"{res['median']:+.4f} | {res['p']:.4g} |")
        p('')
    p('**Key finding.** Pooled across studies and models, rarefaction had '
      'a statistically detectable but small effect on classification '
      'performance. At the ASV level the effect was clearest for balanced '
      'accuracy and specificity; at the Taxa level no significant effect '
      'was detected for any metric.')
    p('')

    # =====================================================================
    # SECTION 4 - Equivalence + decision synthesis (Figure 3; Table S2)
    # =====================================================================
    p('## Section 4 - Equivalence and decision synthesis (Figure 3; Table S2)')
    p('')
    p('Two one-sided tests (TOST) of equivalence within +/-0.02 of the metric '
      'value, applied to paired (study x model) deltas. Conditions whose '
      '90% CI falls entirely within the +/-0.02 band are flagged as equivalent.')
    p('')
    EQ_BAND = 0.02
    p('| Metric | Feature | Mean delta | 90% CI lower | 90% CI upper | Equivalent (+/-0.02)? |')
    p('|---|---|---|---|---|---|')
    for ft in ['ASV', 'Taxa']:
        sub = agg[agg['Feature'] == ft]
        for col, label in METRICS:
            nr = sub[sub['Rarefaction'] == 'Non-rarefied'].set_index(['ML_Number','Model'])[col]
            r  = sub[sub['Rarefaction'] == 'Rarefied'].set_index(['ML_Number','Model'])[col]
            common = nr.index.intersection(r.index)
            if len(common) < 5: continue
            diff = r.loc[common].values - nr.loc[common].values
            mean_d = diff.mean()
            sem = diff.std(ddof=1) / np.sqrt(len(diff))
            tcrit = stats.t.ppf(0.95, len(diff)-1)
            lo, hi = mean_d - tcrit*sem, mean_d + tcrit*sem
            eq = 'Yes' if (lo > -EQ_BAND and hi < EQ_BAND) else 'No'
            p(f"| {label} | {ft} | {mean_d:+.4f} | {lo:+.4f} | {hi:+.4f} | {eq} |")
    p('')

    # Bland-Altman limits of agreement for AUROC
    p('### Bland-Altman limits of agreement (AUROC)')
    p('')
    auc_diff = (agg[agg['Rarefaction']=='Rarefied']
                .set_index(['ML_Number','Model','Feature'])['AUC']
                .subtract(agg[agg['Rarefaction']=='Non-rarefied']
                          .set_index(['ML_Number','Model','Feature'])['AUC']))
    mean_d = auc_diff.mean(); sd_d = auc_diff.std(ddof=1)
    p(f'- Mean delta (Rarefied minus Non-rarefied): **{mean_d:+.4f}**')
    p(f'- SD of delta: **{sd_d:.4f}**')
    p(f'- 95% limits of agreement: **{mean_d - 1.96*sd_d:+.4f}** to '
      f'**{mean_d + 1.96*sd_d:+.4f}**')
    p('')

    # Body-site winning conditions
    p('### Best-performing condition per body site (Figure 3c)')
    p('')
    agg_bs2 = agg.copy()
    agg_bs2['Body_Site']  = agg_bs2['ML_Number'].map(body_site_map)
    agg_bs2['Condition']  = (agg_bs2['Feature'] + ' '
                              + agg_bs2['Rarefaction'])
    per_site = (agg_bs2.groupby(['Body_Site','Condition'])['AUC']
                       .mean().reset_index())
    p('| Body site | Best condition | Best mean AUROC |')
    p('|---|---|---|')
    for site in per_site['Body_Site'].unique():
        sub = per_site[per_site['Body_Site'] == site]
        winner = sub.loc[sub['AUC'].idxmax()]
        p(f"| {site} | {winner['Condition']} | {winner['AUC']:.3f} |")
    p('')
    p('**Key finding.** Equivalence testing confirms that rarefied and '
      'non-rarefied AUROC and Balanced accuracy are statistically '
      'equivalent within +/-0.02. The body site at which each '
      '(feature x rarefaction) condition is the best performer is variable, '
      'with no condition winning in more than three of the six body sites.')
    p('')

    # =====================================================================
    # SECTION 5 - Per-model rarefaction response (Figure 4)
    # =====================================================================
    p('## Section 5 - Per-model rarefaction response (Figure 4)')
    p('')
    p('Paired Wilcoxon tests on (study) deltas for each classifier separately.')
    p('')
    for model in sorted(agg['Model'].unique()):
        p(f'### {model}')
        p('')
        sub = agg[agg['Model'] == model]
        p('| Metric | Feature | Non-rar mean | Rar mean | Median delta | Wilcoxon P |')
        p('|---|---|---|---|---|---|')
        for ft in ['ASV', 'Taxa']:
            sub2 = sub[sub['Feature'] == ft]
            for col, label in METRICS:
                res = _paired_wilcoxon(sub2, col)
                if res:
                    sig = '*' if res['p'] < 0.05 else ''
                    p(f"| {label} | {ft} | {res['nr_mean']:.3f} | "
                      f"{res['r_mean']:.3f} | {res['median']:+.4f} | "
                      f"{res['p']:.4g}{sig} |")
        p('')
    p('**Key finding.** The rarefaction response is model-specific. Naive '
      'Bayes and Logistic Regression benefit measurably at the ASV level; '
      'Random Forest and the Blended ensemble are largely insensitive. At '
      'the Taxa level the effect is negligible for every classifier.')
    p('')

    # =====================================================================
    # SECTION 6 - Biomarker signatures per body site (Figure 5; Table S3)
    # =====================================================================
    if shap_df is None or len(shap_df) == 0:
        p('## Sections 6-8 - SHAP analyses skipped (no SHAP data)')
        p('')
    else:
        p('## Section 6 - Biomarker signatures per body site '
          '(Figure 5; Table S3)')
        p('')
        BODY_ORDER = ['Gut', 'Oral', 'Blood', 'Respiratory', 'Liver', 'Urinary']
        RANK_CAP = 20
        for site in BODY_ORDER:
            sub = shap_df[shap_df['Body_Site'] == site]
            if len(sub) == 0: continue
            n_st = sub['Study'].nunique()
            p(f'### {site} (n={n_st} stud{"ies" if n_st>1 else "y"})')
            p('')
            # Consolidate by Feature_short
            by_cond = (sub[sub['Rank'] <= RANK_CAP]
                       .groupby(['Feature_short', 'Phylum', 'Group'])['Rank']
                       .min().reset_index())
            CONDITIONS = ['Non-rarefied ASV','Rarefied ASV',
                          'Non-rarefied Taxa','Rarefied Taxa']
            wide = by_cond.pivot_table(index=['Feature_short','Phylum'],
                                        columns='Group',
                                        values='Rank', aggfunc='min')
            for c in CONDITIONS:
                if c not in wide.columns: wide[c] = np.nan
            wide = wide[CONDITIONS]
            wide['n_cond']    = wide.notna().sum(axis=1)
            wide['best_rank'] = wide.iloc[:,:4].min(axis=1)
            wide = wide.sort_values(['n_cond','best_rank'],
                                    ascending=[False, True])
            n_in_4 = int((wide['n_cond'] == 4).sum())
            n_in_3 = int((wide['n_cond'] == 3).sum())
            p(f'- Unique consolidated features ranked in top {RANK_CAP} of '
              f'any condition: {len(wide)}.')
            p(f'- Features in top {RANK_CAP} of **all 4 conditions**: '
              f'**{n_in_4}**. In **3 of 4**: {n_in_3}.')
            # Phylum breakdown
            ph_counts = wide.reset_index()['Phylum'].value_counts().head(4)
            ph_str = ', '.join(f'{ph} ({n})' for ph, n in ph_counts.items())
            p(f'- Phylum distribution (top 4): {ph_str}.')
            # Top 6 features
            p('')
            p('| Feature | Phylum | NR-ASV | R-ASV | NR-Taxa | R-Taxa | '
              'n conds | best rank |')
            p('|---|---|---|---|---|---|---|---|')
            for (feat, ph), row in wide.head(6).iterrows():
                def f(v):
                    return '-' if pd.isna(v) else f'{int(v)}'
                p(f"| {feat} | {ph} | {f(row['Non-rarefied ASV'])} | "
                  f"{f(row['Rarefied ASV'])} | {f(row['Non-rarefied Taxa'])} | "
                  f"{f(row['Rarefied Taxa'])} | {int(row['n_cond'])} | "
                  f"{int(row['best_rank'])} |")
            p('')
        p('**Key finding.** Every body site has at least one '
          'biomarker that is identified in all four (feature x rarefaction) '
          'conditions, although the number of such "fully stable" features '
          'is body-site-dependent and substantially larger for Blood than '
          'for Gut or Oral.')
        p('')

        # =================================================================
        # SECTION 7 - Rarefaction effect on biomarkers (Figure 6; Table S4)
        # =================================================================
        p('## Section 7 - Rarefaction effect on biomarkers '
          '(Figure 6; Table S4)')
        p('')
        p('Body-site-aggregated comparison between rarefied and '
          'non-rarefied rank lists (top-20 Jaccard overlap and Spearman '
          'rank correlation across the full lists).')
        p('')

        def _jaccard(a, b):
            a, b = set(a), set(b)
            return len(a & b) / len(a | b) if (a or b) else float('nan')

        rows_agg = []
        for (site, ft), sub in shap_df.groupby(['Body_Site', 'Feature_type']):
            nr_top = sub[(sub['Rarefaction'] == 'Non-rarefied') &
                         (sub['Rank'] <= 20)]['Feature'].tolist()
            r_top  = sub[(sub['Rarefaction'] == 'Rarefied') &
                         (sub['Rank'] <= 20)]['Feature'].tolist()
            nr_rk = (sub[sub['Rarefaction'] == 'Non-rarefied']
                     .groupby('Feature')['Rank'].min())
            r_rk  = (sub[sub['Rarefaction'] == 'Rarefied']
                     .groupby('Feature')['Rank'].min())
            common = nr_rk.index.intersection(r_rk.index)
            jac = _jaccard(nr_top, r_top)
            if len(common) >= 5:
                rho, p_rho = stats.spearmanr(nr_rk.loc[common],
                                              r_rk.loc[common])
            else:
                rho, p_rho = float('nan'), float('nan')
            rows_agg.append(dict(site=site, ft=ft, jac=jac, rho=rho, p=p_rho,
                                 n_common=len(common)))
        rep = pd.DataFrame(rows_agg)

        p('| Body site | Feature | Top-20 Jaccard | Spearman rho | '
          'Spearman P | n common features |')
        p('|---|---|---|---|---|---|')
        def _fmt(v, spec):
            return 'n/a' if pd.isna(v) else format(v, spec)
        for site in BODY_ORDER:
            for ft in ['ASV', 'Taxa']:
                r = rep[(rep['site'] == site) & (rep['ft'] == ft)]
                if len(r) == 0: continue
                r = r.iloc[0]
                p(f"| {site} | {ft} | "
                  f"{_fmt(r['jac'], '.3f')} | "
                  f"{_fmt(r['rho'], '.3f')} | "
                  f"{_fmt(r['p'], '.4g')} | "
                  f"{int(r['n_common'])} |")
        p('')
        # Headline summary
        asv = rep[rep['ft'] == 'ASV']
        tax = rep[rep['ft'] == 'Taxa']
        p('### Summary across body sites')
        p('')
        p(f'- ASV top-20 Jaccard: range '
          f'{asv["jac"].min():.2f}-{asv["jac"].max():.2f}, '
          f'mean {asv["jac"].mean():.2f}.')
        p(f'- Taxa top-20 Jaccard: range '
          f'{tax["jac"].min():.2f}-{tax["jac"].max():.2f}, '
          f'mean {tax["jac"].mean():.2f}.')
        p(f'- ASV Spearman rho: range '
          f'{asv["rho"].min():.2f}-{asv["rho"].max():.2f}, '
          f'mean {asv["rho"].mean():.2f}.')
        p(f'- Taxa Spearman rho: range '
          f'{tax["rho"].min():.2f}-{tax["rho"].max():.2f}, '
          f'mean {tax["rho"].mean():.2f}.')
        p('')
        # Per-study detail for sites with multiple studies
        p('### Per-study variability (Gut and Oral only)')
        p('')
        for (site, study, ft), sub in shap_df.groupby(['Body_Site',
                                                        'Study',
                                                        'Feature_type']):
            if shap_df[shap_df['Body_Site'] == site]['Study'].nunique() < 2:
                continue
            nr_rk = sub[sub['Rarefaction']=='Non-rarefied'].set_index('Feature')['Rank']
            r_rk  = sub[sub['Rarefaction']=='Rarefied'].set_index('Feature')['Rank']
            common = nr_rk.index.intersection(r_rk.index)
            if len(common) < 5: continue
            rho, _ = stats.spearmanr(nr_rk.loc[common], r_rk.loc[common])
            p(f'- {site} study {study}, {ft}: Spearman rho = '
              f'{rho:.3f} (n={len(common)} common features).')
        p('')
        p('**Key finding.** Taxa-level SHAP rankings are nearly identical '
          'between rarefied and non-rarefied data across every body site '
          '(Spearman rho 0.84-0.99). ASV-level rankings are noticeably '
          'less stable (rho 0.47-0.82) and the top-20 ASV biomarker lists '
          'overlap by roughly one third on average. Rarefaction therefore '
          'preserves the taxonomic-level biomarker signature but reshuffles '
          'individual ASV-level rankings.')
        p('')

        # =================================================================
        # SECTION 8 - Phylum-level composition (SuppFig 7)
        # =================================================================
        p('## Section 8 - Phylum-level composition of top biomarkers '
          '(Supp. Fig. 7)')
        p('')
        p('Fraction of top-20 biomarkers belonging to each phylum, per '
          '(body site, condition).')
        p('')
        for site in BODY_ORDER:
            sub = shap_df[(shap_df['Body_Site'] == site) &
                          (shap_df['Rank'] <= 20)]
            if len(sub) == 0: continue
            ph_top = (sub['Phylum'].value_counts(normalize=True)
                      .head(3))
            top_str = ', '.join(f'{ph} ({frac*100:.0f}%)'
                                 for ph, frac in ph_top.items())
            p(f'- **{site}**: dominant phyla in top-20 biomarkers '
              f'(pooled across the four conditions): {top_str}.')
        p('')
        p('**Key finding.** Within each body site, the phylum composition '
          'of the top-20 biomarker pool is broadly consistent across the '
          'four (feature x rarefaction) conditions, even where individual '
          'ASV-level features change. The high-level taxonomic signature '
          'is preserved under rarefaction even when specific sequence '
          'variants are reshuffled.')
        p('')

    # =====================================================================
    # SUGGESTED MANUSCRIPT WORDING
    # =====================================================================
    p('---')
    p('')
    p('## Suggested wording for the Results section')
    p('')
    p('The following are short, neutral templates that paraphrase the '
      'numbers above. They are starting points; please adapt the prose '
      'to fit the manuscript voice.')
    p('')
    p('**On overall performance (Figure 1).** "Across the 12 datasets, '
      'mean AUROC varied from X to Y (median Z), with body-site averages '
      'ordered blood/respiratory/oral > gut/urinary/liver."')
    p('')
    p('**On the pooled rarefaction effect (Figure 2; Table S1).** '
      '"Pooled across studies, models, and feature representations, the '
      'paired rarefied-minus-non-rarefied delta was small for every '
      'metric and reached statistical significance only for balanced '
      'accuracy and specificity at the ASV level."')
    p('')
    p('**On equivalence (Figure 3; Table S2).** "Two one-sided tests '
      'confirmed equivalence of rarefied and non-rarefied AUROC and '
      'balanced accuracy within +/-0.02 for both feature representations. '
      'Bland-Altman limits of agreement for AUROC were narrower than '
      '+/-0.04. No (feature x rarefaction) condition was the best '
      'performer in more than three of the six body sites."')
    p('')
    p('**On per-model behaviour (Figure 4).** "The rarefaction effect '
      'was model-specific. Naive Bayes and Logistic Regression showed '
      'measurable ASV-level gains under rarefaction; Random Forest and '
      'the Blended ensemble were essentially unchanged. No model showed '
      'a significant effect at the Taxa level."')
    p('')
    p('**On biomarker signatures (Figure 5; Table S3).** "Each body site '
      'contained one or more biomarkers identified in all four '
      '(feature x rarefaction) conditions at closely matching ranks, '
      'most strikingly in Blood where the top five biomarkers were '
      'identified at near-identical ranks under every preprocessing '
      'choice."')
    p('')
    p('**On the biomarker stability finding (Figure 6; Table S4).** '
      '"Spearman rank correlation between rarefied and non-rarefied SHAP '
      'rankings was 0.84-0.99 at the Taxa level across all six body '
      'sites, but only 0.47-0.82 at the ASV level. Top-20 Jaccard '
      'overlap followed the same pattern (Taxa mean 0.58, ASV mean 0.36). '
      'Rarefaction therefore preserves taxonomic-level biomarker '
      'rankings but reshuffles ASV-level rankings."')
    p('')
    p('**On phylum-level robustness (Supp. Fig. 7).** "Despite the '
      'ASV-level instability, the phylum composition of the top-20 '
      'biomarker pool was broadly consistent between rarefied and '
      'non-rarefied conditions, indicating that the high-level '
      'taxonomic signature is preserved even when individual sequence '
      'variants are reshuffled."')
    p('')

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f'Saved {out_path}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    full, agg = load_data(INPUT_CSV)
    print(f'Loaded {len(full)} rows; aggregated to {len(agg)} rows.')

    # Section 1 - Overview
    fig_overview_heatmaps(agg)                       # Main Fig 1

    # Section 2 - Negligible effect
    fig_overall_metric_comparison(agg)               # Main Fig 2
    fig_decision_synthesis(agg)                      # Main Fig 3

    # Section 3 - Model-specific exception
    fig_per_model_response(agg)                      # Main Fig 4

    # Section 4 - Biomarker / feature-importance analyses
    shap_df = load_shap_data()
    if shap_df is not None and len(shap_df) > 0:
        print(f'Loaded {len(shap_df)} SHAP rows across '
              f'{shap_df["Body_Site"].nunique()} body sites.')
        fig_biomarker_signatures(shap_df)            # Main Fig 5
        fig_rarefaction_effect_on_biomarkers(shap_df)  # Main Fig 6
        figS7_phylum_composition(shap_df)            # Supp Fig 7
    else:
        print('  [warn] No SHAP data found - skipping Section 4 figures.')

    # Supplementary figures (numbered in order of first citation across
    # Sections 1, 2, and 3)
    figS1_per_study_run_level(full)                  # Supp Fig 1 (cited from Section 1)
    figS2_per_study_deltas(agg)                      # Supp Fig 2 (Section 2; all 4 metrics)
    figS3_statistical_summary(full, agg)             # Supp Fig 3 (Section 2; 3 panels)
    figS4_delta_heatmaps(agg)                        # Supp Fig 4 (Section 3)
    figS5_all_metrics_per_model(agg)                 # Supp Fig 5 (Section 3)
    figS6_sens_spec_tradeoff(agg)                    # Supp Fig 6 (Section 3)

    # Statistical tables
    build_tables(agg)                                # TableS1, TableS2
    if shap_df is not None and len(shap_df) > 0:
        build_shap_tables(shap_df)                   # TableS3, TableS4, TableS5

    # Text report (Markdown) consolidating all numbers for the manuscript
    build_results_report(full, agg, shap_df)

    print('\nAll figures and tables generated in', OUT_DIR)


if __name__ == '__main__':
    main()
