#!/usr/bin/env python3
"""
biomarker_figures.py
====================
Biomarker figures: Fig 4 (signatures), Fig 5 (rarefaction overlap),
Supp Fig 8 (phylum composition) and Supp Fig 10 (top-K Jaccard sensitivity).

Draws these from the per-body-site SHAP tables (../../data/shap_bodysite_*.csv)
using the figure functions in biomarker_core.py and robustness_analyses.py,
and applies the shared figure styling (uniform panel-letter size, panel-letter
positions, tilted axis labels and legend spacing). Outputs -> output/.
"""

import re, os, importlib.util
SCALE = 1.4
CORE = os.path.join(os.path.dirname(__file__), 'biomarker_core.py')
ROB  = os.path.join(os.path.dirname(__file__), 'robustness_analyses.py')

def scale_text(s):
    b=lambda m:f"{m.group(1)}{round(float(m.group(2))*SCALE)}"
    s=re.sub(r'(fontsize=)(\d+\.?\d*)',b,s); s=re.sub(r'(labelsize=)(\d+\.?\d*)',b,s)
    for k in ['font.size','axes.labelsize','axes.titlesize','xtick.labelsize',
              'ytick.labelsize','legend.fontsize','legend.title_fontsize','figure.titlesize']:
        s=re.sub(rf"('{re.escape(k)}':\s*)(\d+\.?\d*)",
                 lambda m:f"{m.group(1)}{round(float(m.group(2))*SCALE)}",s)
    return s
def pin26(s):
    def fx(m):
        inner=re.sub(r',\s*fontsize\s*=\s*\d+\.?\d*','',m.group(1))
        return f"panel_label({inner}, fontsize=26)"
    return re.sub(r'panel_label\(([^()]*)\)',fx,s)

# Fig4: base sizing, panel letters set to 26 and pushed left
s4=open(CORE).read(); s4=pin26(s4)
s4=s4.replace("panel_label(ax, 'abcdef'[i], x=-0.32, y=1.04, fontsize=26)",
              "panel_label(ax, 'abcdef'[i], x=-0.42, y=1.04, fontsize=26)")
open('/tmp/_bio_base.py','w').write(s4)

# Fig5 + Supp Fig8
s=open(CORE).read()
for old,new in [
 ("ax.set_title('Top-20 biomarker overlap', fontsize=14,\n                 fontweight='bold', pad=4)","pass"),
 ("ax.set_title('Rank-correlation of SHAP ranks', fontsize=14,\n                 fontweight='bold', pad=4)","pass"),
 ("ax.set_title(f'{ft} feature: rank concordance',\n                     fontsize=14, fontweight='bold', pad=4)","pass"),
 ("ax.set_xticklabels(BODY_ORDER, fontsize=12)","ax.set_xticklabels(BODY_ORDER, fontsize=12, rotation=30, ha='right')"),
]:
    assert s.count(old)>=1, old[:40]; s=s.replace(old,new)
s=scale_text(s); s=pin26(s)
for a in ("'a'","'b'","panel_letter"):
    s=s.replace(f"panel_label(ax, {a}, x=-0.12, y=1.05, fontsize=26)",
                f"panel_label(ax, {a}, x=-0.22, y=1.05, fontsize=26)")
s=s.replace("panel_label(ax, 'abcdef'[i], x=-0.10, y=1.05, fontsize=26)",
            "panel_label(ax, 'abcdef'[i], x=-0.14, y=1.05, fontsize=26)")
s=s.replace("bbox_to_anchor=(0.5, 0.005),","bbox_to_anchor=(0.5, -0.02),")
s=s.replace("plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.13,\n                        wspace=0.15, hspace=0.32)",
            "plt.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.22,\n                        wspace=0.28, hspace=0.32)")
open('/tmp/_bio_scaled.py','w').write(s)

# Supp Fig10: remove titles, panel letter -> 26
rs=open(ROB).read()
rs=rs.replace("ax.set_title(f'{ft} features', fontweight='bold')","pass")
rs=scale_text(rs)  # scale body fonts first
rs=re.sub(r"(ax\.text\(-0\.06, 1\.05, chr\(ord\('a'\)\+i\), transform=ax\.transAxes,\s*fontsize=)\d+(\s*,\s*fontweight='bold'\))",r"\g<1>26\g<2>",rs)  # then pin panel letter to 26
open('/tmp/_bio_topk.py','w').write(rs)

os.makedirs('output',exist_ok=True)
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
core_base=load('bio_base','/tmp/_bio_base.py'); shap=core_base.load_shap_data()
core_base.fig_biomarker_signatures(shap)
os.replace('output/MainFig5_biomarker_signatures.png','output/MainFig4_biomarker_signatures.png')
core_scaled=load('bio_scaled','/tmp/_bio_scaled.py')
core_scaled.fig_rarefaction_effect_on_biomarkers(shap); core_scaled.figS7_phylum_composition(shap)
os.replace('output/MainFig6_rarefaction_effect_on_biomarkers.png','output/MainFig5_rarefaction_effect_on_biomarkers.png')
os.replace('output/SuppFig7_phylum_composition.png','output/SuppFig8_phylum_composition.png')
load('rev_big','/tmp/_bio_topk.py')
print('done')
