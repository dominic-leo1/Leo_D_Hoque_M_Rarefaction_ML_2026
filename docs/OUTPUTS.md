# Outputs

Every figure and table below is written to `scripts/04_figures/output/` when the
corresponding script is run.

## Figures

| Figure | Script |
|---|---|
| Fig 1 overview heatmaps                 | performance_figures.py |
| Fig 2 overall metric comparison         | performance_figures.py |
| Fig 3 model selection                   | performance_figures.py |
| Fig 4 biomarker signatures              | biomarker_figures.py |
| Fig 5 rarefaction effect on biomarkers  | biomarker_figures.py |
| Supp Fig 1 per-study run-level AUC       | performance_figures.py |
| Supp Fig 2 per-study deltas             | performance_figures.py |
| Supp Fig 3 statistical summary          | performance_figures.py |
| Supp Fig 4 decision synthesis           | performance_figures.py |
| Supp Fig 5 delta heatmaps               | performance_figures.py |
| Supp Fig 6 all metrics per condition    | performance_figures.py |
| Supp Fig 7 sensitivity-specificity      | performance_figures.py |
| Supp Fig 8 phylum composition           | biomarker_figures.py |
| Supp Fig 9 win / equivalent / loss      | performance_figures.py |
| Supp Fig 10 top-K Jaccard sensitivity   | biomarker_figures.py |

`biomarker_figures.py` is the entry point for Fig 4, Fig 5, Supp Fig 8 and
Supp Fig 10; it draws on the figure functions in `biomarker_core.py` and
`robustness_analyses.py` and applies the shared figure styling.

## Supplementary tables

| Table | Script |
|---|---|
| Table S2 SHAP magnitude robustness      | robustness_analyses.py |
| Table S3 win / equivalent / loss tally  | performance_figures.py |
| Table S4 model selection frequency      | performance_figures.py |
| Table S5 top-K Jaccard sensitivity      | robustness_analyses.py |
| Table S6 full rarefaction statistics    | performance_figures.py |
| Table S6b pooled performance summary    | performance_figures.py |
| Table S7 biomarker ranks                | biomarker_core.py |
| Table S8 biomarker set agreement        | biomarker_core.py |
| Table S9 rarefaction-sensitive features | biomarker_core.py |

Table S1 (primer sets and target regions) is a curated table and is not script
generated. The supplementary tables are collated into a single workbook for
submission.
