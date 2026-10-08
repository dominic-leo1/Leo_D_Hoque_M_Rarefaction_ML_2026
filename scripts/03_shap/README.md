# SHAP feature importance

This stage computes the SHAP feature-importance values that the biomarker
figures use. Its outputs, the per-body-site SHAP tables
(`shap_bodysite_*.csv`), are provided in `../../data/`, so the figure stage
runs directly.

Approach:

- SHAP values are computed for every (study x model x condition x seed)
  combination on the fixed, prevalence-filtered feature tables from stage 1
  (that is, without the per-fold univariate selection used for the
  classification benchmark, so the feature space is identical across models and
  seeds).
- Model-appropriate explainers are used: TreeExplainer for tree models,
  LinearExplainer for logistic regression, and KernelExplainer with a k-means
  background for naive Bayes and the Blended ensemble (a mean-background
  KernelExplainer fallback when the sample count is below the feature count).
- Mean absolute SHAP per feature is averaged over samples and classes, ranked
  within each cell, and aggregated to (study x model x condition x feature) by
  mean and standard deviation, then pooled per body site.
- Output: one `shap_bodysite_<site>.csv` per body site, consumed by
  `../04_figures/biomarker_figures.py` and `../04_figures/robustness_analyses.py`.
