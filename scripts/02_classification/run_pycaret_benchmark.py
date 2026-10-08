#!/usr/bin/env python
"""
run_pycaret_benchmark.py
========================
Binary disease-classification benchmark comparing rarefied and non-rarefied
16S feature tables.

For each study, feature representation (ASV / Taxa), rarefaction condition and
each of 10 random seeds:
  * PyCaret compare_models ranks a whitelist of twelve classifiers
    (lr, nb, rf, et, gbc, xgboost, lightgbm, ada, knn, lda, qda, dt) by
    Balanced accuracy and keeps the top three.
  * Each of the top three is tuned (choose_better=True, so tuning that would
    lower Balanced accuracy is discarded).
  * The three tuned models are combined into a soft-voting Blended ensemble
    (choose_better=True, so an inferior blend falls back to the best base
    learner).
  * Steps that could leak label information are confined to each cross-validation
    fold: univariate feature selection (top 20% of features), z-score
    normalisation, and SMOTE on the training folds only (sample-size-aware
    k_neighbors, skipped for near-balanced data).
  * The single best model per cell is selected by cross-validation Balanced
    accuracy.

Reported metrics are cross-validation means (mean over folds), captured from
PyCaret's results grid before any best-model selection, which avoids the
winner's-curse bias of reporting the holdout score of a model that was itself
chosen for its holdout score. Recorded metrics: AUROC, AUPRC, Sensitivity,
Specificity, Balanced accuracy and MCC (Matthews correlation coefficient).
MCC is recorded for reporting only; the selection criterion is Balanced
accuracy.

Output: a summary TSV of per-cell cross-validation metrics.
"""

from __future__ import annotations

import csv
import gc
import glob
import json
import logging
import os
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Recursion limit increase
# ---------------------------------------------------------------------
# PyCaret's internal joblib.Memory caching layer (used by setup()/
# compare_models()) hashes the dataframe and pipeline objects to build
# cache keys. On wide dataframes (many ASV/taxa columns), this hashing
# builds deeply nested tuple structures that can exceed Python's default
# recursion limit (1000), raising a RecursionError deep inside
# pycaret.internal.memory / joblib.hashing. Raising the limit here
# avoids that failure mode. Combined with memory=False in pcc.setup()
# below (which disables the caching layer entirely) for extra safety.
sys.setrecursionlimit(20000)

import matplotlib

matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

try:
    from imblearn.over_sampling import SMOTE
    _IMBLEARN_AVAILABLE = True
except ImportError:
    _IMBLEARN_AVAILABLE = False

# ---------------------------------------------------------------------
# joblib / PyCaret compatibility patch (must run BEFORE pycaret import)
# ---------------------------------------------------------------------
try:
    import joblib as _joblib
    import inspect as _inspect

    _orig_memory_init = _joblib.Memory.__init__
    _params = _inspect.signature(_orig_memory_init).parameters
    if "bytes_limit" not in _params:
        def _patched_memory_init(self, *args, **kwargs):
            bl = kwargs.pop("bytes_limit", None)
            _orig_memory_init(self, *args, **kwargs)
            try:
                self.bytes_limit = bl
            except Exception:
                pass
        _joblib.Memory.__init__ = _patched_memory_init
        try:
            _joblib.Memory.bytes_limit = None
        except Exception:
            pass

    _orig_reduce_size = _joblib.Memory.reduce_size
    def _safe_reduce_size(self, *args, **kwargs):
        if not hasattr(self, "store_backend"):
            return None
        return _orig_reduce_size(self, *args, **kwargs)
    _joblib.Memory.reduce_size = _safe_reduce_size

    from joblib.memory import MemorizedFunc as _MF
    if not hasattr(_MF, "_get_output_identifiers"):
        def _get_output_identifiers(self, *args, **kwargs):
            args_id = self._get_args_id(*args, **kwargs)
            return self.func_id, args_id
        _MF._get_output_identifiers = _get_output_identifiers
except Exception as _e:  # pragma: no cover
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "joblib.Memory patch could not be applied: %s", _e
    )

import pycaret.classification as pcc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# CONFIGURATION
# =====================================================================

# --- paths ---
INPUT_DIR  = "/home/mozammel/01.moz/07.rarefaction_ml/07.dom_phyloseq_data/02.preprocessed_files_leakfree"
OUTPUT_DIR = "/home/mozammel/01.moz/07.rarefaction_ml/07.dom_phyloseq_data/03.smote_top3_results_leakfree"

# --- training protocol ---
N_RUNS         = 10        # 10 seeds
TRAIN_FRACTION = 0.80
N_JOBS         = 8
TARGET_COL     = "type"
CONTROL_LABEL  = "control"
# Metric used for compare_models ranking, tune_model optimisation, blend
# selection, and the "single best" pick per (study x dataset x seed).
# Custom PyCaret metric name added below via pcc.add_metric().
TUNE_METRIC    = "Balanced_acc"

# Fixed CV and tuning depth (speed vs. thoroughness trade-off)
N_FOLDS = 5
N_ITER  = 5        # hyperparameter tuning iterations

# ---- Per-fold feature selection (label-aware, confined to training folds) ----
# The label-aware feature selection is done by PyCaret INSIDE each CV fold,
# so it only ever sees training-fold samples and cannot leak label
# information into the held-out folds.
#   FEATURE_SELECTION        : enable PyCaret's per-fold selection
#   FEATURE_SELECTION_METHOD : 'univariate' (SelectKBest, fast) |
#                              'classic' (SelectFromModel) | 'sequential'
#   N_FEATURES_TO_SELECT     : int (count) or float (fraction of features)
# Per-fold standardisation is done via NORMALIZE, fit on the training folds
# only.
FEATURE_SELECTION        = True
FEATURE_SELECTION_METHOD = "univariate"
N_FEATURES_TO_SELECT     = 0.20    # keep top 20% of features per fold
NORMALIZE                = True
NORMALIZE_METHOD         = "zscore"

# Whitelist of PyCaret model IDs to consider in compare_models().
# Aggressive: only fast, well-known classifiers.
MODEL_WHITELIST = [
    "lr",       # Logistic Regression
    "nb",       # Gaussian Naive Bayes
    "rf",       # Random Forest
    "et",       # Extra Trees
    "gbc",      # Gradient Boosting
    "xgboost",  # XGBoost
    "lightgbm", # LightGBM
    "ada",      # AdaBoost
    "knn",      # K Nearest Neighbours
    "lda",      # Linear Discriminant Analysis
    "qda",      # Quadratic Discriminant Analysis
    "dt",       # Decision Tree
]
TOP_N = 3      # keep the top-3 by AUROC

# --- dataset discovery ---
STUDY_FOLDER_RE = re.compile(r"^(\d+)$")
DATASET_RE = re.compile(
    r"(?P<rare>nonrarefied|rarefied)_(?P<level>asv|taxa)",
    re.IGNORECASE,
)

# --- summary file layout ---
SUMMARY_TSV        = "ml_summary_means.tsv"
TUNING_SUMMARY_CSV = "tuning_comparison_summary.csv"

# Human-readable model name normalisation for the summary TSV.
# compare_models can return many different underlying estimators, so the
# mapping is by scikit-learn class name -- see _display_model_name below.
MODEL_DISPLAY_NAME = {
    "LogisticRegression":                 "Logistic_Regression",
    "GaussianNB":                         "Naive_Bayes",
    "RandomForestClassifier":             "Random_Forest",
    "ExtraTreesClassifier":               "Extra_Trees",
    "GradientBoostingClassifier":         "Gradient_Boosting",
    "XGBClassifier":                      "XGBoost",
    "LGBMClassifier":                     "LightGBM",
    "AdaBoostClassifier":                 "AdaBoost",
    "KNeighborsClassifier":               "K_Neighbors",
    "LinearDiscriminantAnalysis":         "LDA",
    "QuadraticDiscriminantAnalysis":      "QDA",
    "DecisionTreeClassifier":             "Decision_Tree",
    "VotingClassifier":                   "Blended",
    "StackingClassifier":                 "Stacked",
    "Pipeline":                           "Pipeline",   # fallback
}

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ml_pipeline")


# =====================================================================
# DATASET DISCOVERY
# =====================================================================

def discover_datasets(input_dir: str) -> List[Tuple[int, str, str]]:
    """Discover TSV files from a flat directory.

    Expected filename pattern:
        <study_id>_<disease>_<rarefaction>_<level>.tsv
    Examples:
        1_IBD_nonrarefied_asv.tsv
        1_IBD_rarefied_taxa.tsv
        17_UTI_nonrarefied_asv.tsv

    Returns a list of (study_id, dataset_stem, file_path) tuples where
    dataset_stem is the full filename stem (without .tsv), matching the
    format used by parse_dataset_label().
    """
    out: List[Tuple[int, str, str]] = []
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {input_dir}")

    # Match: <digits>_<anything>_nonrarefied|rarefied_asv|taxa.tsv
    FILE_RE = re.compile(
        r"^(?P<study_id>\d+)_.+_(?:nonrarefied|rarefied)_(?:asv|taxa)\.tsv$",
        re.IGNORECASE,
    )

    for tsv in sorted(root.glob("*.tsv")):
        m = FILE_RE.match(tsv.name)
        if not m:
            log.debug("Skipping unrecognised file: %s", tsv.name)
            continue
        study_id = int(m.group("study_id"))
        stem     = tsv.stem   # e.g. "1_IBD_nonrarefied_asv"
        out.append((study_id, stem, str(tsv)))

    # Sort by study_id then stem so runs are deterministic
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def parse_dataset_label(stem: str) -> str:
    m = DATASET_RE.search(stem.lower())
    if not m:
        return stem
    return f"{m.group('rare')}_{m.group('level')}"


# =====================================================================
# CUSTOM METRICS
# =====================================================================

def sensitivity(y_true, y_pred, **_):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def specificity(y_true, y_pred, **_):
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def balanced_acc(y_true, y_pred, **_):
    return balanced_accuracy_score(y_true, y_pred)


def auprc(y_true, y_score, **_):
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return float("nan")


def mcc(y_true, y_pred, **_):
    """Matthews Correlation Coefficient.

    Imbalance-independent binary classification metric ranging from
    -1 (perfect inverse prediction) through 0 (random prediction) to
    +1 (perfect prediction). Recommended over accuracy/F1 for imbalanced
    data (Chicco & Jurman 2020; Boughorbel et al. 2017).
    """
    try:
        return matthews_corrcoef(y_true, y_pred)
    except Exception:
        return float("nan")


# =====================================================================
# UTILITIES
# =====================================================================

def make_label_map(df: pd.DataFrame, target_col: str) -> Tuple[Dict[str, int], str]:
    labels = (
        df[target_col].astype(str).str.strip().str.lower().unique().tolist()
    )
    if CONTROL_LABEL not in labels:
        raise ValueError(f"'{CONTROL_LABEL}' not in target. Found: {labels}")
    if len(labels) != 2:
        raise ValueError(f"Expected binary classification, found {labels}")
    case_label = next(l for l in labels if l != CONTROL_LABEL)
    return {CONTROL_LABEL: 0, case_label: 1}, case_label


def get_estimator(model):
    """Unwrap PyCaret's pipeline to expose the underlying estimator."""
    if hasattr(model, "named_steps"):
        return (
            model.named_steps.get("actual_estimator")
            or model.named_steps.get("trained_model")
            or list(model.named_steps.values())[-1]
        )
    return model


def _display_model_name(model) -> str:
    """Return a stable display name for a PyCaret-wrapped model."""
    est = get_estimator(model)
    cls = type(est).__name__
    return MODEL_DISPLAY_NAME.get(cls, cls)


def choose_smote(
    y: pd.Series, *, n_folds: int, seed: int, max_k: int = 5,
    imbalance_ratio_threshold: float = 1.5,
) -> Tuple[Optional["SMOTE"], int, str]:
    if not _IMBLEARN_AVAILABLE:
        return None, 0, "imblearn not installed"

    counts = y.value_counts()
    majority = int(counts.max())
    minority = int(counts.min())
    ratio = majority / minority if minority > 0 else float("inf")
    if ratio < imbalance_ratio_threshold:
        return None, 0, (
            f"skipped -- classes near-balanced "
            f"(majority={majority}, minority={minority}, ratio={ratio:.2f} "
            f"< threshold={imbalance_ratio_threshold})"
        )

    minority_after_holdout = int(minority * TRAIN_FRACTION)
    fold_drop_factor = (n_folds - 1) / n_folds if n_folds > 1 else 1.0
    worst_case_minority = max(1, int(np.floor(minority_after_holdout * fold_drop_factor)))
    k = min(max_k, worst_case_minority - 1)

    if k < 1:
        return None, 0, (
            f"minority too small for SMOTE "
            f"(total={minority}, est. worst-fold={worst_case_minority})"
        )

    try:
        smote = SMOTE(k_neighbors=k, random_state=seed)
    except TypeError:
        smote = SMOTE(k_neighbors=k)
    reason = (
        f"k_neighbors={k} (majority={majority}, minority={minority}, "
        f"ratio={ratio:.1f}, est. worst-fold={worst_case_minority})"
    )
    return smote, k, reason


# =====================================================================
# PLOTS / REPORTS
# =====================================================================

def save_confusion_matrix(
    y_true, y_pred, *, model_name: str, case_label: str, out_dir: str
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["control", case_label],
        yticklabels=["control", case_label],
        ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {model_name}")
    out_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.tiff")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, format="tiff")
    plt.close(fig)
    return out_path


# =====================================================================
# FEATURE IMPORTANCE (only for the single best final model per dataset)
# =====================================================================

def _builtin_importance(model, model_name: str, n_features: int) -> np.ndarray:
    est = get_estimator(model)
    try:
        if hasattr(est, "feature_importances_"):
            return est.feature_importances_
        if hasattr(est, "coef_"):
            return np.abs(est.coef_).mean(axis=0)
        if hasattr(est, "theta_"):  # GaussianNB
            return np.abs(est.theta_ - est.theta_.mean(axis=0)).mean(axis=0)
        if hasattr(est, "named_estimators_"):  # VotingClassifier
            imps = []
            for _, sub in est.named_estimators_.items():
                if hasattr(sub, "feature_importances_"):
                    imps.append(sub.feature_importances_)
                elif hasattr(sub, "coef_"):
                    imps.append(np.abs(sub.coef_).mean(axis=0))
            return np.mean(imps, axis=0) if imps else np.zeros(n_features)
    except Exception as exc:
        log.warning("built-in importance failed for %s: %s", model_name, exc)
    return np.zeros(n_features)


def _shap_importance(model, model_name: str, X_arr: np.ndarray) -> np.ndarray:
    est = get_estimator(model)
    try:
        if hasattr(est, "feature_importances_") and not hasattr(est, "named_estimators_"):
            # Tree-family: TreeExplainer is fastest.
            sv = shap.TreeExplainer(est).shap_values(X_arr)
        elif hasattr(est, "coef_"):
            sv = shap.LinearExplainer(est, X_arr).shap_values(X_arr)
        else:
            try:
                n_bg = min(10, len(X_arr))
                bg = shap.kmeans(X_arr, n_bg)
                sv = shap.KernelExplainer(model.predict_proba, bg).shap_values(
                    X_arr, nsamples=100, l1_reg="num_features(10)"
                )
            except Exception:
                bg = X_arr.mean(axis=0, keepdims=True)
                sv = shap.KernelExplainer(model.predict_proba, bg).shap_values(
                    X_arr, nsamples=100, l1_reg=0
                )

        sv = np.asarray(sv)
        if sv.ndim == 3:
            return (
                np.abs(sv).mean(axis=(0, 2))
                if sv.shape[0] == len(X_arr)
                else np.abs(sv).mean(axis=(0, 1))
            )
        if sv.ndim == 2:
            return np.abs(sv).mean(axis=0)
        return np.abs(sv).flatten()
    except Exception as exc:
        log.warning("SHAP failed for %s: %s", model_name, exc)
        return np.zeros(X_arr.shape[1])


def save_feature_importance_for_best(
    model, X_test, y_test, *, model_name: str, seed: int, out_dir: str
) -> None:
    """Compute built-in, permutation and SHAP importance, save CSV + plots.
    Called for the single best final model per (study x dataset) only."""
    os.makedirs(out_dir, exist_ok=True)
    feature_names = X_test.columns.tolist()

    results = {"builtin": _builtin_importance(model, model_name, len(feature_names))}

    try:
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=seed,
            scoring="balanced_accuracy", n_jobs=N_JOBS,
        )
        results["permutation"] = perm.importances_mean
    except Exception as exc:
        log.warning("permutation importance failed for %s: %s", model_name, exc)
        results["permutation"] = np.zeros(len(feature_names))

    results["shap"] = _shap_importance(model, model_name, X_test.values)

    df_out = pd.DataFrame(results, index=feature_names)
    df_out.index.name = "feature"
    df_out.to_csv(os.path.join(out_dir, f"{model_name}_seed{seed}_feature_importance.csv"))

    top_n = 20
    for method, vals in results.items():
        imp = pd.Series(vals, index=feature_names).nlargest(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(imp)), imp.values[::-1], color="steelblue", alpha=0.85)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels([f[:40] for f in imp.index[::-1]], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"{model_name} | seed {seed} | {method} - Top {top_n} features")
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{model_name}_seed{seed}_{method}_importance.tiff"),
            dpi=300, format="tiff",
        )
        plt.close(fig)


# =====================================================================
# HOLDOUT METRICS
# =====================================================================

def compute_holdout_metrics(
    model, *, model_name: str, confmtx_dir: str, case_label: str,
) -> Dict[str, float]:
    # Score on PyCaret's held-out TEST split explicitly. Passing the test
    # features by name avoids any ambiguity about what predict_model()
    # defaults to, and guarantees we never score on SMOTE-augmented rows
    # (SMOTE is applied only to training folds, never to X_test).
    try:
        X_test = pcc.get_config("X_test")
        y_test = pcc.get_config("y_test")
        have_holdout = X_test is not None and y_test is not None
    except Exception:
        have_holdout = False

    # raw_score=True makes PyCaret return per-class probability columns
    # (prediction_score_0 and prediction_score_1) instead of a single
    # prediction_score that reports the probability of the PREDICTED class.
    # AUROC/AUPRC require P(class = positive), so we must use the class-1
    # column, not prediction_score (which mixes classes and silently
    # corrupts AUROC while leaving label-based metrics intact).
    if have_holdout:
        preds = pcc.predict_model(model, data=X_test, raw_score=True, verbose=False)
        y_true = np.asarray(y_test).astype(int).ravel()
    else:
        preds = pcc.predict_model(model, raw_score=True, verbose=False)
        y_true = preds[TARGET_COL].to_numpy()

    y_pred = preds["prediction_label"].to_numpy()

    # Locate the positive-class probability column. With raw_score=True
    # PyCaret emits 'prediction_score_<label>' for each class label. The
    # positive class is encoded as 1 (see make_label_map). Fall back to
    # the generic 'prediction_score' only if the raw columns are absent
    # (in which case AUROC/AUPRC will be unreliable and we warn).
    pos_col = None
    for cand in ("prediction_score_1", "prediction_score_1.0",
                 "Score_1", "prediction_score_True"):
        if cand in preds.columns:
            pos_col = cand
            break
    if pos_col is not None:
        y_score = preds[pos_col].to_numpy()
    elif "prediction_score" in preds.columns:
        # Reconstruct P(class=1) from the predicted-class probability:
        # if predicted label is 1, score is P(1); if 0, P(1) = 1 - score.
        raw = preds["prediction_score"].to_numpy()
        y_score = np.where(y_pred == 1, raw, 1.0 - raw)
        log.warning(
            "  %s: raw per-class score column not found; reconstructed "
            "P(class=1) from prediction_score.", model_name,
        )
    else:
        y_score = None

    sens    = sensitivity(y_true, y_pred)
    spec    = specificity(y_true, y_pred)
    bacc    = balanced_acc(y_true, y_pred)
    mcc_val = mcc(y_true, y_pred)

    if y_score is not None:
        try:
            roc = roc_auc_score(y_true, y_score)
        except Exception:
            roc = float("nan")
        prc = auprc(y_true, y_score)
    else:
        roc, prc = float("nan"), float("nan")

    save_confusion_matrix(
        y_true, y_pred, model_name=model_name,
        case_label=case_label, out_dir=confmtx_dir,
    )

    return {
        "AUC":               round(roc,  4),
        "AUPRC":             round(prc,  4),
        "Sensitivity":       round(sens, 4),
        "Specificity":       round(spec, 4),
        "Balanced_Accuracy": round(bacc, 4),
        "MCC":               round(mcc_val, 4),
    }


# =====================================================================
# CROSS-VALIDATION METRICS (reported in the summary)
# =====================================================================

def cv_metrics_from_pull(pull_df: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
    """Extract cross-validation mean metrics from a pcc.pull() results grid.

    PyCaret's create_model/tune_model/blend_models produce a per-fold CV
    grid whose last rows are the fold Mean and SD. We report the Mean row.
    These CV metrics are averaged over the training folds and are computed
    BEFORE any 'best model' selection, so they are not inflated by the
    winner's-curse / selection-on-test bias that affects holdout metrics
    of a selected-best model on small test sets.

    Returns None if the grid has no 'Mean' row.
    """
    if pull_df is None:
        return None
    # Locate the Mean row (label can be 'Mean' in PyCaret 3.x)
    mean_label = None
    for cand in ("Mean", "mean"):
        if cand in pull_df.index:
            mean_label = cand
            break
    if mean_label is None:
        return None
    row = pull_df.loc[mean_label]

    def g(*names):
        for n in names:
            if n in row.index:
                try:
                    val = float(row[n])
                    if not np.isnan(val):
                        return val
                except Exception:
                    pass
        return float("nan")

    return {
        # AUROC: PyCaret's built-in column is "AUC"
        "AUC":               round(g("AUC"), 4),
        # AUPRC: our custom metric registered as "AUPRC"
        "AUPRC":             round(g("AUPRC"), 4),
        # Sensitivity == recall of the positive class; custom "Sensitivity"
        # falls back to built-in "Recall"
        "Sensitivity":       round(g("Sensitivity", "Recall"), 4),
        # Specificity: our custom metric
        "Specificity":       round(g("Specificity"), 4),
        # Balanced accuracy: our custom metric registered as "Balanced_acc"
        "Balanced_Accuracy": round(g("Balanced_acc", "Balanced Accuracy"), 4),
        # MCC: PyCaret built-in "MCC"
        "MCC":               round(g("MCC"), 4),
    }



def save_tuning_comparison(
    before_metrics: Optional[Dict[str, float]],
    after_metrics: Optional[Dict[str, float]],
    *,
    model_name: str, seed: int, out_dir: str,
) -> None:
    """Write a before/after-tuning comparison of CV metrics.

    before_metrics : the base model's cross-validation means (taken from
                     the compare_models grid row for this model)
    after_metrics  : the tuned/chosen model's cross-validation means

    Both are the dicts returned by cv_metrics_from_pull (or None), captured
    directly for each model so the 'before' and 'after' grids always refer to
    the intended model.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        if not before_metrics and not after_metrics:
            return
        keys = ["AUC", "AUPRC", "Sensitivity", "Specificity",
                "Balanced_Accuracy", "MCC"]
        before = before_metrics or {k: float("nan") for k in keys}
        after  = after_metrics  or {k: float("nan") for k in keys}
        rows = {
            "Before Tuning": [before.get(k, float("nan")) for k in keys],
            "After Tuning":  [after.get(k, float("nan")) for k in keys],
        }
        comp = pd.DataFrame(rows, index=keys).T
        comp.loc["Delta"] = comp.loc["After Tuning"] - comp.loc["Before Tuning"]
        comp.insert(0, "model", model_name)
        comp.insert(1, "seed", seed)
        comp.to_csv(os.path.join(out_dir, f"{model_name}_seed{seed}_tuning_comparison.csv"))
    except Exception as exc:
        log.warning("tuning-comparison save failed for %s: %s", model_name, exc)


# =====================================================================
# RUN ONE (study_id, file_path, seed)
# =====================================================================

def run_one(
    study_id: int, dataset_stem: str, file_path: str, seed: int,
    summary_rows: list,
) -> None:
    """Execute one (dataset, seed) iteration.

    Pipeline:
      1. compare_models on MODEL_WHITELIST, ranked by Balanced accuracy -> top 3
      2. Each top-3 model tuned with choose_better=True (Balanced_acc)
      3. Blended (soft voting) of the 3 tuned models, also choose_better=True
      4. Score each of {top3 tuned models, blended} on the holdout set
      5. Pick the single best by holdout Balanced accuracy -> compute feature importance
      6. Append one summary row per model to summary_rows (incl. MCC)
    """
    base_run_dir = Path(OUTPUT_DIR) / dataset_stem / f"run_{seed}"
    results_dir  = base_run_dir / "results"
    tuning_dir   = results_dir / "tuning"
    models_dir   = base_run_dir / "models"
    plots_dir    = base_run_dir / "plots"
    confmtx_dir  = plots_dir / "confusion_matrix"
    featimp_dir  = plots_dir / "feature_importance"
    for d in [results_dir, tuning_dir, models_dir, confmtx_dir, featimp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    data = pd.read_csv(file_path, sep="\t")
    data.columns = (
        data.columns.str.replace(" ", "_", regex=False)
                    .str.replace("-", "_", regex=False)
    )

    # ---- de-duplicate column names ----
    # PyCaret's df_shrink_dtypes() crashes on duplicate columns due to a
    # pandas __finalize__ bug (df[c] returns a DataFrame instead of a
    # Series when c is duplicated). This can happen after the R
    # preprocessing collapses distinct taxonomy strings that differ only
    # in punctuation into the same column name. Rename duplicates with a
    # suffix before PyCaret ever sees the dataframe.
    if data.columns.duplicated().any():
        dup_cols = data.columns[data.columns.duplicated()].unique().tolist()
        log.warning(
            "  Found %d duplicate column name(s) in %s: %s -- renaming with "
            "suffixes to make unique.",
            len(dup_cols), file_path, dup_cols,
        )
        new_cols = []
        seen: Dict[str, int] = {}
        for col in data.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_dup{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        data.columns = new_cols

    data[TARGET_COL] = data[TARGET_COL].astype(str).str.strip().str.lower()
    label_map, case_label = make_label_map(data, TARGET_COL)
    data[TARGET_COL] = data[TARGET_COL].map(label_map)
    n_samples = len(data)
    n_features = data.shape[1] - 1   # exclude the target column

    # ---- Adaptive feature selection ----
    # On low-dimensional tables (e.g. Taxa-level, which can have only a few
    # dozen features after collapsing + prevalence filtering), selecting a
    # 20% fraction can round to a tiny number or conflict with PyCaret's
    # feature-selection step and crash setup(). Disable feature selection
    # when there are few features; otherwise keep a fraction but never fewer
    # than MIN_FEATURES_KEPT and never more than the available features.
    MIN_FEATURES_KEPT = 10
    FS_DISABLE_BELOW  = 25   # if <= this many features, skip FS entirely
    if not FEATURE_SELECTION or n_features <= FS_DISABLE_BELOW:
        fs_flag = False
        fs_n = None
    else:
        fs_flag = True
        if isinstance(N_FEATURES_TO_SELECT, float):
            k = int(round(N_FEATURES_TO_SELECT * n_features))
        else:
            k = int(N_FEATURES_TO_SELECT)
        k = max(MIN_FEATURES_KEPT, min(k, n_features - 1))
        fs_n = k

    # ---- SMOTE configuration ----
    smote_obj, smote_k, smote_reason = choose_smote(
        data[TARGET_COL], n_folds=N_FOLDS, seed=seed
    )
    fix_imbalance_flag   = smote_obj is not None
    fix_imbalance_method = smote_obj if smote_obj is not None else None

    log.info(
        "  Setup: n=%d, features=%d, folds=%d, n_iter=%d, "
        "feature_selection=%s (keep=%s), SMOTE=%s (%s)",
        n_samples, n_features, N_FOLDS, N_ITER,
        fs_flag, fs_n if fs_flag else "-",
        "True" if fix_imbalance_flag else "DISABLED", smote_reason,
    )

    # ---- PyCaret setup ----
    setup_kwargs = dict(
        data=data, target=TARGET_COL,
        session_id=seed,
        train_size=TRAIN_FRACTION,
        fold_strategy="stratifiedkfold",
        fold=N_FOLDS,
        fix_imbalance=fix_imbalance_flag,
        # Label-free steps already done in R (rarefaction, taxa collapse,
        # prevalence filter). The following are done PER-FOLD by PyCaret so
        # they never see held-out samples:
        #   - normalize: per-fold z-score standardisation (replaces the
        #     center/scale that used to run on the full dataset in R)
        #   - feature_selection: per-fold label-aware selection (replaces
        #     the R Venn/zv-nzv step that leaked labels). Adaptive: skipped
        #     on low-dimensional tables (see above).
        preprocess=True,
        imputation_type=None,
        normalize=NORMALIZE,
        normalize_method=NORMALIZE_METHOD,
        transformation=False,
        pca=False,
        low_variance_threshold=None,
        remove_multicollinearity=False,
        remove_outliers=False,
        feature_selection=fs_flag,
        polynomial_features=False,
        bin_numeric_features=None,
        n_jobs=N_JOBS,
        verbose=False,
        # Disable PyCaret's internal joblib.Memory caching. On wide
        # dataframes (many ASV/taxa columns) the caching layer's hashing
        # step can trigger a RecursionError deep inside
        # pycaret.internal.memory / joblib.hashing. We don't need
        # result caching in a single-pass batch pipeline, so turning it
        # off avoids the failure mode entirely.
        memory=False,
    )
    if fix_imbalance_method is not None:
        setup_kwargs["fix_imbalance_method"] = fix_imbalance_method
    if fs_flag:
        setup_kwargs["feature_selection_method"] = FEATURE_SELECTION_METHOD
        setup_kwargs["n_features_to_select"] = fs_n

    pcc.setup(**setup_kwargs)

    # Register custom metrics. These persist in PyCaret's global state
    # across setup() calls, so on seed 2+ they already exist. Silently
    # skip if already registered.
    for mid, mname, mfunc, mkw in [
        ("sensitivity",  "Sensitivity",  sensitivity, {}),
        ("specificity",  "Specificity",  specificity, {}),
        ("balanced_acc", "Balanced_acc", balanced_acc, {}),
        ("auprc",        "AUPRC",        auprc,       {"target": "pred_proba", "greater_is_better": True}),
        ("mcc",          "MCC",          mcc,         {}),
    ]:
        try:
            pcc.add_metric(mid, mname, mfunc, **mkw)
        except ValueError:
            pass  # already registered from a previous iteration

    # =================================================================
    # 1. compare_models -> top 3 by Balanced accuracy
    # =================================================================
    log.info("  compare_models on %d candidates", len(MODEL_WHITELIST))
    top_models = pcc.compare_models(
        include=MODEL_WHITELIST,
        sort=TUNE_METRIC,
        n_select=TOP_N,
        fold=N_FOLDS,
        verbose=False,
        errors="ignore",   # skip any estimator that errors out mid-CV
    )
    if not isinstance(top_models, list):
        top_models = [top_models]

    # Capture the compare_models grid. Its rows are the candidates' CV
    # mean metrics in ranked order, so row i corresponds to top_models[i].
    # This is the correct 'before tuning' CV baseline (the previous code
    # snapshotted a stale pull() grid, which could belong to a different
    # model).
    compare_grid = pcc.pull()

    top_names = [_display_model_name(m) for m in top_models]
    log.info("  Top %d by Balanced accuracy: %s", TOP_N, top_names)

    NAN_METRICS = {k: float("nan") for k in
                   ["AUC", "AUPRC", "Sensitivity", "Specificity",
                    "Balanced_Accuracy", "MCC"]}

    def _cv_metrics_with_fallback(grid: Optional[pd.DataFrame],
                                  model_obj: Any, label: str) -> Dict[str, float]:
        """Extract CV means from a pull() grid; if the grid has no usable
        Mean row (can happen with choose_better keeping the base model),
        re-evaluate the exact model object with cross-validation via
        create_model to obtain an authoritative CV grid."""
        cvm = cv_metrics_from_pull(grid)
        if cvm is not None and not all(
            (v is None or (isinstance(v, float) and np.isnan(v))) for v in cvm.values()
        ):
            return cvm
        try:
            _ = pcc.create_model(model_obj, fold=N_FOLDS, verbose=False)
            cvm2 = cv_metrics_from_pull(pcc.pull())
            if cvm2 is not None:
                return cvm2
        except Exception as exc:
            log.warning("  CV fallback (create_model) failed for %s: %s", label, exc)
        log.warning("  CV grid unavailable for %s; recording NaNs.", label)
        return dict(NAN_METRICS)

    def _compare_row_metrics(i: int) -> Optional[Dict[str, float]]:
        """CV means for the i-th ranked candidate from the compare grid."""
        try:
            if compare_grid is None or i >= len(compare_grid):
                return None
            row = compare_grid.iloc[i]

            def g(*names):
                for n in names:
                    if n in row.index:
                        try:
                            v = float(row[n])
                            if not np.isnan(v):
                                return v
                        except Exception:
                            pass
                return float("nan")

            return {
                "AUC":               round(g("AUC"), 4),
                "AUPRC":             round(g("AUPRC"), 4),
                "Sensitivity":       round(g("Sensitivity", "Recall"), 4),
                "Specificity":       round(g("Specificity"), 4),
                "Balanced_Accuracy": round(g("Balanced_acc", "Balanced Accuracy"), 4),
                "MCC":               round(g("MCC"), 4),
            }
        except Exception:
            return None

    # =================================================================
    # 2. Tune each top-3 model with choose_better=True. Report the tuned
    #    model's CV means; use the compare-grid row as the honest
    #    'before tuning' baseline for the diagnostic comparison file.
    # =================================================================
    tuned_models: List[Tuple[str, Any, Dict[str, float]]] = []
    for i, (base_model, name) in enumerate(zip(top_models, top_names)):
        try:
            tuned = pcc.tune_model(
                base_model,
                fold=N_FOLDS,
                optimize=TUNE_METRIC,
                n_iter=N_ITER,
                choose_better=True,     # discard tuning if it degrades Balanced acc
                verbose=False,
            )
            tuned_grid = pcc.pull()
            cvm = _cv_metrics_with_fallback(tuned_grid, tuned, name)
            save_tuning_comparison(
                _compare_row_metrics(i), cvm,
                model_name=name, seed=seed, out_dir=str(tuning_dir),
            )
            tuned_models.append((name, tuned, cvm))
        except Exception as exc:
            log.warning("  tune_model failed for %s: %s -- using untuned base", name, exc)
            cvm = _cv_metrics_with_fallback(None, base_model, name)
            tuned_models.append((name, base_model, cvm))

    # =================================================================
    # 3. Blended (soft voting) of the three tuned models, choose_better=True.
    #    Capture its CV metrics from pull() too.
    # =================================================================
    blended_final: Optional[Any] = None
    blended_cvm: Optional[Dict[str, float]] = None
    try:
        blender = pcc.blend_models(
            [m for _, m, _ in tuned_models],
            fold=N_FOLDS,
            optimize=TUNE_METRIC,
            choose_better=True,
            method="soft",
            verbose=False,
        )
        blended_final = blender
        blended_cvm = _cv_metrics_with_fallback(pcc.pull(), blender, "Blended")
    except Exception as exc:
        log.warning("  blend_models failed: %s -- Blended will be omitted", exc)

    # =================================================================
    # 4. Assemble final models with their CROSS-VALIDATION metrics.
    #    (Holdout confusion matrices are still drawn for reference, but
    #    the reported/summary metrics are the CV means captured above.)
    # =================================================================
    final_models: List[Tuple[str, Any, Dict[str, float]]] = []
    for name, model, cvm in tuned_models:
        # confusion matrix on the real (non-augmented) holdout, for the plot only
        try:
            _ = compute_holdout_metrics(
                model, model_name=f"{name.lower()}_run{seed}",
                confmtx_dir=str(confmtx_dir), case_label=case_label,
            )
        except Exception as exc:
            log.warning("  holdout CM failed for %s: %s", name, exc)
        final_models.append((name, model, cvm))
    if blended_final is not None:
        try:
            _ = compute_holdout_metrics(
                blended_final, model_name=f"blended_run{seed}",
                confmtx_dir=str(confmtx_dir), case_label=case_label,
            )
        except Exception as exc:
            log.warning("  holdout CM failed for Blended: %s", exc)
        final_models.append((
            "Blended", blended_final,
            blended_cvm if blended_cvm is not None else dict(NAN_METRICS),
        ))

    # =================================================================
    # 5. Pick the single best by CROSS-VALIDATION Balanced accuracy.
    #    Selection is on the same CV quantity that is reported, and since
    #    CV is averaged over folds (not a single small holdout), it is a
    #    far less biased basis for both selecting and reporting.
    # =================================================================
    best_name = None
    if final_models:
        def _key(item):
            v = item[2]["Balanced_Accuracy"]
            return v if not (v is None or (isinstance(v, float) and np.isnan(v))) else -1
        best_name, best_model, best_metrics = max(final_models, key=_key)
        log.info(
            "  BEST for study=%d dataset=%s run=%d -> %s "
            "(CV Balanced_Accuracy=%.3f, CV AUROC=%.3f, CV MCC=%.3f)",
            study_id, dataset_stem, seed, best_name,
            best_metrics["Balanced_Accuracy"], best_metrics["AUC"],
            best_metrics["MCC"],
        )

        # Feature importance / SHAP for the best model is intentionally
        # DISABLED in this pipeline. With per-fold feature selection
        # (feature_selection=True), the selected feature set differs across
        # folds and seeds, so a single "best model" SHAP ranking is neither
        # dimensionally consistent (the estimator sees the reduced feature
        # space while X_test is full-width, causing shape-mismatch errors)
        # nor scientifically aggregatable across seeds. Biomarker / SHAP
        # analysis is performed separately on a fixed feature set. We still
        # save the best-model pickle below for reference.

        # Save the best-model pickle for downstream inspection.
        try:
            pcc.save_model(
                best_model,
                os.path.join(str(models_dir), f"BEST_{best_name}_seed{seed}"),
                verbose=False,
            )
        except Exception as exc:
            log.warning("save_model failed for BEST (%s): %s", best_name, exc)

    # =================================================================
    # 6. One tidy row per final model into the summary TSV. Metrics are
    #    CROSS-VALIDATION means (see cv_metrics_from_pull), not selected-
    #    best holdout scores.
    # =================================================================
    for name, model, metrics in final_models:
        summary_rows.append({
            "ML_Number":         study_id,
            "Run":               seed,
            "Model":             name,
            "Is_Best":           (name == best_name),
            "AUC":               metrics["AUC"],
            "AUPRC":             metrics["AUPRC"],
            "Sensitivity":       metrics["Sensitivity"],
            "Specificity":       metrics["Specificity"],
            "Balanced_Accuracy": metrics["Balanced_Accuracy"],
            "MCC":               metrics["MCC"],
            "Dataset":           parse_dataset_label(dataset_stem),
        })

    plt.close("all")
    gc.collect()


# =====================================================================
# RESUME / AGGREGATION
# =====================================================================

def load_completed_runs(summary_path: str) -> set:
    """A (study, dataset, run) is complete if there is at least one row
    with Is_Best=True for it in the existing summary."""
    path = Path(summary_path)
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(summary_path, sep="\t")
    except Exception as exc:
        log.warning("Could not read existing summary TSV: %s -- fresh start.", exc)
        return set()

    required = {"ML_Number", "Dataset", "Run", "Is_Best"}
    if not required.issubset(df.columns):
        log.warning("Summary TSV missing columns %s -- fresh start.", required - set(df.columns))
        return set()

    completed: set = set()
    for (ml_num, dataset, run), grp in df.groupby(["ML_Number", "Dataset", "Run"]):
        if bool(grp["Is_Best"].any()):
            completed.add((int(ml_num), str(dataset), int(run)))
    return completed


def aggregate_tuning_comparisons(base_dir: str, out_path: str) -> None:
    files = glob.glob(
        os.path.join(base_dir, "**", "*tuning_comparison*.csv"), recursive=True
    )
    if not files:
        log.warning("No tuning comparison files under %s", base_dir)
        return
    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        parts = Path(f).parts
        try:
            ds = parts[parts.index(os.path.basename(base_dir.rstrip("/"))) + 1]
        except (ValueError, IndexError):
            ds = "unknown"
        df.insert(0, "dataset", ds)
        dfs.append(df)
    pd.concat(dfs).to_csv(out_path)
    log.info("Saved tuning summary: %s", out_path)


def write_summary_tsv(rows: list, out_path: str) -> None:
    if not rows:
        log.warning("No summary rows to write.")
        return
    df = pd.DataFrame(rows, columns=[
        "ML_Number", "Run", "Model", "Is_Best",
        "AUC", "AUPRC", "Sensitivity", "Specificity", "Balanced_Accuracy",
        "MCC",
        "Dataset",
    ])
    df.sort_values(["ML_Number", "Run", "Model", "Dataset"], inplace=True)
    df.to_csv(out_path, sep="\t", index=False)
    log.info("Wrote %d rows to %s", len(df), out_path)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = discover_datasets(INPUT_DIR)
    if not datasets:
        raise SystemExit(f"No TSV inputs discovered under {INPUT_DIR}")

    log.info("Discovered %d (study, dataset) inputs:", len(datasets))
    for sid, stem, _ in datasets:
        log.info("  study=%d  dataset=%s", sid, stem)

    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_TSV)

    completed = load_completed_runs(summary_path)
    if completed:
        log.info(
            "RESUME: %d already-complete (study, dataset, run) entries -- will skip.",
            len(completed),
        )
    else:
        log.info("Starting fresh run (no existing summary found).")

    summary_rows: list = []
    if completed and Path(summary_path).exists():
        try:
            existing_df = pd.read_csv(summary_path, sep="\t")
            summary_rows = existing_df.to_dict("records")
            log.info("Loaded %d existing rows.", len(summary_rows))
        except Exception as exc:
            log.warning("Could not reload existing rows: %s -- empty list.", exc)
            summary_rows = []

    total_iters = len(datasets) * N_RUNS
    skipped = 0
    iter_count = 0

    for study_id, dataset_stem, file_path in datasets:
        dataset_label = parse_dataset_label(dataset_stem)
        for seed in range(1, N_RUNS + 1):
            iter_count += 1
            resume_key = (study_id, dataset_label, seed)
            if resume_key in completed:
                skipped += 1
                log.info(
                    "[%d/%d] SKIP (already complete) study=%d dataset=%s run=%d",
                    iter_count, total_iters, study_id, dataset_stem, seed,
                )
                continue

            log.info(
                "\n=== [%d/%d] study=%d  dataset=%s  run=%d ===",
                iter_count, total_iters, study_id, dataset_stem, seed,
            )
            try:
                run_one(
                    study_id=study_id,
                    dataset_stem=dataset_stem,
                    file_path=file_path,
                    seed=seed,
                    summary_rows=summary_rows,
                )
            except Exception as exc:
                log.exception(
                    "FAILED study=%d dataset=%s run=%d: %s",
                    study_id, dataset_stem, seed, exc,
                )
            # Checkpoint after every (dataset, seed) attempt
            write_summary_tsv(summary_rows, summary_path)

    log.info(
        "Finished. Ran %d / %d iterations (%d skipped as already complete).",
        iter_count - skipped, total_iters, skipped,
    )
    aggregate_tuning_comparisons(
        base_dir=OUTPUT_DIR,
        out_path=os.path.join(OUTPUT_DIR, TUNING_SUMMARY_CSV),
    )
    log.info("All done.")


if __name__ == "__main__":
    main()