#!/usr/bin/env python
# /media/dominic/Behemoth/1_Programming/2_Python/6_bioinformatics/rarefaction_vs_nrarefaction/ML/ml_multi_3.py

## this is the script for multi class classification however includes looping

### opening pycaret and loading package
import pycaret
import pandas as pd
import numpy as np
import os
import glob
import random
from pycaret.classification import *
from sklearn.metrics import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import csv
import shap
from pathlib import Path

# ===================== paths =====================

DATA_DIR = "prepro_rarefied_data"
# plt.use("Agg")  # Non-GUI backend for headless plotting

# ===================== custom metrics =====================

def sensitivity(y_test, y_pred, **kwargs):
    return recall_score(y_test, y_pred, average="macro", zero_division=0)

def specificity(y_test, y_pred, **kwargs):
    cm = confusion_matrix(y_test, y_pred)
    spec_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_per_class.append(spec)
    return np.mean(spec_per_class)

def bal_acc(y_test, y_pred, **kwargs):
    # Uses a unique metric ID ("bal_acc") to avoid conflicting with PyCaret's
    # internal "balanced_accuracy" metric, which silently maps to macro recall
    # (identical to sensitivity). sklearn's balanced_accuracy_score computes
    # the mean of per-class recall correctly as a distinct value.
    return balanced_accuracy_score(y_test, y_pred)

def get_fold_count(n_samples):
    if n_samples < 50:
        return 3   # absolute minimum for tiny datasets
    elif n_samples < 100:
        return 5   # standard but cautious
    else:
        return 10  # reliable estimate with enough data
    
def get_n_iter(n_samples):
    if n_samples < 50:
        return 10   # small data — don't overfit the tuning
    elif n_samples < 100:
        return 20
    elif n_samples < 300:
        return 50
    else:
        return 100

def save_confusion_matrix(
    y_test,
    y_pred,
    model_name,
    out_dir,
    label_map
    ):
    os.makedirs(out_dir, exist_ok=True)
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in sorted(inv_label_map)]
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))    
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – {model_name}")
    out_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.tiff")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format="tiff")
    plt.close()
    return out_path

def save_classification_report(model, model_name, classrep_dir):
    """
    Saves PyCaret classification report plot as TIFF
    """
    os.makedirs(classrep_dir, exist_ok=True)
    # PyCaret saves the plot and returns the file path
    png_path = plot_model(model, plot="class_report", save=True)
    # make absolute if relative
    if not os.path.isabs(png_path):
        png_path = os.path.abspath(png_path)
    # target TIFF path
    out_path = os.path.join(classrep_dir, f"{model_name}_classification_report.tiff")
    # convert PNG → TIFF
    img = plt.imread(png_path)
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format="tiff")
    plt.close()
    # remove PNG after conversion
    os.remove(png_path)
    return out_path


def save_model_outputs(model, model_name, data, results,
                       results_dir, models_dir,
                       confmtx_dir, classrep_dir):
    # ---- save CV results ----
    results.to_csv(
        os.path.join(results_dir, f"{model_name}_cv_results.csv"),
        index=False
    )
    # ---- predictions ----
    preds = predict_model(model, data=data)

    # ---- confusion matrix ----
    save_confusion_matrix(
        preds["type"],
        preds["prediction_label"],
        model_name=model_name,
        out_dir=confmtx_dir,
        label_map=label_map
    )

    # ---- classification report ----
    save_classification_report(model, model_name, classrep_dir)
    # ---- save model ----
    save_model(
        model,
        os.path.join(models_dir, model_name)
    )

def make_label_map(df, target_col="type"):
    # normalize
    labels = (
        df[target_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    if len(labels) < 2:
        raise ValueError(
            f"Need at least 2 classes, found {labels}"
        )

    label_map = {}

    if "control" in labels:
        label_map["control"] = 0
        other_labels = sorted([l for l in labels if l != "control"])
        for i, lbl in enumerate(other_labels, start=1):
            label_map[lbl] = i
    else:
        # no control → encode alphabetically
        for i, lbl in enumerate(sorted(labels)):
            label_map[lbl] = i

    return label_map

def get_estimator(model):
    """Safely unwrap PyCaret pipeline to get the actual estimator."""
    if hasattr(model, "named_steps"):
        # Try both possible step names PyCaret uses
        return (model.named_steps.get("actual_estimator") or
                model.named_steps.get("trained_model") or
                list(model.named_steps.values())[-1])  # fallback: last step
    return model  # already a raw estimator

def compute_and_save_feature_importance(model, X_test, y_test, model_name, seed, out_dir):
    """
    Computes built-in, permutation, and SHAP feature importance
    for RF, LR, NB, and Blended models. Saves CSVs and plots.
    """
    feature_names = X_test.columns.tolist()
    results = {}

    # ── 1. Built-in importance ──────────────────────────
    try:
        est = get_estimator(model)

        if "RandomForest" in model_name:
            builtin = est.feature_importances_

        elif "LogisticRegression" in model_name:
            builtin = np.abs(est.coef_).mean(axis=0)

        elif "NaiveBayes" in model_name:
            builtin = np.abs(est.theta_ - est.theta_.mean(axis=0)).mean(axis=0)

        elif "Blended" in model_name:
            importances = []
            for _, sub_est in est.named_estimators_.items():
                if hasattr(sub_est, "feature_importances_"):
                    importances.append(sub_est.feature_importances_)
                elif hasattr(sub_est, "coef_"):
                    importances.append(np.abs(sub_est.coef_).mean(axis=0))
            builtin = np.mean(importances, axis=0) if importances else np.zeros(len(feature_names))

        results["builtin"] = builtin

    except Exception as e:
        print(f"  Built-in importance failed for {model_name}: {e}")
        results["builtin"] = np.zeros(len(feature_names))

    # ── 2. Permutation importance ────────────────────────
    try:
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=seed,
            scoring="balanced_accuracy"
        )
        results["permutation"] = perm.importances_mean
    except Exception as e:
        print(f"  Permutation importance failed for {model_name}: {e}")
        results["permutation"] = np.zeros(len(feature_names))

    # ── 3. SHAP ──────────────────────────────────────────
    try:
        est = get_estimator(model)
        X_arr = X_test.values

        if "RandomForest" in model_name:
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(X_arr)

        elif "LogisticRegression" in model_name:
            explainer = shap.LinearExplainer(est, X_arr)
            sv = explainer.shap_values(X_arr)

        else:  # NB, Blended
            try:
                n_bg = min(10, len(X_arr))
                bg = shap.kmeans(X_arr, n_bg)
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                sv = explainer.shap_values(
                    X_arr,
                    nsamples=100,
                    l1_reg="num_features(10)"
                )
            except Exception:
                # Fallback for n_samples < n_features: use a single background point
                bg = X_arr.mean(axis=0, keepdims=True)
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                sv = explainer.shap_values(
                    X_arr,
                    nsamples=100,
                    l1_reg=0   # skip regularisation entirely
                )

        sv_arr = np.array(sv)
        if sv_arr.ndim == 3:
            shap_mean = np.abs(sv_arr).mean(axis=(0, 2)) if sv_arr.shape[0] == len(X_arr) \
                        else np.abs(sv_arr).mean(axis=(0, 1))
        elif sv_arr.ndim == 2:
            shap_mean = np.abs(sv_arr).mean(axis=0)
        else:
            shap_mean = np.abs(sv_arr).flatten()

        results["shap"] = shap_mean

    except Exception as e:
        print(f"  SHAP failed for {model_name}: {e}")
        results["shap"] = np.zeros(len(feature_names))

    # ── 4. Save CSV ──────────────────────────────────────
    df_out = pd.DataFrame(results, index=feature_names)
    df_out.index.name = "feature"
    csv_path = os.path.join(out_dir, f"{model_name}_seed{seed}_feature_importance.csv")
    df_out.to_csv(csv_path)

    # ── 5. Save plots ────────────────────────────────────
    top_n = 20
    for method in ["builtin", "permutation", "shap"]:
        imp = pd.Series(results[method], index=feature_names).nlargest(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(imp)), imp.values[::-1], color="steelblue", alpha=0.85)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels([f[:40] for f in imp.index[::-1]], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"{model_name} | seed {seed} | {method} — Top {top_n} features")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"{model_name}_seed{seed}_{method}_importance.tiff")
        fig.savefig(plot_path, dpi=300, format="tiff")
        plt.close(fig)

def save_tuning_comparison(base_results, tuned_results, model_name, seed, out_dir):
    """
    Compares CV metrics before and after tuning and saves to CSV.
    base_results  — pull() after create_model()
    tuned_results — pull() after tune_model()
    """
    os.makedirs(out_dir, exist_ok=True)

    # PyCaret results tables have a Mean and SD row at the bottom
    # Extract just the Mean row for comparison
    # Handle both named and numeric index formats
    if "Mean" in base_results.index:
        base_mean  = base_results[base_results.index == "Mean"].copy()
        tuned_mean = tuned_results[tuned_results.index == "Mean"].copy()
    else:
        # Fall back to last row if no Mean label
        base_mean  = base_results.iloc[[-1]].copy()
        tuned_mean = tuned_results.iloc[[-1]].copy()

    base_mean.index  = ["Before Tuning"]
    tuned_mean.index = ["After Tuning"]

    comparison = pd.concat([base_mean, tuned_mean])

    # Add a delta row
    delta = tuned_mean.values - base_mean.values
    delta_df = pd.DataFrame(delta, columns=comparison.columns, index=["Delta"])
    comparison = pd.concat([comparison, delta_df])

    # Flag whether tuning actually changed anything
    tuning_improved = (delta_df > 0).any(axis=1).values[0]
    comparison["tuning_improved"] = ["", "", str(tuning_improved)]

    comparison.insert(0, "model", model_name)
    comparison.insert(1, "seed", seed)

    out_path = os.path.join(out_dir, f"{model_name}_seed{seed}_tuning_comparison.csv")
    comparison.to_csv(out_path)
    return comparison
    
def aggregate_tuning_comparisons(base_results_dir, out_path):
    """
    Walks results directories and concatenates all tuning comparison CSVs
    into one summary file.
    """
    all_files = glob.glob(
        os.path.join(base_results_dir, "**", "*tuning_comparison*.csv"),
        recursive=True
    )
    if not all_files:
        print("No tuning comparison files found.")
        return

    dfs = []
    for f in all_files:
        df = pd.read_csv(f, index_col=0)
        # Extract dataset name from path
        parts = Path(f).parts
        dataset = parts[-5] if len(parts) >= 5 else "unknown"
        df.insert(0, "dataset", dataset)
        dfs.append(df)

    summary = pd.concat(dfs, ignore_index=False)
    summary.to_csv(out_path)
    print(f"Saved tuning summary: {out_path}")
    return summary
















# ===================== loop over tsv files =====================
print("starting")
print(os.getcwd())
print(os.listdir())

tsv_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
print(tsv_files)

for file_path in tsv_files:

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for seed in range(1, 11):  # 10 runs
        print(f"\n=== Processing: {base_name} | Run {seed}/10 ===")

        # ---------- directories per run ----------
        BASE_RUN_DIR = os.path.join(
            "ML", "pycaret_results", base_name, f"run_{seed}"
        )

        RESULTS_DIR  = os.path.join(BASE_RUN_DIR, "results")
        MODELS_DIR   = os.path.join(BASE_RUN_DIR, "models")
        PLOTS_DIR    = os.path.join(BASE_RUN_DIR, "plots")
        CONFMTX_DIR  = os.path.join(PLOTS_DIR, "confusion_matrix")
        TEST_DIR     = os.path.join(PLOTS_DIR, "test")
        CLASSREP_DIR = os.path.join(PLOTS_DIR, "class_reports")
        FEATIMP_DIR = os.path.join(PLOTS_DIR, "feature_importance")
        TUNING_DIR = os.path.join(RESULTS_DIR, "tuning")

        for d in [RESULTS_DIR, TUNING_DIR, MODELS_DIR, CONFMTX_DIR, TEST_DIR, CLASSREP_DIR, FEATIMP_DIR]:
            os.makedirs(d, exist_ok=True)

        # ---------- load data ----------
        data = pd.read_csv(file_path, sep="\t")

        data.columns = (
            data.columns
                .str.replace(" ", "_", regex=False)
                .str.replace("-", "_", regex=False)
        )

        data["type"] = (
            data["type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        label_map = make_label_map(data, "type")
        data["type"] = data["type"].map(label_map)

        print("Detected label map:", label_map)
        print("Class distribution:",
              data["type"].value_counts().sort_index().to_dict())

        if data["type"].isna().any():
            raise ValueError("NaNs introduced during label mapping")

        n_classes = data["type"].nunique()
        if n_classes < 2:
            raise ValueError("Only one class present after mapping")

        problem_type = "binary" if n_classes == 2 else "multiclass"
        print(f"Detected problem type: {problem_type}")

        # ---------- setup ----------
        n_folds = get_fold_count(len(data))
        n_iter = get_n_iter(len(data))

        setup(
            data=data,
            target="type",
            session_id=seed, 
            train_size=0.8,
            fold_strategy="stratifiedkfold",
            fold=n_folds,
            fix_imbalance=False,
            remove_outliers=False,
            feature_selection=False,
            verbose=False
        )

        # ---------- metrics ----------
        add_metric("sensitivity", "Sensitivity", sensitivity)
        add_metric("specificity", "Specificity", specificity)
        # Use "bal_acc" as the ID — PyCaret's internal "balanced_accuracy"
        # maps to macro recall which duplicates sensitivity. "bal_acc" is a
        # distinct ID that forces PyCaret to use our custom function instead.
        add_metric("bal_acc", "Balanced_Accuracy", bal_acc)

        # ---------- base models ----------
        # LR: increase max_iter and use saga solver with L2 regularisation
        # to prevent convergence failure on underdetermined datasets (n << p).
        # This fixes the AUC = 0 artefact caused by degenerate probability
        # estimates when lbfgs fails to converge.
        lr = create_model("lr", max_iter=5000, solver="saga", penalty="l2")

        lr_results = pull()
        
        # tune model — keep saga solver and high max_iter during tuning
        tuned_lr = tune_model(
            lr,
            fold=n_folds,
            optimize="AUC",
            n_iter=n_iter,
            choose_better=True,
            custom_grid={"solver": ["saga"], "max_iter": [5000],
                         "C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]}
            )

        lr_tuned_results = pull()

        # Save comparison
        save_tuning_comparison(
            lr_results,
            lr_tuned_results,
            model_name="LogisticRegression",
            seed=seed,
            out_dir=TUNING_DIR
        )

        save_model_outputs(
            tuned_lr, f"lr_run{seed}", data, lr_tuned_results,
            RESULTS_DIR, MODELS_DIR,
            CONFMTX_DIR, CLASSREP_DIR
        )

        nb = create_model("nb")
                
        nb_results = pull()

        # tune model
        tuned_nb = tune_model(
            nb,
            fold=n_folds,        # match the setup fold count
            optimize="AUC",
            n_iter=n_iter,
            choose_better = True 
            )


        nb_tuned_results = pull()

        # Save comparison
        save_tuning_comparison(
            nb_results,
            nb_tuned_results,
            model_name="NaiveBayes",
            seed=seed,
            out_dir=TUNING_DIR
        )

        save_model_outputs(
            tuned_nb, f"nb_run{seed}", data, nb_tuned_results,
            RESULTS_DIR, MODELS_DIR,
            CONFMTX_DIR, CLASSREP_DIR
        )

        rf = create_model("rf")

        rf_results = pull()

        # tune model
        tuned_rf = tune_model(
            rf, 
            fold=n_folds,        # match the setup fold count
            optimize="AUC",
            n_iter=n_iter,
            choose_better = True 
            )

        rf_tuned_results = pull()

        # Save comparison
        save_tuning_comparison(
            rf_results,
            rf_tuned_results,
            model_name="RandomForest",
            seed=seed,
            out_dir=TUNING_DIR
        )

        save_model_outputs(
            tuned_rf, f"rf_run{seed}", data, rf_tuned_results,
            RESULTS_DIR, MODELS_DIR,
            CONFMTX_DIR, CLASSREP_DIR
        )

        # ---------- blending ----------
        lr_copy = create_model("lr")
        nb_copy = create_model("nb")
        rf_copy = create_model("rf")

        blender = blend_models(
            [lr_copy, nb_copy, rf_copy],
            optimize="AUC"
        )

        blender_results = pull()

        # randomgrid search by default
        tuned_blended = tune_model(
            blender,
            fold=n_folds,        # match the setup fold count
            optimize="AUC",
            n_iter=n_iter,
            choose_better = True 
        )

        blender_tuned_results = pull()

        # Save comparison
        save_tuning_comparison(
            blender_results,
            blender_tuned_results,
            model_name="BlendedModel",
            seed=seed,
            out_dir=TUNING_DIR
        )

        save_model_outputs(
            tuned_blended, f"blended_run{seed}", data, blender_tuned_results,
            RESULTS_DIR, MODELS_DIR,
            CONFMTX_DIR, CLASSREP_DIR
        )

        # ---------- test evaluation ----------
        models = {
            "LogisticRegression": tuned_lr,
            "NaiveBayes": tuned_nb,
            "RandomForest": tuned_rf,
            "BlendedModel": tuned_blended
        }

        class_names = sorted(label_map, key=label_map.get)

        for name, model in models.items():

            preds = predict_model(model)
            y_test = preds["type"]
            y_pred = preds["prediction_label"]

            # ── Feature Importance ──────────────────────────────
            # Get test features (drop pycaret-added columns)
            drop_cols = ["type", "prediction_label", "prediction_score"]
            X_test = preds.drop(columns=[c for c in drop_cols if c in preds.columns])
            
            compute_and_save_feature_importance(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=name,
                seed=seed,
                out_dir=FEATIMP_DIR
            )

            save_confusion_matrix(
                y_test,
                y_pred,
                model_name=f"{name}_run{seed}",
                out_dir=TEST_DIR,
                label_map=label_map
            )

            report_path = os.path.join(
                TEST_DIR,
                f"{name}_run{seed}_classification_report.txt"
            )

            with open(report_path, "w") as f:
                f.write(
                    classification_report(
                        y_test,
                        y_pred,
                        target_names=class_names
                    )
                )

# Call after all loops finish
aggregate_tuning_comparisons(
    base_results_dir="ML/pycaret_results",
    out_path="ML/tuning_comparison_summary.csv"
)