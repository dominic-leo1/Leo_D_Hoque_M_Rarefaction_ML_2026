#!/usr/bin/env python
# /media/dominic/Behemoth/1_Programming/2_Python/6_bioinformatics/ml_pipeline_binary_2_DL.py

### UPDATES ###
# This pipeline includes looping 10 seeds

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
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# ===================== paths =====================

DATA_DIR = "prepro_rarefied_data"
# plt.use("Agg")  # Non-GUI backend for headless plotting

# ===================== custom metrics =====================

def sensitivity(y_test, y_pred, **kwargs):
    return recall_score(y_test, y_pred, pos_label=1)

def specificity(y_test, y_pred, **kwargs):
    return recall_score(y_test, y_pred, pos_label=0)

def balanced_accuracy(y_test, y_pred, **kwargs):
    sens = recall_score(y_test, y_pred, pos_label=1)
    spec = recall_score(y_test, y_pred, pos_label=0)
    return (sens + spec) / 2

def save_confusion_matrix(
    y_test,
    y_pred,
    model_name,
    out_dir
    ):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["control", case_label],
        yticklabels=["control", case_label]
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
        out_dir=confmtx_dir
    )
    # ---- classification report ----
    save_classification_report(model, model_name, classrep_dir)
    # ---- save model ----
    save_model(
        model,
        os.path.join(models_dir, model_name)
    )

def make_label_map(df, target_col):
    labels = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    if "control" not in labels:
        raise ValueError(
            f"'control' not found in {target_col}. Found: {labels}"
        )

    if len(labels) != 2:
        raise ValueError(
            f"Expected binary classification, found {labels}"
        )

    case_label = [l for l in labels if l != "control"][0]

    return {
        "control": 0,
        case_label: 1
    }, case_label


# ===================== loop over tsv files =====================
print("starting")
print(os.getcwd())
print(os.listdir())

os.listdir("prepro_rarefied_data")
tsv_files = glob.glob(os.path.join(DATA_DIR, "*.tsv"))
print(tsv_files)
for file_path in tsv_files:

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for seed in range(1, 11):  # 10 runs
        print(f"\n=== Processing: {base_name} | Run {seed}/10 ===")

        # ---- per-run directory ----
        BASE_RUN_DIR = os.path.join(
            "ML", "pycaret_results", base_name, f"run_{seed}"
        )

        RESULTS_DIR  = os.path.join(BASE_RUN_DIR, "results")
        MODELS_DIR   = os.path.join(BASE_RUN_DIR, "models")
        PLOTS_DIR    = os.path.join(BASE_RUN_DIR, "plots")
        CONFMTX_DIR  = os.path.join(PLOTS_DIR, "confusion_matrix")
        TEST_DIR     = os.path.join(PLOTS_DIR, "test")
        CLASSREP_DIR = os.path.join(PLOTS_DIR, "class_reports")

        for d in [RESULTS_DIR, MODELS_DIR, CONFMTX_DIR, TEST_DIR, CLASSREP_DIR]:
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

        label_map, case_label = make_label_map(data, "type")
        data["type"] = data["type"].map(label_map)

        # ---------- PyCaret setup ----------

        setup(
            data=data,
            target="type",
            session_id=seed,   # 🔥 key change
            train_size=0.8,
            fix_imbalance=False,
            remove_outliers=False,
            feature_selection=False,
            verbose=False
        )

        # ---------- metrics ----------
        add_metric("sensitivity", "Sensitivity", sensitivity)
        add_metric("specificity", "Specificity", specificity)
        add_metric("balanced_accuracy", "Balanced_accuracy", balanced_accuracy)

        # ---------- models ----------
        lr = create_model("lr")
        lr_results = pull()
        save_model_outputs(
            lr, f"lr_run{seed}", data, lr_results,
            RESULTS_DIR, MODELS_DIR,
            CONFMTX_DIR, CLASSREP_DIR
        )

        nb = create_model("nb")
        nb_results = pull()
        save_model_outputs(
            nb, f"nb_run{seed}", data, nb_results,
            RESULTS_DIR, MODELS_DIR,
            CONFMTX_DIR, CLASSREP_DIR
        )

        rf = create_model("rf")
        rf_results = pull()
        save_model_outputs(
            rf, f"rf_run{seed}", data, rf_results,
            RESULTS_DIR, MODELS_DIR,
            CONFMTX_DIR, CLASSREP_DIR
        )

        # ---------- blend ----------
        lr_copy = create_model("lr")
        nb_copy = create_model("nb")
        rf_copy = create_model("rf")

        blender = blend_models(
            [lr_copy, nb_copy, rf_copy],
            optimize="AUC"
        )

        tuned = tune_model(
            blender,
            fold=5,
            optimize="AUC",
            n_iter=5
        )

        tuned_results = pull()

        save_model_outputs(
            tuned, f"blended_run{seed}", data, tuned_results,
            RESULTS_DIR, MODELS_DIR,
            CONFMTX_DIR, CLASSREP_DIR
        )

        # ---------- test confusion matrices ----------
        models = {
            "LogisticRegression": lr,
            "NaiveBayes": nb,
            "RandomForest": rf,
            "BlendedModel": tuned
        }

        for name, model in models.items():
            preds = predict_model(model)
            y_test = preds["type"]
            y_pred = preds["prediction_label"]

            save_confusion_matrix(
                y_test,
                y_pred,
                model_name=f"{name}_run{seed}",
                out_dir=TEST_DIR
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
                        target_names=["control", case_label]
                    )
                )
