#!/usr/bin/env Rscript
# ============================================================================
# preprocess_from_rds.R
# ============================================================================
# Label-free preprocessing of the 12 study phyloseq objects into the feature
# tables used for machine-learning classification.
#
# Steps (none of which use the class labels):
#   1. Standardise labels to {control, <disease>}.
#   2. Repeated rarefaction: 1000 iterations averaged, with the depth set to
#      the 10th percentile of each study's per-sample read-count distribution.
#   3. Taxonomic collapsing (for the *_taxa outputs).
#   4. Prevalence filter: drop features present in fewer than MIN_PREVALENCE
#      of samples.
#
# Label-aware feature selection and per-fold scaling are handled downstream,
# inside the cross-validation loop of the classification pipeline, so they
# only ever see training-fold samples.
#
# Outputs per study (4 files) -> OUTPUT_DIR:
#   <study_id>_<disease>_nonrarefied_asv.tsv
#   <study_id>_<disease>_nonrarefied_taxa.tsv
#   <study_id>_<disease>_rarefied_asv.tsv
#   <study_id>_<disease>_rarefied_taxa.tsv
# Each has the label column "type" plus the prevalence-filtered feature set.
# 12 studies x 4 files = 48 TSVs.
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(phyloseq)
  library(microeco)
  library(file2meco)
  library(parallel)
})

# ── PATHS ────────────────────────────────────────────────────────────────────
INPUT_DIR  <- "/home/mozammel/01.moz/07.rarefaction_ml/07.dom_phyloseq_data/01.physeq_obj"
OUTPUT_DIR <- "/home/mozammel/01.moz/07.rarefaction_ml/07.dom_phyloseq_data/02.preprocessed_files_leakfree"

# ── STUDY -> DISEASE LABEL MAP ───────────────────────────────────────────────
STUDY_MAP <- list(
  "1"="IBD","2"="HCC","6"="VAP","8"="breast_cancer","9"="Parkinson",
  "10"="gout","11"="CRC","12"="periodontitis","13"="S_ECC",
  "15"="HIV","16"="HIV","17"="UTI"
)
CONTROL_TOKENS <- c("control","ctrl","healthy","hc","ab","normal","neg")

# ── SETTINGS ─────────────────────────────────────────────────────────────────
N_REPS         <- 1000     # repeated rarefaction iterations
SEED           <- 123
N_CORES        <- 4        # study-level + iteration-level parallelism
MIN_PREVALENCE <- 0.10     # label-FREE prevalence filter: keep features
                           # present (nonzero) in >= 10% of samples

options(future.globals.maxSize = 10 * 1024^3)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)


# ─────────────────────────────────────────────────────────────────────────────
# Standardise labels to {control, <disease>}
# ─────────────────────────────────────────────────────────────────────────────
standardise_labels <- function(physeq, disease_label) {
  if ("type" %in% colnames(sample_data(physeq))) {
    colnames(sample_data(physeq))[colnames(sample_data(physeq)) == "type"] <- "sample_type"
  }
  if (!"sample_type" %in% colnames(sample_data(physeq))) {
    stop("No 'sample_type' or 'type' column in sample_data.")
  }
  labels <- as.character(sample_data(physeq)$sample_type)
  cat("  Original labels:", paste(unique(labels), collapse = ", "), "\n")
  new_labels <- vapply(labels, function(x) {
    lx <- tolower(trimws(x))
    if (any(vapply(CONTROL_TOKENS, function(tok) identical(lx, tok), logical(1)))) {
      "control"
    } else {
      disease_label
    }
  }, character(1))
  sample_data(physeq)$sample_type <- new_labels
  cat("  Standardised counts:\n"); print(table(sample_data(physeq)$sample_type))
  sample_data(physeq) <- sample_data(physeq)[, "sample_type", drop = FALSE]
  physeq
}


# ─────────────────────────────────────────────────────────────────────────────
# safe_rrarefy + repeated_rarefy  (unchanged from the working version)
# ─────────────────────────────────────────────────────────────────────────────
safe_rrarefy <- function(x, sample) {
  t(apply(x, 1, function(row) {
    reads <- rep(seq_along(row), times = row)
    tabulate(base::sample(reads, size = sample, replace = FALSE), nbins = length(row))
  }))
}

repeated_rarefy <- function(physeq_obj, depth, n_reps, seed = 123) {
  otu_mat <- as.matrix(t(otu_table(physeq_obj)))
  storage.mode(otu_mat) <- "integer"
  keep <- rowSums(otu_mat) >= depth
  if (sum(!keep) > 0) {
    cat("  Excluding", sum(!keep), "sample(s) below depth", depth, "\n")
  }
  otu_mat <- otu_mat[keep, , drop = FALSE]
  cat("  OTU matrix:", nrow(otu_mat), "samples x", ncol(otu_mat), "taxa\n")

  reps_per_core  <- ceiling(n_reps / N_CORES)
  seeds_per_core <- seed + seq(0, N_CORES - 1) * reps_per_core
  core_results <- mclapply(seq_len(N_CORES), function(ci) {
    set.seed(seeds_per_core[ci])
    reps <- min(reps_per_core, n_reps - (ci - 1) * reps_per_core)
    if (reps <= 0) return(matrix(0, nrow(otu_mat), ncol(otu_mat), dimnames = dimnames(otu_mat)))
    acc <- matrix(0, nrow(otu_mat), ncol(otu_mat), dimnames = dimnames(otu_mat))
    for (i in seq_len(reps)) acc <- acc + safe_rrarefy(otu_mat, sample = depth)
    acc
  }, mc.cores = N_CORES)

  accumulated <- Reduce("+", core_results)
  mean_mat <- round(accumulated / n_reps)
  mean_mat <- mean_mat[, colSums(mean_mat) > 0, drop = FALSE]
  retained <- colnames(mean_mat)
  cat("  Features retained after averaging:", ncol(mean_mat), "\n")

  new_otu <- otu_table(t(mean_mat), taxa_are_rows = TRUE)
  kept_sam <- sample_data(physeq_obj)[rownames(mean_mat), , drop = FALSE]
  tax_mat <- as(tax_table(physeq_obj), "matrix")
  kept_tax <- tax_table(tax_mat[retained, , drop = FALSE])
  new_physeq <- phyloseq(new_otu, kept_tax, kept_sam)
  tree <- phy_tree(physeq_obj, errorIfNULL = FALSE)
  if (!is.null(tree)) new_physeq <- merge_phyloseq(new_physeq, prune_taxa(retained, tree))
  new_physeq
}


# ─────────────────────────────────────────────────────────────────────────────
# LABEL-FREE feature tables. No class labels used for selection: only a
# prevalence filter (fraction of samples in which a feature is nonzero).
# The label-aware selection is deferred to the ML CV loop.
# ─────────────────────────────────────────────────────────────────────────────
prevalence_filter <- function(mat, min_prev) {
  # mat: samples x features
  prev <- colMeans(mat > 0)
  keep <- prev >= min_prev
  mat[, keep, drop = FALSE]
}

build_asv_table <- function(physeq_obj) {
  otu <- as.data.frame(t(physeq_obj@otu_table))        # samples x ASVs
  labels <- as.character(physeq_obj@sam_data$sample_type)
  otu <- prevalence_filter(as.matrix(otu), MIN_PREVALENCE)
  out <- as.data.frame(otu, check.names = FALSE)
  out <- cbind(type = labels, out)
  out
}

build_taxa_table <- function(physeq_obj) {
  meco <- phyloseq2meco(physeq_obj)
  meco$tidy_dataset(); meco$cal_abund()
  d1 <- trans_classifier$new(dataset = meco,
                             y.response = "sample_type",
                             x.predictors = "All")
  taxa <- d1$data_feature

  # trans_classifier encodes the response (first column) as a numeric/factor
  # code, NOT the original "control"/disease string. Do NOT use taxa[[1]]
  # as the label. Instead, take the feature matrix and re-attach the true
  # string labels from the phyloseq sample_data, matched by sample rowname.
  #
  # Identify the response column by name if present, else assume column 1,
  # and drop it from the feature block regardless.
  resp_candidates <- c("sample_type", "Response", "response", "Class", "class")
  resp_col <- which(colnames(taxa) %in% resp_candidates)
  if (length(resp_col) >= 1) {
    feats_df <- taxa[, -resp_col[1], drop = FALSE]
  } else {
    feats_df <- taxa[, -1, drop = FALSE]   # fallback: drop first (coded) col
  }
  feats <- as.matrix(feats_df)

  # Recover the true labels. data_feature rows correspond to samples; use
  # its rownames to index sample_data. If rownames are missing, fall back
  # to sample order (data_feature preserves the meco sample order).
  meta_labels <- as.character(sample_data(physeq_obj)$sample_type)
  names(meta_labels) <- rownames(sample_data(physeq_obj))
  rn <- rownames(taxa)
  if (!is.null(rn) && all(rn %in% names(meta_labels))) {
    labels <- meta_labels[rn]
  } else if (nrow(feats) == length(meta_labels)) {
    labels <- meta_labels   # positional fallback
  } else {
    stop("Cannot align taxa feature rows to sample labels ",
         "(nrow feats=", nrow(feats), ", n samples=", length(meta_labels), ")")
  }
  labels <- unname(labels)

  # Sanity: the label vector must contain 'control' and exactly two classes
  tab <- table(labels)
  if (!("control" %in% names(tab)) || length(tab) != 2) {
    stop("Taxa label vector invalid: found levels {",
         paste(names(tab), collapse = ", "), "}")
  }

  colnames(feats) <- gsub("[[:punct:]]", "_", colnames(feats))
  colnames(feats) <- gsub("k__Bacteria_p", "p", colnames(feats))
  colnames(feats) <- gsub("__", "_", colnames(feats))
  if (any(duplicated(colnames(feats)))) {
    colnames(feats) <- make.unique(colnames(feats), sep = "_dup")
  }
  feats <- prevalence_filter(feats, MIN_PREVALENCE)
  out <- as.data.frame(feats, check.names = FALSE)
  out <- cbind(type = labels, out)
  out
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-study driver
# ─────────────────────────────────────────────────────────────────────────────
process_study <- function(rds_path, study_id, disease_label) {
  cat("\n", strrep("=", 70), "\nSTUDY ", study_id, " (", disease_label, ")\n",
      strrep("=", 70), "\n", sep = "")
  physeq1 <- readRDS(rds_path)
  physeq1 <- standardise_labels(physeq1, disease_label)

  depth <- as.integer(quantile(sample_sums(physeq1), probs = 0.10))
  cat("  10th-percentile depth:", depth, "\n")
  ps.rar <- repeated_rarefy(physeq1, depth = depth, n_reps = N_REPS, seed = SEED)

  prefix <- paste0(study_id, "_", disease_label)

  # rarefied
  write_tsv(build_asv_table(ps.rar),
            file.path(OUTPUT_DIR, paste0(prefix, "_rarefied_asv.tsv")))
  write_tsv(build_taxa_table(ps.rar),
            file.path(OUTPUT_DIR, paste0(prefix, "_rarefied_taxa.tsv")))
  # non-rarefied
  write_tsv(build_asv_table(physeq1),
            file.path(OUTPUT_DIR, paste0(prefix, "_nonrarefied_asv.tsv")))
  write_tsv(build_taxa_table(physeq1),
            file.path(OUTPUT_DIR, paste0(prefix, "_nonrarefied_taxa.tsv")))

  cat("  Study", study_id, "done -- 4 TSVs written (label-free).\n")
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
main <- function() {
  cat("Preprocessing\nInput:", INPUT_DIR, "\nOutput:", OUTPUT_DIR, "\n")
  pending <- Filter(function(sid) {
    disease  <- STUDY_MAP[[sid]]
    prefix   <- paste0(sid, "_", disease)
    rds_path <- file.path(INPUT_DIR, paste0(sid, "_physeq.rds"))
    if (!file.exists(rds_path)) { cat("!! MISSING RDS study", sid, "\n"); return(FALSE) }
    expected <- file.path(OUTPUT_DIR, paste0(prefix, c(
      "_nonrarefied_asv.tsv","_nonrarefied_taxa.tsv",
      "_rarefied_asv.tsv","_rarefied_taxa.tsv")))
    if (all(file.exists(expected))) { cat("Study", sid, "complete -- skip\n"); return(FALSE) }
    TRUE
  }, names(STUDY_MAP))

  for (sid in pending) {
    tryCatch(
      process_study(file.path(INPUT_DIR, paste0(sid, "_physeq.rds")), sid, STUDY_MAP[[sid]]),
      error = function(e) cat("!! FAILED study", sid, ":", conditionMessage(e), "\n")
    )
  }
  cat("\nBatch complete. Outputs in:", OUTPUT_DIR, "\n")
}

main()