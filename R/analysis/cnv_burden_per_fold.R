source("lib/utils.R")
source("lib/cnv_helpers.R")

meta <- get_meta()
sample_map <- fread(path("methylation_samples_cleaned"))

# Read the classification patient-stratified k-fold splits (written by python/ml/export_folds.py --task classification).
# This per-fold pipeline is classification-only: its sole purpose is to keep a validation *nevus* out of its own nevus
# reference. The ordinal task (IM/NIM only) has no nevi among its samples, so it never uses these per-fold files.
folds_path <- results_dir("classifier", "cv", "cv_folds__classification.csv")
folds <- fread(folds_path)
folds[, slideId := as.character(slideId)]
n_folds <- length(unique(folds$val_fold))
cat("Loaded", nrow(folds), "fold assignments across", n_folds, "folds from", folds_path, "\n")

# Union analysis set (all train+val slides across all folds).
analysis_keys <- sample_map[as.character(slideId) %in% folds$slideId, sentrix_key]
cat("Analysis sample set:", length(analysis_keys), "sentrix keys (all kfold slides)\n")

# Load IDAT intensities once per array type — reused across all folds.
cat("=== Loading EPICv1 intensities (one-time) ===\n")
mat_v1 <- load_intensities(path("idat_v1_dir"), analysis_keys)
cat("=== Loading EPICv2 intensities (one-time) ===\n")
mat_v2 <- load_intensities(path("idat_v2_dir"), analysis_keys)

out_dir <- CNV_BURDEN_DIR
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Per-fold CNV analysis: rebuild nevus reference from training-fold nevi only.
for (k in sort(unique(folds$val_fold))) {
  cat(sprintf("\n=== Fold %d/%d ===\n", k + 1, n_folds))

  train_slides_k <- folds[val_fold != k, slideId]
  nevus_slides_k <- meta[as.character(slideId) %in% train_slides_k &
                           groupedPrimaryDiagnosisPatho == "NV", as.character(slideId)]
  nevus_keys_k <- sample_map[as.character(slideId) %in% nevus_slides_k, sentrix_key]
  cat("Fold", k, "training-fold nevus reference:", length(nevus_keys_k), "samples\n")

  run_cnv_for_reference(
    mat_v1, mat_v2, nevus_keys_k, sample_map,
    genome_csv = file.path(out_dir, sprintf("cnv_burden_results_fold%d.csv", k))
  )
  gc(verbose = FALSE)
}

cat("Done.\n")
