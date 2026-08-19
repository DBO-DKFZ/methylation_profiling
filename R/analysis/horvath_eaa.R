# Fold-aware, leakage-safe Horvath epigenetic age acceleration (EAA).
#
# EAA is the residual of the skinHorvath epigenetic age regressed on chronological age — methylclock's `ageAcc2`,
# which is literally `resid(lm(skinHorvath ~ age))`. methylclock fits that line across the whole batch passed to
# DNAmAge, so held-out rows leak into their own acceleration value. This script recomputes the residual with the
# regression fit on the appropriate training set only, using base-R `lm` (the same model methylclock uses internally,
# so this is not a reimplementation of the clock — the skinHorvath ages still come from `horvath_clock.R`):
#
#   * eaa_base : line fit on the training cohort (clinic not in test_clinic), per array version, scored on every row.
#                Used by the final model and the external test set.
#   * eaa_oof  : each sample scored against the line fit on the training rows of the CV fold in which it is held out,
#                per array version. Used in cross-validation (swapped into the validation rows). Leakage-safe per
#                sample: each value's regression was fit on a fold that excluded that sample.
#
# Inputs : results/horvath_clock/horvath_ages_v{1,2}.csv  (skinHorvath + chronological age, per array)
#          results/classifier/cv/cv_folds__<task>.csv     (val_fold per slideId; written by python/ml/export_folds.py)
#          meta (clinic)                                  (train/test split)
# Output : results/horvath_clock/horvath_eaa__<task>.csv   [slideId, version, eaa_base, eaa_oof], one file per task.
#          eaa_base is task-independent (training-cohort fit); eaa_oof follows each task's fold assignment, so the
#          classification and ordinal folds — which stratify differently — each get their own out-of-fold residuals.
# Run order: horvath_clock.R and `python -m python.ml.export_folds --task <task>` (independent) must both have run first.

source("lib/utils.R")

TASKS <- c("classification", "ordinal")

meta <- get_meta()
test_clinics <- as.character(config$test_clinic)  # scalar or list, matching python/preprocessing.py
train_ids <- meta[!(as.character(clinic) %in% test_clinics), as.character(slideId)]
cat("Training cohort (clinic not in", paste(test_clinics, collapse = "/"), "):", length(train_ids), "samples\n")

# Residual of skinHorvath ~ age with the OLS line fit on `fit_ids` only, scored on every row of `d`.
# Reproduces methylclock's ageAcc2 with a controllable fit population.
residual_against_fit <- function(d, fit_ids) {
  fit <- d[slideId %in% fit_ids & !is.na(skinHorvath) & !is.na(age)]
  model <- lm(skinHorvath ~ age, data = fit)
  as.numeric(d$skinHorvath - predict(model, newdata = d))
}

compute_eaa <- function(ages_path, version_label, folds) {
  cat("\n=== EAA for", version_label, "===\n")
  d <- fread(ages_path)
  d <- d[, .(slideId = as.character(id), skinHorvath, age)]
  d[, version := version_label]

  # Base: line fit on the training cohort (this array), scored on everyone — final model + test set.
  d[, eaa_base := residual_against_fit(d, train_ids)]

  # OOF: each sample scored against the line fit on its hold-out fold's training rows — used in CV.
  d[, eaa_oof := NA_real_]
  for (k in sort(unique(folds$val_fold))) {
    resid_k <- residual_against_fit(d, folds[val_fold != k, slideId])
    val_rows <- d$slideId %in% folds[val_fold == k, slideId]
    d[val_rows, eaa_oof := resid_k[val_rows]]
  }
  cat("  ", nrow(d), "samples |", sum(!is.na(d$eaa_oof)), "with OOF residual (in a CV fold)\n")
  d
}

# One EAA file per task: eaa_base is identical across tasks, eaa_oof follows each task's own fold assignment.
for (task in TASKS) {
  cat("\n########## Task:", task, "##########\n")
  folds_path <- results_dir("classifier", "cv", sprintf("cv_folds__%s.csv", task))
  folds <- fread(folds_path)
  folds[, slideId := as.character(slideId)]
  cat("Loaded", nrow(folds), "fold assignments across", length(unique(folds$val_fold)), "folds from", folds_path, "\n")

  eaa <- rbindlist(list(
    compute_eaa(path("horvath_ages_v1"), "v1", folds),
    compute_eaa(path("horvath_ages_v2"), "v2", folds)
  ))

  out_path <- path(sprintf("horvath_eaa_%s", task))
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  fwrite(eaa, out_path)
  cat("Wrote", nrow(eaa), "EAA rows to", out_path, "\n")
}
