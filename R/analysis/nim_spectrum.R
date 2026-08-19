# Where does NIM sit between NV and IM?
#
# Builds a methylation axis for the putative progression NV -> IM and places NIM on it: the CpGs whose mean beta
# changes most between benign nevus (NV) and invasive melanoma (IM) define two sets (lost and gained methylation), and
# every lesion is summarised as its mean beta over each set. NIM never enters the CpG selection, so both comparisons
# involving it are out-of-sample; NV vs IM is deliberately not tested, since the axis CpGs were picked to maximise
# exactly that difference.
#
# Inputs : paths.betas_imputed_train, paths.betas_imputed_test   (imputed beta matrices, rows = genome coordinates)
#          meta (diagnosis)                                      (cohort groups via attach_diagnosis)
# Steps  : stack both splits over their shared CpGs -> rank CpGs by mean IM - NV delta-beta -> the N_AXIS most extreme
#          per direction -> per-lesion mean beta over each set -> Wilcoxon rank-sum NV/NIM and NIM/IM, Holm-adjusted
# Outputs: results/nim_spectrum/  axis_mean_betas.csv (one mean per lesion per axis set, the figure's table),
#                                 axis_pairwise_tests.csv (the two tests per axis set; also the figures' bracket labels)
#
# The axis is defined over the whole cohort: this is a descriptive contrast.
#
# Compute only - R/report/plots.R draws the two violin panels from the two tables above (means for the violins,
# adjusted p's for the brackets), so the figures can be adjusted without re-reading the multi-GB matrices.
#
# Run order: R/preprocessing/impute_betas.R must have run first.

source("lib/utils.R")

# CpGs per direction on the axis.
N_AXIS <- 10000

PAIRS <- list(c("NV", "NIM"), c("NIM", "IM"))

out_dir <- results_dir("nim_spectrum")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)


## 1) Load the cohort

betas_train <- load_betas(path("betas_imputed_train"))
betas_test  <- load_betas(path("betas_imputed_test"))
shared_cpgs <- intersect(rownames(betas_train), rownames(betas_test))
cat("Shared CpGs:", length(shared_cpgs), "( train-only", nrow(betas_train) - length(shared_cpgs),
    ", test-only", nrow(betas_test) - length(shared_cpgs), ")\n")

betas <- cbind(betas_train[shared_cpgs, , drop = FALSE], betas_test[shared_cpgs, , drop = FALSE])
rm(betas_train, betas_test); invisible(gc())
stopifnot(!anyDuplicated(colnames(betas)), !anyNA(betas))

cohort <- attach_diagnosis(data.table(slideId = colnames(betas)))
betas  <- betas[, cohort$slideId, drop = FALSE]   # attach_diagnosis re-sorts on merge
cat("Cohort:", ncol(betas), "samples x", nrow(betas), "CpGs (",
    paste(sprintf("%s=%d", levels(cohort$diagnosis), table(cohort$diagnosis)), collapse = ", "), ")\n")


## 2) Define the NV -> IM axis

delta <- rowMeans(betas[, cohort$diagnosis == "IM", drop = FALSE]) -
         rowMeans(betas[, cohort$diagnosis == "NV", drop = FALSE])
ord <- order(delta)

axis_sets <- list(loss = rownames(betas)[head(ord, N_AXIS)],   # hypomethylated in IM relative to NV
                  gain = rownames(betas)[tail(ord, N_AXIS)])   # hypermethylated in IM relative to NV
cat(sprintf("Axis delta-beta: loss %.3f - %.3f, gain %.3f - %.3f\n",
            min(delta[axis_sets$loss]), max(delta[axis_sets$loss]),
            min(delta[axis_sets$gain]), max(delta[axis_sets$gain])))


## 3) Per-lesion aggregation

# One value per lesion per axis set.
axis_means <- rbindlist(lapply(names(axis_sets), function(set) data.table(
  slideId   = cohort$slideId,
  diagnosis = cohort$diagnosis,
  set       = set,
  mean_beta = colMeans(betas[axis_sets[[set]], , drop = FALSE])
)))
fwrite(axis_means, file.path(out_dir, "axis_mean_betas.csv"))
cat("Saved axis_mean_betas.csv (", nrow(axis_means), "rows )\n")


## 4) Pairwise tests

# Wilcoxon rank-sum on the per-lesion means (no normality assumed), Holm-adjusted across the four reported tests:
# a small confirmatory family, so family-wise error control matches the rest of the project (python/ml/stats.py).
tests <- rbindlist(lapply(names(axis_sets), function(key) {
  d <- axis_means[set == key]
  rbindlist(lapply(PAIRS, function(p) {
    x1 <- d$mean_beta[d$diagnosis == p[1]]
    x2 <- d$mean_beta[d$diagnosis == p[2]]
    data.table(set = key, group1 = p[1], group2 = p[2],
               n1 = length(x1), n2 = length(x2),
               median1 = median(x1), median2 = median(x2),
               p_value = wilcox.test(x1, x2)$p.value)
  }))
}))
tests[, p_adj := p.adjust(p_value, method = "holm")]
fwrite(tests, file.path(out_dir, "axis_pairwise_tests.csv"))
cat("Saved axis_pairwise_tests.csv\n")
print(tests)

cat("Done. Figures are drawn by R/report/plots.R.\n")
