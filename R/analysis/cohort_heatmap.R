# Beta slice for the cohort heatmap: the CpGs most strongly associated with the three diagnoses.
#
# Ranking by Kruskal-Wallis which CpGs separate IM, NIM and NV across the cohort.
#   * The ranking is computed on the training split only
#   * The Figure describes the whole cohort (both splits).
#
# Inputs : paths.betas_imputed_train, paths.betas_imputed_test   (imputed beta matrices, rows = genome coordinates)
#          meta (diagnosis)                                      (cohort groups via attach_diagnosis)
# Steps  : stack both splits over their shared CpGs -> Kruskal-Wallis per CpG on the training split -> top TOP_N_CPGS
# Outputs: results/cohort/  classification_cpgs_betas.csv (the slice the report draws),
#                           classification_cpgs_stats.csv (H, p, effect sizes and group means for those CpGs)
#
# Compute only - R/report/plots.R draws the heatmap from the slice above, so the figure can be adjusted without
# re-reading the multi-GB matrices. The slice has the same shape as the DMR one (first column `cpg`, one column per
# slideId), so both report blocks read it the same way.
#
# Run order: R/preprocessing/impute_betas.R must have run first.

library(matrixStats)

source("lib/utils.R")

# Enough rows for the heatmap to show texture while keeping the persisted slice a few MB (results/ is versioned).
TOP_N_CPGS <- 500

out_dir <- results_dir("cohort")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)


## 1) Load the cohort

betas_train <- load_betas(path("betas_imputed_train"))
betas_test  <- load_betas(path("betas_imputed_test"))
shared_cpgs <- intersect(rownames(betas_train), rownames(betas_test))
cat("Shared CpGs:", length(shared_cpgs), "( train-only", nrow(betas_train) - length(shared_cpgs),
    ", test-only", nrow(betas_test) - length(shared_cpgs), ")\n")

betas <- cbind(betas_train[shared_cpgs, , drop = FALSE], betas_test[shared_cpgs, , drop = FALSE])
train_ids <- colnames(betas_train)
rm(betas_train, betas_test); invisible(gc())
stopifnot(!anyDuplicated(colnames(betas)), !anyNA(betas))

cohort <- attach_diagnosis(data.table(slideId = colnames(betas)))
betas  <- betas[, cohort$slideId, drop = FALSE]   # attach_diagnosis re-sorts on merge
is_train <- cohort$slideId %in% train_ids
train  <- betas[, is_train, drop = FALSE]
grp    <- cohort$diagnosis[is_train]
cat("Cohort:", ncol(betas), "samples x", nrow(betas), "CpGs | selection uses the training split:", ncol(train),
    "samples (", paste(sprintf("%s=%d", levels(grp), table(grp)), collapse = ", "), ")\n")


## 2) Kruskal-Wallis per CpG

# Vectorised over the whole matrix: rank within each CpG, then sum the ranks per diagnosis via an indicator matrix.
ranks <- rowRanks(train, ties.method = "average")
indicator  <- model.matrix(~ grp - 1)
rank_sums  <- ranks %*% indicator
n_per_grp  <- colSums(indicator)
n          <- ncol(train)
H <- 12 / (n * (n + 1)) * rowSums(sweep(rank_sums^2, 2, n_per_grp, "/")) - 3 * (n + 1)
cat(sprintf("CpGs containing tied values: %.2f%%\n", 100 * mean(rowSums(ranks != round(ranks)) > 0)))
rm(ranks); invisible(gc())

group_means <- (betas %*% model.matrix(~ cohort$diagnosis - 1)) %*% diag(1 / as.integer(table(cohort$diagnosis)))
colnames(group_means) <- levels(cohort$diagnosis)

stats <- data.table(cpg = rownames(betas), H = H,
                    p_value = pchisq(H, df = nlevels(grp) - 1, lower.tail = FALSE),
                    eta_squared = (H - (nlevels(grp) - 1)) / (n - nlevels(grp)),
                    max_delta = rowMaxs(group_means) - rowMins(group_means),
                    as.data.table(group_means))
stats <- stats[order(-H)][seq_len(min(TOP_N_CPGS, .N))]
cat(sprintf("Selected %d CpGs: eta^2 %.3f - %.3f, max pairwise delta-beta %.3f - %.3f, p <= %.2e\n",
            nrow(stats), min(stats$eta_squared), max(stats$eta_squared),
            min(stats$max_delta), max(stats$max_delta), max(stats$p_value)))


## 3) Persist

selected <- sort(stats$cpg)
fwrite(data.table(cpg = selected, as.data.table(round(betas[selected, , drop = FALSE], 4))),
       file.path(out_dir, "classification_cpgs_betas.csv"))
fwrite(stats, file.path(out_dir, "classification_cpgs_stats.csv"))
cat("Saved classification_cpgs_betas.csv (", length(selected), "CpGs x", ncol(betas), "samples )\n")
