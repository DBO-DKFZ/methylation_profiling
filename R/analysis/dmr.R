# Differentially methylated region (DMR) analysis of the classification model's selected CpGs.
#
# Asks whether the CpGs selected for the 3-class diagnosis model sit inside regions that are differentially
# methylated between the diagnoses — i.e. whether the classifier's features are regional biology rather than isolated
# probes. Classification-only by design: the contrasts are the three diagnosis pairs. The ordinal task has no
# equivalent (its therapeutic groups are ordered, not a set of pairwise contrasts); for it, see go_enrichment.R.
#
# Inputs : paths.betas_imputed_train,
#          paths.betas_imputed_test             (imputed beta matrices, rows = genome coordinates)
#          paths.selected_cpgs_classification   (the classifier's CpG selection; `python -m python.ml.export_cpgs`)
#          paths.cpg_position_mapping           (genome coordinate <-> EPIC probe ID bridge)
#          meta (diagnosis)                     (cohort groups via attach_diagnosis)
# Steps  : stack both splits over their shared CpGs -> DMRcate per diagnosis contrast -> significant DMRs
#          (HMFDR < DMR_FDR) -> high-effect subset (|Δβ| >= DELTA_CUTOFF) -> intersect with the selected CpGs
# Outputs: results/dmr/  dmr_summary.csv, dmr_regions.csv, selected_cpgs_in_dmrs.csv, selected_cpgs_betas.csv
#
# The contrasts are computed over the whole cohort rather than the training split: this is a descriptive analysis. The
# CpG selection it is intersected with is still the one derived on the training split alone.
#
# Compute only — every figure is drawn by R/report/plots.R from the tables above, so plots can be adjusted without
# re-running DMRcate (a ~20 min, ~20 GB job). The last output exists purely to feed that report.
#
# Run order: R/preprocessing/impute_betas.R and `python -m python.ml.export_cpgs` must both have run first.

library(DMRcate)
library(limma)
library(GenomicRanges)
library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
library(dplyr)

source("lib/utils.R")

CONTRASTS <- c("IM_vs_NV", "NIM_vs_NV", "IM_vs_NIM")

PROBE_FDR <- 0.05  # probe-level FDR handed to cpg.annotate

DMR_FDR      <- config$thresholds$dmr_fdr    # region-level HMFDR threshold
DELTA_CUTOFF <- config$thresholds$dmr_delta  # high-effect threshold on |meandiff| (Δβ)

out_dir <- results_dir("dmr")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)


## 1) Load data

# Beta matrix: rows = CpG genome coordinates, columns = slide IDs. Both splits are stacked over the CpGs they share,
# then restricted to the samples carrying one of the three diagnoses.
betas_train <- load_betas(path("betas_imputed_train"))
betas_test  <- load_betas(path("betas_imputed_test"))
shared_cpgs <- intersect(rownames(betas_train), rownames(betas_test))
cat("Shared CpGs:", length(shared_cpgs), "( train-only", nrow(betas_train) - length(shared_cpgs),
    ", test-only", nrow(betas_test) - length(shared_cpgs), ")\n")

betas_all <- cbind(betas_train[shared_cpgs, , drop = FALSE], betas_test[shared_cpgs, , drop = FALSE])
rm(betas_train, betas_test); invisible(gc())
stopifnot(!anyDuplicated(colnames(betas_all)))

cohort <- attach_diagnosis(data.table(slideId = colnames(betas_all)))
betas_mat <- betas_all[, cohort$slideId, drop = FALSE]   # attach_diagnosis re-sorts on merge
rm(betas_all); invisible(gc())
stopifnot(identical(colnames(betas_mat), cohort$slideId))
cat("Cohort:", nrow(cohort), "samples (",
    paste(sprintf("%s=%d", levels(cohort$diagnosis), table(cohort$diagnosis)), collapse = ", "), ")\n")

selected_cpgs <- load_selected_cpgs("classification")

# Coordinate mapping to Illumina probe IDs — DMRcate's array annotation is probe-ID based.
cpg_map <- fread(path("cpg_position_mapping"), data.table = FALSE)
rownames(cpg_map) <- cpg_map$genome_coordinates

selected_in_map <- intersect(selected_cpgs, rownames(cpg_map))
selected_ids    <- cpg_map[selected_in_map, "Probe_ID"]
cat("Selected CpGs mapped to an EPIC probe ID:", length(selected_ids), "\n")

# DMRcate names its annotated CpGs by hg19 position ("chr1:935861"), not by probe ID, so locating the selection among
# them needs the same key: probe ID -> hg19 chr:pos from the EPICv1 annotation package DMRcate itself annotates against.
data(list = "Locations", package = "IlluminaHumanMethylationEPICanno.ilm10b4.hg19")
selected_ids   <- selected_ids[selected_ids %in% rownames(Locations)]
selected_keys  <- paste0(Locations[selected_ids, "chr"], ":", Locations[selected_ids, "pos"])
# hg19 key -> hg38 genome coordinate, so reported CpGs carry the label the rest of the project uses.
selected_label <- setNames(cpg_map[match(selected_ids, cpg_map$Probe_ID), "genome_coordinates"], selected_keys)
cat("Selected CpGs with an hg19 position:", length(selected_keys), "\n")


## 2) Map CpG identifiers to EPIC (v1) probe IDs

betas_annot <- betas_mat[rownames(betas_mat) %in% rownames(cpg_map), ]
rownames(betas_annot) <- cpg_map[rownames(betas_annot), "Probe_ID"]
stopifnot(!anyDuplicated(rownames(betas_annot)))
cat("Probe-annotated matrix:", nrow(betas_annot), "probes x", ncol(betas_annot), "samples\n")


## 3) DMRcate per contrast

design <- model.matrix(~ 0 + cohort$diagnosis)
colnames(design) <- levels(cohort$diagnosis)

cont_matrix <- makeContrasts(
  IM_vs_NV  = IM  - NV,
  NIM_vs_NV = NIM - NV,
  IM_vs_NIM    = IM  - NIM,
  levels = design
)

# Returns the CpG annotation (all probes, needed to locate selected CpGs inside regions) and the full region table.
run_dmrcate <- function(coef) {
  cat("\n=== DMRcate:", coef, "===\n")
  # `cpg_position_mapping` bridges to the v1 `Probe_ID` column, and the EPICv1 annotation is hg19 — hence
  # arraytype "EPICv1" here and `genome = "hg19"` in extractRanges below. DMRcate >= 3 rejects the legacy "EPIC".
  ann <- cpg.annotate("array", betas_annot, what = "Beta", arraytype = "EPICv1",
                      analysis.type = "differential", design = design,
                      contrasts = TRUE, cont.matrix = cont_matrix,
                      fdr = PROBE_FDR, coef = coef)
  res <- extractRanges(dmrcate(ann, lambda = 1000, C = 2), genome = "hg19")
  cat(length(res), "regions called\n")
  list(ann = ann, res = res)
}
results <- setNames(lapply(CONTRASTS, run_dmrcate), CONTRASTS)

# Two-step filter: statistically significant (HMFDR < DMR_FDR), then high-effect (|meandiff| >= DELTA_CUTOFF).
sig_dmrs <- lapply(results, function(r) r$res[r$res$HMFDR < DMR_FDR])
he_dmrs  <- lapply(sig_dmrs, function(r) r[abs(r$meandiff) >= DELTA_CUTOFF])


## 4) Overlap of DMRs with the selected CpGs

# Selected CpGs (as annotated probes) falling inside `dmrs`, for one contrast.
overlap_selected <- function(dmrs, ann) {
  ann_gr      <- ann@ranges                              # all annotated CpGs, named by hg19 position
  selected_gr <- ann_gr[names(ann_gr) %in% selected_keys]  # the model's selection among them
  list(ann_gr      = ann_gr,
       selected_gr = selected_gr,
       hits        = findOverlaps(dmrs, selected_gr))
}
he_overlap <- setNames(lapply(CONTRASTS, function(nm) overlap_selected(he_dmrs[[nm]], results[[nm]]$ann)),
                       CONTRASTS)

# Per-contrast counts: how much of the array, and how much of the selection, the high-effect DMRs cover.
dmr_summary <- rbindlist(lapply(CONTRASTS, function(nm) {
  o <- he_overlap[[nm]]
  data.table(contrast         = nm,
             n_regions        = length(results[[nm]]$res),
             n_significant    = length(sig_dmrs[[nm]]),
             n_high_effect    = length(he_dmrs[[nm]]),
             n_array_cpgs     = length(o$ann_gr),
             n_selected_cpgs  = length(o$selected_gr),
             cpgs_in_dmr      = length(unique(subjectHits(findOverlaps(he_dmrs[[nm]], o$ann_gr)))),
             selected_in_dmr  = length(unique(subjectHits(o$hits))))
}))
fwrite(dmr_summary, file.path(out_dir, "dmr_summary.csv"))
cat("\nSaved", file.path(out_dir, "dmr_summary.csv"), "\n")
print(dmr_summary)

# Per-DMR list of the selected CpGs it contains (genome-coordinate labels).
selected_hits <- bind_rows(lapply(CONTRASTS, function(nm) {
  o <- he_overlap[[nm]]
  if (length(o$hits) == 0) return(NULL)
  ids_per_dmr <- split(names(o$selected_gr)[subjectHits(o$hits)], queryHits(o$hits))
  df <- as.data.frame(he_dmrs[[nm]][as.integer(names(ids_per_dmr))])
  df$contrast      <- nm
  df$selected_cpgs <- vapply(ids_per_dmr,
                             function(keys) paste(selected_label[keys], collapse = "; "),
                             character(1))
  df
}))
if (nrow(selected_hits) > 0) {
  fwrite(selected_hits, file.path(out_dir, "selected_cpgs_in_dmrs.csv"))
  cat("Saved", file.path(out_dir, "selected_cpgs_in_dmrs.csv"), "\n")
} else {
  cat("No selected CpG falls inside a high-effect DMR\n")
}

# Combined region table across contrasts — every region, not just the significant ones, since the report's volcano
# needs the full cloud. `significant` is precomputed here so the figure cannot apply a different cutoff.
dmr_df <- bind_rows(lapply(CONTRASTS, function(nm) {
  cbind(as.data.frame(results[[nm]]$res), contrast = nm)
}))
dmr_df$contrast    <- factor(dmr_df$contrast, levels = CONTRASTS)
dmr_df$significant <- dmr_df$HMFDR < DMR_FDR  # colours the volcano
fwrite(dmr_df, file.path(out_dir, "dmr_regions.csv"))
cat("Saved", file.path(out_dir, "dmr_regions.csv"), "\n")


## 5) Report inputs

# Betas at the CpGs the report's heatmap draws (selected CpGs inside significant, high-effect DMRs). Persisting this
# ~200-row slice is what lets R/report/plots.R redraw the heatmap without re-reading the multi-GB beta matrix. Written
# for the whole cohort; the heatmap itself keeps only the training columns, since it describes the classifier's own
# features.
heatmap_cpgs <- if (nrow(selected_hits) > 0) {
  hit_cpgs <- unique(unlist(strsplit(selected_hits$selected_cpgs, "; ")))
  hit_cpgs[hit_cpgs %in% rownames(betas_mat)]
} else {
  character(0)
}
if (length(heatmap_cpgs) > 0) {
  betas_slice <- data.table(cpg = heatmap_cpgs,
                            as.data.table(betas_mat[heatmap_cpgs, , drop = FALSE]))
  fwrite(betas_slice, file.path(out_dir, "selected_cpgs_betas.csv"))
  cat("Saved", file.path(out_dir, "selected_cpgs_betas.csv"), "|",
      length(heatmap_cpgs), "CpGs x", ncol(betas_mat), "samples\n")
} else {
  cat("No selected CpGs in high-effect DMRs - no beta slice written\n")
}

cat("Done. Figures are drawn by R/report/plots.R.\n")
