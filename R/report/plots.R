library(pheatmap)
library(dplyr)

source("lib/utils.R")
source("lib/plot_utils.R")

plots_root <- path("plots")
dir.create(plots_root, recursive = TRUE, showWarnings = FALSE)

# --- Quality control ---
qc_v1 <- fread(path("epicv1_detectionrate"))[, version := "EpicV1"]
qc_v2 <- fread(path("epicv2_detectionrate"))[, version := "EpicV2"]
qc_all <- rbind(
  data.frame(name = qc_v1$slideId, value = qc_v1$detection_rate, version = "EpicV1"),
  data.frame(name = qc_v2$slideId, value = qc_v2$detection_rate, version = "EpicV2")
)

ph_detection <- ggplot(qc_all, aes(x = value)) +
  geom_histogram(binwidth = 0.05, fill = "skyblue", color = "black", boundary = 0,
                 linewidth = fig_lw) +
  geom_text(stat = "bin", binwidth = 0.05, aes(label = after_stat(count)),
            vjust = -0.5, color = "black", size = fig_annot_size, boundary = 0) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  facet_wrap(~ version, ncol = 2) +
  labs(x = "Probe detection rate", y = "Number of samples") +
  theme_manuscript()
ggsave_panel(file.path(plots_root, "ProbeDetectionRate.pdf"), ph_detection,
             slot = "full", aspect = 0.4)

# --- Deconvolution ---
deconv_plot_dir <- file.path(plots_root, "deconvolution")
dir.create(deconv_plot_dir, recursive = TRUE, showWarnings = FALSE)

df_episcore <- read_diagnosis_csv(path("deconv_episcore"))
cell_types <- setdiff(colnames(df_episcore), c("slideId", "array", "diagnosis"))

# Half width: the two tasks' bars are a figure of their own (see panels.py), one per column.
make_stacked_bar(df_episcore, cell_types, file.path(deconv_plot_dir, "stacked_bar.pdf"))

# Ordinal therapeutic-group variant: attach AJCC-derived groups (IM/NIM only) and stack by them.
df_tg <- attach_therapeutic_group(copy(df_episcore))
make_stacked_bar(df_tg, cell_types, file.path(deconv_plot_dir, "stacked_bar_therapeutic.pdf"),
                 group_col = "therapeutic_group", legend = FALSE)

# --- Horvath clock ---
horvath_plot_dir <- file.path(plots_root, "horvath_clock")
dir.create(horvath_plot_dir, recursive = TRUE, showWarnings = FALSE)

plot_horvath_scatter <- function(ages_path, label) {
  ages <- fread(ages_path)
  ages <- ages[!is.na(age) & !is.na(skinHorvath)]
  if (nrow(ages) == 0) {
    cat("No matched ages for", label, "- skipping scatter\n"); return()
  }
  make_correlation_scatter(
    ages, x_col = "age", y_col = "skinHorvath",
    filename = file.path(horvath_plot_dir,
                         paste0("age_vs_chronological_", tolower(label), ".pdf")),
    x_label = "Chronological age", y_label = "Predicted age (skinHorvath)",
    title = paste("Skin & Blood (Horvath) -", label),
    abline = TRUE
  )
}
plot_horvath_scatter(path("horvath_ages_v1"), "EPICv1")
plot_horvath_scatter(path("horvath_ages_v2"), "EPICv2")

# --- Cohort overview ---
# Every sample over the CpGs most associated with the diagnoses (the slice R/analysis/cohort_heatmap.R persisted),
# annotated with the phenotype tracks below. Columns are blocked by diagnosis; samples clustered inside each block.
cohort_betas_path <- file.path(results_dir("cohort"), "classification_cpgs_betas.csv")
if (!file.exists(cohort_betas_path)) {
  cat("No beta slice at", cohort_betas_path, "- run R/analysis/cohort_heatmap.R first, skipping cohort heatmap\n")
} else {
  cohort_plot_dir <- file.path(plots_root, "cohort")
  dir.create(cohort_plot_dir, recursive = TRUE, showWarnings = FALSE)

  cohort_slice <- fread(cohort_betas_path)
  cohort_betas <- as.matrix(cohort_slice[, -1])
  rownames(cohort_betas) <- cohort_slice[[1]]

  cohort <- attach_diagnosis(data.table(slideId = colnames(cohort_betas)))
  # Left join: NV and unstaged tumours have no therapeutic group and stay in the heatmap.
  cohort <- merge(cohort, attach_therapeutic_group(data.table(slideId = cohort$slideId)),
                  by = "slideId", all.x = TRUE)
  therapeutic_colors <- c(therapeutic_palette, `n/a` = "grey80")
  cohort[, therapeutic_group := factor(fifelse(is.na(therapeutic_group), "n/a", as.character(therapeutic_group)),
                                       levels = names(therapeutic_colors))]
  cohort_meta <- get_meta()[match(cohort$slideId, as.character(slideId))]
  v1_samples <- colnames(fread(path("betas_v1"), nrows = 0))[-1]
  v2_samples <- colnames(fread(path("betas_v2"), nrows = 0))[-1]
  stopifnot(all(cohort$slideId %in% c(v1_samples, v2_samples)))
  cohort[, `:=`(
    age      = cohort_meta$approxAge,
    array    = factor(fifelse(slideId %in% v2_samples, "EPICv2", "EPICv1"), levels = names(array_palette)),
    hospital = factor(unlist(yaml::yaml.load_file(path("hospital_mapping")))[as.character(cohort_meta$clinic)],
                      levels = names(config$colors$hospital), labels = names(hospital_palette))
  )]

  # Diagnosis sets the column order; clustering arranges samples inside each block.
  cohort_order <- unlist(lapply(levels(cohort$diagnosis), function(g) {
    idx <- which(cohort$diagnosis == g)
    idx[hclust(dist(t(cohort_betas[, cohort$slideId[idx], drop = FALSE])))$order]
  }))
  cohort <- cohort[cohort_order]
  cohort_betas <- cohort_betas[, cohort$slideId, drop = FALSE]
  cohort_gaps <- head(cumsum(as.integer(table(cohort$diagnosis))), -1)

  cohort_anno <- data.frame(
    Array               = cohort$array,
    Hospital            = cohort$hospital,
    Age                 = cohort$age,
    `Therapeutic group` = cohort$therapeutic_group,
    Diagnosis           = cohort$diagnosis,
    row.names           = cohort$slideId,
    check.names         = FALSE
  )
  ph_cohort <- pheatmap(
    cohort_betas,
    annotation_col    = cohort_anno,
    annotation_colors = list(Array = array_palette,
                             Hospital = hospital_palette,
                             Age = colorRampPalette(c("#F0F0F0", "#252525"))(100),
                             `Therapeutic group` = therapeutic_colors,
                             Diagnosis = diagnosis_palette),
    color             = colorRampPalette(c("#2166AC", "white", "#D6604D"))(100),
    show_colnames     = FALSE,
    show_rownames     = FALSE,
    cluster_cols      = FALSE,
    gaps_col          = cohort_gaps,
    treeheight_row    = 8,
    fontsize          = fig_cfg$base_size,
    border_color      = NA,
    silent            = TRUE
  )

  cohort_sz <- panel_size("full", height_mm = fig_cfg$max_height_mm - 45)
  cairo_pdf(file.path(cohort_plot_dir, "heatmap_classification_cpgs.pdf"),
            width = cohort_sz$width, height = cohort_sz$height, family = fig_font)
  grid::grid.draw(ph_cohort$gtable)
  dev.off()
  cat("Saved heatmap_classification_cpgs.pdf (", nrow(cohort_betas), "CpGs x", ncol(cohort_betas),
      "samples;", paste(sprintf("%s=%d", levels(cohort$diagnosis), table(cohort$diagnosis)), collapse = ", "), ")\n")
}

# --- NIM spectrum ---
# Where NIM sits on the NV -> IM methylation axis, one panel per axis set, drawn from the per-lesion means
# R/analysis/nim_spectrum.R persisted.
nim_means_path <- file.path(results_dir("nim_spectrum"), "axis_mean_betas.csv")
nim_tests_path <- file.path(results_dir("nim_spectrum"), "axis_pairwise_tests.csv")
PANEL_HEIGHT_MM <- 66
NIM_Y_LABELS <- c(loss = "Mean β at NV-to-IM hypomethylated CpGs",
                  gain = "Mean β at NV-to-IM hypermethylated CpGs")

# Bracket label for one Holm-adjusted p, matching the stars and the wording python/visualization.py annotates with.
# Bounded below: a Wilcoxon p over ~1000 lesions underflows to 0, which "%.3g" would print as the false claim "p=0".
nim_p_label <- function(p) {
  stars <- if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*" else ""
  paste0(if (p < 0.001) "p<0.001" else sprintf("p=%.3g", p), stars)
}

if (!file.exists(nim_means_path)) {
  cat("No axis means at", nim_means_path, "- run R/analysis/nim_spectrum.R first, skipping NIM spectrum figures\n")
} else {
  nim_plot_dir <- file.path(plots_root, "nim_spectrum")
  dir.create(nim_plot_dir, recursive = TRUE, showWarnings = FALSE)

  axis_means <- read_diagnosis_csv(nim_means_path)
  # The panels annotate the tests nim_spectrum.R already ran and adjusted, rather than recomputing them here.
  axis_tests <- if (file.exists(nim_tests_path)) fread(nim_tests_path) else NULL
  if (is.null(axis_tests)) {
    cat("No pairwise tests at", nim_tests_path, "- drawing the axis panels without p-value brackets\n")
  }

  for (key in names(NIM_Y_LABELS)) {
    d <- axis_means[set == key]
    if (nrow(d) == 0) {
      cat("No", key, "rows in", nim_means_path, "- skipping\n")
      next
    }
    med <- d[, .(median_beta = median(mean_beta)), by = diagnosis]

    p_axis <- ggplot(d, aes(x = diagnosis, y = mean_beta)) +
      geom_violin(aes(fill = diagnosis), color = NA, alpha = 0.5, width = 0.85, trim = FALSE) +
      geom_boxplot(width = 0.14, outlier.size = 0.3, linewidth = fig_lw, fill = "white") +
      geom_line(data = med, aes(y = median_beta, group = 1), color = "grey25",
                linewidth = fig_lw, linetype = "22") +
      geom_point(data = med, aes(y = median_beta), color = "grey25", size = 0.6) +
      scale_fill_manual(values = diagnosis_palette, guide = "none") +
      labs(x = NULL, y = NIM_Y_LABELS[[key]]) +
      theme_manuscript()

    # One bracket per tested pair (NV/NIM and NIM/IM, both adjacent on the x axis, so a shared height cannot collide).
    # geom_violin(trim = FALSE) draws past the data range, hence the offset off the panel's own upper limit rather
    # than off max(mean_beta).
    brackets <- if (!is.null(axis_tests)) axis_tests[set == key] else NULL
    if (!is.null(brackets) && nrow(brackets) > 0) {
      y_top <- max(layer_scales(p_axis)$y$range$range)
      span  <- diff(range(d$mean_beta))
      brackets[, `:=`(y.position = y_top + 0.04 * span,
                      label      = vapply(p_adj, nim_p_label, character(1)))]
      p_axis <- p_axis +
        stat_pvalue_manual(brackets, label = "label", size = fig_annot_size,
                           bracket.size = fig_lw, tip.length = 0.01) +
        scale_y_continuous(expand = expansion(mult = c(0.05, 0.14)))
    }
    ggsave_panel(file.path(nim_plot_dir, paste0("axis_", key, ".pdf")), p_axis,
                 slot = "half", height_mm = PANEL_HEIGHT_MM)
  }
}

# --- DMR ---
# Redraws every DMR figure from the tables R/analysis/dmr.R persisted, so tweaking a figure never costs a DMRcate run.
# DMR_FDR comes from config (the same value dmr.R filtered on) purely to label the volcano.
dmr_dir      <- results_dir("dmr")
dmr_plot_dir <- file.path(plots_root, "dmr")
DMR_FDR      <- config$thresholds$dmr_fdr
# The volcano and the count bar are the top row of the differential-methylation figure, side by side at half width
# (see panels.py and PANEL_HEIGHT_MM above). The selected-CpG heatmap instead belongs to the classification figure,
# where it takes the wide slot beside the GO bar chart - it needs the room for its dendrogram.
DMR_WIDE       <- "two_thirds"
# Display form of a contrast; the full "IM_vs_NV" does not fit a facet strip or tick label at these widths. Shared by
# both panels so they name a contrast the same way. Stored values keep dmr.R's spelling.
short_contrast <- function(x) gsub("_vs_", "/", x)

if (!file.exists(file.path(dmr_dir, "dmr_summary.csv"))) {
  cat("No DMR results in", dmr_dir, "- run R/analysis/dmr.R first, skipping DMR figures\n")
} else {
  dir.create(dmr_plot_dir, recursive = TRUE, showWarnings = FALSE)
  dmr_summary <- fread(file.path(dmr_dir, "dmr_summary.csv"))
  contrasts   <- dmr_summary$contrast  # the order dmr.R computed them in

  # High-effect DMR counts per contrast (significant AND |Δβ| >= config$thresholds$dmr_delta)
  p_counts <- ggplot(dmr_summary, aes(x = factor(contrast, levels = contrasts,
                                                 labels = short_contrast(contrasts)),
                                      y = n_high_effect)) +
    geom_col(width = 0.6, fill = "#607D8B") +
    geom_text(aes(label = n_high_effect), vjust = -0.4, size = fig_annot_size) +
    labs(x = NULL, y = "Number of high-effect DMRs") +
    theme_manuscript()
  ggsave_panel(file.path(dmr_plot_dir, "dmr_counts_per_contrast.pdf"), p_counts,
               slot = "half", height_mm = PANEL_HEIGHT_MM)

  # Volcano over every called region; `significant` was precomputed by dmr.R at DMR_FDR.
  dmr_df <- fread(file.path(dmr_dir, "dmr_regions.csv"))
  dmr_df$contrast <- factor(dmr_df$contrast, levels = contrasts, labels = short_contrast(contrasts))
  p_volcano <- ggplot(dmr_df, aes(x = meandiff, y = -log10(HMFDR), color = significant)) +
    geom_point(alpha = 0.6, size = 0.3) +
    geom_hline(yintercept = -log10(DMR_FDR), linetype = "dashed", color = "grey40",
               linewidth = fig_lw) +
    facet_wrap(~ contrast) +
    scale_x_continuous(breaks = c(-0.2, 0, 0.2)) +   # the default five collide across three narrow facets
    scale_color_manual(values = c(`TRUE` = "#D6604D", `FALSE` = "grey70"),
                       labels = c(`TRUE`  = sprintf("HMFDR < %.2f", DMR_FDR),
                                  `FALSE` = "n.s."),
                       name = NULL) +
    # Plain text, not plotmath: cairo substitutes another face for plotmath's symbol glyphs, mixing typefaces in one
    # figure. Same for the GO panel's x label.
    labs(x = "Mean methylation difference Δβ",
         y = "-log10(HMFDR)") +
    theme_manuscript() +
    theme(legend.position = "bottom")
  ggsave_panel(file.path(dmr_plot_dir, "dmr_volcano.pdf"), p_volcano,
               slot = "half", height_mm = PANEL_HEIGHT_MM)

  # Heatmap of the persisted beta slice (selected CpGs inside significant, high-effect DMRs), restricted to the
  # training samples below. Sample annotation is rebuilt from metadata rather than stored, since attach_diagnosis
  # needs no betas.
  betas_path <- file.path(dmr_dir, "selected_cpgs_betas.csv")
  if (!file.exists(betas_path)) {
    cat("No beta slice at", betas_path, "- skipping DMR heatmap\n")
  } else {
    slice <- fread(betas_path)
    betas <- as.matrix(slice[, -1])
    rownames(betas) <- slice[[1]]
    cohort <- attach_diagnosis(data.table(slideId = colnames(betas)))
    # Training samples only. dmr.R calls the regions over the whole cohort, but this heatmap is about the classifier's
    # own features, so it shows the split those features were selected on.
    train_head    <- colnames(fread(path("betas_imputed_train"), nrows = 0))
    train_samples <- setdiff(train_head[-1], "genome_coordinates")   # mirrors load_betas' dropped columns
    cohort <- cohort[slideId %in% train_samples]
    stopifnot(nrow(cohort) > 0)
    betas <- betas[, cohort$slideId, drop = FALSE]
    ph <- pheatmap(
      betas,
      annotation_col    = data.frame(Diagnosis = cohort$diagnosis, row.names = cohort$slideId),
      annotation_colors = list(Diagnosis = diagnosis_palette),
      color             = colorRampPalette(c("#2166AC", "white", "#D6604D"))(100),
      treeheight_col    = 16,
      treeheight_row    = 16,
      show_colnames     = FALSE,
      show_rownames     = nrow(betas) <= 60,
      fontsize          = fig_cfg$base_size,
      fontsize_row      = fig_cfg$small_size,
      border_color      = NA,
      silent            = TRUE
    )
    # pheatmap draws through grid, not ggplot2: the font comes from the device's `family` and the sizes from the
    # fontsize arguments above rather than theme_manuscript().
    dmr_sz <- panel_size(DMR_WIDE, height_mm = PANEL_HEIGHT_MM)
    cairo_pdf(file.path(dmr_plot_dir, "heatmap_selected_high_effect.pdf"),
              width = dmr_sz$width, height = dmr_sz$height, family = fig_font)
    grid::grid.draw(ph$gtable)
    dev.off()
    cat("Saved heatmap_selected_high_effect.pdf (", nrow(betas), "CpGs x", ncol(betas),
        "training samples )\n")
  }
}

# --- GO enrichment ---
# Term bar chart per task from go_results.csv (R/analysis/go_enrichment.R). gometh's bias diagnostic is written by that
# script instead, since gometh draws it as a side effect of the enrichment call.
GO_FDR <- config$thresholds$go_fdr
GO_MIN_DE <- 5   # only show terms with >= 5 hit genes (avoids fragile small-n hits); display-only, so it lives here

# Top terms per ontology by nominal P.DE, restricted to FDR-significant ones where any qualify.
plot_go_terms <- function(go_res, filename) {
  top_go <- go_res %>%
    filter(FDR < GO_FDR, DE >= GO_MIN_DE) %>%
    group_by(ONTOLOGY) %>%
    slice_min(P.DE, n = 10, with_ties = FALSE) %>%
    ungroup()

  if (nrow(top_go) == 0) {
    cat(sprintf("No GO terms with FDR < %.2f and DE >= %d - showing top 10 by P.DE per ontology (DE >= %d only).\n",
                GO_FDR, GO_MIN_DE, GO_MIN_DE))
    top_go <- go_res %>%
      filter(DE >= GO_MIN_DE) %>%
      group_by(ONTOLOGY) %>%
      slice_min(P.DE, n = 10, with_ties = FALSE) %>%
      ungroup()
  }
  if (nrow(top_go) == 0) {
    cat("No GO terms to plot - skipping\n")
    return(invisible(NULL))
  }

  p <- top_go %>%
    mutate(GO_ID = factor(GO_ID, levels = rev(unique(GO_ID)))) %>%
    ggplot(aes(x = -log10(P.DE), y = GO_ID, fill = ONTOLOGY)) +
    geom_col() +
    geom_text(aes(label = sprintf("n=%d", DE)), hjust = -0.15, size = fig_annot_size, color = "grey25") +
    facet_grid(ONTOLOGY ~ ., scales = "free_y", space = "free_y") +
    scale_fill_manual(values = c(BP = "#6DA8C9", MF = "#9b4466", CC = "red3")) +
    scale_x_continuous(breaks = scales::breaks_width(2), expand = expansion(mult = c(0, 0.3))) +
    labs(x = "-log10(P_DE)", y = NULL) +   # plain text, see the volcano's y label
    theme_manuscript() +
    theme(axis.text.y     = element_text(size = fig_cfg$small_size, color = "black"),
          legend.position = "none")
  # Third-width: shares a row with the two_thirds-wide selected-CpG heatmap (see panels.py), whose height is the floor
  # here. Height follows the term count rather than an aspect ratio.
  ggsave_panel(filename, p, slot = "third", height_mm = max(20 + 3.5 * nrow(top_go), PANEL_HEIGHT_MM))
}

for (task in c("classification", "ordinal")) {
  go_path <- results_dir("go", task, "go_results.csv")
  if (!file.exists(go_path)) {
    cat("No GO results at", go_path, "- run R/analysis/go_enrichment.R first, skipping\n")
    next
  }
  cat("--- GO:", task, "---\n")
  go_plot_dir <- file.path(plots_root, "go", task)
  dir.create(go_plot_dir, recursive = TRUE, showWarnings = FALSE)
  plot_go_terms(fread(go_path), file.path(go_plot_dir, "go_top_terms.pdf"))
}

cat("Done.\n")
