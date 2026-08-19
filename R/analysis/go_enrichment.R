# GO enrichment of each task's selected CpGs.
#
# Asks which biological processes the model's selected CpGs sit in. Each CpG is annotated to its nearest gene
# (hg38, ChIPseeker) and the selection is tested against the array background with missMethyl::gometh, which corrects
# for the number of CpGs per gene — the dominant bias in array-based enrichment. Runs for the classification and the
# ordinal task in one go, each against its own selection.
#
# Inputs : paths.selected_cpgs_<task>   (each task's CpG selection; `python -m python.ml.export_cpgs`)
#          paths.cpg_position_mapping   (genome coordinate <-> EPIC probe ID bridge)
#          paths.betas_imputed_train    (only its CpG index — defines the enrichment universe)
# Outputs: results/go/<task>/  selected_gene_annotation.csv, go_results.csv, go_top15_per_ontology.csv
#          plots/go/<task>/    gometh_bias.pdf
#
# Compute only, with one exception: the enrichment bar chart is drawn by R/report/plots.R from go_results.csv, but
# gometh's bias diagnostic cannot be split off — gometh draws it as a side effect of the (expensive) enrichment call,
# so it is emitted here.
#
# Run order: R/preprocessing/impute_betas.R and `python -m python.ml.export_cpgs` must both have run first.

library(GenomicRanges)
library(ChIPseeker)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(org.Hs.eg.db)
library(missMethyl)
library(dplyr)

source("lib/utils.R")

TASKS <- c("classification", "ordinal")

TSS_REGION <- c(-1500, 200)  # promoter window TSS-1500 .. TSS+200

# Shared with R/report/plots.R, which filters the reported terms by it (see config).
GO_FDR <- config$thresholds$go_fdr

txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene

# Genome positions, shared across tasks.
cpg_map           <- fread(path("cpg_position_mapping"), data.table = FALSE)
cpg_map$chr       <- sub("-[0-9]+$", "", cpg_map$genome_coordinates)
cpg_map$pos       <- as.integer(sub("^.*-", "", cpg_map$genome_coordinates))
rownames(cpg_map) <- cpg_map$genome_coordinates

# Universe = every CpG in the training beta matrix that maps to an EPIC probe ID. Only the CpG index is needed, so read
# the first column rather than the (multi-GB) matrix.
betas_cpgs      <- fread(path("betas_imputed_train"), select = 1)[[1]]
universe_coords <- intersect(betas_cpgs, rownames(cpg_map))
all_probes      <- cpg_map[universe_coords, "Probe_ID"]
all_probes      <- unique(all_probes[!is.na(all_probes) & all_probes != ""])
cat("Enrichment universe:", length(all_probes), "EPIC probes\n")


# Annotates each selected CpG to its nearest gene and returns one row per CpG.
annotate_to_genes <- function(selected_map) {
  selected_gr <- GRanges(
    seqnames = selected_map$chr,
    ranges   = IRanges(start = selected_map$pos, end = selected_map$pos),
    probe_id = selected_map$Probe_ID,
    coord_id = selected_map$genome_coordinates
  )
  names(selected_gr) <- selected_map$genome_coordinates

  peak_anno <- annotatePeak(selected_gr, TxDb = txdb, level = "gene", annoDb = "org.Hs.eg.db",
                            verbose = FALSE, tssRegion = TSS_REGION)
  as.data.frame(peak_anno) %>%
    transmute(
      coord_id       = coord_id,
      probe_id       = probe_id,
      seqnames, start,
      annotation,
      distanceToTSS,
      gene_entrez    = geneId,
      gene_symbol    = SYMBOL,
      gene_name_full = GENENAME
    )
}

for (task in TASKS) {
  cat("\n########## Task:", task, "##########\n")
  out_dir  <- results_dir("go", task)
  plot_dir <- plots_dir("go", task)
  dir.create(out_dir,  recursive = TRUE, showWarnings = FALSE)
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

  selected_cpgs   <- load_selected_cpgs(task)
  selected_in_map <- intersect(selected_cpgs, rownames(cpg_map))
  cat(length(selected_in_map), "of them map to an EPIC probe ID\n")

  # Nearest-gene annotation per selected CpG (hg38).
  gene_anno <- annotate_to_genes(cpg_map[selected_in_map, ])
  fwrite(gene_anno, file.path(out_dir, "selected_gene_annotation.csv"))
  cat("Saved", file.path(out_dir, "selected_gene_annotation.csv"), "\n")

  # Selection = the selected CpGs mapped to EPIC probe IDs; must be a subset of the universe.
  sig_probes <- cpg_map[selected_in_map, "Probe_ID"]
  sig_probes <- unique(sig_probes[!is.na(sig_probes) & sig_probes != ""])
  sig_probes <- intersect(sig_probes, all_probes)
  cat("Selection:", length(sig_probes), "probes\n")

  # Enrichment with bias correction for n(CpGs) per gene. plot.bias draws gometh's diagnostic into the open device, so
  # the results and the diagnostic come out of a single (expensive) run.
  pdf(file.path(plot_dir, "gometh_bias.pdf"), width = 6, height = 5)
  go_res <- gometh(
    sig.cpg    = sig_probes,
    all.cpg    = all_probes,
    collection = "GO",
    array.type = "EPIC",
    plot.bias  = TRUE,
    sig.genes  = TRUE   # attach gene symbols driving each term
  )
  dev.off()

  go_res <- go_res %>%
    tibble::rownames_to_column("GO_ID") %>%
    arrange(P.DE)
  cat(sprintf("GO terms with FDR < %.2f: %d\n", GO_FDR, sum(go_res$FDR < GO_FDR, na.rm = TRUE)))

  fwrite(go_res, file.path(out_dir, "go_results.csv"))
  cat("Saved", file.path(out_dir, "go_results.csv"), "\n")

  # Top 15 GO terms per ontology (ranked by nominal P.DE)
  go_top15 <- go_res %>%
    group_by(ONTOLOGY) %>%
    slice_head(n = 15) %>%
    ungroup()
  fwrite(go_top15, file.path(out_dir, "go_top15_per_ontology.csv"))
  cat("Saved", file.path(out_dir, "go_top15_per_ontology.csv"), "\n")
}

cat("\nDone. The enrichment bar chart is drawn by R/report/plots.R.\n")
